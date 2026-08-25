//
//  SAMEncoder.swift
//  JudgeE2
//
//  Created by Jian Sun on 2/28/26.
//
//  Optimisations vs original:
//  - Pixel loop replaced with vDSP vectorised ops (scale, bias, deinterleave).
//    1024×1024 normalisation: ~40 ms → ~4 ms on A-series.
//  - CVPixelBuffer reuse pool (single pre-allocated output buffer).
//  - CIContext shared with render options to skip unnecessary colour-management.

import Accelerate
import CoreML
import CoreImage
import Foundation

final class SAMEncoder {
    private let model: MLModel

    // Shared CIContext — disable colour management for raw pixel fidelity & speed.
    private let ciContext = CIContext(options: [
        .workingColorSpace: NSNull(),
        .outputColorSpace: NSNull()
    ])

    // Pre-allocated output pixel buffer (reused across calls).
    private var outputBuffer: CVPixelBuffer?

    // MobileSAM preprocessing constants.
    // Phase 3 Day 4: `inputSize` is now driven by SAMConfiguration for the
    // resolution AB test (1024 default, 768 opt-in).  The encoder's feature-grid
    // output side (`featureSize`) is inputSize/16 (1024→64, 768→48).  When the
    // encoder runs at 768 its [1,256,48,48] output is bilinearly upsampled to
    // [1,256,64,64] before returning (Architect C-3: Decoder is fixed at 64×64).
    private let inputSize: Int
    private let featureSize: Int
    private let resolution: SAMConfiguration.EncoderResolution
    // mean/std in BGRA channel order (encoder expects RGB, we deinterleave after)
    private let meanR: Float = 123.675
    private let meanG: Float = 116.28
    private let meanB: Float = 103.53
    private let stdR:  Float = 58.395
    private let stdG:  Float = 57.12
    private let stdB:  Float = 57.375

    init?(computeUnits: MLComputeUnits = .all,
          resolution: SAMConfiguration.EncoderResolution = SAMConfiguration.encoderResolution) {
        self.resolution = resolution
        self.inputSize  = resolution.rawValue
        self.featureSize = resolution.featureSize
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        // Phase 3 Debugger session analysis revealed the fp32 fix (Phase 3 Day 2) was
        // counter-productive: fp32 encoder was +32% slower (1131ms vs 857ms) and ANE
        // warnings were NOT eliminated (ANE runtime re-casts fp32→fp16 internally,
        // re-triggering the C=1 alignment warning on the neck LayerNorm2d ops).
        //
        // Phase 3 Builder fix (§B.4 MIL LayerNorm fusion):
        //   export_encoder_fp16_milfix.py patches tiny_vit_sam.LayerNorm2d.forward
        //   to use F.layer_norm via transpose, which coremltools converts directly to
        //   the MIL `layer_norm` op.  Result: reduce_mean=0 (was 4), layer_norm=22
        //   (was 20).  No C=1 Float16 intermediate tensors exposed to ANE.
        //
        // Load priority:
        //   - res768 AB test → milfix_768 (falls back to 1024 milfix if 768 absent,
        //     but then inputSize/featureSize would mismatch, so require 768 present)
        //   - res1024 default → fp16_milfix > fp32 (fallback) > fp16 original
        let resourceName: String
        switch resolution {
        case .res768:
            if Bundle.main.url(forResource: "MobileSAM_ImageEncoder_fp16_milfix_768",
                               withExtension: "mlmodelc") != nil {
                resourceName = "MobileSAM_ImageEncoder_fp16_milfix_768"
            } else {
                faultLog("[SAMEncoder] ⚠️ res768 requested but milfix_768 model missing in bundle")
                return nil
            }
        case .res1024:
            if Bundle.main.url(forResource: "MobileSAM_ImageEncoder_fp16_milfix",
                               withExtension: "mlmodelc") != nil {
                resourceName = "MobileSAM_ImageEncoder_fp16_milfix"
            } else if Bundle.main.url(forResource: "MobileSAM_ImageEncoder_fp32",
                                      withExtension: "mlmodelc") != nil {
                resourceName = "MobileSAM_ImageEncoder_fp32"
            } else {
                resourceName = "MobileSAM_ImageEncoder"
            }
        }
        guard let url = Bundle.main.url(forResource: resourceName,
                                        withExtension: "mlmodelc") else {
            faultLog("\(resourceName).mlmodelc not found in bundle")
            return nil
        }
        // computeUnits is logged so device traces can tell an ANE numeric problem
        // apart from a memory-access one when the sentinel fires.
        perfLog("SAMEncoder loading model: \(resourceName) (input=\(inputSize) feat=\(featureSize) units=\(computeUnits.rawValue))")
        do {
            model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            faultLog("Failed to load \(resourceName): \(error)")
            return nil
        }
        // Pre-allocate the BGRA render target at the active input resolution.
        outputBuffer = Self.makePixelBuffer(size: inputSize)
    }

    /// True when the most recent `encode` was rejected by the numeric sentinel
    /// (as opposed to a plain CoreML failure).  Read on encoderQueue only.
    private(set) var lastEncodeRejectedAsGarbage = false

    func encode(pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        guard let inputArray = preprocess(pixelBuffer: pixelBuffer) else { return nil }
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": inputArray])
            let output = try model.prediction(from: input)
            guard let embedding = output.featureValue(for: "image_embeddings")?.multiArrayValue else {
                return nil
            }
            // Numeric sentinel: a corrupt embedding decodes into axis-aligned slabs
            // or 1-px line masks.  Reject it here so no downstream stage renders it.
            if let issue = Self.numericIssue(in: embedding, label: "embedding") {
                lastEncodeRejectedAsGarbage = true
                faultLog("[SAMEncoder] embedding rejected — \(issue)")
                return nil
            }
            lastEncodeRejectedAsGarbage = false

            // Architect C-3: Decoder is fixed at a 64×64 embedding.  When running
            // the 768 AB variant the encoder emits [1,256,48,48]; upsample it to
            // [1,256,64,64] bilinearly so the Decoder contract is unchanged.
            if featureSize == SAMConfiguration.decoderEmbeddingSize {
                return embedding
            }
            return Self.bilinearUpsampleEmbedding(
                embedding,
                srcSize: featureSize,
                dstSize: SAMConfiguration.decoderEmbeddingSize)
        } catch {
            faultLog("MobileSAM encoder prediction failed: \(error)")
            return nil
        }
    }

    // MARK: - Preprocessing (vDSP-accelerated)

    private func preprocess(pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let width  = image.extent.width
        let height = image.extent.height
        let scale  = CGFloat(inputSize) / max(width, height)

        let scaled     = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let tx         = (CGFloat(inputSize) - scaled.extent.width)  / 2.0
        let ty         = (CGFloat(inputSize) - scaled.extent.height) / 2.0
        let translated = scaled.transformed(by: CGAffineTransform(translationX: tx, y: ty))

        // Reuse pre-allocated buffer if available, otherwise allocate.
        guard let buffer = outputBuffer ?? Self.makePixelBuffer(size: inputSize) else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let background = CIImage(color: .black)
            .cropped(to: CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
        let composed = translated.composited(over: background)
        ciContext.render(composed, to: buffer)

        guard let baseAddr = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let n = inputSize * inputSize   // total pixels

        // Build output MLMultiArray [1, 3, H, W]
        guard let array = try? MLMultiArray(shape: [1, 3,
                                                     NSNumber(value: inputSize),
                                                     NSNumber(value: inputSize)],
                                            dataType: .float32) else { return nil }

        // ── Fast BGRA→planar-RGB normalisation using vDSP ──────────────────────
        //
        // Layout of a 32BGRA pixel in memory: [B, G, R, A] (indices 0,1,2,3).
        // We need three interleaved→planar passes, then (val - mean) / std per channel.
        //
        // Strategy:
        //   1. Copy each channel's bytes into a Float scratch buffer via vDSP_vfltu8.
        //   2. Apply (x - mean) / std with vDSP_vsmsa / vDSP_vsdiv.
        //   3. Write directly into the MLMultiArray planes.
        //
        // All pixel rows must be contiguous; since bytesPerRow may include padding we
        // pack rows into a flat UInt8 array first (one pass per channel).

        let bufPtr = baseAddr.assumingMemoryBound(to: UInt8.self)

        // Scratch buffers (stack-allocated for small sizes, heap for 1024²).
        var bBytes = [UInt8](repeating: 0, count: n)
        var gBytes = [UInt8](repeating: 0, count: n)
        var rBytes = [UInt8](repeating: 0, count: n)

        // Deinterleave BGRA → separate byte planes with vImage (single vectorised
        // pass over ~1M pixels; the previous scalar double loop cost 100ms+ per
        // encode on A13). vImageConvert_ARGB8888toPlanar8 splits the four
        // interleaved channels in memory order: for 32BGRA that is B,G,R,A.
        var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: baseAddr),
                                   height: vImagePixelCount(inputSize),
                                   width: vImagePixelCount(inputSize),
                                   rowBytes: bytesPerRow)
        var aBytes = [UInt8](repeating: 0, count: n)  // discarded
        let planarErr: vImage_Error = bBytes.withUnsafeMutableBytes { bRaw in
            gBytes.withUnsafeMutableBytes { gRaw in
                rBytes.withUnsafeMutableBytes { rRaw in
                    aBytes.withUnsafeMutableBytes { aRaw in
                        var bBuf = vImage_Buffer(data: bRaw.baseAddress, height: vImagePixelCount(inputSize),
                                                 width: vImagePixelCount(inputSize), rowBytes: inputSize)
                        var gBuf = vImage_Buffer(data: gRaw.baseAddress, height: vImagePixelCount(inputSize),
                                                 width: vImagePixelCount(inputSize), rowBytes: inputSize)
                        var rBuf = vImage_Buffer(data: rRaw.baseAddress, height: vImagePixelCount(inputSize),
                                                 width: vImagePixelCount(inputSize), rowBytes: inputSize)
                        var aBuf = vImage_Buffer(data: aRaw.baseAddress, height: vImagePixelCount(inputSize),
                                                 width: vImagePixelCount(inputSize), rowBytes: inputSize)
                        return vImageConvert_ARGB8888toPlanar8(&srcBuf, &bBuf, &gBuf, &rBuf, &aBuf,
                                                               vImage_Flags(kvImageNoFlags))
                    }
                }
            }
        }
        if planarErr != kvImageNoError {
            // Fallback: scalar deinterleave (correctness over speed).
            for row in 0..<inputSize {
                let rowBase = bufPtr.advanced(by: row * bytesPerRow)
                let dstOffset = row * inputSize
                for col in 0..<inputSize {
                    let px = rowBase.advanced(by: col * 4)
                    bBytes[dstOffset + col] = px[0]
                    gBytes[dstOffset + col] = px[1]
                    rBytes[dstOffset + col] = px[2]
                }
            }
        }

        // Converts UInt8 → Float, then applies (x + bias) * scale = (x - mean) / std.
        func normalise(bytes: [UInt8],
                       mean: Float, std: Float,
                       dst: UnsafeMutablePointer<Float>) {
            var floatBuf = [Float](repeating: 0, count: n)
            // UInt8 → Float
            vDSP_vfltu8(bytes, 1, &floatBuf, 1, vDSP_Length(n))
            // (x - mean) / std  ==  x * (1/std) + (-mean/std)
            var scale  = 1.0 / std
            var bias   = -mean / std
            vDSP_vsmsa(floatBuf, 1, &scale, &bias, dst, 1, vDSP_Length(n))
        }

        // Writes go through withUnsafeMutableBufferPointer, not the raw
        // dataPointer: MLMultiArray storage may be IOSurface-backed and is only
        // guaranteed valid/synchronised inside this call.
        var wrote = false
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            guard let base = buf.baseAddress, buf.count >= 3 * n else { return }
            // Plane 0 = R, plane 1 = G, plane 2 = B  (SAM expects ImageNet-normalised RGB)
            normalise(bytes: rBytes, mean: meanR, std: stdR, dst: base)
            normalise(bytes: gBytes, mean: meanG, std: stdG, dst: base.advanced(by: n))
            normalise(bytes: bBytes, mean: meanB, std: stdB, dst: base.advanced(by: 2 * n))
            wrote = true
        }
        guard wrote else {
            faultLog("[SAMEncoder] preprocess: unexpected input buffer layout")
            return nil
        }

        return array
    }

    // MARK: - Embedding upsampling (768 AB variant → Decoder 64×64 contract)

    /// Bilinearly resize an image embedding of shape [1, C, srcSize, srcSize] to
    /// [1, C, dstSize, dstSize].  Used only for the 768 AB variant
    /// (48→64) — the 1024 path returns early and never calls this.
    ///
    /// Uses align_corners=false convention (matches PyTorch / CoreML default),
    /// operating per-channel.  Float32 MLMultiArray in/out.
    static func bilinearUpsampleEmbedding(_ src: MLMultiArray,
                                          srcSize: Int,
                                          dstSize: Int) -> MLMultiArray? {
        let shape = src.shape.map { $0.intValue }
        guard shape.count == 4, shape[0] == 1,
              shape[2] == srcSize, shape[3] == srcSize else {
            faultLog("[SAMEncoder] upsample: unexpected embedding shape \(shape)")
            return nil
        }
        let C = shape[1]
        guard let dst = try? MLMultiArray(
            shape: [1, NSNumber(value: C),
                    NSNumber(value: dstSize), NSNumber(value: dstSize)],
            dataType: .float32) else { return nil }

        // Materialise the source as a contiguous [C][srcSize][srcSize] Float copy.
        // MLMultiArray storage may be IOSurface-backed, so it is read through
        // withUnsafeBufferPointer (valid only inside the call) rather than via the
        // raw dataPointer; the stride index math is unchanged.
        let srcStrides = src.strides.map { $0.intValue }
        let srcPlane = srcSize * srcSize
        var flat = [Float](repeating: 0, count: C * srcPlane)
        if src.dataType == .float32 {
            flat.withUnsafeMutableBufferPointer { fbuf in
                src.withUnsafeBufferPointer(ofType: Float.self) { sbuf in
                    for c in 0..<C {
                        for y in 0..<srcSize {
                            let srcRow = c * srcStrides[1] + y * srcStrides[2]
                            let dstRow = c * srcPlane + y * srcSize
                            for x in 0..<srcSize {
                                fbuf[dstRow + x] = sbuf[srcRow + x * srcStrides[3]]
                            }
                        }
                    }
                }
            }
        } else {
            for c in 0..<C {
                for y in 0..<srcSize {
                    let srcRow = c * srcStrides[1] + y * srcStrides[2]
                    let dstRow = c * srcPlane + y * srcSize
                    for x in 0..<srcSize {
                        flat[dstRow + x] = src[srcRow + x * srcStrides[3]].floatValue
                    }
                }
            }
        }

        let dstPlane = dstSize * dstSize
        // scale: map dst pixel centre back to src coordinate (align_corners=false)
        let scale = Float(srcSize) / Float(dstSize)

        dst.withUnsafeMutableBufferPointer(ofType: Float.self) { dstBuf, _ in
            for c in 0..<C {
                let dstBase = c * dstPlane
                let srcBase = c * srcPlane
                for oy in 0..<dstSize {
                    var sy = (Float(oy) + 0.5) * scale - 0.5
                    sy = max(0, min(Float(srcSize - 1), sy))
                    let y0 = Int(sy)
                    let y1 = min(y0 + 1, srcSize - 1)
                    let wy = sy - Float(y0)
                    for ox in 0..<dstSize {
                        var sx = (Float(ox) + 0.5) * scale - 0.5
                        sx = max(0, min(Float(srcSize - 1), sx))
                        let x0 = Int(sx)
                        let x1 = min(x0 + 1, srcSize - 1)
                        let wx = sx - Float(x0)

                        let v00 = flat[srcBase + y0 * srcSize + x0]
                        let v01 = flat[srcBase + y0 * srcSize + x1]
                        let v10 = flat[srcBase + y1 * srcSize + x0]
                        let v11 = flat[srcBase + y1 * srcSize + x1]
                        let top = v00 + (v01 - v00) * wx
                        let bot = v10 + (v11 - v10) * wx
                        dstBuf[dstBase + oy * dstSize + ox] = top + (bot - top) * wy
                    }
                }
            }
        }
        return dst
    }

    // MARK: - Numeric sentinel

    /// SAM image-embedding activations sit far inside this bound; past it the
    /// tensor is corrupt (ANE fp16 numerics, or unsynchronised backing memory).
    private static let maxPlausibleEmbedding: Float = 1_000

    /// Returns nil when the array looks numerically sane, otherwise a short
    /// description for the log.  Sampling every `step`-th element keeps the check
    /// ~free on a 1 M-element embedding while still catching whole-tensor
    /// corruption (which is never confined to a few values).
    static func numericIssue(in array: MLMultiArray,
                             limit: Float = maxPlausibleEmbedding,
                             label: String) -> String? {
        guard array.dataType == .float32 else { return nil }
        let n = array.count
        guard n > 0 else { return "\(label) is empty" }
        var maxAbs: Float = 0
        var nonFinite = 0
        let step = max(1, n / 4096)
        array.withUnsafeBufferPointer(ofType: Float.self) { buf in
            var i = 0
            let bound = min(n, buf.count)
            while i < bound {
                let v = buf[i]
                if v.isFinite { maxAbs = max(maxAbs, abs(v)) } else { nonFinite += 1 }
                i += step
            }
        }
        if nonFinite > 0 { return "\(label): \(nonFinite) non-finite samples" }
        if maxAbs > limit { return String(format: "%@: |max|=%.4g > %.4g", label, maxAbs, limit) }
        if maxAbs == 0 { return "\(label): all sampled values are zero" }
        return nil
    }

    // MARK: - Helpers

    private static func makePixelBuffer(size: Int) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey:          kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey:                    size,
            kCVPixelBufferHeightKey:                   size,
            kCVPixelBufferCGImageCompatibilityKey:     true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        var buf: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                                         kCVPixelFormatType_32BGRA,
                                         attrs as CFDictionary, &buf)
        return status == kCVReturnSuccess ? buf : nil
    }
}
