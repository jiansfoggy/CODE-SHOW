import Foundation
import CoreML
import CoreVideo

enum ModelLoader {
    static func testLoad(backend: InferenceBackend = .all) {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = backend.computeUnits
            let start = CFAbsoluteTimeGetCurrent()
            let model = try yolov9_c(configuration: config)
            let end = CFAbsoluteTimeGetCurrent()
            let elapsedMs = (end - start) * 1000.0
            PerfLogger.logComputeUnits(config)
            PerfLogger.logInputInfo(model.model)
            perfLog(String(format: "Model loaded in %.2f ms", elapsedMs))
        } catch {
            faultLog("Model load failed: \(error)")
        }
    }

    static func testSingleInference(backend: InferenceBackend = .all) {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = backend.computeUnits
            let model = try yolov9_c(configuration: config)
            PerfLogger.logComputeUnits(config)
            PerfLogger.logInputInfo(model.model)

            guard let pixelBuffer = makeDummyPixelBuffer(width: 640, height: 640) else {
                faultLog("Failed to create dummy pixel buffer")
                return
            }

            let start = CFAbsoluteTimeGetCurrent()
            let output = try model.prediction(image: pixelBuffer)
            let end = CFAbsoluteTimeGetCurrent()

            PerfLogger.logOutputShapesOnce(var_3019: output.var_3019, var_3022: output.var_3022)

            let elapsedMs = (end - start) * 1000.0
            perfLog(String(format: "Single inference time: %.2f ms", elapsedMs))
        } catch {
            faultLog("Single inference failed: \(error)")
        }
    }

    static func testMobileSAMLoad() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            let memBefore = PerfLogger.currentMemoryMB()
            let start = CFAbsoluteTimeGetCurrent()

            // Phase 3 Builder: load milfix encoder instead of original fp16.
            // Load priority mirrors SAMEncoder.init(): milfix > fp32 > original fp16.
            //
            // ⚠️ CORRECTION (debug_report.md §16.5): an earlier version of this
            // comment claimed the original MobileSAM_ImageEncoder emitted the three
            // `Invalid layer: Invalid input tensor channel 1 and format size 2 bytes,
            // must be aligned on 64 bytes` warnings.  That was a misattribution.
            // The Debugger's isolated-load experiment measured, on device:
            //   - milfix encoder loaded alone .............. 0 warnings
            //   - MobileSAM_PromptMaskDecoder loaded alone . 3 warnings
            // and the decoder package has never changed while pre-milfix builds
            // showed the same count of 3, so the original encoder contributed 0 too.
            // The warnings below therefore come from the decoder load a few lines
            // down, not from this encoder.  They are noise, not a fault: TAP#1
            // logits stayed at min=-9.74/max=3.60 and every iou_pred inside [0,1].
            // Swapping in milfix may still pay off for other reasons — "removes the
            // alignment warnings" is simply not one of them.
            let encoderModel: MLModel
            let encoderName: String
            if let url = Bundle.main.url(forResource: "MobileSAM_ImageEncoder_fp16_milfix",
                                         withExtension: "mlmodelc") {
                encoderModel = try MLModel(contentsOf: url, configuration: config)
                encoderName  = "MobileSAM_ImageEncoder_fp16_milfix"
            } else if let url = Bundle.main.url(forResource: "MobileSAM_ImageEncoder_fp32",
                                                withExtension: "mlmodelc") {
                encoderModel = try MLModel(contentsOf: url, configuration: config)
                encoderName  = "MobileSAM_ImageEncoder_fp32"
            } else {
                encoderModel = try MobileSAM_ImageEncoder(configuration: config).model
                encoderName  = "MobileSAM_ImageEncoder"
            }
            perfLog("[ModelLoader] encoder loaded: \(encoderName)")
            let encoder = encoderModel   // used for PerfLogger.logInputInfo below
            let memAfterEncoder = PerfLogger.currentMemoryMB()

            let decoder = try MobileSAM_PromptMaskDecoder(configuration: config)
            let end = CFAbsoluteTimeGetCurrent()
            let loadMs = (end - start) * 1000.0
            let memAfter = PerfLogger.currentMemoryMB()

            PerfLogger.logComputeUnits(config)
            perfLog(String(format: "MobileSAM models loaded in %.2f ms", loadMs))
            perfLog(String(format: "MobileSAM load memory: before=%.1f MB | afterEncoder=%.1f MB | afterDecoder=%.1f MB",
                         memBefore, memAfterEncoder, memAfter))
            perfLog(String(format: "MobileSAM load delta: encoder=%.1f MB | decoder=%.1f MB | total=%.1f MB",
                         memAfterEncoder - memBefore, memAfter - memAfterEncoder, memAfter - memBefore))
            PerfLogger.logInputInfo(encoder)          // encoder is already MLModel
            PerfLogger.logInputInfo(decoder.model)

            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                let memSettled = PerfLogger.currentMemoryMB()
                perfLog(String(format: "MobileSAM load memory (settled +2s): %.1f MB | delta=%.1f MB",
                             memSettled, memSettled - memBefore))
            }
        } catch {
            faultLog("MobileSAM load failed: \(error)")
        }
    }

    private static func makeDummyPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        if let baseAddress = CVPixelBufferGetBaseAddress(buffer) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            memset(baseAddress, 0, bytesPerRow * height)
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }
}
