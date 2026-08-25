//
//  MaskPNGCodec.swift
//  JudgeE2
//
//  Phase 4B — Day 4 (Builder) · B-22
//
//  Lossless serialisation of a tap mask: 256×256 `[UInt8]` with values in
//  {0, 255} ⇄ 8-bit grayscale PNG.  Contract: architect_output.md §19.2.6.
//
//  ⛔ 256×256, NOT 128×128.
//  128 corresponds to no invariant anywhere in this system.  The real invariant
//  is 256×256 — `MaskRenderer.AlphaResult.alpha`, `compositeLayers`' 256*256
//  guard, `TapInstance.maskAlpha` (§35.6.3).  Downscaling a *binary* mask means
//  resampling it into intermediate greys and then re-binarising through some
//  threshold — a parameter that has never been adjudicated, which is precisely
//  the class of defect that produced ISSUE-P4-GATE.  It would also save about
//  1 KB.  The 128 pt list thumbnail is produced by the display layer at render
//  time (UIImage / contentsGravity), at zero persistence cost.
//
//  Consequently this file has:
//    • no interpolation      (`shouldInterpolate: false`, `.none` quality)
//    • no resampling         (source and destination are both 256×256)
//    • no threshold parameter — there is nothing here to tune.
//
//  A DEBUG assertion checks encode→decode byte identity on every encode, so a
//  colour-management or row-padding regression surfaces at the call site that
//  caused it rather than months later as "the thumbnails look wrong".
//

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// B-34 (§22.1.7): `nonisolated` — a stateless codec called from `pin.store.io`
/// (`FilePinStore.create` / `loadMaskImage`) and, for the DEBUG round-trip
/// assertion, from wherever `encode` is called.  No actor affinity anywhere.
nonisolated enum MaskPNGCodec {

    /// The one and only mask side length (§35.6.3).
    static let side = 256
    static let pixelCount = side * side

    enum CodecError: Error, CustomStringConvertible {
        case badPixelCount(expected: Int, got: Int)
        case imageCreationFailed
        case destinationCreationFailed
        case encodeFailed
        case sourceCreationFailed
        case decodeFailed
        case unexpectedDimensions(width: Int, height: Int)

        var description: String {
            switch self {
            case .badPixelCount(let e, let g):     return "badPixelCount(expected:\(e) got:\(g))"
            case .imageCreationFailed:             return "imageCreationFailed"
            case .destinationCreationFailed:       return "destinationCreationFailed"
            case .encodeFailed:                    return "encodeFailed"
            case .sourceCreationFailed:            return "sourceCreationFailed"
            case .decodeFailed:                    return "decodeFailed"
            case .unexpectedDimensions(let w, let h): return "unexpectedDimensions(\(w)x\(h))"
            }
        }
    }

    // MARK: - Encode

    /// 256×256 binary alpha ⇒ 8-bit grayscale PNG bytes.
    static func encode(_ alpha: [UInt8]) throws -> Data {
        guard alpha.count == pixelCount else {
            throw CodecError.badPixelCount(expected: pixelCount, got: alpha.count)
        }

        guard let provider = CGDataProvider(data: Data(alpha) as CFData) else {
            throw CodecError.imageCreationFailed
        }
        guard let image = CGImage(width: side,
                                  height: side,
                                  bitsPerComponent: 8,
                                  bitsPerPixel: 8,
                                  bytesPerRow: side,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider,
                                  decode: nil,
                                  shouldInterpolate: false,
                                  intent: .defaultIntent)
        else {
            throw CodecError.imageCreationFailed
        }

        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out as CFMutableData,
                                                          UTType.png.identifier as CFString,
                                                          1, nil)
        else {
            throw CodecError.destinationCreationFailed
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CodecError.encodeFailed }

        let data = out as Data

        #if DEBUG
        // B-22: the round-trip must be byte-identical.  Cheap (≈65 k bytes) and
        // it only runs in DEBUG, so it never touches a measurement session.
        if let decoded = try? decode(data) {
            assert(decoded == alpha,
                   "[PIN] MaskPNGCodec round-trip is not byte-identical — the blob format has drifted")
        } else {
            assertionFailure("[PIN] MaskPNGCodec produced a blob it cannot decode")
        }
        #endif

        return data
    }

    // MARK: - Decode

    /// 8-bit grayscale PNG bytes ⇒ 256×256 alpha.
    ///
    /// Throws `unexpectedDimensions` rather than resizing: a blob that is not
    /// 256×256 is a *degraded* record (§19.5.4), and the caller's job is to
    /// exclude it from compositing, not to guess it back into shape.
    static func decode(_ data: Data) throws -> [UInt8] {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image  = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw CodecError.sourceCreationFailed
        }
        guard image.width == side, image.height == side else {
            throw CodecError.unexpectedDimensions(width: image.width, height: image.height)
        }

        var buffer = [UInt8](repeating: 0, count: pixelCount)
        let ok: Bool = buffer.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let ctx = CGContext(data: base,
                                      width: side,
                                      height: side,
                                      bitsPerComponent: 8,
                                      bytesPerRow: side,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { return false }
            ctx.interpolationQuality = .none
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
            return true
        }
        guard ok else { throw CodecError.decodeFailed }
        return buffer
    }

    // MARK: - Helpers

    /// Non-zero pixel count over the full 256×256 tile.
    ///
    /// Full traversal, `stride: 1` — 65 536 iterations is 30–60 µs, whereas a
    /// strided estimate would turn a self-check into a coin flip (§35).
    static func nonZeroCount(_ alpha: [UInt8]) -> Int {
        var n = 0
        for v in alpha where v != 0 { n += 1 }
        return n
    }
}
