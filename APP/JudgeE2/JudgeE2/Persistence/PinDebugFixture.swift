//
//  PinDebugFixture.swift
//  JudgeE2
//
//  Phase 4B — Day 5 (Builder) · B-29
//
//  Contract: architect_output.md §22.1.6 (scope + why not a test target) and
//  §22.1.5 / §37.6.1 (what P-1…P-4 need to be executable at all).
//
//  `#if DEBUG` only — compiled into the App target, not `JudgeE2Tests` (§22.1.6:
//  P-3's kill-and-restart protocol has to kill the *host app process*, which a
//  separate test-process target cannot do; and `JudgeE2Tests`' Sources phase is
//  empty and out of scope for this ruling anyway).  Does not exist in Release.
//
//  ⛔ Constraints (carried over verbatim from §22.1.6 / §37.6.1):
//    • must not reference `TapInstance` or any re-anchor field
//    • must not post any work to `videoQueue` / `decoderQueue` / `encoderQueue`
//    • only calls the `PinStore` protocol surface already exposed
//      (`create` / `flush`) — never touches `CameraManager`.
//
//  Triggered by a launch argument, read once in `JudgeE2App.init()`
//  (`#if DEBUG`), never from any UI control:
//
//      -PinFixtureBatch 10:500     ⇒ batchSave(count: 10, intervalMs: 500)
//

import Foundation

#if DEBUG
enum PinDebugFixture {

    /// Construct one synthetic `(PinRecordV1, maskAlpha)` pair without going
    /// through a live tap / decode at all — the only way P-1's five boundary
    /// classes (empty tag, a 64-UTF16-unit tag, an emoji+newline note, a
    /// `maskFile == nil` record, a minimal-area mask) can be produced on
    /// demand, since none of them is reachable by tapping a real scene.
    ///
    /// - Parameters:
    ///   - tag: forwarded as-is (⛔ NOT normalised here — that is
    ///     `FilePinStore`'s job, B-35; a fixture that pre-normalised would
    ///     hide exactly the bug class P-1 is meant to catch).
    ///   - note: forwarded as-is, same reasoning.
    ///   - maskAreaPx: number of non-zero pixels in the synthetic 256×256
    ///     mask (clamped to `[0, MaskPNGCodec.pixelCount]`); pass `1` for the
    ///     "minimal-area mask" boundary case.  Ignored when `includeMaskFile`
    ///     is `false`.
    ///   - includeMaskFile: `false` produces the `maskFile == nil` boundary
    ///     case — a record with no accompanying mask bytes at all.
    /// - Returns: a record ready for `PinStore.create`, plus the mask bytes to
    ///   pass alongside it (`nil` when `includeMaskFile` is `false`).
    static func makeSyntheticRecord(tag: String?,
                                    note: String?,
                                    maskAreaPx: Int?,
                                    includeMaskFile: Bool) -> (PinRecordV1, [UInt8]?) {
        let id = UUID().uuidString.lowercased()
        let now = Date().timeIntervalSince1970

        var alpha: [UInt8]?
        var maskFile: String?
        var maskNonZero = 0

        if includeMaskFile {
            let area = min(max(maskAreaPx ?? 0, 0), MaskPNGCodec.pixelCount)
            var buffer = [UInt8](repeating: 0, count: MaskPNGCodec.pixelCount)
            for i in 0..<area { buffer[i] = 255 }
            alpha = buffer
            maskNonZero = area
            maskFile = PinRecordV1.maskFileName(for: id)
        }

        // Geometry is a fixed, self-consistent stand-in (§19.2.3's two required
        // quantities, origSize + promptSpace) — the fixture never touches
        // `FrameGeometry` or the camera pipeline, so there is no live geometry
        // to snapshot.
        let geometry = PinGeometryV1(origW: 1080, origH: 1920,
                                     promptSpace: SAMConfiguration.pointPromptSpace,
                                     rotationDeg: 0, mirrored: false,
                                     encoderInputSize: SAMConfiguration.encoderInputSize)

        let record = PinRecordV1(
            id: id,
            pointX: Double.random(in: 0..<1080),
            pointY: Double.random(in: 0..<1920),
            geometry: geometry,
            maskFile: maskFile,
            maskWidth: MaskPNGCodec.side,
            maskHeight: MaskPNGCodec.side,
            maskNonZero: maskNonZero,
            iouPredAtCreation: nil,
            createdAt: now,
            tag: tag,
            note: note,
            updatedAt: now,
            previewFile: nil)

        return (record, alpha)
    }

    /// Save `count` synthetic Pins, `intervalMs` apart, driving `store` exactly
    /// the way real UI traffic would: through `PinStore.create` on the main
    /// thread, nothing more privileged.  Serves:
    ///   • P-4's B arm ("script saves 1 Pin every 500 ms")
    ///   • P-3(iii) ("mid-burst of N consecutive saves, kill the process")
    ///
    /// Each save's tag is `"fixture-<i>"` so a `[PIN] create … ok/FAILED` log
    /// line can be attributed to this driver rather than to a real user
    /// action.  Uses `DispatchQueue.main.asyncAfter` — the existing main
    /// queue, not a new one.
    static func batchSave(store: PinStore, count: Int, intervalMs: Int) {
        guard count > 0 else { return }

        func saveOne(_ i: Int) {
            let (record, alpha) = makeSyntheticRecord(tag: "fixture-\(i)",
                                                       note: nil,
                                                       maskAreaPx: 4096,
                                                       includeMaskFile: true)
            store.create(record, maskAlpha: alpha) { result in
                switch result {
                case .success:
                    pinLog("[PIN][FIXTURE] batchSave \(i + 1)/\(count) ok")
                case .failure(let err):
                    pinFault("[PIN][FIXTURE] batchSave \(i + 1)/\(count) FAILED err=\(err)")
                }
            }
            let next = i + 1
            guard next < count else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(intervalMs)) {
                saveOne(next)
            }
        }
        saveOne(0)
    }

    /// Parse `-PinFixtureBatch <count>:<intervalMs>` out of the process launch
    /// arguments and, if present, kick off `batchSave`.  Called once from
    /// `JudgeE2App.init()`.  A malformed or absent argument is a silent no-op —
    /// this is a debug-only convenience, not a user-facing control surface.
    static func runFromLaunchArgumentsIfPresent(store: PinStore) {
        let args = ProcessInfo.processInfo.arguments
        guard let flagIndex = args.firstIndex(of: "-PinFixtureBatch"),
              flagIndex + 1 < args.count else { return }
        let parts = args[flagIndex + 1].split(separator: ":")
        guard parts.count == 2,
              let count = Int(parts[0]),
              let intervalMs = Int(parts[1]) else { return }
        pinLog("[PIN][FIXTURE] launch-argument batchSave count=\(count) intervalMs=\(intervalMs)")
        batchSave(store: store, count: count, intervalMs: intervalMs)
    }
}
#endif
