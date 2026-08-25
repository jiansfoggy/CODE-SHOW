//
//  DriftDetector.swift
//  JudgeE2
//
//  Phase 4 — Day 2–3 (Builder), revised per architect_output.md §17.
//
//  Drift signal for the re-anchor loop: **anchor-region content divergence**
//  (§17.3).  Contract: §17.3 (signal), §17.4 (thresholds), §17.5 (interface).
//
//  ⛔ WHAT THIS FILE USED TO BE, AND WHY IT ISN'T ANY MORE.
//  §16.3.1 nominated `letterboxOffset` + `videoRotationAngle` as the drift
//  components.  Both are *calibration* quantities: `letterboxToSquare` derives
//  them from the camera buffer's dimensions and the fixed 640 model input alone,
//  so panning the device leaves them bit-identical and `hasDrifted` could never
//  fire.  §17.1 confirms the defect and supersedes that table; §17.5 replaces
//  the whole `FrameGeometry`-based interface.  No compatibility overload is left
//  behind — a predicate that is permanently asleep is the easiest thing in the
//  world to call by accident.
//
//  WHAT IT MEASURES NOW.  The input is `latestCameraBuffer`, the raw capture
//  buffer `AVCaptureVideoDataOutput` hands over every frame (CameraManager.swift
//  :2645).  Its contents ARE what the camera is looking at — an observed
//  quantity, not a computed one (§17.1.3's A-7 self-check).  Camera moves ⇒
//  window contents change; camera still but the object moves ⇒ they change too;
//  everything still ⇒ only sensor noise.  Those are exactly the three states a
//  refresh loop has to tell apart.
//
//  SCOPE GUARD (R3): this file decides *when* a re-anchor batch may fire and,
//  via `alphaIoU`, whether a produced mask is geometrically consistent with the
//  one it would replace.  It carries no mask-quality logic — no candidate
//  selection, no `iou_pred` / stability reading, no area caps, no coordinate
//  transform, no model call.  `MaskRenderer.buildTapAlpha` is untouched and its
//  verdict is accepted in full; the consistency gate sits strictly *outside and
//  after* it and can only ever veto, falling back to the already-approved
//  "keep the previous mask" branch (§17.3.3).
//

import CoreGraphics
import CoreVideo
import Foundation

// MARK: - AnchorSignature

/// The content of one anchor neighbourhood on one frame, reduced to a small
/// grid of box-averaged luma samples (§17.3.2).
///
/// Storage is a fixed 64-byte inline tuple, never a heap array: §17.6's first
/// implementation constraint asks for inline or reused storage so the per-frame
/// sampling allocates nothing.  That caps the grid at 8×8 = 64 samples, which is
/// the shipped `anchorGridSide = 8`; raising the grid past 8 requires widening
/// `Storage` here, and `DriftDetector.effectiveGridSide` clamps to that bound so
/// a mis-tuned constant degrades visibly rather than corrupting memory.
struct AnchorSignature {

    /// Inline capacity in samples (8 × UInt64 = 64 bytes).
    static let capacity = 64

    private typealias Storage = (UInt64, UInt64, UInt64, UInt64,
                                 UInt64, UInt64, UInt64, UInt64)
    private var storage: Storage = (0, 0, 0, 0, 0, 0, 0, 0)

    /// How many of the 64 slots carry a sample.  Two signatures are comparable
    /// only when their counts match (a re-tuned `anchorGridSide` mid-session
    /// invalidates the baseline; the caller re-seeds instead of comparing).
    let count: Int

    /// Build a signature by letting `fill` write `count` bytes into the inline
    /// storage.  Fails when `count` exceeds the inline capacity.
    init?(count: Int, fill: (UnsafeMutableBufferPointer<UInt8>) -> Void) {
        guard count > 0, count <= Self.capacity else { return nil }
        self.count = count
        withUnsafeMutableBytes(of: &storage) { raw in
            let base = raw.bindMemory(to: UInt8.self)
            fill(UnsafeMutableBufferPointer(start: base.baseAddress!, count: count))
        }
    }

    /// Read-only access to the samples, without copying them out.
    @inline(__always)
    func withSamples<R>(_ body: (UnsafeBufferPointer<UInt8>) -> R) -> R {
        withUnsafeBytes(of: storage) { raw in
            let base = raw.bindMemory(to: UInt8.self)
            return body(UnsafeBufferPointer(start: base.baseAddress!, count: count))
        }
    }
}

// MARK: - DriftDetector

/// Decides whether the content under a tap anchor has changed enough to justify
/// re-decoding the live tap instances, and whether a re-decoded mask may replace
/// the one it was meant to refresh.
enum DriftDetector {

    // MARK: - Tunable constants (§17.4)
    //
    // The §16.1.2 values (10.0 pt / 3.0°) are NOT migrated: the unit changed, so
    // the numbers carry no information.  The tuning-surface *shape* Builder
    // established is kept verbatim — named `static var`s a device session can
    // reassign with no other code change.

    /// Drift threshold, in 8-bit luma levels of mean-removed MAD (§17.4).
    /// Sensor noise σ ≈ 1–3 levels puts the mean-removed noise floor near 1–2;
    /// panning a textured window's worth of content gives 20–60.  8.0 sits ≈4×
    /// above the floor and well under real motion.
    /// Raise it if a still camera keeps firing; lower it if obvious content
    /// changes do not refresh.
    static var contentThresholdLuma: Double = 8.0

    /// Side length of the sampled window around `canonicalPoint`, canonical px
    /// (§17.4).  ~8.9 % of a 1080p short edge.  Too small and it lands inside a
    /// flat colour patch and never moves; too large and unrelated background
    /// dominates it.
    static var anchorWindowPx: CGFloat = 96.0

    /// Samples per window edge (§17.4).  8 ⇒ 64 samples, the inline capacity of
    /// `AnchorSignature`.  Normally not touched; changing it moves the noise
    /// floor and therefore requires re-checking `contentThresholdLuma`.
    /// Values outside 1...8 are clamped (see `effectiveGridSide`).
    static var anchorGridSide: Int = 8

    /// Hard lower bound between two re-anchor batches, ms (§17.4).
    /// D-15.2 explicitly allows a fixed interval **as a lower bound**; the
    /// load-adaptive throttle is still the single-in-flight-batch machine of
    /// §16.2.3.  300 ms caps firing at ≈3.3 batches/s, i.e. ≈64 % duty at N = 3
    /// with the measured 63.7 ms decode, leaving room for background encode.
    static var minReAnchorIntervalMs: Double = 300.0

    /// Consistency-gate threshold (§17.4, comparison basis revised by §18.2.5):
    /// a refreshed mask whose IoU with **the alpha this instance's tap
    /// produced** (`TapInstance.originAlpha`) is below this is rejected and the
    /// currently displayed mask kept.  0.5 separates "the same object segmented
    /// twice" from "a different object" loosely but unambiguously; the gate is
    /// deliberately biased towards letting a slightly-off refresh through rather
    /// than blocking a correct one.
    ///
    /// ⚠️ The number is unchanged, **the quantity it thresholds is not**: under
    /// §17.3.3 it scored the similarity of two consecutive refreshes; under RE-3
    /// it scores similarity to the user's original selection.  Same digits,
    /// different measurement (§18.2.5).  ⛔ R21: it may **not** be re-tuned
    /// until D-6' produces an acceptance rate under the new basis — the 0/216
    /// reading from the Day 2–3 session was taken against the old basis and is
    /// void as an input.
    static var reAnchorAcceptIoU: Double = 0.5

    /// Capability C's own consistency-gate threshold (§24.1.5 / §24.4 B-46),
    /// for `centroidAlignedIoU` — a physically DIFFERENT quantity than
    /// `reAnchorAcceptIoU` above, in a different coordinate space, with its
    /// own independent `static var` storage. It is NOT a rename or an alias
    /// of `reAnchorAcceptIoU`, and the two must never be conflated: raw
    /// `alphaIoU` is measured in the absolute (un-translated) frame, while
    /// `centroidAlignedIoU` re-centres both masks first, so the three-cluster
    /// [0.02–0.25]/[0.57–0.89]/1.00 empirical framework built up around
    /// `reAnchorAcceptIoU` (§20/§21/§23.2.10) does not transfer here — this is
    /// a different, unmeasured quantity, not a re-tuning of a known one.
    ///
    /// 0.5 is a zero-data starting value (same status `reAnchorAcceptIoU` had
    /// at its own inception): §24.1.5's reasoning for that number — "biased
    /// towards letting a slightly-off refresh through rather than blocking a
    /// correct one" — applies unchanged in the centroid-aligned space, but
    /// this is an independent engineering judgement, not an inherited one.
    /// Real values must come from an on-device three-cluster capture
    /// (migrated / tracked-same-object / no-op) before this number is
    /// trusted (§24.4 §24.5 P-15).
    ///
    /// ⛔ R21 continues to forbid retuning `reAnchorAcceptIoU` itself; this is
    /// a separate constant and touching it is not covered by, and does not
    /// touch, that constraint.
    static var trackConsistencyAcceptIoU: Double = 0.5

    /// Single-variable rollback switch for the consistency gate (C-6 discipline,
    /// §17.4).  Exists so the Debugger can run a controlled A/B; **must not ship
    /// disabled** — without the gate the refresh loop is a net regression
    /// (§17.2.0 capability B).
    static var reAnchorConsistencyGateEnabled: Bool = true

    /// Feature master switch for the whole re-anchor loop (§18.3.2 / B-16).
    ///
    /// ✅ **SHIPS `true`.**  The 4th re-anchor device round cleared the ⏱️ STOP
    /// RULE on both criteria, so the 「暂停合入」 hold recorded in D-4 is lifted
    /// and Day 2–3 closes:
    ///   • D-6''  — the consistency gate is demonstrably live: 11 `[REANCHOR]`
    ///     lines, 9 with iou ≠ 1.00 spanning 0.02–0.89 (the two 1.00 readings
    ///     are genuine bit-identical no-op re-anchors within one embedding
    ///     generation, not the pre-fix constant).  Rejection rate 45 % (5/11),
    ///     which finally gives R21 an input.
    ///   • D-1c'' — no mask migration: the 5 rejections blocked masks 47.5× /
    ///     30.1× / 20.1× / 36.9× / 1.7× larger than the frozen origin, and the
    ///     screen recording shows the mask held at ~490 px throughout.
    ///   • REC-1 — accepted refreshes stayed area-stable (482→481, 482→487,
    ///     491→490, 491→487).
    /// This flip is the Architect's separate approval under §18.3.5; a Builder
    /// still may not change it unilaterally.  OFF ⇒ `checkAndFireReAnchor`
    /// returns at its condition 0 and `.tapToSegment` behaves bit-identically to
    /// the frozen Phase 3 build, which remains the rollback path.
    ///
    /// ⚠️ **OPPOSITE POLARITY TO `reAnchorConsistencyGateEnabled`, and the two
    /// must never be confused.**  That one is the *safe half*'s single-variable
    /// control (C-6) and §17.4 forbids shipping it disabled; this one is the
    /// *whole feature*'s merge gate, now authorised to ship enabled.
    static var reAnchorEnabled: Bool = true

    /// Capability C's own master switch — in-session object tracking (§24.2 /
    /// §24.4's "主开关" row). Mirrors `reAnchorEnabled`'s shape (B-16): a single
    /// `static var` a future Architect approval can flip with no other code
    /// change.
    ///
    /// ⛔ **SHIPS `false`.** OFF ⇒ every `TapInstance.trackState` stays
    /// `.locked` forever (B-42's guarantee, `TapInstanceManager.addInstance`)
    /// ⇒ `trackedPoint` is never written a second time ⇒ `trackedPoint ≡
    /// canonicalPoint` for the instance's whole life ⇒ every new branch B-44 /
    /// B-45 add in `checkAndFireReAnchor` / `reAnchorDecode` that is gated on
    /// `objectTrackingEnabled` or on `trackState != .locked` is unreachable,
    /// and `.tapToSegment` behaves bit-identically to the pre-capability-C
    /// build. Same C-6 single-variable-rollback discipline as
    /// `reAnchorEnabled`.
    ///
    /// Flipping this `true` requires ALL THREE of the following (§24.4's
    /// "主开关" row, verbatim — a Builder may not decide this alone, same
    /// discipline §18.3.5 condition 6 already established for
    /// `reAnchorEnabled`):
    ///   1. B-42 … B-47 all landed and compiling.
    ///   2. §24.5's judging criteria (P-10 … P-17) have completed one
    ///      on-device reading pass.
    ///   3. `ISSUE-P4-DECODE` (§24.0.2 condition 2) is closed, OR the user has
    ///      explicitly waived that precondition.
    /// Architect sign-off required for the flip itself; this constant landing
    /// as part of B-44/B-45 does not flip it.
    ///
    /// Re-enabled `true` (2026-08-24, R36 remediation, user-directed — see
    /// shared/tasks.md Day 7 re-anchor item). Was reverted to `false` after a
    /// controlled on-device test session confirmed R36: the local
    /// block-matching search (§24.2.2 candidate B) cannot distinguish camera
    /// ego-motion from object motion, and a real-device session produced a
    /// confirmed case of the displayed mask visibly jumping to the wrong
    /// position while the user panned the phone (object itself stationary).
    ///
    /// Remediation implemented: `CameraMotionGate.swift` adds the
    /// suppression gate architect_output.md §17.2.1 pre-approved for exactly
    /// this failure mode — while the device's gyroscope reports it panning
    /// (`CameraMotionGate.isPanning`), `CameraManager.checkAndFireReAnchor`
    /// skips the tracking/recovery search entirely for that cycle, falling
    /// back to Capability A/B's frozen-`canonicalPoint` behaviour (identical
    /// to this flag being `false`). This does not fix candidate B's inherent
    /// inability to tell object motion from camera motion — it prevents the
    /// search from ever running during the one scenario (a panning camera)
    /// where that blindness produces a wrong-object jump instead of just a
    /// missed real-object track. A still phone with a moving object is
    /// unaffected and continues to use the search exactly as before.
    ///
    /// ⚠️ Not yet re-validated on a real device after this fix — see the
    /// Day 7 tasks.md entry for what evidence exists and what is still
    /// outstanding. `CameraMotionGate.panningThresholdRadPerSec` (0.20
    /// rad/s) is a first-cut value, not yet tuned against real pan-vs-hold
    /// data.
    static var objectTrackingEnabled: Bool = true

    /// B-47 (§24.3.2) — heavy-drift force-refresh threshold, in the same 8-bit
    /// mean-removed MAD luma units as `contentThresholdLuma`. 4× that constant
    /// (8.0) — far above the "worth a decode" bar; only a genuinely severe
    /// divergence should justify the much more expensive re-*encode* this
    /// bypass triggers, not merely the ordinary re-anchor-worthy drift that
    /// `contentThresholdLuma` already gates. Independent storage: not derived
    /// from `contentThresholdLuma` at read time, so retuning one does not move
    /// the other.
    static var heavyDriftForceRefreshLuma: Double = 32.0

    /// B-47 (§24.3.2) — minimum cache age, ms, before heavy drift is allowed to
    /// force a background re-encode. Even a severe single-frame divergence must
    /// not re-trigger a refresh whose cache is still effectively brand new —
    /// without this floor, a cache refreshed 100 ms ago could be torn down
    /// again by one transient large-divergence reading, which is jitter, not a
    /// real scene change. Independent of `minReAnchorIntervalMs` (governs
    /// re-anchor decode cadence, not re-encode).
    static var minHeavyDriftAgeFloorMs: Double = 1500.0

    /// B-47 (§24.3.2) — independent cooldown between two heavy-drift-triggered
    /// re-encodes, ms. Deliberately NOT the same clock as `minReAnchorIntervalMs`
    /// (300 ms): re-encoding the SAM image embedding is far more expensive than
    /// a decode-only re-anchor, so it needs its own, much wider throttle window.
    /// 5000 ms means this bypass can at most double the steady-state re-encode
    /// cadence set by the existing 5 s TTL refresh — it is a bounded speed-up,
    /// not an unbounded one.
    static var minHeavyDriftRefreshIntervalMs: Double = 5000.0

    /// DEBUG ONLY — treat every comparison as drifted (§17.7).
    ///
    /// Ratified and retained, with its meaning pinned down: data produced with
    /// this ON is valid for **mechanism** (qwait, decode, batch counters,
    /// throttle deadlock, memory, main-thread cost, log plumbing) and invalid
    /// for **behaviour** (whether a refresh happened when it should have, the
    /// gate's acceptance rate, false-fire rate — anything about the signal's
    /// quality).  Any report quoting `[REANCHOR]` data must state this switch's
    /// state for that session.  Default `false`; never enabled in a shipped
    /// build; never exposed in UI.
    static var forceDriftForTesting: Bool = false

    /// `anchorGridSide` clamped to what the inline storage can hold.
    static var effectiveGridSide: Int {
        min(max(anchorGridSide, 1), Int(Double(AnchorSignature.capacity).squareRoot()))
    }

    // MARK: - Result (§17.6 B-3)

    /// One divergence measurement plus its verdict.
    ///
    /// `hasDrifted` alone is not enough for the caller: §17.5.3 requires the
    /// `[REANCHOR]` line to carry the magnitude **that triggered this batch**,
    /// per instance, captured at the moment of the check.  Recomputing it inside
    /// the decode closure would sample a later frame and report a number that
    /// triggered nothing.  This is the same design reason §16.2.5 gave for the
    /// old two-component `Drift`; only the components changed.
    struct Drift {
        /// Mean-removed mean absolute difference, in 8-bit luma levels.
        let divergenceLuma: Double
        /// True when the divergence is over `contentThresholdLuma` (or the test
        /// override is on).
        let exceedsThreshold: Bool
    }

    // MARK: - Sampling (§17.3.2)

    /// Sample the anchor neighbourhood of `point` on `buffer`.
    ///
    /// ⚠️ **videoQueue only.**  The only buffer that may be passed here is
    /// `CameraManager.latestCameraBuffer`, which videoQueue owns; it is also the
    /// only one that is safe to lock — `latestInputBuffer` is a `CIContext`
    /// render target and locking its base address stalls on the GPU (§17.3.2).
    ///
    /// Returns `nil` when the buffer is not the expected 32BGRA layout, when the
    /// base address cannot be locked, or when the window degenerates.  §17.4's
    /// graceful-degradation rule: the caller treats `nil` as "no drift" and
    /// returns — silently not firing costs a slightly staler mask (a known,
    /// bounded Phase 4A limit), while firing on a guess costs a misplaced decode
    /// plus a main-thread hop.
    static func signature(from buffer: CVPixelBuffer,
                          atCanonical point: CGPoint) -> AnchorSignature? {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else {
            return nil
        }
        let grid = effectiveGridSide
        let sampleCount = grid * grid
        guard sampleCount <= AnchorSignature.capacity else { return nil }

        guard CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
            return nil
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        guard width > 0, height > 0, bytesPerRow >= width * 4 else { return nil }

        // The capture buffer and canonical space are the same space, 1:1, with no
        // transform: `LetterboxInfo.origW/origH` come from this buffer's extent
        // and rotation is already baked in by `AVCaptureConnection`
        // (§17.3.2).  So `canonicalPoint` indexes this buffer directly.
        let px = UnsafeRawPointer(base).assumingMemoryBound(to: UInt8.self)
        let step = Double(anchorWindowPx) / Double(grid)
        let x0 = Double(point.x) - Double(anchorWindowPx) * 0.5
        let y0 = Double(point.y) - Double(anchorWindowPx) * 0.5

        return AnchorSignature(count: sampleCount) { out in
            for j in 0..<grid {
                let cy = y0 + (Double(j) + 0.5) * step
                for i in 0..<grid {
                    let cx = x0 + (Double(i) + 0.5) * step
                    let sx = clampIndex(cx, width)
                    let sy = clampIndex(cy, height)
                    // 3×3 box average — the cheapest possible low-pass.  A
                    // sparse 8×8 grid on a textured surface aliases hard: 1–2 px
                    // of hand shake would swing the raw samples wildly, and that
                    // is a sampling artefact, not a content change (§17.3.2).
                    var acc = 0
                    for dy in -1...1 {
                        let ry = clampIndex(Double(sy + dy), height)
                        let row = px + ry * bytesPerRow
                        for dx in -1...1 {
                            let rx = clampIndex(Double(sx + dx), width)
                            let p = row + rx * 4
                            // 32BGRA: B, G, R, A.  `lumaApprox` need not be
                            // colorimetric — it feeds a threshold comparison, not
                            // a colour computation — but it must stay the SAME
                            // formula for an instance's whole life, because both
                            // sides of every comparison come from it.
                            acc += lumaApprox(b: p[0], g: p[1], r: p[2])
                        }
                    }
                    out[j * grid + i] = UInt8(truncatingIfNeeded: acc / 9)
                }
            }
        }
    }

    // MARK: - Divergence (§17.3.2)

    /// Mean-removed mean absolute difference between two signatures, in luma
    /// levels.  Pure function; safe on any queue.
    ///
    /// The means are removed so that auto-exposure — which ramps the whole frame
    /// up or down while the user pans — cannot be read as a content change.  The
    /// result stays immune to an *additive* brightness offset and fully
    /// sensitive to content replacement.  (A multiplicative gain change is not
    /// covered; that is recorded as risk R22.)
    ///
    /// Signatures of differing length are not comparable and yield 0 — the
    /// caller re-seeds rather than comparing across a grid change.
    static func divergence(from anchor: AnchorSignature,
                           to current: AnchorSignature) -> Double {
        guard anchor.count == current.count, anchor.count > 0 else { return 0 }
        let n = anchor.count
        return anchor.withSamples { a in
            current.withSamples { b in
                var sumA = 0, sumB = 0
                for k in 0..<n { sumA += Int(a[k]); sumB += Int(b[k]) }
                let meanA = Double(sumA) / Double(n)
                let meanB = Double(sumB) / Double(n)
                var acc = 0.0
                for k in 0..<n {
                    acc += abs((Double(a[k]) - meanA) - (Double(b[k]) - meanB))
                }
                return acc / Double(n)
            }
        }
    }

    /// §17.5.1's named entry point: measure and judge in one call.
    static func drift(from anchor: AnchorSignature, to current: AnchorSignature) -> Drift {
        let d = divergence(from: anchor, to: current)
        return Drift(divergenceLuma: d,
                     exceedsThreshold: forceDriftForTesting || d > contentThresholdLuma)
    }

    /// §17.5.1's boolean form.
    static func hasDrifted(from anchor: AnchorSignature, to current: AnchorSignature) -> Bool {
        drift(from: anchor, to: current).exceedsThreshold
    }

    // MARK: - Mask consistency gate (§17.3.3)

    /// The side length, in elements, of every binary mask alpha in this app.
    ///
    /// This is not a tunable — it is a hard invariant of the render path:
    /// `MaskRenderer.AlphaResult.alpha` is documented as `// 256x256, values 0
    /// or 255`, `MaskRenderer.compositeLayers` rejects any layer whose
    /// `alpha.count != 256 * 256`, and `TapInstance.maskAlpha` / `originAlpha`
    /// simply carry that array.  It exists here so the gate's call site cannot
    /// re-derive the wrong dimension a second time (ISSUE-P4-GATE, §35.6.3),
    /// where `origW x origH` was passed for a 256 x 256 buffer and silently
    /// disabled the gate.
    static let maskAlphaSide = 256


    /// Intersection-over-union of two binary mask alphas, walked on a
    /// `stride`-spaced 2-D grid.  Pure function; the caller runs it on
    /// decoderQueue, never on main.
    ///
    /// ⛔ ISSUE-P4-GATE (debug_report §35.6.3) — this doc comment used to read
    /// "the alphas are origW × origH bytes (≈2.07 M at 1080p) and a full
    /// traversal costs 1–2 ms.  Stride 4 keeps ≈130 k points (≈0.1 ms)."  That
    /// premise was **wrong, and the §17.3.3 spec and the call site were wrong
    /// with it in the same direction**, which is why static review never caught
    /// it: every alpha in this app is **256 × 256 = 65 536 bytes**
    /// (`MaskRenderer.AlphaResult.alpha`, `MaskLayer.alpha`,
    /// `TapInstance.maskAlpha` / `originAlpha`; `compositeLayers` enforces
    /// `alpha.count == 256 * 256` as a hard contract).  The caller passed
    /// `origW × origH = 2 073 600`, so the size guard below was permanently
    /// false and this function returned 1.0 — "consistent, do not veto" — on
    /// **every call since it landed**, making `reAnchorRejectUpdate`
    /// unreachable.  Pass `maskAlphaSide` for both dimensions.
    ///
    /// Why `stride` now defaults to 1: on the real 256 × 256 grid, stride 4
    /// would leave 4 096 sample points, and a mask of 80–431 px (the sizes
    /// actually observed in §35.6.4) lands on only 5–27 of them — a standard
    /// error of ±0.10…±0.22 near p ≈ 0.5, i.e. the gate's verdict would be a
    /// coin flip exactly where the threshold lives.  A full traversal of 65 536
    /// elements costs ≈30–60 µs, which is negligible beside the ≈50–60 ms
    /// decode that precedes it, so the subsample buys nothing and costs
    /// discrimination.  The parameter is kept so a caller can still trade
    /// accuracy for time deliberately.
    ///
    /// §17.6's second implementation constraint: no intermediate array is built;
    /// the two alphas are walked in place.
    ///
    /// Returns 1.0 (≙ "consistent, do not veto") whenever the comparison cannot
    /// be made — mismatched sizes, or two empty masks.  The gate may only ever
    /// veto on evidence.  ⚠️ That safe default is also what hid the defect: it
    /// is indistinguishable, in the log, from a genuine pass.  §35.7.2's `iou:`
    /// field on the accept branch is the instrumentation that fixes that.
    static func alphaIoU(_ a: [UInt8], _ b: [UInt8],
                         width: Int, height: Int, stride: Int = 1) -> Double {
        let step = max(1, stride)
        guard width > 0, height > 0,
              a.count >= width * height, b.count >= width * height else { return 1.0 }
        var intersection = 0
        var union = 0
        a.withUnsafeBufferPointer { pa in
            b.withUnsafeBufferPointer { pb in
                var y = 0
                while y < height {
                    let row = y * width
                    var x = 0
                    while x < width {
                        let ia = pa[row + x] != 0
                        let ib = pb[row + x] != 0
                        if ia && ib { intersection += 1 }
                        if ia || ib { union += 1 }
                        x += step
                    }
                    y += step
                }
            }
        }
        guard union > 0 else { return 1.0 }
        return Double(intersection) / Double(union)
    }

    // MARK: - Translation-invariant consistency gate (§24.1.5, §24.4 B-46)

    /// Capability C's consistency comparison: intersection-over-union of two
    /// binary mask alphas after each is translated so its own centroid lands
    /// on the grid centre.
    ///
    /// ⚠️ WHY `alphaIoU` CANNOT BE REUSED FOR THIS: `alphaIoU` compares two
    /// masks in the raw, un-translated 256×256 frame — correct for
    /// capability A/B, where `canonicalPoint` never moves and the object is
    /// expected to still be roughly where it was. Once `trackedPoint` is
    /// free to follow a moving object (capability C, §24.2), the object is
    /// EXPECTED to sit at a different absolute position: a tracked object
    /// that walked across the frame produces two masks with near-zero raw
    /// overlap even though the track is working correctly, and `alphaIoU`
    /// would misjudge that success as drift. Re-centring both masks first
    /// turns the question from "is it still in the same place" into "does it
    /// still look like the same object" (A-20, §24.6.4) — the answer capability
    /// C's gate actually needs.
    ///
    /// Pure function; safe on any queue, like `alphaIoU`. Same "no
    /// intermediate array" discipline as `alphaIoU` — the two masks are
    /// walked in place, with only the two centroids computed as
    /// scratch scalars.
    ///
    /// Returns 1.0 (≙ "consistent, do not veto") whenever the comparison
    /// cannot be made — mismatched sizes, or either mask empty — mirroring
    /// `alphaIoU`'s safe-default discipline: the gate may only ever veto on
    /// evidence.
    static func centroidAlignedIoU(_ a: [UInt8], _ b: [UInt8],
                                   width: Int, height: Int) -> Double {
        guard width > 0, height > 0,
              a.count >= width * height, b.count >= width * height else { return 1.0 }
        guard let centroidA = maskCentroid(a, width: width, height: height),
              let centroidB = maskCentroid(b, width: width, height: height) else {
            return 1.0
        }

        // Integer pixel shift that carries each mask's own centroid onto the
        // grid centre. Rounding to whole pixels keeps the walk below a plain
        // index remap — no resampling / interpolation, matching `alphaIoU`'s
        // "no intermediate array, no float pixel work" discipline.
        let cx = Double(width) * 0.5
        let cy = Double(height) * 0.5
        let shiftAX = Int((cx - centroidA.x).rounded())
        let shiftAY = Int((cy - centroidA.y).rounded())
        let shiftBX = Int((cx - centroidB.x).rounded())
        let shiftBY = Int((cy - centroidB.y).rounded())

        var intersection = 0
        var union = 0
        a.withUnsafeBufferPointer { pa in
            b.withUnsafeBufferPointer { pb in
                for y in 0..<height {
                    for x in 0..<width {
                        let ia = sampleShifted(pa, x: x, y: y, shiftX: shiftAX, shiftY: shiftAY,
                                               width: width, height: height)
                        let ib = sampleShifted(pb, x: x, y: y, shiftX: shiftBX, shiftY: shiftBY,
                                               width: width, height: height)
                        if ia && ib { intersection += 1 }
                        if ia || ib { union += 1 }
                    }
                }
            }
        }
        guard union > 0 else { return 1.0 }
        return Double(intersection) / Double(union)
    }

    /// Centroid (mean x, mean y) of the nonzero elements of a binary mask
    /// alpha, or nil when the mask is entirely zero. O(width×height), the
    /// same cost class `alphaIoU` already pays for a full traversal.
    private static func maskCentroid(_ alpha: [UInt8], width: Int, height: Int) -> (x: Double, y: Double)? {
        var sumX = 0, sumY = 0, count = 0
        alpha.withUnsafeBufferPointer { p in
            for y in 0..<height {
                let row = y * width
                for x in 0..<width where p[row + x] != 0 {
                    sumX += x
                    sumY += y
                    count += 1
                }
            }
        }
        guard count > 0 else { return nil }
        return (Double(sumX) / Double(count), Double(sumY) / Double(count))
    }

    /// Read the binary value at `(x, y)` in the OUTPUT (post-shift) grid from
    /// a mask that is being translated by `(shiftX, shiftY)`: this is the
    /// value that was at `(x - shiftX, y - shiftY)` in the source mask, or
    /// `false` when that source coordinate falls outside the grid (the
    /// translated mask is zero-padded, not wrapped).
    @inline(__always)
    private static func sampleShifted(_ p: UnsafeBufferPointer<UInt8>,
                                      x: Int, y: Int, shiftX: Int, shiftY: Int,
                                      width: Int, height: Int) -> Bool {
        let sx = x - shiftX
        let sy = y - shiftY
        guard sx >= 0, sx < width, sy >= 0, sy < height else { return false }
        return p[sy * width + sx] != 0
    }

    // MARK: - Helpers

    @inline(__always)
    private static func lumaApprox(b: UInt8, g: UInt8, r: UInt8) -> Int {
        (Int(r) + 2 * Int(g) + Int(b)) / 4          // 0…255, integer, no FP
    }

    @inline(__always)
    private static func clampIndex(_ v: Double, _ limit: Int) -> Int {
        if v.isNaN { return 0 }
        let i = Int(v.rounded())
        if i < 0 { return 0 }
        if i >= limit { return limit - 1 }
        return i
    }
}
