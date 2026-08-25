//
//  AnchorTracker.swift
//  JudgeE2
//
//  Phase 4C — Day 8 (Builder) · B-43
//
//  Local block-matching search primitive for Capability C (in-session object
//  tracking).  Contract: architect_output.md §24.2.2 (search primitive,
//  §24.2.3 (lost / recovery predicates).
//
//  SCOPE GUARD (historical, B-43): this file originally delivered a search
//  PRIMITIVE only, with zero call sites. B-44 (Phase 4C Day 8) wired
//  `trackSearch` / `recoverySearch` into `CameraManager.checkAndFireReAnchor`,
//  so that is no longer true — this file's public functions ARE called now,
//  from videoQueue, whenever `DriftDetector.objectTrackingEnabled == true`
//  and a `.tracking` / `.lost` instance is picked for re-anchor (RE-2). The
//  algorithm below is unchanged by B-44/B-45; only this comment is updated so
//  it does not keep asserting something no longer true (A-17: the documented
//  surface must track actual behaviour). `objectTrackingEnabled` still ships
//  `false`, so in practice these functions have zero LIVE call sites until
//  that switch is flipped — see `DriftDetector.objectTrackingEnabled`'s doc
//  comment for the flip preconditions.
//
//  WHY THIS FILE ADDS NO NEW SAMPLING / DIVERGENCE CODE. §24.2's selected
//  design (candidate B, "reuse the AnchorSignature sampling window for local
//  block matching") is explicitly a zero-new-ANE/GPU-load, zero-new-algorithm
//  choice: content sampling stays `DriftDetector.signature(from:atCanonical:)`
//  and comparison stays `DriftDetector.divergence(from:to:)`, both already
//  verified on-device (§17.3.2). What is new here is only the SEARCH LOOP
//  around those two functions — a small grid of candidate centre points,
//  each scored by the existing primitives, reduced to a single best match.
//
//  ⚠️ QUEUE DISCIPLINE (videoQueue only, inherited from `DriftDetector
//  .signature`): every function below that touches a `CVPixelBuffer` may
//  ONLY be called with `CameraManager.latestCameraBuffer`, and only from
//  videoQueue — that buffer is videoQueue's own, and it is the only buffer
//  safe to lock (`latestInputBuffer` is a CIContext render target; locking
//  it stalls on the GPU, per `DriftDetector.signature`'s own doc comment).
//  This file adds no queue hop of its own and no lock of its own — it is a
//  set of pure functions (module-level `static var` tunables aside), "safe
//  on any queue" in the same sense `DriftDetector.divergence` is, PROVIDED
//  the caller respects the buffer-ownership rule above. B-44 owns enforcing
//  that at the call site; this file only documents the obligation.
//

import CoreGraphics
import CoreVideo
import Foundation

// MARK: - AnchorTracker

/// Local block-matching search over the `DriftDetector.signature` /
/// `divergence` primitives: "where, in a small neighbourhood, does the
/// content look most like a baseline signature" (§24.2.2), plus the lost /
/// recovery verdicts built on top of that search (§24.2.3).
enum AnchorTracker {

    // MARK: - Tunable constants (§24.2.2 / §24.2.3)
    //
    // Same shape as `DriftDetector`'s tunables: named `static var`s a device
    // session can reassign with no other code change, each with its
    // derivation written down so a future re-tune has something to check
    // against.

    /// Search radius around the current tracked point, canonical px
    /// (§24.2.2). Half of `DriftDetector.anchorWindowPx` (96.0): if the true
    /// displacement exceeds half the sampled window's side, the window's own
    /// content has already shifted out from under it and the search
    /// degenerates into "find something passable" rather than genuine
    /// tracking. This value also caps the fastest per-frame object motion
    /// this tracker can follow — anything faster is, by construction, a lost
    /// track (§24.2.3).
    static var trackSearchRadiusPx: CGFloat = 48.0

    /// Search grid step, canonical px (§24.2.2).
    ///
    /// ⚠️ DOCUMENTED DISCREPANCY, DO NOT "FIX" WITHOUT ARCHITECT SIGN-OFF:
    /// §24.2.2's prose derivation for this constant says it equals
    /// `anchorWindowPx / anchorGridSide` (96.0 / 8 = 12.0), but the same
    /// section's worked cost estimate ("取 R=48pt, S=8pt ⇒ 13×13=169 个候选")
    /// and its own initial-value table both give **8.0**, not 12.0. The two
    /// numbers in the source spec disagree with each other — almost
    /// certainly a typo in the prose explanation, not in the tabulated
    /// value, since the 169-candidate cost derivation (the number §24.2.2
    /// actually uses to justify the <1ms budget) was computed with 8.0. This
    /// implementation follows the spec's literal initial-value table (8.0)
    /// and records the inconsistency here rather than silently resolving it
    /// to 12.0 — a future Architect pass should reconcile the prose, not a
    /// Builder guessing which of two spec numbers was the "real" one.
    static var trackSearchStepPx: CGFloat = 8.0

    /// Lost-track multiplier on `DriftDetector.contentThresholdLuma`
    /// (§24.2.3). A track is lost only when even its BEST candidate match is
    /// worse than "content has obviously changed" by this factor — 2.0, i.e.
    /// twice as bad as the ordinary drift threshold, not merely equal to it.
    static var trackLostFactor: Double = 2.0

    /// Search radius for the post-lost recovery search, canonical px
    /// (§24.2.3). Twice `DriftDetector.anchorWindowPx` (96.0) — a
    /// deliberately much wider net than `trackSearchRadiusPx`, because a
    /// recovery search runs far less often (once per embedding generation,
    /// not every drift check) and is looking for an object that may have
    /// travelled a long way while off-track.
    static var trackLostRecoverySearchRadiusPx: CGFloat = 192.0

    /// Floor between two recovery-search attempts while `.lost`, ms
    /// (§24.2.3). Recovery search normally rides the existing ≈5s embedding
    /// generation cadence (no new timer); this floor only matters if that
    /// cadence stalls (e.g. a static camera), so a `.lost` instance is not
    /// left with zero retry opportunity indefinitely.
    ///
    /// ⚠️ Nothing in this batch reads or enforces this constant — B-43
    /// delivers the constant only; the caller that checks elapsed time
    /// against it is B-44's wiring work.
    static var minTrackLostRecoveryIntervalMs: Double = 2000.0

    /// Minimum displacement of `trackedPoint`, canonical px, that justifies
    /// actually dispatching a decode rather than just updating internal
    /// tracking state (§24.1.3 / §24.4 B-45's RE-1 dispatch-gate formula).
    /// ≈1.5 search steps (`trackSearchStepPx = 8.0`): a decode costs ≈55-65ms,
    /// 50-100× the ≈1ms search that found the candidate, so a displacement
    /// this small is not worth spending one on — below it, only `trackedPoint`
    /// / `anchorSignature` / `trackState` advance (§24.4 B-44/B-45 wiring);
    /// above it, "the position has moved enough that the last decoded mask is
    /// visibly stale" (architect_output §24.2.2's value table).
    ///
    /// ⚠️ Declared here, not in `CameraManager.swift` (where B-45's table row
    /// nominally lives): this constant is consumed exclusively by the
    /// dispatch-gate arithmetic in `checkAndFireReAnchor`, but it belongs
    /// alongside this file's other track-tunables (`trackSearchRadiusPx` etc.)
    /// for the same "one place to look for every knob" reason those are here
    /// rather than scattered at their call sites.
    static var trackReDecodeMinDeltaPx: CGFloat = 12.0

    /// Search grid step for `recoverySearch` only, canonical px (B-44 fix, not
    /// part of the original B-43/§24.2.3 spec).
    ///
    /// ⚠️ COST BUG FOUND DURING B-44'S REVIEW OF THE B-43 DELIVERABLE, FIXED
    /// HERE: `recoverySearch` originally reused `trackSearchStepPx` (8.0) at
    /// its much wider `trackLostRecoverySearchRadiusPx` (192.0) radius. Candidate
    /// count is `(2R/S + 1)²`: at R=192, S=8 that is 49×49 = 2401 candidates —
    /// ≈14× `trackSearch`'s 169 — for an estimated ≈120ms on videoQueue, a
    /// perceptible stall against the ≈33ms/frame budget. architect_output
    /// §24.2.3 specifies the WIDER RADIUS for recovery search but never says
    /// whether the step should widen with it — a genuine gap in the spec, not
    /// a value it deliberately chose and this file is overriding. Doubling the
    /// step to 16.0 brings the candidate count down to 25×25 = 625 (≈31ms) —
    /// recovery search is inherently coarser-grained (its job is "is the
    /// object roughly back near where the user tapped", not sub-pixel
    /// precision) and it fires far less often than `trackSearch` (once per
    /// `.lost` embedding-generation tick, not every drift check), so the
    /// coarser grid costs little. `trackSearch`'s own step
    /// (`trackSearchStepPx`) is UNCHANGED — this constant affects
    /// `recoverySearch` only.
    static var trackRecoverySearchStepPx: CGFloat = 16.0

    // MARK: - Results

    /// One grid-search's best-matching candidate: the canonical point that
    /// minimised `DriftDetector.divergence` against the baseline, and that
    /// minimum divergence itself.
    struct Candidate {
        let point: CGPoint
        let divergenceLuma: Double
    }

    /// Outcome of a normal-radius tracking search (§24.2.2), centred on the
    /// CURRENT tracked point, plus the lost verdict (§24.2.3) derived from
    /// it. Distinct type from `RecoveryResult` on purpose (§24.2.3's search
    /// centre / radius / verdict basis all differ between the two searches;
    /// see `RecoveryResult`'s doc comment for the full contrast) — B-44
    /// should not be able to accidentally feed a tracking-search verdict
    /// through the recovery-search code path or vice versa.
    struct TrackResult {
        /// The best-matching candidate found in the search window.
        let best: Candidate
        /// True iff `best.divergenceLuma >= DriftDetector.contentThresholdLuma
        /// * trackLostFactor` — the §24.2.3 lost predicate. Because `best`
        /// is by construction the MINIMUM divergence over every candidate in
        /// the window, "the best candidate still exceeds the scaled
        /// threshold" and "every candidate exceeds the scaled threshold" are
        /// the same statement; this field is exactly the "全部候选位置的
        /// divergence 都 ≥ threshold" judgement §24.2.3 asks for, just
        /// computed via the minimum rather than a separate all-candidates
        /// scan.
        let isLost: Bool
    }

    /// Outcome of a wide-radius recovery search (§24.2.3), run only while an
    /// instance is `.lost`, centred on `canonicalPoint` — deliberately NOT
    /// the position the track was lost at. §24.2.3's reasoning: an object
    /// that left frame and came back is most plausibly near where the user
    /// originally tapped (camera panned away and back, or the object walked
    /// out and back in near the same spot), not near the last tracked
    /// position, which is often the stale edge of the old search window —
    /// the least likely place for the object to reappear.
    struct RecoveryResult {
        /// The best-matching candidate found in the (wide) search window.
        let best: Candidate
        /// True iff `best.divergenceLuma < DriftDetector.contentThresholdLuma`
        /// — the UNSCALED threshold (not `* trackLostFactor`; that scaling
        /// is specific to the lost predicate, not the recovery predicate).
        /// A recovered track means content has come back to looking
        /// "merely normal", not merely "less bad than lost".
        let hasRecovered: Bool
    }

    // MARK: - Public search entry points (§24.2.2 / §24.2.3)
    //
    // Two distinctly-named, distinctly-typed entry points rather than one
    // function with a mode flag: §24.2.3 draws a hard semantic line between
    // "search near where we currently think the object is" (tracking) and
    // "search widely near where the user originally tapped" (recovery) —
    // different centres, different radii, different pass/fail bases. Naming
    // and typing that difference makes the wrong call (e.g. recovery-style
    // search feeding a `.isLost` verdict, or vice versa) a compile error for
    // B-44 rather than a silent logic bug.

    /// Normal-radius tracking search (§24.2.2): search around the current
    /// `trackedPoint` with `trackSearchRadiusPx` / `trackSearchStepPx`, and
    /// report the §24.2.3 lost verdict against `trackLostFactor`.
    ///
    /// ⚠️ videoQueue only — see the file-level queue-discipline note; `buffer`
    /// must be `CameraManager.latestCameraBuffer`.
    ///
    /// - Parameters:
    ///   - buffer: the live camera buffer to sample from.
    ///   - baseline: the signature the search is trying to re-find (e.g. the
    ///     instance's anchor signature at the point tracking last locked on).
    ///   - trackedPoint: the current tracked point, canonical space — the
    ///     search CENTRE, not a value this function mutates.
    /// - Returns: `nil` only when every candidate in the window failed to
    ///   sample (mirrors `DriftDetector.signature`'s graceful-degradation
    ///   contract: nothing to report is not the same as "lost", and the
    ///   caller should treat it the same way a single failed sample is
    ///   treated today — skip this cycle, try again next time).
    static func trackSearch(in buffer: CVPixelBuffer,
                            baseline: AnchorSignature,
                            around trackedPoint: CGPoint) -> TrackResult? {
        guard let best = bestCandidate(in: buffer,
                                       baseline: baseline,
                                       center: trackedPoint,
                                       radius: trackSearchRadiusPx,
                                       step: trackSearchStepPx) else {
            return nil
        }
        let lostThreshold = DriftDetector.contentThresholdLuma * trackLostFactor
        return TrackResult(best: best, isLost: best.divergenceLuma >= lostThreshold)
    }

    /// Wide-radius recovery search (§24.2.3): search around `canonicalPoint`
    /// — the original tap location, NOT the position a track was lost at —
    /// with `trackLostRecoverySearchRadiusPx` / `trackRecoverySearchStepPx`
    /// (its own, coarser step — see that constant's doc comment for why it
    /// differs from `trackSearch`'s `trackSearchStepPx`), and report the
    /// §24.2.3 recovery verdict against the unscaled
    /// `DriftDetector.contentThresholdLuma`.
    ///
    /// ⚠️ videoQueue only — see the file-level queue-discipline note; `buffer`
    /// must be `CameraManager.latestCameraBuffer`.
    ///
    /// - Parameters:
    ///   - buffer: the live camera buffer to sample from.
    ///   - baseline: the signature the recovery search is trying to re-find.
    ///   - canonicalPoint: the instance's frozen original tap point,
    ///     canonical space — the search CENTRE for recovery, by §24.2.3's
    ///     explicit ruling (never the last tracked / lost position).
    /// - Returns: `nil` only when every candidate in the window failed to
    ///   sample (same graceful-degradation contract as `trackSearch`).
    static func recoverySearch(in buffer: CVPixelBuffer,
                               baseline: AnchorSignature,
                               around canonicalPoint: CGPoint) -> RecoveryResult? {
        guard let best = bestCandidate(in: buffer,
                                       baseline: baseline,
                                       center: canonicalPoint,
                                       radius: trackLostRecoverySearchRadiusPx,
                                       step: trackRecoverySearchStepPx) else {
            return nil
        }
        return RecoveryResult(best: best, hasRecovered: best.divergenceLuma < DriftDetector.contentThresholdLuma)
    }

    // MARK: - Shared grid-search primitive (§24.2.2)

    /// Grid-search `[center.x - radius, center.x + radius] ×
    /// [center.y - radius, center.y + radius]` at `step` spacing, sampling
    /// each candidate centre with `DriftDetector.signature` and scoring it
    /// against `baseline` with `DriftDetector.divergence`. Returns the
    /// candidate with the SMALLEST divergence (the best match), or `nil` if
    /// no candidate in the window could be sampled at all.
    ///
    /// No new sampling or comparison logic lives here — this loop is the
    /// entire contribution of B-43 over what `DriftDetector` already
    /// provided (see the file-level comment). No caching, no reuse of
    /// neighbouring candidates' partial sums: §24.2.2's cost derivation
    /// (169 candidates × ≈50µs ≈ <1ms) already assumes the naive per-candidate
    /// cost and found it acceptable, and the spec explicitly asks that no
    /// optimisation be added here (it would only add complexity and a new
    /// place to get the bookkeeping wrong).
    ///
    /// Candidate count matches §24.2.2's `(2R/S + 1)²` derivation exactly:
    /// index-based stepping (`center - radius + Double(i) * step`) is used
    /// instead of accumulating `x += step` in a `while` loop so the
    /// candidate count cannot drift by one due to floating-point
    /// accumulation error — at R=48, S=8 this must produce exactly 13
    /// candidates per axis (169 total), matching the number the cost budget
    /// was computed against.
    private static func bestCandidate(in buffer: CVPixelBuffer,
                                      baseline: AnchorSignature,
                                      center: CGPoint,
                                      radius: CGFloat,
                                      step: CGFloat) -> Candidate? {
        guard radius >= 0, step > 0 else { return nil }

        let perAxis = Int(((radius * 2) / step).rounded()) + 1
        guard perAxis > 0 else { return nil }

        var best: Candidate?
        for j in 0..<perAxis {
            let dy = -radius + CGFloat(j) * step
            for i in 0..<perAxis {
                let dx = -radius + CGFloat(i) * step
                let candidatePoint = CGPoint(x: center.x + dx, y: center.y + dy)
                guard let signature = DriftDetector.signature(from: buffer, atCanonical: candidatePoint) else {
                    continue
                }
                let divergence = DriftDetector.divergence(from: baseline, to: signature)
                if best == nil || divergence < best!.divergenceLuma {
                    best = Candidate(point: candidatePoint, divergenceLuma: divergence)
                }
            }
        }
        return best
    }
}

// MARK: - Self-check (§24.5, no call sites this batch)

#if DEBUG
extension AnchorTracker {

    /// Off-line correctness check for the search primitive above.
    ///
    /// B-43 lands with zero production call sites (wiring into
    /// `checkAndFireReAnchor` is B-44's job, per §24.4's dependency table),
    /// so there is no way to exercise this code by running the app — the
    /// usual "point the camera at something and watch the log" verification
    /// this project relies on for the rest of the re-anchor loop does not
    /// apply here yet. This function is NOT called from anywhere in the app
    /// (mirrors `PinDebugFixture`'s "debug-only, not wired into the live
    /// flow" precedent, `Persistence/PinDebugFixture.swift`) — it exists so
    /// the search math can be exercised and `assert()`-checked by calling it
    /// explicitly (this Builder session compiled and ran it standalone,
    /// outside the Xcode project, alongside `DriftDetector.swift`, against a
    /// synthetic `CVPixelBuffer` — see `shared/builder_progress.md` for the
    /// harness); a future caller (B-44's author, or a scratch `main.swift`)
    /// can invoke it the same way.
    ///
    /// Builds two synthetic 32BGRA buffers, both a uniform dark background
    /// with one bright square "marker" on it, and checks:
    ///   1. `trackSearch` finds the marker after it moves by a known,
    ///      asymmetric offset (dx ≠ dy, to catch an axis mix-up) that is
    ///      still within `trackSearchRadiusPx`, landing exactly on the true
    ///      new centre with near-zero divergence and `isLost == false`.
    ///   2. `trackSearch` reports `isLost == true` when the marker has moved
    ///      far enough (dx = 96pt) that NONE of the candidates in the
    ///      `trackSearchRadiusPx` window overlap it — every sampled
    ///      candidate sees only flat background, which cannot match the
    ///      structured baseline.
    ///   3. `recoverySearch`, centred on the ORIGINAL (unmoved) point with
    ///      the wider `trackLostRecoverySearchRadiusPx`, finds that same
    ///      far-moved marker and reports `hasRecovered == true` — the exact
    ///      scenario `trackSearch` in (2) could not resolve, demonstrating
    ///      the two searches' distinct centre/radius contract, not just
    ///      their distinct types.
    ///
    /// All move offsets are chosen as exact multiples of `trackSearchStepPx`
    /// (16, -8, 96 — never e.g. 100) so the true marker centre always lands
    /// exactly on a searched grid point. This is deliberate, not an
    /// oversight: block matching against a hard, high-contrast synthetic
    /// edge is highly sensitive to sub-step misalignment (a candidate a few
    /// px off the true centre samples an asymmetric slice of marker vs.
    /// background and reads a real, non-buggy divergence spike — this was
    /// caught and confirmed by hand while authoring this check, using a
    /// dx = 100pt offset, which is 4px off the nearest grid point and reads
    /// divergence ≈21 there instead of the ≈0 the exact centre gives). That
    /// grid-quantization sensitivity is an inherent, already-documented
    /// property of coarse-step block matching (§24.2 lists "ambiguity on
    /// repeated/high-contrast texture" among B's known limitations), not a
    /// defect in `bestCandidate`'s coordinate math — so this check is
    /// designed to isolate coordinate-system correctness (wrong axis, wrong
    /// sign, off-by-one candidate count) from that separate, real phenomenon
    /// rather than conflate the two.
    ///
    /// Returns `true` iff every check passed; also `assert()`s inline so a
    /// DEBUG run traps immediately at the first violated invariant.
    @discardableResult
    static func selfCheck() -> Bool {
        let side = 500
        let markerHalf = 30
        let background: UInt8 = 20
        let marker: UInt8 = 220

        func makeBuffer(markerCenter: (x: Int, y: Int)) -> CVPixelBuffer? {
            var pb: CVPixelBuffer?
            let attrs: [CFString: Any] = [kCVPixelBufferCGImageCompatibilityKey: true,
                                          kCVPixelBufferCGBitmapContextCompatibilityKey: true]
            let status = CVPixelBufferCreate(kCFAllocatorDefault, side, side,
                                             kCVPixelFormatType_32BGRA,
                                             attrs as CFDictionary, &pb)
            guard status == kCVReturnSuccess, let buffer = pb else { return nil }
            guard CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess else { return nil }
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            let px = base.assumingMemoryBound(to: UInt8.self)
            for y in 0..<side {
                let row = px + y * bytesPerRow
                let inMarkerRow = abs(y - markerCenter.y) <= markerHalf
                for x in 0..<side {
                    let inMarker = inMarkerRow && abs(x - markerCenter.x) <= markerHalf
                    let v = inMarker ? marker : background
                    let p = row + x * 4
                    p[0] = v; p[1] = v; p[2] = v; p[3] = 255   // BGRA
                }
            }
            return buffer
        }

        let originalCenter = (x: 250, y: 250)
        guard let baselineBuffer = makeBuffer(markerCenter: originalCenter),
              let baselineSig = DriftDetector.signature(from: baselineBuffer,
                                                        atCanonical: CGPoint(x: originalCenter.x,
                                                                             y: originalCenter.y)) else {
            assertionFailure("[TRACK][SELFCHECK] failed to build baseline buffer/signature")
            return false
        }

        var ok = true

        // (1) small, in-radius, asymmetric move: dx=16, dy=-8.
        let smallMoveCenter = (x: originalCenter.x + 16, y: originalCenter.y - 8)
        if let movedBuffer = makeBuffer(markerCenter: smallMoveCenter),
           let result = trackSearch(in: movedBuffer, baseline: baselineSig,
                                    around: CGPoint(x: originalCenter.x, y: originalCenter.y)) {
            let dx = abs(result.best.point.x - CGFloat(smallMoveCenter.x))
            let dy = abs(result.best.point.y - CGFloat(smallMoveCenter.y))
            // Grid-aligned offset (16, -8 are multiples of trackSearchStepPx):
            // the true centre is itself a searched candidate, so an exact
            // match is expected, not just "close".
            let landedOnTrueOffset = dx < 0.01 && dy < 0.01
            assert(landedOnTrueOffset,
                  "[TRACK][SELFCHECK] trackSearch best \(result.best.point) did not land exactly on grid-aligned true centre \(smallMoveCenter)")
            assert(result.best.divergenceLuma < DriftDetector.contentThresholdLuma,
                  "[TRACK][SELFCHECK] trackSearch divergence \(result.best.divergenceLuma) unexpectedly high for a matched marker")
            assert(!result.isLost, "[TRACK][SELFCHECK] trackSearch reported isLost for an in-radius, matched marker")
            ok = ok && landedOnTrueOffset && result.best.divergenceLuma < DriftDetector.contentThresholdLuma && !result.isLost
        } else {
            assertionFailure("[TRACK][SELFCHECK] small-move buffer/search failed to produce a result")
            ok = false
        }

        // (2) + (3): large, out-of-radius move: dx=96 (a multiple of
        // trackSearchStepPx, and > trackSearchRadiusPx=48 but well inside
        // trackLostRecoverySearchRadiusPx=192), dy=0.
        let farMoveCenter = (x: originalCenter.x + 96, y: originalCenter.y)
        guard let farBuffer = makeBuffer(markerCenter: farMoveCenter) else {
            assertionFailure("[TRACK][SELFCHECK] failed to build far-move buffer")
            return false
        }

        if let lostResult = trackSearch(in: farBuffer, baseline: baselineSig,
                                        around: CGPoint(x: originalCenter.x, y: originalCenter.y)) {
            assert(lostResult.isLost,
                  "[TRACK][SELFCHECK] trackSearch failed to report isLost when the marker moved outside trackSearchRadiusPx")
            ok = ok && lostResult.isLost
        } else {
            assertionFailure("[TRACK][SELFCHECK] far-move trackSearch failed to produce a result")
            ok = false
        }

        if let recovered = recoverySearch(in: farBuffer, baseline: baselineSig,
                                          around: CGPoint(x: originalCenter.x, y: originalCenter.y)) {
            let dx = abs(recovered.best.point.x - CGFloat(farMoveCenter.x))
            let dy = abs(recovered.best.point.y - CGFloat(farMoveCenter.y))
            let landedOnTrueOffset = dx < 0.01 && dy < 0.01
            assert(landedOnTrueOffset,
                  "[TRACK][SELFCHECK] recoverySearch best \(recovered.best.point) did not land exactly on grid-aligned true centre \(farMoveCenter)")
            assert(recovered.hasRecovered,
                  "[TRACK][SELFCHECK] recoverySearch failed to report hasRecovered for a marker within its wider radius")
            ok = ok && landedOnTrueOffset && recovered.hasRecovered
        } else {
            assertionFailure("[TRACK][SELFCHECK] recoverySearch failed to produce a result for the far-moved marker")
            ok = false
        }

        return ok
    }
}
#endif
