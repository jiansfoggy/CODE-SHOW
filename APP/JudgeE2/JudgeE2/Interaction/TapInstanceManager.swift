//
//  TapInstanceManager.swift
//  JudgeE2
//
//  Phase 3 — Day 5 (Builder)
//
//  Multi-instance pool for Tap-to-Segment.  Contract: architect_output.md
//  §3.1 (TapInstance fields), §3.2 (pool rules), §3.3 (colour palette),
//  §3.4 (primary vs secondary rendering).
//
//  SCOPE GUARD (Architect reserved item R3): this file carries NO mask-quality
//  logic.  Candidate selection, the 60 %/85 % caps, the shape gates, the
//  `iou_pred >= 0.1` gate and the flood fill all stay in MaskRenderer /
//  CameraManager, untouched.  The pool only decides *which* masks exist and
//  *what colour* they draw in.
//
//  THREADING: instances are added from the gesture thread (handleTap runs the
//  fast-path decision inline — see Requirement A), updated from decoderQueue,
//  and read from decoderQueue + main.  There is no single owning queue, so the
//  pool guards its own array with an NSLock.  This is deliberately NOT a new
//  dispatch queue (Architect §10.4 A forbids a third queue); a lock around a
//  ≤3-element array costs nothing and adds no scheduling hop.
//

import CoreML
import Foundation
import UIKit

// MARK: - TrackState

/// Capability C's per-instance tracking state machine (architect_output.md
/// §24.2.3 / §24.4 B-42).
///
/// `.locked` is the state every instance is constructed in and — until
/// B-43/B-44 land — the ONLY state any instance ever reaches: nothing in
/// this batch transitions it.  That is what keeps this batch's observable
/// behaviour bit-identical to today's: `.locked` means "tracking is not
/// engaged", which reads the same as "capability C does not exist yet".
nonisolated enum TrackState: Equatable {
    /// Tracking not engaged. `trackedPoint` sits at `canonicalPoint` and
    /// never moves — today's (Phase 3/4A) behaviour, unchanged.
    case locked
    /// Actively following the object; `trackedPoint` may differ from
    /// `canonicalPoint` (B-43/B-44 own the movement logic, not this batch).
    case tracking
    /// A tracking search failed to find the object; `trackedPoint` is frozen
    /// at its last known position (B-43/B-44 own the transition into and
    /// out of this state, not this batch).
    case lost
}

// MARK: - TapInstance

/// One user tap and the mask decoded for it (architect_output §3.1).
struct TapInstance {
    let id: UUID
    /// Tap location in Canonical pixel space (original camera frame, display
    /// orientation) — the same space `PointPromptBuilder` consumes.
    let canonicalPoint: CGPoint
    /// Tap location in preview-view coordinates — stored at intake time so the
    /// anchor-marker overlay can place itself without an inverse coordinate
    /// transform at render time.  Nil only when a tap was constructed
    /// programmatically (tests, drain-path synthetic taps).
    let viewPoint: CGPoint?
    /// FIFO ordering key (§3.2: "超出 N=3 时 FIFO 删除 createdAt 最早的实例").
    let createdAt: Date
    /// Raw decoder output for this instance.  Held per §3.1; nothing reads it
    /// today — `maskAlpha` is what the renderer composites.
    var mask: MLMultiArray?
    /// When `mask` was produced.  Pure telemetry timestamp; not used for routing
    /// or display decisions.  Architect B-2: `maskTTL` / `isMaskValid` removed.
    var maskTimestamp: Date?
    /// iou_pred of the candidate that was actually drawn (0 = not decoded yet).
    var iouPred: Float
    /// Presentation slot colour, fixed for the instance's whole life.
    /// One distinct hue per slot (§3.3.1 L3 — "which tap did this come from").
    /// It carries L3 ONLY: L1 visibility belongs to the two-tone outline and
    /// must never be inferred from this field.
    let color: UIColor
    /// true = most recent tap (§3.2). Exactly one live instance is primary.
    var isPrimary: Bool

    /// 256×256 binary alpha (0 / 255) produced by `MaskRenderer`.  This — not
    /// `mask` — is what `compositeInstances` blends, so re-compositing after a
    /// later tap never re-runs candidate selection (R3: the pick must not be
    /// re-decided when an unrelated instance updates).
    var maskAlpha: [UInt8]?

    /// Tap sequence number (`CameraManager.tapGeneration`) of the newest request
    /// issued against this instance.  A decode whose `requestGen` no longer
    /// matches has been superseded and must be dropped — this is how R4's
    /// "同一实例的新 tap 取代旧请求" is enforced without unbounded queueing.
    var requestGen: Int

    /// Content baseline of this instance's anchor neighbourhood — the frame the
    /// current mask is an assertion about (Phase 4, §17.5.2).
    ///
    /// Written on videoQueue by `checkAndFireReAnchor` only: seeded once the
    /// instance becomes drawable, then advanced at every batch **start** (§16.4.2
    /// — the frame that triggered the batch is the correct next baseline;
    /// deferring to batch end would let the intervening frames' accumulated
    /// divergence fire a second, redundant batch).  Nothing else reads or writes
    /// it.  It is not part of the render path.
    var anchorSignature: AnchorSignature?

    /// The `CameraManager.embeddingGeneration` this instance has already been
    /// re-anchored against (§18.1.4, RE-1).  nil = never re-anchored.
    ///
    /// WHY IT EXISTS.  `decode → numeric sentinel → buildTapAlpha` is a pure
    /// function of `(embedding, canonicalPoint)` (§18.1.2's four source-level
    /// checks): `SAMDecoder` keeps only lazy model caches, `MaskRenderer` keeps
    /// only a constant, `canonicalPoint` is frozen by §16.7 and a change in
    /// `origW/origH` clears the pool via C4 first.  So a second re-anchor decode
    /// against the *same* embedding reproduces the mask already on screen
    /// bit-for-bit.  Gating on the generation deletes that work; it does not
    /// trade anything away — §18.1.4: "没有任何一个在 RE-1 之前会出现的画面，
    /// 在 RE-1 之后不会出现".
    ///
    /// Written on videoQueue at the moment the decode is **dispatched**, not
    /// when it returns — same timing and same reason as `anchorSignature`
    /// (§16.4.2): this is a throttle quantity, and advancing it early can only
    /// ever skip a refresh, never duplicate one.  The tap path neither reads nor
    /// writes it (taps are ordered by `requestGen`, an orthogonal counter).
    var lastReAnchorEmbeddingGen: UInt64?

    /// When this instance was last picked for a re-anchor, in
    /// `PerfLogger.nowMs()` (§18.1.5, RE-2).  nil = never picked.
    ///
    /// A batch is fixed at one instance (RE-2), so a selection rule is needed;
    /// "least recently refreshed first" makes it a fair rotation and starves
    /// nobody.  Written on videoQueue at dispatch, alongside
    /// `lastReAnchorEmbeddingGen`.
    var lastReAnchorAtMs: Double?

    /// The alpha this instance's **tap** produced — the frozen comparison basis
    /// for the consistency gate (§18.2.2, RE-3).
    ///
    /// ⚠️ WRITTEN BY THE TAP PATH ONLY.  A re-anchor updates `maskAlpha` and
    /// must never touch this field.  That asymmetry is the whole point: §17.3.3
    /// compared each refresh against `maskAlpha`, which a *successful* re-anchor
    /// had itself just written, so the gate constrained adjacent steps only and
    /// a chain of high step-wise IoUs said nothing about end-to-end drift
    /// (§18.2.1, long-term convention A-14).  Comparing against the tap product
    /// instead makes the invariant statable: *every mask an instance ever shows
    /// has IoU ≥ `reAnchorAcceptIoU` with the mask the user's tap produced.*
    ///
    /// Recovery from a vetoed state needs no extra machinery (§18.2.3):
    /// REC-1 pan back ⇒ the IoU recovers and refreshes resume by the same gate;
    /// REC-2 tap again ⇒ `requestGen` bumps, the tap path rewrites this field;
    /// REC-3 FIFO eviction / C4 ⇒ the instance and the field go together.
    var originAlpha: [UInt8]?

    /// Current object-tracking anchor position (§24.2 capability C, §24.4
    /// B-42).  Initialised to `canonicalPoint` at construction and, in THIS
    /// batch, never written again — the code that moves it as the tracked
    /// object moves is B-43/B-44, not here.
    ///
    /// Distinct field from `canonicalPoint` by design, not by accident:
    /// `canonicalPoint` is frozen forever (§16.7) because it is PIN-1's only
    /// read source — a saved Pin must record the user's original tap, never
    /// a tracking result.  `trackedPoint` is the point capability C's
    /// re-anchor decode is meant to move; the two must never be conflated
    /// (see also §24.4 B-48, `PinFactory` defensive read discipline).
    var trackedPoint: CGPoint

    /// Snapshot of `trackedPoint` at the last successful *tracked* re-anchor
    /// dispatch (§24.2, §24.4 B-44).  nil until such a re-anchor has fired at
    /// least once. This batch never writes a non-nil value — B-42 only seeds
    /// `nil` at construction; the write path is B-44.
    var lastReAnchorTrackedPoint: CGPoint?

    /// Capability C tracking state (see `TrackState`).  Every instance is
    /// constructed `.locked` and this batch contains no code that
    /// transitions it — B-43/B-44 own the state machine.
    var trackState: TrackState

    /// Age of the mask in milliseconds, or nil if no mask has been decoded yet.
    /// Telemetry only — the rendering path does not gate on mask age.
    func maskAgeMs(now: Date) -> Double? {
        guard let ts = maskTimestamp else { return nil }
        return now.timeIntervalSince(ts) * 1000
    }

    /// The slot index (0, 1, or 2) assigned to this instance, derived from its
    /// color identity.  Uses reference equality (`===`) against the palette
    /// singletons — the same rule as `addInstance`'s allocator.
    /// Falls back to 0 if the colour is not found (should never occur).
    func slotIndex(in palette: [UIColor]) -> Int {
        palette.firstIndex { $0 === color } ?? 0
    }
}

// MARK: - TapInstanceManager

final class TapInstanceManager {

    /// Maximum live instances (architect_output §9.5: N = 3, FINAL).
    static let maxInstances = 3

    // MARK: - Fill palette (C-7 admission record)
    //
    // ⚠️ SPEC AMENDMENT, PENDING ARCHITECT RATIFICATION.
    // architect_output §3.3.3 / §12.1 Q6 withdrew per-instance hue and froze a
    // SINGLE cyan for all three slots.  The user has since required visually
    // distinct masks, and the premise §12 rested on has changed: the two-tone
    // outline (§3.4, `MaskOutline` + `PreviewView`) now carries L1 on its own,
    // so hue no longer decides whether a mask is *visible* — only which tap it
    // came from (§3.3.1 L3).  Under that split, per-instance hue is defensible
    // again.  The hard constraints C-1/C-2/C-3 are NOT relaxed, and the
    // withdrawn systemBlue/systemGreen/systemOrange remain banned outright.
    //
    // ── C-7 (a): arithmetic admission test ──────────────────────────────────
    // sRGB relative luminance Y = 0.2126·R_lin + 0.7152·G_lin + 0.0722·B_lin,
    // H/S/V re-derived from the 8-bit values that actually ship.
    //
    //  slot  sRGB            H        S     V     Y       C-1  C-2  C-3
    //  ----  --------------  -------  ----  ----  ------  ---  ---  ---
    //   0    (0, 217, 255)   188.94°  1.00  1.00  0.5685   ✅   ✅   ✅
    //   1    (0, 255, 242)   176.94°  1.00  1.00  0.7793   ✅   ✅   ✅
    //   2    (0, 255, 170)   160.00°  1.00  1.00  0.7442   ✅   ✅   ✅
    //  (rejected, for the record)
    //   systemBlue   (0,122,255)  211.3°  1.00  1.00  0.2114  ❌   ❌   ✅
    //   systemGreen  (52,199,89)  135.1°  0.74  0.78  0.4230  ❌   ❌   ❌
    //   systemOrange (255,149,0)   35.1°  1.00  1.00  0.4275  ❌   ❌   ✅
    //
    // ── Why three hues DO fit, contrary to §3.3.3's reasoning ──────────────
    // §3.3.3 asserts "允许带装不下三个彼此可区分色相".  Worked through:
    //   • Band 2, H ∈ [280°,330°] (magenta), is *vacuous* under C-2 ∩ C-3.  Its
    //     maximum attainable luminance with S ≥ 0.85 and V ≥ 0.90 is Y = 0.2988
    //     (at H = 300°, S = 0.85) — it cannot reach 0.45 anywhere.  The magenta
    //     fallback in C-1 is therefore unusable and should be struck.
    //   • Band 1, H ∈ [160°,200°], is cut by C-2 at H = 194.78° (Y crosses 0.45).
    //     The feasible arc is H ∈ [160°, 194.78°], ~34.8° wide.
    //   • With slot 0 pinned at 189° (the Day 4 anchor), the triple that
    //     maximises the minimum CIEDE2000 separation is (160.0°, 177.5°, 189°),
    //     ΔE00_min = 17.4.  The shipped triple (160°, 177°, 189°) gives
    //     ΔE00_min = 17.1 — above the ΔE00 ≈ 10 rule of thumb for reliable
    //     categorical identification.
    // So the constraint set admits three distinguishable hues; what it does not
    // admit is three hues *spread across the colour wheel*, which is what the
    // withdrawn palette did (its ΔE00_min was 50.2 — and that spread is exactly
    // what dragged two of its three hues into the banned bands).
    //
    // ── C-7 (b): collision-risk assessment ─────────────────────────────────
    //   slot 0  189° cyan       — Day 4 anchor.  Residual risk R10 unchanged
    //                             (cyan mugs, teal book covers, screen content).
    //   slot 1  177° aqua       — same risk class as slot 0; S = 1.00/V = 1.00
    //                             keeps it off real surfaces, which photograph
    //                             desaturated.
    //   slot 2  160° spring cyan— the highest-risk of the three: it sits ON the
    //                             lower edge of the allowed band, 5° from the
    //                             [90°,155°] green ban.  Chroma-key green
    //                             (0,177,64) is H = 142°, 18° away; mint/teal
    //                             paint and fabric are S < 0.4 and fail C-3 by a
    //                             wide margin, so they cannot collide at full
    //                             saturation.  Accepted with ZERO band margin —
    //                             flagged, not hidden.
    // The visual same-hue-object test C-7 (b) demands is a DEVICE step and has
    // NOT been performed by this build.  It must be run (a cyan object, an aqua
    // object and a mint/spring-green object) before this palette is treated as
    // admitted.
    //
    // ── Overlap mixing (§3.3.3 objection 2) ────────────────────────────────
    // `compositeLayers` blends source-over, so overlapping instances do mix.
    // The old palette's mixtures fell out of the constraint set entirely
    // (blue+orange → grey-brown).  This palette is CLOSED under mixing: all
    // three colours have R = 0 and G,B ≥ 170, so every convex combination also
    // has R = 0 (⇒ S = 1.000 exactly) and stays inside the band.  Enumerating
    // every ordering `compositeLayers` can produce at 0.40/0.40/0.60:
    //     H ∈ [163.6°, 186.5°]   S = 1.000   V ∈ [0.930, 1.000]   Y ∈ [0.600, 0.771]
    // — i.e. every overlap product passes C-1, C-2 and C-3 in its own right.
    //
    // ⚠️ All three are plain sRGB literals, **never** dynamic system colours:
    // `MaskRenderer.compositeLayers` resolves them with `getRed` on
    // decoderQueue, where a dynamic colour would resolve against a background
    // thread's trait collection (D-15).

    /// Slot 0 — cyan, H = 188.94°.  Kept as the first-allocated colour so a
    /// single tap paints exactly what the 44/50 = 88 % manual-scoring baseline
    /// was scored on; the N = 1 case is unchanged by this amendment (§11.9 R9).
    static let slot0Color = UIColor(red: 0.0, green: 217.0 / 255.0, blue: 1.0, alpha: 1.0)
    /// Slot 1 — aqua, H = 176.94°.
    static let slot1Color = UIColor(red: 0.0, green: 1.0, blue: 242.0 / 255.0, alpha: 1.0)
    /// Slot 2 — spring cyan, H = 160.00° (lower edge of the allowed band).
    static let slot2Color = UIColor(red: 0.0, green: 1.0, blue: 170.0 / 255.0, alpha: 1.0)

    /// The colour meant when only one is meant — slot 0, the Day 4 anchor.
    static let fillColor = slot0Color

    /// Presentation slots (§3.3.3).  Allocation is first-unused-slot, so the
    /// colour of an evicted instance is recycled and the same physical slot
    /// never shows two hues at once.
    static let palette: [UIColor] = [slot0Color, slot1Color, slot2Color]

    /// Fill opacity by role (§3.4).  0.60 is bit-identical to Day 4's fill
    /// (153/255), which is what keeps the 44/50 = 88 % manual-scoring baseline
    /// comparable; 0.40 gives secondary a 1.5:1 step down (mitigation M1 —
    /// an older mask should read as "this was selected", not as a current
    /// assertion), still above the "visible on a natural scene" floor.
    static let primaryOpacity: CGFloat = 0.60
    static let secondaryOpacity: CGFloat = 0.40

    private let lock = NSLock()
    private var instances: [TapInstance] = []

    // MARK: Mutation

    /// Create an instance for a new tap and make it primary.
    /// Evicts the oldest instance first when the pool is already full (§3.2).
    /// - Returns: the new instance, plus the id of the instance evicted (if any)
    ///   so the caller can cancel that instance's in-flight request.
    @discardableResult
    func addInstance(point: CGPoint, viewPoint: CGPoint? = nil,
                     requestGen: Int, now: Date = Date())
        -> (added: TapInstance, evicted: UUID?) {
        lock.lock(); defer { lock.unlock() }

        var evicted: UUID? = nil
        if instances.count >= Self.maxInstances {
            // FIFO by createdAt — "不论是否 primary" (§3.2).
            if let oldestIdx = instances.indices.min(by: { instances[$0].createdAt < instances[$1].createdAt }) {
                evicted = instances[oldestIdx].id
                instances.remove(at: oldestIdx)
            }
        }

        for i in instances.indices { instances[i].isPrimary = false }

        // Slot allocation: the lowest-numbered slot no live instance holds, so an
        // evicted instance's colour is recycled and two live instances can never
        // share a hue.
        //
        // Matching is by REFERENCE (`===`) against the palette singletons rather
        // than by UIColor equality: `TapInstance.color` is always one of the three
        // `palette` objects, so identity is exact and the allocator never depends
        // on how UIColor implements `isEqual:` / `hash` across colour spaces.
        // Eviction ran above, so at most 2 instances remain and a free slot always
        // exists — the `??` is an unreachable defensive fallback.
        let color = Self.palette.first { candidate in
            !instances.contains { $0.color === candidate }
        } ?? Self.palette[instances.count % Self.palette.count]

        let instance = TapInstance(id: UUID(),
                                   canonicalPoint: point,
                                   viewPoint: viewPoint,
                                   createdAt: now,
                                   mask: nil,
                                   maskTimestamp: nil,
                                   iouPred: 0,
                                   color: color,
                                   isPrimary: true,
                                   maskAlpha: nil,
                                   requestGen: requestGen,
                                   anchorSignature: nil,
                                   lastReAnchorEmbeddingGen: nil,
                                   lastReAnchorAtMs: nil,
                                   originAlpha: nil,
                                   trackedPoint: point,
                                   lastReAnchorTrackedPoint: nil,
                                   trackState: .locked)
        instances.append(instance)
        return (instance, evicted)
    }

    /// Attach a decoded mask to an instance.  Returns false when the instance is
    /// gone (evicted / cleared) or the request was superseded — the caller must
    /// then drop the result instead of publishing it.
    ///
    /// - Parameter recordOrigin: pass `true` **only** from the tap path.  It is
    ///   the single write point of `originAlpha` (§18.2.2 / B-15): the frozen
    ///   basis of the consistency gate must record what the *user's tap*
    ///   produced, so a re-anchor — which reaches this method on its own success
    ///   path — must leave it alone.  Defaulting to `false` makes the re-anchor
    ///   call site correct by omission rather than by remembering.  The write
    ///   happens in the same lock acquisition as `maskAlpha`, so the two can
    ///   never be observed out of step.
    @discardableResult
    func updateMask(id: UUID,
                    requestGen: Int,
                    mask: MLMultiArray?,
                    alpha: [UInt8],
                    iouPred: Float,
                    recordOrigin: Bool = false,
                    now: Date = Date()) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard let idx = instances.firstIndex(where: { $0.id == id }),
              instances[idx].requestGen == requestGen else { return false }
        instances[idx].mask = mask
        instances[idx].maskAlpha = alpha
        instances[idx].maskTimestamp = now
        instances[idx].iouPred = iouPred
        if recordOrigin {
            // Arrays are COW: this is a retain, not a 2 MB copy.  The buffer
            // splits from `maskAlpha` only when a re-anchor later succeeds
            // (§18.5.2's +2 MB/instance allowance).
            instances[idx].originAlpha = alpha
        }
        return true
    }

    /// Install the re-anchor content baseline for an instance (§17.5.2 / B-7).
    ///
    /// Uses the pool's existing `lock` — the same one that already guards
    /// `mask` / `maskAlpha` / `requestGen` — so §16.7's "no new lock" holds.  The
    /// caller is videoQueue and is the only writer *and* the only reader, so the
    /// lock is here for the pool's invariant, not for this field's ordering.
    ///
    /// Silently no-ops when the instance is gone (evicted / cleared while the
    /// frame was being processed).
    func setAnchorSignature(id: UUID, signature: AnchorSignature) {
        lock.lock(); defer { lock.unlock() }
        guard let idx = instances.firstIndex(where: { $0.id == id }) else { return }
        instances[idx].anchorSignature = signature
    }

    /// Record that a re-anchor decode was **dispatched** for this instance
    /// (§18.1.4 RE-1 + §18.1.5 RE-2 / B-13 + B-14).
    ///
    /// Both fields are throttle state, and both advance at dispatch rather than
    /// at completion — the §16.4.2 argument applies unchanged: advancing early
    /// can only cost one skipped refresh, while advancing late lets the frames
    /// elapsed during the decode fire a second, redundant one.
    ///
    /// Uses the pool's existing `lock`, like `setAnchorSignature` (B-7), so
    /// §16.7's "no new lock" holds.  Silently no-ops when the instance is gone.
    ///
    /// - Parameter trackedPoint: Capability C only (§24.1.3 / §24.4 B-45).
    ///   When non-nil, also snapshots `lastReAnchorTrackedPoint` — the point
    ///   the RE-1 dispatch-gate distance check measures FROM on the next
    ///   cycle. Default `nil` leaves `lastReAnchorTrackedPoint` untouched, so
    ///   every call site that predates B-44/B-45 (and the `.locked` arm of the
    ///   one call site that now exists) is byte-for-byte unchanged.
    func markReAnchorDispatched(id: UUID, embeddingGeneration: UInt64, atMs: Double,
                                trackedPoint: CGPoint? = nil) {
        lock.lock(); defer { lock.unlock() }
        guard let idx = instances.firstIndex(where: { $0.id == id }) else { return }
        instances[idx].lastReAnchorEmbeddingGen = embeddingGeneration
        instances[idx].lastReAnchorAtMs = atMs
        if let trackedPoint {
            instances[idx].lastReAnchorTrackedPoint = trackedPoint
        }
    }

    /// Compound update for Capability C tracking state (§24.2 / §24.4 B-44):
    /// `trackedPoint`, `trackState`, and `anchorSignature` are the outcome of
    /// ONE `AnchorTracker.trackSearch` / `recoverySearch` call and are written
    /// in a single lock acquisition rather than three separate calls —
    /// splitting the write would let a reader observe, e.g., a moved
    /// `trackedPoint` still paired with the PREVIOUS position's
    /// `anchorSignature`, exactly the "buffer shape must travel with its
    /// data" hazard this project's A-18 discipline warns about elsewhere.
    ///
    /// - Parameters:
    ///   - trackedPoint: `nil` leaves the stored `trackedPoint` untouched.
    ///     The `.lost` transition (§24.2.3: "`trackedPoint` 冻结在跟丢那一刻
    ///     的位置") passes `nil` here **on purpose** — there is deliberately
    ///     no way to call this method and move `trackedPoint` while also
    ///     marking the instance `.lost`, mirroring `AnchorTracker`'s own
    ///     distinct-types-per-search-kind discipline (B-43) at the call-site
    ///     level instead.
    ///   - trackState: always written — every caller of this method is
    ///     transitioning the state machine (`.tracking` on a found / recovered
    ///     match, `.lost` on a failed search).
    ///   - anchorSignature: `nil` leaves the stored baseline untouched (the
    ///     `.lost` case: nothing was found, so nothing new is worth recording
    ///     as "what the object should look like"). Non-nil MUST be a signature
    ///     sampled at the NEW `trackedPoint`, never the pre-search sample —
    ///     recording the old position's content here would make the tracker
    ///     drift toward whatever it lost the object near, not toward the
    ///     object itself (see `checkAndFireReAnchor`'s step 6c comment for the
    ///     full argument against reusing the pre-search signature).
    ///
    /// Uses the pool's existing `lock`, like `setAnchorSignature` /
    /// `markReAnchorDispatched` — no new lock (§16.7). Silently no-ops when
    /// the instance is gone (evicted / cleared while the frame was being
    /// processed).
    func updateTracking(id: UUID, trackedPoint: CGPoint?, trackState: TrackState,
                        anchorSignature: AnchorSignature?) {
        lock.lock(); defer { lock.unlock() }
        guard let idx = instances.firstIndex(where: { $0.id == id }) else { return }
        if let trackedPoint {
            instances[idx].trackedPoint = trackedPoint
        }
        instances[idx].trackState = trackState
        if let anchorSignature {
            instances[idx].anchorSignature = anchorSignature
        }
    }

    /// Long-press / eviction path (§3.2).
    func removeInstance(id: UUID) {
        lock.lock(); defer { lock.unlock() }
        instances.removeAll { $0.id == id }
        // The removed instance may have been primary; keep exactly one primary.
        if !instances.isEmpty && !instances.contains(where: { $0.isPrimary }) {
            if let newestIdx = instances.indices.max(by: { instances[$0].createdAt < instances[$1].createdAt }) {
                instances[newestIdx].isPrimary = true
            }
        }
    }

    /// Double-tap → clear everything (§3.2).
    func clearAll() {
        lock.lock(); defer { lock.unlock() }
        instances.removeAll()
    }

    /// Promote an existing instance to primary without re-decoding (§3.2).
    /// Day 6 wires the "tap inside an existing mask" hit test to this; kept here
    /// because the rule is part of the locked §3.2 contract.
    @discardableResult
    func promoteToPrimary(id: UUID) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard instances.contains(where: { $0.id == id }) else { return false }
        for i in instances.indices { instances[i].isPrimary = (instances[i].id == id) }
        return true
    }

    // MARK: Queries

    /// True while `requestGen` is still the newest request for this instance and
    /// the instance is still in the pool.  Replaces the Day 4 global
    /// "is this the latest tap" check, which would have cancelled instance 1's
    /// decode the moment instance 2 was tapped.
    func isRequestCurrent(id: UUID, requestGen: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard let inst = instances.first(where: { $0.id == id }) else { return false }
        return inst.requestGen == requestGen
    }

    var count: Int {
        lock.lock(); defer { lock.unlock() }
        return instances.count
    }

    var isEmpty: Bool { count == 0 }

    func snapshot() -> [TapInstance] {
        lock.lock(); defer { lock.unlock() }
        return instances
    }

    /// Instances that currently have something to draw, oldest first.
    /// Draw order matters: the caller composites in this order so the primary
    /// (newest, always last) lands on top — see §3.4.
    func drawableInstances() -> [TapInstance] {
        lock.lock(); defer { lock.unlock() }
        return instances
            .filter { $0.maskAlpha != nil }
            .sorted { $0.createdAt < $1.createdAt }
    }

    /// One-line pool state for the tap log chain.
    func debugSummary() -> String {
        lock.lock(); defer { lock.unlock() }
        guard !instances.isEmpty else { return "pool=[]" }
        let parts = instances
            .sorted { $0.createdAt < $1.createdAt }
            .map { inst -> String in
                String(format: "#%d%@%@iou=%.2f",
                       inst.requestGen,
                       inst.isPrimary ? "*" : "",
                       inst.maskAlpha == nil ? "(pending) " : " ",
                       inst.iouPred)
            }
        return "pool=[\(parts.joined(separator: " | "))] n=\(instances.count)"
    }
}
