//
//  ReAnchorLoop.swift
//  JudgeE2
//
//  Phase 4 — Day 2–3 (Builder)
//
//  Throttle state machine for the re-anchor refresh loop.
//  Contract: shared/architect_output.md §16.2.3 (节流策略, D-15.2 compliant),
//  §16.4.2 (anchor frame snapshot timing), §16.5.3 (C4 atomic reset).
//
//  WHY THIS IS A STATE MACHINE AND NOT A TIMER
//  D-15.2 (§15.7.3) rejected the fixed 100 ms throttle that tasks.md's Day 2–3
//  bullet still names: at N = 3 a serial batch costs 191.1 ms with the fast-path
//  decode (63.7 ms) and 109.2 ms with the slow-path decode (36.4 ms), so the
//  100 ms window is under-provisioned in *both* measured regimes.  The batch is
//  therefore made to pace itself: at most one batch may be in flight, and a
//  drift event arriving while one is running is DROPPED — never queued, and
//  never allowed to cancel the running batch.
//
//  THREADING — NO NEW LOCK (§16.7).
//  Every member of this class is unsynchronised on purpose.  The owner
//  (`CameraManager`) mutates it only while its existing `stateLock` is held, so
//  this type adds no lock, no queue and no scheduling hop.  Each method below
//  restates that precondition; violating it is a data race.
//

import Foundation

/// Single-in-flight-batch gate for `CameraManager`'s re-anchor loop.
///
/// ⚠️ **Every property and method requires `CameraManager.stateLock` to be held
/// by the caller.**  This type deliberately owns no lock of its own.
final class ReAnchorLoop {

    /// True while a batch is in flight.  This is the whole of D-15.2's
    /// load-adaptive throttle: `checkAndFire` drops the drift event when set.
    private(set) var isReAnchoring: Bool = false

    /// Units of the current batch that have not finished yet.  Reaching zero
    /// releases `isReAnchoring`.
    private(set) var pendingCount: Int = 0

    /// Monotonic batch identifier.  Each dispatched unit captures the id of the
    /// batch it belongs to and may only decrement that batch's counter.
    ///
    /// This is what makes `reset()` safe.  C4 can fire while three decodes are
    /// still queued on `decoderQueue`; those closures cannot be cancelled, so
    /// they will still run and still call `completeUnit`.  Without the id they
    /// would decrement whatever batch happened to be current by then, driving
    /// the counter negative (which never re-reaches zero, deadlocking the
    /// throttle) or ending a *newer* batch early.  `reset()` bumps the id, so
    /// every orphaned unit's `completeUnit` becomes a no-op.
    private(set) var batchId: UInt64 = 0

    /// Geometry of the frame that started the current batch.
    ///
    /// ⛔ **TELEMETRY ONLY (§17.5.2).**  It was the drift baseline under §16.3.1;
    /// that signal was superseded (§17.1), and the baseline is now per-instance
    /// content (`TapInstance.anchorSignature`).  It is kept because a C4
    /// post-mortem may want to know the geometry a batch started under, and it
    /// must not be fed into any decision.  §16.4.2's *timing* rule (the baseline
    /// moves at batch start) survives unchanged — only its content changed.
    var lastAnchorGeometry: FrameGeometry?

    // MARK: - Batch lifecycle

    /// Claim the single in-flight slot for a batch of `count` instances.
    ///
    /// - Returns: the new batch id, or `nil` when a batch is already in flight
    ///   (D-15.2: drop the event, do not queue it) or `count <= 0`.
    /// - Precondition: `stateLock` held.
    func beginBatch(count: Int, anchor: FrameGeometry) -> UInt64? {
        guard !isReAnchoring, count > 0 else { return nil }
        batchId &+= 1
        isReAnchoring = true
        pendingCount = count
        lastAnchorGeometry = anchor          // telemetry only (§17.5.2)
        return batchId
    }

    /// Report one unit of `batch` as finished, successfully or not (§16.2.4: the
    /// counter must be decremented on *both* outcomes).
    ///
    /// - Returns: true when this call completed the batch and released the slot.
    /// - Precondition: `stateLock` held.
    @discardableResult
    func completeUnit(batch: UInt64) -> Bool {
        guard isReAnchoring, batch == batchId else { return false }  // orphaned unit
        pendingCount -= 1
        guard pendingCount <= 0 else { return false }
        pendingCount = 0
        isReAnchoring = false
        return true
    }

    /// Free the slot and orphan every unit of the current batch.
    ///
    /// §16.2.3 / §16.5.3: C4 (geometry-signature change → pool cleared) can land
    /// mid-batch.  If the flag were left set, the batch that owns it would never
    /// complete against a pool that no longer contains its instances and the
    /// throttle would block *permanently* on the next entry into `.tapToSegment`.
    ///
    /// - Precondition: `stateLock` held.
    func reset() {
        batchId &+= 1                 // orphan in-flight units (see `batchId`)
        isReAnchoring = false
        pendingCount = 0
        // `lastAnchorGeometry` is intentionally left alone: §16.4.2 notes that
        // with an empty pool no batch can fire, so its value is inert, and the
        // next batch overwrites it anyway.
    }
}
