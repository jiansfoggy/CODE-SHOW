//
//  CameraMotionGate.swift
//  JudgeE2
//
//  Phase 4C — R36 remediation (2026-08-24, user-directed direct fix; see
//  shared/tasks.md Day 7 re-anchor item).
//
//  R36: Capability C's tracking search (`AnchorTracker`, local block-matching
//  around `DriftDetector.signature`/`divergence`) cannot tell "the object
//  moved" apart from "the camera panned and the background slid under a
//  stationary search window" — a real-device session produced a confirmed
//  case of the displayed mask jumping onto unrelated background content
//  while the user panned the phone with the object itself stationary.
//
//  The fix implemented here is NOT a new design invented at this layer — it
//  is the suppression gate architect_output.md §17.2.1 already scoped for
//  exactly this failure mode when it first evaluated and rejected CoreMotion
//  as capability C's PRIMARY signal:
//
//    "CoreMotion 设备姿态 —— 否决（保留为未来的抑制门）... 保留用途：若 §17.4
//    的阈值在真机上被手抖误触发困扰，CoreMotion 的角速度可以作为**抑制门**
//    （角速度高于阈值时判定为甩动，跳过 re-anchor）在届时再行立项。它适合当
//    否决项，不适合当触发项。"
//
//  and R36's own disposition note (tasks.md) names the same idea as the
//  first listed remediation option: "只在检测到相机基本静止时才允许
//  trackedPoint 移动，可能需要 CoreMotion 或帧间全局位移估计作为门槛。"
//
//  ⚠️ SCOPE — VETO ONLY, never a trigger. This gate can only make Capability
//  C MORE conservative than it already is: while the device is judged to be
//  panning, `CameraManager.checkAndFireReAnchor` skips the tracking/recovery
//  search entirely, `trackedPoint` is left exactly where it was, and the
//  instance falls back to Capability A/B's ordinary frozen-`canonicalPoint`
//  re-anchor behaviour for that cycle — bit-identical to what already
//  happens whenever `DriftDetector.objectTrackingEnabled == false`. Nothing
//  here touches `DriftDetector`, `AnchorTracker`, the consistency gate, or
//  the accept path. A still phone with a moving object reads near-zero
//  rotation rate — this gate has nothing useful to say about that case
//  (CoreMotion's documented blind spot, §17.2.1 point 1) and correctly
//  leaves the search running for it, which is the case the search already
//  handles correctly.
//
//  Cost: pure CPU-side sensor fusion CoreMotion already runs continuously
//  system-wide — zero new ANE/GPU load, so this does not touch the
//  ISSUE-P4-DECODE contention surface that ruled out candidate A (Vision
//  optical flow) as a heavier alternative fix for the same defect.
//

import CoreMotion
import Foundation

/// Cheap, ANE/GPU-free "is the device panning right now" signal. Read by
/// `CameraManager.checkAndFireReAnchor` to veto Capability C's tracking
/// search during a pan (R36 remediation) — see file header for the full
/// rationale and scope.
enum CameraMotionGate {

    /// Rotation-rate magnitude, rad/s, at or above which the device is
    /// judged to be panning rather than merely held. Hand tremor while
    /// aiming a phone is typically well under 0.1 rad/s; a deliberate
    /// reframing pan (the R36 failure mode) is commonly upward of
    /// 0.3–0.5 rad/s. 0.20 rad/s (~11.5°/s) is a conservative first cut,
    /// deliberately biased toward suppressing the search too often rather
    /// than too rarely: a missed tracking opportunity costs a stale mask
    /// (the same, already-accepted Capability A/B fallback that runs today
    /// with tracking off); an unsuppressed pan costs the R36 wrong-object
    /// jump. Named `static var`, same tuning-surface shape as
    /// `DriftDetector`'s constants — a Debugger session may retune this on
    /// real-device data with no other code change. Not yet retuned against
    /// real-device data as of this fix; treat as a first cut pending a
    /// dedicated pan-vs-hold capture.
    static var panningThresholdRadPerSec: Double = 0.20

    private static let motionManager = CMMotionManager()
    private static let motionQueue: OperationQueue = {
        let q = OperationQueue()
        q.name = "camera.motion.gate.queue"
        q.maxConcurrentOperationCount = 1
        return q
    }()

    /// Guards `latestRotationRateMagnitude` / `hasSample`. CoreMotion
    /// delivers on `motionQueue`; `checkAndFireReAnchor` reads on
    /// videoQueue — both are background queues and neither may block on the
    /// other, so this is a plain lock around two scalars, not a queue hop.
    private static let lock = NSLock()
    private static var latestRotationRateMagnitude: Double = 0.0

    /// True once CoreMotion has delivered at least one sample since the most
    /// recent `start()`. Before that — no gyroscope hardware, or `start()`
    /// not yet called — `isPanning` reads `false` (fail open, the same "do
    /// not veto without evidence" discipline `DriftDetector`'s own gates
    /// already use, e.g. `alphaIoU`/`centroidAlignedIoU`'s "return 1.0 when
    /// the comparison cannot be made"). A device or session with no motion
    /// data must not silently disable Capability C.
    private static var hasSample = false

    /// Whether Capability C's tracking/recovery search should be skipped
    /// this cycle. Safe to call from any queue.
    static var isPanning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return hasSample && latestRotationRateMagnitude >= panningThresholdRadPerSec
    }

    /// Begins device-motion updates. Idempotent — safe to call more than
    /// once (e.g. `CameraManager.start()` being invoked again after a
    /// background/foreground cycle). No-op on hardware without a gyroscope
    /// (`isDeviceMotionAvailable == false`): `isPanning` then stays
    /// permanently `false` via the `hasSample` guard above, i.e. Capability
    /// C behaves exactly as it did before this file existed on such
    /// hardware — never as if panning were permanently detected.
    static func start() {
        guard motionManager.isDeviceMotionAvailable else { return }
        guard !motionManager.isDeviceMotionActive else { return }
        // 30 Hz: comfortably above this gate's only consumer
        // (`checkAndFireReAnchor`, itself throttled to
        // `DriftDetector.minReAnchorIntervalMs` = 300 ms, i.e. ≤3.3 Hz), so
        // every read sees a fresh-enough sample. This is pure CPU-side
        // sensor fusion the OS already runs continuously — not a new
        // capture stream, and no ANE/GPU cost.
        motionManager.deviceMotionUpdateInterval = 1.0 / 30.0
        motionManager.startDeviceMotionUpdates(to: motionQueue) { motion, _ in
            guard let motion = motion else { return }
            let r = motion.rotationRate
            let magnitude = (r.x * r.x + r.y * r.y + r.z * r.z).squareRoot()
            lock.lock()
            latestRotationRateMagnitude = magnitude
            hasSample = true
            lock.unlock()
        }
    }

    /// Stops device-motion updates, so the gyroscope is not kept spinning
    /// while the camera pipeline itself is stopped. Mirrors
    /// `CameraManager.stop()`'s existing session-lifecycle discipline.
    static func stop() {
        motionManager.stopDeviceMotionUpdates()
        lock.lock()
        hasSample = false
        lock.unlock()
    }
}
