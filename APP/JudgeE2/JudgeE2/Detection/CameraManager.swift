import AVFoundation
import Accelerate
import Combine
import CoreImage
import CoreML
import Foundation
import SwiftUI
import MachO
import UIKit

final class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()
    @Published var boxes: [CGRect] = []
    @Published var maskImage: CGImage?
    @Published var maskRotationAngle: CGFloat = 0
    @Published var maskMirrored: Bool = false

    /// Phase 3 Day 5 — vector outlines for the tap masks currently on screen
    /// (architect_output §3.4).  Kept SEPARATE from `maskImage` on purpose:
    /// constraint C-5 forbids baking the stroke into the 256×256 alpha tile, so
    /// the outline travels as geometry and `PreviewView` strokes it in pt.
    /// nil / empty ⇒ no stroke (also the C-6 "outline off" state).
    @Published var maskOutlines: MaskOutlineSet?

    // Phase 3 Day 4 — Tap segmentation UI feedback.
    // `tapProcessing` is true from the moment a tap is accepted until its mask is
    // rendered (or the request fails/aborts).  The overlay (Day 6) uses it to show
    // a loading pulse while the encoder is busy; Day 4 drives the flag so busy /
    // fallback states are observable and testable.
    @Published var tapProcessing: Bool = false
    // Last tap point in canonical space — Day 6 ripple-animation anchor.
    @Published var lastTapCanonicalPoint: CGPoint?
    // Last tap point in preview-view coordinates — anchors the loading indicator.
    @Published var lastTapViewPoint: CGPoint?
    // Sequence number of the most recently accepted tap (== `tapGeneration`).
    // Rendered on screen so a screen recording carries the same `#N` that every
    // `[TAP#N]` log line does — frame and log align without timestamp guessing.
    @Published var lastTapIndex: Int = 0

    /// Phase 3 Day 5 — Requirement C: a tap that failed must SAY SO.
    ///
    /// Architect §10.4 C: "静默清除加载指示是被禁止的行为".  Every path that
    /// clears `tapProcessing` without producing a mask publishes one of these
    /// instead; the overlay shows a red ring + reason at the tap point and
    /// auto-dismisses.  nil = nothing to report.
    @Published var tapFailure: TapFailure?

    /// True while the SAM encoder/decoder cold-start warmup is in flight.
    /// The UI shows an explicit "initialising" state instead of letting a tap
    /// fall into a wait that can run 1.3–8.6 s (Architect §10.4 C, G-3).
    @Published var samWarmingUp: Bool = false

    // Phase 3 Day 6 — Tap anchor markers (L3 visual carrier, architect §3.3.1).
    // One marker per instance whose mask has been placed; updated on the main
    // thread whenever the instance pool changes.  Keyed by instance UUID so
    // SwiftUI's `ForEach(tapAnchorMarkers)` can animate individual markers in
    // and out without re-rendering the whole set.
    @Published var tapAnchorMarkers: [TapAnchorMarker] = []

    // Phase 4B Day 5 — long-press-on-existing-mask hands a single locked
    // snapshot to the UI (§22.2.2 decision tree, PIN-6).  Present the
    // PinCreationSheet on `!= nil`; clear it on dismiss (save or cancel).
    // CameraManager itself never imports Persistence/PinStore/PinFactory —
    // this struct carries only Interaction-layer types (TapInstance,
    // FrameGeometry), keeping PIN-3's "zero Persistence symbols in
    // CameraManager.swift" invariant (§37.2) intact for Day 5 as it was Day 4.
    @Published var pinCreationDraft: PinCreationDraft?

    // Phase 4B Day 5 — which live instances have a saved Pin, main-thread only.
    // Purely a display concern (§22.2.4: the 📌 decorates the instance, not
    // any persisted record) — set by the UI after a successful
    // `PinStore.create`, via `markInstancePinned(id:tag:)`, never by this file
    // reading PinStore state itself.
    private var pinnedInstanceIDs: Set<UUID> = []
    // Phase 4B Day 6 — the tag text to show beside a pinned marker (task
    // instruction: label rendered next to the pin icon). A plain String, not
    // a Persistence symbol: PIN-3's whole-file invariant on this file is
    // unaffected. Populated from the same successful-save callback that sets
    // `pinnedInstanceIDs`; never read back from PinStore.
    private var pinnedInstanceTags: [UUID: String] = [:]
    // Phase 4B Day 6 — B-37 (§23.1.4): which live instances were produced by a
    // Pin revisit decode, and the origin Pin's tag (value nil = untagged Pin).
    // Provenance-only decoration under PIN-7 — it states "this segmentation
    // was initiated from that Pin's record" (record + branch facts), never
    // object identity.  Same lifecycle discipline as `pinnedInstanceIDs`
    // (§22.2.4: decorates the TapInstance, dies with it); populated from the
    // revisit flow's placed-observation callback, never read from PinStore.
    // Main-thread only, like every other `tapAnchorMarkers` input.
    private var revisitOriginPinTags: [UUID: String?] = [:]

    // Phase 4B Day 6 — revisit instrumentation hooks (§19.4.6, §23.1.9 B-40).
    // All three fire from the SAME main-thread blocks that already publish
    // `maskImage`/`tapFailure`/promote state for every tap, real or
    // revisit-originated — nothing here changes tap routing, timing, or any
    // existing invariant; a nil closure costs one optional check.
    //
    // PIN-8 (§23.5): these are PURE OBSERVATION exits — no control flow may
    // read them or branch on them (that would be a second decode path in
    // disguise, violating PIN-5) — and `PinInterfaces.swift`'s
    // `PinRevisitTracker` is the SOLE installer of all three in the whole
    // app; every tap generation it registers must be released by exactly one
    // of placed / failed / promoted.
    var onTapMaskPlaced: ((_ gen: Int, _ path: TapPath, _ instanceID: UUID, _ alpha: [UInt8]) -> Void)?
    var onTapFailed: ((_ gen: Int, _ message: String) -> Void)?
    // B-40 ① (§23.1.9) — the third hook: a tap resolved by the §3.2 promote
    // branch (no decode). Closes outcome R-D's observability gap: without it
    // a revisit that lands inside an existing live mask produced no log line
    // and leaked its tracker entry.
    var onTapPromoted: ((_ gen: Int, _ promotedInstanceID: UUID) -> Void)?

    /// Phase 4B Day 6 — outcome of the most recent Pin revisit that did NOT
    /// end in an ordinary active mask (§19.4.4 R-B / R-C). Deliberately no
    /// Persistence-layer symbol (PIN-3): `pinID` is enough for a caller that
    /// already holds a `PinStore` to look the record back up and, e.g., open
    /// the static fallback viewer.
    struct PinRevisitEvent: Equatable {
        /// `.promoted` is outcome R-D (§23.1.9): the revisit's derived point
        /// landed inside an existing live mask, so the §3.2 promote rule
        /// resolved it with no decode.  `fromPinTag` is the origin Pin's tag
        /// (nil = untagged), rendered ONLY inside the `From Pin "…"` phrase
        /// (PIN-7 / §23.1.4 wording rule).  A plain String — PIN-3 intact.
        enum Kind: Equatable { case refused(reason: String), failed, promoted(fromPinTag: String?) }
        let pinID: UUID
        let kind: Kind
        // Makes two textually-identical outcomes distinguishable to
        // `onChange`, so a second consecutive rejection re-shows its banner.
        private let token = UUID()
    }
    @Published var pinRevisitEvent: PinRevisitEvent?

    /// Phase 4B Day 6 (§19.4.4 R-A) — true for the duration of the most
    /// recently accepted tap iff that tap originated from `handleTap(fromPin:)`
    /// rather than a screen tap. Reset to false at the top of every ordinary
    /// `handleTap(canonicalPoint:viewPoint:)` call (below) and set true by the
    /// revisit wrapper immediately afterward — purely a UI copy switch ("在该
    /// 位置重新分割" vs no caption), not a second waiting mechanism: the
    /// existing `tapProcessing` pulse (§15.3 D-15.1) is what actually drives
    /// the UI state.
    @Published var lastTapWasRevisit: Bool = false

    // Phase 3 Day 6 — "请点击分割" hint after clearAll.
    // Shown for ~1.5 s then auto-dismissed.  Not published directly: managed
    // via the `showSegmentHint` flag on the main thread.
    @Published var showSegmentHint: Bool = false

    // MARK: - DEBUG: 强制慢路径 (slow-path latency sampling)
    //
    // When true:
    //   • embeddingCache is cleared immediately (slow path must encode from scratch)
    //   • refreshTapEmbeddingIfNeeded() is skipped (background refresh cannot refill the cache)
    // This ensures every tap is a genuine slow-path encode+decode, making
    // slow-path latency samples collectable without timing the background refresh.
    //
    // Default is false — zero effect on normal operation when disabled.
    // Remove this property (and its two use-sites below) when sampling is done.
    @Published var forceSlowPath: Bool = false {
        didSet {
            if forceSlowPath {
                stateLock.lock(); embeddingCache = nil; stateLock.unlock()
                perfLog("[DEBUG] forceSlowPath=ON — embeddingCache cleared, background refresh suspended")
            } else {
                perfLog("[DEBUG] forceSlowPath=OFF — background refresh resumed")
            }
        }
    }

    // MARK: - DEBUG: suspend background refresh ONLY (§18.4.3 / B-12)
    //
    // When true, `refreshTapEmbeddingIfNeeded` returns immediately and NOTHING
    // ELSE happens: the embedding already in `embeddingCache` stays exactly
    // where it is.
    //
    // WHY IT IS NOT `forceSlowPath`.  `forceSlowPath` does two things at once —
    // it clears the cache AND suspends refresh — which makes the "no contention"
    // reference workload it produces a *compound* one ("no contention AND a hot
    // embedding").  §18.4.2's A-11 rules that a reference differing from the
    // measured workload in two variables cannot separate anything, which is
    // precisely why ISSUE-P4-DECODE's hypothesis (b) never had a clean baseline.
    // This switch changes one variable.  The two are deliberately independent:
    // neither implies the other, and they must not be merged.
    //
    // Not exposed in UI; default false; nothing reads it outside
    // `refreshTapEmbeddingIfNeeded`.  This round only installs the switch —
    // §18.4.3 defers the 2×2 factorial that uses it.
    @Published var suspendRefreshOnly: Bool = false

    /// Which *route* a tap took to reach the decoder.
    ///
    /// Deliberately separate from `reusedEmbedding`, which answers a different
    /// question — "did this tap pay for its own encode?".  The two were one
    /// parameter until Day 5, and because `drainPendingTaps` hard-coded
    /// `reusedEmbedding: true` (correct as billing: a parked tap rides someone
    /// else's encode) every parked tap was logged `fast/decode-only`.  Those
    /// taps had waited out a full encode, so they silently inflated the
    /// fast-path p95 they were never part of (Debugger D-12).
    enum TapPath {
        /// A usable cached embedding existed at tap time → straight to decoderQueue.
        case fast
        /// No usable cache → this tap started, and waited for, its own encode.
        case slow
        /// Parked behind an in-flight encode (tap / warmup / background refresh),
        /// then decoded when that encode published its embedding.  Decode-only in
        /// billing terms, but the user waited for the encode — it is NOT fast.
        case parked

        var label: String {
            switch self {
            case .fast:   return "fast/decode-only"
            case .slow:   return "slow/encode+decode"
            case .parked: return "slow/parked→decode-only"
            }
        }
    }

    /// A tap that could not be served, with the reason the user sees.
    struct TapFailure: Equatable {
        let index: Int          // tap sequence number (`[TAP#N]`)
        let message: String     // short, user-facing
        let viewPoint: CGPoint? // anchor in preview-view coordinates
    }

    /// Persistent anchor dot shown at a placed mask's tap position (L3 carrier).
    /// Published on the main thread; one per drawable instance in the pool.
    struct TapAnchorMarker: Identifiable {
        let id: UUID           // == TapInstance.id
        let viewPoint: CGPoint // view-space position of the original tap
        let slotIndex: Int     // 0, 1, or 2 — matches the instance's colour slot
        let requestGen: Int    // tap sequence number; telemetry / debug only
        let isPrimary: Bool    // newest live instance — drawn with extra weight
        /// Phase 4B Day 5 (§22.2.4) — true once this instance has a saved Pin.
        /// ⚠️ Decorates the *instance*, not any persisted record: when the
        /// instance is evicted (FIFO) or cleared (C1/C3/C4), this flag — and
        /// the marker itself — disappear with it, exactly like every other
        /// tap-marker attribute.  The saved Pin on disk is unaffected; finding
        /// it again is PinList's job (Day 6), not this live view's.
        let isPinned: Bool
        /// Phase 4B Day 6 — the tag text to render beside the marker, when
        /// the instance is pinned and was saved with a non-empty tag. nil
        /// otherwise (unpinned, or pinned with no tag).
        let tag: String?
        /// B-37 (§23.1.4, PIN-7) — true when this instance's mask was decoded
        /// by a Pin revisit.  Drawn with the ↻ glyph (never 📌 — the mask has
        /// no disk record, §23.1.5) and the `From Pin "…"` provenance label.
        /// Replaced by 📌 if the instance is later long-press saved
        /// (§23.1.6-2); dies with the instance like every other decoration.
        let isRevisitOrigin: Bool
        /// The origin Pin's tag (nil = untagged Pin).  Meaningful only when
        /// `isRevisitOrigin`; per §23.1.4's wording rule it may appear ONLY
        /// inside the quoted `From Pin "<tag>"` phrase, never bare.
        let revisitPinTag: String?
    }

    /// Phase 4B Day 5 — a single-snapshot capture of "long-press hit an
    /// existing mask", ready for `PinCreationSheet`.
    ///
    /// PIN-6 (§22.2.3): `instance` is captured **once**, under
    /// `TapInstanceManager`'s existing lock (via `.snapshot()`), at long-press
    /// time.  Everything the sheet shows (preview) and everything it saves
    /// (`PinStore.create`) must read from this same value — never re-fetch
    /// from `tapInstances` between preview and save.
    struct PinCreationDraft: Identifiable {
        var id: UUID { instance.id }
        let instance: TapInstance
        let geometry: FrameGeometry
    }

    var currentMode: AppMode = .detectionOnly

    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let videoQueue = DispatchQueue(label: "camera.video.queue")
    private let ciContext = CIContext()
    private var model: yolov9_c?
    private var isProcessing = false
    private var currentPosition: AVCaptureDevice.Position = .back
    private var backend: InferenceBackend = .all
    private var videoConnection: AVCaptureConnection?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var lastRotationAngle: CGFloat = 90   // portrait default; kept as-is when device lies flat

    // MARK: Rotation (P1 180° 修复)
    // iOS 17+ 用 AVCaptureDevice.RotationCoordinator 作为唯一角度来源:
    // 它按具体相机的传感器装配方向给角度(前/后摄横屏相差 180°),
    // 手写 UIDeviceOrientation 映射表无法同时对两颗摄像头正确。
    // 用 AnyObject 存储是因为 stored property 不能加 @available 限定。
    private var rotationCoordinator: AnyObject?
    private var captureRotationObservation: NSKeyValueObservation?
    private var previewRotationObservation: NSKeyValueObservation?
    private weak var previewLayer: AVCaptureVideoPreviewLayer?
    private var activeDevice: AVCaptureDevice?

    private var inferenceTimesMs: [Double] = []
    // 50 frames ≈ 21 s at the ~2.4 FPS seen in segmentation / tapToSegment,
    // so a capture session yields two stats lines instead of one (100 frames
    // took ~42 s and users typically stopped just short of the second line).
    private let inferenceStatsWindow = 50

    // Companion window to `inferenceTimesMs`: post-NMS detection count per frame.
    // Quiet mode suppresses the per-frame `final_detections:` line, but
    // checklist §2.6 uses that number to confirm the four scenes were shot with
    // comparable content — so quiet mode emits this window mean instead.
    // Appended exactly once per successful frame, on the same videoQueue pass
    // that appends `inferenceTimesMs`, so both windows always close together.
    private var detectionCounts: [Int] = []

    // SAM encoder sliding-window stats (n=100, cold-start excluded)
    // Only accessed on encoderQueue — no extra lock needed.
    private var samEncoderTimesMs: [Double] = []
    private let samEncoderStatsWindow = 100
    private var samEncoderCallCount = 0    // first call is ANE cold-start; skipped from stats

    // SAM decoder sliding-window stats (n=100, cold-start excluded)
    // Only accessed on decoderQueue — no extra lock needed.
    private var samDecoderTimesMs: [Double] = []
    private let samDecoderStatsWindow = 100
    private var samDecoderCallCount = 0    // first call is ANE cold-start; skipped from stats

    // Phase 3 Day 4 — Tap-path encoder latency stats for the resolution AB test.
    // Accessed only on encoderQueue (serial), so no extra lock needed.
    private var tapEncoderTimesMs: [Double] = []
    private var tapEncoderCallCount = 0    // first call is ANE cold-start; excluded

    private var lastPreprocessMs: Double = 0

    struct Detection {
        let x1: Float
        let y1: Float
        let x2: Float
        let y2: Float
        let score: Float
        let classId: Int
    }

    private struct LetterboxInfo {
        let origW: Float
        let origH: Float
        let scale: Float
        let padX: Float
        let padY: Float
        let inputSize: Float
    }
    private var lastLetterbox: LetterboxInfo?
    private var latestInputBuffer: CVPixelBuffer?
    private var latestCameraBuffer: CVPixelBuffer?

    // MARK: - Segmentation queues & locks
    // Encoder and decoder run on *separate* serial queues so a slow encode
    // never delays mask updates from the decoder.
    private let encoderQueue = DispatchQueue(label: "sam.encoder.queue", qos: .userInitiated)
    private let decoderQueue = DispatchQueue(label: "sam.decoder.queue", qos: .userInitiated)

    // stateLock guards: isEncoding, isDecoding, embeddingCache (written on encoderQueue, read on videoQueue)
    private let stateLock = NSLock()
    private var isEncoding = false
    private var isDecoding = false

    // D-1 observability for the two `guard !isEncoding` exits that used to return
    // silently.  Both are read/written only while `stateLock` is already held at
    // the exit itself — no extra lock, no extra queue.
    /// Times `refreshTapEmbeddingIfNeeded` lost the encoder slot to another queue
    /// between its two lock acquisitions.  Expected to stay at or near 0.
    private var refreshSlotLostCount = 0
    /// Times `scheduleEncoder` (Phase 2 per-frame path) found the slot busy.
    /// Expected to grow steadily — an encode takes ~1 s and frames arrive at 30 Hz.
    private var encoderSlotBusyDropCount = 0

    // App background state, cached from UIApplication notifications.
    // The inference queues MUST NOT read UIApplication.applicationState via
    // DispatchQueue.main.sync: when the main thread is itself waiting (mask
    // publish, UI work) the inference queue blocks with it, the encoder slot is
    // never released and the watchdog terminates the app — which the user
    // perceives as a random crash.
    private let appStateLock = NSLock()
    private var appIsBackgrounded = false
    private var isAppBackgrounded: Bool {
        appStateLock.lock(); defer { appStateLock.unlock() }
        return appIsBackgrounded
    }

    // Pre-allocated transpose buffer: [8400 × 84] Floats = 2.8 MB, reused every frame.
    // Avoids the per-frame heap alloc that caused Post ~250-400 ms regression.
    private var transposeBuffer = [Float](repeating: 0, count: 8400 * 84)

    // Pre-allocated 640×640 BGRA letterbox output buffer (reused every frame on videoQueue).
    // Eliminates per-frame CVPixelBufferCreate heap alloc that caused Pre spikes of 10-16 ms.
    private var letterboxOutputBuffer: CVPixelBuffer?

    // Model ownership by queue (fixes a data race that could hand a partially
    // initialised model to another queue):
    //   samEncoder — created and used ONLY on encoderQueue
    //   samDecoder — created and used ONLY on decoderQueue
    // Callers go through encoderForQueue / decoderForQueue, which also rebuild the
    // model when the requested compute units changed.
    private var samEncoder: SAMEncoder?
    private var samEncoderUnits: MLComputeUnits?
    private var samDecoder: SAMDecoder?
    private var samDecoderUnits: MLComputeUnits?
    // True once the box decoder MLModel has been constructed on decoderQueue for the
    // current samDecoder instance.  Reset in decoderForQueue when a new SAMDecoder is
    // created (e.g. after setBackend).  decoderQueue only — no lock needed.
    private var boxDecoderPrebuilt = false
    private let maskRenderer = MaskRenderer()

    // Cadence is TIME-based, not frame-based: under heavy load (YOLO + SAM
    // sharing the ANE) the effective fps can collapse to well under 1, and a
    // frame-counted cadence collapses with it (12 frames ≈ tens of seconds —
    // observed on device as "stable mask takes ~30 s").
    private var encoderRefreshMs: Double = 4000  // re-encode when embedding older than 4 s
    private var decoderRefreshMs: Double = 600   // re-decode at most ~1.7 Hz
    private var lastSegDecodeMs: Double = 0      // videoQueue-only
    private var tapModeFrameCount: Int = 0       // videoQueue-only; YOLO throttle in tap mode
    private var frameIndex: Int = 0

    // Access to embeddingCache must be done while holding stateLock.
    private var embeddingCache: (embedding: MLMultiArray, timestampMs: Double)?

    /// How many *newly computed* embeddings have been published into
    /// `embeddingCache` (§18.1.4, RE-1).  Written on encoderQueue at each of the
    /// four write-back sites, read on videoQueue by `checkAndFireReAnchor`;
    /// **guarded by the same `stateLock` as `embeddingCache` itself**, always in
    /// the same acquisition, so a reader can never see a generation number that
    /// does not belong to the embedding it also read.  No new lock (§16.7).
    ///
    /// Only a write-back increments it.  Clearing the cache to nil does **not**:
    /// the counter names embeddings, not cache states, and an instance that has
    /// consumed generation g must still be considered "already refreshed
    /// against g" after a clear-and-recompute — the recompute publishes g+1 and
    /// re-opens the gate on its own.
    ///
    /// UInt64 at ≈1 embedding/5 s does not wrap in any reachable session, so no
    /// wrap handling is written; `!=` (rather than `<`) is the eligibility test
    /// anyway, which is wrap-safe by construction.
    private var embeddingGeneration: UInt64 = 0

    // Tap sequence number. Every accepted tap gets one; it stamps the whole
    // `[TAP#N]` log chain and the on-screen counter.
    //
    // Day 5 note: it is NO LONGER a global supersession token.  With up to three
    // live instances, "only the newest tap renders" would cancel instance 1 the
    // moment instance 2 is tapped.  Supersession is now per-instance —
    // `TapInstanceManager.isRequestCurrent(id:requestGen:)` (Architect R4).
    // Guarded by stateLock.
    private var tapGeneration: Int = 0
    // Taps that arrived while the encoder was busy — decoded as soon as the
    // in-flight encode caches its embedding. Guarded by stateLock.
    // Bounded by TapInstanceManager.maxInstances: R4 forbids unbounded queueing.
    private var pendingTaps: [(id: UUID, point: CGPoint, gen: Int, startMs: Double,
                               lockDoneMs: Double)] = []
    // Tap sequence number → owning instance, for every request that has neither
    // rendered nor failed yet.  `tapProcessing` is simply "this map is non-empty"
    // — with several instances in flight a single Bool owned by the last writer
    // would flicker off early.  The instance id is needed so an eviction can
    // retire its request quietly instead of leaving it to time out.
    // Guarded by stateLock.
    private var inFlightTaps: [Int: UUID] = [:]
    // When the last tap was accepted; the background refresh yields to recent
    // taps so it never competes with them for the encoder. Guarded by stateLock.
    private var lastTapAcceptedMs: Double = 0

    // MARK: Requirement A — state the fast path reads off videoQueue
    //
    // The tap fast path needs letterbox geometry + the cached embedding and
    // NOTHING from the camera buffer, so Architect §10.4 A moves its decision
    // onto the gesture thread and straight to decoderQueue.  That turns
    // `lastLetterbox` / `backend` / `lastRotationAngle` / `currentPosition` —
    // until now videoQueue- and sessionQueue-owned — into cross-queue reads.
    // Rather than leave them unsynchronised, videoQueue/sessionQueue publish a
    // snapshot under `stateLock` and the gesture thread reads only that.
    // Trading a queue hop for a data race would not be an improvement.
    private struct TapGeometrySnapshot {
        let letterbox: LetterboxInfo
        let rotation: CGFloat
        let mirrored: Bool
    }
    /// Published on videoQueue at the end of every letterbox pass. stateLock.
    private var tapGeometryMirror: TapGeometrySnapshot?
    /// Published on sessionQueue by setBackend. stateLock.
    private var backendMirror: InferenceBackend = .all

    /// Tap instance pool (Day 5, architect_output §3).
    private let tapInstances = TapInstanceManager()

    /// Re-anchor throttle state (Phase 4 Day 2–3, architect_output §16.2.3).
    /// **Guarded by `stateLock`** — the type owns no lock of its own, which is
    /// how §16.7's "禁止新增 DispatchQueue 或锁对象" is honoured.
    private let reAnchor = ReAnchorLoop()

    /// How many consecutive frames `checkAndFireReAnchor` has skipped for want
    /// of an embedding (§16.6.3).  Log rate limiter only — `checkAndFireReAnchor`
    /// runs on videoQueue exclusively, so this needs no synchronisation.
    private var reAnchorNoEmbeddingSkips: Int = 0

    /// When the last re-anchor batch fired, in `PerfLogger.nowMs()` (§17.5.2 /
    /// B-8).  videoQueue-exclusive, like `tapModeFrameCount` — written and read
    /// only by `checkAndFireReAnchor`, so it needs no synchronisation.
    /// Enforces `DriftDetector.minReAnchorIntervalMs`, which §17.4 defines as a
    /// hard LOWER BOUND on firing rate, not as the throttle: load-adaptive
    /// throttling remains the single-in-flight-batch machine of §16.2.3.
    private var lastReAnchorFireMs: Double = 0

    /// B-47 (§24.3.2/§24.3.4) — the largest `DriftDetector.Drift.divergenceLuma`
    /// observed across all instances measured in the most recent
    /// `checkAndFireReAnchor` pass, regardless of whether that pass fired a
    /// re-anchor batch. Written once per pass, videoQueue-only (same discipline
    /// as `lastReAnchorFireMs`), and read by `refreshTapEmbeddingIfNeeded`'s
    /// heavy-drift bypass so a severe divergence can justify a re-*encode* even
    /// though `checkAndFireReAnchor` only ever triggers a decode-only re-anchor.
    /// This is purely a read of a value `checkAndFireReAnchor` already computes
    /// for its own `measured` array — no new sampling.
    private var lastObservedMaxDriftLuma: Double = 0

    /// B-47 (§24.3.2) — when the heavy-drift bypass in `refreshTapEmbeddingIfNeeded`
    /// last actually triggered a background re-encode, in `PerfLogger.nowMs()`.
    /// Deliberately a separate clock from `lastReAnchorFireMs`: that one throttles
    /// decode-only re-anchor batches, this one throttles the much more expensive
    /// re-encode bypass, and the two must stay independent so neither throttle's
    /// bookkeeping leaks into the other's semantics. videoQueue-only, like
    /// `lastReAnchorFireMs`.
    private var lastHeavyDriftRefreshMs: Double = 0

    // Requirement C — tiered fallback timeouts (Architect §10.4 C).
    // Fast path is decode-only (~61 ms measured), so 1.5 s is already 20×.
    // Slow path must cover the measured encoder cold-start upper bound of
    // 8605 ms with margin; a 3 s blanket value was logically incompatible with
    // that distribution and produced the "点了没反应" failure mode (G-3).
    private static let fastPathTimeoutSec: TimeInterval = 1.5
    private static let slowPathTimeoutSec: TimeInterval = 12.0
    /// How long a failure banner stays on screen before auto-dismissing.
    private static let tapFailureDisplaySec: TimeInterval = 1.6

    /// Set on videoQueue when warmup could not run yet **because no camera frame
    /// exists**.  The next frame kicks it. videoQueue-only.
    ///
    /// D-14: this used to cover the encoder-slot-busy case as well, and that is
    /// what made warmup spin.  "No frame yet" is genuinely a per-frame condition
    /// — the next frame either resolves it or does not — so retrying on the next
    /// frame is the right answer here and only here.  A busy encoder slot is a
    /// *one event away* condition and is now handled by `warmupWaitingOnEncode`.
    private var warmupPending = false

    /// One-shot continuation: warmup deferred because another encode already
    /// owned the slot, and that encode will publish an embedding when it ends.
    ///
    /// D-14: before this existed, warmup re-armed `warmupPending` on every such
    /// deferral, so during the ~8 s ANE cold start (owned by the background
    /// refresh) every single camera frame ran a full warmup attempt, found the
    /// slot busy, re-armed, and logged two lines — 40+ round trips that could
    /// not possibly succeed, because nothing about the state changed until that
    /// one encode finished.  Waiting for *that* encode is the correct trigger.
    ///
    /// **Guarded by `stateLock`, and always written under the same lock
    /// acquisition that observed `isEncoding == true`.**  The owner flips
    /// `isEncoding` to false under that same lock before calling
    /// `encodeSlotDidFinish`, which consumes this flag — so the wakeup can
    /// never be lost between the observation and the registration.
    private var warmupWaitingOnEncode = false

    /// Extra deferrals that arrived while a continuation was already pending.
    /// Reported once on resume instead of once per attempt.  stateLock-guarded.
    private var warmupDeferralsFolded = 0

    /// Which call site currently owns the encoder slot, or nil when the slot is
    /// free.  stateLock-guarded; written at exactly the points that flip
    /// `isEncoding`.  Exists so the two warmup deferral reasons can be told
    /// apart in code rather than assumed (see `EncodeSlotOwner`).
    private var encodeSlotOwner: EncodeSlotOwner?

    /// Owner of the single encoder slot (`isEncoding`).
    private enum EncodeSlotOwner: String {
        case warmup            = "warmup"
        case tap               = "tap"
        case backgroundRefresh = "background-refresh"
        case segmentationFrame = "segmentation-frame"

        /// True when a successful run of this encode publishes into
        /// `embeddingCache` **and** releases the slot through
        /// `encodeSlotDidFinish`.  Only such an encode can satisfy a deferred
        /// warmup by finishing; anything else must fall back to the next-frame
        /// re-arm or the warmup would wait forever.
        ///
        /// The switch is deliberately exhaustive with no `default`: a future
        /// encode path that does neither has to declare itself here instead of
        /// silently stranding a waiting warmup.
        var canSatisfyDeferredWarmup: Bool {
            switch self {
            case .warmup, .tap, .backgroundRefresh, .segmentationFrame:
                return true
            }
        }
    }

    /// True once the decoder has executed one rehearsal decode.  Guarded by
    /// `stateLock`; reset wherever `samDecoder` is dropped, since a new decoder
    /// instance is cold again.
    private var decoderWarmupDecodeDone = false

    /// D-14 log de-duplication for the decoder rehearsal: "decode deferred,
    /// no embedding yet" is one *state*, not one event per frame.  It is printed
    /// on the first deferral of an arm cycle; later deferrals are counted and
    /// reported once when the rehearsal finally runs.  Both stateLock-guarded
    /// and reset wherever `decoderWarmupDecodeDone` is reset.
    private var decoderWarmupDeferralLogged = false
    private var decoderWarmupDeferralsFolded = 0

    private var encoderHitCount: Int = 0
    private var encoderMissCount: Int = 0
    private var decoderCount: Int = 0
    private let debugSegmentation: Bool = true
    private var debugFrameCount: Int = 0

    // TemporalManager owns: primary-object selection, drift classification,
    // mask cache + TTL (default 1200 ms), geometry-signature tracking,
    // and embedding TTL validation (default 8000 ms).
    private let temporal = TemporalManager()

    override init() {
        super.init()
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(handleOrientationChange),
                                               name: UIDevice.orientationDidChangeNotification,
                                               object: nil)
        // Background state is tracked by notification (main thread) and read
        // lock-free-ish from the inference queues — see `isAppBackgrounded`.
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(handleAppDidEnterBackground),
                                               name: UIApplication.didEnterBackgroundNotification,
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(handleAppWillEnterForeground),
                                               name: UIApplication.willEnterForegroundNotification,
                                               object: nil)
        DispatchQueue.main.async { [weak self] in
            self?.setAppBackgrounded(UIApplication.shared.applicationState == .background)
        }
        sessionQueue.async { [weak self] in
            self?.configureSession()
        }
    }

    @objc private func handleAppDidEnterBackground() { setAppBackgrounded(true) }
    @objc private func handleAppWillEnterForeground() { setAppBackgrounded(false) }

    private func setAppBackgrounded(_ value: Bool) {
        appStateLock.lock(); appIsBackgrounded = value; appStateLock.unlock()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        UIDevice.current.endGeneratingDeviceOrientationNotifications()
    }

    func start() {
        // R36 remediation: bracket the camera session's active lifetime,
        // same as `session.startRunning()`/`stopRunning()` below — see
        // `CameraMotionGate.swift` for why this exists and its scope.
        CameraMotionGate.start()
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }

    func setBackend(_ backend: InferenceBackend) {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            // D-17: avoid redundant reloadModel + encoder drops when the same
            // backend is applied more than once (e.g. onAppear + onChange both
            // firing at startup with the same persisted value).
            guard self.backend != backend else { return }
            self.backend = backend
            // Requirement A: the gesture thread reads compute units from the
            // stateLock mirror, never from `backend` (sessionQueue-owned).
            self.stateLock.lock(); self.backendMirror = backend; self.stateLock.unlock()
            self.reloadModel()
            // SAM encoder/decoder are lazily created and cached — drop them so the
            // next encode/decode rebuilds with the new compute units (mirrors
            // setEncoderResolution). Without this the picker has no effect on SAM.
            self.videoQueue.async {
                self.stateLock.lock()
                self.embeddingCache = nil
                self.isEncoding = false
                self.encodeSlotOwner = nil
                self.isDecoding = false
                // The slot is being force-released, not completed, so no
                // `encodeSlotDidFinish` will run — drop any pending warmup
                // continuation instead of leaving it waiting on an encode that
                // will never report in.  Warmup is re-kicked below anyway.
                self.warmupWaitingOnEncode = false
                self.warmupDeferralsFolded = 0
                self.pendingTaps.removeAll()
                self.inFlightTaps.removeAll()
                self.stateLock.unlock()
                // Each model is dropped on its owning queue (never cross-queue).
                // D-17: if a cold-load raced and completed before this block
                // executes, the log line below flags it so race regressions are
                // visible in device logs without a second guessing exercise.
                self.encoderQueue.async {
                    if self.samEncoder != nil {
                        diagLog("[SAM] encoder: dropped by setBackend cleanup (was built)")
                    }
                    self.samEncoder = nil
                    self.samEncoderUnits = nil
                }
                self.decoderQueue.async {
                    self.samDecoder = nil
                    self.samDecoderUnits = nil
                    // A rebuilt decoder is cold again — re-arm its warmup.
                    self.stateLock.lock()
                    self.decoderWarmupDecodeDone = false
                    self.decoderWarmupDeferralLogged = false
                    self.decoderWarmupDeferralsFolded = 0
                    self.stateLock.unlock()
                }
                self.temporal.invalidateMask()
                self.temporal.resetTapGeometry()
                self.tapInstances.clearAll()
                self.resetReAnchorState()   // §16.2.3 — pool emptied, free the slot
                DispatchQueue.main.async {
                    self.maskImage = nil
                    self.maskOutlines = nil
                    self.tapProcessing = false
                    self.tapFailure = nil
                }
                // Re-warm for modes that rely on a cached embedding.
                if self.currentMode == .segmentation || self.currentMode == .tapToSegment {
                    self.warmupSegmentationIfPossible()
                }
            }
        }
    }

    /// Phase 3 Day 4 — AB test toggle for encoder input resolution (1024 default / 768).
    /// Switching invalidates the embedding cache + SAM encoder instance because the
    /// geometry signature (inputSize) and the loaded model both change.  Architect
    /// C-1: default is 1024; 768 is opt-in for the AB test only.
    func setEncoderResolution(_ resolution: SAMConfiguration.EncoderResolution) {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            guard SAMConfiguration.encoderResolution != resolution else { return }
            SAMConfiguration.encoderResolution = resolution
            perfLog("[AB] encoder resolution → \(resolution.rawValue) (feat=\(resolution.featureSize))")
            // Drop the loaded encoder + caches so the next encode rebuilds at the
            // new resolution.  Must run on videoQueue for cache/state safety.
            self.videoQueue.async {
                self.stateLock.lock()
                self.embeddingCache = nil
                self.isEncoding = false
                self.encodeSlotOwner = nil
                // Force-release, not completion — see the identical note in
                // `setBackend`.
                self.warmupWaitingOnEncode = false
                self.warmupDeferralsFolded = 0
                self.pendingTaps.removeAll()
                self.inFlightTaps.removeAll()
                self.stateLock.unlock()
                // Dropped on encoderQueue (its owning queue); rebuilt lazily at the
                // new resolution on the next encode.
                // AB stats are encoderQueue-owned — reset them there as well.
                self.encoderQueue.async {
                    self.samEncoder = nil
                    self.samEncoderUnits = nil
                    self.tapEncoderTimesMs.removeAll()
                    self.tapEncoderCallCount = 0
                }
                self.temporal.invalidateMask()
                self.temporal.resetPrimary()
                self.temporal.resetTapGeometry()
                self.tapInstances.clearAll()
                self.resetReAnchorState()   // §16.2.3 — pool emptied, free the slot
                DispatchQueue.main.async {
                    self.maskImage = nil
                    self.maskOutlines = nil
                    self.tapProcessing = false
                    self.tapFailure = nil
                }
            }
        }
    }

    func setMode(_ mode: AppMode) {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            // O1 (checklist §2.4): a loud, greppable separator so a single
            // performance log can be sliced into its four per-mode segments and
            // the first stats line after each switch discarded.  Kept in quiet
            // mode — without it the segments are unidentifiable.
            perfLog("=== MODE SWITCH → \(mode.rawValue) ===")
            self.currentMode = mode
            // The rehearsal decode is now mode-specific (point vs box), and the
            // box decoder is built on demand, so a latch left set by the
            // previous mode would leave the incoming mode's decoder cold —
            // exactly the first-use stall this warmup exists to prevent.
            // Re-arming costs at most one extra throwaway decode.
            self.stateLock.lock()
            self.decoderWarmupDecodeDone = false
            self.decoderWarmupDeferralLogged = false
            self.decoderWarmupDeferralsFolded = 0
            self.stateLock.unlock()
            // §3.2: leaving .tapToSegment clears every instance.  Done for all
            // three destinations (including re-entering tapToSegment) because a
            // pool built against an older geometry must never survive a switch.
            self.discardAllTapWork(reason: "mode switch → \(mode.rawValue)")
            switch mode {
            case .segmentation:
                self.warmupSegmentationIfPossible()
                // P-3: pre-build box decoder so the first segmentation frame does
                // not pay the ANE first-compilation cost on demand.  Runs on
                // decoderQueue asynchronously; `boxDecoderPrebuilt` prevents double
                // builds.  If `.segmentation` first-frame decode races ahead, the
                // existing on-demand path remains correct (just slower that once).
                self.scheduleBoxDecoderPrebuild(backend: self.backend)
            case .tapToSegment:
                // Clear the previous mode's mask but keep any cached embedding —
                // then warm up encoder+decoder (ANE cold start ≈1s+) so the first
                // tap hits the decode-only fast path instead of a cold encode.
                //
                // Requirement C: this is the whole point of warming on mode
                // ENTRY rather than on first tap — the 1283–8605 ms cold-start
                // window moves off the user's first tap, where no timeout value
                // could have covered it honestly.
                self.temporal.invalidateMask()
                DispatchQueue.main.async {
                    self.maskImage = nil
                    self.maskOutlines = nil
                }
                self.warmupSegmentationIfPossible()
                // P-3 (speculative): pre-build box decoder while the user is in
                // tapToSegment so it is already compiled if they switch to
                // segmentation next.  tapToSegment never calls the box decoder
                // itself; this is purely a background investment.
                self.scheduleBoxDecoderPrebuild(backend: self.backend)
            case .detectionOnly:
                // Clear all caches when leaving segmentation modes to free memory.
                self.stateLock.lock()
                self.embeddingCache = nil
                self.stateLock.unlock()
                self.temporal.invalidateMask()
                DispatchQueue.main.async {
                    self.maskImage = nil
                    self.maskOutlines = nil
                }
            }
        }
    }

    /// Encoder for the current work item. **encoderQueue only.**
    private func encoderForQueue(computeUnits: MLComputeUnits) -> SAMEncoder? {
        if let encoder = samEncoder, samEncoderUnits == computeUnits { return encoder }
        // D-16 duplicate-model-loading investigation: distinguish first load from
        // rebuild so the log makes the cause unambiguous.  Rebuilds happen when
        // setBackend / setEncoderResolution drops the instance on encoderQueue;
        // first-load is the ANE cold-start.  Both are correct behaviour; seeing
        // two "rebuild" lines in a single session with no settings change would
        // indicate a bug — this log makes that detectable.
        let loadReason = (samEncoder == nil) ? "first load" : "rebuild (units changed)"
        diagLog("[SAM] encoder: loading model (units=\(computeUnits.rawValue), reason=\(loadReason))")
        samEncoder = SAMEncoder(computeUnits: computeUnits)
        samEncoderUnits = (samEncoder == nil) ? nil : computeUnits
        return samEncoder
    }

    /// Decoder for the current work item. **decoderQueue only.**
    private func decoderForQueue(computeUnits: MLComputeUnits) -> SAMDecoder? {
        if let decoder = samDecoder, samDecoderUnits == computeUnits { return decoder }
        samDecoder = SAMDecoder(computeUnits: computeUnits)
        samDecoderUnits = (samDecoder == nil) ? nil : computeUnits
        // New SAMDecoder instance → its boxModel is nil again.
        boxDecoderPrebuilt = false
        return samDecoder
    }

    /// Pre-build the SAM box decoder asynchronously so the first `.segmentation`
    /// frame does not pay the ANE first-compilation cost on demand.
    ///
    /// Triggered when entering `.tapToSegment` (speculative — the user may switch
    /// to `.segmentation` next) and when entering `.segmentation` directly.
    /// A dummy all-zero embedding is used to drive the first call to
    /// `SAMDecoder.decode(embedding:prompt:)`, which is the only public entry point
    /// that materialises the lazy box MLModel.  The inference result is discarded.
    ///
    /// **Safe to call from any queue.** The actual work runs on `decoderQueue`.
    /// `boxDecoderPrebuilt` (decoderQueue-only) ensures at most one build per
    /// decoder lifetime — `decoderForQueue` resets it whenever a new SAMDecoder
    /// is created.
    private func scheduleBoxDecoderPrebuild(backend: InferenceBackend) {
        decoderQueue.async { [weak self] in
            guard let self = self else { return }
            guard !self.boxDecoderPrebuilt else { return }
            guard !self.isAppBackgrounded else { return }
            guard let decoder = self.decoderForQueue(computeUnits: backend.computeUnits) else { return }
            // A zero-filled embedding of the expected shape is enough to reach the
            // lazy MLModel constructor inside boxModelForDecode().  The inference
            // result produced by an all-zero input is garbage and is intentionally
            // discarded.  The embedding shape is [1, 256, 64, 64] Float32.
            guard let dummyEmb = try? MLMultiArray(shape: [1, 256, 64, 64], dataType: .float32),
                  let prompt = PromptBuilder.buildBoxPrompt(x1: 256, y1: 256,
                                                            x2: 768, y2: 512,
                                                            origW: 1280, origH: 720,
                                                            inputSize: 1024) else { return }
            _ = decoder.decode(embedding: dummyEmb, prompt: prompt)
            self.boxDecoderPrebuilt = true
            diagLog("[SAM] box decoder pre-built (async, dummy decode)")
        }
    }

    /// Encode `buffer` and release nothing — the caller owns the isEncoding slot.
    /// **encoderQueue only.**  Numerically broken embeddings are rejected inside
    /// SAMEncoder; `tag` identifies the call site in the log.
    private func encodeChecked(buffer: CVPixelBuffer,
                               computeUnits: MLComputeUnits,
                               tag: String) -> MLMultiArray? {
        guard let encoder = encoderForQueue(computeUnits: computeUnits) else { return nil }
        guard let embedding = encoder.encode(pixelBuffer: buffer) else {
            if encoder.lastEncodeRejectedAsGarbage {
                faultLog("[SAM] \(tag): embedding failed the numeric sentinel (units=\(computeUnits.rawValue))")
            }
            return nil
        }
        return embedding
    }

    private func warmupSegmentationIfPossible() {
        // §2.1 fix: dispatch to videoQueue first to take safe snapshots of
        // latestCameraBuffer and lastLetterbox (both written on videoQueue).
        // Previously this ran on sessionQueue, causing cross-queue reads without locks.
        videoQueue.async { [weak self] in
            guard let self = self else { return }
            guard let cameraBuffer = self.latestCameraBuffer else {
                // No frame has arrived yet (typical when .tapToSegment is the
                // launch mode).  Requirement C: do NOT drop the warmup — arm it
                // so the first frame triggers it, otherwise the cold start
                // silently moves back onto the user's first tap.
                self.warmupPending = true
                diagLog("[SAM] warmup deferred — no camera frame yet, armed for next frame")
                return
            }
            let capturedLetterbox = self.lastLetterbox                         // snapshot on videoQueue
            let capturedBackend   = self.backend
            // Snapshot on videoQueue alongside the other two — the decoder
            // warmup runs on decoderQueue and must not read `currentMode` there.
            let capturedMode      = self.currentMode

            // D-2: warm the DECODER first and unconditionally.  It used to live
            // inside the encode-success branch below, so whoever lost the race
            // for `isEncoding` also lost decoder warmup — and the loser was
            // usually warmup itself (see the guard below).  decoderQueue and
            // encoderQueue are independent, so this needs no queue of its own
            // (§10.4 A: no third queue) and cannot be starved by the encoder.
            self.warmupDecoderIfPossible(letterbox: capturedLetterbox,
                                         backend: capturedBackend,
                                         origin: "warmup",
                                         mode: capturedMode)

            // Claim the encoder slot so the main pipeline knows encoding is in progress.
            self.stateLock.lock()
            if self.isEncoding {
                // D-1 made this exit observable (it used to be a bare `return`,
                // which silently dropped the warmup and put the cold start back
                // on the user's first tap).  D-14 fixes what D-1's retry did to
                // the cold start itself: re-arming `warmupPending` meant asking
                // the same question on every camera frame for as long as the
                // other encode ran — 40+ futile attempts across the 7.9 s ANE
                // cold start, two log lines each, and not one of them could have
                // succeeded, because the only thing that changes the answer is
                // that encode finishing.
                //
                // So: wait for the encode instead of polling for it — but only
                // when the encode in flight is one whose completion actually
                // reports in (`encodeSlotDidFinish`) with an embedding to show
                // for it.  Registering the flag here, while still holding the
                // lock that observed `isEncoding`, is what makes the wakeup
                // race-free.
                let owner = self.encodeSlotOwner
                if let owner = owner, owner.canSatisfyDeferredWarmup {
                    let isFirstWait = !self.warmupWaitingOnEncode
                    self.warmupWaitingOnEncode = true
                    if !isFirstWait { self.warmupDeferralsFolded += 1 }
                    self.stateLock.unlock()
                    if isFirstWait {
                        diagLog("[SAM] warmup deferred — \(owner.rawValue) encode in flight; awaiting its completion (one-shot, no per-frame retry)")
                    }
                } else {
                    // Slot busy but nobody claims it (or the owner cannot wake
                    // us).  There is no completion to wait for, so the old
                    // next-frame re-arm is still the only honest option.
                    self.stateLock.unlock()
                    self.warmupPending = true
                    diagLog("[SAM] warmup deferred — encoder slot busy with no wakeable owner (\(owner?.rawValue ?? "unknown")), re-armed for next frame")
                }
                return
            }
            // The encode itself is only worth doing if the cache is not already
            // warm.  Without this, losing the race to the background refresh
            // meant re-arming forever and then paying a redundant ~650 ms
            // encode on top of the one that just finished.
            let nowMs = PerfLogger.nowMs()
            let cacheAgeMs = self.embeddingCache.map { nowMs - $0.timestampMs }
            if let age = cacheAgeMs, age <= 5000 {
                self.stateLock.unlock()
                diagLog(String(format: "[SAM] warmup encode skipped — embedding already fresh (%.0f ms old)", age))
                return
            }
            self.isEncoding = true
            self.encodeSlotOwner = .warmup
            self.stateLock.unlock()
            self.setWarmingUp(true)

            self.encoderQueue.async { [weak self] in
                guard let self = self else { return }

                // Guard: iOS aborts GPU/ANE work when the app is in the background,
                // causing "Insufficient Permission" CoreML errors.  Skip inference and
                // release the encoding slot so the pipeline retries when foregrounded.
                guard !self.isAppBackgrounded else {
                    faultLog("[SAM] warmup skipped: app is in background")
                    self.releaseEncodeSlot()
                    self.setWarmingUp(false)
                    self.encodeSlotDidFinish(originTag: "warmup-aborted")
                    return
                }

                let t0 = PerfLogger.nowMs()
                if let embedding = self.encodeChecked(buffer: cameraBuffer,
                                                      computeUnits: capturedBackend.computeUnits,
                                                      tag: "warmup") {
                    let t1 = PerfLogger.nowMs()
                    self.stateLock.lock()
                    self.embeddingCache = (embedding: embedding, timestampMs: t1)
                    // RE-1 (§18.1.4): a *newly computed* embedding lands ⇒ every
                    // instance becomes eligible for one re-anchor decode again.
                    self.embeddingGeneration &+= 1
                    self.isEncoding = false
                    self.encodeSlotOwner = nil
                    self.stateLock.unlock()
                    perfLog(String(format: "SAM encoder warmup latency: %.2f ms", t1 - t0))
                    self.setWarmingUp(false)
                    // Taps that arrived during warmup are parked — decode them now.
                    self.encodeSlotDidFinish(originTag: "warmup")

                    // Second decoder-warmup attempt: the pre-encode one above
                    // may have had no embedding to decode with.  Idempotent —
                    // `decoderWarmupDecodeDone` makes the decode run once.
                    self.warmupDecoderIfPossible(letterbox: capturedLetterbox,
                                                 backend: capturedBackend,
                                                 origin: "warmup-encoded",
                                                 mode: capturedMode)
                } else {
                    // encode() returned nil — CoreML error already logged inside SAMEncoder.
                    // Common cause: app went to background mid-inference (GPU abort).
                    // The isEncoding flag is released so the pipeline can retry on next frame.
                    faultLog("[SAM] warmup encode returned nil — releasing encoder slot")
                    self.releaseEncodeSlot()
                    self.setWarmingUp(false)
                    // Requirement C: a tap parked behind a failed warmup must not
                    // die silently — this is exactly the Day 4 lost-tap (G-2).
                    self.encodeSlotDidFinish(originTag: "warmup-failed")
                }
            }
        }
    }

    /// Warm the prompt decoder **independently of the encoder slot** (D-2).
    ///
    /// The expensive half is constructing `SAMDecoder` — two MLModel loads plus
    /// ANE/GPU compilation, > 1.5 s cold.  That half needs nothing but
    /// decoderQueue, so it runs on every call regardless of what the encoder is
    /// doing; before Day 5 it was buried in the encode-success branch and
    /// therefore never ran when warmup lost the `isEncoding` race, which is how
    /// the user's first tap ended up being the call site that built the decoder.
    ///
    /// The second half (one throwaway decode, which is what actually exercises
    /// the compiled graph) needs an embedding, so it is attempted on every call
    /// and latched by `decoderWarmupDecodeDone` once it succeeds.
    ///
    /// `mode` selects **which decoder is rehearsed**, and therefore which one is
    /// built at all now that `SAMDecoder`'s box model is lazy:
    ///   - `.tapToSegment` → point prompt → multimask point decoder (the only
    ///     model that mode ever calls).  Rehearsing a box prompt here would
    ///     construct the box decoder purely to warm a model that is never used,
    ///     re-introducing the load this change exists to remove.
    ///   - anything else (Phase 2 `.segmentation`) → box prompt → box decoder on
    ///     the caller's compute units, i.e. ANE, exactly as before.
    /// In the single-mask fallback (`isMultimask == false`) both prompts land on
    /// the same model, so either rehearsal warms the right thing.
    ///
    /// **decoderQueue only** — the model may only be created and called there.
    private func warmupDecoderIfPossible(letterbox: LetterboxInfo?,
                                         backend: InferenceBackend,
                                         origin: String,
                                         mode: AppMode) {
        decoderQueue.async { [weak self] in
            guard let self = self else { return }
            self.stateLock.lock()
            let alreadyDecoded = self.decoderWarmupDecodeDone
            self.stateLock.unlock()
            if alreadyDecoded { return }

            guard !self.isAppBackgrounded else {
                faultLog("[SAM] decoder warmup skipped (\(origin)): app is in background")
                return
            }
            // Constructing the decoder IS the cold start being moved off the
            // first tap; do it even if no embedding exists yet.
            guard let decoder = self.decoderForQueue(computeUnits: backend.computeUnits) else {
                faultLog("[SAM] decoder warmup (\(origin)): decoder unavailable")
                return
            }

            self.stateLock.lock()
            let embedding = self.embeddingCache?.embedding
            self.stateLock.unlock()

            guard let embedding = embedding, let lb = letterbox else {
                // Not a failure: the model is loaded, the rehearsal decode just
                // has nothing to run on yet.  The next encode to publish an
                // embedding calls back in ("warmup-encoded" / "refresh-encoded")
                // and the rehearsal runs then — exactly once, latched by
                // `decoderWarmupDecodeDone`.
                //
                // D-14: this is a *state* ("no embedding exists yet"), not a
                // per-call event, and it used to be printed on every call — tens
                // of identical lines across the cold start.  Print it on the
                // first deferral of an arm cycle and fold the rest into a count
                // reported when the rehearsal finally runs.
                self.stateLock.lock()
                let isFirstDeferral = !self.decoderWarmupDeferralLogged
                self.decoderWarmupDeferralLogged = true
                if !isFirstDeferral { self.decoderWarmupDeferralsFolded += 1 }
                self.stateLock.unlock()
                if isFirstDeferral {
                    diagLog("[SAM] decoder warmup (\(origin)): model ready, decode deferred until an embedding exists (embedding=\(embedding == nil ? "none" : "ok") letterbox=\(letterbox == nil ? "none" : "ok"))")
                }
                return
            }

            let t0 = PerfLogger.nowMs()
            if mode == .tapToSegment {
                // Centre of the canonical frame — a plausible foreground tap.
                guard let prompt = PointPromptBuilder.buildPointPrompt(
                        canonicalPoint: CGPoint(x: CGFloat(lb.origW) * 0.5,
                                                y: CGFloat(lb.origH) * 0.5),
                        origSize: CGSize(width: CGFloat(lb.origW), height: CGFloat(lb.origH)),
                        inputSize: 1024) else {
                    diagLog("[SAM] decoder warmup (\(origin)): point prompt build failed, decode deferred")
                    return
                }
                _ = decoder.decode(embedding: embedding, point: prompt,
                                   logTag: "[SAM][warmup]")
            } else {
                guard let prompt = PromptBuilder.buildBoxPrompt(
                        x1: 0.25 * lb.origW, y1: 0.25 * lb.origH,
                        x2: 0.75 * lb.origW, y2: 0.75 * lb.origH,
                        origW: lb.origW, origH: lb.origH, inputSize: 1024) else {
                    diagLog("[SAM] decoder warmup (\(origin)): box prompt build failed, decode deferred")
                    return
                }
                _ = decoder.decode(embedding: embedding, prompt: prompt)
            }
            perfLog(String(format: "SAM decoder warmup latency: %.2f ms (%@, %@ prompt)",
                           PerfLogger.nowMs() - t0, origin,
                           mode == .tapToSegment ? "point" : "box"))
            self.stateLock.lock()
            self.decoderWarmupDecodeDone = true
            let folded = self.decoderWarmupDeferralsFolded
            self.decoderWarmupDeferralsFolded = 0
            self.decoderWarmupDeferralLogged = false
            self.stateLock.unlock()
            if folded > 0 {
                diagLog("[SAM] decoder warmup: \(folded) further deferral(s) folded in while waiting for the first embedding")
            }
        }
    }

    /// Publish the cold-start state so the UI can say "initialising" rather than
    /// let a tap wait on an invisible 1.3–8.6 s encode (Architect §10.4 C).
    private func setWarmingUp(_ value: Bool) {
        DispatchQueue.main.async { [weak self] in self?.samWarmingUp = value }
    }

    /// Rebuild `tapAnchorMarkers` from the current instance pool.
    ///
    /// MUST be called on the main thread (it writes @Published directly).
    /// Only instances that have a placed mask (maskAlpha != nil) and a stored
    /// viewPoint get a marker; pending instances show the loading indicator
    /// instead, so they are excluded here.
    private func publishAnchorMarkersOnMain() {
        assert(Thread.isMainThread, "publishAnchorMarkersOnMain must run on main thread")
        let instances = tapInstances.drawableInstances()
        tapAnchorMarkers = instances.compactMap { inst in
            guard let vp = inst.viewPoint else { return nil }
            // B-37: `revisitOriginPinTags[id]` is `String??` — the outer
            // optional answers "is this a revisit product?", the inner one
            // carries the origin Pin's tag (nil = untagged Pin).
            let revisitTag = revisitOriginPinTags[inst.id]
            return TapAnchorMarker(
                id: inst.id,
                viewPoint: vp,
                slotIndex: inst.slotIndex(in: TapInstanceManager.palette),
                requestGen: inst.requestGen,
                isPrimary: inst.isPrimary,
                isPinned: pinnedInstanceIDs.contains(inst.id),
                tag: pinnedInstanceTags[inst.id],
                isRevisitOrigin: revisitTag != nil,
                revisitPinTag: revisitTag ?? nil)
        }
    }

    /// Phase 4B Day 5 — mark an instance's anchor marker as pinned (📌 style)
    /// after its Pin has been saved.  Called from the UI layer (which does the
    /// actual `PinStore.create`) on success; this method itself never touches
    /// PinStore.  Main-thread only, like every other `tapAnchorMarkers` writer.
    /// - Parameter tag: the tag the Pin was saved with, if any and non-empty
    ///   (Day 6 — shown beside the marker; a plain String carries no
    ///   Persistence symbol, so PIN-3 is unaffected).
    func markInstancePinned(id: UUID, tag: String? = nil) {
        assert(Thread.isMainThread, "markInstancePinned must run on main thread")
        pinnedInstanceIDs.insert(id)
        if let tag = tag, !tag.isEmpty {
            pinnedInstanceTags[id] = tag
        }
        // §23.1.6-2: saving a revisit product creates a NEW Pin record, and at
        // that moment 📌 becomes TRUE for this mask — the ↻ provenance
        // decoration is replaced by 📌, not stacked under it.
        revisitOriginPinTags.removeValue(forKey: id)
        publishAnchorMarkersOnMain()
    }

    /// B-37 (§23.1.4) — mark an instance as a revisit product: its mask was
    /// decoded from a Pin's stored coordinate.  Called from the revisit flow's
    /// placed-observation callback (`PinInterfaces.swift`) on the main thread;
    /// this method itself never touches PinStore (PIN-3).  `pinTag` is the
    /// origin Pin's tag (nil = untagged Pin) — displayed only inside the
    /// `From Pin "…"` phrase (PIN-7).
    func markInstanceRevisitOrigin(id: UUID, pinTag: String?) {
        assert(Thread.isMainThread, "markInstanceRevisitOrigin must run on main thread")
        revisitOriginPinTags[id] = pinTag
        publishAnchorMarkersOnMain()
    }

    /// B-37 route A1 (§23.1.4 二) — project a Canonical pixel point into the
    /// preview view's coordinate space, so a revisit can hand
    /// `handleTap(canonicalPoint:viewPoint:)` the `viewPoint` it previously
    /// could not supply (M-23.1: `viewPoint: nil` was the cost of PIN-5's
    /// "reuse the existing entry point" when no forward mapping existed).
    ///
    /// The math lives in `FrameGeometry.projectCanonicalPoint` — the single
    /// coordinate-chain source (§2) — as the algebraic inverse of
    /// `invertViewPoint`, reading the same geometry snapshot
    /// (`currentFrameGeometry()`), not a second transform path.  Gated by P-9.
    ///
    /// Main-thread only: `layerPointConverted(fromCaptureDevicePoint:)` is a
    /// layer-geometry query.  Reading the weak `previewLayer` here matches the
    /// existing main-thread read in `setupRotationCoordinator`'s preview KVO
    /// block.  Returns nil before the preview layer or first geometry snapshot
    /// exists — the caller then degrades to the pre-B-37 markerless behaviour.
    func viewPoint(forCanonicalPoint canonical: CGPoint) -> CGPoint? {
        assert(Thread.isMainThread, "viewPoint(forCanonicalPoint:) must run on main thread")
        guard let layer = previewLayer, let geo = currentFrameGeometry() else { return nil }
        return geo.projectCanonicalPoint(canonical, previewLayer: layer)
    }

    /// **TEST-ONLY** — run P-9's canonical↔view round trip (§23.2.8) against the
    /// current live geometry and print the result.  Triggered by hand from the
    /// Debug Options panel; ⛔ never called from any capture / inference path.
    ///
    /// Reads the same two things `viewPoint(forCanonicalPoint:)` reads — the
    /// weak `previewLayer` and `currentFrameGeometry()` — so the measurement is
    /// taken against the geometry the production revisit path would actually
    /// use, not a synthesized one.  When either is missing it says so out loud:
    /// a self-check that can silently no-op is worse than no self-check.
    func runCanonicalRoundTripSelfCheck() {
        assert(Thread.isMainThread, "runCanonicalRoundTripSelfCheck must run on main thread")
        guard let layer = previewLayer else {
            faultLog("[P9] unavailable reason=noPreviewLayer — start the camera first")
            return
        }
        guard let geo = currentFrameGeometry() else {
            faultLog("[P9] unavailable reason=noGeometrySnapshot — no frame has been processed yet")
            return
        }
        geo.runRoundTripSelfCheck(previewLayer: layer)
    }

    /// Recomposite the entire instance pool after a `promoteToPrimary` call that
    /// changed isPrimary flags without touching any mask data.  Dispatches to
    /// decoderQueue (same as the normal decode path) so the blend never blocks
    /// the main thread.  `gen` is used only for log attribution.
    private func recompositeForPromote(letterbox lb: LetterboxInfo, gen: Int) {
        decoderQueue.async { [weak self] in
            guard let self = self else { return }
            let drawable = self.tapInstances.drawableInstances()
            guard !drawable.isEmpty else { return }

            let layers = drawable.map { inst in
                MaskRenderer.MaskLayer(
                    alpha: inst.maskAlpha ?? [],
                    color: inst.color,
                    opacity: inst.isPrimary ? TapInstanceManager.primaryOpacity
                                           : TapInstanceManager.secondaryOpacity)
            }
            let composed = self.maskRenderer.compositeLayers(layers,
                                                             origW: Int(lb.origW),
                                                             origH: Int(lb.origH),
                                                             tapIndex: gen)

            let outlineSet: MaskOutlineSet?
            if MaskOutlineStyle.isEnabled {
                let outlines = drawable.compactMap { inst -> MaskOutline? in
                    guard let alpha = inst.maskAlpha else { return nil }
                    let polys = self.maskRenderer.traceOutline(alpha: alpha,
                                                               origW: Int(lb.origW),
                                                               origH: Int(lb.origH))
                    guard !polys.isEmpty else { return nil }
                    return MaskOutline(polygons: polys, isPrimary: inst.isPrimary)
                }
                outlineSet = MaskOutlineSet(
                    canvasSize: CGSize(width: CGFloat(lb.origW), height: CGFloat(lb.origH)),
                    outlines: outlines)
            } else {
                outlineSet = nil
            }

            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.maskImage = composed
                self.maskOutlines = outlineSet
                self.publishAnchorMarkersOnMain()
            }
        }
    }

    func stop() {
        CameraMotionGate.stop()
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    // MARK: - Phase 3 Tap-to-Segment (Day 5: fast path off videoQueue + instance pool)

    /// Snapshot of the current canonical frame geometry for TouchHandler.
    /// Canonical = original camera frame in display orientation (origW/origH already
    /// carry device rotation, matching the mask pipeline). Returns nil until the
    /// first letterbox pass has published a snapshot.
    ///
    /// Day 5: reads the `stateLock` mirror instead of the videoQueue-owned
    /// `lastLetterbox` / `currentPosition` / `lastRotationAngle`.  This is called
    /// from the gesture thread, so the old direct reads were three unsynchronised
    /// cross-queue reads; Requirement A makes that path hotter, not colder, so
    /// they had to be closed rather than left as-is.
    func currentFrameGeometry() -> FrameGeometry? {
        stateLock.lock()
        let snap = tapGeometryMirror
        stateLock.unlock()
        guard let snap = snap else { return nil }
        return Self.frameGeometry(from: snap)
    }

    /// The single construction point for `FrameGeometry`.  Shared by the tap
    /// path (`currentFrameGeometry`) and the re-anchor path so both compare and
    /// invert against bit-identical geometry (§16.3.1).
    private static func frameGeometry(from snap: TapGeometrySnapshot) -> FrameGeometry {
        FrameGeometry(origW: CGFloat(snap.letterbox.origW),
                      origH: CGFloat(snap.letterbox.origH),
                      mirrored: snap.mirrored,
                      rotation: snap.rotation,
                      letterboxOffset: CGPoint(x: CGFloat(snap.letterbox.padX),
                                               y: CGFloat(snap.letterbox.padY)),
                      scale: CGFloat(snap.letterbox.scale))
    }

    /// Publish the geometry the tap fast path reads.  **videoQueue only** —
    /// called once per letterbox pass, right where `lastLetterbox` is written.
    private func publishTapGeometry(_ info: LetterboxInfo) {
        let snap = TapGeometrySnapshot(letterbox: info,
                                       rotation: lastRotationAngle,
                                       mirrored: (currentPosition == .front))
        stateLock.lock()
        tapGeometryMirror = snap
        stateLock.unlock()
    }

    // MARK: Tap intake

    /// Called from TouchHandler (main thread) when the user taps in .tapToSegment mode.
    ///
    /// **Requirement A (architect_output §10.4 A).**  The reuse decision is taken
    /// right here, under `stateLock`, from a snapshot — it needs letterbox
    /// geometry and the cached embedding and nothing else:
    ///
    ///   fast path (cached embedding usable) → decoderQueue directly
    ///   slow path (needs a fresh embedding) → videoQueue (for the pixel buffer)
    ///
    /// The §4.2 trigger semantics are unchanged (TTL + geometry decide reuse,
    /// `isEncoding` is still the single shared encoder slot, encode still
    /// precedes decode).  Only *which queue evaluates the decision* moved: the
    /// old code queued the whole decision behind a 400–670 ms YOLO frame on
    /// videoQueue, which was ~90 % of the measured 620 ms median tap→mask.
    ///
    /// - Returns: the tap sequence number, or 0 when the tap was not accepted.
    @discardableResult
    func handleTap(canonicalPoint: CGPoint, viewPoint: CGPoint? = nil) -> Int {
        guard currentMode == .tapToSegment else { return 0 }

        // True e2e timing starts here (tap acceptance): queue wait is part of
        // what the user perceives.
        let tapStartMs = PerfLogger.nowMs()

        stateLock.lock()
        tapGeneration += 1
        let myGen = tapGeneration
        lastTapAcceptedMs = tapStartMs
        let geo             = tapGeometryMirror
        let capturedBackend = backendMirror
        let entry           = embeddingCache.map {
            EmbeddingEntry(embedding: $0.embedding, timestampMs: $0.timestampMs)
        }
        let cacheAgeMs      = embeddingCache.map { tapStartMs - $0.timestampMs }
        // NOTE: `isEncoding` is deliberately NOT snapshotted here.  §4.2's
        // busy/free branch is re-evaluated on videoQueue at dispatch time (see
        // the slow path below) with a fresher value; the fast path never reads
        // it at all, which is the P2 fix Day 4 landed — a cached embedding can
        // be decoded while an unrelated encode is in flight.
        stateLock.unlock()
        // D-7' T2 (debug_report §30.1) — stateLock released.  I1 = T2 − T1
        // isolates lock contention.  Pure telemetry: never read for routing.
        let lockDoneMs = PerfLogger.nowMs()

        guard let snap = geo else {
            // No letterbox yet ⇒ no coordinate mapping ⇒ no prompt is possible.
            failTap(gen: myGen, viewPoint: viewPoint,
                    message: "camera not ready — tap again in a moment")
            return myGen
        }
        let lb = snap.letterbox

        // Geometry change ⇒ every cached instance mask belongs to a dead
        // coordinate space, and the cached embedding no longer matches the tap.
        let geoSig = TemporalManager.GeometrySignature(
            origW: lb.origW, origH: lb.origH,
            scale: lb.scale, padX: lb.padX, padY: lb.padY,
            rotation: snap.rotation,
            mirrored: snap.mirrored,
            inputSize: lb.inputSize)
        let geometryChanged = temporal.tapGeometryChanged(geoSig)
        if geometryChanged && !tapInstances.isEmpty {
            diagLog("[TAP#\(myGen)] geometry changed → clearing \(tapInstances.count) stale instance(s)")
            discardAllTapWork(reason: "geometry change")
        }

        let ttlValid = temporal.isEmbeddingValid(entry: entry, nowMs: tapStartMs)
        let canReuse = ttlValid && !geometryChanged

        // Day 6 — tap inside existing mask → promote to primary, no re-decode.
        // Runs BEFORE addInstance so the pool is not polluted with a pending
        // instance that never gets a mask.  Only checked when geometry is stable
        // (after a geometry change, discardAllTapWork already cleared the pool,
        // so the snapshot would be empty and the loop is a no-op anyway).
        if !geometryChanged, let hit = hitTestExistingInstance(canonicalPoint: canonicalPoint, letterbox: lb) {
            // Tap falls inside this instance's rendered mask region.
            if tapInstances.promoteToPrimary(id: hit.id) {
                diagLog("[TAP#\(myGen)] tap inside existing mask (gen #\(hit.requestGen)) — promote to primary, no re-decode")
                recompositeForPromote(letterbox: lb, gen: myGen)
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.lastTapViewPoint = viewPoint
                    self.lastTapIndex = myGen
                    // B-40 ② (§23.1.9) — the promote branch returns `gen > 0`
                    // but used to skip the reset below, so an earlier revisit's
                    // flag leaked onto ordinary promote taps (false "Re-
                    // segmenting" copy).  Every gen>0 path now resets.
                    self.lastTapWasRevisit = false
                    // B-40 ① (§23.1.9) — third pure-observation hook, same
                    // discipline as `onTapMaskPlaced`/`onTapFailed`: fired in
                    // this pre-existing main-thread block, reads only values
                    // already computed above (`myGen`, `hit.id`), adds no
                    // decision branch and changes no promote behaviour.
                    self.onTapPromoted?(myGen, hit.id)
                }
                return myGen
            }
        }

        // Pool bookkeeping (§3.2): new instance, primary, FIFO-evict the oldest.
        let (instance, evicted) = tapInstances.addInstance(point: canonicalPoint,
                                                           viewPoint: viewPoint,
                                                           requestGen: myGen)
        if let evicted = evicted {
            diagLog("[TAP#\(myGen)] pool full → FIFO evicted oldest instance")
            cancelRequests(forInstance: evicted, reason: "FIFO eviction")
            // Phase 4B Day 5 (§22.2.4): the 📌 decoration is instance-scoped —
            // an evicted instance's marker is already gone from
            // `tapAnchorMarkers` (it is rebuilt from `drawableInstances()`),
            // so drop its id here too rather than let it sit forever in a set
            // nothing will ever look up again.
            DispatchQueue.main.async { [weak self] in
                self?.pinnedInstanceIDs.remove(evicted)
                self?.pinnedInstanceTags.removeValue(forKey: evicted)
                // B-37: the ↻ decoration is instance-scoped exactly like 📌.
                self?.revisitOriginPinTags.removeValue(forKey: evicted)
            }
        }
        beginTapRequest(gen: myGen, instanceID: instance.id)

        // Publish tap anchor + processing state immediately for UI feedback.
        DispatchQueue.main.async { [weak self] in
            self?.lastTapCanonicalPoint = canonicalPoint
            self?.lastTapViewPoint = viewPoint
            self?.lastTapIndex = myGen
            self?.tapProcessing = true
            self?.tapFailure = nil
            // Phase 4B Day 6 (§19.4.4 R-A) — every ordinary tap resets this;
            // `handleTap(fromPin:)` sets it back to true immediately after
            // this call returns, synchronously, so the flag is correct by the
            // time SwiftUI next reads it.
            self?.lastTapWasRevisit = false
        }

        let ageDesc = cacheAgeMs.map { String(format: "%.0f", $0) } ?? "n/a"

        if canReuse, let emb = entry?.embedding {
            // ── FAST PATH — decode only, straight to decoderQueue ─────────────
            // R2: `cacheAge` is logged (and only logged) so "the embedding was
            // stale" can be confirmed or ruled out as an over-segmentation
            // contributor.  TTL behaviour itself is untouched — Day 6 item.
            diagLog(String(format: "[TAP#%d] reuse cached embedding (ttlValid=%@ geoChanged=%@ cacheAge=%@ms) → decode point=(%.1f,%.1f) [fast]",
                         myGen, ttlValid ? "Y" : "N", geometryChanged ? "Y" : "N",
                         ageDesc, canonicalPoint.x, canonicalPoint.y))
            scheduleTapTimeout(gen: myGen, seconds: Self.fastPathTimeoutSec, label: "fast")
            tapDecodeWithPoint(embedding: emb,
                               instanceID: instance.id,
                               canonicalPoint: canonicalPoint,
                               letterbox: lb,
                               backend: capturedBackend,
                               reusedEmbedding: true,
                               path: .fast,
                               cacheAgeMs: cacheAgeMs,
                               tapStartMs: tapStartMs,
                               lockDoneMs: lockDoneMs,
                               gen: myGen)
            return myGen
        }

        // ── SLOW PATH — a fresh embedding is needed, which needs the newest
        // camera pixel buffer, which lives on videoQueue.  Unchanged from Day 4.
        let reason = geometryChanged ? "geometry change" : (ttlValid ? "no cache" : "ttl expired")
        // Item 6 — log the trigger source for the embedding re-encode so cache
        // efficiency can be measured (geometry_change vs ttl_expired vs manual_tap).
        let cacheLogKey: String
        switch true {
        case geometryChanged: cacheLogKey = "geometry_change"
        case !ttlValid:       cacheLogKey = "ttl_expired"
        default:              cacheLogKey = "manual_tap"
        }
        diagLog("[CACHE] re-encode reason: \(cacheLogKey) (tap #\(myGen))")
        scheduleTapTimeout(gen: myGen, seconds: Self.slowPathTimeoutSec, label: "slow")
        videoQueue.async { [weak self] in
            guard let self = self else { return }
            guard self.tapInstances.isRequestCurrent(id: instance.id, requestGen: myGen) else {
                diagLog("[TAP#\(myGen)] instance retired before processing — dropped")
                self.endTapRequest(gen: myGen)
                return
            }
            guard let buffer = self.latestCameraBuffer else {
                self.failTap(gen: myGen, viewPoint: viewPoint,
                             message: "no camera frame — tap again")
                self.tapInstances.removeInstance(id: instance.id)
                return
            }

            // Re-check the encoder slot at dispatch time (§4.2): it may have been
            // taken or released between the gesture-thread snapshot and here.
            self.stateLock.lock()
            let busyNow = self.isEncoding
            self.stateLock.unlock()

            if !busyNow {
                diagLog(String(format: "[TAP#%d] encode + decode (reason=%@ cacheAge=%@ms) → point=(%.1f,%.1f) [slow]",
                             myGen, reason, ageDesc, canonicalPoint.x, canonicalPoint.y))
                self.tapEncodeAndDecode(buffer: buffer,
                                        instanceID: instance.id,
                                        canonicalPoint: canonicalPoint,
                                        letterbox: lb,
                                        backend: capturedBackend,
                                        tapStartMs: tapStartMs,
                                        lockDoneMs: lockDoneMs,
                                        gen: myGen)
            } else {
                // Encoder busy and no usable cached embedding — park the tap.
                // It is decoded the moment the in-flight encode caches its
                // embedding (drainPendingTaps).  Requirement C: a parked tap is
                // never dropped in silence; every drain path either decodes it
                // or reports a visible failure.
                diagLog("[TAP#\(myGen)] encoder busy, no cache — tap parked until encode completes [slow]")
                self.parkTap(id: instance.id, point: canonicalPoint,
                             gen: myGen, startMs: tapStartMs,
                             lockDoneMs: lockDoneMs)
            }
        }
        return myGen
    }

    // MARK: Phase 4B Day 5 — long-press hit-test (Pin creation channel, §22.2)

    /// The canonical→256-space + `alpha[idx] > 0` hit test §3.2's promote path
    /// already used inline.  Extracted so `handleTap`'s promote branch and
    /// `handleLongPress`'s Pin-creation branch share one implementation rather
    /// than two copies of the same coordinate arithmetic (§22.2.2: "Builder
    /// 应把这段映射抽成一个私有 helper 供两处调用").
    ///
    /// Single lock acquisition: `tapInstances.snapshot()` takes
    /// `TapInstanceManager`'s lock once and returns value-type copies: the
    /// `TapInstance` this returns (if any) is already a frozen snapshot, which
    /// is what `handleLongPress` needs for PIN-6.
    private func hitTestExistingInstance(canonicalPoint: CGPoint,
                                         letterbox lb: LetterboxInfo) -> TapInstance? {
        let ps  = CGFloat(SAMConfiguration.pointPromptSpace)
        let msX = ps / max(CGFloat(lb.origW), CGFloat(lb.origH))
        let tx  = Int(((canonicalPoint.x * msX + (ps - CGFloat(lb.origW) * msX) * 0.5)
                       * 256.0 / ps).rounded())
        let ty  = Int(((canonicalPoint.y * msX + (ps - CGFloat(lb.origH) * msX) * 0.5)
                       * 256.0 / ps).rounded())
        for inst in tapInstances.snapshot() {
            guard let alpha = inst.maskAlpha else { continue }
            guard tx >= 0, tx < 256, ty >= 0, ty < 256 else { continue }
            let idx = ty * 256 + tx
            guard idx < alpha.count, alpha[idx] > 0 else { continue }
            return inst
        }
        return nil
    }

    /// Long-press entry point (§22.2.2's decision tree).  Called from
    /// `TouchHandler.onLongPress`, itself only fired after the system
    /// long-press recognizer's `minimumPressDuration` — which is why this
    /// needs no duration parameter of its own: a plain tap never reaches here,
    /// and a long-press that released early never fires the recognizer that
    /// calls this (§22.2.1 — the two gestures are mutually exclusive on the
    /// same touch by construction, not by any priority rule this file adds).
    ///
    /// **Hit an existing mask** → publish a `PinCreationDraft` for the UI to
    /// present `PinCreationSheet` with.  **Not** `promoteToPrimary`, **not** a
    /// re-decode (§22.2.2: Pin creation is orthogonal to "who is primary").
    /// **Missed every mask** → no-op; long-press is not an `addInstance`
    /// trigger (§22.2.2 — that rule is not extended here).
    ///
    /// This method — like the rest of `CameraManager.swift` — references no
    /// `Pin` / `PinStore` / `PinFactory` symbol (PIN-3 stays a whole-file
    /// invariant, not just a videoQueue/decoderQueue/encoderQueue one;
    /// `PinCreationDraft` carries only `TapInstance` + `FrameGeometry`,
    /// already-existing Interaction-layer types).
    func handleLongPress(canonicalPoint: CGPoint, viewPoint: CGPoint) {
        guard currentMode == .tapToSegment else { return }

        stateLock.lock()
        let geo = tapGeometryMirror
        stateLock.unlock()
        guard let snap = geo else { return }
        let lb = snap.letterbox

        guard let hit = hitTestExistingInstance(canonicalPoint: canonicalPoint, letterbox: lb) else {
            diagLog("[PIN] long-press outside all masks — no-op")
            return
        }

        // PIN-6 (§22.2.3): `hit` is already the single locked snapshot this
        // long-press takes of the instance — `hitTestExistingInstance` reads
        // `tapInstances.snapshot()` exactly once above.  From here on, this
        // `TapInstance` value (not `tapInstances`) is the only thing the Pin
        // creation UI may read.
        let geometry = Self.frameGeometry(from: snap)
        diagLog("[PIN] long-press hit existing mask (gen #\(hit.requestGen)) — opening PinCreationSheet, no promote, no decode")
        DispatchQueue.main.async { [weak self] in
            self?.pinCreationDraft = PinCreationDraft(instance: hit, geometry: geometry)
        }
    }

    // MARK: In-flight request bookkeeping

    private func beginTapRequest(gen: Int, instanceID: UUID) {
        stateLock.lock(); inFlightTaps[gen] = instanceID; stateLock.unlock()
    }

    /// Retire a request that completed (or was retired without user-visible
    /// failure) and refresh the loading indicator.
    private func endTapRequest(gen: Int) {
        stateLock.lock()
        inFlightTaps.removeValue(forKey: gen)
        let anyLeft = !inFlightTaps.isEmpty
        stateLock.unlock()
        DispatchQueue.main.async { [weak self] in self?.tapProcessing = anyLeft }
    }

    /// Retire every request belonging to an instance that is no longer in the
    /// pool (FIFO eviction, clearAll).  No failure banner: nothing failed, the
    /// user replaced it.
    private func cancelRequests(forInstance id: UUID, reason: String) {
        stateLock.lock()
        let gens = inFlightTaps.filter { $0.value == id }.map { $0.key }
        for g in gens { inFlightTaps.removeValue(forKey: g) }
        pendingTaps.removeAll { $0.id == id }
        let anyLeft = !inFlightTaps.isEmpty
        stateLock.unlock()
        for g in gens { diagLog("[TAP#\(g)] request retired — \(reason)") }
        if !gens.isEmpty {
            DispatchQueue.main.async { [weak self] in self?.tapProcessing = anyLeft }
        }
    }

    /// **Requirement C — failure must be visible.**  Every path that gives up on
    /// a tap goes through here instead of quietly clearing `tapProcessing`;
    /// Architect §10.4 C names silent clearing the worst failure mode, because
    /// the user reads it as "点了没反应".
    private func failTap(gen: Int, viewPoint: CGPoint? = nil, message: String) {
        faultLog("[TAP#\(gen)] \(message)")
        stateLock.lock()
        inFlightTaps.removeValue(forKey: gen)
        let anyLeft = !inFlightTaps.isEmpty
        stateLock.unlock()

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.tapProcessing = anyLeft
            self.tapFailure = TapFailure(index: gen,
                                         message: message,
                                         viewPoint: viewPoint ?? self.lastTapViewPoint)
            // B-40 ② (§23.1.9) — `handleTap`'s early-exit path ("camera not
            // ready") returns `gen > 0` without ever reaching the reset in the
            // normal publish block; this covers it (and clears a stale revisit
            // flag on any later failure, where the R-C banner takes over).
            self.lastTapWasRevisit = false
            // Phase 4B Day 6 (§19.4.6) — observation only; every failure path
            // for this tap already converges here (Requirement C), so one
            // hook site covers R-C for the revisit flow with no duplicated
            // failure logic.
            self.onTapFailed?(gen, message)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.tapFailureDisplaySec) { [weak self] in
            guard let self = self else { return }
            if self.tapFailure?.index == gen { self.tapFailure = nil }
        }
    }

    /// **Requirement C — tiered timeout.**  The old blanket 3 s was incompatible
    /// with the measured 1283–8605 ms encoder cold start: the 8.6 s case cleared
    /// the indicator first and the mask arrived to an empty UI.
    ///   fast (decode-only, ~61 ms measured) → 1.5 s
    ///   slow (encode in flight)             → 12 s, covering the 8605 ms upper bound
    /// The real cold-start fix is warmup on mode entry, not a longer timeout.
    ///
    /// **A timeout reports; it does not retire (D-3).**  Until Day 5 this also
    /// called `removeInstance`, which threw away work that was still on its way:
    /// the TAP#1 mask that arrived 220 ms after the 1.5 s deadline was decoded
    /// successfully and then discarded.  A mask that shows up 1.7 s late is
    /// strictly better than a failure that never resolves, so the deadline now
    /// only raises the visible failure signal (Architect forbids clearing the
    /// indicator with nothing in its place) and leaves the instance and any
    /// parked request alone.  A late arrival renders normally and clears the
    /// banner via `tapFailure = nil` on the publish path.  Instances are removed
    /// only by the §3.2.1 C1–C6 events.
    private func scheduleTapTimeout(gen: Int, seconds: TimeInterval, label: String) {
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { [weak self] in
            guard let self = self else { return }
            self.stateLock.lock()
            let stillInFlight = self.inFlightTaps[gen] != nil
            self.stateLock.unlock()
            guard stillInFlight else { return }                 // already resolved
            self.failTap(gen: gen,
                         message: String(format: "timed out after %.1fs (%@ path) — still working",
                                         seconds, label))
        }
    }

    // MARK: Encoder slot release

    /// Release the single encoder slot.  Always followed by
    /// `encodeSlotDidFinish` at the call site — the two are separate only
    /// because some paths do bookkeeping (stats, logging) in between.
    private func releaseEncodeSlot() {
        stateLock.lock()
        isEncoding = false
        encodeSlotOwner = nil
        stateLock.unlock()
    }

    /// The single completion hook for "an encode just ended", success or
    /// failure.  Two things wait on that event and both used to be handled by
    /// their own retry loop:
    ///
    ///   1. parked taps, drained against whatever embedding the encode produced
    ///      (pre-existing; unchanged);
    ///   2. a warmup that deferred behind this encode (D-14) — resumed here
    ///      exactly once instead of re-asking on every camera frame.
    ///
    /// Called on encoderQueue from every path that flips `isEncoding` back to
    /// false as the *result* of an encode.  Paths that force-release the slot
    /// (backend / resolution switches) deliberately do not call this; they clear
    /// the continuation flag themselves, because no encode is reporting in.
    private func encodeSlotDidFinish(originTag: String) {
        drainPendingTaps(originTag: originTag)
        resumeDeferredWarmupIfWaiting(originTag: originTag)
    }

    /// Consume the one-shot warmup continuation registered by
    /// `warmupSegmentationIfPossible`, if any.
    ///
    /// Consume-and-clear happens under `stateLock` in a single acquisition, so
    /// two encodes finishing back-to-back can only wake the warmup once, and a
    /// deferral registered at the same instant cannot be lost (it is written
    /// under the same lock acquisition that saw `isEncoding == true`, which the
    /// owner only clears before calling this).
    private func resumeDeferredWarmupIfWaiting(originTag: String) {
        stateLock.lock()
        let waiting = warmupWaitingOnEncode
        warmupWaitingOnEncode = false
        let folded = warmupDeferralsFolded
        warmupDeferralsFolded = 0
        stateLock.unlock()
        guard waiting else { return }

        // The mode may have changed while we waited (a `.detectionOnly` switch
        // has no use for a warm embedding, and `setMode` already re-kicks warmup
        // for the two modes that do).  `currentMode` is read on videoQueue, the
        // same queue every other reader of it uses — no new cross-queue read.
        videoQueue.async { [weak self] in
            guard let self = self else { return }
            guard self.currentMode == .segmentation || self.currentMode == .tapToSegment else {
                diagLog("[SAM] deferred warmup dropped — mode left segmentation before the \(originTag) encode finished")
                return
            }
            let foldedNote = folded > 0 ? " (\(folded) further attempt(s) folded in)" : ""

            // D-16: before calling warmupSegmentationIfPossible() check whether
            // the encode we were waiting for has already populated a fresh cache.
            // If it has, the re-run would only reach the "skipped — already fresh"
            // branch and log "SAM encoder warmup latency" from the earlier
            // (real) encode appearing in the log after a "skipped" notice —
            // the exact confusing ordering that D-16 describes.  Log a clean
            // "resolved" message instead and skip the redundant call entirely.
            self.stateLock.lock()
            let nowMs = PerfLogger.nowMs()
            let postEncodeAge = self.embeddingCache.map { nowMs - $0.timestampMs }
            self.stateLock.unlock()

            if let age = postEncodeAge, age <= 5000 {
                diagLog(String(format: "[SAM] deferred warmup resolved — \(originTag) encode populated cache (%.0f ms old)\(foldedNote)", age))
                // Latency was already logged by the encode that populated the cache;
                // logging it again here would be the D-16 double-report.
            } else {
                diagLog("[SAM] deferred warmup resumed after \(originTag) encode — cache still stale, re-running\(foldedNote)")
                self.warmupSegmentationIfPossible()
            }
        }
    }

    // MARK: Parked taps (encoder busy / warming up)

    private func parkTap(id: UUID, point: CGPoint, gen: Int, startMs: Double,
                         lockDoneMs: Double) {
        stateLock.lock()
        pendingTaps.removeAll { $0.id == id }          // one request per instance (R4)
        pendingTaps.append((id: id, point: point, gen: gen, startMs: startMs,
                            lockDoneMs: lockDoneMs))
        // Bounded by the pool size: an unbounded park queue is exactly what R4
        // forbids.  Overflow can only happen if the pool shrank underneath us.
        var overflow: [(id: UUID, point: CGPoint, gen: Int, startMs: Double,
                        lockDoneMs: Double)] = []
        while pendingTaps.count > TapInstanceManager.maxInstances {
            overflow.append(pendingTaps.removeFirst())
        }
        stateLock.unlock()
        for p in overflow {
            tapInstances.removeInstance(id: p.id)
            failTap(gen: p.gen, message: "too many pending taps — oldest discarded")
        }
    }

    /// Decode every parked tap against the freshly cached embedding.  Reached
    /// after ANY encode finishes (warmup, tap encode, background refresh,
    /// Phase 2 frame encode) — success or failure — through
    /// `encodeSlotDidFinish`, which is the only call site.  A parked tap must leave this function either decoded
    /// or visibly failed; it must never simply disappear (Architect §10.4 C;
    /// Day 4's single lost parked tap is the reason G-2 has no slow-path data).
    private func drainPendingTaps(originTag: String) {
        stateLock.lock()
        let pending = pendingTaps
        pendingTaps.removeAll()
        let emb = embeddingCache?.embedding
        let embTs = embeddingCache?.timestampMs
        let geo = tapGeometryMirror
        let capturedBackend = backendMirror
        stateLock.unlock()

        guard !pending.isEmpty else { return }

        guard let embedding = emb, let snap = geo else {
            for p in pending {
                tapInstances.removeInstance(id: p.id)
                failTap(gen: p.gen,
                        message: "segmentation could not start (\(originTag)) — tap again")
            }
            return
        }

        let nowMs = PerfLogger.nowMs()
        for p in pending {
            guard tapInstances.isRequestCurrent(id: p.id, requestGen: p.gen) else {
                diagLog("[TAP#\(p.gen)] parked tap retired before resume — dropped")
                endTapRequest(gen: p.gen)
                continue
            }
            diagLog("[TAP#\(p.gen)] parked tap resumed (\(originTag)) — decoding with fresh embedding")
            tapDecodeWithPoint(embedding: embedding,
                               instanceID: p.id,
                               canonicalPoint: p.point,
                               letterbox: snap.letterbox,
                               backend: capturedBackend,
                               // Billing: this tap ran no encode of its own.
                               reusedEmbedding: true,
                               // Route: it waited out somebody else's encode (D-12).
                               path: .parked,
                               cacheAgeMs: embTs.map { nowMs - $0 },
                               tapStartMs: p.startMs,
                               lockDoneMs: p.lockDoneMs,
                               gen: p.gen)
        }
    }

    /// Drop every instance, parked tap and in-flight request, and clear the
    /// overlay.  Used by mode / backend / resolution switches and by geometry
    /// changes — anything after which a cached instance mask would be a lie.
    private func discardAllTapWork(reason: String) {
        stateLock.lock()
        let pending = pendingTaps
        let inFlight = inFlightTaps
        pendingTaps.removeAll()
        inFlightTaps.removeAll()
        stateLock.unlock()

        tapInstances.clearAll()
        // §16.2.3 / §16.5.3 — C4 lands here.  Freeing the throttle slot is
        // mandatory: a batch in flight against a pool that no longer holds its
        // instances would never complete, and the flag would block every future
        // re-anchor for the lifetime of the process.
        resetReAnchorState()
        if !pending.isEmpty || !inFlight.isEmpty {
            diagLog("[TAP] discarded \(inFlight.count) in-flight / \(pending.count) parked tap(s) — \(reason)")
        }
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.tapProcessing = false
            self.maskImage = nil
            self.maskOutlines = nil
            // Day 6: clear anchor markers when all instances are removed.
            self.tapAnchorMarkers = []
            // Phase 4B Day 5: every live instance is gone, so every pinned
            // marker decoration goes with it (§22.2.4).
            self.pinnedInstanceIDs.removeAll()
            // Day 6 hygiene: the tag dictionary must be cleared at the same
            // points as the id set (it was missed here originally — display
            // was unaffected because markers rebuild from live instances,
            // but the dictionary grew without bound across clearAll cycles).
            self.pinnedInstanceTags.removeAll()
            // B-37: same lifecycle for the ↻ provenance decoration.
            self.revisitOriginPinTags.removeAll()
        }
    }

    // MARK: Encode / decode

    /// Encode the current camera frame on encoderQueue, then decode this tap and
    /// drain anything parked behind the same encode.
    ///
    /// Draining is what implements "同一 geometry 内实例间共享同一 embedding，
    /// 只 encode 一次": the second and third taps of a burst never start their
    /// own encode, they ride this one.
    private func tapEncodeAndDecode(buffer: CVPixelBuffer,
                                    instanceID: UUID,
                                    canonicalPoint: CGPoint,
                                    letterbox: LetterboxInfo,
                                    backend: InferenceBackend,
                                    tapStartMs: Double,
                                    lockDoneMs: Double,
                                    gen: Int) {
        // Claim the encoding slot
        stateLock.lock()
        guard !isEncoding else {
            stateLock.unlock()
            diagLog("[TAP#\(gen)] encode race — slot already claimed, parking instead")
            parkTap(id: instanceID, point: canonicalPoint, gen: gen,
                    startMs: tapStartMs, lockDoneMs: lockDoneMs)
            return
        }
        isEncoding = true
        encodeSlotOwner = .tap
        stateLock.unlock()

        encoderQueue.async { [weak self] in
            guard let self = self else { return }

            // Background guard — iOS aborts GPU/ANE work when in background
            guard !self.isAppBackgrounded else {
                self.releaseEncodeSlot()
                self.tapInstances.removeInstance(id: instanceID)
                self.failTap(gen: gen, message: "app was backgrounded — tap again")
                self.encodeSlotDidFinish(originTag: "encode-backgrounded")
                return
            }

            let t0 = PerfLogger.nowMs()
            guard let embedding = self.encodeChecked(buffer: buffer,
                                                     computeUnits: backend.computeUnits,
                                                     tag: "tap encode") else {
                self.releaseEncodeSlot()
                self.tapInstances.removeInstance(id: instanceID)
                self.failTap(gen: gen, message: "image encode failed — tap again")
                self.encodeSlotDidFinish(originTag: "encode-failed")
                return
            }
            let latencyMs = PerfLogger.nowMs() - t0

            // Cache the fresh embedding (reusable for subsequent taps on same frame)
            self.stateLock.lock()
            self.embeddingCache = (embedding: embedding, timestampMs: PerfLogger.nowMs())
            self.embeddingGeneration &+= 1                  // RE-1 (§18.1.4)
            self.isEncoding = false
            self.encodeSlotOwner = nil
            self.stateLock.unlock()

            // AB-test encoder latency stats (excludes ANE cold start).
            self.recordTapEncoderLatency(latencyMs)
            diagLog(String(format: "[TAP#%d] encode done %.2f ms (res=%d) → decode",
                         gen, latencyMs, SAMConfiguration.encoderInputSize))

            if self.tapInstances.isRequestCurrent(id: instanceID, requestGen: gen) {
                self.tapDecodeWithPoint(embedding: embedding,
                                        instanceID: instanceID,
                                        canonicalPoint: canonicalPoint,
                                        letterbox: letterbox,
                                        backend: backend,
                                        reusedEmbedding: false,
                                        path: .slow,
                                        cacheAgeMs: 0,
                                        tapStartMs: tapStartMs,
                                        lockDoneMs: lockDoneMs,
                                        gen: gen)
            } else {
                diagLog("[TAP#\(gen)] instance retired during encode — embedding cached anyway")
                self.endTapRequest(gen: gen)
            }
            // Anything parked behind this encode rides the same embedding.
            self.encodeSlotDidFinish(originTag: "tap-encode")
        }
    }

    /// Decode one instance's point prompt on decoderQueue, then re-composite the
    /// whole pool and publish it.
    ///
    /// decoderQueue is serial, so several instances decoded back-to-back are
    /// processed in order and never concurrently — the Day 5 "按顺序入队 decode
    /// （不并发）" requirement needs no extra machinery, and Architect §10.4 A
    /// forbids adding a queue to get it.
    ///
    /// ⚠️ Reserved item R3: nothing between here and `MaskRenderer` filters on
    /// mask quality.  The `iou_pred >= 0.1` gate, the candidate rules and the
    /// caps are exactly the Day 4 code — multi-instance must not perturb the
    /// 88 % baseline that the low-contrast attribution rests on.
    private func tapDecodeWithPoint(embedding: MLMultiArray,
                                    instanceID: UUID,
                                    canonicalPoint: CGPoint,
                                    letterbox lb: LetterboxInfo,
                                    backend: InferenceBackend,
                                    reusedEmbedding: Bool,
                                    path: TapPath,
                                    cacheAgeMs: Double?,
                                    tapStartMs: Double,
                                    lockDoneMs: Double,
                                    gen: Int) {
        let origSize = CGSize(width: CGFloat(lb.origW), height: CGFloat(lb.origH))

        // D-7' T3 (debug_report §30.1) — MUST be taken at the `decoderQueue.async`
        // CALL SITE, not inside the closure: T4 − T3 is the decoderQueue wait, the
        // only direct observation of R4 decode pile-up.  Inside the closure it
        // would be identically zero and the pile-up would be invisible.
        let enqueueMs = PerfLogger.nowMs()

        decoderQueue.async { [weak self] in
            guard let self = self else { return }
            guard self.tapInstances.isRequestCurrent(id: instanceID, requestGen: gen) else {
                diagLog("[TAP#\(gen)] instance retired before decode — dropped")
                self.endTapRequest(gen: gen)
                return
            }

            // Background guard — iOS aborts GPU/ANE work when in background.
            // tapDecodeWithPoint can have multiple tasks queued on decoderQueue;
            // each must individually check before touching the GPU/ANE.
            guard !self.isAppBackgrounded else {
                self.tapInstances.removeInstance(id: instanceID)
                self.failTap(gen: gen, message: "app was backgrounded — tap again")
                return
            }

            guard let decoder = self.decoderForQueue(computeUnits: backend.computeUnits) else {
                self.tapInstances.removeInstance(id: instanceID)
                self.failTap(gen: gen, message: "decoder unavailable — tap again")
                return
            }

            // Point prompt coordinates are ALWAYS in 1024 space (Architect C-2 /
            // model_plan §C.4): the Decoder normalises against 1024 regardless of
            // the encoder input resolution, so PointPromptBuilder uses inputSize=1024.
            guard let prompt = PointPromptBuilder.buildPointPrompt(
                canonicalPoint: canonicalPoint,
                origSize: origSize,
                inputSize: SAMConfiguration.pointPromptSpace
            ) else {
                self.tapInstances.removeInstance(id: instanceID)
                self.failTap(gen: gen, message: "could not build prompt — tap again")
                return
            }

            // Run model
            // D-7' T4 — decode entry (after dispatch wait + the guards above).
            let decodeStartMs = PerfLogger.nowMs()
            guard let result = decoder.decode(embedding: embedding, point: prompt,
                                              tapIndex: gen) else {
                self.tapInstances.removeInstance(id: instanceID)
                self.failTap(gen: gen, message: "segmentation failed — tap again")
                return
            }
            // D-7' T5 — decode returned; everything after this is post-processing.
            let decodeEndMs = PerfLogger.nowMs()
            // Numeric sentinel: iou_pred is a bounded quality score, so a value
            // outside [0,1] (or non-finite) means the decode is corrupt — the same
            // failure mode that renders as a slab or a 1-px line.  Discard it
            // rather than trying to interpret the mask.
            let iouSane = result.iouPreds.allSatisfy { $0.isFinite && $0 >= -0.01 && $0 <= 1.01 }
            guard iouSane else {
                self.tapInstances.removeInstance(id: instanceID)
                self.failTap(gen: gen,
                             message: "corrupt decode discarded (iou_pred \(result.iouPreds))")
                return
            }

            // Map the tap into 256×256 mask space (same ResizeLongestSide +
            // centered-pad transform as PointPromptBuilder, then ÷4) so the
            // renderer can keep only the connected component under the tap.
            let promptSpace = CGFloat(SAMConfiguration.pointPromptSpace)
            let maskScale = promptSpace / max(origSize.width, origSize.height)
            let tapPoint256 = CGPoint(
                x: (canonicalPoint.x * maskScale + (promptSpace - origSize.width  * maskScale) * 0.5) * 256.0 / promptSpace,
                y: (canonicalPoint.y * maskScale + (promptSpace - origSize.height * maskScale) * 0.5) * 256.0 / promptSpace
            )

            // Binarize through the SAME MaskRenderer code the single-mask path
            // uses (`buildTapAlpha` → `buildAlpha`): absolute logit>0 threshold,
            // multimask candidate selection, flood fill from the tap.
            // Candidate selection runs FIRST so the gate below tests the
            // SELECTED candidate's own iou_pred (D-15 fix), not the max over
            // all candidates which can be dominated by a degenerate full-frame
            // candidate with artificially high iou while the real pick is much
            // lower quality.
            guard let built = self.maskRenderer.buildTapAlpha(
                lowResMask: result.mask,
                origW: Int(lb.origW), origH: Int(lb.origH),
                tapPoint256: tapPoint256,
                iouPreds: result.iouPreds,
                tapIndex: gen
            ) else {
                self.tapInstances.removeInstance(id: instanceID)
                self.failTap(gen: gen, message: "no mask region at tap — try the object's centre")
                return
            }
            let selected = built.selected

            // D-15: gate on the SELECTED candidate's own iou_pred, not the
            // max over all candidates.  Threshold 0.1 is R3-frozen (unchanged).
            // `selected` is nil only in single-mask mode; in that case fall back
            // to the legacy max (behaviour-neutral for single-mask decodes).
            let gateIouPred = selected?.iou ?? (result.iouPreds.max() ?? 0)
            guard gateIouPred >= 0.1 else {
                self.tapInstances.removeInstance(id: instanceID)
                self.failTap(gen: gen,
                             message: String(format: "no object at tap (iou_pred=%.2f)", gateIouPred))
                return
            }

            // Attach to the instance; false ⇒ superseded/evicted mid-decode.
            //
            // `recordOrigin: true` — THE single write point of `originAlpha`
            // (§18.2.2 / B-15).  This is the tap path, so this alpha is by
            // definition what the user asked for, and it becomes the frozen
            // basis every later re-anchor is judged against.  A re-tap on the
            // same instance comes back through here with a bumped `requestGen`
            // and legitimately installs a new origin — that is REC-2.  No other
            // call site may pass `true`.
            guard self.tapInstances.updateMask(id: instanceID,
                                               requestGen: gen,
                                               mask: result.mask,
                                               alpha: built.alpha,
                                               iouPred: selected?.iou ?? gateIouPred,
                                               recordOrigin: true) else {
                diagLog("[TAP#\(gen)] instance retired during decode — mask not shown")
                self.endTapRequest(gen: gen)
                return
            }

            // Composite the whole pool: secondary instances first, primary last
            // (§3.4).  Done here on decoderQueue, never on main — the blend is
            // ~200 k pixel ops and the main thread is the thing being measured.
            let drawable = self.tapInstances.drawableInstances()
            let layers = drawable.map { inst in
                MaskRenderer.MaskLayer(
                    alpha: inst.maskAlpha ?? [],
                    color: inst.color,
                    opacity: inst.isPrimary ? TapInstanceManager.primaryOpacity
                                            : TapInstanceManager.secondaryOpacity)
            }
            let composed = self.maskRenderer.compositeLayers(layers,
                                                             origW: Int(lb.origW),
                                                             origH: Int(lb.origH),
                                                             tapIndex: gen)

            // Outlines (§3.4, C-5): traced here on decoderQueue in the same draw
            // order, published as geometry.  `MaskOutlineStyle.isEnabled` is the
            // C-6 single switch — when it is off nothing is traced at all, so
            // the Day 4 "no outline" condition costs nothing to reproduce.
            let outlineSet: MaskOutlineSet?
            if MaskOutlineStyle.isEnabled {
                let outlines = drawable.compactMap { inst -> MaskOutline? in
                    guard let alpha = inst.maskAlpha else { return nil }
                    let polys = self.maskRenderer.traceOutline(alpha: alpha,
                                                               origW: Int(lb.origW),
                                                               origH: Int(lb.origH))
                    guard !polys.isEmpty else { return nil }
                    return MaskOutline(polygons: polys, isPrimary: inst.isPrimary)
                }
                outlineSet = MaskOutlineSet(
                    canvasSize: CGSize(width: CGFloat(lb.origW), height: CGFloat(lb.origH)),
                    outlines: outlines)
            } else {
                outlineSet = nil
            }

            // What is actually on screen: the picked candidate's own numbers.
            // `gate` stays alongside so a large gate/pick gap is visible at a glance.
            let selDesc = selected.map {
                String(format: "sel=ch%d iou=%.3f area=%dpx fill=%.2f stab=%.2f%@",
                       $0.ch, $0.iou, $0.area, $0.fill, $0.stability,
                       $0.degraded ? " (degraded)" : "")
            } ?? "sel=single-mask"
            let pathLabel = path.label
            let ageDesc = cacheAgeMs.map { String(format: "%.0f", $0) } ?? "n/a"
            // The other half of the split: billing, not routing.  `own` = this
            // tap paid for an encode, `shared` = it reused one.
            let encodeDesc = reusedEmbedding ? "shared" : "own"
            let poolDesc = self.tapInstances.debugSummary()

            // ── Requirement B: the e2e window closes INSIDE this main-thread
            // block, after `maskImage` is assigned — i.e. after the mask has been
            // committed for rendering, not before it is published.  The Day 4
            // number stopped at the decoderQueue boundary and is a lower bound.
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.maskImage = composed
                self.maskOutlines = outlineSet
                self.maskRotationAngle = self.lastRotationAngle
                self.maskMirrored = (self.currentPosition == .front)
                self.tapFailure = nil
                // Day 6: rebuild anchor markers now that a new mask is placed.
                self.publishAnchorMarkersOnMain()
                // Phase 4B Day 6 (§19.4.6) — observation only, using values
                // already computed above for the log lines that follow; does
                // not affect routing, timing or any existing invariant.
                self.onTapMaskPlaced?(gen, path, instanceID, built.alpha)
                // D-7' T6 — reuses the existing e2e stamp (§30.1 recommendation),
                // so T6 − T1 is bit-identical to `tap→mask` and the 22 prior
                // samples stay comparable.  The `mask displayed` line below is
                // UNCHANGED — §29/§27 sampling regexes depend on it verbatim.
                let displayedMs = PerfLogger.nowMs()
                let e2eMs = displayedMs - tapStartMs
                perfLog(String(format: "[TAP#%d] mask displayed — %@ | gate iou_pred(selected)=%.3f | cacheAge=%@ms | %@ | tap→mask %.1f ms (%@, encode=%@)",
                             gen, selDesc, gateIouPred, ageDesc, poolDesc, e2eMs, pathLabel, encodeDesc))
                // D-7' six-segment breakdown (debug_report §30.4).  Separate line,
                // `[D7']` prefix, %.1f precision, path label included so the slow
                // path's `decide` (which contains the whole encode) is never mixed
                // into the fast path's distribution.
                perfLog(String(format:
                    "[D7'][TAP#%d] lock=%.1f decide=%.1f qwait=%.1f decode=%.1f post=%.1f | total=%.1f ms (%@)",
                    gen,
                    lockDoneMs - tapStartMs,
                    enqueueMs - lockDoneMs,
                    decodeStartMs - enqueueMs,
                    decodeEndMs - decodeStartMs,
                    displayedMs - decodeEndMs,
                    displayedMs - tapStartMs,
                    pathLabel))
                self.endTapRequest(gen: gen)
            }

            // Post-tap warmup: during active tapping the quiet-window rule
            // starves the background refresh and the cache goes stale, pushing
            // the NEXT tap onto the ~1 s encode slow path.  Kick a refresh now
            // (tap is done, encoder idle) so follow-up taps stay decode-only.
            self.videoQueue.async { [weak self] in
                self?.refreshTapEmbeddingIfNeeded(ignoreQuietWindow: true)
            }
        }
    }

    // MARK: - Re-anchor loop (Phase 4 Day 2–3, architect_output §16)

    /// Free the re-anchor throttle slot.  Called from every path that empties
    /// the instance pool.
    ///
    /// §16.2.3 / §16.5.3: if C4 empties the pool while a batch is in flight and
    /// the flag is left set, the batch can never complete against instances that
    /// no longer exist, and the throttle blocks *permanently* the next time the
    /// user enters `.tapToSegment`.  `ReAnchorLoop.reset()` also bumps the batch
    /// id so the orphaned decodes still queued on `decoderQueue` decrement
    /// nothing when they land.
    private func resetReAnchorState() {
        stateLock.lock()
        reAnchor.reset()
        stateLock.unlock()
    }

    /// §16.2.2 Step 4 — evaluated once per video frame, immediately after
    /// `refreshTapEmbeddingIfNeeded` so it reads the freshest `embeddingCache`.
    ///
    /// **videoQueue only.**  Never blocks: every decode leaves through
    /// `decoderQueue.async` and the function returns immediately.
    ///
    /// Condition order is §17.5.3's, and it is deliberate: the cheapest gates go
    /// first so that on the overwhelming majority of frames the function returns
    /// before touching a single pixel.  The throttle claim is taken atomically
    /// with the embedding snapshot so two frames cannot both win the slot.
    private func checkAndFireReAnchor() {
        // ── 0 — feature master switch (§18.3.2 / B-16). ───────────────────────
        // tasks.md's D-4 entry says "上报并暂停合入"; §18.3.2 elects this flag as
        // the executable form of that hold, so the branch neither reverts nor
        // diverges.  OFF ⇒ `.tapToSegment` is bit-identical to the frozen Phase 3
        // build, and the Debugger can run the on/off comparison without a second
        // build.  ⚠️ Ships `false`; NOT the same polarity as
        // `reAnchorConsistencyGateEnabled`, which ships `true`.
        guard DriftDetector.reAnchorEnabled else { return }

        // ── 1 — Phase 2 protection gate (§16.10). ─────────────────────────────
        guard currentMode == .tapToSegment else { return }
        // ── 2 — foreground only. ──────────────────────────────────────────────
        guard !isAppBackgrounded else { return }

        // ── 3 — batch membership (§17.9.2, ratified as a contract clause). ────
        // Only instances that already carry a mask are re-anchored.  Under the
        // §17.3 signal this is a *necessary* condition, not an optimisation: an
        // instance with no mask has no content baseline either (the baseline is
        // seeded once it becomes drawable), so it has no divergence to be
        // selected on; and the consistency gate needs a comparison basis — with
        // none it would degrade to unconditional acceptance on exactly the
        // instance that most deserves caution.  That basis is now `originAlpha`
        // (§18.2.2), and this filter is what keeps it non-nil in practice: the
        // first mask an instance ever receives comes from the tap path, which
        // writes `maskAlpha` and `originAlpha` in one lock acquisition, so
        // `maskAlpha != nil` implies `originAlpha != nil`.  It also still avoids
        // racing that instance's in-flight first tap for the same slot.
        let batch = tapInstances.drawableInstances()
        guard !batch.isEmpty else { return }

        // ── 4 — time lower bound (§17.4), BEFORE any pixel work. ──────────────
        // D-15.2 permits a fixed interval as a lower bound.  This is also the
        // third layer of the false-fire defence (§17.4): even if box filtering
        // and the threshold were both defeated by some pathological texture, the
        // worst case stays ≈3.3 batches/s instead of one per frame.
        let nowMs = PerfLogger.nowMs()
        guard nowMs - lastReAnchorFireMs >= DriftDetector.minReAnchorIntervalMs else { return }

        // Cheap reads only while the lock is held (§10.4 requirement A).
        stateLock.lock()
        let geoSnap = tapGeometryMirror
        let hasEmbedding = (embeddingCache != nil)
        // Read in the SAME acquisition as `embeddingCache` (§18.1.4): a reader
        // must never pair a generation number with an embedding it does not
        // belong to.
        let currentEmbeddingGen = embeddingGeneration
        let capturedBackend = backendMirror
        stateLock.unlock()

        guard let snap = geoSnap else { return }
        let currentGeometry = Self.frameGeometry(from: snap)

        // ── 5 — embedding present (§16.4.1 / §16.6.3). ────────────────────────
        // Baselines are deliberately NOT advanced here: the divergence went
        // unhandled, so the next frame must still see it and retry the moment
        // background refresh restores the embedding.
        guard hasEmbedding else {
            reAnchorNoEmbeddingSkips += 1
            if reAnchorNoEmbeddingSkips % 30 == 1 {
                diagLog("[REANCHOR] skipped — no embedding")
            }
            return
        }
        reAnchorNoEmbeddingSkips = 0

        // ── 5b — RE-1: embedding generation gate (§18.1.4). ───────────────────
        // Deliberately placed AFTER the embedding check and BEFORE any pixel
        // sampling, exactly as §18.1.4 specifies.  `decode → sentinel →
        // buildTapAlpha` is a pure function of `(embedding, canonicalPoint)`
        // (§18.1.2), and `canonicalPoint` is frozen by §16.7, so a second decode
        // of the same instance against the same embedding would reproduce the
        // mask already on screen bit-for-bit.  Skipping it removes work, not
        // behaviour — no frame that would have been drawn before RE-1 fails to
        // be drawn after it.
        //
        // Placing it here also means that on every frame where all instances
        // have already consumed the current generation — the common case, since
        // embeddings arrive on a ~5 s cadence (A-12 / R26) while this function
        // runs per frame — the whole of step 6's sampling is skipped too.
        //
        // ⚠️ CAPABILITY C READING NOTE (§24.1.3, B-44/B-45) — recorded here so
        // a future Architect pass can check this interpretation:
        //
        // architect_output §24.1.3's RE-1 formula reads —
        //   eligible = batch.filter {
        //     $0.lastReAnchorEmbeddingGen != currentEmbeddingGen
        //     || ($0.trackState == .tracking && distance($0.trackedPoint,
        //         $0.lastReAnchorTrackedPoint ?? $0.trackedPoint) > trackReDecodeMinDeltaPx)
        //   }
        // — and it CANNOT be applied here, verbatim, as this function's
        // sampling-eligibility gate for `.tracking` / `.lost` instances: the
        // second disjunct's `trackedPoint` only ever moves as the RESULT of a
        // search running (step 6c below), and a search only runs for an
        // instance that already passed THIS filter — using the formula here
        // would make `trackedPoint` motion a precondition of its own
        // precondition, so a `.tracking` instance could never become eligible
        // in the first place. The first disjunct (generation changed) is not
        // affected by that problem, but relying on it alone would cap a
        // `.tracking` instance at one drift sample per ~5s embedding
        // generation — identical to today's capability A/B cadence, which
        // defeats tracking's whole purpose (every cheap 300ms
        // `minReAnchorIntervalMs` tick should be able to advance
        // `trackedPoint`, not just each ~5s generation).
        //
        // Resolution applied: the formula's two disjuncts are kept verbatim
        // but split across two DIFFERENT points in the pipeline instead of
        // gating one `eligible` set with both at once:
        //   • HERE (sampling gate): `.locked` keeps today's generation-only
        //     rule, byte-for-byte. `.tracking` / `.lost` sample
        //     unconditionally, gated only by `objectTrackingEnabled` — content
        //     sampling is a pure pixel read against `latestCameraBuffer`,
        //     unrelated to `embeddingCache`, so the generation-gate's original
        //     justification (§18.1.2's pure-function argument: same embedding
        //     + same point ⇒ same output, so re-sampling is pointless) only
        //     ever applied to a point that cannot move — which is exactly
        //     `.locked`'s definition and exactly why `.locked` keeps the old
        //     rule unchanged.
        //   • Step 6c (below, after RE-2 picks one instance): once a search
        //     has produced a new candidate point, THIS is where §24.1.3's full
        //     formula (both disjuncts) decides whether that candidate is worth
        //     the cost of an actual decode versus just updating internal
        //     tracking state and letting more displacement accumulate.
        // This is not a reinterpretation of the architecture — it is resolving
        // an application-site ambiguity the spec's prose did not disambiguate
        // (same category as B-43's `trackSearchStepPx` prose/table conflict:
        // recorded, not silently resolved by guessing).
        //
        // `objectTrackingEnabled == false` never reaches the `.tracking` /
        // `.lost` arm below in practice: B-42 guarantees every instance stays
        // `.locked` for its whole life while the capability is off, so this
        // filter's observable behaviour is unchanged bit-for-bit until the
        // switch flips.
        let eligible = batch.filter { instance in
            if instance.trackState == .locked {
                return instance.lastReAnchorEmbeddingGen != currentEmbeddingGen
            }
            return DriftDetector.objectTrackingEnabled
        }
        guard !eligible.isEmpty else { return }

        // ── 6 — anchor-region content divergence (§17.3). ─────────────────────
        // Sampled HERE, per instance, and each magnitude is carried into that
        // instance's closure.  Recomputing inside a decode closure would sample
        // a later frame and report a number that triggered nothing.
        //
        // `latestCameraBuffer` is the raw capture buffer and is videoQueue-owned,
        // like this function — same queue, no cross-queue read, no new lock.  It
        // is also the only buffer that may be sampled: `latestInputBuffer` is a
        // CIContext render target and locking it would wait on the GPU (§17.3.2).
        guard let cameraBuffer = latestCameraBuffer else { return }

        var measured: [(instance: TapInstance,
                        drift: DriftDetector.Drift,
                        signature: AnchorSignature)] = []
        measured.reserveCapacity(eligible.count)
        var seededThisFrame = false

        for instance in eligible {
            // Capability C (§24.4 B-44): sample at `trackedPoint`, not the
            // frozen `canonicalPoint` — for `.locked` instances the two are
            // always equal (nothing ever moves `trackedPoint` while locked),
            // so this is a no-op change in observable behaviour for every
            // instance that exists while `objectTrackingEnabled == false`.
            guard let current = DriftDetector.signature(from: cameraBuffer,
                                                        atCanonical: instance.trackedPoint) else {
                // §17.4 graceful degradation: an unusable sample is treated as
                // "no drift" — not as a failure, and not as a conservative fire.
                return
            }
            guard let baseline = instance.anchorSignature,
                  baseline.count == current.count else {
                // No baseline yet (or `anchorGridSide` was re-tuned mid-session,
                // which makes the old baseline incomparable): seed and skip.
                // §17.5.3 step 6 — a seeding frame does not fire.
                tapInstances.setAnchorSignature(id: instance.id, signature: current)
                seededThisFrame = true
                continue
            }
            let drift = DriftDetector.drift(from: baseline, to: current)
            measured.append((instance, drift, current))
        }

        // A frame that seeded any baseline does not fire: the seeded instances
        // have nothing to compare against yet, so firing now would refresh them
        // against a divergence that was never measured.
        guard !seededThisFrame, !measured.isEmpty else { return }

        // B-47 (§24.3.2/§24.3.4) — record this pass's largest measured
        // divergence for `refreshTapEmbeddingIfNeeded`'s heavy-drift bypass.
        // Purely a `max()` over `measured`, which is already computed above;
        // no new sampling, no new DriftDetector call, and no change to any of
        // the re-anchor selection logic that follows.
        lastObservedMaxDriftLuma = measured.map { $0.drift.divergenceLuma }.max() ?? lastObservedMaxDriftLuma

        // ── 6b — RE-2: pick exactly ONE instance (§18.1.5). ───────────────────
        // ⛔ The batch maximum is GONE, not disabled (B-17).  §17.5.3 fired one
        // batch per drift event covering every drawable instance, which made
        // batch size 1…3 and put up to three decodes on the serial
        // `decoderQueue` back to back: measured qwait max 189.90 ms at N = 3
        // against 1.5 ms at N = 1 (§34.3.1/.2), i.e. D-4's 50 ms line was broken
        // purely by queue accumulation inside a batch.  §18.1.3 rules the line
        // stays and the work goes.
        //
        // The §17.5.3 justification for taking the max was not wrong, it was
        // aimed at something else: D-15.2 rejects *concurrent batches*, and this
        // is one unit inside one batch.  The single-in-flight-batch invariant
        // below is untouched, so instances still cannot queue up independently.
        //
        // Candidates are filtered by their OWN divergence — which also closes
        // R20: an instance under threshold can no longer ride along on a
        // neighbour's trigger.  (`exceedsThreshold` already folds in
        // `forceDriftForTesting`, so §17.7's "everything is drifted" override
        // keeps working and simply makes every eligible instance a candidate.)
        let candidates = measured.filter { $0.drift.exceedsThreshold }
        // Least-recently-refreshed first, so the three slots rotate rather than
        // one starving: `nil` (never refreshed) sorts ahead of any timestamp,
        // ties break on the lower slot index so the choice is fully
        // deterministic and primary wins.  With `minReAnchorIntervalMs = 300`
        // the worst-case per-instance period is 900 ms at N = 3 — far inside the
        // ~5 s staleness bound the embedding cadence already imposes (A-12), so
        // rotation costs capability A nothing measurable (§18.1.5).
        guard let pick = candidates.min(by: { lhs, rhs in
            let l = lhs.instance.lastReAnchorAtMs ?? -Double.greatestFiniteMagnitude
            let r = rhs.instance.lastReAnchorAtMs ?? -Double.greatestFiniteMagnitude
            if l != r { return l < r }
            return lhs.instance.slotIndex(in: TapInstanceManager.palette)
                 < rhs.instance.slotIndex(in: TapInstanceManager.palette)
        }) else { return }

        // ── 6c — Capability C: local block-matching search + dispatch gate
        // (§24.2.2/.3, B-44; §24.1.3 dispatch half, B-45). ─────────────────────
        //
        // ⚠️ WHY THIS RUNS HERE, ON videoQueue, BEFORE THE THROTTLE CLAIM AND
        // BEFORE ANY decoderQueue HOP: `AnchorTracker.trackSearch` /
        // `recoverySearch` call `DriftDetector.signature(from:atCanonical:)`
        // internally, which may ONLY be called on videoQueue against
        // `latestCameraBuffer` (`AnchorTracker.swift`'s file-level queue
        // discipline note; `cameraBuffer` here IS that same buffer, already in
        // scope from step 6). `reAnchorDecode`'s actual decode work runs on
        // `decoderQueue` — dispatching the search there instead would be a
        // cross-queue read of a videoQueue-owned buffer, which this project's
        // queue discipline forbids outright (§10.4). So the search must run to
        // completion here; only its RESULT — a decode target point — crosses
        // into `reAnchorDecode` as a plain `CGPoint` parameter.
        //
        // Default: today's path, byte-for-byte. `decodePoint` /
        // `newBaselineSignature` stay at their legacy values (`canonicalPoint`
        // / the pre-search signature step 6 already sampled) whenever tracking
        // is off — no branch below runs, `pick` flows straight to step 7
        // exactly as it always has.
        var decodePoint = pick.instance.canonicalPoint
        var newBaselineSignature = pick.signature
        // Non-nil only on the "found while tracking / recovered while lost"
        // path — the only case §24.1.3 wants `lastReAnchorTrackedPoint`
        // snapshotted at dispatch. `nil` whenever tracking is off, so step 8's
        // `markReAnchorDispatched` call leaves `lastReAnchorTrackedPoint`
        // untouched exactly as it does today.
        var dispatchedTrackedPoint: CGPoint?

        // ⚠️ `.locked → .tracking` ACTIVATION (coordinator-flagged gap, closed
        // here): B-42's `TapInstance.trackState` starts every instance
        // `.locked` and nothing else ever writes that field except this
        // function. If this entry guard excluded `.locked` (as it did before
        // this fix), no instance could EVER reach `.tracking` — the whole
        // capability would stay permanently inert even with
        // `objectTrackingEnabled == true`, because there was no path from the
        // construction-time state to any state this block's search logic
        // would run for. §24 never specified a separate bootstrap mechanism,
        // and deliberately none is added here: a `.locked` instance is folded
        // into the SAME `trackSearch` arm `.tracking` already uses (see the
        // merged `case .tracking, .locked:` below) — for a `.locked` instance
        // `trackedPoint` is by construction still `== canonicalPoint` (never
        // moved), so "search around `trackedPoint`" is exactly "search around
        // where the user tapped", i.e. the correct first search for an
        // instance that has never been searched before. Activation is
        // therefore LAZY and TRIGGER-DRIVEN — a `.locked` instance is only
        // ever considered here because RE-2 already picked it for exceeding
        // the drift threshold (step 6b) — not a batch conversion the moment
        // `objectTrackingEnabled` flips true. That keeps this change inside
        // the "no new trigger path" discipline B-43/B-44 already established:
        // the trigger is still "this instance's own drift crossed
        // threshold and RE-2 chose it", identical to every other path through
        // this function.
        // R36 remediation: veto the search entirely while the device is
        // panning (`CameraMotionGate.swift`) — this is exactly the failure
        // mode the local block-matching search cannot self-detect (a panning
        // camera slides unrelated background through the search window, and
        // the search reports it as a confident match). Suppressed cycles
        // fall straight through with `decodePoint`/`newBaselineSignature`
        // still at their default (`pick.instance.canonicalPoint` /
        // `pick.signature`, set above) — identical to `objectTrackingEnabled
        // == false` for this one cycle, not a dropped frame.
        if DriftDetector.objectTrackingEnabled && !CameraMotionGate.isPanning {
            let slot = pick.instance.slotIndex(in: TapInstanceManager.palette)
            let searchStartMs = PerfLogger.nowMs()
            // Both search kinds need a baseline; every instance that reached
            // `measured` (and therefore `candidates`/`pick`) already passed
            // step 6's `guard let baseline = instance.anchorSignature` unwrap
            // for THIS instance, so `pick.instance.anchorSignature` is
            // guaranteed non-nil here — the `guard` below is defensive, not
            // expected to fire, and degrades gracefully (§17.4) if it ever did.
            guard let baseline = pick.instance.anchorSignature else { return }

            // `found` is set by either arm below on a successful match — the
            // shared post-search bookkeeping (re-sample at the NEW point,
            // write it back, then apply the §24.1.3 dispatch gate) is common
            // to both "found while tracking" and "recovered while lost", so it
            // lives once, after the switch, rather than duplicated in each arm.
            var found: AnchorTracker.Candidate?

            switch pick.instance.trackState {
            case .tracking, .locked:
                // `.locked` shares this exact arm with `.tracking` (the
                // coordinator's fix for the activation gap): a `.locked`
                // instance's `trackedPoint` is by construction still
                // `== canonicalPoint` (B-42 — nothing else has ever written
                // it), so "search around `trackedPoint`" here is precisely
                // "search around where the user tapped" — the correct first
                // search for an instance that has never been searched before.
                // No separate code path, no separate log format for the
                // search itself — only a one-time activation line so a log
                // reader can tell "just activated" from "Nth tracking tick"
                // apart.
                let wasLocked = pick.instance.trackState == .locked
                if wasLocked {
                    perfLog(String(format: "[TRACK][inst#%d] locked → tracking — activating on first drift-triggered candidacy",
                                   slot))
                }
                guard let result = AnchorTracker.trackSearch(in: cameraBuffer,
                                                              baseline: baseline,
                                                              around: pick.instance.trackedPoint) else {
                    // §17.4 graceful degradation: an unsampleable search this
                    // cycle is not a verdict, just nothing to report — the
                    // instance's state is untouched and it will be
                    // re-evaluated the next time RE-2 picks it.
                    return
                }
                if result.isLost {
                    // §24.2.3: freeze `trackedPoint` in place — `updateTracking`
                    // with `trackedPoint: nil` makes that the only possible
                    // outcome of this call, not just this call's intent. A
                    // `.locked` instance whose very first search already
                    // fails the lost threshold is a legitimate outcome, not a
                    // special case: it means the object had already moved out
                    // of `trackSearchRadiusPx` by the time drift crossed
                    // threshold, so there is nothing to exempt it from.
                    tapInstances.updateTracking(id: pick.instance.id, trackedPoint: nil,
                                                trackState: .lost, anchorSignature: nil)
                    perfLog(String(format: "[TRACK][inst#%d] %@ → lost — search:%.1fms best divergence:%.1flum (≥ lost threshold)",
                                   slot, wasLocked ? "locked" : "tracking",
                                   PerfLogger.nowMs() - searchStartMs, result.best.divergenceLuma))
                    // §24.2.3 point 2: mask stays exactly as it was, via the
                    // existing stale-mask branch — no new degradation path.
                    reAnchorKeepStaleMask(slot: slot, reason: "tracking lost")
                    return
                }
                perfLog(String(format: "[TRACK][inst#%d] %@ search:%.1fms best divergence:%.1flum → (%.1f, %.1f)",
                               slot, wasLocked ? "locked" : "tracking",
                               PerfLogger.nowMs() - searchStartMs, result.best.divergenceLuma,
                               result.best.point.x, result.best.point.y))
                found = result.best
            case .lost:
                // §24.2.3: recovery search centres on `canonicalPoint`, NOT
                // `trackedPoint` (which is frozen at the stale lost position —
                // searching around it would look in the least likely place).
                guard let recovery = AnchorTracker.recoverySearch(in: cameraBuffer,
                                                                   baseline: baseline,
                                                                   around: pick.instance.canonicalPoint) else {
                    return
                }
                guard recovery.hasRecovered else {
                    perfLog(String(format: "[TRACK][inst#%d] recovery search:%.1fms best divergence:%.1flum — still lost",
                                   slot, PerfLogger.nowMs() - searchStartMs, recovery.best.divergenceLuma))
                    return
                }
                perfLog(String(format: "[TRACK][inst#%d] lost → recovered — search:%.1fms best divergence:%.1flum → (%.1f, %.1f)",
                               slot, PerfLogger.nowMs() - searchStartMs, recovery.best.divergenceLuma,
                               recovery.best.point.x, recovery.best.point.y))
                found = recovery.best
            }
            // No `.locked` case left to write: it is merged into the
            // `.tracking` arm above (the coordinator's activation-gap fix) —
            // `TrackState` has exactly three cases and both are now handled,
            // so the switch is exhaustive without one.

            guard let candidate = found else { return }

            // Re-sample the content signature AT THE NEW POINT, not `pick
            // .signature` (that was sampled at the OLD `trackedPoint`, before
            // this search ran, and is by definition the drifted content that
            // triggered the search in the first place). Using it as the new
            // baseline would record "what the object looked like right before
            // we lost track of it" as "what the object should look like",
            // biasing every subsequent search toward the drift instead of
            // away from it — the self-reinforcing template drift §24.2.1
            // candidate B already lists as a known limitation, made WORSE, not
            // just tolerated, by an implementation bug. Cost is the same
            // ≈50µs primitive already verified on-device (§17.3.2) — one extra
            // call, not a new algorithm.
            guard let refreshedSignature = DriftDetector.signature(from: cameraBuffer,
                                                                    atCanonical: candidate.point) else {
                return
            }

            // Commit the new tracking state UNCONDITIONALLY at this point —
            // §24.1.3's dispatch gate below only decides whether to spend a
            // decode, never whether the search's own result is kept. A
            // candidate found this cycle but not worth decoding yet still
            // moves `trackedPoint` / `anchorSignature` forward so the NEXT
            // cycle's search starts from an up-to-date position instead of
            // re-discovering the same displacement.
            tapInstances.updateTracking(id: pick.instance.id, trackedPoint: candidate.point,
                                        trackState: .tracking, anchorSignature: refreshedSignature)
            decodePoint = candidate.point
            newBaselineSignature = refreshedSignature
            dispatchedTrackedPoint = candidate.point

            // §24.1.3's dispatch gate (the formula's application site this
            // batch settled on — see the step-5b comment above for the full
            // derivation): worth an actual decode (~55-65ms, 50-100× the
            // search that found it) only if the embedding generation moved,
            // or the candidate moved far enough from the last DISPATCHED
            // point to matter.
            let lastDispatchedPoint = pick.instance.lastReAnchorTrackedPoint ?? candidate.point
            let movedPx = hypot(candidate.point.x - lastDispatchedPoint.x,
                                candidate.point.y - lastDispatchedPoint.y)
            let genChanged = pick.instance.lastReAnchorEmbeddingGen != currentEmbeddingGen
            guard genChanged || movedPx > AnchorTracker.trackReDecodeMinDeltaPx else {
                perfLog(String(format: "[TRACK][inst#%d] moved %.1fpx ≤ %.1fpx since last decode — state updated, decode skipped",
                               slot, movedPx, AnchorTracker.trackReDecodeMinDeltaPx))
                return
            }
        }

        // ── 7 — throttle claim (§16.2.3, D-15.2). ─────────────────────────────
        // Claiming the slot and moving the baseline (§16.4.2) happen in one
        // acquisition so a second frame cannot slip between them.
        //
        // `embeddingCache?.embedding` is read in the SAME acquisition.  §10.4 A
        // forbids *heavy* work under the lock; this is a single retain of an
        // object that already exists, no allocation and no copy, and it is the
        // established pattern everywhere else in this file (`drainPendingTaps`,
        // `scheduleEncoder`).  Reading it after unlocking would instead be an
        // unsynchronised cross-queue read of a field the encoderQueue writes.
        // Every decode dispatch below happens outside the lock.
        //
        // Order inside the block matters: the embedding is checked out FIRST and
        // the slot claimed second, so a cache that emptied between the two
        // acquisitions returns without ever having claimed — a claim that is
        // never released is precisely the permanent throttle deadlock §16.2.3
        // warns about.
        stateLock.lock()
        // `beginBatch(count: 1)` — RE-2 (§18.1.5).  The counter and `batchId`
        // generation machinery are unchanged line for line; fixing the count at
        // 1 *strengthens* rather than weakens the fixed-batch-size premise they
        // were built on (§18.1.5's revision table).
        guard let capturedEmbedding = embeddingCache?.embedding,
              let batchId = reAnchor.beginBatch(count: 1,
                                                anchor: currentGeometry) else {
            // A batch is already in flight → drop this drift event, never queue
            // it and never cancel the running batch (D-15.2).
            stateLock.unlock()
            return
        }
        // Taken here, with the embedding that will actually be decoded against,
        // rather than reusing the step-4 read: this is the generation the picked
        // instance is about to consume, and recording anything else would either
        // grant it a spare decode or deny it a real one.
        let dispatchedEmbeddingGen = embeddingGeneration
        stateLock.unlock()

        // ── 8 — advance the rate limiter and the baseline (§16.4.2 timing). ───
        // The baseline moves at batch START, not at completion: the frame that
        // triggered the batch is what the refreshed mask will be an assertion
        // about, and deferring the update would let the frames elapsing during
        // the batch accumulate divergence and fire a second, redundant batch.
        //
        // ⚠️ ONLY the picked instance's baseline moves.  The other candidates
        // were measured but not refreshed, so their current masks are still
        // assertions about their own older frames — and their un-advanced
        // baselines are precisely what keeps them over threshold on the next
        // fire, which is what makes RE-2's rotation a rotation instead of a
        // silent drop (§18.1.5).
        // §24.4 B-44/B-45: `newBaselineSignature` is `pick.signature`
        // unchanged whenever tracking is off (byte-for-byte the pre-existing
        // write); on any capability-C found/recovered/activated path — which
        // now includes a `.locked` instance's first successful search, per
        // the coordinator's activation-gap fix — it is the signature step 6c
        // re-sampled AT THE NEW POINT — the same value `updateTracking`
        // already wrote, so this is a harmless, idempotent re-write there,
        // not a second, different write.
        lastReAnchorFireMs = nowMs
        tapInstances.setAnchorSignature(id: pick.instance.id, signature: newBaselineSignature)
        tapInstances.markReAnchorDispatched(id: pick.instance.id,
                                            embeddingGeneration: dispatchedEmbeddingGen,
                                            atMs: nowMs,
                                            trackedPoint: dispatchedTrackedPoint)

        // ── 9 — dispatch the single decode. ───────────────────────────────────
        let lb = snap.letterbox
        // `reAnchorDecode` reads `instance.trackState` to pick its consistency
        // gate (§24.1.5/B-46, untouched by this batch). `pick.instance` is a
        // snapshot taken before step 6c's search ran, so on the
        // found/recovered/activated path it still carries the PRE-search
        // state (`.tracking` about to be reconfirmed, `.lost` about to become
        // `.tracking`, or — since the coordinator's activation-gap fix —
        // `.locked` about to become `.tracking` for the first time) — not
        // what this particular decode is actually about. Patch just that one
        // field on a local copy (`instance` is a struct; `trackState` is the
        // only `var` this needs) so the gate sees the state this dispatch is
        // FOR, matching what `updateTracking` already committed to the pool
        // above. Tracking-off path: `dispatchedTrackedPoint == nil`, so this
        // copy is untouched and behaves exactly like passing `pick.instance`
        // directly, byte for byte.
        var dispatchInstance = pick.instance
        if dispatchedTrackedPoint != nil {
            dispatchInstance.trackState = .tracking
        }
        reAnchorDecode(instance: dispatchInstance,
                       embedding: capturedEmbedding,
                       letterbox: lb,
                       backend: capturedBackend,
                       drift: pick.drift,
                       batchId: batchId,
                       decodePoint: decodePoint)
    }

    /// Re-decode one instance at `decodePoint` and republish the composited
    /// pool.
    ///
    /// ⚠️ §16.7 UPDATE (§24.1.2, B-44): when `instance.trackState == .locked`
    /// (capability C off, or this instance never engaged tracking), the
    /// caller always passes `instance.canonicalPoint` here, unchanged from
    /// every build before this one — the point is fixed at tap time and
    /// re-anchor may not move it. When `instance.trackState == .tracking`
    /// (capability C on and this instance's tracked position was just found or
    /// recovered by `checkAndFireReAnchor`'s step 6c), the caller instead
    /// passes the search's resulting point — `canonicalPoint` itself is still
    /// never touched (§16.7's original guarantee, restated in §24.1.2: it
    /// remains `PinFactory`'s sole read source, frozen for the instance's
    /// whole life), only the DECODE INPUT moves. This function itself does not
    /// decide which point that is — it decodes at whatever `decodePoint` it is
    /// given and is agnostic to how the caller chose it.
    ///
    /// This mirrors `tapDecodeWithPoint`'s decode + composite steps exactly —
    /// same `SAMDecoder`, same `MaskRenderer.buildTapAlpha`, same R3-frozen
    /// gates — and differs only in its failure semantics: a re-anchor that fails
    /// keeps the previous mask instead of failing the tap (§16.6.1).  Nothing
    /// here calls the encoder (§16.7).
    private func reAnchorDecode(instance: TapInstance,
                                embedding: MLMultiArray,
                                letterbox lb: LetterboxInfo,
                                backend: InferenceBackend,
                                drift: DriftDetector.Drift,
                                batchId: UInt64,
                                decodePoint: CGPoint) {
        let instanceID = instance.id
        let capturedGen = instance.requestGen                 // §16.5.1 generation snapshot
        let slot = instance.slotIndex(in: TapInstanceManager.palette)
        let origSize = CGSize(width: CGFloat(lb.origW), height: CGFloat(lb.origH))
        let divergenceLuma = drift.divergenceLuma
        // The comparison basis for the consistency gate: the alpha the USER'S
        // TAP produced, not the mask currently on screen (§18.2.2, RE-3).
        //
        // ⛔ This used to be `instance.maskAlpha`, and that was the D-1c defect.
        // A successful re-anchor writes `maskAlpha` itself, so the gate compared
        // each refresh against the previous refresh's own output: it constrained
        // adjacent steps, and a chain of adjacent steps each ≥ 0.5 implies
        // nothing whatever about the distance from where the user started.  On
        // device the mask walked across three unrelated objects with 0/216
        // rejections (§34.4.4 / A-14).  `originAlpha` is written on the tap path
        // only, so the chain cannot form.  Arrays are COW — capturing it costs a
        // retain, not a 2 MB copy.
        let originAlpha = instance.originAlpha
        // Capability C consistency-gate selector (§24.1.5, §24.4 B-46).
        // `.locked` for every instance while `DriftDetector
        // .objectTrackingEnabled == false` (the shipping default) or for an
        // instance that never engaged tracking. B-44's caller
        // (`checkAndFireReAnchor` step 6c) passes an `instance` whose
        // `trackState` already reflects the OUTCOME of this dispatch's own
        // search — `.tracking` when a track search found (or a recovery
        // search recovered) the object THIS cycle, matching `decodePoint`
        // above 1:1 — not the state the pool held before that search ran.
        // Captured outside the closure, same as the other per-instance
        // snapshots above, so the gate branch below is decided once at
        // dispatch time and cannot be confused by a later mutation.
        let trackState = instance.trackState

        // ⚠️ T3 — taken at the `decoderQueue.async` CALL SITE, outside the
        // closure.  `qwait` (T4 − T3) is the only direct observation of R4
        // decode pile-up, and re-anchor is exactly the cadence that can revive
        // it (debug_report §33.2.3).  Moved onto the closure's first line it
        // would read identically zero and the pile-up would be invisible.
        let enqueueMs = PerfLogger.nowMs()

        decoderQueue.async { [weak self] in
            guard let self = self else { return }
            // The batch counter must fall on EVERY exit path, success or
            // failure (§16.2.4).  A `defer` immediately after the self guard is
            // the only formulation that survives the eleven early returns below.
            defer { self.finishReAnchorUnit(batchId: batchId) }

            // Superseded by a newer tap on this instance, or the instance was
            // evicted / cleared while queued → drop silently (§16.5.1).
            guard self.tapInstances.isRequestCurrent(id: instanceID,
                                                     requestGen: capturedGen) else { return }
            guard !self.isAppBackgrounded else { return }
            guard let decoder = self.decoderForQueue(computeUnits: backend.computeUnits) else {
                self.reAnchorKeepStaleMask(slot: slot, reason: "decoder unavailable")
                return
            }
            guard let prompt = PointPromptBuilder.buildPointPrompt(
                canonicalPoint: decodePoint,
                origSize: origSize,
                inputSize: SAMConfiguration.pointPromptSpace
            ) else {
                self.reAnchorKeepStaleMask(slot: slot, reason: "prompt build failed")
                return
            }

            // T4 — decode entry.
            let decodeStartMs = PerfLogger.nowMs()
            let decodeResult = decoder.decode(embedding: embedding, point: prompt,
                                              tapIndex: capturedGen)
            // T5 — decode returned.
            let decodeEndMs = PerfLogger.nowMs()

            // §17.5.3 log format.  Only the drift prefix changed (pt/deg → lum);
            // `qwait:` and `decode:` keep their names, formats and relative
            // position CHARACTER FOR CHARACTER, so §16.9.3's grep commands and
            // the D-4 acceptance are unaffected.  The magnitude reported is this
            // instance's own divergence, not the batch maximum — per-instance
            // values are what makes the number useful for tuning.  Emitted for
            // every dispatched unit, on every exit path, so `qwait` / `decode`
            // are never lost to a rejected mask (§16.8's ISSUE-P4-DECODE passive
            // collection depends on exactly that).
            //
            // §35.7.2 appends the consistency gate's own fields to this same
            // line (`| iou: … origin: …px new: …px`).  Those values do not exist
            // until `buildTapAlpha` has run, so the line is now BUILT here and
            // EMITTED at the first of: the gate (with the fields), or any
            // earlier exit path (prefix only).  `emitReAnchorLine` is
            // idempotent and a `defer` backstops every path, including ones
            // added later — the §16.8 invariant is enforced by construction, not
            // by remembering to call it.
            //
            // ⚠️ Consequence for log readers: on the paths that reach
            // `buildTapAlpha`, this line now appears AFTER that call's
            // `[TAP#g] candidates: …` line instead of before it.  §35.6.4's
            // pairing recipe ("the `[TAP#g]` line following each `[REANCHOR]`")
            // therefore no longer applies — but it is also no longer needed,
            // because `new:` carries that area on the `[REANCHOR]` line itself.
            let reAnchorLinePrefix = String(
                format: "[REANCHOR][inst#%d] drifted %.1flum → qwait: %.1fms decode: %.1fms",
                slot, divergenceLuma,
                decodeStartMs - enqueueMs,
                decodeEndMs - decodeStartMs)
            var reAnchorLineEmitted = false
            func emitReAnchorLine(_ suffix: String = "") {
                guard !reAnchorLineEmitted else { return }
                reAnchorLineEmitted = true
                perfLog(reAnchorLinePrefix + suffix)
            }
            // Runs before the `finishReAnchorUnit` defer above (LIFO).
            defer { emitReAnchorLine() }

            guard let result = decodeResult else {
                emitReAnchorLine()
                self.reAnchorKeepStaleMask(slot: slot, reason: "decode returned nil")
                return
            }
            // Numeric sentinel — identical to the tap path (R3-frozen).
            guard result.iouPreds.allSatisfy({ $0.isFinite && $0 >= -0.01 && $0 <= 1.01 }) else {
                emitReAnchorLine()
                self.reAnchorKeepStaleMask(slot: slot,
                                           reason: "corrupt decode (iou_pred \(result.iouPreds))")
                return
            }

            // Same ResizeLongestSide + centred-pad transform ÷4 as the tap path.
            // Seeded from `decodePoint` — the same point the decode itself was
            // prompted at (`buildPointPrompt` above) — so `buildTapAlpha`'s
            // connected-region search starts from where this decode actually
            // looked, not from a stale `canonicalPoint` that may be far away
            // once tracking has moved `decodePoint` off it.
            let promptSpace = CGFloat(SAMConfiguration.pointPromptSpace)
            let maskScale = promptSpace / max(origSize.width, origSize.height)
            let tapPoint256 = CGPoint(
                x: (decodePoint.x * maskScale + (promptSpace - origSize.width  * maskScale) * 0.5) * 256.0 / promptSpace,
                y: (decodePoint.y * maskScale + (promptSpace - origSize.height * maskScale) * 0.5) * 256.0 / promptSpace
            )

            guard let built = self.maskRenderer.buildTapAlpha(
                lowResMask: result.mask,
                origW: Int(lb.origW), origH: Int(lb.origH),
                tapPoint256: tapPoint256,
                iouPreds: result.iouPreds,
                tapIndex: capturedGen
            ) else {
                emitReAnchorLine()
                self.reAnchorKeepStaleMask(slot: slot, reason: "no mask region at point")
                return
            }
            let selected = built.selected
            let gateIouPred = selected?.iou ?? (result.iouPreds.max() ?? 0)
            guard gateIouPred >= 0.1 else {
                emitReAnchorLine()
                self.reAnchorKeepStaleMask(slot: slot,
                                           reason: String(format: "iou_pred=%.2f below gate", gateIouPred))
                return
            }

            // §17.3.3 — mask consistency veto (capability B), basis revised by
            // §18.2.2.  Without it the refresh loop is a net regression: with
            // `canonicalPoint` frozen, a camera that has panned off the target
            // re-decodes "whatever is at that screen position now", which may be
            // the table, and the user's selection would silently become
            // something else.
            //
            // The invariant it now enforces is statable and directly checkable:
            // **every mask an instance ever displays has IoU ≥
            // `reAnchorAcceptIoU` with the mask that instance's tap produced.**
            // A veto therefore freezes the mask rather than nudging it, and
            // §18.2.3 rules that terminal state CORRECT, not a bug to be fixed:
            // when the anchor no longer covers the user's object, not updating
            // is the right output.  It is visible to the user and it recovers by
            // itself — pan back and the IoU rises through the same threshold
            // (REC-1), or tap again and the tap path installs a new origin
            // (REC-2), or the instance ages out (REC-3).  Costs are booked as
            // R25: a large but legitimate deformation is vetoed too.
            //
            // This is NOT mask-quality logic (R3): it never evaluates whether a
            // mask is good, never compares candidates, never reads `iou_pred` or
            // stability, and `buildTapAlpha` above ran untouched with its verdict
            // accepted in full.  It compares the geometry of two products that
            // each already passed every R3 gate, and its only possible effect is
            // to VETO — falling back to the already-approved "keep the previous
            // mask" branch.  Nothing it does can produce an outcome that would
            // not occur with the gate absent (§13's E-3 form).  Tap and
            // `.segmentation` paths never reach here.
            //
            // ⛔ ISSUE-P4-GATE (§35.6.3): until this fix the dimensions below
            // were `Int(lb.origW)` / `Int(lb.origH)` = 1080 x 1920, while both
            // alphas are 256 x 256 = 65 536 bytes.  `alphaIoU`'s size guard was
            // therefore permanently false, it returned its safe default of 1.0
            // on every call, and `reAnchorRejectUpdate` was unreachable — the
            // gate has never made a single real comparison since it landed.  The
            // 0/216 and 0/76 rejection counts from two device sessions measured
            // a system that effectively had no gate.  `maskAlphaSide` is the
            // invariant, not a literal, so this cannot silently drift again.
            //
            // §35.7.2: the IoU is computed OUTSIDE the enable check and outside
            // the `originAlpha != nil` unwrap, so the number reaches the log in
            // every world — gate on, gate off, no origin.  A missing comparison
            // prints `n/a`, never a fabricated number: "gate did not run" must
            // never again be indistinguishable from "gate ran and passed".
            let originNonzero: Int? = originAlpha.map { alpha in
                // ~65 k byte scan, ≈20 µs on decoderQueue, next to a ≈55 ms
                // decode.  Deliberately not cached on TapInstance: a stored
                // field would be a second copy of a derived value to keep in
                // sync, and §35.7.2 explicitly books this as a detail, not a
                // criterion.
                var n = 0
                alpha.withUnsafeBufferPointer { p in
                    for v in p where v != 0 { n += 1 }
                }
                return n
            }
            // Capability C branch (§24.1.5, §24.4 B-46): `.tracking` means
            // `trackedPoint` may legitimately have moved away from
            // `canonicalPoint` (B-44's `checkAndFireReAnchor` step 6c), so the
            // object is EXPECTED to sit at a different absolute position than
            // the tap-time origin mask. Comparing that with raw `alphaIoU`
            // would read a successful track as failed drift and veto every
            // correct refresh — see `centroidAlignedIoU`'s doc comment for the
            // full argument. `.locked` / `.lost` take the untouched `alphaIoU`
            // arm, byte-for-byte the same call this gate has always made.
            //
            // While `DriftDetector.objectTrackingEnabled == false` (the
            // shipping default), every instance stays `.locked` for its whole
            // life (B-42's guarantee), so this switch takes the `alphaIoU` arm
            // on every call that can occur — bit-identical to the build before
            // B-44/B-45. The `.tracking` arm is now a live call path once the
            // switch is on (B-44 wires `checkAndFireReAnchor` to pass a
            // dispatch-time `.tracking` state here on a found/recovered
            // search), not merely a real-but-unreachable implementation as it
            // was at B-46 landing time.
            let gateIoU: Double? = originAlpha.map { origin -> Double in
                switch trackState {
                case .tracking:
                    return DriftDetector.centroidAlignedIoU(origin, built.alpha,
                                                            width: DriftDetector.maskAlphaSide,
                                                            height: DriftDetector.maskAlphaSide)
                case .locked, .lost:
                    return DriftDetector.alphaIoU(origin, built.alpha,
                                                  width: DriftDetector.maskAlphaSide,
                                                  height: DriftDetector.maskAlphaSide,
                                                  stride: 1)
                }
            }
            // The threshold must track which quantity `gateIoU` actually is —
            // `trackConsistencyAcceptIoU` and `reAnchorAcceptIoU` are two
            // physically different, independently-stored constants (B-46);
            // comparing a centroid-aligned IoU against `reAnchorAcceptIoU`
            // (or vice versa) would silently mix the two spaces.
            let acceptThreshold: Double = (trackState == .tracking)
                ? DriftDetector.trackConsistencyAcceptIoU
                : DriftDetector.reAnchorAcceptIoU
            let iouField = gateIoU.map { String(format: "%.2f", $0) } ?? "n/a"
            let originField = originNonzero.map { "\($0)px" } ?? "n/a"
            emitReAnchorLine(" | iou: \(iouField) origin: \(originField) new: \(built.nonzeroCount)px")

            if DriftDetector.reAnchorConsistencyGateEnabled,
               let iou = gateIoU, iou < acceptThreshold {
                self.reAnchorRejectUpdate(slot: slot, iou: iou, threshold: acceptThreshold,
                                          originPx: originNonzero ?? 0,
                                          newPx: built.nonzeroCount)
                return
            }

            // Second generation check, this time atomic with the write
            // (§16.5.1): false ⇒ a tap superseded this instance, or it was
            // evicted, while the decode ran.
            guard self.tapInstances.updateMask(id: instanceID,
                                               requestGen: capturedGen,
                                               mask: result.mask,
                                               alpha: built.alpha,
                                               iouPred: selected?.iou ?? gateIouPred) else { return }

            // Composite on decoderQueue, exactly as the tap path does — the
            // blend is ~200 k pixel ops and must not land on main.
            let drawable = self.tapInstances.drawableInstances()
            guard !drawable.isEmpty else { return }
            let layers = drawable.map { inst in
                MaskRenderer.MaskLayer(
                    alpha: inst.maskAlpha ?? [],
                    color: inst.color,
                    opacity: inst.isPrimary ? TapInstanceManager.primaryOpacity
                                            : TapInstanceManager.secondaryOpacity)
            }
            let composed = self.maskRenderer.compositeLayers(layers,
                                                             origW: Int(lb.origW),
                                                             origH: Int(lb.origH),
                                                             tapIndex: capturedGen)
            let outlineSet: MaskOutlineSet?
            if MaskOutlineStyle.isEnabled {
                let outlines = drawable.compactMap { inst -> MaskOutline? in
                    guard let alpha = inst.maskAlpha else { return nil }
                    let polys = self.maskRenderer.traceOutline(alpha: alpha,
                                                               origW: Int(lb.origW),
                                                               origH: Int(lb.origH))
                    guard !polys.isEmpty else { return nil }
                    return MaskOutline(polygons: polys, isPrimary: inst.isPrimary)
                }
                outlineSet = MaskOutlineSet(
                    canvasSize: CGSize(width: CGFloat(lb.origW), height: CGFloat(lb.origH)),
                    outlines: outlines)
            } else {
                outlineSet = nil
            }

            // §16.2.6 — the existing publish path.  N = 3 means three separate
            // main-thread hops, each carrying the ~16.69 ms `post` cost measured
            // in D-7'.  If the main thread starts dropping frames during a
            // re-anchor session this is the first place to look (§15.7.4).
            // No `tapProcessing` / `tapFailure` / anchor-marker work here: a
            // re-anchor is not a user request, it must not raise the loading
            // indicator, and it never moves an anchor point.
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.maskImage = composed
                self.maskOutlines = outlineSet
                self.maskRotationAngle = self.lastRotationAngle
                self.maskMirrored = (self.currentPosition == .front)
            }
        }
    }

    /// §16.6.1 — a failed re-anchor keeps the previous mask.  Clearing it would
    /// flicker the overlay for what is usually a transient condition (embedding
    /// swap, memory pressure), and decode failure is not one of the C1–C6
    /// reasons a mask may legally disappear.
    private func reAnchorKeepStaleMask(slot: Int, reason: String) {
        faultLog("[REANCHOR][inst#\(slot)] decode failed — keeping stale mask (\(reason))")
    }

    /// §17.3.3 / §17.5.3 — the consistency gate refused this refresh; the
    /// previous mask stays.  Logged on its own line, distinct from the decode
    /// failures above, so D-6 can count acceptances against rejections without
    /// disentangling the two causes.
    ///
    /// `perfLog`, not `diagLog`: D-6 is a ratio against the `[REANCHOR]` lines,
    /// which are always printed, so a quiet-mode session must not silence one
    /// side of it and turn a rejecting gate into an invisible one.
    ///
    /// §35.7.2 note 3: the existing text is unchanged and `origin:` / `new:` are
    /// appended, so the accept and reject branches carry the same field set and
    /// one regex extracts both.
    ///
    /// `threshold` is passed in by the caller (§24.4 B-46) rather than read
    /// from `DriftDetector.reAnchorAcceptIoU` here, because which constant
    /// applies now depends on the instance's `trackState` — the caller has
    /// already made that choice once (to decide whether to reject at all) and
    /// this log line must report the SAME number, not re-derive a possibly
    /// different one. For every instance that exists today (`trackState ==
    /// .locked`) the caller always passes `DriftDetector.reAnchorAcceptIoU`,
    /// so the printed line is byte-for-byte what it was before this batch.
    private func reAnchorRejectUpdate(slot: Int, iou: Double, threshold: Double, originPx: Int, newPx: Int) {
        perfLog(String(format: "[REANCHOR][inst#%d] rejected — mask IoU %.2f < %.2f, keeping previous mask | origin: %dpx new: %dpx",
                       slot, iou, threshold, originPx, newPx))
    }

    /// Retire one unit of a re-anchor batch; the last one releases the throttle.
    private func finishReAnchorUnit(batchId: UInt64) {
        stateLock.lock()
        reAnchor.completeUnit(batch: batchId)
        stateLock.unlock()
    }

    /// Keep the embedding cache warm while idling in tapToSegment mode so a tap
    /// can take the decode-only fast path (<100 ms) instead of a cold ~1 s encode.
    /// Runs on videoQueue (called from runDetectionPipeline).
    ///
    /// Two rules learned from device logs (encode ≈ 0.8–1.3 s on iPhone 11):
    /// 1. Refresh *proactively* at age > 5 s — before the 8 s TTL — so a tap
    ///    almost never lands on an expired cache and gets stuck on the slow path.
    /// 2. Yield to taps: never start a refresh while a tap is in flight or was
    ///    accepted in the last 1.5 s, so a refresh never occupies the serial
    ///    encoderQueue right when a tap needs it (root cause of the observed
    ///    2.6–3.6 s tap→mask latencies).
    private func refreshTapEmbeddingIfNeeded(ignoreQuietWindow: Bool = false) {
        // DEBUG: forceSlowPath suspends all background refresh so the cache stays
        // empty and every tap is forced through the full slow encode+decode path.
        guard !forceSlowPath else { return }
        // DEBUG (B-12): suspend refresh WITHOUT touching the cache, so the
        // "refresh suspended" and "cache cold" variables can be moved
        // independently (§18.4.3).  Early return only — no other effect.
        guard !suspendRefreshOnly else { return }
        guard let buffer = latestCameraBuffer else { return }
        let nowMs = PerfLogger.nowMs()

        stateLock.lock()
        let cacheAgeMs = embeddingCache.map { nowMs - $0.timestampMs }
        let busy = isEncoding
        let parked = !pendingTaps.isEmpty
        let msSinceTap = nowMs - lastTapAcceptedMs
        stateLock.unlock()

        let cacheFresh = (cacheAgeMs ?? .infinity) <= 5000
        let quiet = ignoreQuietWindow || msSinceTap > 1500

        // B-47 (§24.3.2) — heavy-drift bypass. Strictly rate-limited and
        // deliberately independent of the TTL check: it must never let drift
        // fire an unconditional re-encode (that reopens the "frequent
        // re-encode" failure mode this pipeline already fixed, and capability
        // C makes drift events structurally more frequent), so every one of
        // these conditions is a hard AND, not a scored heuristic.
        //
        // `quiet` is a NECESSARY condition here, not an optional refinement —
        // it may not be dropped even for a severe divergence. `quiet == false`
        // means a tap was accepted within the last 1.5 s, i.e. the user is
        // actively interacting right now; that window is exactly when the
        // decode-side re-anchor/tracking machinery is already the correct tool
        // to respond, and forcing a re-encode here would steal the serial
        // encoderQueue from the tap the user is actively waiting on — reopening
        // the historically-fixed "background refresh pre-empts encoderQueue,
        // tap→mask 2.6–3.6 s" failure mode. Conversely `quiet == true` is the
        // cheapest, highest-value moment to pay for a re-encode: the scene has
        // settled and no tap is in flight, and it is also plausibly the moment
        // right after the user re-aimed the camera at something new.
        let heavyDriftBypass = DriftDetector.objectTrackingEnabled
            && quiet
            && cacheAgeMs != nil && cacheAgeMs! >= DriftDetector.minHeavyDriftAgeFloorMs
            && lastObservedMaxDriftLuma >= DriftDetector.heavyDriftForceRefreshLuma
            && (nowMs - lastHeavyDriftRefreshMs) >= DriftDetector.minHeavyDriftRefreshIntervalMs

        guard !busy, !parked, quiet, (!cacheFresh || heavyDriftBypass) else { return }

        // Log the actual trigger condition so device logs clearly show *why*
        // this refresh fired (age-based TTL vs. drift), not just that it fired.
        let triggerReason: String
        if cacheAgeMs == nil {
            triggerReason = "cold_start (no cache)"
        } else if cacheAgeMs! >= 5000 {
            triggerReason = String(format: "age=%.0f ms ≥ 5000 ms threshold", cacheAgeMs!)
        } else {
            // B-47: the only way to reach this branch with `cacheAgeMs! < 5000`
            // is `heavyDriftBypass` — see the guard above — so report the
            // actual cause (age AND the drift magnitude that forced it),
            // rather than the age-threshold wording that would misleadingly
            // claim age ≥ 5000 ms when it is not.
            triggerReason = String(format: "heavy_drift bypass (age=%.0f ms, maxDrift=%.1f lum ≥ %.1f threshold)",
                                    cacheAgeMs!, lastObservedMaxDriftLuma, DriftDetector.heavyDriftForceRefreshLuma)
        }
        diagLog("[CACHE] background refresh triggered: \(triggerReason)")

        // Snapshot the backend on videoQueue — `backend` is written on sessionQueue.
        let capturedBackend = backend
        let capturedLetterbox = lastLetterbox   // videoQueue-owned, snapshot here

        stateLock.lock()
        guard !isEncoding else {
            // D-1: this used to be a bare `return`, the second of the two silent
            // encoder-slot exits.  The `busy` read a few lines up is taken under
            // a *different* lock acquisition, so another queue can claim the slot
            // in between; losing here is benign (the next frame retries) but it
            // was indistinguishable from "the refresh rule decided not to run",
            // which is what made the Day 5 cold-start race unreadable in the log.
            // Rare by construction, so every occurrence is printed, with a
            // running count so a pathological loop is obvious at a glance.
            refreshSlotLostCount += 1
            let n = refreshSlotLostCount
            stateLock.unlock()
            diagLog("[SAM] background refresh lost the encoder slot (race, n=\(n)) — retrying next frame")
            // Losing the encoder race must not also cost the decoder its warmup:
            // decoderQueue is independent of encoderQueue, and the call is
            // latched by `decoderWarmupDecodeDone`, so kicking it here is free
            // and removes the last path where warmup depended on who won.
            // This function only ever runs in .tapToSegment (its two call sites
            // are both behind `currentMode == .tapToSegment`), so the point
            // decoder is the one that needs warming.
            warmupDecoderIfPossible(letterbox: capturedLetterbox,
                                    backend: capturedBackend,
                                    origin: "refresh-slot-lost",
                                    mode: .tapToSegment)
            return
        }
        isEncoding = true
        encodeSlotOwner = .backgroundRefresh
        stateLock.unlock()

        // B-47: this refresh is only ever won here via `heavyDriftBypass` when
        // `cacheAgeMs! < 5000` (the `!cacheFresh` branch of the guard above
        // would otherwise have required age ≥ 5000 ms) — record the cooldown
        // timestamp so the bypass's own independent throttle
        // (`minHeavyDriftRefreshIntervalMs`) is honoured on the next pass.
        // videoQueue-only, no lock, same discipline as `lastReAnchorFireMs`.
        if heavyDriftBypass {
            lastHeavyDriftRefreshMs = nowMs
        }

        // Item 6 — log background refresh trigger source.
        // Three cases:
        //   cold_start      — no embedding yet; this refresh IS the first encode
        //   ttl_approaching — cache age ≥ 5000 ms → proactive TTL refresh
        //                     (normal steady-state; NOT heavy object motion)
        //   heavy_drift     — cache is young (< 5000 ms) but reached this call
        //                     with the encoder slot won, which (B-47) is only
        //                     possible when `heavyDriftBypass` was true for
        //                     this pass — see the guard above. There is no
        //                     other path to `cacheAgeMs! < 5000` here.
        let refreshLogReason: String
        if cacheAgeMs == nil {
            refreshLogReason = "cold_start"
        } else if cacheAgeMs! >= 5000 {
            refreshLogReason = "ttl_approaching"
        } else {
            refreshLogReason = "heavy_drift"
        }
        diagLog("[CACHE] re-encode reason: \(refreshLogReason) (background refresh, age=\(cacheAgeMs.map { String(format: "%.0f ms", $0) } ?? "none"), threshold=5000ms)")

        // D-2: whichever path wins the encoder slot, the decoder must still be
        // warmed.  This call used to sit inside the `isColdStart` branch below,
        // so a refresh that ran with a warm cache never touched decoderQueue and
        // the ~1.5 s decoder model load stayed on the user's first tap whenever
        // the cold-start refresh had been the loser.  Idempotent and latched.
        warmupDecoderIfPossible(letterbox: capturedLetterbox,
                                backend: capturedBackend,
                                origin: "refresh",
                                mode: .tapToSegment)

        // When there is no cache at all this refresh IS the cold start — it is
        // the call that beats warmup to the encoder slot on the first frame
        // after entering .tapToSegment.  It must therefore raise the same
        // "initialising" state warmup would have, or the banner is missing for
        // exactly the seconds it exists for (Debugger §17.6-3).
        let isColdStart = (cacheAgeMs == nil)
        if isColdStart {
            setWarmingUp(true)
            diagLog("[SAM] cold-start encode taken by background refresh — warming state raised")
        }

        encoderQueue.async { [weak self] in
            guard let self = self else { return }

            guard !self.isAppBackgrounded else {
                self.releaseEncodeSlot()
                if isColdStart { self.setWarmingUp(false) }
                self.encodeSlotDidFinish(originTag: "refresh-backgrounded")
                return
            }

            let t0 = PerfLogger.nowMs()
            if let embedding = self.encodeChecked(buffer: buffer,
                                                 computeUnits: capturedBackend.computeUnits,
                                                 tag: "background refresh") {
                let t1 = PerfLogger.nowMs()
                self.stateLock.lock()
                self.embeddingCache = (embedding: embedding, timestampMs: t1)
                // RE-1 (§18.1.4).  This is the site that actually paces the
                // re-anchor loop: §18.1.2 measured 15 background refreshes
                // against 91 re-anchor decodes, and A-12/R26 name this 5000 ms
                // gate — not `minReAnchorIntervalMs` — as capability A's real
                // staleness bound (~5 s).
                self.embeddingGeneration &+= 1
                self.isEncoding = false
                self.encodeSlotOwner = nil
                self.stateLock.unlock()
                // Deliberately un-indexed: this encode belongs to no tap, so a
                // bare `[TAP]` keeps it out of every `[TAP#N]` chain.
                diagLog(String(format: "[TAP] background embedding refresh %.2f ms", t1 - t0))
                if isColdStart {
                    perfLog(String(format: "SAM encoder warmup latency: %.2f ms (via background refresh)", t1 - t0))
                    self.setWarmingUp(false)
                }
                // The rehearsal decode now has an embedding to run on.  Attempted
                // after EVERY refresh encode, not only the cold-start one: the
                // pre-encode attempt above may have found no embedding yet, and
                // `decoderWarmupDecodeDone` latches it to exactly one real decode.
                self.warmupDecoderIfPossible(letterbox: capturedLetterbox,
                                             backend: capturedBackend,
                                             origin: "refresh-encoded",
                                             mode: .tapToSegment)
                // A tap may have arrived (and been parked) during this encode.
                self.encodeSlotDidFinish(originTag: "background-refresh")
            } else {
                faultLog("[TAP] background embedding refresh failed — releasing slot")
                self.releaseEncodeSlot()
                if isColdStart { self.setWarmingUp(false) }
                self.encodeSlotDidFinish(originTag: "refresh-failed")
            }
        }
    }

    /// Sliding-window stats for tap-path encoder latency (AB test).
    /// Cold start (first call) is excluded, matching the Phase 2 stat convention.
    private func recordTapEncoderLatency(_ latencyMs: Double) {
        encoderQueue.async { [weak self] in
            guard let self = self else { return }
            let isCold = (self.tapEncoderCallCount == 0)
            self.tapEncoderCallCount += 1
            if isCold {
                perfLog(String(format: "[TAP][AB] encoder cold start %.2f ms (res=%d, excluded)",
                             latencyMs, SAMConfiguration.encoderInputSize))
                return
            }
            self.tapEncoderTimesMs.append(latencyMs)
            let n = self.tapEncoderTimesMs.count
            if n >= 5 {
                let mean = self.tapEncoderTimesMs.reduce(0, +) / Double(n)
                let sorted = self.tapEncoderTimesMs.sorted()
                let p95 = sorted[max(0, Int(ceil(0.95 * Double(n))) - 1)]
                perfLog(String(format: "[TAP][AB] encoder stats res=%d (n=%d): mean=%.2f ms | p95=%.2f ms",
                             SAMConfiguration.encoderInputSize, n, mean, p95))
            }
        }
    }

    /// Double-tap → clear every tap instance (§3.2).
    func handleClearAllTapMasks() {
        guard currentMode == .tapToSegment else { return }
        // Consume a sequence number so a gap in `[TAP#N]` is attributable to a
        // clearAll rather than to a lost tap.
        stateLock.lock()
        tapGeneration += 1
        let clearedAt = tapGeneration
        stateLock.unlock()
        diagLog("[TAP#\(clearedAt)] pipeline received clearAll — clears \(tapInstances.count) instance(s) + loading indicator")
        discardAllTapWork(reason: "clearAll")
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.lastTapCanonicalPoint = nil
            self.lastTapViewPoint = nil
            self.tapFailure = nil
            self.lastTapIndex = clearedAt
            // Day 6: show "请点击分割" hint for 1.5 s after clearing all masks.
            self.showSegmentHint = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                self?.showSegmentHint = false
            }
        }
    }

    @objc private func handleOrientationChange() {
        // iOS 17+ 由 RotationCoordinator 的 KVO 驱动,通知路径只服务 <17 的 fallback。
        if #available(iOS 17.0, *), rotationCoordinator != nil { return }
        sessionQueue.async { [weak self] in self?.updateRotationLegacy() }
    }

    /// Legacy (<iOS 17) fallback,仅后摄语义。
    /// ⚠️ P1 根因:UIDeviceOrientation 与视频方向是【交叉】对应的 ——
    /// 设备向左转 (.landscapeLeft, Home 在右) 需要视频角度 0(即 video .landscapeRight),
    /// 反之亦然。旧实现按同名直觉映射,两个横屏分支恰好各差 180°。
    /// 返回 nil 表示 faceUp/faceDown/unknown → 保持上一个有效角度,不重置。
    private func desiredRotationAngleLegacy() -> CGFloat? {
        switch UIDevice.current.orientation {
        case .portrait:             return 90
        case .landscapeLeft:        return 0     // 旧值 180 ✗
        case .landscapeRight:       return 180   // 旧值 0 ✗
        case .portraitUpsideDown:   return 270
        default:                    return nil   // 平放/未知:保持 lastRotationAngle
        }
    }

    private func updateRotationLegacy() {
        guard let output = videoOutput,
              let connection = output.connection(with: .video) else { return }
        guard let angle = desiredRotationAngleLegacy() else { return }  // 平放:不动
        if #available(iOS 17.0, *) {
            if connection.isVideoRotationAngleSupported(angle) {
                connection.videoRotationAngle = angle
                lastRotationAngle = angle
            }
        } else {
            if connection.isVideoOrientationSupported {
                let orientation: AVCaptureVideoOrientation
                switch UIDevice.current.orientation {
                case .landscapeLeft:        orientation = .landscapeRight  // 交叉映射(见上)
                case .landscapeRight:       orientation = .landscapeLeft   // 交叉映射(见上)
                case .portraitUpsideDown:   orientation = .portraitUpsideDown
                default:                    orientation = .portrait
                }
                connection.videoOrientation = orientation
                lastRotationAngle = orientation == .portrait ? 90
                    : orientation == .portraitUpsideDown ? 270
                    : orientation == .landscapeLeft ? 180 : 0
            }
        }
        publishRotation()
    }

    private func publishRotation() {
        if debugSegmentation {
            diagLog(String(format: "[DBG] device rotation updated: %.0f", lastRotationAngle))
        }
        DispatchQueue.main.async { [weak self] in
            self?.maskRotationAngle = self?.lastRotationAngle ?? 0
        }
    }

    /// iOS 17+ 唯一角度来源。coordinator 与具体 device 绑定,
    /// 每次切换前/后摄都必须重建(见 toggleCamera)。
    @available(iOS 17.0, *)
    private func setupRotationCoordinator() {
        captureRotationObservation?.invalidate()
        previewRotationObservation?.invalidate()
        guard let device = activeDevice else { return }

        let coordinator = AVCaptureDevice.RotationCoordinator(device: device,
                                                              previewLayer: previewLayer)
        rotationCoordinator = coordinator   // 必须强持有,否则 KVO 随 coordinator 释放而失效

        // 1) buffer 输出角度(喂给 YOLO/SAM 的帧)—— 坐标链的根基。
        captureRotationObservation = coordinator.observe(
            \.videoRotationAngleForHorizonLevelCapture, options: [.initial, .new]
        ) { [weak self] coordinator, _ in
            let angle = coordinator.videoRotationAngleForHorizonLevelCapture
            self?.sessionQueue.async { [weak self] in
                self?.applyCaptureRotation(angle)
            }
        }

        // 2) 预览层角度 —— 只影响肉眼所见,不影响坐标链。
        previewRotationObservation = coordinator.observe(
            \.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]
        ) { [weak self] coordinator, _ in
            let angle = coordinator.videoRotationAngleForHorizonLevelPreview
            DispatchQueue.main.async { [weak self] in
                guard let conn = self?.previewLayer?.connection,
                      conn.isVideoRotationAngleSupported(angle) else { return }
                conn.videoRotationAngle = angle
            }
        }
    }

    @available(iOS 17.0, *)
    private func applyCaptureRotation(_ angle: CGFloat) {
        guard let output = videoOutput,
              let connection = output.connection(with: .video),
              connection.isVideoRotationAngleSupported(angle) else { return }
        connection.videoRotationAngle = angle
        // C4: rotation changes the coordinate space; clear stale tap masks immediately
        // rather than waiting for the next handleTap to detect geometry mismatch.
        if angle != lastRotationAngle && currentMode == .tapToSegment && !tapInstances.isEmpty {
            discardAllTapWork(reason: "rotation (C4)")
        }
        lastRotationAngle = angle
        publishRotation()
    }

    /// 由 CameraPreview 在创建 AVCaptureVideoPreviewLayer 后调用一次
    /// (例如 makeUIView 里:manager.attachPreviewLayer(layer))。
    /// 会重建 coordinator,让预览角度 KVO 接管 previewLayer.connection。
    func attachPreviewLayer(_ layer: AVCaptureVideoPreviewLayer) {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.previewLayer = layer
            if #available(iOS 17.0, *) {
                self.setupRotationCoordinator()
            }
        }
    }

    func toggleCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.currentPosition = (self.currentPosition == .back) ? .front : .back
            self.session.beginConfiguration()
            for input in self.session.inputs { self.session.removeInput(input) }
            if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: self.currentPosition),
               let input = try? AVCaptureDeviceInput(device: device),
               self.session.canAddInput(input) {
                self.session.addInput(input)
                self.activeDevice = device
            } else {
                faultLog("Camera input unavailable when toggling")
            }
            self.session.commitConfiguration()
            if #available(iOS 17.0, *) {
                // coordinator 与 device 绑定:切前/后摄必须重建。
                // 前摄传感器装配方向与后摄相反,coordinator 会自动给出
                // 横屏相差 180° 的正确角度 —— 这正是弃用手写映射表的原因。
                self.setupRotationCoordinator()
            } else {
                self.updateRotationLegacy()
            }
            if let output = self.videoOutput,
               let conn = output.connection(with: .video),
               conn.isVideoMirroringSupported {
                conn.automaticallyAdjustsVideoMirroring = false
                conn.isVideoMirrored = (self.currentPosition == .front)
            }
            DispatchQueue.main.async { [weak self] in
                self?.maskMirrored = (self?.currentPosition == .front)
            }
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .high
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: currentPosition),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            faultLog("Camera input unavailable")
            return
        }
        session.addInput(input)
        activeDevice = device

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            faultLog("Camera output unavailable")
            return
        }
        session.addOutput(output)
        videoOutput = output
        if #available(iOS 17.0, *) {
            setupRotationCoordinator()   // KVO .initial 会立即推一次正确角度
        } else {
            updateRotationLegacy()
        }
        if let conn = output.connection(with: .video), conn.isVideoMirroringSupported {
            conn.isVideoMirrored = (currentPosition == .front)
        }
        DispatchQueue.main.async { [weak self] in
            self?.maskMirrored = (self?.currentPosition == .front)
        }
        session.commitConfiguration()
        reloadModel()
    }

    private func reloadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = backend.computeUnits
            model = try yolov9_c(configuration: config)
            if let loaded = model {
                PerfLogger.logComputeUnits(config)
                PerfLogger.logInputInfo(loaded.model)
            }
        } catch {
            faultLog("Model load failed in CameraManager: \(error)")
        }
    }
}

// MARK: - Camera delegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        PerfLogger.logFpsAndMemoryEverySecond()
        guard model != nil else { return }
        if isProcessing { return }
        isProcessing = true
        defer { isProcessing = false }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        latestCameraBuffer = pixelBuffer

        // Requirement C: a warmup requested before any frame existed is armed,
        // not dropped — otherwise the 1283–8605 ms cold start silently slides
        // back onto the user's first tap (the failure mode G-3 describes).
        if warmupPending {
            warmupPending = false
            diagLog("[SAM] first frame arrived — running deferred warmup")
            warmupSegmentationIfPossible()
        }

        let tPreStart = PerfLogger.nowMs()
        guard let inputBuffer = letterboxToSquare(pixelBuffer: pixelBuffer, size: 640) else { return }
        lastPreprocessMs = PerfLogger.nowMs() - tPreStart
        latestInputBuffer = inputBuffer

        if debugSegmentation {
            debugFrameCount += 1
            if debugFrameCount % 30 == 0 {
                let camW = CVPixelBufferGetWidth(pixelBuffer)
                let camH = CVPixelBufferGetHeight(pixelBuffer)
                let inW  = CVPixelBufferGetWidth(inputBuffer)
                let inH  = CVPixelBufferGetHeight(inputBuffer)
                if let info = lastLetterbox {
                    diagLog(String(format: "[DBG] camera=%dx%d | modelInput=%dx%d | letterbox scale=%.4f padX=%.2f padY=%.2f | rot=%.0f mirrored=%@",
                                 camW, camH, inW, inH, info.scale, info.padX, info.padY,
                                 lastRotationAngle, (currentPosition == .front) ? "true" : "false"))
                } else {
                    diagLog(String(format: "[DBG] camera=%dx%d | modelInput=%dx%d | rot=%.0f mirrored=%@",
                                 camW, camH, inW, inH, lastRotationAngle, (currentPosition == .front) ? "true" : "false"))
                }
            }
        }
        runDetectionPipeline()
    }

    func runDetectionPipeline() {
        guard let model = model, let inputBuffer = latestInputBuffer else { return }

        // tapToSegment: YOLO output is discarded (boxes hidden) — it runs only
        // so FrameGeometry stays warm, which doesn't need every frame.  Run it
        // on every 3rd frame to free the ANE/CPU for SAM encode/decode; the
        // letterbox itself is still updated per frame in captureOutput, so tap
        // coordinate mapping is unaffected.
        if currentMode == .tapToSegment {
            tapModeFrameCount += 1
            if tapModeFrameCount % 3 != 0 {
                refreshTapEmbeddingIfNeeded()
                // §16.2.2 Step 4 — after the refresh, so it reads the freshest
                // `embeddingCache`.  Evaluated on the two-out-of-three frames
                // that skip YOLO as well: drift is a property of the frame, not
                // of whether this frame happened to run a detection.
                checkAndFireReAnchor()
                return
            }
        }

        let tInferStart = PerfLogger.nowMs()
        do {
            let output = try model.prediction(image: inputBuffer)
            let tInferEnd = PerfLogger.nowMs()
            let inferMs = tInferEnd - tInferStart

            PerfLogger.logOutputShapesOnce(var_3019: output.var_3019, var_3022: output.var_3022)

            inferenceTimesMs.append(inferMs)
            if inferenceTimesMs.count >= inferenceStatsWindow {
                let mean = inferenceTimesMs.reduce(0.0, +) / Double(inferenceTimesMs.count)
                let sorted = inferenceTimesMs.sorted()
                let p95 = sorted[max(0, Int(ceil(0.95 * Double(sorted.count))) - 1)]
                perfLog(String(format: "Inference time stats (n=%d): mean=%.2f ms | p95=%.2f ms",
                             sorted.count, mean, p95))
                inferenceTimesMs.removeAll(keepingCapacity: true)
            }

            let tPostStart = PerfLogger.nowMs()
            let detections = decodeDetections(from: output.var_3019, confidenceThreshold: 0.25)
            let topK = 10
            let topDetections = detections.sorted { $0.score > $1.score }.prefix(topK)
            let nmsDetections = classAwareNMS(Array(topDetections), iouThreshold: 0.45)

            let top5 = Array(nmsDetections.prefix(5))
            diagLog(String(format: "Frame inference time: %.2f ms | raw_in_boxes: %d | topK: %d | final_detections: %d",
                         inferMs, detections.count, min(topK, detections.count), nmsDetections.count))
            for (idx, det) in top5.enumerated() {
                diagLog(String(format: "det[%d]: class=%d score=%.3f box=[%.1f, %.1f, %.1f, %.1f]",
                             idx, det.classId, det.score, det.x1, det.y1, det.x2, det.y2))
            }

            // Quiet-mode stand-in for the per-frame `final_detections:` above.
            // Lands one frame-phase after `Inference time stats` (same frame,
            // just after post-processing), so the two read as adjacent lines.
            detectionCounts.append(nmsDetections.count)
            if detectionCounts.count >= inferenceStatsWindow {
                let detMean = Double(detectionCounts.reduce(0, +)) / Double(detectionCounts.count)
                quietSummaryLog(String(format: "Detection count stats (n=%d): mean=%.2f",
                                       detectionCounts.count, detMean))
                detectionCounts.removeAll(keepingCapacity: true)
            }

            // tapToSegment: YOLO 仅用于维持 FrameGeometry，不展示 bbox（architect §1.1）
            // detectionOnly / segmentation: 正常展示 bbox
            if currentMode == .tapToSegment {
                DispatchQueue.main.async { [weak self] in self?.boxes = [] }
            } else {
                let rects = top5.compactMap { mapToMetadataRect($0) }
                DispatchQueue.main.async { [weak self] in self?.boxes = rects }
            }

            // maskImage 管理规则：
            //   detectionOnly  → 每帧清除（无分割功能）
            //   segmentation   → 由 runSegmentationPipeline 驱动，不在此清除
            //   tapToSegment   → 由 tapDecodeWithPoint 驱动，绝对不能在此清除！
            //                  （旧错误：currentMode != .segmentation 把 tapToSegment 也清掉）
            if currentMode == .detectionOnly {
                DispatchQueue.main.async { [weak self] in
                    self?.maskImage = nil
                    self?.maskOutlines = nil
                }
            }
            if currentMode == .segmentation {
                runSegmentationPipeline(using: nmsDetections)
            }
            if currentMode == .tapToSegment {
                refreshTapEmbeddingIfNeeded()
                checkAndFireReAnchor()      // §16.2.2 Step 4
            }

            let postMs = PerfLogger.nowMs() - tPostStart
            PerfLogger.logTimings(preMs: lastPreprocessMs, inferMs: inferMs, postMs: postMs)
        } catch {
            faultLog("Frame inference failed: \(error)")
        }
    }
}

// MARK: - Segmentation pipeline
extension CameraManager {

    func runSegmentationPipeline(using detections: [Detection]) {
        frameIndex += 1
        let nowMs = PerfLogger.nowMs()

        guard let info = lastLetterbox else { return }

        // 1. Geometry change → full invalidation
        let geoSig = TemporalManager.GeometrySignature(
            origW: info.origW, origH: info.origH,
            scale: info.scale, padX: info.padX, padY: info.padY,
            rotation: lastRotationAngle,
            mirrored: (currentPosition == .front),
            inputSize: info.inputSize)
        if temporal.geometryChanged(geoSig) {
            invalidateEmbeddingAndMask(reason: "geometry change")
            temporal.resetPrimary()
        }

        // 2. Primary object selection (via TemporalManager)
        let prevPrimaryRect  = temporal.primaryRect
        let prevPrimaryScore = temporal.primaryScore
        let tmInputs = detections.map {
            TemporalManager.DetectionInput(x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2,
                                           score: $0.score, classId: $0.classId)
        }
        guard let selection = temporal.selectPrimary(
            from: tmInputs,
            toOrigRect: { [weak self] det in
                guard let self = self else { return nil }
                return self.detectionToOriginalRect(Detection(
                    x1: det.x1, y1: det.y1, x2: det.x2, y2: det.y2,
                    score: det.score, classId: det.classId))
            },
            origW: info.origW,
            origH: info.origH
        ) else { return }

        if selection.primaryChanged {
            invalidateEmbeddingAndMask(reason: "primary changed")
        }

        if debugSegmentation {
            diagLog(String(format: "[DET] primary box orig=%.1f,%.1f,%.1f,%.1f score=%.3f class=%d",
                         selection.rect.minX, selection.rect.minY,
                         selection.rect.maxX, selection.rect.maxY,
                         selection.detection.score, selection.detection.classId))
            diagLog(String(format: "[DET] letterbox scale=%.4f padX=%.2f padY=%.2f",
                         info.scale, info.padX, info.padY))
        }

        // 3. Drift classification (via TemporalManager)
        // Skip drift check when primary just changed — the embedding is already
        // invalidated and we must re-encode regardless; drift on a new object is meaningless.
        var needsEncoder = selection.primaryChanged
        var needsImmediateDecode = false
        if !selection.primaryChanged, let prev = prevPrimaryRect, prev != selection.rect {
            let drift = temporal.classifyDrift(
                prev: prev, current: selection.rect,
                prevScore: prevPrimaryScore, currentScore: selection.detection.score,
                origW: info.origW, origH: info.origH,
                debugEnabled: debugSegmentation)
            switch drift {
            case .heavyDrift:
                // Large jump — embedding unreliable, must re-encode and clear mask.
                temporal.invalidateMask()
                if debugSegmentation { diagLog("[SEG] invalidate mask: heavy drift") }
                needsEncoder = true
            case .lightDrift:
                // Moderate shift — embedding still valid, prompt box changed.
                // Keep existing mask as visual fallback; force a decoder run this frame.
                needsImmediateDecode = true
                if debugSegmentation { diagLog("[SEG] schedule re-decode: light drift (mask retained)") }
            case .noDrift:
                break
            }
        }

        // 4. Embedding validity check
        stateLock.lock()
        let embeddingEntry = embeddingCache.map {
            EmbeddingEntry(embedding: $0.embedding, timestampMs: $0.timestampMs)
        }
        let hasValidEmbedding = temporal.isEmbeddingValid(entry: embeddingEntry, nowMs: nowMs)
        let currentlyEncoding = isEncoding
        stateLock.unlock()

        let embeddingAgeMs = embeddingEntry.map { nowMs - $0.timestampMs } ?? .infinity
        let needsRencode = !hasValidEmbedding || needsEncoder || embeddingAgeMs > encoderRefreshMs
        if hasValidEmbedding { encoderHitCount += 1 } else { encoderMissCount += 1 }
        logEncoderCacheStatsIfNeeded()
        if needsRencode { scheduleEncoder(cameraBuffer: latestCameraBuffer) }

        // 5. Fallback if no embedding yet
        if currentlyEncoding && !hasValidEmbedding {
            if debugSegmentation { diagLog("[SEG] fallback: bbox-only (encoding in progress, no valid embedding)") }
            return
        }

        stateLock.lock()
        let cachedEmbedding = embeddingCache?.embedding
        stateLock.unlock()
        guard let embedding = cachedEmbedding else { return }

        // 6. Expire stale mask
        if !temporal.isMaskValid(nowMs: nowMs) {
            temporal.invalidateMask()
        }

        // 7. Decoder cadence
        // Run decoder when:
        //   a) cadence tick
        //   b) encoder just fired (embedding is fresh)
        //   c) light drift — prompt box changed, run decoder immediately regardless of cadence
        //   d) no mask at all
        let maskValid = (temporal.maskCache != nil)
        let shouldDecodeByCadence = (nowMs - lastSegDecodeMs) > decoderRefreshMs
        if maskValid && !shouldDecodeByCadence && !needsEncoder && !needsImmediateDecode { return }
        lastSegDecodeMs = nowMs

        stateLock.lock()
        guard !isDecoding else { stateLock.unlock(); return }
        isDecoding = true
        stateLock.unlock()

        let selectedRect    = selection.rect
        let capturedInfo    = info
        let rotAngle        = lastRotationAngle
        let isFront         = (currentPosition == .front)
        let capturedBackend = backend

        decoderQueue.async { [weak self] in
            guard let self = self else { return }

            // Model load happens on decoderQueue — samDecoder is single-queue owned.
            guard let decoder = self.decoderForQueue(computeUnits: capturedBackend.computeUnits) else {
                self.stateLock.lock(); self.isDecoding = false; self.stateLock.unlock()
                return
            }

            guard let prompt = PromptBuilder.buildBoxPrompt(
                x1: Float(selectedRect.minX), y1: Float(selectedRect.minY),
                x2: Float(selectedRect.maxX), y2: Float(selectedRect.maxY),
                origW: capturedInfo.origW, origH: capturedInfo.origH,
                inputSize: 1024
            ) else {
                self.stateLock.lock(); self.isDecoding = false; self.stateLock.unlock()
                faultLog("SAM decoder: prompt build failed — isDecoding reset")
                return
            }

            let t0 = PerfLogger.nowMs()
            guard let mask = decoder.decode(embedding: embedding, prompt: prompt) else {
                self.stateLock.lock(); self.isDecoding = false; self.stateLock.unlock()
                faultLog("SAM decoder failed to produce mask — isDecoding reset")
                return
            }
            let t1 = PerfLogger.nowMs()
            let latency = t1 - t0

            // Record mask in TemporalManager — also updates lastMaskRefreshMs.
            let refreshInterval = self.temporal.recordMask(mask, timestampMs: t1)
            self.decoderCount += 1

            if let rendered = self.maskRenderer.renderMask(
                lowResMask: mask,
                origW: Int(capturedInfo.origW),
                origH: Int(capturedInfo.origH),
                box: selectedRect) {
                DispatchQueue.main.async {
                    self.maskImage         = rendered.image
                    self.maskOutlines      = nil   // Phase 2 box path: no stroke (§3.4 is Phase 3 only)
                    self.maskRotationAngle = rotAngle
                    self.maskMirrored      = isFront
                }
            }

            // SAM decoder sliding-window stats (Task 1)
            let isFirstDecodeCall = (self.samDecoderCallCount == 0)
            self.samDecoderCallCount += 1
            if isFirstDecodeCall {
                // First run contains ANE compilation; exclude from stats.
                // Kept in quiet mode (unlike the per-decode lines below): it
                // fires once per session and is the ANE cold-start measurement,
                // mirroring the encoder cold-start line.
                if refreshInterval > 0 {
                    perfLog(String(format: "SAM decoder latency: %.2f ms (cold start — excluded from stats) | mask refresh: %.2f ms (%.2f Hz)",
                                 latency, refreshInterval, 1000.0 / refreshInterval))
                } else {
                    perfLog(String(format: "SAM decoder latency: %.2f ms (cold start — excluded from stats)", latency))
                }
            } else {
                if refreshInterval > 0 {
                    diagLog(String(format: "SAM decoder latency: %.2f ms | mask refresh: %.2f ms (%.2f Hz)",
                                 latency, refreshInterval, 1000.0 / refreshInterval))
                } else {
                    diagLog(String(format: "SAM decoder latency: %.2f ms", latency))
                }
                self.samDecoderTimesMs.append(latency)
                if self.samDecoderTimesMs.count >= self.samDecoderStatsWindow {
                    let mean = self.samDecoderTimesMs.reduce(0.0, +) / Double(self.samDecoderTimesMs.count)
                    let sorted = self.samDecoderTimesMs.sorted()
                    let p95 = sorted[max(0, Int(ceil(0.95 * Double(sorted.count))) - 1)]
                    perfLog(String(format: "SAM decoder stats (n=%d): mean=%.2f ms | p95=%.2f ms",
                                 self.samDecoderTimesMs.count, mean, p95))
                    self.samDecoderTimesMs.removeAll(keepingCapacity: true)
                }
            }

            self.stateLock.lock(); self.isDecoding = false; self.stateLock.unlock()
        }
    }

    /// Dispatches an encode task to encoderQueue.
    /// If an encode is already running the request is dropped — the in-flight
    /// encode will produce a fresh embedding for subsequent frames.
    ///
    /// D-1: the drop used to be *silent*.  Unlike the two tap-path exits this one
    /// is frequent and by design (Phase 2 asks every frame while an encode takes
    /// ~1 s), so printing each occurrence would flood the console and land inside
    /// the `Post=` measurement window.  It is counted instead and reported every
    /// 30 drops, which is enough to tell "the encoder is saturated, as expected"
    /// apart from "the slot is stuck and never released".
    private func scheduleEncoder(cameraBuffer: CVPixelBuffer?) {
        guard let buffer = cameraBuffer else { return }

        stateLock.lock()
        guard !isEncoding else {
            encoderSlotBusyDropCount += 1
            let n = encoderSlotBusyDropCount
            stateLock.unlock()
            if n % 30 == 0 {
                diagLog("[SAM] encode request dropped — encoder busy (\(n) so far; expected while an encode is in flight)")
            }
            return
        }
        isEncoding = true
        encodeSlotOwner = .segmentationFrame
        stateLock.unlock()

        // NOTE: encoderMissCount already incremented by runSegmentationPipeline; don't double-count here.

        // Capture backend before leaving videoQueue.
        let capturedBackend = backend

        encoderQueue.async { [weak self] in
            guard let self = self else {
                return
            }
            // Guard: GPU/ANE work is aborted by iOS when the app enters the background,
            // producing "kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted".
            // Skip the encode and release the slot; TemporalManager will re-trigger next frame.
            guard !self.isAppBackgrounded else {
                if self.debugSegmentation { faultLog("[SAM] encode skipped: app is in background") }
                self.releaseEncodeSlot()
                self.encodeSlotDidFinish(originTag: "segmentation-backgrounded")
                return
            }

            // Model load happens INSIDE encoderQueue — keeps videoQueue unblocked
            // and keeps `samEncoder` single-queue owned.
            let t0 = PerfLogger.nowMs()
            if let embedding = self.encodeChecked(buffer: buffer,
                                                 computeUnits: capturedBackend.computeUnits,
                                                 tag: "segmentation encode") {
                let t1 = PerfLogger.nowMs()
                self.stateLock.lock()
                self.embeddingCache = (embedding: embedding, timestampMs: t1)
                self.embeddingGeneration &+= 1                  // RE-1 (§18.1.4)
                self.isEncoding = false
                self.encodeSlotOwner = nil
                self.stateLock.unlock()
                // SAM encoder sliding-window stats (Task 1)
            let latencyMs = t1 - t0
            let isFirstEncoderCall = (self.samEncoderCallCount == 0)
            self.samEncoderCallCount += 1
            if isFirstEncoderCall {
                // First run contains ANE compilation; exclude from stats.
                perfLog(String(format: "SAM encoder latency: %.2f ms (cold start — excluded from stats)", latencyMs))
            } else {
                self.samEncoderTimesMs.append(latencyMs)
                if self.samEncoderTimesMs.count >= self.samEncoderStatsWindow {
                    let mean = self.samEncoderTimesMs.reduce(0.0, +) / Double(self.samEncoderTimesMs.count)
                    let sorted = self.samEncoderTimesMs.sorted()
                    let p95 = sorted[max(0, Int(ceil(0.95 * Double(sorted.count))) - 1)]
                    perfLog(String(format: "SAM encoder stats (n=%d): mean=%.2f ms | p95=%.2f ms",
                                 self.samEncoderTimesMs.count, mean, p95))
                    self.samEncoderTimesMs.removeAll(keepingCapacity: true)
                } else {
                    diagLog(String(format: "SAM encoder latency: %.2f ms", latencyMs))
                }
            }
                // D-14: Phase 2's per-frame encode is an encode like any other —
                // it publishes into `embeddingCache` and releases the same slot,
                // so a warmup parked behind it must be woken here too, or
                // entering `.segmentation` while this path owns the slot would
                // leave the warmup waiting forever.  `drainPendingTaps` inside
                // the hook is a no-op in this mode (mode switches discard every
                // parked tap), and a strict improvement if one ever survived.
                self.encodeSlotDidFinish(originTag: "segmentation-frame")
            } else {
                // encode() returned nil — CoreML error already logged inside SAMEncoder.
                // Most common cause: app went to background mid-inference (GPU abort).
                // Release the encoding slot so the pipeline can retry on next active frame.
                faultLog("[SAM] encode returned nil — releasing encoder slot (background GPU abort?)")
                self.releaseEncodeSlot()
                self.encodeSlotDidFinish(originTag: "segmentation-failed")
            }
        }
    }

    private func logEncoderCacheStatsIfNeeded() {
        let total = encoderHitCount + encoderMissCount
        guard total > 0, total % 30 == 0 else { return }
        let hitRate = Double(encoderHitCount) / Double(total) * 100.0
        diagLog(String(format: "SAM encoder cache hit rate: %.1f%% (%d/%d)",
                     hitRate, encoderHitCount, total))
    }

    private func invalidateEmbeddingAndMask(reason: String) {
        stateLock.lock()
        embeddingCache = nil
        stateLock.unlock()
        temporal.invalidateMask()
        if debugSegmentation { diagLog("[SEG] invalidate embedding+mask: \(reason)") }
    }

    // MARK: Geometry helpers

    private func detectionToOriginalRect(_ det: Detection) -> CGRect? {
        guard let info = lastLetterbox else { return nil }
        let x1 = (det.x1 - info.padX) / info.scale
        let y1 = (det.y1 - info.padY) / info.scale
        let x2 = (det.x2 - info.padX) / info.scale
        let y2 = (det.y2 - info.padY) / info.scale
        let bx = max(0, min(info.origW, x1))
        let by = max(0, min(info.origH, y1))
        let bw = max(0, min(info.origW, x2)) - bx
        let bh = max(0, min(info.origH, y2)) - by
        if bw <= 1 || bh <= 1 { return nil }
        return CGRect(x: CGFloat(bx), y: CGFloat(by), width: CGFloat(bw), height: CGFloat(bh))
    }

    private func mapToMetadataRect(_ det: Detection) -> CGRect? {
        guard let info = lastLetterbox else { return nil }
        let x1 = (det.x1 - info.padX) / info.scale
        let y1 = (det.y1 - info.padY) / info.scale
        let x2 = (det.x2 - info.padX) / info.scale
        let y2 = (det.y2 - info.padY) / info.scale
        let bx = max(0, min(info.origW, x1))
        let by = max(0, min(info.origH, y1))
        let bw = max(0, min(info.origW, x2)) - bx
        let bh = max(0, min(info.origH, y2)) - by
        if bw <= 1 || bh <= 1 { return nil }

        var rect: CGRect
        switch Int(lastRotationAngle) {
        case 90:
            let normX = by / info.origH
            let normY = (info.origW - bx - bw) / info.origW
            let normW = bh / info.origH
            let normH = bw / info.origW
            rect = CGRect(x: CGFloat(normX), y: CGFloat(normY), width: CGFloat(normW), height: CGFloat(normH))
        case 180:
            rect = CGRect(x: CGFloat((info.origW - bx - bw) / info.origW),
                          y: CGFloat((info.origH - by - bh) / info.origH),
                          width: CGFloat(bw / info.origW),
                          height: CGFloat(bh / info.origH))
        case 270:
            let normX = (info.origH - by - bh) / info.origH
            let normY = bx / info.origW
            let normW = bh / info.origH
            let normH = bw / info.origW
            rect = CGRect(x: CGFloat(normX), y: CGFloat(normY), width: CGFloat(normW), height: CGFloat(normH))
        default:
            rect = CGRect(x: CGFloat(bx / info.origW), y: CGFloat(by / info.origH),
                          width: CGFloat(bw / info.origW), height: CGFloat(bh / info.origH))
        }
        if currentPosition == .front {
            rect = CGRect(x: 1.0 - rect.origin.x - rect.size.width,
                          y: rect.origin.y,
                          width: rect.size.width, height: rect.size.height)
        }
        return rect
    }
}

// MARK: - Detection decode (optimised)
extension CameraManager {

    /// Decodes YOLO output [1, 84, 8400] into Detection structs.
    ///
    /// **Optimisations:**
    /// 1. Transpose src[84×8400] → transposeBuffer[8400×84] using vDSP_mtrans (SIMD,
    ///    ~3 ms) into a pre-allocated instance buffer — zero per-frame heap allocation.
    /// 2. Single sigmoid call per location (argmax on raw logits, sigmoid only on winner).
    /// Expected Post: ~8 ms.
    private func decodeDetections(from multiArray: MLMultiArray,
                                   confidenceThreshold: Float) -> [Detection] {
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count == 3, shape[1] == 84 else { return [] }
        let L = shape[2]   // 8400 locations
        let C = shape[1]   // 84 channels
        let strides = multiArray.strides.map { $0.intValue }
        let sC = strides[1]
        let sI = strides[2]
        guard multiArray.dataType == .float32 else { return [] }

        // ── Transpose into pre-allocated buffer using vDSP_mtrans ───────────────
        // vDSP_mtrans requires contiguous row-major input; if strides are unit we
        // can call it directly, otherwise fall back to scalar copy.
        // The model output is read through withUnsafeBufferPointer — the raw
        // dataPointer of an IOSurface-backed output is not safe to hold.
        transposeBuffer.withUnsafeMutableBufferPointer { dstBuf in
            guard let dst = dstBuf.baseAddress else { return }
            multiArray.withUnsafeBufferPointer(ofType: Float.self) { srcBuf in
                guard let src = srcBuf.baseAddress else { return }
                if sC == L && sI == 1 {
                    // Contiguous [C × L] → [L × C]: one vDSP call (~0.5 ms for 84×8400)
                    vDSP_mtrans(src, 1, dst, 1, vDSP_Length(L), vDSP_Length(C))
                } else {
                    // Fallback scalar path (rare)
                    for c in 0..<C {
                        let srcRow = src + c * sC
                        for i in 0..<L { dst[i * C + c] = srcRow[i * sI] }
                    }
                }
            }
        }

        let logitThreshold: Float = log(confidenceThreshold / (1.0 - confidenceThreshold))

        var detections = [Detection]()
        detections.reserveCapacity(256)

        for i in 0..<L {
            let base = i * C

            // Argmax on class logits (ch 4…83) — sequential read in transposed buffer.
            var bestRaw: Float = -Float.greatestFiniteMagnitude
            var bestClass = -1
            for c in 4..<C {
                let raw = transposeBuffer[base + c]
                if raw > bestRaw { bestRaw = raw; bestClass = c - 4 }
            }
            if bestRaw < logitThreshold { continue }

            let bestScore = sigmoid(bestRaw)
            if bestScore < confidenceThreshold { continue }

            var cx = transposeBuffer[base + 0]
            var cy = transposeBuffer[base + 1]
            var w  = transposeBuffer[base + 2]
            var h  = transposeBuffer[base + 3]

            if cx >= -0.1 && cx <= 1.5 { cx *= 640; cy *= 640; w *= 640; h *= 640 }

            let x1 = max(0, min(640, cx - w / 2))
            let y1 = max(0, min(640, cy - h / 2))
            let x2 = max(0, min(640, cx + w / 2))
            let y2 = max(0, min(640, cy + h / 2))
            if x2 - x1 <= 1 || y2 - y1 <= 1 { continue }

            detections.append(Detection(x1: x1, y1: y1, x2: x2, y2: y2,
                                        score: bestScore, classId: bestClass))
        }
        return detections
    }

    private func classAwareNMS(_ detections: [Detection], iouThreshold: Float) -> [Detection] {
        var grouped = [Int: [Detection]]()
        for det in detections { grouped[det.classId, default: []].append(det) }
        var results = [Detection]()
        for (_, dets) in grouped {
            let sorted = dets.sorted { $0.score > $1.score }
            var kept = [Detection]()
            for det in sorted {
                if !kept.contains(where: { iou(det, $0) > iouThreshold }) { kept.append(det) }
            }
            results.append(contentsOf: kept)
        }
        return results
    }

    private func iou(_ a: Detection, _ b: Detection) -> Float {
        let xA = max(a.x1, b.x1); let yA = max(a.y1, b.y1)
        let xB = min(a.x2, b.x2); let yB = min(a.y2, b.y2)
        let inter = max(0, xB - xA) * max(0, yB - yA)
        let union = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter
        return union <= 0 ? 0 : inter / union
    }

    @inline(__always) private func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + exp(-x))
    }

    // MARK: Letterbox

    private func letterboxToSquare(pixelBuffer: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let width = image.extent.width, height = image.extent.height
        let scale = min(CGFloat(size) / width, CGFloat(size) / height)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let x = (CGFloat(size) - scaled.extent.width) / 2.0
        let y = (CGFloat(size) - scaled.extent.height) / 2.0
        let translated = scaled.transformed(by: CGAffineTransform(translationX: x, y: y))

        let info = LetterboxInfo(
            origW: Float(width), origH: Float(height),
            scale: Float(scale), padX: Float(x), padY: Float(y),
            inputSize: Float(size))
        lastLetterbox = info
        // Requirement A: mirror it under stateLock so the tap fast path (gesture
        // thread) can read geometry without a videoQueue hop and without racing.
        publishTapGeometry(info)

        // Reuse pre-allocated output buffer; allocate once on first call.
        if letterboxOutputBuffer == nil {
            let outputAttrs: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: size,
                kCVPixelBufferHeightKey: size,
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true
            ]
            var newBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                                            kCVPixelFormatType_32BGRA,
                                            outputAttrs as CFDictionary, &newBuffer)
            guard status == kCVReturnSuccess else { return nil }
            letterboxOutputBuffer = newBuffer
        }
        guard let buffer = letterboxOutputBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        let background = CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
        let composed = translated.composited(over: background)
        ciContext.render(composed, to: buffer)
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }
}
