//
//  TouchHandler.swift
//  JudgeE2
//
//  Phase 3 — Day 2 (Builder)
//
//  Interaction layer for Tap-to-Segment. Installs a single-tap + double-tap
//  gesture recognizer on the preview view and converts each tap to a Canonical
//  pixel coordinate via the one approved path: FrameGeometry.invertViewPoint
//  (see shared/architect_output.md §2). This module performs NO coordinate math
//  of its own — it only routes taps through FrameGeometry and emits callbacks.
//
//  Z-order contract (architect_output.md §1.3): the gesture layer sits above the
//  mask renderer but uses its own recognizer instances and must not steal touches
//  from other layers. We attach recognizers to the preview view; single-tap waits
//  for double-tap to fail so "clear all" (double-tap) and "segment" (single-tap)
//  do not collide.
//

import AVFoundation
import UIKit

final class TouchHandler: NSObject {

    /// Fired on a single tap, carrying the Canonical-pixel coordinate and the
    /// raw view-space point (used to anchor the loading indicator in the UI).
    ///
    /// Returns the tap sequence number the pipeline assigned (0 = tap not
    /// accepted).  The counter is owned by CameraManager under its state lock —
    /// TouchHandler must not keep a second one — so the contract log below stays
    /// here (this is where the geometry data lives) and only borrows the index.
    var onTap: ((_ canonical: CGPoint, _ viewPoint: CGPoint) -> Int)?

    /// Fired on a double tap → clear all active tap-mask instances.
    var onClearAll: (() -> Void)?

    /// Phase 4B Day 5 (§22.2) — fired on a long-press (≥ 0.5 s), carrying the
    /// same Canonical-pixel coordinate and view-space point as `onTap`.  Pin
    /// creation's entry point; `CameraManager.handleLongPress` decides what
    /// happens (open `PinCreationSheet` on hit, no-op on miss — §22.2.2).
    var onLongPress: ((_ canonical: CGPoint, _ viewPoint: CGPoint) -> Void)?

    /// Supplies the current frame geometry snapshot at tap time.
    /// Returning nil (e.g. geometry not ready) causes the tap to be ignored.
    private let geometryProvider: () -> FrameGeometry?

    private weak var targetView: UIView?
    private weak var previewLayer: AVCaptureVideoPreviewLayer?

    private lazy var singleTap: UITapGestureRecognizer = {
        let g = UITapGestureRecognizer(target: self, action: #selector(handleSingleTap(_:)))
        g.numberOfTapsRequired = 1
        g.delegate = self
        return g
    }()

    private lazy var doubleTap: UITapGestureRecognizer = {
        let g = UITapGestureRecognizer(target: self, action: #selector(handleDoubleTap(_:)))
        g.numberOfTapsRequired = 2
        g.delegate = self
        return g
    }()

    /// Phase 4B Day 5 (§22.2.1) — long-press for Pin creation.  ≥ 0.5 s is a
    /// standard-length system gesture; deliberately no `require(toFail:)`
    /// relationship with `singleTap`/`doubleTap` — the two families are
    /// mutually exclusive by construction (a release before 0.5 s never
    /// transitions this recognizer to `.began`, so it simply never fires and
    /// the tap recognizers see the touch normally; §22.2.1 explains why no
    /// additional priority rule is needed or wanted here).
    private lazy var longPress: UILongPressGestureRecognizer = {
        let g = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPress(_:)))
        g.minimumPressDuration = 0.5
        g.delegate = self
        return g
    }()

    /// - Parameters:
    ///   - view: the preview view that hosts the AVCaptureVideoPreviewLayer.
    ///   - previewLayer: the layer used for captureDevicePointConverted inversion.
    ///   - geometryProvider: returns the live FrameGeometry (origW/origH/mirror).
    init(view: UIView,
         previewLayer: AVCaptureVideoPreviewLayer,
         geometryProvider: @escaping () -> FrameGeometry?) {
        self.targetView = view
        self.previewLayer = previewLayer
        self.geometryProvider = geometryProvider
        super.init()

        // Single-tap must wait for a possible double-tap to fail.
        singleTap.require(toFail: doubleTap)
        view.isUserInteractionEnabled = true
        view.addGestureRecognizer(singleTap)
        view.addGestureRecognizer(doubleTap)
        view.addGestureRecognizer(longPress)
    }

    @objc private func handleSingleTap(_ gesture: UITapGestureRecognizer) {
        guard let view = targetView,
              let previewLayer = previewLayer,
              let geometry = geometryProvider() else {
            faultLog("[TAP] ignored — geometry not ready")
            return
        }

        let viewPoint = gesture.location(in: view)
        let result = geometry.invertViewPoint(viewPoint,
                                              viewSize: view.bounds.size,
                                              previewLayer: previewLayer)

        // Dispatch first so the returned sequence number can stamp the log line;
        // handleTap only allocates the index and enqueues, so this is cheap.
        let gen = onTap?(result.canonical, viewPoint) ?? 0
        // Contract-required log (architect_output.md §2.2), stamped `[TAP#N]` so
        // it greps out with the rest of that tap's chain.  gen == 0 means the
        // pipeline rejected the tap (not in .tapToSegment) — no chain follows.
        let tag = gen > 0 ? "[TAP#\(gen)]" : "[TAP][not-accepted]"
        diagLog(String(format: "%@ view=(%.1f,%.1f) normalized=(%.4f,%.4f) canonical=(%.1f,%.1f) orientation=%.0f mirrored=%@",
                     tag,
                     viewPoint.x, viewPoint.y,
                     result.normalized.x, result.normalized.y,
                     result.canonical.x, result.canonical.y,
                     geometry.rotation,
                     geometry.mirrored ? "true" : "false"))
    }

    /// Phase 4B Day 5 (§22.2) — long-press → Pin creation channel.  Only the
    /// `.began` transition is acted on (fired once per press, the instant
    /// `minimumPressDuration` elapses) — `.changed`/`.ended`/`.cancelled` carry
    /// finger movement and lift-off, neither of which this feature needs.
    @objc private func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
        guard gesture.state == .began else { return }
        guard let view = targetView,
              let previewLayer = previewLayer,
              let geometry = geometryProvider() else {
            faultLog("[PIN] long-press ignored — geometry not ready")
            return
        }

        let viewPoint = gesture.location(in: view)
        let result = geometry.invertViewPoint(viewPoint,
                                              viewSize: view.bounds.size,
                                              previewLayer: previewLayer)
        onLongPress?(result.canonical, viewPoint)
    }

    @objc private func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
        diagLog("[TAP] double-tap → clearAll")
        // Day 6 — haptic feedback so the user gets tactile confirmation that
        // all masks were cleared (architect_output §10.4 clearAll requirement).
        let haptic = UIImpactFeedbackGenerator(style: .heavy)
        haptic.prepare()
        haptic.impactOccurred()
        onClearAll?()
    }
}

// MARK: - UIGestureRecognizerDelegate
extension TouchHandler: UIGestureRecognizerDelegate {
    // Do not interfere with other recognizers / rendering layers (Z-order contract).
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
        return true
    }
}
