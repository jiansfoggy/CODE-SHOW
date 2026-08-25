import AVFoundation
import SwiftUI
import UIKit

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let boxes: [CGRect]
    let maskImage: CGImage?
    let maskRotationAngle: CGFloat
    let maskMirrored: Bool
    /// Phase 3: manager reference used to install the Tap-to-Segment gesture layer.
    let cameraManager: CameraManager

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill

        // Phase 3 Day 2 — install TouchHandler once. Callbacks route to CameraManager,
        // which gates on `.tapToSegment` mode. Detection/Segmentation modes ignore taps.
        let handler = TouchHandler(
            view: view,
            previewLayer: view.videoPreviewLayer,
            geometryProvider: { [weak cameraManager] in cameraManager?.currentFrameGeometry() }
        )
        handler.onTap = { [weak cameraManager] canonicalPoint in
            cameraManager?.handleTap(canonicalPoint: canonicalPoint)
        }
        handler.onClearAll = { [weak cameraManager] in
            cameraManager?.handleClearAllTapMasks()
        }
        view.touchHandler = handler   // retain for the lifetime of the view
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.videoPreviewLayer.session = session
        uiView.updateBoxes(boxes)
        uiView.updateMask(image: maskImage, rotationAngle: maskRotationAngle, mirrored: maskMirrored)
//        new added
        if let conn = uiView.videoPreviewLayer.connection {
            if #available(iOS 17.0, *) {
                if conn.isVideoRotationAngleSupported(maskRotationAngle) {
                    conn.videoRotationAngle = maskRotationAngle
                }
            }
            if conn.isVideoMirroringSupported {
                conn.automaticallyAdjustsVideoMirroring = false
                conn.isVideoMirrored = maskMirrored
            }
        }
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    private let overlayLayer = CAShapeLayer()
    private let maskLayer = CALayer()

    /// Phase 3: retains the Tap-to-Segment gesture handler for the view's lifetime.
    var touchHandler: TouchHandler?

    private var lastBounds: CGRect = .zero
    private var debugMaskCount: Int = 0

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupOverlay()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupOverlay()
    }

    private func setupOverlay() {
        maskLayer.contentsGravity = .resizeAspectFill
        maskLayer.opacity = 1.0
        layer.addSublayer(maskLayer)

        overlayLayer.strokeColor = UIColor.green.cgColor
        overlayLayer.lineWidth = 2.0
        overlayLayer.fillColor = UIColor.clear.cgColor
        layer.addSublayer(overlayLayer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        overlayLayer.frame = bounds

        // maskLayer must always fill the full PreviewView bounds (same as overlayLayer).
        // Do NOT use layerRectConverted(fromMetadataOutputRect:) here — that converts
        // a normalized video-coordinate rect into the layer's coordinate system, which
        // for resizeAspectFill produces a frame that extends *outside* the visible area
        // (negative y when the video is letterboxed), causing the mask to be clipped.
        // The mask CGImage was rendered to the full original-frame dimensions, so
        // resizeAspectFill gravity aligns it with the video preview automatically.
        maskLayer.frame = bounds

        if bounds != lastBounds {
            lastBounds = bounds
            print("[DBG] preview bounds=\(bounds.size), gravity=\(videoPreviewLayer.videoGravity)")
        }
    }

    func updateBoxes(_ boxes: [CGRect]) {
        let path = UIBezierPath()
        for box in boxes {
            let rect = videoPreviewLayer.layerRectConverted(fromMetadataOutputRect: box)
            path.append(UIBezierPath(rect: rect))
        }
        overlayLayer.path = path.cgPath
    }

    func updateMask(image: CGImage?, rotationAngle: CGFloat, mirrored: Bool) {
        guard let image = image else {
            maskLayer.isHidden = true
            maskLayer.contents = nil
            return
        }

        // maskLayer.frame is permanently set to `bounds` in layoutSubviews.
        // Do NOT override it here with layerRectConverted(fromMetadataOutputRect:):
        // that API maps a normalized video-coordinate rect into the layer coordinate
        // system and for resizeAspectFill returns a rect that extends outside the view
        // (negative y ≈ −44 pt on iPhone 11 portrait), clipping the top of the mask.
        //
        // The mask CGImage is rendered at the original camera frame dimensions
        // (e.g. 1080 × 1920). With contentsGravity = .resizeAspectFill and
        // frame = bounds, CoreAnimation scales it identically to the video preview,
        // so alignment is automatic — no extra frame or position adjustment needed.
        //
        // rotationAngle / mirrored are already baked into the CGImage by the time
        // renderMask() returns (AVCaptureConnection.videoRotationAngle rotates the
        // pixel buffer before it reaches the encoder, so origW/origH are already in
        // the display orientation). No CATransform3D needed here.
        maskLayer.contents = image
        maskLayer.isHidden = false

        debugMaskCount += 1
        if debugMaskCount % 30 == 0 {
            print(String(format: "[DBG] mask layer frame=%@ | bounds=%@ | rot=%.0f mirrored=%@",
                         NSCoder.string(for: maskLayer.frame),
                         NSCoder.string(for: bounds),
                         rotationAngle, mirrored ? "true" : "false"))
        }
    }
}
