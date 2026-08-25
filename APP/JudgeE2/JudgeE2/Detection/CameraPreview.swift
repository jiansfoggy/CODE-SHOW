import AVFoundation
import SwiftUI
import UIKit

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let boxes: [CGRect]
    let maskImage: CGImage?
    /// Phase 3 Day 5 — vector outlines for the tap masks (§3.4).  Travels
    /// separately from `maskImage` because constraint C-5 forbids baking the
    /// stroke into the 256×256 alpha tile; the width has to stay in pt.
    let maskOutlines: MaskOutlineSet?
    let maskRotationAngle: CGFloat
    let maskMirrored: Bool
    /// Phase 3: manager reference used to install the Tap-to-Segment gesture layer.
    let cameraManager: CameraManager

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill

        // P1 旋转修复:把 previewLayer 交给 CameraManager,由
        // AVCaptureDevice.RotationCoordinator 的 KVO 统一驱动预览角度
        // (videoRotationAngleForHorizonLevelPreview)。预览角度不要在
        // updateUIView 里手动设置 —— 见下方注释。
        cameraManager.attachPreviewLayer(view.videoPreviewLayer)

        // Phase 3 Day 2 — install TouchHandler once. Callbacks route to CameraManager,
        // which gates on `.tapToSegment` mode. Detection/Segmentation modes ignore taps.
        let handler = TouchHandler(
            view: view,
            previewLayer: view.videoPreviewLayer,
            geometryProvider: { [weak cameraManager] in cameraManager?.currentFrameGeometry() }
        )
        handler.onTap = { [weak cameraManager] canonicalPoint, viewPoint in
            // Returns the tap sequence number (0 = not accepted) for log stamping.
            cameraManager?.handleTap(canonicalPoint: canonicalPoint, viewPoint: viewPoint) ?? 0
        }
        handler.onClearAll = { [weak cameraManager] in
            cameraManager?.handleClearAllTapMasks()
        }
        // Phase 4B Day 5 — long-press → Pin creation channel (§22.2).
        handler.onLongPress = { [weak cameraManager] canonicalPoint, viewPoint in
            cameraManager?.handleLongPress(canonicalPoint: canonicalPoint, viewPoint: viewPoint)
        }
        view.touchHandler = handler   // retain for the lifetime of the view
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        if uiView.videoPreviewLayer.session !== session {
            uiView.videoPreviewLayer.session = session
        }
        uiView.updateBoxes(boxes)
        uiView.updateMask(image: maskImage, rotationAngle: maskRotationAngle, mirrored: maskMirrored)
        uiView.updateMaskOutlines(maskOutlines)
        // ⚠️ 不要在这里设置 videoPreviewLayer.connection 的旋转或镜像:
        // 1. maskRotationAngle 是 capture(buffer)角度,不是 preview 角度 ——
        //    两者仅在 UI 锁竖屏时碰巧相等,界面可旋转时会造成二次旋转。
        // 2. updateUIView 随 @Published(boxes ~3fps)高频触发,会持续覆盖
        //    RotationCoordinator KVO 写入的正确预览角度,两套机制互相打架。
        // 3. 预览镜像保持系统默认 automaticallyAdjustsVideoMirroring = true,
        //    前摄自动镜像,行为与手动设置一致且不会与 tap 坐标链产生歧义。
        // 预览角度的唯一写入方是 CameraManager.setupRotationCoordinator() 的
        // preview KVO(经 attachPreviewLayer 接线)。
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    private let overlayLayer = CAShapeLayer()
    private let maskLayer = CALayer()

    /// Two-tone mask outline (architect_output §3.4).  Four layers, drawn
    /// bottom-to-top: secondary outer → secondary inner → primary outer →
    /// primary inner.  The dark ring goes underneath the light line so the pair
    /// reads as one stroke with a dark halo, which is what makes it legible at
    /// both ends of the background-luminance range.
    ///
    /// ⚠️ C-5: these are `CAShapeLayer`s in VIEW coordinates.  `lineWidth` is
    /// therefore in pt and stays 2.0 / 1.5 / 1.0 pt no matter how far the
    /// 256×256 mask tile is scaled up — which is precisely the property that
    /// baking the stroke into the tile would destroy.
    private let secondaryOuterStroke = CAShapeLayer()
    private let secondaryInnerStroke = CAShapeLayer()
    private let primaryOuterStroke = CAShapeLayer()
    private let primaryInnerStroke = CAShapeLayer()

    /// Last published outlines, retained so `layoutSubviews` can rebuild the
    /// paths after a bounds change without waiting for the next mask.
    private var lastOutlines: MaskOutlineSet?

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

        // Outline layers sit directly above the fill and below the bbox overlay
        // — same mask layer in the §1.3 Z order, no new level.
        let strokeSpecs: [(CAShapeLayer, UIColor, CGFloat, CGFloat)] = [
            (secondaryOuterStroke, MaskOutlineStyle.outerColor,
             MaskOutlineStyle.secondaryOuterWidth, MaskOutlineStyle.secondaryOuterAlpha),
            (secondaryInnerStroke, MaskOutlineStyle.innerColor,
             MaskOutlineStyle.secondaryInnerWidth, MaskOutlineStyle.secondaryInnerAlpha),
            (primaryOuterStroke, MaskOutlineStyle.outerColor,
             MaskOutlineStyle.primaryOuterWidth, MaskOutlineStyle.primaryOuterAlpha),
            (primaryInnerStroke, MaskOutlineStyle.innerColor,
             MaskOutlineStyle.primaryInnerWidth, MaskOutlineStyle.primaryInnerAlpha),
        ]
        for (shape, color, width, alpha) in strokeSpecs {
            shape.strokeColor = color.withAlphaComponent(alpha).cgColor
            shape.lineWidth = width          // pt — C-5
            shape.fillColor = UIColor.clear.cgColor
            shape.lineJoin = .round
            shape.lineCap = .round
            // Paths are rebuilt per mask; implicit animation would smear one
            // instance's contour into the next one's on every tap.
            shape.actions = ["path": NSNull(), "hidden": NSNull()]
            shape.isHidden = true
            layer.addSublayer(shape)
        }

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
        for shape in [secondaryOuterStroke, secondaryInnerStroke,
                      primaryOuterStroke, primaryInnerStroke] {
            shape.frame = bounds
        }

        if bounds != lastBounds {
            lastBounds = bounds
            diagLog("[DBG] preview bounds=\(bounds.size), gravity=\(videoPreviewLayer.videoGravity)")
            // Outline paths are baked in view points, so a bounds change
            // invalidates them; rebuild from the retained geometry.
            rebuildOutlinePaths()
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
            diagLog(String(format: "[DBG] mask layer frame=%@ | bounds=%@ | rot=%.0f mirrored=%@",
                         NSCoder.string(for: maskLayer.frame),
                         NSCoder.string(for: bounds),
                         rotationAngle, mirrored ? "true" : "false"))
        }
    }

    /// Publish the tap masks' outlines (architect_output §3.4).
    ///
    /// nil / empty hides every stroke layer, which is also what the C-6 switch
    /// (`MaskOutlineStyle.isEnabled == false`) produces upstream — the Day 4
    /// "no outline" visual condition needs no change here.
    func updateMaskOutlines(_ set: MaskOutlineSet?) {
        lastOutlines = (set?.isEmpty == false) ? set : nil
        rebuildOutlinePaths()
    }

    /// Convert the normalised contours into view-space `CGPath`s.
    ///
    /// The transform below is exactly what CoreAnimation applies to the fill
    /// image (`contentsGravity = .resizeAspectFill` into `bounds`): scale by the
    /// LARGER of the two axis ratios, then centre.  Deriving the outline from
    /// the same rule is what guarantees stroke and fill cannot separate — and it
    /// is why the contour arrives normalised rather than in pixels.
    private func rebuildOutlinePaths() {
        let primaryPath = CGMutablePath()
        let secondaryPath = CGMutablePath()

        if let set = lastOutlines,
           set.canvasSize.width > 0, set.canvasSize.height > 0,
           bounds.width > 0, bounds.height > 0 {
            let scale = max(bounds.width / set.canvasSize.width,
                            bounds.height / set.canvasSize.height)
            let drawW = set.canvasSize.width * scale
            let drawH = set.canvasSize.height * scale
            let originX = (bounds.width - drawW) * 0.5
            let originY = (bounds.height - drawH) * 0.5

            for outline in set.outlines {
                let path = outline.isPrimary ? primaryPath : secondaryPath
                for polygon in outline.polygons where polygon.count >= 3 {
                    for (i, p) in polygon.enumerated() {
                        let pt = CGPoint(x: originX + p.x * drawW,
                                         y: originY + p.y * drawH)
                        if i == 0 { path.move(to: pt) } else { path.addLine(to: pt) }
                    }
                    path.closeSubpath()
                }
            }
        }

        let hasSecondary = !secondaryPath.isEmpty
        let hasPrimary = !primaryPath.isEmpty
        secondaryOuterStroke.path = secondaryPath
        secondaryInnerStroke.path = secondaryPath
        primaryOuterStroke.path = primaryPath
        primaryInnerStroke.path = primaryPath
        secondaryOuterStroke.isHidden = !hasSecondary
        secondaryInnerStroke.isHidden = !hasSecondary
        primaryOuterStroke.isHidden = !hasPrimary
        primaryInnerStroke.isHidden = !hasPrimary
    }
}
