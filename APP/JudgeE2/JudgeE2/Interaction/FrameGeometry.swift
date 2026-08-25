//
//  FrameGeometry.swift
//  JudgeE2
//
//  Phase 3 — Day 2 (Builder) · P1-Critical fix applied
//
//  Single source of truth for the preview-view → Canonical-pixel coordinate
//  inversion used by Tap-to-Segment. Implements the transform chain frozen in
//  shared/architect_output.md §2 (Tap 坐标变换契约).
//
//  Canonical space == original camera-frame pixel space (origW × origH), in the
//  *display orientation*. Device rotation is baked into origW/origH by the
//  capture pipeline (AVCaptureConnection.videoRotationAngle rotates the buffer
//  before it reaches the model / mask renderer — see CameraPreview.swift notes).
//

import AVFoundation
import CoreGraphics
import Foundation

/// Immutable snapshot of the current canonical frame geometry, plus the one
/// approved inversion path from UIKit view coordinates to Canonical pixel space.
struct FrameGeometry {
    /// Canonical (original camera frame) width in pixels, display orientation.
    let origW: CGFloat
    /// Canonical (original camera frame) height in pixels, display orientation.
    let origH: CGFloat
    /// Front-camera mirroring applied to the preview (matches CameraManager.maskMirrored).
    let mirrored: Bool
    /// Current device rotation angle (degrees). ⚠️ 参与坐标变换(不再只是日志用途):
    /// 用于把传感器原生归一化坐标旋转映射到 buffer(显示方向)归一化坐标。
    let rotation: CGFloat

    // MARK: - Phase 4 §16.3.1 — drift components
    //
    // Added for `DriftDetector`.  Neither field participates in
    // `invertViewPoint`; the tap transform chain frozen in §2 is untouched.

    /// Letterbox centering offset in view points (§16.3.2 — view/UIKit layout
    /// space, NOT canonical pixel space).  Mirrors `LetterboxInfo.padX/padY`.
    ///
    /// ⚠️ See the measurement note at the top of `DriftDetector.swift`: this is
    /// a pure function of the camera buffer's dimensions and the fixed model
    /// input size, so it does not vary with camera pose.
    let letterboxOffset: CGPoint

    /// Letterbox scale factor (`LetterboxInfo.scale`).  Carried for
    /// completeness and for `FrameGeometry`'s role as the geometry snapshot;
    /// §16.3.3 rules it OUT of drift detection — a real change to `scale` also
    /// changes the geometry signature and is already covered by C4, which
    /// empties the instance pool before any re-anchor could observe it.
    let scale: CGFloat

    /// `rotation` under the name §16.1.1 uses, in degrees.
    var videoRotationAngle: Double { Double(rotation) }

    /// Invert a UIKit tap point (in `previewLayer`'s view coordinate system) back
    /// to Canonical pixel space.
    ///
    func invertViewPoint(_ point: CGPoint,
                         viewSize: CGSize,
                         previewLayer: AVCaptureVideoPreviewLayer)
        -> (canonical: CGPoint, normalized: CGPoint)
    {
        // Step 1 — AspectFill inverse → sensor-native normalized [0,1] coords.
        let sensor = previewLayer.captureDevicePointConverted(fromLayerPoint: point)

        // Step 2 — rotation mapping: sensor (landscape) → buffer (display orientation).
        // Normalize the angle to {0, 90, 180, 270}.
        let angle = ((rotation.truncatingRemainder(dividingBy: 360)) + 360)
            .truncatingRemainder(dividingBy: 360)

        var bufX: CGFloat
        var bufY: CGFloat
        switch angle {
        case 90:
            // 竖持 (portrait)。真机 12 组数据精确验证:
            //   canonical_x = (1 - sensor_y) * origW   传感器 Y 翻转 → buffer X
            //   canonical_y = sensor_x * origH         传感器 X       → buffer Y
            bufX = 1.0 - sensor.y
            bufY = sensor.x
        case 180:
            // 倒持 landscape。
            bufX = 1.0 - sensor.x
            bufY = 1.0 - sensor.y
        case 270:
            // 倒持 portrait — rot=90 的逆向。
            bufX = sensor.y
            bufY = 1.0 - sensor.x
        default:
            // rot=0: buffer 与传感器同向，直通。
            bufX = sensor.x
            bufY = sensor.y
        }

        // Step 3 — mirror correction on the DISPLAY-space horizontal axis.
        // 前摄预览是显示方向上的水平镜像，对应 buffer/Canonical 的 X 轴，
        // 必须在旋转映射之后翻转，才能与 CameraManager.maskMirrored /
        // mapToMetadataRect 的镜像语义保持一致。
        if mirrored {
            bufX = 1.0 - bufX
        }

        let normalized = CGPoint(x: bufX, y: bufY)

        // Step 4 — buffer normalized → Canonical pixel space.
        var cx = bufX * origW
        var cy = bufY * origH

        // Step 5 — clamp to valid Canonical range (越界裁剪, tap 不排除).
        cx = min(max(0, cx), origW - 1)
        cy = min(max(0, cy), origH - 1)

        return (CGPoint(x: cx, y: cy), normalized)
    }

    /// Project a Canonical pixel point forward into the preview layer's view
    /// coordinate system — B-37 route A1 (architect_output.md §23.1.4 二).
    ///
    /// ⚠️ **This function is the inverse of `invertViewPoint`, not a second
    /// transform path.**  Each step below is the algebraic inverse of the
    /// correspondingly numbered step of `invertViewPoint`, executed in reverse
    /// order (4⁻¹ → 3⁻¹ → 2⁻¹ → 1⁻¹), reading the SAME `origW`/`origH`/
    /// `mirrored`/`rotation` fields of the same geometry snapshot.  Step 1's
    /// inverse uses AVFoundation's documented dual of
    /// `captureDevicePointConverted(fromLayerPoint:)`:
    /// `layerPointConverted(fromCaptureDevicePoint:)`.
    ///
    /// `invertViewPoint`'s Step 5 (boundary clamp) has no inverse and is not
    /// mirrored here: a clamp is not injective, and every caller feeds this
    /// function an in-bounds Canonical point (a stored / derived tap point,
    /// already clamped when it was produced).
    ///
    /// Correctness is gated by P-9 (§23.2.8): round-trip
    /// `projectCanonicalPoint` → `invertViewPoint` must return to the origin
    /// within a Chebyshev distance of 1.0 px on a 9-point grid, in all four
    /// rotation × mirroring configurations.
    func projectCanonicalPoint(_ canonical: CGPoint,
                               previewLayer: AVCaptureVideoPreviewLayer) -> CGPoint
    {
        // Step 4⁻¹ — Canonical pixel space → buffer normalized [0,1].
        var bufX = canonical.x / origW
        let bufY = canonical.y / origH   // only X carries the mirror flip below

        // Step 3⁻¹ — mirror correction is an involution: applying the same
        // display-space horizontal flip again undoes it.
        if mirrored {
            bufX = 1.0 - bufX
        }

        // Step 2⁻¹ — buffer (display orientation) → sensor-native normalized.
        // Same angle normalization as the forward direction, each case solved
        // for `sensor` from the equations in `invertViewPoint`.
        let angle = ((rotation.truncatingRemainder(dividingBy: 360)) + 360)
            .truncatingRemainder(dividingBy: 360)

        let sensor: CGPoint
        switch angle {
        case 90:
            // forward: bufX = 1 − sensor.y, bufY = sensor.x
            sensor = CGPoint(x: bufY, y: 1.0 - bufX)
        case 180:
            // forward: bufX = 1 − sensor.x, bufY = 1 − sensor.y
            sensor = CGPoint(x: 1.0 - bufX, y: 1.0 - bufY)
        case 270:
            // forward: bufX = sensor.y, bufY = 1 − sensor.x
            sensor = CGPoint(x: 1.0 - bufY, y: bufX)
        default:
            // rot=0: pass-through, same as the forward direction.
            sensor = CGPoint(x: bufX, y: bufY)
        }

        // Step 1⁻¹ — sensor-native normalized → layer/view point, via the
        // documented AVFoundation dual of Step 1's conversion.
        return previewLayer.layerPointConverted(fromCaptureDevicePoint: sensor)
    }
}

// MARK: - P-9 self-check (§23.2.8) — TEST-ONLY
//
// WHY THIS EXISTS
// P-9 gates B-37's route A1: for the current geometry, a 9-point grid in
// Canonical space must survive `projectCanonicalPoint` (canonical → view)
// followed by `invertViewPoint` (view → canonical) with a Chebyshev distance
// back to the origin point of ≤ 1.0 px, in four device configurations
// (portrait/landscape × back/front).  Until now `projectCanonicalPoint` was
// reachable only from the production revisit path, so the round trip was a
// pure numerical property with no way for a tester to observe it — P-9 was
// unexecutable, which is exactly why B-37 shipped with its correctness
// deferred to P-9 rather than self-certified.  This extension makes the
// measurement triggerable by hand from the Debug Options panel.
//
// ⛔ NOT A HOT PATH.  Main thread only, once per button press.  Nothing here
// is referenced from videoQueue / decoderQueue / encoderQueue.
//
// ⛔ MEASURES, DOES NOT FIX.  If the round trip exceeds 1.0 px that is a
// finding about the transform chain, not a licence to touch
// `projectCanonicalPoint` / `invertViewPoint` ahead of an Architect ruling.
//
// ON THE STEP-5 CLAMP (reported to the Architect with this batch)
// `invertViewPoint`'s Step 5 clamps to [0, origW-1] × [0, origH-1] and has no
// inverse.  Its effect on this measurement is one-directional: at a point on
// the clamp boundary an *outward* round-trip error is silently truncated to
// zero, while an *inward* error passes through untouched.  So the clamp can
// only ever MASK a real error (spurious PASS); it can never manufacture one
// (spurious FAIL).  §23.2.8 insets the four corners by 1 px, which keeps them
// clear of the clamp for any error small enough to pass; the four EDGE
// MIDPOINTS are not inset and each has one coordinate sitting exactly on the
// boundary, so they stay maskable.  The grid below is implemented **exactly as
// §23.2.8 specifies** — widening the inset so the numbers look better would be
// the A-7 / A-19 failure mode this project keeps catching.  Instead each line
// also reports `errNoClamp` (the same round trip measured from the pre-clamp
// `normalized` output of the very same `invertViewPoint` call) and
// `clamp=on/off`, so a Debugger can see when the clamp participated without
// the criterion itself being altered.  `err` is the criterion; `errNoClamp` is
// diagnostic only.

extension FrameGeometry {

    /// §23.2.8's bar: Chebyshev distance back to the origin point.
    static let roundTripTolerancePx: CGFloat = 1.0

    /// The §23.2.8 grid in Canonical space: four corners inset 1 px, four edge
    /// midpoints, centre — over the valid Canonical range that
    /// `invertViewPoint`'s Step 5 defines, i.e. [0, origW-1] × [0, origH-1].
    static func roundTripGrid(origW: CGFloat, origH: CGFloat) -> [CGPoint] {
        let maxX = origW - 1, maxY = origH - 1
        let midX = maxX / 2,  midY = maxY / 2
        return [
            CGPoint(x: 1,        y: 1),          // corners, inset 1 px
            CGPoint(x: maxX - 1, y: 1),
            CGPoint(x: 1,        y: maxY - 1),
            CGPoint(x: maxX - 1, y: maxY - 1),
            CGPoint(x: midX,     y: 0),          // edge midpoints (not inset)
            CGPoint(x: midX,     y: maxY),
            CGPoint(x: 0,        y: midY),
            CGPoint(x: maxX,     y: midY),
            CGPoint(x: midX,     y: midY),       // centre
        ]
    }

    /// Run the round trip against this geometry snapshot and print the result.
    /// Main thread only (`layerPointConverted` is a layer-geometry query).
    func runRoundTripSelfCheck(previewLayer layer: AVCaptureVideoPreviewLayer) {
        assert(Thread.isMainThread, "runRoundTripSelfCheck must run on main thread")

        let tol = Self.roundTripTolerancePx
        let bounds = layer.bounds
        let mirroredField = mirrored ? "true" : "false"

        perfLog(String(format:
            "[P9] cfg rot=%.0f mirrored=%@ canonical=%.0fx%.0f preview=%.1fx%.1f tol=%.1fpx",
            rotation, mirroredField, origW, origH,
            bounds.width, bounds.height, tol))

        let points = Self.roundTripGrid(origW: origW, origH: origH)
        var maxErr: CGFloat = 0
        var worstIdx = 0
        var clampedCount = 0

        for (i, pt) in points.enumerated() {
            let view = projectCanonicalPoint(pt, previewLayer: layer)
            let back = invertViewPoint(view, viewSize: bounds.size, previewLayer: layer)
            let err = max(abs(back.canonical.x - pt.x), abs(back.canonical.y - pt.y))
            // Pre-clamp reconstruction, from the same call's `normalized` output.
            let raw = CGPoint(x: back.normalized.x * origW, y: back.normalized.y * origH)
            let rawErr = max(abs(raw.x - pt.x), abs(raw.y - pt.y))
            let clamped = abs(raw.x - back.canonical.x) > 1e-6
                       || abs(raw.y - back.canonical.y) > 1e-6
            if clamped { clampedCount += 1 }
            if err > maxErr { maxErr = err; worstIdx = i + 1 }

            perfLog(String(format:
                "[P9] i=%d/%d pt=(%.1f,%.1f) → view=(%.1f,%.1f) → back=(%.1f,%.1f) err=%.2fpx errNoClamp=%.2fpx clamp=%@",
                i + 1, points.count, pt.x, pt.y, view.x, view.y,
                back.canonical.x, back.canonical.y, err, rawErr,
                clamped ? "on" : "off"))
        }

        // `scope=1of4cfg`: one press covers ONE of P-9's four configurations.
        // PASS here is NOT P-9 PASS — that needs all four (§23.2.8).  The
        // config fields are repeated on this line so a Debugger grepping only
        // the result line can still tell which configuration produced it.
        perfLog(String(format:
            "[P9] result n=%d maxErr=%.2fpx worstIdx=%d clamped=%d/%d rot=%.0f mirrored=%@ canonical=%.0fx%.0f verdict=%@(<=%.1fpx) scope=1of4cfg",
            points.count, maxErr, worstIdx, clampedCount, points.count,
            rotation, mirroredField, origW, origH,
            maxErr <= tol ? "PASS" : "FAIL", tol))
    }
}
