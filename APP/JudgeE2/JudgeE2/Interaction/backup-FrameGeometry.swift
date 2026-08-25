//
//  FrameGeometry.swift
//  JudgeE2
//
//  Phase 3 — Day 2 (Builder)
//
//  Single source of truth for the preview-view → Canonical-pixel coordinate
//  inversion used by Tap-to-Segment. Implements the transform chain frozen in
//  shared/architect_output.md §2 (Tap 坐标变换契约).
//
//  Canonical space == original camera-frame pixel space (origW × origH), in the
//  *display orientation*. Device rotation is already baked into origW/origH by the
//  capture pipeline (AVCaptureConnection.videoRotationAngle rotates the buffer
//  before it reaches the model / mask renderer — see CameraPreview.swift notes),
//  so this inversion must NOT re-apply a rotation. It only inverts the aspectFill
//  mapping + mirror, then clamps. This keeps a single geometry chain shared with
//  Phase 1/2 (YOLO bbox + SAM mask) — no duplicate transforms.
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
    /// Current device rotation angle (degrees) — carried for logging/parity only.
    let rotation: CGFloat

    /// Invert a UIKit tap point (in `previewLayer`'s view coordinate system) back
    /// to Canonical pixel space.
    ///
    /// Transform chain (architect_output.md §2.1):
    ///   Step 1: AspectFill inverse via `captureDevicePointConverted(fromLayerPoint:)`
    ///           → normalized sensor coords [0,1]×[0,1]. Apple handles the aspectFill
    ///           crop offset internally.
    ///   Step 2: normalized × (origW, origH) → Canonical pixel coords.
    ///   Step 3: mirror correction (front camera).
    ///   Step 4: clamp to [0, orig-1].
    ///
    /// - Returns: the corrected Canonical pixel point, plus the intermediate
    ///            normalized point (for the `[TAP]` log required by the contract).
    func invertViewPoint(_ point: CGPoint,
                         viewSize: CGSize,
                         previewLayer: AVCaptureVideoPreviewLayer)
        -> (canonical: CGPoint, normalized: CGPoint)
    {
        // Step 1 — AspectFill inverse (normalized [0,1] sensor coords).
        var normalized = previewLayer.captureDevicePointConverted(fromLayerPoint: point)

        // Step 3a — mirror correction on the normalized axis (front camera flips X).
        // Applied on the normalized coordinate so it composes cleanly with Step 2.
        if mirrored {
            normalized.x = 1.0 - normalized.x
        }

        // Step 2 — normalized → Canonical pixel space.
        var cx = normalized.x * origW
        var cy = normalized.y * origH

        // Step 4 — clamp to valid Canonical range (越界裁剪, tap 不排除).
        cx = min(max(0, cx), origW - 1)
        cy = min(max(0, cy), origH - 1)

        return (CGPoint(x: cx, y: cy), normalized)
    }
}
