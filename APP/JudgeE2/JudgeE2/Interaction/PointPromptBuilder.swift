//
//  PointPromptBuilder.swift
//  JudgeE2
//
//  Phase 3 Day 3 — Point-Based SAM Prompt (Tap-to-Segment)
//
//  Converts a user tap in Canonical pixel space → SAM 1024-space prompt tensors.
//  I/O contract: model_plan.md §A.2–A.6 (Phase 3 Day 1 ML_Vision)
//
//  ┌──────────────────────────────────────────────────────────────────────────┐
//  │ Tensor layout (model_plan.md §A.1)                                       │
//  │   point_coords : [1, 2, 2]   Float32  (2 points × 2 coords)             │
//  │   point_labels : [1, 2]      Float32  (label per point)                 │
//  │   mask_input   : [1,1,256,256] Float32 (all zeros — no prior mask)      │
//  │   has_mask_input:[1]          Float32 (0.0)                             │
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Encoding rule (model_plan.md §A.2)                                       │
//  │   Point 0: foreground tap   label = 1.0                                  │
//  │   Point 1: padding at (0,0) label = -1.0                                 │
//  │   Coords are SAM pixel space (0 ~ 1023); model divides by 1024 internally│
//  ├──────────────────────────────────────────────────────────────────────────┤
//  │ Transform: matches PromptBuilder.buildBoxPrompt exactly                  │
//  │   scale = inputSize / max(origW, origH)   ← ResizeLongestSide           │
//  │   padX  = (inputSize - origW*scale) / 2   ← centered pad               │
//  │   samX  = canonicalPoint.x * scale + padX                               │
//  └──────────────────────────────────────────────────────────────────────────┘

import CoreML
import Foundation

struct PointPromptBuilder {

    // MARK: - Output

    struct PointPrompt {
        let pointCoords:  MLMultiArray   // [1, 2, 2]       Float32
        let pointLabels:  MLMultiArray   // [1, 2]          Float32
        let maskInput:    MLMultiArray   // [1, 1, 256, 256] Float32 (zeros)
        let hasMaskInput: MLMultiArray   // [1]             Float32 (0.0)
    }

    // MARK: - Build

    /// Map a Canonical-pixel tap point to SAM prompt tensors.
    ///
    /// - Parameters:
    ///   - canonicalPoint: Tap location in original camera pixel space (origW × origH).
    ///                     Produced by `FrameGeometry.invertViewPoint`.
    ///   - origSize:       Original camera frame size in pixels (e.g. 1080×1920 portrait).
    ///   - inputSize:      SAM encoder input edge length (default 1024).
    /// - Returns: Ready-to-feed PointPrompt, or nil if array allocation fails.
    static func buildPointPrompt(
        canonicalPoint: CGPoint,
        origSize: CGSize,
        inputSize: Int = 1024
    ) -> PointPrompt? {
        guard origSize.width > 0, origSize.height > 0 else { return nil }

        let origW  = Float(origSize.width)
        let origH  = Float(origSize.height)
        let target = Float(inputSize)

        // ── ResizeLongestSide(inputSize) + centered pad ──────────────────────
        // Identical to PromptBuilder.buildBoxPrompt to guarantee the same
        // coordinate frame is used for both box-prompt and point-prompt paths.
        let scale = target / max(origW, origH)
        let newW  = origW * scale
        let newH  = origH * scale
        let padX  = (target - newW) * 0.5
        let padY  = (target - newH) * 0.5

        // Map canonical → SAM pixel space
        var samX = Float(canonicalPoint.x) * scale + padX
        var samY = Float(canonicalPoint.y) * scale + padY

        // Clamp to valid range [0, inputSize-1]
        samX = max(0, min(samX, target - 1))
        samY = max(0, min(samY, target - 1))

        // ── Allocate tensors ─────────────────────────────────────────────────
        guard
            let coords  = try? MLMultiArray(shape: [1, 2, 2],         dataType: .float32),
            let labels  = try? MLMultiArray(shape: [1, 2],            dataType: .float32),
            let maskIn  = try? MLMultiArray(shape: [1, 1, 256, 256],  dataType: .float32),
            let hasMask = try? MLMultiArray(shape: [1],               dataType: .float32)
        else { return nil }

        // Helper — indexed write
        @inline(__always)
        func set(_ arr: MLMultiArray, _ idx: [Int], _ v: Float) {
            arr[idx.map { NSNumber(value: $0) }] = NSNumber(value: v)
        }

        // Point 0 — foreground (label = 1.0)
        set(coords, [0, 0, 0], samX)
        set(coords, [0, 0, 1], samY)
        // Point 1 — padding (label = -1.0, coords = 0,0)
        set(coords, [0, 1, 0], 0.0)
        set(coords, [0, 1, 1], 0.0)
        set(labels, [0, 0],  1.0)
        set(labels, [0, 1], -1.0)

        // Zero mask_input (no prior mask hint).  Written through the checked
        // buffer API, never `dataPointer`: MLMultiArray storage may be
        // IOSurface-backed and is only valid inside this call.
        maskIn.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            buf.update(repeating: 0.0)
        }
        set(hasMask, [0], 0.0)

        return PointPrompt(pointCoords:  coords,
                           pointLabels:  labels,
                           maskInput:    maskIn,
                           hasMaskInput: hasMask)
    }
}
