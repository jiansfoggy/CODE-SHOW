//
//  PromptBuilder.swift
//  JudgeE2
//
//  Created by Jian Sun on 2/28/26.
//  Modified: include detectorInputSize and produce a coarse mask_input (256x256) from bbox.
//

import CoreML
import Foundation

struct PromptBuilder {
    struct BoxPrompt {
        let pointCoords: MLMultiArray
        let pointLabels: MLMultiArray
        let maskInput: MLMultiArray
        let hasMaskInput: MLMultiArray
    }

    /// Build SAM box prompt from bbox.
    ///
    /// - x1,y1,x2,y2 : either pixel coordinates in the original image space OR normalized (0..1) if boxIsNormalized==true.
    /// - origW, origH : original image pixel size (camera frame)
    /// - inputSize : SAM encoder input size (default 1024)
    /// - boxIsNormalized : true if passed box are normalized 0..1
    /// - detectorInputSize : if non-nil, indicates that the box coordinates were produced on a detector input of
    ///                       size `detectorInputSize` (e.g. 640). In that case we scale the box from detector space
    ///                       back to original image pixel space before applying ResizeLongestSide(1024)+pad.
    static func buildBoxPrompt(x1: Float, y1: Float, x2: Float, y2: Float,
                               origW: Float, origH: Float,
                               inputSize: Int = 1024,
                               boxIsNormalized: Bool = false,
                               detectorInputSize: Float? = nil) -> BoxPrompt? {
        guard origW > 0, origH > 0 else { return nil }

        // Copy inputs into mutable vars
        var x1p = x1, y1p = y1, x2p = x2, y2p = y2

        // If coordinates came from a detector that ran at a fixed input size (e.g. 640),
        // scale them back into the original image pixel space before further processing.
        if let detSize = detectorInputSize, detSize > 0 {
            let scaleX = origW / detSize
            let scaleY = origH / detSize
            x1p = x1p * scaleX
            x2p = x2p * scaleX
            y1p = y1p * scaleY
            y2p = y2p * scaleY
        }

        // If inputs are normalized (0~1), convert to original pixel space first
        if boxIsNormalized {
            x1p *= origW; y1p *= origH; x2p *= origW; y2p *= origH
        }

        // Ensure min/max ordering
        if x2p < x1p { swap(&x1p, &x2p) }
        if y2p < y1p { swap(&y1p, &y2p) }

        // Now we have box coordinates in original image pixel space (origW x origH).
        // Apply SAM ResizeLongestSide(inputSize) + centered padding transform.
        let target = Float(inputSize)
        let scale = target / max(origW, origH)
        let newW = origW * scale
        let newH = origH * scale
        let padX = (target - newW) * 0.5
        let padY = (target - newH) * 0.5

        // Resize + pad to SAM input frame (e.g. 1024x1024)
        var sx1 = (x1p * scale + padX)
        var sy1 = (y1p * scale + padY)
        var sx2 = (x2p * scale + padX)
        var sy2 = (y2p * scale + padY)

        // Clip to valid range [0, target-1]
        let maxCoord = target - 1
        sx1 = min(max(0, sx1), maxCoord)
        sy1 = min(max(0, sy1), maxCoord)
        sx2 = min(max(0, sx2), maxCoord)
        sy2 = min(max(0, sy2), maxCoord)
        if sx2 < sx1 { swap(&sx1, &sx2) }
        if sy2 < sy1 { swap(&sy1, &sy2) }

        // Debug print: mapping summary in SAM space
        let toMaskScaleDbg = Float(256) / Float(inputSize)
        let mx1_dbg = sx1 * toMaskScaleDbg
        let my1_dbg = sy1 * toMaskScaleDbg
        let mx2_dbg = sx2 * toMaskScaleDbg
        let my2_dbg = sy2 * toMaskScaleDbg
        diagLog("[SAM] orig(\(origW)x\(origH)) box=\(x1p),\(y1p),\(x2p),\(y2p) | sam=\(sx1),\(sy1),\(sx2),\(sy2) | mask256=\(mx1_dbg),\(my1_dbg),\(mx2_dbg),\(my2_dbg)")
        // Interpolated rather than variadic so it can go through diagLog; the
        // rendered text is unchanged (print's default separator is a space).
        diagLog("prompt box (px\(inputSize)): \(sx1) \(sy1) \(sx2) \(sy2)")

        // Build MLMultiArray prompts
        guard let pointCoords = try? MLMultiArray(shape: [1, 2, 2], dataType: .float32),
              let pointLabels = try? MLMultiArray(shape: [1, 2], dataType: .float32),
              let maskInput = try? MLMultiArray(shape: [1, 1, 256, 256], dataType: .float32),
              let hasMaskInput = try? MLMultiArray(shape: [1], dataType: .float32) else {
            return nil
        }

        func set(_ array: MLMultiArray, _ idx: [Int], _ value: Float) {
            let key = idx.map { NSNumber(value: $0) }
            array[key] = NSNumber(value: value)
        }

        // SAM box prompt format: two points (top-left, bottom-right) with labels 2 and 3
        set(pointCoords, [0, 0, 0], sx1)
        set(pointCoords, [0, 0, 1], sy1)
        set(pointCoords, [0, 1, 0], sx2)
        set(pointCoords, [0, 1, 1], sy2)

        set(pointLabels, [0, 0], 2.0)   // top-left
        set(pointLabels, [0, 1], 3.0)   // bottom-right

        // Zero mask_input and set has_mask_input=0.  Written through the checked
        // buffer API, never `dataPointer`: MLMultiArray storage may be
        // IOSurface-backed and is only valid inside this call.
        maskInput.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            buf.update(repeating: 0.0)
        }
        set(hasMaskInput, [0], 0.0)

        return BoxPrompt(pointCoords: pointCoords,
                         pointLabels: pointLabels,
                         maskInput: maskInput,
                         hasMaskInput: hasMaskInput)
    }
}
