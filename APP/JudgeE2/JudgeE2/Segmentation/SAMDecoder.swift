//
//  SAMDecoder.swift
//  JudgeE2
//
//  Created by Jian Sun on 2/28/26.
//

import CoreML
import Foundation

/// Owns the MobileSAM prompt decoders.
///
/// **Queue discipline:** an instance is created and used on `decoderQueue` only
/// (`CameraManager.decoderForQueue`).  The lazy box-decoder state below relies on
/// that confinement and therefore carries no lock of its own.
final class SAMDecoder {
    // MARK: - Box decoder (lazy)

    /// Where the single-mask (box-prompt) decoder lives, and the compute units it
    /// must be built with.  Resolved in `init` — a bundle URL lookup is free, the
    /// `MLModel` construction behind it is not.
    private let boxModelURL: URL
    private let boxComputeUnits: MLComputeUnits

    /// Single-mask decoder — box prompts (Segmentation mode, Phase 2 behavior).
    ///
    /// **Built on first use, not in `init`.**  It used to be loaded
    /// unconditionally, which cost a full `MLModel` construction (plus ANE
    /// compilation) even in `.tapToSegment`, where the box path is never called:
    /// the tap path goes through `pointModelForDecode()`.  A Debugger isolated-load
    /// experiment (debug_report.md §16.5) pinned all three
    /// `Invalid layer: Invalid input tensor channel 1 … aligned on 64 bytes`
    /// warnings on *this* load — the milfix encoder loaded alone emits none.
    ///
    /// Phase 2 `.segmentation` still gets the ANE units the caller asked for:
    /// this is on-demand loading, not a compute-unit downgrade.
    private var boxModel: MLModel?
    /// Latches a failed lazy load so a broken bundle is reported once, not per frame.
    private var boxModelLoadFailed = false

    // MARK: - Point decoder

    /// Multimask decoder (low_res_masks [1,3,256,256] + iou_predictions [1,3])
    /// for point prompts: a single tap is an ambiguous prompt and the
    /// single-mask export collapses ambiguity into oversized masks.
    ///
    /// `nil` when the multimask package is not bundled — point prompts then fall
    /// back to the (lazily built) box decoder, exactly as before.
    private let multimaskPointModel: MLModel?
    /// True when point-prompt decodes produce 3 candidates.
    let isMultimask: Bool

    init?(computeUnits: MLComputeUnits = .all) {
        guard let url = Bundle.main.url(forResource: "MobileSAM_PromptMaskDecoder", withExtension: "mlmodelc") else {
            faultLog("MobileSAM_PromptMaskDecoder.mlmodelc not found in bundle")
            return nil
        }
        boxModelURL = url
        boxComputeUnits = computeUnits

        // Point decoder runs on CPU+GPU only: on A13's ANE the multimask
        // decoder produced garbage-magnitude logits (|logit| ≈ 1e7–1e8,
        // iou_pred > 1.0, degenerate half-frame masks) via the fp16 C=1
        // alignment bug.  The decoder is small — GPU decode is ~100–200 ms.
        let pointConfig = MLModelConfiguration()
        pointConfig.computeUnits = .cpuAndGPU
        if let multiURL = Bundle.main.url(forResource: "MobileSAM_PromptMaskDecoder_multi", withExtension: "mlmodelc"),
           let multi = try? MLModel(contentsOf: multiURL, configuration: pointConfig) {
            multimaskPointModel = multi
            isMultimask = true
            perfLog("[SEG] point decoder: multimask (3 candidates, CPU+GPU)")
        } else {
            // Fallback: the box decoder IS the point decoder.  `isMultimask`
            // stays false so callers keep treating the output as 1 candidate,
            // and the box model is materialised on the first decode of either
            // kind — the fallback route is preserved, only deferred.
            multimaskPointModel = nil
            isMultimask = false
            perfLog("[SEG] point decoder: single-mask (multimask package not found) — shares the box decoder")
        }
    }

    // MARK: - Lazy model access (decoderQueue only)

    /// The box decoder, built on first request.  Returns nil if the load fails.
    private func boxModelForDecode() -> MLModel? {
        if let boxModel = boxModel { return boxModel }
        if boxModelLoadFailed { return nil }
        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            let loaded = try MLModel(contentsOf: boxModelURL, configuration: {
                let config = MLModelConfiguration()
                config.computeUnits = boxComputeUnits
                return config
            }())
            boxModel = loaded
            perfLog(String(format: "[SEG] box decoder built on demand in %.2f ms (units=%d)",
                           (CFAbsoluteTimeGetCurrent() - t0) * 1000.0, boxComputeUnits.rawValue))
            return loaded
        } catch {
            boxModelLoadFailed = true
            faultLog("Failed to load MobileSAM_PromptMaskDecoder: \(error)")
            return nil
        }
    }

    /// The model that serves point prompts: the multimask package when bundled,
    /// otherwise the box decoder (single-mask fallback).
    private func pointModelForDecode() -> MLModel? {
        if let multimaskPointModel = multimaskPointModel { return multimaskPointModel }
        return boxModelForDecode()
    }

    func decode(embedding: MLMultiArray, prompt: PromptBuilder.BoxPrompt) -> MLMultiArray? {
        guard let model = boxModelForDecode() else {
            faultLog("MobileSAM box decoder unavailable — box decode skipped")
            return nil
        }
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image_embeddings": embedding,
                "point_coords": prompt.pointCoords,
                "point_labels": prompt.pointLabels,
                "mask_input": prompt.maskInput,
                "has_mask_input": prompt.hasMaskInput
            ])
            let output = try model.prediction(from: input)
            if let m = output.featureValue(for: "low_res_masks")?.multiArrayValue {
                return m
            }
            if let m = output.featureValue(for: "low_res_logits")?.multiArrayValue {
                return m
            }
            if let m = output.featureValue(for: "masks")?.multiArrayValue {
                return m
            }
            faultLog("MobileSAM decoder output keys: \(output.featureNames)")
            return nil
        } catch {
            faultLog("MobileSAM decoder prediction failed: \(error)")
            return nil
        }
    }

    // MARK: - Phase 3 Point Prompt (Tap-to-Segment)

    /// Decode with a user-tap point prompt (Phase 3).
    ///
    /// Same model, same I/O contract — only the prompt tensors differ:
    ///   point_coords: [[tapX, tapY], [0, 0]]  (foreground + padding)
    ///   point_labels: [1.0, -1.0]
    ///
    /// - Returns: (lowResMask, iouPreds), or nil on model error.  With the
    ///   multimask decoder the mask is [1,3,256,256] and iouPreds has 3 values
    ///   (one per candidate); with the single-mask fallback both have 1.
    ///
    /// - Parameter tapIndex: tap sequence number, diagnostics only — stamps this
    ///   call's lines as `[SEG][TAP#N]` so they join the rest of that tap's chain.
    /// - Parameter logTag: overrides the log prefix entirely.  Used by the
    ///   decoder warmup, whose rehearsal decode belongs to no tap and must not
    ///   look like one in the log.
    ///
    /// Prints: `[SEG][TAP#N] decode latency: %.2f ms iou_preds: […]`
    func decode(embedding: MLMultiArray, point: PointPromptBuilder.PointPrompt,
                tapIndex: Int? = nil,
                logTag: String? = nil) -> (mask: MLMultiArray, iouPreds: [Float])? {
        let tag = logTag ?? tapIndex.map { "[SEG][TAP#\($0)]" } ?? "[SEG][TAP]"
        guard let pointModel = pointModelForDecode() else {
            faultLog("\(tag) point decoder unavailable — decode skipped")
            return nil
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image_embeddings": embedding,
                "point_coords":     point.pointCoords,
                "point_labels":     point.pointLabels,
                "mask_input":       point.maskInput,
                "has_mask_input":   point.hasMaskInput
            ])
            let output = try pointModel.prediction(from: input)
            let latencyMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

            var iouPreds: [Float] = []
            if let iouArr = output.featureValue(for: "iou_predictions")?.multiArrayValue {
                for i in 0..<iouArr.count { iouPreds.append(iouArr[i].floatValue) }
            }
            if iouPreds.isEmpty { iouPreds = [0.0] }

            diagLog(String(format: "%@ decode latency: %.2f ms iou_preds: %@",
                         tag, latencyMs, iouPreds.map { String(format: "%.3f", $0) }.joined(separator: ", ")))

            // Extract mask (same output key priority as box-prompt path)
            for key in ["low_res_masks", "low_res_logits", "masks"] {
                if let m = output.featureValue(for: key)?.multiArrayValue {
                    return (m, iouPreds)
                }
            }
            faultLog("\(tag) decoder output keys not found: \(output.featureNames)")
            return nil

        } catch {
            faultLog("\(tag) decoder prediction failed: \(error)")
            return nil
        }
    }
}


