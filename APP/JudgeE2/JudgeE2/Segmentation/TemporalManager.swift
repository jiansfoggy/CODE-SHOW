//
//  TemporalManager.swift
//  JudgeE2
//
//  Phase 2 Day 6 — Temporal Manager
//
//  Responsibilities:
//  - Primary-object selection with class-change hysteresis
//  - Two-level bbox drift classification (lightDrift / heavyDrift)
//  - Embedding TTL + mask TTL tracking
//  - Cache invalidation triggers
//
//  Contract (Phase 2 architecture):
//  - All geometry stays in original-image pixel space (origW × origH).
//  - No coordinate transforms live here; caller passes CGRect in pixel space.
//  - Thread safety: all public methods must be called from a single serial queue
//    (videoQueue in CameraManager). The embedding cache itself is protected by
//    CameraManager's stateLock for cross-queue reads.

import CoreML
import Foundation
import UIKit

// MARK: - Public types

/// Result of primary-object selection for a single frame.
struct PrimarySelection {
    /// The chosen detection and its rect in original-image pixel space.
    let detection: TemporalManager.DetectionInput
    let rect: CGRect
    /// True when the primary object identity changed (class switch or first detection).
    let primaryChanged: Bool
}

/// Embedding cache entry — owned by TemporalManager, exposed read-only.
struct EmbeddingEntry {
    let embedding: MLMultiArray
    let timestampMs: Double
}

// MARK: - TemporalManager

final class TemporalManager {

    // MARK: - Input type (mirrors CameraManager.Detection but decoupled)

    struct DetectionInput {
        let x1: Float
        let y1: Float
        let x2: Float
        let y2: Float
        let score: Float
        let classId: Int
    }

    // MARK: - Configuration

    /// TTL for the image embedding (ms). Must be >= encoder cadence period.
    var embeddingTTLms: Double = 8000

    /// TTL for the decoded mask (ms).
    /// 2000 ms gives ~2 frames at 1 Hz decode rate as visual fallback.
    var maskTTLms: Double = 2000

    /// Number of consecutive frames a new class must be top-1 before we accept it
    /// as a class switch. Prevents rapid panning from thrashing the encoder.
    var classHysteresisThreshold: Int = 6

    // MARK: - State (all accessed on videoQueue)

    private(set) var primaryRect: CGRect?
    private(set) var primaryScore: Float = 0
    private(set) var primaryClassId: Int = -1

    private var primaryLostCount: Int = 0
    private var primaryScoreDropCount: Int = 0
    private var primaryClassCandidateId: Int = -1
    private var primaryClassCandidateCount: Int = 0

    // Mask cache — WRITTEN on decoderQueue (recordMask) and read/cleared on
    // videoQueue.  A strong MLMultiArray reference mutated from two queues can
    // be over-released (intermittent crash), so these two fields are the only
    // TemporalManager state guarded by a lock.
    private let maskLock = NSLock()
    private var storedMaskCache: (mask: MLMultiArray, timestampMs: Double)?
    private var storedLastMaskRefreshMs: Double = 0

    var maskCache: (mask: MLMultiArray, timestampMs: Double)? {
        maskLock.lock(); defer { maskLock.unlock() }
        return storedMaskCache
    }

    // Last mask refresh timestamp — used to compute refresh rate for logging.
    var lastMaskRefreshMs: Double {
        maskLock.lock(); defer { maskLock.unlock() }
        return storedLastMaskRefreshMs
    }

    // Geometry signature — detect camera / orientation changes that invalidate embeddings.
    private var lastGeometrySignature: GeometrySignature?

    // Tap-path geometry signature (Phase 3 Day 5).
    //
    // Kept SEPARATE from `lastGeometrySignature` for two reasons:
    //  1. Requirement A (architect_output §10.4 A) moved the tap fast-path
    //     decision off videoQueue onto the gesture thread.  `geometryChanged`
    //     mutates videoQueue-owned state and must not be called from there;
    //     this variant carries its own lock, like the mask cache above.
    //  2. `.segmentation` and `.tapToSegment` never run at the same time, so
    //     splitting the signature is behaviour-neutral inside a mode while
    //     removing the cross-mode contamination the shared field allowed.
    private let tapGeometryLock = NSLock()
    private var lastTapGeometrySignature: GeometrySignature?

    // MARK: - Geometry signature

    struct GeometrySignature: Equatable {
        let origW: Float
        let origH: Float
        let scale: Float
        let padX: Float
        let padY: Float
        let rotation: CGFloat
        let mirrored: Bool
        let inputSize: Float

        static func == (lhs: GeometrySignature, rhs: GeometrySignature) -> Bool {
            func close(_ a: Float, _ b: Float, eps: Float = 0.5) -> Bool { abs(a-b) <= eps }
            return close(lhs.origW, rhs.origW)
                && close(lhs.origH, rhs.origH)
                && close(lhs.scale, rhs.scale, eps: 0.001)
                && close(lhs.padX, rhs.padX)
                && close(lhs.padY, rhs.padY)
                && abs(lhs.rotation - rhs.rotation) < 0.5
                && lhs.mirrored == rhs.mirrored
                && close(lhs.inputSize, rhs.inputSize)
        }
    }

    // MARK: - Drift classification

    enum DriftLevel { case noDrift, lightDrift, heavyDrift }

    // MARK: - Public API

    /// Check whether the geometry has changed since the last frame.
    /// Returns true and updates the stored signature if changed.
    func geometryChanged(_ sig: GeometrySignature) -> Bool {
        guard let prev = lastGeometrySignature else {
            lastGeometrySignature = sig
            return false
        }
        if prev != sig {
            lastGeometrySignature = sig
            return true
        }
        return false
    }

    /// Tap-path counterpart of `geometryChanged`, callable from any thread.
    ///
    /// Same semantics: compares the frame geometry seen by *this* tap against the
    /// one seen by the previous tap, returns true and adopts the new signature
    /// when they differ.  Because every live TapInstance is decoded against the
    /// embedding cached for the current geometry, a `false` here is exactly the
    /// condition under which the whole instance pool shares one embedding and
    /// the encoder runs once (Day 5: "同一 geometry 内实例间共享同一 embedding").
    /// A `true` means every cached instance mask belongs to a dead geometry and
    /// the caller must clear the pool.
    func tapGeometryChanged(_ sig: GeometrySignature) -> Bool {
        tapGeometryLock.lock(); defer { tapGeometryLock.unlock() }
        guard let prev = lastTapGeometrySignature else {
            lastTapGeometrySignature = sig
            return false
        }
        if prev != sig {
            lastTapGeometrySignature = sig
            return true
        }
        return false
    }

    /// Forget the tap-path geometry so the next tap re-baselines (mode switch,
    /// backend switch, resolution switch — anything that drops the embedding).
    func resetTapGeometry() {
        tapGeometryLock.lock(); defer { tapGeometryLock.unlock() }
        lastTapGeometrySignature = nil
    }

    /// Reset all primary-object tracking state (call on geometry change or mode reset).
    func resetPrimary() {
        primaryRect = nil
        primaryScore = 0
        primaryClassId = -1
        primaryLostCount = 0
        primaryScoreDropCount = 0
        primaryClassCandidateId = -1
        primaryClassCandidateCount = 0
    }

    /// Invalidate the mask cache (e.g. on light drift or TTL expiry).
    func invalidateMask() {
        maskLock.lock()
        storedMaskCache = nil
        storedLastMaskRefreshMs = 0
        maskLock.unlock()
    }

    /// Record a newly rendered mask and update the refresh-rate log.
    /// Returns the refresh interval in ms (0 on first update).
    @discardableResult
    func recordMask(_ mask: MLMultiArray, timestampMs: Double) -> Double {
        maskLock.lock(); defer { maskLock.unlock() }
        storedMaskCache = (mask: mask, timestampMs: timestampMs)
        let interval = (storedLastMaskRefreshMs > 0) ? (timestampMs - storedLastMaskRefreshMs) : 0
        storedLastMaskRefreshMs = timestampMs
        return interval
    }

    /// True if the mask cache exists and has not exceeded maskTTLms.
    func isMaskValid(nowMs: Double) -> Bool {
        maskLock.lock(); defer { maskLock.unlock() }
        guard let mc = storedMaskCache else { return false }
        return (nowMs - mc.timestampMs) <= maskTTLms
    }

    /// True if an embedding exists and has not exceeded embeddingTTLms.
    func isEmbeddingValid(entry: EmbeddingEntry?, nowMs: Double) -> Bool {
        guard let e = entry else { return false }
        return (nowMs - e.timestampMs) <= embeddingTTLms
    }

    // MARK: - Primary selection

    /// Select the primary object to segment from a list of detections.
    ///
    /// - Parameters:
    ///   - detections: NMS-filtered detections in detector-input space.
    ///   - toOrigRect: closure that converts a DetectionInput into original-image pixel space.
    ///   - origW/origH: original camera frame size (for diagonal distance calculation).
    /// - Returns: Optional PrimarySelection, or nil if no suitable candidate found.
    func selectPrimary(
        from detections: [DetectionInput],
        toOrigRect: (DetectionInput) -> CGRect?,
        origW: Float,
        origH: Float
    ) -> PrimarySelection? {
        let candidates: [(det: DetectionInput, rect: CGRect)] = detections.compactMap { det in
            guard let r = toOrigRect(det) else { return nil }
            return (det, r)
        }

        guard let top1 = candidates.max(by: { $0.det.score < $1.det.score }) else {
            primaryLostCount += 1
            return nil
        }

        // First detection ever — accept immediately.
        if primaryRect == nil {
            primaryRect = top1.rect
            primaryScore = top1.det.score
            primaryClassId = top1.det.classId
            primaryClassCandidateId = top1.det.classId
            primaryClassCandidateCount = 0
            primaryLostCount = 0
            primaryScoreDropCount = 0
            return PrimarySelection(detection: top1.det, rect: top1.rect, primaryChanged: true)
        }

        let diag = sqrt(Double(origW * origW + origH * origH))
        let matches = candidates.filter { $0.det.classId == primaryClassId }
        let bestMatch = matches.max { rectIoU($0.rect, primaryRect!) < rectIoU($1.rect, primaryRect!) }

        if let match = bestMatch {
            let iou = rectIoU(match.rect, primaryRect!)
            let centerDist = rectCenterDistance(match.rect, primaryRect!)
            let scoreDiff = abs(Double(top1.det.score - match.det.score)) /
                            max(0.0001, Double(max(top1.det.score, match.det.score)))
            if scoreDiff < 0.15 && (iou >= 0.5 || centerDist <= 0.1 * diag) {
                // Still tracking same object.
                primaryRect = match.rect
                primaryScore = match.det.score
                primaryLostCount = 0
                primaryClassCandidateCount = 0
                return PrimarySelection(detection: match.det, rect: match.rect, primaryChanged: false)
            }
        } else {
            primaryLostCount += 1
            if primaryLostCount < 3 { return nil }
        }

        // Potential class / object switch — apply hysteresis.
        if top1.det.classId != primaryClassId {
            if top1.det.classId == primaryClassCandidateId {
                primaryClassCandidateCount += 1
            } else {
                primaryClassCandidateId = top1.det.classId
                primaryClassCandidateCount = 1
            }
            if primaryClassCandidateCount < classHysteresisThreshold {
                return nil
            }
        }

        let changed = (primaryClassId != top1.det.classId) || (primaryRect != top1.rect)
        primaryRect = top1.rect
        primaryScore = top1.det.score
        primaryClassId = top1.det.classId
        primaryClassCandidateId = top1.det.classId
        primaryClassCandidateCount = 0
        primaryLostCount = 0
        primaryScoreDropCount = 0
        return PrimarySelection(detection: top1.det, rect: top1.rect, primaryChanged: changed)
    }

    // MARK: - Drift classification

    /// Two-level drift classification between consecutive primary-object bboxes.
    ///
    /// - heavyDrift: large spatial change → stale embedding unreliable → re-encode
    ///   Triggers: iou < 0.15, center > 0.3·diag, area change > 1.0, ratio change > 0.60, score drop ≥ 2
    /// - lightDrift: moderate change → embedding still valid, mask stale → re-decode only
    ///   Triggers: iou 0.15–0.60, center 0.1–0.3·diag, area change 0.25–1.0, ratio change 0.20–0.60
    /// - noDrift: no significant change
    func classifyDrift(
        prev: CGRect,
        current: CGRect,
        prevScore: Float,
        currentScore: Float,
        origW: Float,
        origH: Float,
        debugEnabled: Bool = true
    ) -> DriftLevel {
        let diag = sqrt(Double(origW * origW + origH * origH))

        let iou = rectIoU(prev, current)
        let centerDist = rectCenterDistance(prev, current)
        let prevArea = Double(prev.width * prev.height)
        let currArea = Double(current.width * current.height)
        let areaChange = prevArea > 0 ? abs(currArea - prevArea) / prevArea : 0
        let prevRatio = Double(prev.width / max(prev.height, 1))
        let currRatio = Double(current.width / max(current.height, 1))
        let ratioChange = prevRatio > 0 ? abs(currRatio - prevRatio) / prevRatio : 0

        if currentScore < prevScore * 0.7 { primaryScoreDropCount += 1 } else { primaryScoreDropCount = 0 }

        // Heavy drift — large spatial change, stale embedding unreliable, re-encode needed.
        // Thresholds deliberately loose: only fires on large jumps.
        var heavyReasons: [String] = []
        if iou < 0.10              { heavyReasons.append(String(format: "iou<0.10(%.3f)", iou)) }
        if centerDist > 0.35 * diag { heavyReasons.append(String(format: "center>0.35*diag(%.1f)", centerDist)) }
        if areaChange > 1.5        { heavyReasons.append(String(format: "area>1.5(%.2f)", areaChange)) }
        if ratioChange > 0.80      { heavyReasons.append(String(format: "ratio>0.80(%.2f)", ratioChange)) }
        if primaryScoreDropCount >= 3 {
            heavyReasons.append(String(format: "scoreDrop>=3(prev=%.3f cur=%.3f)", prevScore, currentScore))
        }
        if !heavyReasons.isEmpty {
            if debugEnabled { diagLog("[SEG] heavy drift → re-encode: \(heavyReasons.joined(separator: ", "))") }
            return .heavyDrift
        }

        // Light drift — moderate change; embedding is still fresh, mask needs refresh.
        // Triggers at iou<0.55 or significant center/area/ratio shift.
        // Only fires when something meaningful changed — small jitter is tolerated.
        var lightReasons: [String] = []
        if iou < 0.55              { lightReasons.append(String(format: "iou<0.55(%.3f)", iou)) }
        if centerDist > 0.15 * diag { lightReasons.append(String(format: "center>0.15*diag(%.1f)", centerDist)) }
        if areaChange > 0.45       { lightReasons.append(String(format: "area>0.45(%.2f)", areaChange)) }
        if ratioChange > 0.35      { lightReasons.append(String(format: "ratio>0.35(%.2f)", ratioChange)) }
        if !lightReasons.isEmpty {
            if debugEnabled { diagLog("[SEG] light drift → re-decode: \(lightReasons.joined(separator: ", "))") }
            return .lightDrift
        }

        return .noDrift
    }

    // MARK: - Geometry helpers

    func rectIoU(_ a: CGRect, _ b: CGRect) -> Double {
        let ix1 = max(a.minX, b.minX); let iy1 = max(a.minY, b.minY)
        let ix2 = min(a.maxX, b.maxX); let iy2 = min(a.maxY, b.maxY)
        let inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        let areaA = max(0, a.width) * max(0, a.height)
        let areaB = max(0, b.width) * max(0, b.height)
        let union = areaA + areaB - inter
        return union <= 0 ? 0 : Double(inter / union)
    }

    func rectCenterDistance(_ a: CGRect, _ b: CGRect) -> Double {
        let dx = Double(a.midX - b.midX)
        let dy = Double(a.midY - b.midY)
        return sqrt(dx * dx + dy * dy)
    }
}
