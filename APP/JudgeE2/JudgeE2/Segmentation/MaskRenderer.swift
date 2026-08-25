//
//  MaskRenderer.swift
//  JudgeE2
//
//  Created by Jian Sun on 2/28/26.
//

import Accelerate
import CoreML
import Foundation
import UIKit

final class MaskRenderer {
    private let inputSize: CGFloat = 1024.0

    // ── Candidate acceptance thresholds (tap / multimask path) ────────────────
    // Corrupt logits do not produce "no candidate" — they produce *degenerate*
    // candidates: 1-px lines spanning a whole row/column and axis-aligned slabs.
    // A 4-px floor let those win the "smallest candidate" contest, so the
    // thresholds below reject on shape, not only on size.
    private static let minComponentPx      = 30       // in 256×256 mask space
    private static let minComponentSidePx  = 3        // bbox side; ≤2 px == line artefact
    private static let minComponentFill: Float = 0.05 // area / bbox area
    /// Numeric sentinel for the low-res logit tensor.  Device measurements
    /// (2026-08-01 session) show healthy taps reaching |logit| ≈ 65, so the
    /// earlier ±30/50 assumption was wrong and silently dropped ~13 % of taps.
    /// Real corruption on the A13 ANE fp16 path is |logit| ≈ 1e6–1e8, four
    /// orders of magnitude away; 500 sits between the two and still catches it.
    private static let maxPlausibleLogit: Float = 500.0

    /// Offset used by SAM's official `calculate_stability_score`
    /// (`area(logit > +delta) / area(logit > -delta)`).
    ///
    /// OBSERVABILITY ONLY — nothing downstream branches on the resulting score
    /// yet.  1.0 is SAM's own default; the real cut-off for this app will be
    /// picked once the on-device distribution of good vs. bad taps has been
    /// collected, so treat this value as un-calibrated.
    private static let stabilityDelta: Float = 1.0

    /// One 4-connected component: pixel count + bounding box in mask space.
    private struct Component {
        let count: Int
        let minX: Int, minY: Int, maxX: Int, maxY: Int
        var boxW: Int { maxX - minX + 1 }
        var boxH: Int { maxY - minY + 1 }
        var fill: Float { count > 0 ? Float(count) / Float(max(1, boxW * boxH)) : 0 }
    }

    /// One multimask candidate that survived the shape filters.
    private struct Candidate {
        let ch: Int
        let alpha: [UInt8]
        let comp: Component
        let iou: Float
        let stability: Float   // diagnostics only — never a selection input
    }

    /// The candidate that was actually drawn (multimask tap path only).
    /// Callers log these instead of `iouPreds.max()`, which is the *gate* value
    /// across all three candidates and does not describe the pick.
    struct SelectedCandidate {
        let ch: Int
        let iou: Float
        let area: Int      // pixels in 256×256 mask space
        let fill: Float    // area / bbox area
        let degraded: Bool // true when the 60 % cap fallback picked it
        /// SAM stability score of this candidate's logit field (see
        /// `stabilityDelta`).  Reported so the low-contrast failure mode can be
        /// characterised; it does NOT influence which candidate was picked.
        let stability: Float
    }

    /// renderMask output. `selected` is nil on the box path and on the
    /// single-mask tap fallback — there is no per-candidate choice to report.
    struct RenderResult {
        let image: CGImage
        let selected: SelectedCandidate?
    }

    /// `buildAlpha` output: the binarised 256×256 mask plus what was picked.
    /// Phase 3 Day 5 splits this out of `renderMask` so the multi-instance path
    /// can composite several alphas without re-running candidate selection.
    struct AlphaResult {
        let alpha: [UInt8]           // 256×256, values 0 or 255
        let selected: SelectedCandidate?
        let nonzeroCount: Int
    }

    /// One instance's contribution to a multi-mask overlay (Day 5, §3.3/§3.4).
    struct MaskLayer {
        let alpha: [UInt8]     // 256×256 from `buildTapAlpha`
        /// ⚠️ Must be a NON-dynamic colour.  `compositeLayers` resolves it with
        /// `getRed` on decoderQueue; a dynamic UIColor (`.systemBlue` & co.)
        /// resolves against the *background thread's* default trait collection
        /// there, so the overlay would disagree with the rest of the UI in dark
        /// mode (D-15).  Every entry of `TapInstanceManager.palette` is a plain
        /// sRGB literal for exactly this reason.
        let color: UIColor
        let opacity: CGFloat   // 0.60 primary / 0.40 secondary (§3.4)
    }

    // MARK: - Public entry points

    /// Phase 2 box path and the Day 4 single-instance tap path — unchanged
    /// behaviour and unchanged output bytes.  It is now a thin wrapper: all the
    /// decision logic lives in `buildAlpha`, all the geometry in `drawTile`.
    ///
    /// - Parameter tapPoint256: tap location in 256×256 mask space (point-prompt
    ///   mode only). When set, binarization uses SAM's absolute logit>0 threshold
    ///   and only the connected component containing the tap is kept — the
    ///   relative mean+0.5σ threshold with no spatial clipping lit up large
    ///   unrelated regions of the frame. Returns nil if no component covers the tap.
    /// - Parameter iouPreds: per-candidate iou_prediction from the multimask
    ///   decoder. Used only for the degraded fallback pick (every candidate over
    ///   the 60 % cap) so complex scenes return a coarse mask instead of nothing.
    /// - Parameter tapIndex: tap sequence number (`CameraManager.tapGeneration`).
    ///   Diagnostics only: when set, every line this call emits is stamped
    ///   `[TAP#N]` so one tap's whole chain greps out together. nil (box path)
    ///   leaves the legacy prefixes byte-identical.
    func renderMask(lowResMask: MLMultiArray, origW: Int, origH: Int, box: CGRect?,
                    tapPoint256: CGPoint? = nil, iouPreds: [Float]? = nil,
                    tapIndex: Int? = nil) -> RenderResult? {
        guard let built = buildAlpha(lowResMask: lowResMask, origW: origW, origH: origH,
                                     box: box, tapPoint256: tapPoint256,
                                     iouPreds: iouPreds, tapIndex: tapIndex) else { return nil }
        let tag = tapIndex.map { "[TAP#\($0)] " } ?? ""

        // Cyan at 60 % straight (non-premultiplied) alpha — the Phase 2 look.
        // Kept as a literal RGBA fill (not routed through `compositeLayers`) so
        // this path's output stays byte-identical to before the Day 5 split.
        let total = 256 * 256
        let cyanR: UInt8 = 0
        let cyanG: UInt8 = 217   // round(0.85 * 255)
        let cyanB: UInt8 = 255
        let cyanA: UInt8 = 153   // round(0.60 * 255)
        var rgba = [UInt8](repeating: 0, count: total * 4)  // zeros = fully transparent
        for i in 0..<total where built.alpha[i] > 0 {
            rgba[i * 4 + 0] = cyanR
            rgba[i * 4 + 1] = cyanG
            rgba[i * 4 + 2] = cyanB
            rgba[i * 4 + 3] = cyanA
        }
        guard let out = drawTile(rgba: rgba, origW: origW, origH: origH,
                                 nonzeroCount: built.nonzeroCount, tag: tag) else { return nil }
        return RenderResult(image: out, selected: built.selected)
    }

    /// Day 5 multi-instance entry point: binarise one decoder output into the
    /// 256×256 alpha the instance pool stores.  Runs the *same* `buildAlpha`
    /// the single-mask path runs — candidate selection, the 60 %/85 % caps, the
    /// shape gates and the flood fill are shared code, not a copy (Architect
    /// reserved item R3: Day 5 must not perturb the 88 % baseline).
    func buildTapAlpha(lowResMask: MLMultiArray, origW: Int, origH: Int,
                       tapPoint256: CGPoint, iouPreds: [Float]?,
                       tapIndex: Int?) -> AlphaResult? {
        buildAlpha(lowResMask: lowResMask, origW: origW, origH: origH, box: nil,
                   tapPoint256: tapPoint256, iouPreds: iouPreds, tapIndex: tapIndex)
    }

    /// Blend up to N instance masks into one overlay image (§3.4).
    ///
    /// `layers` must already be in draw order — oldest/secondary first, primary
    /// last — so the primary lands on top.  Overlaps are alpha-blended
    /// source-over rather than overwritten, so two overlapping instances stay
    /// mutually visible ("不相互遮挡") instead of the newer one punching a hole.
    func compositeLayers(_ layers: [MaskLayer], origW: Int, origH: Int,
                         tapIndex: Int? = nil) -> CGImage? {
        guard !layers.isEmpty else { return nil }
        let width = 256, height = 256, total = width * height
        let tag = tapIndex.map { "[TAP#\($0)] " } ?? ""

        // Straight-alpha accumulation buffer, float to avoid rounding drift
        // across three source-over passes.
        var accR = [Float](repeating: 0, count: total)
        var accG = [Float](repeating: 0, count: total)
        var accB = [Float](repeating: 0, count: total)
        var accA = [Float](repeating: 0, count: total)

        for (idx, layer) in layers.enumerated() {
            // A dropped layer is an instance silently vanishing from the screen
            // (D-14).  `drawableInstances()` filters out the nil-alpha case, so
            // neither branch is reachable today — which is precisely why they
            // have to be loud if they ever become reachable.
            guard layer.alpha.count == total else {
                faultLog("\(tag)[MASK] composite dropped layer \(idx): alpha count \(layer.alpha.count) != \(total)")
                continue
            }
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            guard layer.color.getRed(&r, green: &g, blue: &b, alpha: &a) else {
                faultLog("\(tag)[MASK] composite dropped layer \(idx): colour is not RGB-convertible (\(layer.color))")
                continue
            }
            let sr = Float(r), sg = Float(g), sb = Float(b)
            let sa = Float(max(0, min(1, layer.opacity)))
            for i in 0..<total where layer.alpha[i] > 0 {
                let dstA = accA[i]
                let outA = sa + dstA * (1 - sa)
                guard outA > 0 else { continue }
                accR[i] = (sr * sa + accR[i] * dstA * (1 - sa)) / outA
                accG[i] = (sg * sa + accG[i] * dstA * (1 - sa)) / outA
                accB[i] = (sb * sa + accB[i] * dstA * (1 - sa)) / outA
                accA[i] = outA
            }
        }

        var rgba = [UInt8](repeating: 0, count: total * 4)
        var nonzero = 0
        for i in 0..<total where accA[i] > 0 {
            rgba[i * 4 + 0] = UInt8(max(0, min(255, (accR[i] * 255).rounded())))
            rgba[i * 4 + 1] = UInt8(max(0, min(255, (accG[i] * 255).rounded())))
            rgba[i * 4 + 2] = UInt8(max(0, min(255, (accB[i] * 255).rounded())))
            rgba[i * 4 + 3] = UInt8(max(0, min(255, (accA[i] * 255).rounded())))
            nonzero += 1
        }
        return drawTile(rgba: rgba, origW: origW, origH: origH,
                        nonzeroCount: nonzero, tag: tag)
    }

    // MARK: - Outline tracing (§3.4 two-tone stroke — geometry only)

    /// Trace the boundary of one instance's 256×256 alpha and return it in
    /// **normalised canvas coordinates** ([0,1] over the origW × origH canvas).
    ///
    /// ⚠️ **C-5.**  This produces vector geometry and nothing else — the stroke
    /// is never rasterised here.  Widths live in pt on the `CAShapeLayer` that
    /// `PreviewView` builds, so they stay 2.0 / 1.5 pt regardless of the 4–8×
    /// upscale the fill tile goes through, and the future D-13 refactor (hand
    /// the 256×256 tile to the GPU) leaves this path untouched.
    ///
    /// Reads only the binarised alpha, so it cannot perturb candidate selection
    /// or any threshold (reserved item R3).
    func traceOutline(alpha: [UInt8], origW: Int, origH: Int) -> [[CGPoint]] {
        let width = 256, height = 256
        guard alpha.count == width * height else { return [] }

        // ---- 1. Emit the directed unit edges of the region boundary ----------
        // For every foreground pixel, each side whose neighbour is background is
        // one boundary edge.  Emitting them in a fixed rotational order (top →
        // right → bottom → left, i.e. clockwise in the y-down mask space) makes
        // the edges of a blob chain into closed loops with no ambiguity, and
        // holes come out as their own loops — which is what we want, a hole's
        // rim is a real mask boundary and should be stroked.
        //
        // Vertices sit on pixel corners, so doubling them makes every
        // coordinate an exact Int: no floating-point keys, no epsilon matching.
        @inline(__always) func filled(_ x: Int, _ y: Int) -> Bool {
            guard x >= 0, x < width, y >= 0, y < height else { return false }
            return alpha[y * width + x] > 0
        }
        @inline(__always) func key(_ x: Int, _ y: Int) -> Int { y * (width + 2) + x }

        // start-vertex key → list of end-vertex keys
        var next = [Int: [Int]]()
        next.reserveCapacity(1024)
        @inline(__always) func edge(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) {
            next[key(x0, y0), default: []].append(key(x1, y1))
        }

        for y in 0..<height {
            for x in 0..<width where alpha[y * width + x] > 0 {
                // Corner coordinates of pixel (x, y) in "corner units".
                let l = x, r = x + 1, t = y, b = y + 1
                if !filled(x, y - 1) { edge(l, t, r, t) }   // top:    left → right
                if !filled(x + 1, y) { edge(r, t, r, b) }   // right:  top → bottom
                if !filled(x, y + 1) { edge(r, b, l, b) }   // bottom: right → left
                if !filled(x - 1, y) { edge(l, b, l, t) }   // left:   bottom → top
            }
        }
        guard !next.isEmpty else { return [] }

        // ---- 2. Chain the edges into closed loops ---------------------------
        @inline(__always) func point(_ k: Int) -> CGPoint {
            CGPoint(x: CGFloat(k % (width + 2)), y: CGFloat(k / (width + 2)))
        }
        // Every remaining value is non-empty (a key is dropped when drained), so
        // consuming edges terminates: each iteration removes exactly one edge.
        var loops: [[CGPoint]] = []
        while let start = next.keys.first {
            var loop: [CGPoint] = []
            var cursor = start
            while let outgoing = next[cursor] {
                var rest = outgoing
                let to = rest.removeLast()
                if rest.isEmpty { next.removeValue(forKey: cursor) } else { next[cursor] = rest }
                loop.append(point(cursor))
                cursor = to
                if cursor == start { break }
            }
            if loop.count >= 4 { loops.append(loop) }
        }

        // ---- 3. Round the staircase off once (Chaikin) ----------------------
        // The raw loop is an axis-aligned staircase; at ~3.5 pt per mask pixel
        // on an iPhone 11 preview those steps read as jaggies against a fill
        // that CoreAnimation has smoothed.  One Chaikin pass replaces each
        // corner with a 45° chamfer — cheap, closed-form, and it never moves a
        // vertex further than half a mask pixel, so the outline still traces the
        // same binary alpha the fill was built from.
        // ---- 4. Normalise into canvas space --------------------------------
        let rect = tileRect(origW: origW, origH: origH)
        let sx = rect.width / CGFloat(width) / CGFloat(origW)
        let sy = rect.height / CGFloat(height) / CGFloat(origH)
        let ox = rect.minX / CGFloat(origW)
        let oy = rect.minY / CGFloat(origH)

        return loops.map { loop in
            var out: [CGPoint] = []
            out.reserveCapacity(loop.count * 2)
            for i in 0..<loop.count {
                let p = loop[i]
                let q = loop[(i + 1) % loop.count]
                out.append(CGPoint(x: ox + (p.x * 0.75 + q.x * 0.25) * sx,
                                   y: oy + (p.y * 0.75 + q.y * 0.25) * sy))
                out.append(CGPoint(x: ox + (p.x * 0.25 + q.x * 0.75) * sx,
                                   y: oy + (p.y * 0.25 + q.y * 0.75) * sy))
            }
            return out
        }
    }

    // MARK: - Alpha construction (decision logic — unchanged since Day 4)

    /// Binarise the decoder output into a 256×256 alpha and report which
    /// candidate was drawn.  Extracted verbatim from the Day 4 `renderMask`;
    /// every threshold, cap, gate and log line below is the pre-Day-5 code.
    private func buildAlpha(lowResMask: MLMultiArray, origW: Int, origH: Int, box: CGRect?,
                            tapPoint256: CGPoint?, iouPreds: [Float]?,
                            tapIndex: Int?) -> AlphaResult? {
        let width = 256
        let height = 256
        let total = width * height

        // Log prefixes — `tag` prepends the tap stamp to lines that carry no
        // bracket prefix of their own; `tapTag` replaces the bare `[TAP]` one.
        let tag = tapIndex.map { "[TAP#\($0)] " } ?? ""
        let tapTag = tapIndex.map { "[TAP#\($0)]" } ?? "[TAP]"

        // Accept [1,1,256,256], [1,3,256,256], or [3,256,256]
        let shape = lowResMask.shape.map { $0.intValue }
        let count = lowResMask.count
        let validCounts = [1 * 1 * 256 * 256, 1 * 3 * 256 * 256, 3 * 256 * 256]
        guard validCounts.contains(count) else { return nil }

        let channels = (shape.count == 4) ? shape[1] : (shape.count == 3 ? shape[0] : 1)
        guard let logits = extractLogits(lowResMask, channels: channels,
                                         width: width, height: height,
                                         tag: tag) else { return nil }

        @inline(__always) func valueAt(c: Int, y: Int, x: Int) -> Float {
            logits[c * total + y * width + x]
        }

        /// SAM's official `calculate_stability_score` for one candidate channel:
        /// `count(logit > +delta) / count(logit > -delta)`, 0 when the
        /// denominator is empty.  A crisp object gives ≈1 (moving the threshold
        /// barely moves the area); a flat, low-contrast logit field gives ≈0.
        ///
        /// CONSTRAINT: both counts are taken over the FULL 256×256 field, not
        /// over the flood-filled component this candidate ends up drawing.  That
        /// is the official definition, and it is the point of the metric — the
        /// failure mode being measured is the *surrounding* field going flat.
        /// Restricting the counts to the connected component would silently
        /// change the semantics; do not "tidy" it that way.
        ///
        /// Reads the already-materialised `logits` buffer from `extractLogits`;
        /// it must not reopen the MLMultiArray (IOSurface-backed `dataPointer`
        /// is unsafe outside `withUnsafeBufferPointer`).  Both counts are
        /// accumulated in a single traversal.
        @inline(__always) func stabilityScore(channel c: Int) -> Float {
            let base = c * total
            var areaHigh = 0
            var areaLow  = 0
            for i in base..<(base + total) {
                let v = logits[i]
                if v > -Self.stabilityDelta {
                    areaLow += 1
                    if v > Self.stabilityDelta { areaHigh += 1 }
                }
            }
            return areaLow > 0 ? Float(areaHigh) / Float(areaLow) : 0
        }

        var alpha = [UInt8](repeating: 0, count: total)

        /// True when the multimask candidate branch below will run.  That branch
        /// binarises each candidate at SAM's absolute `logit > 0` and never
        /// reads `thresh`, so everything feeding `thresh` is dead work on the
        /// tap path — 65 k floats copied and sorted per tap, measured at
        /// 5.96 ms on Mac / 79 % of the Swift-side work (Debugger D-6).
        /// Gating it here removes the cost without touching a single decision:
        /// the alpha this function returns is bit-identical either way.
        let usesMultimask = (tapPoint256 != nil && channels > 1)

        // Global stats over channel 0 (threshold inputs for the box path).
        var minV: Float = 1e9
        var maxV: Float = -1e9
        var sum: Float = 0
        var sumsq: Float = 0
        // Percentile-threshold scratch — only the box-less single-mask branch
        // reads it, so it is not even allocated on the multimask tap path.
        var allLogits = [Float](repeating: 0, count: usesMultimask ? 0 : total)
        for y in 0..<height {
            for x in 0..<width {
                let v = valueAt(c: 0, y: y, x: x)
                if !usesMultimask { allLogits[y * width + x] = v }
                minV = min(minV, v)
                maxV = max(maxV, v)
                sum += v
                sumsq += v * v
            }
        }
        let mean = sum / Float(total)
        let varv = max(0, (sumsq / Float(total)) - mean * mean)
        let std = sqrt(varv)

        // Numeric sentinel over ALL channels — a corrupt tensor MUST NOT reach the
        // screen; rendering it is exactly what produced the half-frame slabs and
        // 1-px line masks the user reported.
        var maxAbs: Float = 0
        var nonFinite = 0
        for v in logits {
            if v.isFinite { maxAbs = max(maxAbs, abs(v)) } else { nonFinite += 1 }
        }
        if nonFinite > 0 || maxAbs > Self.maxPlausibleLogit {
            faultLog(String(format: "%@[MASK] garbage logits rejected — nonFinite=%d |max|=%.4g (limit %.0f) shape=%@",
                         tag, nonFinite, maxAbs, Self.maxPlausibleLogit, String(describing: shape)))
            return nil
        }

        // ---- Compute maskBox in 256x256 mask space ----
        var maskBox: CGRect? = nil
        if let b = box, origW > 0, origH > 0 {
            let origWf = CGFloat(origW)
            let origHf = CGFloat(origH)
            let scale = inputSize / max(origWf, origHf)
            let newW = origWf * scale
            let newH = origHf * scale
            let padX = (inputSize - newW) * 0.5
            let padY = (inputSize - newH) * 0.5
            let toMaskScale = CGFloat(width) / inputSize  // 256/1024 = 0.25

            let x1_input = b.minX * scale + padX
            let y1_input = b.minY * scale + padY
            let x2_input = (b.minX + b.width) * scale + padX
            let y2_input = (b.minY + b.height) * scale + padY

            let mx1 = x1_input * toMaskScale
            let my1 = y1_input * toMaskScale
            let mx2 = x2_input * toMaskScale
            let my2 = y2_input * toMaskScale

            var mb = CGRect(x: mx1, y: my1, width: mx2 - mx1, height: my2 - my1)
            if mb.width < 0 { mb.size.width = 0 }
            if mb.height < 0 { mb.size.height = 0 }

            mb.origin.x = max(0, min(CGFloat(width),  floor(mb.origin.x)))
            mb.origin.y = max(0, min(CGFloat(height), floor(mb.origin.y)))
            mb.size.width  = min(CGFloat(width)  - mb.origin.x, ceil(mb.size.width))
            mb.size.height = min(CGFloat(height) - mb.origin.y, ceil(mb.size.height))

            if mb.width > 0 && mb.height > 0 { maskBox = mb }

            diagLog("\(tag)maskBox (mask-space) = \(String(describing: maskBox)), cropRect (input-space) = CGRect(x:\(padX), y:\(padY), width:\(newW), height:\(newH))")
        }

        // ---- Local threshold: computed within maskBox, not globally ----
        // Both branches are skipped on the multimask tap path (`usesMultimask`):
        // there `thresh` is written, logged and never read, and the p30 branch
        // pays a 65 k-element sort for it (D-6).  The seed value below is what
        // the box path's degenerate case already used.
        var thresh: Float = mean + 0.5 * std
        if !usesMultimask, let mb = maskBox {
            let iy1 = max(0, Int(mb.minY))
            let iy2 = min(height, Int(mb.maxY))
            let ix1 = max(0, Int(mb.minX))
            let ix2 = min(width, Int(mb.maxX))

            var localSum: Float = 0
            var localSumSq: Float = 0
            var localMax: Float = -1e9
            var localCount: Int = 0

            for ly in iy1..<iy2 {
                for lx in ix1..<ix2 {
                    let v = valueAt(c: 0, y: ly, x: lx)
                    localMax = max(localMax, v)
                    localSum += v
                    localSumSq += v * v
                    localCount += 1
                }
            }

            if localCount > 0 {
                let localMean = localSum / Float(localCount)
                let localVar = max(0, localSumSq / Float(localCount) - localMean * localMean)
                let localStd = sqrt(localVar)
                let adaptive = localMean + 0.5 * localStd
                thresh = min(adaptive, localMax - 1.0)
            } else {
                thresh = mean + 0.5 * std
            }
        } else if !usesMultimask {
            // No box (single-mask tap fallback): adaptive threshold to prevent
            // full-screen masks.
            //
            // Problem: mean+0.5σ works for small objects (background pulls mean
            // negative) but fails for large objects (object pulls mean positive,
            // so the threshold is too permissive and fills the entire frame).
            //
            // Fix: take MAX(mean+0.5σ, 30th-percentile-from-top threshold).
            //   • Small objects: mean+0.5σ is already tighter than p30 → unchanged.
            //   • Large objects: p30 caps coverage at 30% of the mask → no full-screen.
            // Connected-component filtering (tapPoint256 path below) then removes
            // disconnected background blobs regardless.
            let baseThresh = mean + 0.5 * std
            let maxPx = total * 30 / 100        // 30% cap = ~19,661 px
            let sortedDesc = allLogits.sorted(by: >)
            let p30Thresh = (maxPx > 0 && maxPx < total) ? sortedDesc[maxPx] : maxV
            thresh = max(baseThresh, p30Thresh)
            if thresh > baseThresh {
                diagLog(String(format: "%@[MASK] adaptive cap: base=%.3f → p30=%.3f (large-object guard)",
                             tag, baseThresh, thresh))
            }
        }

        // ---- Fill alpha: solid binary (no gradient / polka dots) ----
        // NOTE: thresh is NOT overridden to 0.0 for tap mode any more.
        // Doing so caused full-screen coverage for large objects because SAM
        // assigns positive logits across the entire foreground.  The adaptive
        // cap above handles large objects; connected-component filtering below
        // handles scattered background blobs.

        var nonzeroCount = 0
        var selectedCandidate: SelectedCandidate? = nil

        if let tap = tapPoint256, channels > 1 {
            // ---- Multimask candidate selection (tap mode, 3-candidate decoder) ----
            // Each candidate is binarized at SAM's absolute logit>0 (valid
            // per-candidate — the ambiguity lives across candidates, not in the
            // threshold), flood-filled from the tap, then filtered on size AND
            // shape.  The smallest survivor wins: the most specific object under
            // the tap.  A 60%-of-content-area cap rejects the whole-scene
            // candidate; an 85% cap gates the degraded fallback below.
            let contentScale = 256.0 / max(CGFloat(origW), CGFloat(origH))
            let contentPx = max(1, Int((CGFloat(origW) * contentScale) * (CGFloat(origH) * contentScale)))
            let cap60 = contentPx * 60 / 100
            let cap85 = contentPx * 85 / 100

            var candidates: [Candidate] = []
            var rejections: [String] = []
            for c in 0..<channels {
                var a = [UInt8](repeating: 0, count: total)
                for y in 0..<height {
                    for x in 0..<width {
                        if valueAt(c: c, y: y, x: x) > 0 { a[y * width + x] = 255 }
                    }
                }
                let comp = keepComponentContaining(tap, alpha: &a, width: width, height: height)
                let iou = (iouPreds?.indices.contains(c) ?? false) ? iouPreds![c] : 0
                // Computed for every channel, including the ones rejected below —
                // the rejected side of the distribution is exactly what has to be
                // characterised.  Diagnostics only: no branch below reads `stab`.
                let stab = stabilityScore(channel: c)
                guard comp.count > 0 else {
                    rejections.append(String(format: "ch%d:no-component-at-tap(stab=%.2f)", c, stab))
                    continue
                }
                if comp.count < Self.minComponentPx {
                    rejections.append(String(format: "ch%d:tiny(%dpx stab=%.2f)", c, comp.count, stab))
                    continue
                }
                if min(comp.boxW, comp.boxH) < Self.minComponentSidePx {
                    rejections.append(String(format: "ch%d:line(%dx%d stab=%.2f)", c, comp.boxW, comp.boxH, stab))
                    continue
                }
                if comp.fill < Self.minComponentFill {
                    rejections.append(String(format: "ch%d:sparse(fill=%.3f stab=%.2f)", c, comp.fill, stab))
                    continue
                }
                candidates.append(Candidate(ch: c, alpha: a, comp: comp, iou: iou, stability: stab))
            }

            let areas = candidates.map {
                String(format: "ch%d=%dpx bbox=%dx%d fill=%.2f iou=%.2f stab=%.2f",
                       $0.ch, $0.comp.count, $0.comp.boxW, $0.comp.boxH, $0.comp.fill,
                       $0.iou, $0.stability)
            }.joined(separator: " | ")

            let selected: Candidate?
            var degradedPick = false
            if let sel = candidates.sorted(by: { $0.comp.count < $1.comp.count })
                                   .first(where: { $0.comp.count <= cap60 }) {
                diagLog("\(tapTag) candidates: [\(areas)] → picked ch\(sel.ch) area=\(sel.comp.count) (cap=\(cap60)px)")
                selected = sel
            } else if let fb = candidates.filter({ $0.comp.count <= cap85 })
                                         .max(by: { $0.iou < $1.iou }) {
                // Degraded pick: complex scenes where every candidate exceeds the
                // 60 % cap.  A coarse mask beats no feedback at all; the
                // iou_pred ≥ 0.1 gate upstream still rejects a missed tap.
                diagLog(String(format: "%@ all candidates over 60%% cap → degraded pick ch%d area=%d iou=%.3f (cap85=%dpx) | [%@]",
                             tapTag, fb.ch, fb.comp.count, fb.iou, cap85, areas))
                selected = fb
                degradedPick = true
            } else {
                selected = nil
            }

            guard let sel = selected else {
                faultLog("\(tapTag) no valid candidate at tap (\(Int(tap.x)),\(Int(tap.y))) — kept: [\(areas)] rejected: [\(rejections.joined(separator: " "))] cap60=\(cap60)px cap85=\(cap85)px")
                return nil
            }
            alpha = sel.alpha
            nonzeroCount = sel.comp.count
            selectedCandidate = SelectedCandidate(ch: sel.ch, iou: sel.iou,
                                                  area: sel.comp.count, fill: sel.comp.fill,
                                                  degraded: degradedPick,
                                                  stability: sel.stability)
        } else {
            // Box mode, or single-mask fallback in tap mode (adaptive thresh above).
            for y in 0..<height {
                for x in 0..<width {
                    if let mb = maskBox, !mb.contains(CGPoint(x: x, y: y)) {
                        alpha[y * width + x] = 0
                        continue
                    }
                    let v = valueAt(c: 0, y: y, x: x)
                    if v > thresh {
                        alpha[y * width + x] = 255  // solid, no gradient
                        nonzeroCount += 1
                    } else {
                        alpha[y * width + x] = 0
                    }
                }
            }

            if let tap = tapPoint256 {
                let comp = keepComponentContaining(tap, alpha: &alpha,
                                                   width: width, height: height)
                nonzeroCount = comp.count
                guard comp.count >= Self.minComponentPx,
                      min(comp.boxW, comp.boxH) >= Self.minComponentSidePx else {
                    faultLog("\(tag)Mask: no usable component at tap (\(Int(tap.x)),\(Int(tap.y))) — area=\(comp.count) bbox=\(comp.boxW)x\(comp.boxH)")
                    return nil
                }
            }
        }
        // `thresh` is reported as `n/a` on the multimask path rather than
        // printing a number nothing used — the old line invited exactly the
        // misreading that the tap mask had been thresholded at that value.
        let threshDesc = usesMultimask ? "n/a(multimask: logit>0 per candidate)" : "\(thresh)"
        diagLog("\(tag)Mask logits range: min=\(minV), max=\(maxV) | mean=\(mean), std=\(std), thresh=\(threshDesc) | nonzero=\(nonzeroCount)/\(total) | shape=\(shape)")

        return AlphaResult(alpha: alpha, selected: selectedCandidate, nonzeroCount: nonzeroCount)
    }

    // MARK: - Tile drawing (pure geometry — unchanged since Day 4)

    /// Draw a 256×256 straight-alpha RGBA tile into an (origW × origH) canvas so
    /// the non-padded content region fills the canvas exactly.  Extracted
    /// verbatim from the Day 4 `renderMask`; both the single-mask and the
    /// multi-instance paths go through it, so their geometry cannot drift apart.
    /// Where the 256×256 mask tile lands inside the (origW × origH) canvas.
    ///
    /// Factored out of `drawTile` (values unchanged) so the outline tracer maps
    /// mask-space contours through the *same* letterbox arithmetic the fill goes
    /// through.  Two copies of this maths would be two chances for the outline
    /// to drift off the fill.
    private func tileRect(origW: Int, origH: Int) -> CGRect {
        // Letterbox geometry — mirrors PromptBuilder so pixel coords align.
        let origWf   = CGFloat(origW)
        let origHf   = CGFloat(origH)
        let samScale = inputSize / max(origWf, origHf)
        let newW     = origWf * samScale
        let newH     = origHf * samScale
        let padX     = (inputSize - newW) * 0.5
        let padY     = (inputSize - newH) * 0.5

        // Example (portrait 1080×1920): drawW=1920, drawH=1920, drawX=-420, drawY=0
        //   → the 256×256 tile renders at 1920×1920, clipped left 420 px, showing 1080×1920.
        return CGRect(x: -padX * origWf / newW,
                      y: -padY * origHf / newH,
                      width:  inputSize * origWf / newW,
                      height: inputSize * origHf / newH)
    }

    private func drawTile(rgba: [UInt8], origW: Int, origH: Int,
                          nonzeroCount: Int, tag: String) -> CGImage? {
        let width = 256
        let height = 256

        // ---- v3: UIGraphicsImageRenderer — no CIImage, no CIFalseColor ----
        //
        // v1 (grayscale+CIFalseColor): CIFalseColor drops alpha on single-channel CIImage.
        // v2 (RGBA+CIImage): CIImage y-flip + createCGImage format uncertainty still broke rendering.
        // v3 (this): UIGraphicsImageRenderer with explicit alpha, UIKit y-origin=top (same as alpha[]).
        guard let rgbaProvider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        guard let maskCGImage = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), // straight alpha
            provider: rgbaProvider, decode: nil, shouldInterpolate: true,
            intent: .defaultIntent) else { return nil }

        let origWf = CGFloat(origW)
        let origHf = CGFloat(origH)
        let samScale = inputSize / max(origWf, origHf)
        let newW = origWf * samScale
        let newH = origHf * samScale
        let padX = (inputSize - newW) * 0.5
        let padY = (inputSize - newH) * 0.5

        // Draw the 256×256 tile into a (origW × origH) canvas so that the non-padded
        // content region fills the canvas exactly.  UIKit origin = top-left (same as
        // alpha[]) so no extra y-flip is required.
        let drawRect = tileRect(origW: origW, origH: origH)

        let cropRect = CGRect(x: padX, y: padY, width: newW, height: newH)
        diagLog("\(tag)cropRect=\(cropRect) outRect=(0,0,\(origWf),\(origHf)) drawRect=\(drawRect) nonzero=\(nonzeroCount)")

        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale  = 1.0    // 1 pt = 1 px; avoid Retina upscale of the mask tile
        fmt.opaque = false  // preserve alpha channel in output
        let uiRenderer = UIGraphicsImageRenderer(
            size: CGSize(width: origWf, height: origHf), format: fmt)
        let uiImg = uiRenderer.image { _ in
            UIImage(cgImage: maskCGImage).draw(in: drawRect)
        }
        return uiImg.cgImage
    }

    /// Copy the decoder output into a contiguous [C][H][W] Float buffer.
    ///
    /// Access goes through `withUnsafeBufferPointer(ofType:)`, never the raw
    /// `dataPointer`: a CoreML output backed by an IOSurface / pixel buffer (a
    /// GPU-resident decoder output is one) is only guaranteed valid and
    /// synchronised inside that call.  Reading the bare pointer returns
    /// unsynchronised memory — intermittent garbage logits, occasional crash.
    /// The stride index math is unchanged; only the access API is.
    private func extractLogits(_ arr: MLMultiArray, channels: Int,
                               width: Int, height: Int, tag: String = "") -> [Float]? {
        let shape = arr.shape.map { $0.intValue }
        let s = arr.strides.map { $0.intValue }
        let sC: Int, sY: Int, sX: Int
        switch shape.count {
        case 4: (sC, sY, sX) = (s[1], s[2], s[3])   // [1, C, H, W]
        case 3: (sC, sY, sX) = (s[0], s[1], s[2])   // [C, H, W]
        default: (sC, sY, sX) = (width * height, width, 1)
        }

        // Bounds contract: the highest index we will touch must exist.
        let maxIndex = (channels - 1) * sC + (height - 1) * sY + (width - 1) * sX
        guard maxIndex < arr.count else {
            faultLog("\(tag)[MASK] stride/shape mismatch: maxIndex=\(maxIndex) count=\(arr.count) shape=\(shape)")
            return nil
        }

        let plane = width * height
        var out = [Float](repeating: 0, count: channels * plane)

        guard arr.dataType == .float32 else {
            // Both bundled decoders export FLOAT32 (verified from the mlmodel
            // protobuf).  Anything else goes through the slow NSNumber path
            // instead of being mis-typed into Float.
            faultLog("\(tag)[MASK] unexpected mask dataType raw=\(arr.dataType.rawValue) — NSNumber read path")
            for c in 0..<channels {
                for y in 0..<height {
                    for x in 0..<width {
                        out[c * plane + y * width + x] = arr[c * sC + y * sY + x * sX].floatValue
                    }
                }
            }
            return out
        }

        out.withUnsafeMutableBufferPointer { dst in
            arr.withUnsafeBufferPointer(ofType: Float.self) { src in
                for c in 0..<channels {
                    for y in 0..<height {
                        let srcRow = c * sC + y * sY
                        let dstRow = c * plane + y * width
                        if sX == 1 {
                            for x in 0..<width { dst[dstRow + x] = src[srcRow + x] }
                        } else {
                            for x in 0..<width { dst[dstRow + x] = src[srcRow + x * sX] }
                        }
                    }
                }
            }
        }
        return out
    }

    /// Keep only the 4-connected binary component containing `tap`; zero the rest.
    /// The seed tolerates a ±2 px miss (tap landing just off the object edge in
    /// 256-space). Returns the surviving component's pixel count and bounding box
    /// (count = 0 means nothing under the tap).
    private func keepComponentContaining(_ tap: CGPoint, alpha: inout [UInt8],
                                         width: Int, height: Int) -> Component {
        let tx = max(0, min(width - 1, Int(tap.x.rounded())))
        let ty = max(0, min(height - 1, Int(tap.y.rounded())))

        // Find a set seed pixel at/near the tap.
        var seed: Int? = nil
        outer: for r in 0...2 {
            for dy in -r...r {
                for dx in -r...r {
                    let x = tx + dx, y = ty + dy
                    guard x >= 0, x < width, y >= 0, y < height else { continue }
                    if alpha[y * width + x] != 0 { seed = y * width + x; break outer }
                }
            }
        }
        guard let start = seed else {
            for i in 0..<alpha.count { alpha[i] = 0 }
            return Component(count: 0, minX: 0, minY: 0, maxX: 0, maxY: 0)
        }

        // BFS flood fill marking the tap's component as 1 (alpha values are 0/255,
        // so 1 is a safe temporary marker).  The bounding box is accumulated in the
        // same pass — the shape filters need it.
        var queue = [start]
        alpha[start] = 1
        var head = 0
        var count = 1
        var minX = start % width, maxX = minX
        var minY = start / width, maxY = minY
        while head < queue.count {
            let idx = queue[head]; head += 1
            let x = idx % width, y = idx / width
            if x < minX { minX = x } else if x > maxX { maxX = x }
            if y < minY { minY = y } else if y > maxY { maxY = y }
            if x > 0,          alpha[idx - 1] == 255     { alpha[idx - 1] = 1;     queue.append(idx - 1);     count += 1 }
            if x < width - 1,  alpha[idx + 1] == 255     { alpha[idx + 1] = 1;     queue.append(idx + 1);     count += 1 }
            if y > 0,          alpha[idx - width] == 255 { alpha[idx - width] = 1; queue.append(idx - width); count += 1 }
            if y < height - 1, alpha[idx + width] == 255 { alpha[idx + width] = 1; queue.append(idx + width); count += 1 }
        }

        for i in 0..<alpha.count {
            alpha[i] = (alpha[i] == 1) ? 255 : 0
        }
        return Component(count: count, minX: minX, minY: minY, maxX: maxX, maxY: maxY)
    }
}
