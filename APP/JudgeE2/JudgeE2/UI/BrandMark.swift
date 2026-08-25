//
//  BrandMark.swift
//  JudgeE2
//
//  Phase 5 — UI (single-function app + brand visuals).
//
//  The vector twin of `Assets.xcassets/AppIcon.appiconset`: a red "J" drawn as
//  a map pin, standing in a red location ring, on white.  Same geometry
//  constants as the icon generator, so the two cannot drift apart.
//
//  The letter and the landmark are ONE closed contour: the head is a map-pin
//  teardrop (a circle closed off by the two tangents drawn from a virtual tip
//  below it), those tangents are cut where they have narrowed to exactly the
//  stem's width so the join is seamless, and the stem descends into the J's
//  hook.  The eyelet in the head is a real hole in the path — even-odd, not a
//  disc painted over the top — so the mark still reads correctly on a ground
//  that is not white.
//
//  Drawn in a `Canvas` rather than as stacked `Shape`s so the layout can be
//  fitted from the artwork's measured bounding box, exactly as the raster
//  generator does. Hand-placed offsets would have been a second source of
//  truth for the same drawing.
//

import SwiftUI

/// Brand colours.
///
/// ⚠️ Since the 2026-08-24 icon direction (white ground, red landmark) the
/// mark no longer quotes the mask palette, so this type is NOT a mirror of
/// `MaskRenderer`'s colours any more and nothing here is a C-7 surface.  The
/// overlay palette remains the FINAL cyan trio (architect_output.md §3.3.3);
/// icon and overlay are deliberately different families now.
enum BrandPalette {
    /// The landmark red.  White clears 4.67:1 against it, so the mark holds up
    /// on a light home screen and the counter stays legible.
    static let landmark = Color(red: 227 / 255, green: 30 / 255, blue: 46 / 255)

    /// Chrome accent: teal, darkened until white type on it clears 4.5:1
    /// (Y = 0.162 ⇒ 4.95:1).  Used for the Pin-count badge and mirrored by
    /// `Assets.xcassets/AccentColor`; unchanged by the icon revision.
    static let accent = Color(red: 0 / 255, green: 122 / 255, blue: 153 / 255)
}

/// Geometry shared with `scratchpad/make_icon.py`. Fractions of a 1000-unit
/// design square; the `Canvas` fits the result to whatever size it is given.
private enum JMark {
    static let headR: CGFloat = 140        // teardrop head radius
    static let headCY: CGFloat = 265       // head centre
    static let drop: CGFloat = 3.00        // virtual tip distance below it, × headR
    static let stemW: CGFloat = 0.58       // stem half-width, × headR
    static let stemBot: CGFloat = 675      // where the stem becomes the hook
    static let hookR: CGFloat = 165        // hook centreline radius
    static let counter: CGFloat = 0.40     // eyelet radius, × headR
    static let cx: CGFloat = 545
    static let ringR: CGFloat = 215        // inner ring semi-major axis
    static let ringSquash: CGFloat = 0.33  // ring perspective
    static let ringLW: CGFloat = 26        // strokes in design units, so they
    static let ringLWOuter: CGFloat = 20   // scale with the mark, not with pt
}


/// The icon artwork on its own, no wordmark.  Scales with `size`.
struct BrandMark: View {
    var size: CGFloat = 64

    /// Fraction of the square the artwork fills.  Matches the icon generator,
    /// which leaves this margin because iOS masks app icons to a squircle.
    private let fillRatio: CGFloat = 0.76

    var body: some View {
        Canvas { ctx, canvasSize in
            let n = min(canvasSize.width, canvasSize.height)
            let (letter, innerRing, outerRing) = Self.paths()

            // Fit by measured extents, counting the ring strokes' own width.
            let bounds = letter.boundingRect
                .union(innerRing.boundingRect.insetBy(dx: -JMark.ringLW / 2,
                                                      dy: -JMark.ringLW / 2))
                .union(outerRing.boundingRect.insetBy(dx: -JMark.ringLWOuter / 2,
                                                      dy: -JMark.ringLWOuter / 2))
            let scale = (n * fillRatio) / max(bounds.width, bounds.height)
            let fit = CGAffineTransform(translationX: n / 2, y: n / 2)
                .scaledBy(x: scale, y: scale)
                .translatedBy(x: -bounds.midX, y: -bounds.midY)

            ctx.stroke(outerRing.applying(fit), with: .color(BrandPalette.landmark.opacity(0.49)),
                       lineWidth: JMark.ringLWOuter * scale)
            ctx.stroke(innerRing.applying(fit), with: .color(BrandPalette.landmark),
                       lineWidth: JMark.ringLW * scale)
            // Even-odd: the silhouette is a single non-overlapping contour and
            // the eyelet is a second subpath strictly inside it, so this cuts a
            // real hole rather than painting a disc over the head.
            ctx.fill(letter.applying(fit), with: .color(BrandPalette.landmark),
                     style: FillStyle(eoFill: true))
        }
        .frame(width: size, height: size)
        .background(Color.white)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.225, style: .continuous))
        // The ground is white and so is a Settings row in light mode; without
        // this hairline the mark would have no edge at all there.
        .overlay(
            RoundedRectangle(cornerRadius: size * 0.225, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.14), lineWidth: 0.5)
        )
        .accessibilityHidden(true)
    }

    /// The sub-paths in design space: the J (silhouette plus its eyelet, to be
    /// filled even-odd) and the two rings.
    private static func paths() -> (Path, Path, Path) {
        let rh = JMark.headR
        let sw = rh * JMark.stemW
        let R = JMark.hookR
        let cx = JMark.cx
        let hy = JMark.headCY
        let stemBot = JMark.stemBot
        let hookCX = cx - R

        // Teardrop head: tangents from a virtual tip `d` below the centre.
        let d = JMark.drop * rh
        let beta = acos(rh / d)                    // contact angle from straight down
        let a0 = CGFloat.pi / 2 - beta             // right-hand contact
        let a1 = CGFloat.pi / 2 + beta             // left-hand contact
        let pr = CGPoint(x: cx + rh * cos(a0), y: hy + rh * sin(a0))
        // Walk the tangent until its half-width equals the stem's, and start
        // the stem there — that is what makes the join invisible.
        let t = 1 - sw / abs(pr.x - cx)
        let yStem = pr.y + t * ((hy + d) - pr.y)

        var letter = Path()
        letter.move(to: pr)
        addArc(&letter, cx, hy, rh, a0, a1 - 2 * .pi, 260)     // over the top of the head
        letter.addLine(to: CGPoint(x: cx - sw, y: yStem))       // left tangent
        letter.addLine(to: CGPoint(x: cx - sw, y: stemBot))     // left stem edge
        addArc(&letter, hookCX, stemBot, R - sw, 0, .pi, 140)   // hook, inner edge
        // ⛔ The terminal cap sweeps 0 → −π, not 0 → +π.  The centreline at the
        // hook's end is travelling upward, so the cap has to bulge upward; the
        // other way round it dips below the inner edge and cusps against it,
        // and the silhouette reads as a claw rather than a J.
        addArc(&letter, hookCX - R, stemBot, sw, 0, -.pi, 48)   // terminal cap
        addArc(&letter, hookCX, stemBot, R + sw, .pi, 0, 140)   // hook, outer edge
        letter.addLine(to: CGPoint(x: cx + sw, y: stemBot))     // right stem edge
        letter.addLine(to: CGPoint(x: cx + sw, y: yStem))       // right tangent
        letter.closeSubpath()

        // The eyelet, as a second subpath: even-odd turns it into a hole.
        let rr = rh * JMark.counter
        letter.addEllipse(in: CGRect(x: cx - rr, y: hy - rr, width: rr * 2, height: rr * 2))

        // Rings, centred so the J's foot sits *inside* the ring — the letter
        // stands in it rather than in front of it, which is what makes the
        // ring read as ground and not as a halo.
        let foot = stemBot + R + sw
        let ox = ((cx - 2 * R - sw) + (cx + rh)) / 2   // optical centre of the letter
        let ry = foot + JMark.ringR * JMark.ringSquash * 0.42

        func ring(_ rx: CGFloat) -> Path {
            Path(ellipseIn: CGRect(x: ox - rx, y: ry - rx * JMark.ringSquash,
                                   width: rx * 2, height: rx * JMark.ringSquash * 2))
        }
        return (letter, ring(JMark.ringR), ring(JMark.ringR * 1.46))
    }

    private static func addArc(_ p: inout Path, _ cx: CGFloat, _ cy: CGFloat,
                               _ r: CGFloat, _ a0: CGFloat, _ a1: CGFloat, _ steps: Int)
    {
        for i in 0...steps {
            let a = a0 + (a1 - a0) * CGFloat(i) / CGFloat(steps)
            p.addLine(to: CGPoint(x: cx + r * cos(a), y: cy + r * sin(a)))
        }
    }
}

/// Icon + name, for an About row or any place that needs the app to identify
/// itself in one line.
struct BrandLockup: View {
    var size: CGFloat = 56

    var body: some View {
        HStack(spacing: 14) {
            BrandMark(size: size)
            VStack(alignment: .leading, spacing: 2) {
                Text("JudgeE2")
                    .font(.system(size: size * 0.40, weight: .semibold, design: .rounded))
                Text("Tap to Segment")
                    .font(.system(size: size * 0.24, weight: .medium))
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("JudgeE2, Tap to Segment")
    }
}

#Preview {
    VStack(spacing: 24) {
        BrandMark(size: 160)
        BrandMark(size: 64)
        BrandMark(size: 32)
        BrandLockup()
    }
    .padding()
}
