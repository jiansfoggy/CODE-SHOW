//
//  MaskOutline.swift
//  JudgeE2
//
//  Phase 3 — Day 5 (Builder), presentation spec revision.
//  Contract: architect_output.md §3.3.1 (L1/L2/L3 responsibility model),
//  §3.3.2 (C-1 … C-7), §3.4 (two-tone outline, primary / secondary tiers).
//
//  WHY THIS FILE EXISTS
//  --------------------
//  §3.3.1 L1 ("there is a selected region on screen") must NOT be carried by
//  hue: the fill colour competes with the user's own scene, which is an
//  uncontrollable input.  The outline is the only visibility carrier that
//  brings its own contrast — a dark outer ring plus a light inner line means
//  that whatever the background luminance is, at least one of the two lines
//  steps away from it.
//
//  ⚠️ ARCHITECTURE CONSTRAINT C-5 — the reason the geometry below is
//  *normalised* rather than rasterised:  the outline must be generated in the
//  SCREEN coordinate system with its width expressed in **pt**.  Burning it
//  into the 256×256 alpha tile would multiply the line width by the tile's
//  4–8× upscale and would also lock out D-13 (hand the 256×256 rasterisation
//  to the GPU via `contentsRect`).  So `MaskOutline` carries pure geometry —
//  polygons in a unitless [0,1] canvas space — and `PreviewView` turns them
//  into `CAShapeLayer` paths at layout time, where 1 pt really is 1 pt.
//

import CoreGraphics
import UIKit

// MARK: - Geometry

/// One instance's mask boundary, ready for stroking on the preview.
///
/// `polygons` are closed loops (outer contour first is NOT guaranteed — holes
/// are included as separate loops and are stroked exactly like the outer
/// contour, which is correct: a hole's rim is a real mask boundary).
///
/// Coordinates are **normalised against the mask canvas** (origW × origH, the
/// same canvas `MaskRenderer.drawTile` renders the fill into): x, y ∈ [0,1]
/// with the origin at the canvas's top-left.  They therefore carry no pixel
/// scale of their own — the view maps them through exactly the transform
/// CoreAnimation applies to the fill image, so outline and fill cannot drift.
struct MaskOutline {
    let polygons: [[CGPoint]]
    /// Tier selector (§3.4): primary = the most recent tap, full-strength
    /// outline; secondary = "this was selected earlier", reduced outline.
    let isPrimary: Bool
}

/// The whole overlay's outlines plus the canvas they are normalised against.
/// Published as one value so the view never sees polygons and canvas size
/// from two different frames.
struct MaskOutlineSet {
    /// Mask canvas size in pixels (origW × origH) — the space the fill image
    /// is rendered in and the space `outlines` are normalised against.
    let canvasSize: CGSize
    let outlines: [MaskOutline]

    var isEmpty: Bool { outlines.isEmpty }
}

// MARK: - Style

/// Every presentation constant for the tap-mask overlay, in one place
/// (architect_output §3.3.3 / §3.4, frozen in §9.5).
enum MaskOutlineStyle {

    /// ⚠️ **C-6 — THE single switch.**  Set to `false` to reproduce the Day 4
    /// "no outline" visual condition exactly, so Day 6/7 manual scoring can be
    /// compared against the 44/50 = 88 % baseline with only one variable moved.
    /// Nothing else needs touching: `CameraManager` skips contour tracing
    /// entirely (zero cost) and `PreviewView` keeps its shape layers empty.
    static let isEnabled = true

    /// Outer ring — near-black, drawn first and widest so it survives on light
    /// backgrounds.
    static let outerColor = UIColor(red: 0, green: 0, blue: 0, alpha: 1)
    /// Inner line — near-white, drawn on top and narrower so it survives on
    /// dark backgrounds.
    static let innerColor = UIColor(red: 1, green: 1, blue: 1, alpha: 1)

    /// Line widths in **pt** (C-5).  Primary at full strength, secondary
    /// stepped down — §3.4 lists the outline tier as the *strongest* and only
    /// background-independent primary/secondary discriminator.
    static let primaryOuterWidth: CGFloat = 2.0
    static let primaryInnerWidth: CGFloat = 1.5
    static let secondaryOuterWidth: CGFloat = 1.5
    static let secondaryInnerWidth: CGFloat = 1.0

    static let primaryOuterAlpha: CGFloat = 0.85
    static let primaryInnerAlpha: CGFloat = 0.95
    static let secondaryOuterAlpha: CGFloat = 0.70
    static let secondaryInnerAlpha: CGFloat = 0.70
}
