#!/usr/bin/env python3
"""
JudgeE2 app icon generator — a red "J" drawn as a map pin, standing in a
location ring, on white.

Renders the three 1024x1024 AppIcon variants (any / dark / tinted) that
Assets.xcassets/AppIcon.appiconset declares in its Contents.json.

THE CONSTRUCTION.  The letter and the landmark are the same object, traced as
ONE closed contour — nothing is layered on top of anything:

  * the head is a map-pin teardrop: a circle closed off by the two tangents
    drawn from a virtual tip below it;
  * those tangents are cut where they have narrowed to exactly the stem's
    width, so the teardrop flows into the stem with no shoulder and no seam;
  * the stem descends and becomes the J's hook, whose inner and outer edges
    are the two offsets of a semicircular centreline;
  * the head carries a punched-through counter (the eyelet), which is a real
    hole in the path, not a disc painted over it.

Two things this geometry is easy to get wrong, both found by rendering:
  1. The hook's terminal cap must bulge along the direction of travel. The
     centreline at the hook's end is moving upward, so the cap sweeps 0 -> -pi.
     Sweeping 0 -> +pi puts it below the inner edge and cusps against it — the
     silhouette then reads as a claw, not a J.
  2. The framing is fitted from the artwork's measured bounding box, not from
     the construction arithmetic. iOS masks app icons to a squircle, so
     anything computed to sit near a border gets clipped.

Appearance variants:
  any / dark  — red on white, as specified. Both are white: this is a
                white-ground mark, and inverting it for dark mode would make it
                a different mark.
  tinted      — iOS uses this one as a *luminance mask* over a dark ground and
                applies the user's tint itself, so it is rendered light-on-dark.
                Feeding it the white-ground artwork would come back inverted.

This script only writes PNGs into the asset catalog; it touches no app code.
"""

import math
import sys
from PIL import Image, ImageDraw

S = 1024          # final edge length
SS = 4            # supersampling factor
N = S * SS

# Signal red. White against it clears 4.67:1, so the mark holds up on a light
# home screen and the eyelet stays legible.
RED = (227, 30, 46)
WHITE = (255, 255, 255)
# Tinted variant: iOS maps luminance, so this is "how bright", not "what colour".
TINT_FG = (232, 232, 232)
TINT_BG = (0, 0, 0)

# Geometry, in units of a 1000-wide design square.
HEAD_R      = 140.0     # teardrop head radius
HEAD_CY     = 265.0     # head centre
DROP        = 3.00      # virtual tip distance below the head centre, x HEAD_R
STEM_W      = 0.58      # stem half-width, x HEAD_R
STEM_BOT    = 675.0     # where the stem becomes the hook
HOOK_R      = 165.0     # hook centreline radius
COUNTER     = 0.40      # eyelet radius, x HEAD_R
CX          = 545.0
RING_R      = 215.0     # inner location-ring semi-major axis
RING_SQUASH = 0.33      # ring perspective: semi-minor / semi-major
RING_LW     = 26.0      # inner ring stroke, in design units (NOT pixels — a
RING_LW_OUT = 20.0      # pixel width would thin out as SS or S changed)
FILL_RATIO  = 0.76      # fraction of the canvas the artwork fills


def arc(cx, cy, r, a0, a1, n):
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
             cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def j_outline(cx, hy, rh, sw, R, stem_bot, drop):
    """The whole letter as one closed contour, traced clockwise from the head's
    right-hand tangent contact."""
    d = drop * rh
    beta = math.acos(rh / d)                      # contact angle from straight down
    a0, a1 = math.pi / 2 - beta, math.pi / 2 + beta
    pr = (cx + rh * math.cos(a0), hy + rh * math.sin(a0))
    off = abs(pr[0] - cx)
    if off <= sw:
        raise ValueError("head too narrow for this stem width")
    # Walk the tangent until its half-width equals the stem's, and start the
    # stem there — that is what makes the join invisible.
    t = 1.0 - sw / off
    y_stem = pr[1] + t * ((hy + d) - pr[1])
    hcx = cx - R

    p  = arc(cx, hy, rh, a0, a1 - 2 * math.pi, 260)   # over the top of the head
    p += [(cx - sw, y_stem), (cx - sw, stem_bot)]     # left tangent + stem edge
    p += arc(hcx, stem_bot, R - sw, 0, math.pi, 140)  # hook, inner edge
    p += arc(hcx - R, stem_bot, sw, 0, -math.pi, 48)  # terminal cap — see note 1
    p += arc(hcx, stem_bot, R + sw, math.pi, 0, 140)  # hook, outer edge
    p += [(cx + sw, stem_bot), (cx + sw, y_stem)]     # stem edge + right tangent
    return p


def render(mode="light"):
    fg = TINT_FG if mode == "tinted" else RED
    bg = TINT_BG if mode == "tinted" else WHITE

    u = N / 1000.0
    rh, sw, R = HEAD_R * u, HEAD_R * STEM_W * u, HOOK_R * u
    cx, hy, stem_bot = CX * u, HEAD_CY * u, STEM_BOT * u

    art = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(art, "RGBA")
    d.polygon(j_outline(cx, hy, rh, sw, R, stem_bot, DROP), fill=fg + (255,))

    # Location ring, centred so the J's foot sits *inside* it — the letter
    # stands in the ring rather than in front of it, which is what makes the
    # ring read as ground and not as a halo.
    foot = stem_bot + R + sw
    ox = ((cx - 2 * R - sw) + (cx + rh)) / 2          # optical centre of the letter
    rx = RING_R * u
    ry = foot + rx * RING_SQUASH * 0.42
    for r, alpha, w in ((rx, 255, RING_LW), (rx * 1.46, 125, RING_LW_OUT)):
        d.ellipse([ox - r, ry - r * RING_SQUASH, ox + r, ry + r * RING_SQUASH],
                  outline=fg + (alpha,), width=max(1, int(w * u)))

    # The eyelet, punched clean through to transparency so it picks up the
    # ground rather than a pasted-on disc.
    rr = rh * COUNTER
    d.ellipse([cx - rr, hy - rr, cx + rr, hy + rr], fill=(0, 0, 0, 0))

    # Fit by measured extents — see note 2.
    art = art.crop(art.getbbox())
    scale = int(N * FILL_RATIO) / max(art.width, art.height)
    art = art.resize((max(1, int(art.width * scale)), max(1, int(art.height * scale))),
                     Image.LANCZOS)
    img = Image.new("RGBA", (N, N), bg + (255,))
    img.alpha_composite(art, ((N - art.width) // 2, (N - art.height) // 2))
    return img.convert("RGB").resize((S, S), Image.LANCZOS)


if __name__ == "__main__":
    out = sys.argv[1]
    render("light").save(f"{out}/AppIcon-1024.png")
    render("light").save(f"{out}/AppIcon-Dark-1024.png")
    render("tinted").save(f"{out}/AppIcon-Tinted-1024.png")
    print("wrote 3 icons to", out)
