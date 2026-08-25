"""
Phase 3 Day 6 — build canonical (1920x1080) test frames for the token-0 study.

Why this file exists
====================
The Session-A evidence we have for the three failure objects is *screen
recordings* (`shared/S2-1.MP4` / `S3-1.MP4` / `S5-1.MP4`).  A screen recording
is NOT the frame the encoder saw:

  * the recording is the 828x1792 device screen, stored with a +/-90 deg
    display-matrix (ffmpeg decodes it as 1792x828 landscape),
  * `AVCaptureVideoPreviewLayer.videoGravity = .resizeAspectFill`
    (CameraPreview.swift:21) crops the canonical frame to the screen aspect,
  * the app's own chrome (title pill, Settings bar, #N badge) is burnt in.

Canonical space is the capture buffer in display orientation.  `sessionPreset
= .high` (CameraManager.swift:1636) on an iPhone 11 rear camera is 1920x1080,
and the recordings are landscape, so canonical = 1920x1080.

Geometry reconstruction (exact, no free parameters)
---------------------------------------------------
    aspectFill scale s = max(1792/1920, 828/1080) = 1792/1920 = 0.93333
    displayed buffer   = 1792 x 1008   -> cropped to 828 rows
    hidden rows        = (1008-828)/2 / s = 96.4 canonical rows top AND bottom

    => resize 1792x828 -> 1920x887, then pad 96/97 rows top/bottom.

That reproduces the letterbox geometry the device actually used:
    scale = 1024/1920, nw=1024, nh=576, pad_y=224
    content box in the 256x256 low-res grid = 256 x 144 = 36864 px
which is the `contentPx` all device-side area logs are normalised against
(cap60 = 22118, cap85 = 31334).

Fabricated content (declare, do not hide)
-----------------------------------------
  F1  the 96+97 hidden rows are filled by EDGE REPLICATION of the first/last
      visible row.  17.9% of canonical rows are therefore synthetic.  They sit
      at the very top / bottom of frame, far from every tap point.
      (Mirror-padding was tried first and rejected: it manufactures plausible
      *objects* — a second keyboard, a second table edge — that SAM can and
      does latch onto.  A vertical smear manufactures texture, not objects.)
  F2  the three UI rectangles are overpainted with content copied, unflipped,
      from an explicitly clean band of the same frame.  Leaving them in would
      be worse: a full-width dark translucent bar is a strong, wholly
      artificial segment sitting right next to the S5 tap target.

Both are applied identically to every mask token, so the *relative* token
comparison is unaffected; only absolute area fractions carry the caveat.

RUN
===
/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/.venv_export/bin/python \
    shared/prep_token0_frames.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SHARED = ROOT / "shared"

DISP_W, DISP_H = 1792, 828          # decoded screen recording, landscape
CANON_W, CANON_H = 1920, 1080       # capture buffer, display orientation

# UI rectangles in DISPLAY coords (x0, y0, x1, y1, src_y0).  The "Settings" bar
# was located from the row-mean profile of three unrelated frames: it darkens
# rows 124..227 in all of them.  `src_y0` names a band that is scene content in
# every frame, so no rect is ever repaired from another rect.
UI_RECTS = [
    (1450, 618, 1668, 752, 470),   # "#N" tap-counter badge  <- clean band above
    (118,  112, 1680, 238, 240),   # full-width "Settings" bar <- band below
    (118,   26,  706, 112, 300),   # title pill + camera button <- clean band
]


def overpaint(arr: np.ndarray, rect, blur_px: int = 24) -> np.ndarray:
    """Repair a UI rect with a heavily blurred copy of a clean band.

    The blur is deliberate: an unblurred copy manufactures a *duplicate object*
    (a second keyboard, a second bookmark) that SAM will happily segment.  A
    blurred copy keeps the local colour statistics but has no object identity,
    so it behaves like defocused background.
    """
    from PIL import ImageFilter
    x0, y0, x1, y1, sy = rect
    h = y1 - y0
    band = Image.fromarray(arr[sy:sy + h, x0:x1])
    band = band.filter(ImageFilter.GaussianBlur(blur_px))
    arr[y0:y1, x0:x1] = np.asarray(band)
    return arr


def display_to_canonical(img: Image.Image) -> Image.Image:
    a = np.asarray(img.convert("RGB")).copy()
    for r in UI_RECTS:
        a = overpaint(a, r)
    vis = Image.fromarray(a).resize((CANON_W, 887), Image.BICUBIC)
    v = np.asarray(vis)
    top_n = (CANON_H - 887) // 2               # 96
    bot_n = CANON_H - 887 - top_n              # 97
    top = np.repeat(v[:1], top_n, axis=0)      # edge replication, see F1
    bot = np.repeat(v[-1:], bot_n, axis=0)
    return Image.fromarray(np.concatenate([top, v, bot], axis=0))


def disp_pt_to_canonical(x: float, y: float):
    return x * (CANON_W / DISP_W), y * (CANON_W / DISP_W) + (CANON_H - 887) / 2.0


# ---------------------------------------------------------------------------
# Case book.  Tap coordinates are given in DISPLAY pixels (1792x828) because
# that is the space in which the frames were visually inspected.
# ---------------------------------------------------------------------------
CASES = [
    dict(name="S3_marble", video="S3-1.MP4", t=33,
         note="silver closed laptop on white marble table — Session A failure "
              "(tap selected the whole table)",
         taps=[
             ("laptop_lid", 1250, 528, "FAIL-CASE target"),
             ("white_box", 1150, 442, "white paper box on marble (control)"),
             ("mug", 1472, 528, "white mug on marble (control)"),
             ("chair_back", 600, 330, "office chair, high contrast (control)"),
             ("marble_bare", 900, 640, "bare marble surface (degenerate prompt)"),
         ]),
    dict(name="S5_foam", video="S5-1.MP4", t=1,
         note="white USB charger on white foam pad — Session A worst item (2/10)",
         taps=[
             ("charger", 985, 245, "FAIL-CASE target"),
             ("bookmark", 560, 530, "white bookmark on foam (mid failure)"),
             ("ketchup", 1105, 555, "foil packet on foam (mid failure)"),
             ("grey_cable", 1150, 380, "grey cable coil, mild contrast"),
             ("foam_bare", 830, 700, "bare foam pad (degenerate prompt)"),
         ]),
    dict(name="S2_stand", video="S2-1.MP4", t=44,
         note="silver monitor stand on beige desk + white wall — Session A failure",
         taps=[
             ("stand_column", 512, 350, "FAIL-CASE target (column)"),
             ("stand_base", 520, 420, "FAIL-CASE target (base)"),
             ("mouse", 730, 610, "black mouse on beige, high contrast (control)"),
             ("tumbler", 1270, 590, "black tumbler (control)"),
             ("gum_jar", 775, 420, "Mentos jar, high contrast (control)"),
         ]),
    dict(name="S1_clean", video="S1-1.MP4", t=2,
         note="S1 scene — Session A scored 10/10 here; used to test whether "
              "token 0 REGRESSES the cases that already work",
         taps=[
             ("ipad", 470, 520, "yellow iPad (control, was correct)"),
             ("keyboard", 800, 300, "black keyboard (control, was correct)"),
             ("mousepad", 930, 620, "mouse pad (control, was correct)"),
             ("canon_bag", 1430, 470, "Canon bag (control, was correct)"),
             ("tissue_box", 230, 400, "tissue box (control, was correct)"),
         ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "canonical").mkdir(parents=True, exist_ok=True)

    book = []
    for c in CASES:
        raw = out / f"raw_{c['name']}.png"
        if not raw.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", str(c["t"]),
                 "-i", str(SHARED / c["video"]), "-frames:v", "1",
                 "-update", "1", str(raw)], check=True)
        img = Image.open(raw)
        assert img.size == (DISP_W, DISP_H), f"{raw} is {img.size}"
        canon = display_to_canonical(img)
        cp = out / "canonical" / f"{c['name']}.png"
        canon.save(cp)

        taps = []
        for label, dx, dy, desc in c["taps"]:
            cx, cy = disp_pt_to_canonical(dx, dy)
            taps.append(dict(label=label, x=round(cx, 1), y=round(cy, 1),
                             desc=desc, disp=[dx, dy]))
        book.append(dict(name=c["name"], image=str(cp), note=c["note"], taps=taps))
        print(f"{c['name']}: {cp}  {canon.size}  {len(taps)} taps")

    (out / "cases.json").write_text(json.dumps(book, ensure_ascii=False, indent=2))
    print(f"\nwrote {out/'cases.json'}")


if __name__ == "__main__":
    main()
