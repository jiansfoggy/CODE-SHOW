"""
Phase 3 Day 6 — token-0 study, part 2: replay MaskRenderer's ACTUAL pick rule.

`eval_token0.py` measures the raw logit fields.  This script re-implements the
device-side selection so the comparison is apples-to-apples with the
`[TAP#N] candidates:` device logs.  Every constant below is transcribed from
`JudgeE2/Segmentation/MaskRenderer.swift` (read-only; nothing here modifies it):

  keepComponentContaining  :798-843   4-connected BFS from the tap, seed search
                                      radius 2, area/bbox over the FULL 256x256
                                      grid (letterbox padding included — this is
                                      why device logs can print 65536)
  stabilityScore           :369-381   count(logit > +1) / count(logit > -1),
                                      also over the full grid
  minComponentPx           = 30
  minComponentSidePx       = 3
  minComponentFill         = 0.05
  cap60 / cap85            :545-547   contentPx * 60% / 85%, contentPx = 36864
  pick                     :591-600   smallest surviving component with
                                      count <= cap60; else max-iou among
                                      count <= cap85 ("degraded")

Four candidate sets are replayed on the same logits:

  CUR   tokens 1,2,3          — what ships today (decoder_multi)
  T0    token 0 only          — Day 3 (decoder single)
  A4    tokens 0,1,2,3        — "export 4 candidates"
  FB    tokens 1,2,3, and token 0 only if 1..3 all fail the filters/caps
                              — "token 0 as fallback"

RUN
===
/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/.venv_export/bin/python \
    shared/eval_token0_pick.py --cases <cases.json> --out <dir> [--prenormalise]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_token0 import (  # noqa: E402  (reuses the already-reviewed harness)
    Decoder4, ENCODER_PKG, CHECKPOINT, LOWRES, letterbox, make_prompt,
    sam_point, to_encoder_input, upsample_to_image, overlay,
)
from mobile_sam import sam_model_registry  # noqa: E402

MIN_COMPONENT_PX = 30
MIN_COMPONENT_SIDE_PX = 3
MIN_COMPONENT_FILL = 0.05
STABILITY_DELTA = 1.0
CONTENT_PX = 36864          # 256 x 144, iPhone 11 .high preset, landscape
CAP60 = CONTENT_PX * 60 // 100
CAP85 = CONTENT_PX * 85 // 100


def keep_component_containing(binm: np.ndarray, tx: int, ty: int):
    """Port of MaskRenderer.keepComponentContaining (4-connected BFS)."""
    h, w = binm.shape
    tx = max(0, min(w - 1, tx))
    ty = max(0, min(h - 1, ty))
    seed = None
    for r in range(3):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = tx + dx, ty + dy
                if 0 <= x < w and 0 <= y < h and binm[y, x]:
                    seed = (y, x)
                    break
            if seed:
                break
        if seed:
            break
    if seed is None:
        return dict(count=0, boxW=0, boxH=0, fill=0.0), np.zeros_like(binm)

    out = np.zeros_like(binm)
    q = deque([seed])
    out[seed] = True
    cnt = 1
    y0 = y1 = seed[0]
    x0 = x1 = seed[1]
    while q:
        y, x = q.popleft()
        x0, x1 = min(x0, x), max(x1, x)
        y0, y1 = min(y0, y), max(y1, y)
        for ny, nx in ((y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x)):
            if 0 <= nx < w and 0 <= ny < h and binm[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                cnt += 1
                q.append((ny, nx))
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    return dict(count=int(cnt), boxW=int(bw), boxH=int(bh),
                fill=round(cnt / float(bw * bh), 4)), out


def stability(logit: np.ndarray) -> float:
    lo = int((logit > -STABILITY_DELTA).sum())
    hi = int((logit > STABILITY_DELTA).sum())
    return round(hi / lo, 4) if lo else 0.0


def build_candidates(masks: np.ndarray, ious: np.ndarray, tx: int, ty: int, idxs):
    """Returns (kept, rejected) exactly as MaskRenderer:557-580 would."""
    kept, rejected = [], []
    for c in idxs:
        binm = masks[c] > 0.0
        comp, sel = keep_component_containing(binm, tx, ty)
        stab = stability(masks[c])
        rec = dict(tok=c, iou=round(float(ious[c]), 4), stab=stab, **comp)
        if comp["count"] == 0:
            rejected.append({**rec, "why": "no-component-at-tap"}); continue
        if comp["count"] < MIN_COMPONENT_PX:
            rejected.append({**rec, "why": "tiny"}); continue
        if min(comp["boxW"], comp["boxH"]) < MIN_COMPONENT_SIDE_PX:
            rejected.append({**rec, "why": "line"}); continue
        if comp["fill"] < MIN_COMPONENT_FILL:
            rejected.append({**rec, "why": "sparse"}); continue
        rec["_sel"] = sel
        kept.append(rec)
    return kept, rejected


def pick(kept):
    """Port of MaskRenderer:591-600."""
    under = sorted([k for k in kept if k["count"] <= CAP60], key=lambda k: k["count"])
    if under:
        return under[0], False
    fb = [k for k in kept if k["count"] <= CAP85]
    if fb:
        return max(fb, key=lambda k: k["iou"]), True
    return None, False


def strategies(masks, ious, tx, ty):
    out = {}
    k_cur, r_cur = build_candidates(masks, ious, tx, ty, [1, 2, 3])
    k_t0, _ = build_candidates(masks, ious, tx, ty, [0])
    k_a4, _ = build_candidates(masks, ious, tx, ty, [0, 1, 2, 3])

    s_cur, d_cur = pick(k_cur)
    out["CUR"] = (s_cur, d_cur, k_cur, r_cur)
    s_t0, d_t0 = pick(k_t0)
    out["T0"] = (s_t0, d_t0, k_t0, None)
    s_a4, d_a4 = pick(k_a4)
    out["A4"] = (s_a4, d_a4, k_a4, None)
    # FB: token 0 only when the 1..3 path produced nothing at all
    out["FB"] = out["CUR"] if s_cur is not None else out["T0"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prenormalise", action="store_true",
                    help="reproduce SAMEncoder.swift, which normalises on the CPU "
                         "and then feeds a graph that normalises again")
    ap.add_argument("--save-overlays", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import coremltools as ct
    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT)).eval()
    dec4 = Decoder4(sam).eval()
    enc = ct.models.MLModel(str(ENCODER_PKG), compute_units=ct.ComputeUnit.CPU_AND_GPU)

    rows = []
    for case in json.loads(Path(args.cases).read_text()):
        img = Image.open(case["image"]).convert("RGB")
        W, H = img.size
        canvas, scale, pad_x, pad_y, nw, nh = letterbox(img)
        emb = enc.predict({"image": to_encoder_input(canvas, args.prenormalise)}
                          )["image_embeddings"].astype(np.float32)

        for tap in case["taps"]:
            sx, sy = sam_point(tap["x"], tap["y"], scale, pad_x, pad_y)
            coords, labels, mask_in, has_mask = make_prompt(sx, sy)
            m, p = dec4(torch.from_numpy(emb), torch.from_numpy(coords),
                        torch.from_numpy(labels), torch.from_numpy(mask_in),
                        torch.from_numpy(has_mask))
            masks, ious = m[0].numpy(), p[0].numpy()
            tx, ty = int(round(sx / 4.0)), int(round(sy / 4.0))

            st = strategies(masks, ious, tx, ty)
            rec = dict(case=case["name"], tap=tap["label"], desc=tap.get("desc", ""))
            for name, (sel, degraded, kept, _rej) in st.items():
                rec[name] = None if sel is None else dict(
                    tok=sel["tok"], area=sel["count"], frac=round(sel["count"] / CONTENT_PX, 4),
                    bbox=[sel["boxW"], sel["boxH"]], fill=sel["fill"],
                    iou=sel["iou"], stab=sel["stab"], degraded=degraded)
            rec["all"] = [{k: v for k, v in c.items() if k != "_sel"}
                          for c in st["A4"][2]]
            rows.append(rec)
            print(json.dumps(rec, ensure_ascii=False))

            if args.save_overlays:
                for name in ("CUR", "T0"):
                    sel = st[name][0]
                    if sel is None:
                        continue
                    lo = np.where(st[name][2] and sel["_sel"], 1.0, -1.0)
                    full = upsample_to_image(lo.astype(np.float32), scale, pad_x, pad_y, W, H)
                    col = (0, 217, 255) if name == "CUR" else (255, 120, 0)
                    overlay(img, full, col, (tap["x"], tap["y"])).save(
                        out_dir / f"{case['name']}_{tap['label']}_{name}_tok{sel['tok']}.jpg",
                        quality=82)

    (out_dir / "pick.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_dir/'pick.json'} ({len(rows)} taps)")


if __name__ == "__main__":
    main()
