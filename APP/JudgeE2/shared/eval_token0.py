"""
Phase 3 Day 6 — Offline evaluation: should SAM mask-token 0 be a candidate?

Context
=======
`shared/export_decoder_multimask.py:170-172` returns `masks[:, 1:4]` — it drops
token 0.  SAM's mask head emits 4 tokens:

    token 0  — the "single mask" head, trained under multimask_output=False,
               i.e. "give one unambiguous answer for this prompt"
    token 1..3 — the ambiguity candidates (sub-part / part / whole)

Phase 2 / Day 3 shipped `MobileSAM_PromptMaskDecoder.mlpackage`, exported from
`shared_v1/export_mobilesam_split_to_coreml.py`, whose `_select_mask` uses
`best_idx = zeros` over the *raw* `predict_masks` output ⇒ **token 0**.

This script answers, fully offline:
  Q1  where does token 0 sit in the granularity ladder vs tokens 1/2/3?
  Q2  on the Session-A failure cases, is token 0 closer to user intent?
  Q3  what would a 4-candidate export cost?

It also cross-checks the two deployed CoreML decoders against the torch
reference so the token identity claim is measured, not assumed.

RUN
===
/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/.venv_export/bin/python \
    shared/eval_token0.py --cases shared/token0_cases.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MOBILE_SAM_REPO = ROOT / "models" / "MobileSAM"
CHECKPOINT = ROOT / "models" / "mobile_sam.pt"
MODELS_DIR = ROOT / "JudgeE2" / "Segmentation" / "Models"
ENCODER_PKG = MODELS_DIR / "MobileSAM_ImageEncoder_fp16_milfix.mlpackage"
DEC_SINGLE_PKG = MODELS_DIR / "MobileSAM_PromptMaskDecoder.mlpackage"
DEC_MULTI_PKG = MODELS_DIR / "MobileSAM_PromptMaskDecoder_multi.mlpackage"

sys.path.insert(0, str(MOBILE_SAM_REPO))

from mobile_sam.modeling.common import LayerNorm2d  # noqa: E402


def _layernorm2d_forward_milfix(self, x):
    x = x.permute(0, 2, 3, 1)
    x = F.layer_norm(x, [x.shape[-1]], self.weight, self.bias, self.eps)
    return x.permute(0, 3, 1, 2)


LayerNorm2d.forward = _layernorm2d_forward_milfix

from mobile_sam import sam_model_registry  # noqa: E402

INPUT_SIZE = 1024
LOWRES = 256
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ---------------------------------------------------------------------------
# Decoder that keeps ALL FOUR mask tokens
# ---------------------------------------------------------------------------
class Decoder4(nn.Module):
    """Identical math to export_decoder_multimask.MultimaskDecoder, but returns
    masks[:, 0:4] / scores[:, 0:4] so token 0 can be inspected."""

    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.img_size = sam.image_encoder.img_size
        self.register_buffer("dense_pe", sam.prompt_encoder.get_dense_pe())

    def _embed_points(self, point_coords, point_labels):
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        pe = self.sam.prompt_encoder.pe_layer._pe_encoding(point_coords)
        lbl = point_labels.unsqueeze(-1).expand_as(pe)
        pe = pe * (lbl != -1)
        pe = pe + self.sam.prompt_encoder.not_a_point_embed.weight * (lbl == -1)
        for i in range(self.sam.prompt_encoder.num_point_embeddings):
            pe = pe + self.sam.prompt_encoder.point_embeddings[i].weight * (lbl == i)
        return pe

    def _embed_masks(self, input_mask, has_mask_input):
        emb = has_mask_input * self.sam.prompt_encoder.mask_downscaling(input_mask)
        emb = emb + (1 - has_mask_input) * self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return emb

    @torch.no_grad()
    def forward(self, image_embeddings, point_coords, point_labels, mask_input, has_mask_input):
        sparse = self._embed_points(point_coords, point_labels)
        dense = self._embed_masks(mask_input, has_mask_input)
        md = self.sam.mask_decoder
        output_tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], dim=0).unsqueeze(0)
        tokens = torch.cat((output_tokens, sparse), dim=1)
        src = image_embeddings + dense
        hs, src2 = md.transformer(src, self.dense_pe, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : 1 + md.num_mask_tokens, :]
        src2 = src2.transpose(1, 2).view(1, 256, 64, 64)
        upscaled = md.output_upscaling(src2)
        hyper_in = torch.stack(
            [md.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(md.num_mask_tokens)],
            dim=1,
        )
        masks = (hyper_in @ upscaled.view(1, 32, LOWRES * LOWRES)).view(1, -1, LOWRES, LOWRES)
        iou_pred = md.iou_prediction_head(iou_token_out)
        return masks, iou_pred          # [1,4,256,256], [1,4]


# ---------------------------------------------------------------------------
# Geometry — mirrors SAMEncoder.preprocess + PointPromptBuilder (centered pad)
# ---------------------------------------------------------------------------
def letterbox(img: Image.Image, size: int = INPUT_SIZE):
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, nw, nh


def to_encoder_input(canvas: Image.Image, prenormalise: bool) -> np.ndarray:
    """[1,3,1024,1024] float32, RGB.

    prenormalise=False → raw 0..255 (what the *export wrapper* expects; it
                         applies (x-mean)/std inside the graph).
    prenormalise=True  → (x-mean)/std applied here as well, reproducing what
                         SAMEncoder.swift actually feeds the model.
    """
    arr = np.asarray(canvas, dtype=np.float32)              # HWC RGB 0..255
    if prenormalise:
        arr = (arr - PIXEL_MEAN) / PIXEL_STD
    return np.transpose(arr, (2, 0, 1))[None].astype(np.float32)


def sam_point(px, py, scale, pad_x, pad_y):
    return float(px * scale + pad_x), float(py * scale + pad_y)


def make_prompt(sx, sy):
    coords = np.array([[[sx, sy], [0.0, 0.0]]], dtype=np.float32)
    labels = np.array([[1.0, -1.0]], dtype=np.float32)
    mask_in = np.zeros((1, 1, LOWRES, LOWRES), dtype=np.float32)
    has_mask = np.zeros((1,), dtype=np.float32)
    return coords, labels, mask_in, has_mask


# ---------------------------------------------------------------------------
# Metrics (mirror MaskRenderer semantics where it matters)
# ---------------------------------------------------------------------------
def content_box(scale, pad_x, pad_y, nw, nh):
    """Content rect in the 256x256 low-res grid (the letterbox interior)."""
    f = LOWRES / INPUT_SIZE
    x0 = int(round(pad_x * f))
    y0 = int(round(pad_y * f))
    x1 = int(round((pad_x + nw) * f))
    y1 = int(round((pad_y + nh) * f))
    return x0, y0, x1, y1


def mask_stats(logit: np.ndarray, box, stability_delta: float = 1.0):
    x0, y0, x1, y1 = box
    sub = logit[y0:y1, x0:x1]
    binm = sub > 0.0
    area = int(binm.sum())
    content_px = sub.size
    hi = int((sub > stability_delta).sum())
    lo = int((sub > -stability_delta).sum())
    stability = (hi / lo) if lo > 0 else 0.0
    if area > 0:
        ys, xs = np.nonzero(binm)
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        fill = area / float(bw * bh)
    else:
        bw = bh = 0
        fill = 0.0
    return dict(area=area, content_px=content_px, frac=area / content_px,
                stability=round(stability, 4), fill=round(fill, 4),
                bbox_w=int(bw), bbox_h=int(bh))


def iou_of(a: np.ndarray, b: np.ndarray, box):
    x0, y0, x1, y1 = box
    A = a[y0:y1, x0:x1] > 0
    B = b[y0:y1, x0:x1] > 0
    u = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / u) if u else 0.0


def contains_frac(inner: np.ndarray, outer: np.ndarray, box):
    """fraction of `inner` that lies inside `outer` (nesting test)."""
    x0, y0, x1, y1 = box
    A = inner[y0:y1, x0:x1] > 0
    B = outer[y0:y1, x0:x1] > 0
    return float(np.logical_and(A, B).sum() / A.sum()) if A.sum() else 0.0


# ---------------------------------------------------------------------------
def upsample_to_image(logit256, scale, pad_x, pad_y, w, h):
    t = torch.from_numpy(logit256)[None, None]
    full = F.interpolate(t, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()
    crop = full[pad_y:pad_y + int(round(h * scale)), pad_x:pad_x + int(round(w * scale))]
    t2 = torch.from_numpy(np.ascontiguousarray(crop))[None, None]
    return F.interpolate(t2, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()


COLORS = [(255, 64, 64), (0, 217, 255), (64, 255, 96), (255, 200, 0)]


def overlay(img: Image.Image, logit_full: np.ndarray, color, tap, alpha=0.55):
    base = np.asarray(img, dtype=np.float32).copy()
    m = logit_full > 0
    for c in range(3):
        base[..., c][m] = base[..., c][m] * (1 - alpha) + color[c] * alpha
    out = Image.fromarray(base.astype(np.uint8))
    from PIL import ImageDraw
    d = ImageDraw.Draw(out)
    r = max(6, img.size[0] // 120)
    d.ellipse([tap[0] - r, tap[1] - r, tap[0] + r, tap[1] + r], outline=(255, 255, 255), width=4)
    d.ellipse([tap[0] - r // 2, tap[1] - r // 2, tap[0] + r // 2, tap[1] + r // 2], fill=(255, 0, 0))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prenormalise", action="store_true",
                    help="reproduce SAMEncoder.swift (double normalisation)")
    ap.add_argument("--coreml-crosscheck", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import coremltools as ct

    print("loading torch MobileSAM …")
    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT)).eval()
    dec4 = Decoder4(sam).eval()

    print(f"loading CoreML encoder {ENCODER_PKG.name} …")
    enc = ct.models.MLModel(str(ENCODER_PKG), compute_units=ct.ComputeUnit.CPU_AND_GPU)

    dec_single = dec_multi = None
    if args.coreml_crosscheck:
        dec_single = ct.models.MLModel(str(DEC_SINGLE_PKG), compute_units=ct.ComputeUnit.CPU_ONLY)
        dec_multi = ct.models.MLModel(str(DEC_MULTI_PKG), compute_units=ct.ComputeUnit.CPU_ONLY)

    cases = json.loads(Path(args.cases).read_text())
    rows = []

    for case in cases:
        img_path = Path(case["image"])
        if not img_path.is_absolute():
            img_path = ROOT / img_path
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        canvas, scale, pad_x, pad_y, nw, nh = letterbox(img)
        box = content_box(scale, pad_x, pad_y, nw, nh)

        x = to_encoder_input(canvas, prenormalise=args.prenormalise)
        emb = enc.predict({"image": x})["image_embeddings"].astype(np.float32)

        for tap in case["taps"]:
            px, py = tap["x"], tap["y"]
            sx, sy = sam_point(px, py, scale, pad_x, pad_y)
            coords, labels, mask_in, has_mask = make_prompt(sx, sy)

            masks_t, iou_t = dec4(
                torch.from_numpy(emb), torch.from_numpy(coords),
                torch.from_numpy(labels), torch.from_numpy(mask_in),
                torch.from_numpy(has_mask),
            )
            masks = masks_t[0].numpy()      # [4,256,256]
            ious = iou_t[0].numpy()         # [4]

            rec = dict(case=case["name"], tap=tap["label"],
                       image=str(img_path), point=[px, py], size=[W, H])
            for k in range(4):
                st = mask_stats(masks[k], box)
                st["iou_pred"] = round(float(ious[k]), 4)
                rec[f"tok{k}"] = st
            # nesting / overlap structure
            rec["iou_t0_t1"] = round(iou_of(masks[0], masks[1], box), 4)
            rec["iou_t0_t2"] = round(iou_of(masks[0], masks[2], box), 4)
            rec["iou_t0_t3"] = round(iou_of(masks[0], masks[3], box), 4)
            rec["t0_in_t3"] = round(contains_frac(masks[0], masks[3], box), 4)
            rec["t1_in_t0"] = round(contains_frac(masks[1], masks[0], box), 4)

            if args.coreml_crosscheck:
                feed = {"image_embeddings": emb, "point_coords": coords,
                        "point_labels": labels, "mask_input": mask_in,
                        "has_mask_input": has_mask}
                s = dec_single.predict(feed)
                m = dec_multi.predict(feed)
                sm = s["low_res_masks"][0, 0]
                mm = m["low_res_masks"][0]
                rec["cm_single_vs_tok"] = [round(iou_of(sm, masks[k], box), 4) for k in range(4)]
                rec["cm_single_iou_pred"] = round(float(s["iou_predictions"].reshape(-1)[0]), 4)
                rec["cm_multi_vs_tok123"] = [round(iou_of(mm[j], masks[j + 1], box), 4) for j in range(3)]
                rec["cm_multi_iou_pred"] = [round(float(v), 4) for v in m["iou_predictions"].reshape(-1)]

            rows.append(rec)
            print(json.dumps(rec, ensure_ascii=False))

            if not args.no_images:
                tag = f"{case['name']}_{tap['label']}"
                for k in range(4):
                    full = upsample_to_image(masks[k], scale, pad_x, pad_y, W, H)
                    ov = overlay(img, full, COLORS[k], (px, py))
                    ov.save(out_dir / f"{tag}_tok{k}.jpg", quality=82)

    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_dir/'results.json'}  ({len(rows)} taps)")


if __name__ == "__main__":
    main()
