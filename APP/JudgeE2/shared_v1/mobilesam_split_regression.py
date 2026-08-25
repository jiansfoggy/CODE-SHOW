"""MobileSAM split CoreML regression (Python).

Goal: give Builder/Debugger a minimal numeric check to confirm Swift/CoreML
preprocess + prompt mapping is correct.

Runs BOTH:
- PyTorch reference (mobile_sam.pt + vit_t)
- CoreML artifacts (MobileSAM_ImageEncoder.mlpackage + MobileSAM_PromptMaskDecoder.mlpackage)

On shared/golden/bus.jpg with a box prompt (from shared/golden/mobilesam_bus_case.json).
Outputs summary stats for low_res_masks.

Usage:
  source /Users/jiansun/Documents/Doctor Courses/4455/env1/bin/activate
  python shared/mobilesam_split_regression.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MOBILESAM_ROOT = REPO_ROOT / "models" / "MobileSAM"
CHECKPOINT = REPO_ROOT / "models" / "mobile_sam.pt"
ENC_MLPKG = REPO_ROOT / "models" / "MobileSAM_ImageEncoder.mlpackage"
DEC_MLPKG = REPO_ROOT / "models" / "MobileSAM_PromptMaskDecoder.mlpackage"
GOLDEN_IMG = REPO_ROOT / "shared" / "golden" / "bus.jpg"
GOLDEN_CASE = REPO_ROOT / "shared" / "golden" / "mobilesam_bus_case.json"

# Make `import mobile_sam` work
sys.path.insert(0, str(MOBILESAM_ROOT))


@dataclass
class ResizePadTransform:
    scale: float
    new_w: int
    new_h: int
    pad_w: int
    pad_h: int


def sam_resize_longest_side(h: int, w: int, longest_side: int = 1024) -> ResizePadTransform:
    """Match SAM's resize rule used in ONNX helper: floor(x + 0.5).

    scale = longest_side / max(h, w)
    new = floor(scale * size + 0.5)

    Padding is applied to the bottom/right to reach (longest_side, longest_side).
    """
    scale = float(longest_side) / float(max(h, w))
    new_h = int(np.floor(scale * h + 0.5))
    new_w = int(np.floor(scale * w + 0.5))
    pad_h = longest_side - new_h
    pad_w = longest_side - new_w
    assert pad_h >= 0 and pad_w >= 0
    return ResizePadTransform(scale=scale, new_w=new_w, new_h=new_h, pad_w=pad_w, pad_h=pad_h)


def preprocess_to_1024_rgb_float(image_path: Path) -> tuple[np.ndarray, ResizePadTransform]:
    """Return NCHW float32 in 0..255, RGB, padded to 1024x1024."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    tfm = sam_resize_longest_side(h, w, 1024)

    img_resized = img.resize((tfm.new_w, tfm.new_h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (1024, 1024), (0, 0, 0))
    canvas.paste(img_resized, (0, 0))  # pad right/bottom

    arr = np.asarray(canvas).astype(np.float32)  # HWC, 0..255
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # 1x3x1024x1024
    return arr, tfm


def box_to_points_1024(box_xyxy: np.ndarray, tfm: ResizePadTransform) -> tuple[np.ndarray, np.ndarray]:
    """Map original-image xyxy -> 1024 input coords and encode as SAM box points.

    point_coords: (1,2,2)
    point_labels: (1,2) with [2,3]

    Note: padding is only on right/bottom, so top-left origin unchanged.
    """
    x0, y0, x1, y1 = box_xyxy.astype(np.float32).tolist()
    x0n = x0 * tfm.scale
    y0n = y0 * tfm.scale
    x1n = x1 * tfm.scale
    y1n = y1 * tfm.scale

    point_coords = np.array([[[x0n, y0n], [x1n, y1n]]], dtype=np.float32)
    point_labels = np.array([[2.0, 3.0]], dtype=np.float32)
    return point_coords, point_labels


def summarize_mask(low_res_masks: np.ndarray) -> dict:
    m = low_res_masks.astype(np.float32)
    return {
        "shape": list(m.shape),
        "min": float(m.min()),
        "max": float(m.max()),
        "mean": float(m.mean()),
        "std": float(m.std()),
        "area_logits_gt0": int((m > 0).sum()),
    }


def run_pytorch(image_nchw: np.ndarray, point_coords: np.ndarray, point_labels: np.ndarray) -> dict:
    import torch
    from mobile_sam import sam_model_registry

    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT)).eval()

    # encoder wrapper (same as CoreML export script)
    pixel_mean = sam.pixel_mean.cpu().numpy().astype(np.float32)
    pixel_std = sam.pixel_std.cpu().numpy().astype(np.float32)

    img = image_nchw.copy()
    img = (img - pixel_mean[None, :, :, :]) / pixel_std[None, :, :, :]

    with torch.no_grad():
        img_t = torch.from_numpy(img)
        emb = sam.image_encoder(img_t)

        # decoder wrapper
        emb_t = emb
        pc_t = torch.from_numpy(point_coords)
        pl_t = torch.from_numpy(point_labels)
        mask_input = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
        has_mask = torch.zeros(1, dtype=torch.float32)

        # Re-implement the wrapper logic: use SamOnnxModel style embedding functions
        # but easiest is to import our wrapper from export script.
    
    # Import wrapper for exactness
    import importlib.util

    export_py = REPO_ROOT / "shared" / "export_mobilesam_split_to_coreml.py"
    spec = importlib.util.spec_from_file_location("export_mobilesam_split_to_coreml", export_py)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    dec = mod.PromptMaskDecoderWrapper(sam, return_single_mask=True).eval()
    with torch.no_grad():
        low_res_masks_t, iou_t = dec(emb_t, pc_t, pl_t, mask_input, has_mask)

    return {
        "low_res_masks": low_res_masks_t.cpu().numpy(),
        "iou_predictions": iou_t.cpu().numpy(),
    }


def run_coreml(image_nchw: np.ndarray, point_coords: np.ndarray, point_labels: np.ndarray) -> dict:
    import coremltools as ct

    enc = ct.models.MLModel(str(ENC_MLPKG))
    dec = ct.models.MLModel(str(DEC_MLPKG))

    emb = enc.predict({"image": image_nchw})["image_embeddings"]

    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask = np.zeros((1,), dtype=np.float32)

    out = dec.predict(
        {
            "image_embeddings": emb,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask,
        }
    )
    return {
        "image_embeddings": emb,
        "low_res_masks": out["low_res_masks"],
        "iou_predictions": out["iou_predictions"],
    }


def main():
    assert GOLDEN_IMG.exists(), GOLDEN_IMG
    assert GOLDEN_CASE.exists(), GOLDEN_CASE

    case = json.loads(GOLDEN_CASE.read_text())

    # Use YOLO box on the same golden image coordinate system.
    # The case file stores xyxy on a 640x640 resized version; in our repo the golden bus.jpg is 640x640.
    box = np.array(case["yolo_ref"]["box_xyxy_640"], dtype=np.float32)

    image_nchw, tfm = preprocess_to_1024_rgb_float(GOLDEN_IMG)
    point_coords, point_labels = box_to_points_1024(box, tfm)

    pt = run_pytorch(image_nchw, point_coords, point_labels)
    cm = run_coreml(image_nchw, point_coords, point_labels)

    pt_sum = summarize_mask(pt["low_res_masks"])
    cm_sum = summarize_mask(cm["low_res_masks"])

    # Read actual image size for record
    from PIL import Image
    w, h = Image.open(GOLDEN_IMG).size

    report = {
        "image": str(GOLDEN_IMG),
        "orig_size_hw": [int(h), int(w)],
        "bbox_xyxy": box.tolist(),
        "resize_pad": tfm.__dict__,
        "point_coords_1024": point_coords.tolist(),
        "point_labels": point_labels.tolist(),
        "pytorch": {
            "low_res_masks": pt_sum,
            "iou_predictions": {
                "shape": list(np.asarray(pt["iou_predictions"]).shape),
                "value": float(np.asarray(pt["iou_predictions"]).reshape(-1)[0]),
            },
        },
        "coreml": {
            "low_res_masks": cm_sum,
            "iou_predictions": {
                "shape": list(np.asarray(cm["iou_predictions"]).shape),
                "value": float(np.asarray(cm["iou_predictions"]).reshape(-1)[0]),
            },
        },
        "diff": {
            "low_res_masks_mean_abs": float(np.mean(np.abs(pt["low_res_masks"] - cm["low_res_masks"]))),
            "low_res_masks_max_abs": float(np.max(np.abs(pt["low_res_masks"] - cm["low_res_masks"]))),
        },
        "case_mobilesam_mask_area_px_640": int(case["mobilesam"]["mask_area_px"]),
        "case_mobilesam_mask_score": float(case["mobilesam"]["mask_score"]),
    }

    out_path = REPO_ROOT / "shared" / "golden" / "mobilesam_bus_coreml_regression.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(report["pytorch"], indent=2))
    print(json.dumps(report["coreml"], indent=2))
    print(json.dumps(report["diff"], indent=2))


if __name__ == "__main__":
    main()
