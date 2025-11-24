"""Model utilities: load YOLOv9 and MobileSAM, run detection and mask generation.

This module provides pragmatic, robust helpers that try to load models from the
original repos if they are installed in the environment, and otherwise fall back
to simple torch loading. You must adapt the MobileSAM loading to the exact API
in the upstream repo.

See `README_SETUP.md` for instructions on obtaining weights.
"""
from typing import List, Tuple, Dict, Any, NamedTuple
import numpy as np
from PIL import Image
import base64
import io
import os
import sys

try:
    import torch
except Exception as e:
    raise RuntimeError('torch is required: install with `pip install torch`')


class Detection(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    score: float


def load_yolov9_model(weights_path: str, device: str = 'cpu'):
    """Load a YOLOv9 model from weights.

    Strategy:
    - If the `yolov9` package (from the upstream repo) is importable, use its
      `DetectMultiBackend` or equivalent wrapper.
    - Otherwise try `torch.jit.load` or `torch.load` as a fallback.
    """
    device_t = torch.device(device)
    # Try upstream wrapper
    try:
        from yolov9.models.common import DetectMultiBackend  # type: ignore
        model = DetectMultiBackend(weights_path, device=device_t)
        model.to(device_t)
        return model
    except Exception:
        pass

    # Try ultralytics style YOLO if available
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO(weights_path)
        return model
    except Exception:
        pass

    # Fallback: torch.load
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'YOLO weights not found at {weights_path}')
    try:
        model = torch.jit.load(weights_path, map_location=device_t)
        model.eval()
        return model
    except Exception:
        model = torch.load(weights_path, map_location=device_t)
        if hasattr(model, 'eval'):
            model.eval()
        return model


def load_mobilesam_model(weights_path: str, device: str = 'cpu'):
    """Load MobileSAM model.

    The MobileSAM repo provides its own factory functions. If you have cloned or
    installed MobileSAM in the same Python environment, this function will try
    to use that API. Otherwise it raises a helpful error directing you to the
    MobileSAM repo.
    """
    device_t = torch.device(device)
    # Common import name in the MobileSAM project may vary; try a few options.
    tried = []
    try:
        # Attempt to import MobileSAM project structure
        import MobileSAM  # type: ignore
        # The upstream MobileSAM repo typically exposes model building utilities
        if hasattr(MobileSAM, 'build_sam'):
            model = MobileSAM.build_sam(weights_path, device=device_t)
            return model
        elif hasattr(MobileSAM, 'build_mobilesam'):
            model = MobileSAM.build_mobilesam(weights_path, device=device_t)
            return model
    except Exception as e:
        tried.append(str(e))

    # If we reach here, provide actionable guidance
    raise RuntimeError(
        'MobileSAM model loader: could not import MobileSAM.\n'
        'Please follow https://github.com/ChaoningZhang/MobileSAM to install or '
        'place the MobileSAM project on PYTHONPATH, then adapt `load_mobilesam_model`.'
    )


def detect_objects(image: np.ndarray, yolomodel, conf_thres: float = 0.25) -> List[Dict[str, Any]]:
    """Run object detection and return list of dicts: {bbox:[x1,y1,x2,y2], class_id, score}

    This function attempts to handle a few common model wrappers (DetectMultiBackend,
    ultralytics YOLO, or plain torchscript modules). You may need to adapt result
    parsing depending on your exact yolov9 package.
    """
    detections: List[Dict[str, Any]] = []

    # ultralytics YOLO wrapper
    try:
        from ultralytics.yolo.engine.results import Results  # type: ignore
        # If model is ultralytics YOLO
        results = yolomodel(image)
        # results may be a list-like; take first
        r = results[0]
        # r.boxes.xyxy, r.boxes.conf, r.boxes.cls
        boxes = getattr(r, 'boxes', None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            for b, c, cl in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = map(int, b[:4])
                detections.append({'bbox': [x1, y1, x2, y2], 'score': float(c), 'class_id': int(cl)})
            return detections
    except Exception:
        pass

    # DetectMultiBackend-like wrapper used in many YOLO forks
    try:
        res = yolomodel(image)
        # try to access .xyxy or .pred
        if hasattr(res, 'xyxy'):
            arr = res.xyxy[0].cpu().numpy()
            for det in arr:
                x1, y1, x2, y2, score, cls = det[:6]
                detections.append({'bbox': [int(x1), int(y1), int(x2), int(y2)], 'score': float(score), 'class_id': int(cls)})
            return detections
    except Exception:
        pass

    # Torchscript / generic model: we can't parse outputs reliably
    return detections


def run_mobilesam_masks(image: np.ndarray, boxes: List[List[int]], clicks: List[Tuple[int, int]], mobilesam_model) -> List[np.ndarray]:
    """Given image, bounding boxes and optional click points, return masks (numpy arrays) per box.

    This function calls into MobileSAM's API. You must adapt it to the real API.
    Here we attempt a common pattern `predict_box` or `predict` on `mobilesam_model`.
    """
    masks: List[np.ndarray] = []
    h, w = image.shape[:2]
    for box in boxes:
        try:
            if hasattr(mobilesam_model, 'predict_box'):
                mask = mobilesam_model.predict_box(image, box)
            elif hasattr(mobilesam_model, 'predict'):
                # some APIs: predict(image, boxes=...)
                mask = mobilesam_model.predict(image, boxes=[box])
                # if predict returns list
                if isinstance(mask, list):
                    mask = mask[0]
            else:
                raise RuntimeError('Unknown MobileSAM API; adapt run_mobilesam_masks')
            # Ensure mask is uint8 0/255
            mask_arr = np.array(mask)
            if mask_arr.dtype != np.uint8:
                mask_arr = (mask_arr > 0.5).astype(np.uint8) * 255
            masks.append(mask_arr)
        except Exception:
            masks.append(np.zeros((h, w), dtype=np.uint8))
    return masks


def mask_to_base64_png(mask: np.ndarray) -> str:
    pil = Image.fromarray(mask).convert('L')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return b64
