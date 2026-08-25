"""YOLOv9 (JudgeEverything) minimal Python reference decoder

Goal (Day 3 ML_Vision deliverable):
- Decode raw head tensor shaped (1, 84, 8400) -> boxes/classes
- Apply score threshold + NMS

This matches the Detect head implementation in `models/yolov9/models/yolo.py`:
  y = cat(dbox_xywh, cls_sigmoid)
So:
- first 4 channels are bbox in **xywh (center-x, center-y, w, h)**, **absolute pixels** in model input space
- remaining 80 channels are **class probabilities** after sigmoid (COCO classes)
- there is NO separate objectness channel in this exported tensor

Outputs are in model-input coordinates (e.g., 640x640). To map to camera px,
apply the inverse letterbox transform defined in `shared/architect_output.md`.

Usage (example):
  import coremltools as ct
  from PIL import Image
  from yolov9_reference_decoder import decode

  m = ct.models.MLModel('models/yolov9-c.mlmodel')
  img = Image.open('shared/golden/bus.jpg').convert('RGB').resize((640,640))
  out = m.predict({'image': img})
  raw = out['var_3019']
  dets = decode(raw, score_threshold=0.25, iou_threshold=0.65, topk=100)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray  # (4,) float32 in model-input pixels: x1,y1,x2,y2
    score: float
    class_id: int


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    # xywh: (...,4) where xy is center
    x, y, w, h = np.split(xywh, 4, axis=-1)
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.concatenate([x1, y1, x2, y2], axis=-1)


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box (4,) and many boxes (N,4), xyxy."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-9
    return inter / union


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.65,
    topk: int = 100,
) -> np.ndarray:
    """Classic NMS, class-agnostic. Returns kept indices."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    idxs = scores.argsort()[::-1]
    keep: List[int] = []

    while idxs.size > 0 and len(keep) < topk:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou_xyxy(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_threshold]

    return np.array(keep, dtype=np.int64)


def decode(
    raw: np.ndarray,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    topk: int = 100,
    per_class_nms: bool = True,
    clip: Optional[Tuple[float, float]] = (640.0, 640.0),
) -> List[Detection]:
    """Decode YOLOv9 raw output (1,84,8400) to a list of detections.

    Args:
      raw: np.ndarray float32, shape (1,84,N)
      score_threshold: keep candidates where max_class_prob >= threshold
      iou_threshold: NMS IoU
      topk: max kept boxes after NMS
      per_class_nms: if True, run NMS per class; else class-agnostic
      clip: (W,H) to clip xyxy to image bounds; set None to skip

    Returns:
      List[Detection] sorted by score descending.
    """
    raw = np.asarray(raw)
    assert raw.ndim == 3 and raw.shape[0] == 1 and raw.shape[1] == 84, f"expected (1,84,N), got {raw.shape}"

    # (N,4) xywh, (N,80) cls_prob
    xywh = raw[0, 0:4, :].T.astype(np.float32)
    cls = raw[0, 4:, :].T.astype(np.float32)

    class_id = cls.argmax(axis=1)
    score = cls.max(axis=1)

    mask = score >= float(score_threshold)
    if not np.any(mask):
        return []

    xyxy = xywh_to_xyxy(xywh[mask])
    score_f = score[mask]
    class_id_f = class_id[mask]

    if clip is not None:
        W, H = clip
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0.0, W - 1.0)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0.0, W - 1.0)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0.0, H - 1.0)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0.0, H - 1.0)

    keep_all: List[int] = []
    if per_class_nms:
        for c in np.unique(class_id_f):
            idx = np.where(class_id_f == c)[0]
            kept = nms_xyxy(xyxy[idx], score_f[idx], iou_threshold=iou_threshold, topk=topk)
            keep_all.extend(idx[kept].tolist())
        keep_all = sorted(keep_all, key=lambda i: float(score_f[i]), reverse=True)[:topk]
        keep = np.array(keep_all, dtype=np.int64)
    else:
        keep = nms_xyxy(xyxy, score_f, iou_threshold=iou_threshold, topk=topk)

    dets = [
        Detection(xyxy=xyxy[i].astype(np.float32), score=float(score_f[i]), class_id=int(class_id_f[i]))
        for i in keep
    ]
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets
