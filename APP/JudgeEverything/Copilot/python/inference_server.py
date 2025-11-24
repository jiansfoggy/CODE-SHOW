"""FastAPI inference server for real-time segmentation prototype.

Endpoint: POST /segment
 - multipart: `file` (image/jpeg/png), `clicks` (JSON list of [x,y] points)
Response: JSON with detections and mask (base64 PNG)

Note: This is a prototype. You must adapt model loading calls to the exact MobileSAM and YOLOv9 APIs.
"""
import io
import json
import base64
from typing import List, Tuple

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from model_utils import load_yolov9_model, load_mobilesam_model, detect_objects, run_mobilesam_masks, mask_to_base64_png

app = FastAPI()

# Config - adjust paths
YOLO_WEIGHTS = 'python/models/yolov9-c.pt'
MOBILESAM_WEIGHTS = 'python/models/mobilesam.pth'
DEVICE = 'cpu'

print('Loading models...')
try:
    yolomodel = load_yolov9_model(YOLO_WEIGHTS, device=DEVICE)
except Exception as e:
    print('YOLO load warning:', e)
    yolomodel = None
try:
    mobilesam_model = load_mobilesam_model(MOBILESAM_WEIGHTS, device=DEVICE)
except Exception as e:
    print('MobileSAM load warning:', e)
    mobilesam_model = None

@app.post('/segment')
async def segment(file: UploadFile = File(...), clicks: str = Form('[]')):
    """Accepts image file and clicks JSON string. Returns mask for the clicked object.

    clicks: JSON string representing a list of [x,y] coordinates in image pixels.
    """
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    img_np = np.array(img)  # H,W,3 (RGB)

    try:
        click_points = json.loads(clicks)
    except Exception:
        click_points = []

    # 1) Detect objects
    detections = []
    if yolomodel is not None:
        detections = detect_objects(img_np, yolomodel)

    # 2) Choose box for click: find detection box that contains click or closest center
    chosen_box = None
    chosen_idx = None
    if len(detections) > 0 and len(click_points) > 0:
        cx, cy = click_points[0]
        best = None
        for i, det in enumerate(detections):
            x1,y1,x2,y2 = det['bbox']
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                best = i
                break
        if best is None:
            # choose nearest box center
            dmin = float('inf')
            for i, det in enumerate(detections):
                x1,y1,x2,y2 = det['bbox']
                mx = (x1+x2)/2
                my = (y1+y2)/2
                d = (mx-cx)**2 + (my-cy)**2
                if d < dmin:
                    dmin = d
                    best = i
        chosen_idx = best
        chosen_box = detections[best]['bbox']

    # 3) Run MobileSAM on chosen box (or on all boxes)
    masks_b64 = []
    if mobilesam_model is not None and chosen_box is not None:
        boxes = [chosen_box]
        masks = run_mobilesam_masks(img_np, boxes, click_points, mobilesam_model)
        for mask in masks:
            masks_b64.append(mask_to_base64_png(mask))

    resp = {
        'detections': detections,
        'chosen_index': chosen_idx,
        'masks': masks_b64,
    }
    return JSONResponse(resp)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('inference_server:app', host='0.0.0.0', port=8000, reload=False)
