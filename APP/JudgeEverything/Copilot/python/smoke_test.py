"""Simple smoke test to verify model loading for YOLOv9 and MobileSAM.

Run this after placing weights in `python/models/` to see whether models load.
"""
import os
from model_utils import load_yolov9_model, load_mobilesam_model

YOLO_WEIGHTS = 'python/models/yolov9-c.pt'
MOBILESAM_WEIGHTS = 'python/models/mobilesam.pth'

def try_load_yolo():
    print('=== YOLOv9 Load Test ===')
    if not os.path.exists(YOLO_WEIGHTS):
        print('YOLO weights not found at', YOLO_WEIGHTS)
        return False
    try:
        m = load_yolov9_model(YOLO_WEIGHTS)
        print('YOLO model loaded:', type(m))
        return True
    except Exception as e:
        print('YOLO load error:', e)
        return False

def try_load_mobilesam():
    print('=== MobileSAM Load Test ===')
    if not os.path.exists(MOBILESAM_WEIGHTS):
        print('MobileSAM weights not found at', MOBILESAM_WEIGHTS)
        return False
    try:
        m = load_mobilesam_model(MOBILESAM_WEIGHTS)
        print('MobileSAM model loaded:', type(m))
        return True
    except Exception as e:
        print('MobileSAM load error:', e)
        return False

if __name__ == '__main__':
    y = try_load_yolo()
    s = try_load_mobilesam()
    if y and s:
        print('Smoke test: both models loaded')
    else:
        print('Smoke test: one or more models failed to load. See messages above.')
