Real-time Video Segmentation iOS App (Prototype)

This folder contains a prototype scaffold for an iOS app that performs real-time video segmentation using YOLOv9 for detection and MobileSAM for mask generation.

Structure:
- `python/` - Python inference server and model utilities (FastAPI). Intended to run the models and expose an HTTP API that the iOS app can call.
- `swift/` - Swift example snippets for camera capture, touch handling, sending frames to the inference server, and displaying masks.

Important notes:
- The model repositories referenced:
  - MobileSAM: https://github.com/ChaoningZhang/MobileSAM
  - YOLOv9: https://github.com/WongKinYiu/yolov9 (use `yolov9-c.pt`)
- This prototype implements the Python inference side (with clear load/run hooks) and a Swift client example. Running a Python server on-device on iOS requires embedding a Python runtime (non-trivial) or converting models to Core ML and running natively in Swift.
- See `README_SETUP.md` for setup, model download, and conversion guidance.

If you want, I can:
- finish conversion instructions from PyTorch -> Core ML and provide `coremltools` snippets;
- or produce a full Xcode project that runs models natively using converted Core ML models.

---
Refer to subfolders for details.
