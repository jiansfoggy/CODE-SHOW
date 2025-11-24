Setup, model download, conversion and packaging notes

This document explains how to set up the Python inference server locally (for prototyping) and options for running models on-device in an iOS app.

1) Quick local prototype (macOS or Linux) — recommended for development
- Install Python deps:
  - `python3 -m pip install -r python/requirements.txt`
- Download models (see `python/download_models.sh`) and place them into `python/models/`:
  - YOLOv9: use `yolov9-c.pt` from https://github.com/WongKinYiu/yolov9
  - MobileSAM: follow instructions at https://github.com/ChaoningZhang/MobileSAM and get the mobile weights
- Run the server:
  - `cd APP/JudgeEverything/Copilot/python`
  - `python inference_server.py` (runs FastAPI on port 8000)
- Launch the Swift sample in the iOS Simulator or device and point the HTTP endpoint to `http://<host-ip>:8000/segment` (Simulator on same machine can reach `localhost`).

2) iOS on-device deployment (production / App Store)
- Two main options:
  1. Convert models to Core ML and run natively in Swift (recommended for App Store):
     - Convert YOLOv9 (PyTorch) -> TorchScript -> coremltools -> `.mlmodel`.
     - Convert MobileSAM or replace with a SAM variant supported by Core ML. MobileSAM may need custom ops; conversion will require tracing and possibly code adaptation.
     - Use `coremltools` and `torch.jit.trace` to get reliable conversion. This yields best latency and App Store compatibility.
  2. Embed a Python runtime on iOS and run the Python server inside the app (complex and may violate App Store rules):
     - Projects like `Python-Apple-support` and `Pyto` demonstrate embedding Python on iOS.
     - This is fragile and not recommended for App Store distribution.

3) App Store considerations
- Apps must not download executable code after review — shipping model weights is fine but converting/downloading code at runtime may be problematic.
- Choose Core ML conversion whenever possible.

If you want me to attempt model conversion steps (example `coremltools` code) I can generate those next.
