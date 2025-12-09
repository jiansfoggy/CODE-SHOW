Real-time Video Segmentation iOS App (Prototype)

This folder contains a complete scaffold for an iOS app that performs real-time video segmentation using YOLOv9 for object detection and MobileSAM for pixel-level mask generation.

## Quick Links

- **MacBook Air M3 users**: See `M3_MAC_QUICK_START.md` for optimized setup (10-minute conversion!)
- **Full conversion guide**: `CONVERT_ON_MAC.md`
- **Xcode project setup**: `XCODE_SETUP.md`

## Structure

- `python/` - Python inference server and model utilities (FastAPI). Can also be used for local model conversion to Core ML on macOS.
- `swift/` - Swift code for on-device Core ML inference (camera capture, segmentation, mask display).
- `coreml/` - Generated Core ML model files (after conversion on Mac).

## High-level Flow

1. **Step A (Validation)**: Verify model loading with `smoke_test.py`.
2. **Step B (Conversion)**: Run conversion scripts on macOS to generate `.mlmodel` files (2.7.0 & coremltools 6.3).
3. **Step C (iOS App)**: Build Xcode project with on-device Core ML inference for real-time segmentation.

## Model References

- **YOLOv9**: https://github.com/WongKinYiu/yolov9 (use `yolov9-c.pt`)
- **MobileSAM**: https://github.com/ChaoningZhang/MobileSAM (TinyViT encoder)

## Important Notes

- The model repositories have been cloned locally to `python/models/yolov9/` and `python/models/MobileSAM/` for easy reference.
- This is an on-device inference setup (Core ML), so latency is low and data stays private.
- App Store compatible: models are embedded; no runtime code generation.
