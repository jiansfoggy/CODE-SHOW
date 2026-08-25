"""Convert YOLOv9 ONNX -> CoreML (mlpackage).

Day2 goal: produce a *draft* CoreML artifact that loads and runs on iOS.
We intentionally keep YOLO decode/NMS outside the model for debugability.

Usage:
  ./env_export311/bin/python shared/export_yolov9_onnx_to_coreml.py \
    --onnx models/yolov9-c.onnx \
    --out models/yolov9-c-raw-fp16.mlpackage

Notes:
- Input is float tensor (NCHW) named 'images' produced by yolov9/export.py
- Output is 'output0' with shape (1, 84, 8400) for yolov9-c.pt @ 640
"""

import argparse
from pathlib import Path

import coremltools as ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--ios", type=int, default=17)
    args = ap.parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # YOLOv9 repo exports ONNX with input name: 'images'
    # Keep as tensor input (Swift will feed MLMultiArray or use a wrapper).
    # If Builder prefers CVPixelBuffer input, we can later re-export with ImageType.
    mlmodel = ct.converters.onnx.convert(
        model=str(onnx_path),
        minimum_deployment_target=getattr(ct.target, f"iOS{args.ios}"),
        compute_precision=ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32,
    )

    mlmodel.save(str(out_path))
    print(f"Saved CoreML model to: {out_path}")
    print("=== CoreML I/O ===")
    for k, v in mlmodel.input_description.items():
        print("input", k, ":", v)
    for k, v in mlmodel.output_description.items():
        print("output", k, ":", v)


if __name__ == "__main__":
    main()
