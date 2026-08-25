"""Export YOLOv9 .pt -> CoreML .mlpackage (ML Program).

Why this script exists:
- Upstream yolov9/export.py uses `.mlmodel` suffix, but coremltools>=7 defaults
  to `mlprogram` which must be saved as `.mlpackage`.

This script produces a draft artifact for iOS integration / smoke test.
Decode + NMS are *not* included; CoreML output is the raw head tensor.

Usage:
  ./Copilot/python/env_export311/bin/python shared/export_yolov9_pt_to_coreml_mlpackage.py \
    --weights models/yolov9-c.pt \
    --imgsz 640 \
    --out models/yolov9-c-raw.mlpackage \
    --ios 17

Expected I/O (for yolov9-c @ 640):
- input: image (RGB), 640x640, scale=1/255
- output: (1, 84, 8400) float (flattened tuple handled by coremltools)
"""

import argparse
import sys
from pathlib import Path

import torch
import coremltools as ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ios", type=int, default=17)
    ap.add_argument("--compute_precision", type=str, default="fp16", choices=["fp16", "fp32"])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1] / "models" / "yolov9"
    # allow: import models.experimental etc
    sys.path.insert(0, str(repo_root))

    from models.experimental import attempt_load  # noqa: E402

    weights = Path(args.weights).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = attempt_load(str(weights), device=device, inplace=True, fuse=True)
    model.eval()

    # Match yolov9/export.py behavior: force Detect heads to export a single tensor
    # (otherwise forward() may return nested python lists which TorchScript can't trace).
    try:
        from models.yolo import Detect, DDetect, DualDetect, DualDDetect  # noqa: E402

        for _, m in model.named_modules():
            if isinstance(m, (Detect, DDetect, DualDetect, DualDDetect)):
                m.inplace = False
                m.dynamic = False
                m.export = True
    except Exception as e:
        print(f"[WARN] Could not set export flags on Detect modules: {e}")

    # Dummy input for tracing
    im = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)

    # Dry runs (match yolov9/export.py)
    with torch.no_grad():
        for _ in range(2):
            _ = model(im)

    # Trace
    ts = torch.jit.trace(model, im, strict=False)

    # Convert to ML Program (.mlpackage)
    precision = ct.precision.FLOAT16 if args.compute_precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        ts,
        inputs=[ct.ImageType(name="image", shape=im.shape, scale=1 / 255.0, bias=[0.0, 0.0, 0.0])],
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, f"iOS{args.ios}"),
        compute_precision=precision,
    )

    mlmodel.save(str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
