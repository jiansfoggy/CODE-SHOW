"""
Phase 3 — ANE Alignment Fix: Re-export MobileSAM ImageEncoder with Float32 precision

Root Cause:
  MobileSAM_ImageEncoder.mlpackage (current, Float16) contains LayerNorm2d operations
  that produce intermediate tensors with shape [1, 1, 64, 64] (C=1, Float16).
  ANE on iPhone 11 (A13 Bionic) requires Float16 tensor channels to be aligned on
  64-byte boundaries: C * 2 bytes must be divisible by 64, i.e., C must be a multiple of 32.
  C=1 fails this requirement → runtime warning:
    "Invalid input tensor channel 1 ... must be aligned on 64 bytes"
  These layers fall back to CPU, degrading encoder latency.

Specific source:
  LayerNorm2d in the encoder neck:
    x.mean(1, keepdim=True)  →  [1, 1, 64, 64]  (C=1, unaligned in Float16)
  Two such reduce_mean ops found at the end of the 1188-op encoder graph.

Fix Strategy:
  Re-export with compute_precision=ct.precision.FLOAT32 (change one line).
  Float32 on ANE does not trigger the same 64-byte channel alignment warning,
  allowing the full network (including the neck LayerNorm) to run on ANE.

  Trade-off:
    - Model file size: ~22 MB → ~44 MB
    - All ops use Float32 weights and activations
    - Expected benefit: encoder can now fully schedule on ANE / GPU without
      CPU fallback on the LayerNorm neck ops

  Alternative (advanced, Phase 3 Day 3 if Float32 latency still unacceptable):
    Use coremltools MIL PassPipeline to fuse the mean/variance ops into a single
    layer_norm MIL op. The fused op does not expose the C=1 intermediate to the
    ANE scheduler, so Float16 can be kept for the rest of the network.
    See export_encoder_fp16_milfix.py (future).

Usage:
  cd /path/to/JudgeE2
  source /Users/jiansun/Documents/Doctor\ Courses/4455/env1/bin/activate
  python3 shared/export_encoder_fp32_ane_fix.py

Output:
  models/MobileSAM_ImageEncoder_fp32.mlpackage
  (original MobileSAM_ImageEncoder.mlpackage is preserved)
"""

import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct

REPO_ROOT = Path(__file__).resolve().parents[1]
MOBILESAM_ROOT = REPO_ROOT / "models" / "MobileSAM"
CHECKPOINT = REPO_ROOT / "models" / "mobile_sam.pt"
OUT_DIR = REPO_ROOT / "models"

sys.path.insert(0, str(MOBILESAM_ROOT))
from mobile_sam import sam_model_registry  # noqa: E402


class ImageEncoderWrapper(nn.Module):
    """
    Wrapper identical to the original export script.
    Input:  (1, 3, 1024, 1024) Float32, RGB, values in [0, 255]
    Output: (1, 256, 64, 64)  Float32 image embeddings
    Preprocessing (SAM normalization) is fused inside the model.
    """
    def __init__(self, sam):
        super().__init__()
        self.image_encoder = sam.image_encoder
        self.register_buffer("pixel_mean", sam.pixel_mean, persistent=False)
        self.register_buffer("pixel_std", sam.pixel_std, persistent=False)

    def forward(self, image: torch.Tensor):
        # image: (1, 3, 1024, 1024), RGB, float32, 0..255
        x = (image - self.pixel_mean) / self.pixel_std
        emb = self.image_encoder(x)
        return emb


def main():
    assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MobileSAM vit_t checkpoint...")
    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
    sam.eval()

    encoder = ImageEncoderWrapper(sam).eval()

    # Trace encoder with dummy input
    dummy_image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    print("Tracing encoder...")
    traced_encoder = torch.jit.trace(encoder, (dummy_image,))

    # Convert to CoreML with FLOAT32 precision (the key change vs original export)
    print("Converting ImageEncoder to CoreML (Float32 precision — ANE fix)...")
    encoder_mlmodel = ct.convert(
        traced_encoder,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="image", shape=dummy_image.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="image_embeddings", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,   # <-- KEY CHANGE: was FLOAT16
    )

    out_path = OUT_DIR / "MobileSAM_ImageEncoder_fp32.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)
    encoder_mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")

    # Sanity check: verify I/O shapes
    spec = ct.utils.load_spec(str(out_path))
    desc = spec.description
    print("\n--- Verification ---")
    for inp in desc.input:
        t = inp.type
        if t.HasField("multiArrayType"):
            shape = list(t.multiArrayType.shape)
            dtype = t.multiArrayType.dataType
            dtype_str = "Float32" if dtype == 65600 else "Float16" if dtype == 65568 else f"raw({dtype})"
            print(f"  Input  '{inp.name}': {shape}  {dtype_str}")
    for out in desc.output:
        t = out.type
        if t.HasField("multiArrayType"):
            shape = list(t.multiArrayType.shape)
            dtype = t.multiArrayType.dataType
            dtype_str = "Float32" if dtype == 65600 else "Float16" if dtype == 65568 else f"raw({dtype})"
            print(f"  Output '{out.name}': {shape}  {dtype_str}")

    print("\nDone. Deploy MobileSAM_ImageEncoder_fp32.mlpackage to Xcode.")
    print("Replace the existing MobileSAM_ImageEncoder.mlpackage reference in the project.")
    print("Expected: 'Invalid input tensor channel 1 ... aligned on 64 bytes' warning disappears.")


if __name__ == "__main__":
    main()
