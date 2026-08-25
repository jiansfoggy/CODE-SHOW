"""
Phase 3 — Day 4: 768×768 Encoder Export (milfix fp16) for AB Test

Authorised by Architect §8.6 (architect_output.md) — 768 conditionally approved
for Builder AB testing.  This script derives directly from
`export_encoder_fp16_milfix.py`, applying two additional changes required by
model_plan.md §C.5:

  1. dummy_trace = torch.randn(1, 3, 768, 768)   (instead of 1024)
  2. monkeypatch TinyViT.forward_features to compute feat_size dynamically:
         feat_size = self.img_size // 16      # 1024→64, 768→48, 512→32
     (original code hardcodes  x.view(B, 64, 64, C)  → crashes on non-1024 input)

The milfix LayerNorm2d patch (ANE alignment fix, §B.4) is retained unchanged so
the 768 variant keeps the same ANE-alignment behaviour as the 1024 milfix model.

OUTPUT:
    models/MobileSAM_ImageEncoder_fp16_milfix_768.mlpackage  (~14 MB, fp16)

Encoder output shape for 768 input: [1, 256, 48, 48]
  → Swift layer must bilinearly upsample to [1, 256, 64, 64] before Decoder
    (Decoder is fixed at 64×64 embedding — see Architect C-3).

USAGE:
    cd /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2
    source "/Users/jiansun/Documents/Doctor Courses/4455/env1/bin/activate"
    python3 shared/export_encoder_fp16_milfix_768.py
"""

import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MOBILESAM_ROOT = REPO_ROOT / "models" / "MobileSAM"
CHECKPOINT = REPO_ROOT / "models" / "mobile_sam.pt"
OUT_DIR = REPO_ROOT / "models"
OUT_NAME = "MobileSAM_ImageEncoder_fp16_milfix_768"

# Target encoder input resolution for this AB variant.
INPUT_SIZE = 768
FEAT_SIZE = INPUT_SIZE // 16   # 768 → 48

sys.path.insert(0, str(MOBILESAM_ROOT))

# ---------------------------------------------------------------------------
# milfix: patch LayerNorm2d BEFORE importing the model (ANE alignment fix §B.4)
# ---------------------------------------------------------------------------
from mobile_sam.modeling.common import LayerNorm2d  # noqa: E402


def _layernorm2d_forward_milfix(self, x: torch.Tensor) -> torch.Tensor:
    """ANE-friendly LayerNorm2d: transpose → F.layer_norm([C]) → transpose back.
    Maps to the fused MIL `layer_norm` op, avoiding C=1 fp16 intermediates."""
    x = x.permute(0, 2, 3, 1)                              # [B, H, W, C]
    x = F.layer_norm(x, [x.shape[-1]], self.weight, self.bias, self.eps)
    return x.permute(0, 3, 1, 2)                           # [B, C, H, W]


LayerNorm2d.forward = _layernorm2d_forward_milfix
print("[milfix] common.LayerNorm2d.forward patched")

from mobile_sam.modeling.tiny_vit_sam import LayerNorm2d as NeckLayerNorm2d  # noqa: E402
NeckLayerNorm2d.forward = _layernorm2d_forward_milfix
print("[milfix] tiny_vit_sam.LayerNorm2d.forward patched (neck C=1 culprit)")

# ---------------------------------------------------------------------------
# Dynamic feat_size patch (model_plan §C.1): forward_features hardcodes
# x.view(B, 64, 64, C), which crashes on non-1024 input.  Replace with a
# resolution-aware version so 768 input yields a 48×48 feature grid.
#
# NOTE: The internal window-attention layers assert L == H*W based on the
# `resolution` passed when TinyViT is CONSTRUCTED.  Merely patching
# forward_features is not enough (assert fails at build-time resolution).
# We must BUILD TinyViT with img_size=INPUT_SIZE and load the encoder
# weights from the 1024 checkpoint.  Per model_plan §C.1, all attention_biases
# depend on window_size (fixed), not input_resolution, so weights transfer
# losslessly (0 missing / 0 unexpected keys).
# ---------------------------------------------------------------------------
from mobile_sam.modeling.tiny_vit_sam import TinyViT  # noqa: E402


def _forward_features_dynamic(self, x):
    # x: (N, C, H, W)
    x = self.patch_embed(x)
    x = self.layers[0](x)
    for i in range(1, len(self.layers)):
        x = self.layers[i](x)
    B, N, C = x.size()
    feat_size = self.img_size // 16          # 1024→64, 768→48, 512→32
    x = x.view(B, feat_size, feat_size, C)
    x = x.permute(0, 3, 1, 2)
    x = self.neck(x)
    return x


TinyViT.forward_features = _forward_features_dynamic
print(f"[patch] TinyViT.forward_features dynamic (feat_size = img_size // 16 = {FEAT_SIZE})")

from mobile_sam import sam_model_registry  # noqa: E402


def build_encoder_at(input_size: int) -> nn.Module:
    """Build a TinyViT image encoder configured for `input_size`, loading weights
    from the 1024 checkpoint (window-based attention biases transfer losslessly)."""
    encoder = TinyViT(
        img_size=input_size, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )
    # Extract image_encoder.* weights from the full SAM checkpoint.
    state = torch.load(str(CHECKPOINT), map_location="cpu")
    enc_state = {k[len("image_encoder."):]: v
                 for k, v in state.items() if k.startswith("image_encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    # attention_biases_idxs are registered buffers rebuilt for the new resolution;
    # they are non-persistent, so they legitimately appear in `missing`. All learned
    # params (weights/biases/attention_biases) must load cleanly.
    real_missing = [k for k in missing if "attention_bias_idxs" not in k]
    print(f"[build] TinyViT(img_size={input_size}) loaded: "
          f"missing={len(missing)} (real={len(real_missing)}) unexpected={len(unexpected)}")
    if real_missing:
        print(f"[build] ⚠️  real missing keys: {real_missing[:8]}")
    if unexpected:
        print(f"[build] ⚠️  unexpected keys: {unexpected[:8]}")
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------
class ImageEncoderWrapper(nn.Module):
    """Fuses SAM pixel normalisation into the traced graph.
    Input:  (1, 3, INPUT_SIZE, INPUT_SIZE) Float32, RGB, [0, 255]
    Output: (1, 256, FEAT_SIZE, FEAT_SIZE)"""
    def __init__(self, image_encoder: nn.Module):
        super().__init__()
        self.image_encoder = image_encoder
        self.register_buffer("pixel_mean",
                             torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                             persistent=False)
        self.register_buffer("pixel_std",
                             torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
                             persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - self.pixel_mean) / self.pixel_std
        return self.image_encoder(x)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def _count_ops(mlmodel_path: Path) -> dict:
    from coremltools.converters.mil import load as mil_load
    prog = mil_load(str(mlmodel_path))
    counts: dict = {}
    for func in prog.functions.values():
        for op in func.operations:
            counts[op.op_type] = counts.get(op.op_type, 0) + 1
    return counts


def _verify_io(mlmodel_path: Path) -> None:
    DTYPE_NAMES = {65568: "Float16", 65600: "Float32"}
    spec = ct.utils.load_spec(str(mlmodel_path))
    desc = spec.description
    print("\n--- I/O Verification ---")
    for inp in desc.input:
        t = inp.type
        if t.HasField("multiArrayType"):
            shape = list(t.multiArrayType.shape)
            dtype = DTYPE_NAMES.get(t.multiArrayType.dataType,
                                    str(t.multiArrayType.dataType))
            print(f"  Input  '{inp.name}': {shape}  {dtype}")
    for out in desc.output:
        t = out.type
        if t.HasField("multiArrayType"):
            shape = list(t.multiArrayType.shape)
            dtype = DTYPE_NAMES.get(t.multiArrayType.dataType,
                                    str(t.multiArrayType.dataType))
            print(f"  Output '{out.name}': {shape}  {dtype}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building TinyViT image encoder at input={INPUT_SIZE} ...")
    image_encoder = build_encoder_at(INPUT_SIZE)
    print(f"[info] image_encoder.img_size = {image_encoder.img_size}")

    encoder = ImageEncoderWrapper(image_encoder).eval()

    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    with torch.no_grad():
        out = encoder(dummy)
    expected = (1, 256, FEAT_SIZE, FEAT_SIZE)
    assert out.shape == expected, f"Unexpected output shape: {out.shape} (expected {expected})"
    print(f"[sanity] Forward pass OK — output shape {tuple(out.shape)}")

    print(f"Tracing encoder ({INPUT_SIZE}×{INPUT_SIZE}) ...")
    dummy_trace = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(encoder, (dummy_trace,))
    print("Tracing done.")

    print("Converting to CoreML mlprogram (compute_precision=FLOAT16) ...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="image", shape=dummy_trace.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="image_embeddings", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )

    out_path = OUT_DIR / f"{OUT_NAME}.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    print(f"\nSaved: {out_path}")

    _verify_io(out_path)

    try:
        op_counts = _count_ops(out_path)
        print("\n--- Op Counts (key ops) ---")
        print(f"  layer_norm ops  : {op_counts.get('layer_norm', 0)}")
        print(f"  reduce_mean ops : {op_counts.get('reduce_mean', 0)}  "
              f"({'✅ none (milfix ok)' if op_counts.get('reduce_mean', 0) == 0 else '⚠️  present'})")
    except Exception as e:
        print(f"[warn] Could not count ops (non-critical): {e}")

    milfix_size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
    print(f"\n--- File Size ---\n  {OUT_NAME}: {milfix_size_mb:.1f} MB")

    print(f"""
=======================================================
Export complete: {out_path.name}
=======================================================
Encoder input : [1, 3, {INPUT_SIZE}, {INPUT_SIZE}]
Encoder output: [1, 256, {FEAT_SIZE}, {FEAT_SIZE}]  ← Swift must upsample to [1,256,64,64]

Next steps (Xcode integration):
  1. Drag {out_path} into JudgeE2/Segmentation/Models/ (tick target membership).
  2. SAMEncoder loads it when encoderInputSize == 768.
  3. Clean Build → Build → run on device (AB test).
""")


if __name__ == "__main__":
    main()
