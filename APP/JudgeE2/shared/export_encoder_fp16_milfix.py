"""
Phase 3 — ANE Alignment Fix (Option B): fp16 Encoder via MIL LayerNorm Fusion

# ============================================================
# WHY THIS SCRIPT EXISTS
# ============================================================

Previous fix (export_encoder_fp32_ane_fix.py) re-exported the encoder with
Float32 compute precision to eliminate the ANE Float16 alignment warning.
However, real-device testing revealed two regressions:

  1. Encoder latency increased +32%: fp32 mean=1131ms (vs fp16 857ms)
     Root cause: ANE is optimised for fp16/int8; fp32 ops fall back to CPU.

  2. ANE warnings were NOT eliminated (actually +3 new runtime warnings):
     "Invalid input tensor channel 1 ... must be aligned on 64 bytes"
     Root cause: ANE runtime internally casts some fp32 ops back to fp16,
     re-triggering the C=1 alignment check on LayerNorm2d intermediates.

# ============================================================
# ROOT CAUSE
# ============================================================

MobileSAM encoder neck uses LayerNorm2d (mobile_sam/modeling/common.py):

    def forward(self, x):                         # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)               # ← [B, 1, H, W]  ← C=1 trigger
        s = (x - u).pow(2).mean(1, keepdim=True)  # ← [B, 1, H, W]  ← C=1 trigger
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

ANE on iPhone 11 (A13): Float16 channel tensors must satisfy:
    C * sizeof(float16) ≡ 0  (mod 64)  →  C must be a multiple of 32
C=1 → 2 bytes → NOT 64-byte aligned → CPU fallback for the LayerNorm2d ops.

# ============================================================
# THE FIX: Monkeypatch LayerNorm2d → F.layer_norm with transpose
# ============================================================

coremltools converts torch.nn.functional.layer_norm directly to the CoreML
MIL `layer_norm` op (see coremltools/converters/mil/frontend/torch/ops.py).
The MIL `layer_norm` op is a single fused kernel; it does NOT expose any C=1
intermediate tensors to the ANE scheduler.  The compiler schedules the whole
op natively on ANE, satisfying alignment requirements internally.

Equivalence proof for the patched form:
    x_perm = x.permute(0,2,3,1)   # [B, H, W, C]
    F.layer_norm(x_perm, [C], weight, bias, eps)
      → for each (b, h, w): normalises over the C channels
      → output[b, h, w, c] = weight[c] * (x[b,c,h,w] - μ(x[b,:,h,w]))
                                        / σ(x[b,:,h,w]) + bias[c]
This is identical to the original LayerNorm2d.forward for all (b,c,h,w). ✓

# ============================================================
# EXPECTED OUTCOMES
# ============================================================

  1. ANE alignment warning disappears (no C=1 fp16 intermediate exposed)
  2. Encoder latency returns to fp16 baseline (~857ms mean)  — verify with Debugger
  3. YOLO latency regression (+14-25%) disappears (fp16 model is ~14MB vs 28MB fp32,
     less memory-bandwidth pressure on shared CPU/GPU resources)
  4. Model file size: ~14 MB (same order as original fp16, vs 28 MB for fp32 fix)

# ============================================================
# USAGE
# ============================================================

    cd /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2
    source "/Users/jiansun/Documents/Doctor Courses/4455/env1/bin/activate"
    python3 shared/export_encoder_fp16_milfix.py

OUTPUT:
    models/MobileSAM_ImageEncoder_fp16_milfix.mlpackage  (~14 MB, fp16 weights)

XCODE INTEGRATION (after script succeeds):
    1. Drag models/MobileSAM_ImageEncoder_fp16_milfix.mlpackage into
       JudgeE2/Segmentation/Models/ in Xcode (tick target membership).
    2. SAMEncoder.swift already updated to load milfix first (priority:
       milfix > fp32 > fp16 original).
    3. Clean Build (⌘⇧K) → Build → run on device.
    4. Verify in console: "SAMEncoder loading model: MobileSAM_ImageEncoder_fp16_milfix"
    5. Verify ANE warning absent: "Invalid input tensor channel 1 ..." should NOT appear.

NOTE: Do NOT delete MobileSAM_ImageEncoder.mlpackage or _fp32.mlpackage;
      SAMEncoder.swift uses them as fallback in case milfix is absent.
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
OUT_NAME = "MobileSAM_ImageEncoder_fp16_milfix"

sys.path.insert(0, str(MOBILESAM_ROOT))

# ---------------------------------------------------------------------------
# Patch LayerNorm2d BEFORE importing the model (critical: must happen first)
#
# There are TWO LayerNorm2d classes in MobileSAM:
#   1. mobile_sam.modeling.common.LayerNorm2d  — used by transformer blocks
#      (coremltools DEFAULT pipeline already fuses these →20 layer_norm ops)
#   2. mobile_sam.modeling.tiny_vit_sam.LayerNorm2d  — used by the NECK
#      (NOT fused → exposes 4 C=1 reduce_mean Float16 tensors → ANE warning)
#
# We must patch BOTH, but only tiny_vit_sam.LayerNorm2d is strictly necessary.
# ---------------------------------------------------------------------------
from mobile_sam.modeling.common import LayerNorm2d  # noqa: E402


def _layernorm2d_forward_milfix(self, x: torch.Tensor) -> torch.Tensor:
    """
    ANE-friendly replacement for LayerNorm2d.forward.

    Strategy:
      Transpose [B, C, H, W] → [B, H, W, C], apply F.layer_norm over the
      last dimension (C channels), transpose back.

    Why this works in CoreML / ANE:
      - torch.nn.functional.layer_norm is mapped by coremltools to the MIL
        `layer_norm` op (axes=[-1]).  This is a single fused kernel.
      - The MIL `layer_norm` op does NOT expose any [B, 1, H, W] Float16
        intermediate tensor to the ANE scheduler, eliminating the C=1
        "must be aligned on 64 bytes" warning.
      - Weight/bias shapes [C] match normalized_shape=[C], identical to
        the original LayerNorm2d semantics.

    Mathematical equivalence to original LayerNorm2d:
      For every position (b, c, h, w):
        output[b,c,h,w] = weight[c] * (x[b,c,h,w] - μ) / σ + bias[c]
      where μ and σ are computed over the C channels at position (b, h, w).
      This is exactly what F.layer_norm([C]) computes on x_perm[b, h, w, :].
    """
    # x: [B, C, H, W]
    x = x.permute(0, 2, 3, 1)                              # [B, H, W, C]
    x = F.layer_norm(x, [x.shape[-1]], self.weight, self.bias, self.eps)
    return x.permute(0, 3, 1, 2)                           # [B, C, H, W]


# Patch the common.LayerNorm2d (used by transformer blocks)
LayerNorm2d.forward = _layernorm2d_forward_milfix
print("[milfix] common.LayerNorm2d.forward patched")

# CRITICAL: The encoder NECK uses a SEPARATE LayerNorm2d defined in
# tiny_vit_sam.py — this is the one producing the C=1 ANE-alignment warnings.
# Must also patch this class.
from mobile_sam.modeling.tiny_vit_sam import LayerNorm2d as NeckLayerNorm2d  # noqa: E402
NeckLayerNorm2d.forward = _layernorm2d_forward_milfix
print("[milfix] tiny_vit_sam.LayerNorm2d.forward patched (neck — this is the C=1 culprit)")

from mobile_sam import sam_model_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Encoder wrapper (identical to fp32 script, kept for I/O compatibility)
# ---------------------------------------------------------------------------
class ImageEncoderWrapper(nn.Module):
    """
    Thin wrapper that fuses SAM's pixel normalisation into the traced graph.
    Input:  (1, 3, 1024, 1024) Float32, RGB, values in [0, 255]
    Output: (1, 256, 64, 64)  image embeddings
    """
    def __init__(self, sam: nn.Module):
        super().__init__()
        self.image_encoder = sam.image_encoder
        self.register_buffer("pixel_mean", sam.pixel_mean, persistent=False)
        self.register_buffer("pixel_std",  sam.pixel_std,  persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # SAM normalisation: (image - mean) / std
        x = (image - self.pixel_mean) / self.pixel_std
        return self.image_encoder(x)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def _count_ops(mlmodel_path: Path) -> dict:
    """Return a dict with op-type counts for quick sanity checks."""
    from coremltools.converters.mil import load as mil_load
    prog = mil_load(str(mlmodel_path))
    counts: dict[str, int] = {}
    for func in prog.functions.values():
        for op in func.operations:
            counts[op.op_type] = counts.get(op.op_type, 0) + 1
    return counts


def _verify_io(mlmodel_path: Path) -> None:
    """Print I/O tensor names, shapes, and dtypes."""
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

    # ---- Load model --------------------------------------------------------
    print("Loading MobileSAM vit_t checkpoint (with patched LayerNorm2d)...")
    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
    sam.eval()

    encoder = ImageEncoderWrapper(sam).eval()

    # Quick sanity: run one forward pass to confirm patched model works
    dummy = torch.zeros(1, 3, 1024, 1024, dtype=torch.float32)
    with torch.no_grad():
        out = encoder(dummy)
    assert out.shape == (1, 256, 64, 64), f"Unexpected output shape: {out.shape}"
    print(f"[sanity] Forward pass OK — output shape {tuple(out.shape)}")

    # ---- Trace -------------------------------------------------------------
    print("Tracing encoder with torch.jit.trace ...")
    dummy_trace = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(encoder, (dummy_trace,))
    print("Tracing done.")

    # ---- Convert to CoreML (fp16) -----------------------------------------
    # compute_precision=FLOAT16 keeps weights and activations in fp16.
    # The patched LayerNorm2d traces to F.layer_norm which coremltools maps
    # directly to the MIL `layer_norm` op — no C=1 reduce_mean intermediates.
    print("Converting to CoreML mlprogram (compute_precision=FLOAT16) ...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="image",
                              shape=dummy_trace.shape,
                              dtype=np.float32)],
        outputs=[ct.TensorType(name="image_embeddings",
                               dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,   # Keep fp16 for ANE efficiency
        minimum_deployment_target=ct.target.iOS16,
    )

    # ---- Save --------------------------------------------------------------
    out_path = OUT_DIR / f"{OUT_NAME}.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    print(f"\nSaved: {out_path}")

    # ---- Verification ------------------------------------------------------
    _verify_io(out_path)

    # Check op counts: look for reduce_mean with small shapes (C=1 signal)
    # and confirm layer_norm ops are present (fusion signal)
    try:
        op_counts = _count_ops(out_path)
        layernorm_count = op_counts.get("layer_norm", 0)
        reduce_mean_count = op_counts.get("reduce_mean", 0)
        fp16_named_count = sum(v for k, v in op_counts.items() if "_fp16" in k)

        print(f"\n--- Op Counts (key ops) ---")
        print(f"  layer_norm ops  : {layernorm_count}  "
              f"({'✅ >0 means fusion worked' if layernorm_count > 0 else '⚠️  0 — check patch'})")
        print(f"  reduce_mean ops : {reduce_mean_count}  "
              f"({'may still exist outside LayerNorm2d' if reduce_mean_count > 0 else '✅ none'})")
        print(f"  _fp16 named ops : {fp16_named_count}  "
              f"({'✅ 0 — no explicit fp16 casts' if fp16_named_count == 0 else '⚠️  check for fp16 cast ops'})")
    except Exception as e:
        print(f"[warn] Could not count ops (non-critical): {e}")

    # File size comparison
    orig_path = OUT_DIR / "MobileSAM_ImageEncoder.mlpackage"
    fp32_path = OUT_DIR / "MobileSAM_ImageEncoder_fp32.mlpackage"
    milfix_size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6

    print(f"\n--- File Size Comparison ---")
    if orig_path.exists():
        orig_size_mb = sum(f.stat().st_size for f in orig_path.rglob("*") if f.is_file()) / 1e6
        print(f"  Original fp16  : {orig_size_mb:.1f} MB")
    if fp32_path.exists():
        fp32_size_mb = sum(f.stat().st_size for f in fp32_path.rglob("*") if f.is_file()) / 1e6
        print(f"  fp32 (old fix) : {fp32_size_mb:.1f} MB")
    print(f"  fp16_milfix    : {milfix_size_mb:.1f} MB  ← this model")

    print(f"""
=======================================================
Export complete: {out_path.name}
=======================================================

Next steps (Xcode integration):
  1. Drag {out_path} into Xcode
     → JudgeE2/Segmentation/Models/  (tick target membership)
  2. SAMEncoder.swift already updated to prefer milfix.
  3. Clean Build (⌘⇧K) → Build.
  4. Run on device — verify in console:
       SAMEncoder loading model: MobileSAM_ImageEncoder_fp16_milfix
  5. Confirm NO "Invalid input tensor channel 1 ..." warnings.
  6. Debugger: measure encoder latency and compare to Phase 2 baseline (857ms).
""")


if __name__ == "__main__":
    main()
