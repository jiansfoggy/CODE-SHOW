"""
Phase 3 — Multimask Prompt/Mask Decoder Export (Tap-to-Segment ambiguity fix)

WHY
===
The deployed MobileSAM_PromptMaskDecoder.mlpackage was exported with a single
mask output (low_res_masks [1,1,256,256], iou_predictions [1,1]).  A single tap
is an inherently ambiguous prompt (part / object / scene); SAM's design answer
is multimask_output=True: 3 candidate masks at different granularities plus a
predicted IoU for each, with the *caller* choosing.  Forcing single-mask output
makes ambiguous taps collapse to an oversized blend — observed on device as a
full-screen mask when tapping large objects, while small distinct objects
segment fine.

WHAT
====
Exports MobileSAM_PromptMaskDecoder_multi.mlpackage with identical inputs to
the deployed model:

    image_embeddings [1, 256, 64, 64]
    point_coords     [1, 2, 2]     (SAM 1024-pixel space; normalized inside)
    point_labels     [1, 2]
    mask_input       [1, 1, 256, 256]
    has_mask_input   [1]

and multimask outputs:

    low_res_masks    [1, 3, 256, 256]   (mask tokens 1..3)
    iou_predictions  [1, 3]

The point-embedding math is copied from mobile_sam/utils/onnx.py
(SamOnnxModel._embed_points/_embed_masks) — the official trace-friendly form —
so the app's existing PointPromptBuilder tensors work unchanged.

RUN
===
/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/.venv_export/bin/python \
    shared/export_decoder_multimask.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent          # APP/JudgeE2
MOBILE_SAM_REPO = ROOT / "models" / "MobileSAM"
CHECKPOINT = ROOT / "models" / "mobile_sam.pt"
OUT_PACKAGE = ROOT / "JudgeE2" / "Segmentation" / "Models" / "MobileSAM_PromptMaskDecoder_multi.mlpackage"

sys.path.insert(0, str(MOBILE_SAM_REPO))

import torch.nn.functional as F  # noqa: E402

# ---------------------------------------------------------------------------
# LayerNorm2d milfix (same as export_encoder_fp16_milfix.py): the decoder's
# output_upscaling uses common.LayerNorm2d, whose C=1 fp16 intermediates hit
# the ANE 64-byte alignment bug on A13 ("Invalid input tensor channel 1"),
# producing garbage-magnitude logits on device (observed |logit| ≈ 1e7–1e8,
# iou_pred > 1.0, half-frame degenerate masks).  Patch BEFORE model build.
# ---------------------------------------------------------------------------
from mobile_sam.modeling.common import LayerNorm2d  # noqa: E402


def _layernorm2d_forward_milfix(self, x):
    x = x.permute(0, 2, 3, 1)
    x = F.layer_norm(x, [x.shape[-1]], self.weight, self.bias, self.eps)
    return x.permute(0, 3, 1, 2)


LayerNorm2d.forward = _layernorm2d_forward_milfix
print("[milfix] common.LayerNorm2d.forward patched")

from mobile_sam import sam_model_registry  # noqa: E402
from mobile_sam.modeling.transformer import Attention  # noqa: E402

import coremltools as ct  # noqa: E402


def _patch_attention_static_shapes():
    """Replace shape-reading reshapes in Attention with static equivalents.

    The originals do `b, n, c = x.shape` and reshape with those values, which
    traces to aten::Int nodes that coremltools (8.3 + torch 2.5) fails to
    convert.  Batch is always 1 here and channel dims are module constants, so
    -1 reshapes are exactly equivalent.
    """

    def _separate_heads(self, x, num_heads):
        x = x.reshape(1, -1, num_heads, self.internal_dim // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x):
        x = x.transpose(1, 2)
        return x.reshape(1, -1, self.internal_dim)

    Attention._separate_heads = _separate_heads
    Attention._recombine_heads = _recombine_heads


class MultimaskDecoder(nn.Module):
    """Prompt encoder + mask decoder, multimask output, trace-friendly.

    predict_masks is re-implemented here with batch fixed at 1: the original
    uses repeat_interleave(tokens.shape[0]) and dynamic views, which trace to
    aten::Int on symbolic shapes and break coremltools conversion.  The dense
    positional encoding is precomputed as a buffer for the same reason.
    """

    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.img_size = sam.image_encoder.img_size  # 1024
        self.register_buffer("dense_pe", sam.prompt_encoder.get_dense_pe())  # [1,256,64,64]

    def _embed_points(self, point_coords, point_labels):
        # Copied from SamOnnxModel._embed_points (mobile_sam/utils/onnx.py).
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.sam.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.sam.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )
        for i in range(self.sam.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.sam.prompt_encoder.point_embeddings[i].weight * (
                point_labels == i
            )
        return point_embedding

    def _embed_masks(self, input_mask, has_mask_input):
        # Copied from SamOnnxModel._embed_masks.
        mask_embedding = has_mask_input * self.sam.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

    def _predict_masks_b1(self, image_embeddings, sparse, dense):
        # Batch-1 re-implementation of MaskDecoder.predict_masks (vit_t dims:
        # transformer_dim=256, feat 64×64, upscaled 32ch 256×256).
        md = self.sam.mask_decoder
        output_tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], dim=0).unsqueeze(0)
        tokens = torch.cat((output_tokens, sparse), dim=1)

        src = image_embeddings + dense
        hs, src2 = md.transformer(src, self.dense_pe, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : 1 + md.num_mask_tokens, :]

        src2 = src2.transpose(1, 2).view(1, 256, 64, 64)
        upscaled = md.output_upscaling(src2)                       # [1,32,256,256]
        hyper_in = torch.stack(
            [md.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(md.num_mask_tokens)],
            dim=1,
        )                                                          # [1,4,32]
        masks = (hyper_in @ upscaled.view(1, 32, 256 * 256)).view(1, -1, 256, 256)
        iou_pred = md.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    @torch.no_grad()
    def forward(self, image_embeddings, point_coords, point_labels, mask_input, has_mask_input):
        sparse = self._embed_points(point_coords, point_labels)
        dense = self._embed_masks(mask_input, has_mask_input)
        masks, scores = self._predict_masks_b1(image_embeddings, sparse, dense)
        # Token 0 is the single-mask token; tokens 1..3 are the multimask
        # candidates (sub-part / part / whole granularity).
        return masks[:, 1:4], scores[:, 1:4]


def main():
    print(f"Loading checkpoint {CHECKPOINT}")
    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
    sam.eval()

    wrapper = MultimaskDecoder(sam).eval()

    example = (
        torch.randn(1, 256, 64, 64, dtype=torch.float32),
        torch.tensor([[[512.0, 512.0], [0.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        torch.zeros(1, 1, 256, 256, dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
    )

    _patch_attention_static_shapes()

    print("Tracing…")
    traced = torch.jit.trace(wrapper, example)

    print("Converting to Core ML (fp16)…")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="image_embeddings", shape=(1, 256, 64, 64)),
            ct.TensorType(name="point_coords", shape=(1, 2, 2)),
            ct.TensorType(name="point_labels", shape=(1, 2)),
            ct.TensorType(name="mask_input", shape=(1, 1, 256, 256)),
            ct.TensorType(name="has_mask_input", shape=(1,)),
        ],
        outputs=[
            ct.TensorType(name="low_res_masks"),
            ct.TensorType(name="iou_predictions"),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        # iOS15 matches the deployed single-mask decoder (spec v6) — the same
        # ANE compiler path that is known to work on this device.
        minimum_deployment_target=ct.target.iOS15,
    )

    OUT_PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUT_PACKAGE))
    print(f"Saved {OUT_PACKAGE}")

    # ---- Sanity check: CoreML vs torch on the same random inputs ----
    print("Sanity check (CoreML vs torch)…")
    torch_masks, torch_scores = wrapper(*example)
    pred = mlmodel.predict({
        "image_embeddings": example[0].numpy(),
        "point_coords": example[1].numpy(),
        "point_labels": example[2].numpy(),
        "mask_input": example[3].numpy(),
        "has_mask_input": example[4].numpy(),
    })
    cm_masks = pred["low_res_masks"]
    cm_scores = pred["iou_predictions"]
    print(f"  low_res_masks shape:   {cm_masks.shape}  (expect (1, 3, 256, 256))")
    print(f"  iou_predictions shape: {cm_scores.shape}  (expect (1, 3))")
    mask_err = np.abs(cm_masks - torch_masks.numpy()).mean()
    score_err = np.abs(cm_scores - torch_scores.numpy()).max()
    print(f"  mean |mask logit err| = {mask_err:.4f}  (fp16, expect < 0.1)")
    print(f"  max  |iou err|        = {score_err:.4f}  (fp16, expect < 0.05)")
    areas = (cm_masks[0] > 0).reshape(3, -1).sum(axis=1)
    print(f"  candidate areas (logit>0 px): {areas.tolist()} | ious: {cm_scores[0].tolist()}")


if __name__ == "__main__":
    main()
