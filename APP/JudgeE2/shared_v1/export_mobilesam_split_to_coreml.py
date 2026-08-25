"""Export MobileSAM (vit_t) to CoreML as two artifacts:
  1) MobileSAM_ImageEncoder.mlpackage
  2) MobileSAM_PromptMaskDecoder.mlpackage

Designed for iOS usage:
- Run encoder on 1024x1024 RGB image (float32, 0..255). Wrapper applies SAM mean/std.
- Run decoder on cached image_embeddings and box prompt encoded as 2 points.

This script targets stable, static shapes (num_points=2).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct


REPO_ROOT = Path(__file__).resolve().parents[1]  # .../JudgeEverything
MOBILESAM_ROOT = REPO_ROOT / "models" / "MobileSAM"
CHECKPOINT = REPO_ROOT / "models" / "mobile_sam.pt"
OUT_DIR = REPO_ROOT / "models"

# Make `import mobile_sam` work
sys.path.insert(0, str(MOBILESAM_ROOT))

from mobile_sam import sam_model_registry  # noqa: E402


class ImageEncoderWrapper(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.image_encoder = sam.image_encoder
        self.register_buffer("pixel_mean", sam.pixel_mean, persistent=False)
        self.register_buffer("pixel_std", sam.pixel_std, persistent=False)

    def forward(self, image: torch.Tensor):
        # image: (1,3,1024,1024), RGB, float32, 0..255
        x = (image - self.pixel_mean) / self.pixel_std
        emb = self.image_encoder(x)
        return emb


class PromptMaskDecoderWrapper(nn.Module):
    def __init__(self, sam, return_single_mask: bool = True):
        super().__init__()
        self.model = sam
        self.mask_decoder = sam.mask_decoder
        self.img_size = sam.image_encoder.img_size
        self.return_single_mask = return_single_mask

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        # point_coords: (1,2,2) in [0, 1023] input image coords
        # point_labels: (1,2) with labels 2/3 for box corners (SAM convention)
        point_coords = point_coords + 0.5
        point_coords = point_coords / float(self.img_size)
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[i].weight * (
                point_labels == i
            )
        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        # has_mask_input: (1,) float32 0/1
        has_mask_input = has_mask_input.reshape(1, 1, 1, 1)
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (1.0 - has_mask_input) * self.model.prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        )
        return mask_embedding

    def _select_mask(self, masks: torch.Tensor, iou_preds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # masks: (1,3,256,256), iou_preds: (1,3)
        # Prefer index 0 for 2-point box prompts (works well in practice for SAM export)
        best_idx = torch.zeros((masks.shape[0],), dtype=torch.long, device=masks.device)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)
        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        low_res_masks, iou_predictions = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.return_single_mask:
            low_res_masks, iou_predictions = self._select_mask(low_res_masks, iou_predictions)

        return low_res_masks, iou_predictions


def main():
    assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MobileSAM vit_t...")
    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
    sam.eval()

    encoder = ImageEncoderWrapper(sam).eval()
    decoder = PromptMaskDecoderWrapper(sam, return_single_mask=True).eval()

    # ---- Trace encoder ----
    dummy_image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    traced_encoder = torch.jit.trace(encoder, (dummy_image,))

    # ---- Trace decoder (static num_points=2) ----
    dummy_emb = torch.randn(1, 256, 64, 64, dtype=torch.float32)
    dummy_point_coords = torch.tensor([[[0.0, 0.0], [1023.0, 1023.0]]], dtype=torch.float32)
    dummy_point_labels = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    dummy_mask_input = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
    dummy_has_mask = torch.zeros(1, dtype=torch.float32)
    traced_decoder = torch.jit.trace(
        decoder,
        (dummy_emb, dummy_point_coords, dummy_point_labels, dummy_mask_input, dummy_has_mask),
    )

    # ---- Convert to CoreML ----
    print("Converting ImageEncoder to CoreML...")
    encoder_mlmodel = ct.convert(
        traced_encoder,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="image", shape=dummy_image.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="image_embeddings", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
    )

    out_encoder = OUT_DIR / "MobileSAM_ImageEncoder.mlpackage"
    if out_encoder.exists():
        # Overwrite
        import shutil

        shutil.rmtree(out_encoder)
    encoder_mlmodel.save(str(out_encoder))
    print(f"Wrote {out_encoder}")

    print("Converting Prompt+MaskDecoder to CoreML...")
    decoder_mlmodel = ct.convert(
        traced_decoder,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="image_embeddings", shape=dummy_emb.shape, dtype=np.float32),
            ct.TensorType(name="point_coords", shape=dummy_point_coords.shape, dtype=np.float32),
            ct.TensorType(name="point_labels", shape=dummy_point_labels.shape, dtype=np.float32),
            ct.TensorType(name="mask_input", shape=dummy_mask_input.shape, dtype=np.float32),
            ct.TensorType(name="has_mask_input", shape=dummy_has_mask.shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="low_res_masks", dtype=np.float32),
            ct.TensorType(name="iou_predictions", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT16,
    )

    out_decoder = OUT_DIR / "MobileSAM_PromptMaskDecoder.mlpackage"
    if out_decoder.exists():
        import shutil

        shutil.rmtree(out_decoder)
    decoder_mlmodel.save(str(out_decoder))
    print(f"Wrote {out_decoder}")


if __name__ == "__main__":
    main()
