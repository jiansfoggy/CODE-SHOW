# Model Plan — Day 2 (CoreML Model Load)

## Selected Model
- **File:** `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/yolov9-c.mlmodel`
- **Model class (CoreML):** `yolov9_c`
- **Model type:** NeuralNetwork
- **Compute precision:** Float16 (weights), outputs Float32

## Inputs
- **Name:** `image`
- **Type:** Image (Color)
- **Size:** 640 × 640
- **Pixel format (from generated interface):** CVPixelBuffer, kCVPixelFormatType_32BGRA

## Outputs
- **Name:** `var_3019`
  - **Type:** MultiArray (Float32)
  - **Shape:** (1, 84, 8400)
- **Name:** `var_3022`
  - **Type:** MultiArray (Float32)
  - **Shape:** (1, 84, 8400)

> Shape source: compiled model `model.espresso.shape` (rank=3 with n=1, h=84, w=8400). Both outputs match this shape.

## Minimum OS Requirement
- **iOS 13.0** (from compiled model availability metadata)

---

# Phase 2 — Day 2 (MobileSAM CoreML)

## Models (CoreML, split encoder/decoder)
- **Image Encoder:** `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/MobileSAM_ImageEncoder.mlpackage`
- **Prompt+Mask Decoder:** `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/MobileSAM_PromptMaskDecoder.mlpackage`

## Encoder I/O
- **Input**
  - Name: `image`
  - Type: MultiArray (Float32)
  - Shape: **(1, 3, 1024, 1024)**
- **Output**
  - Name: `image_embeddings`
  - Type: MultiArray (Float32)
  - Shape: **(1, 256, 64, 64)**

## Decoder I/O
- **Inputs**
  - `image_embeddings`: (1, 256, 64, 64) Float32
  - `point_coords`: (1, 2, 2) Float32
  - `point_labels`: (1, 2) Float32
  - `mask_input`: (1, 1, 256, 256) Float32
  - `has_mask_input`: (1) Float32
- **Outputs**
  - `low_res_masks`: (1, 1, 256, 256) Float32 (logits)
  - `iou_predictions`: (1, 1) Float32

## Prompt Spec (Box Prompt)
- Use **2 points** for a box: `[[x1,y1],[x2,y2]]`
- `point_labels` should be **[2, 3]** for the box corners (SAM convention)
- Coordinates are **pixel space** in the resized input frame (after ResizeLongestSide)
- `mask_input`: zeros; `has_mask_input`: 0 for first-pass decode

## Preprocessing (from MobileSAM repo)
1. Input image is **RGB**, uint8, H×W, values in **[0,255]**
2. Resize with **ResizeLongestSide(1024)**
3. Convert to float, **CHW**, add batch → (1,3,H,W)
4. Normalize: **(x - mean) / std**
   - mean = **[123.675, 116.28, 103.53]**
   - std = **[58.395, 57.12, 57.375]**
5. Pad to **1024×1024** (bottom/right)

## Postprocess (mask)
- `low_res_masks` are logits at 256×256
- Upsample to 1024×1024, crop to input size, then resize back to original image size
- Threshold at 0 (per MobileSAM `mask_threshold = 0.0`)

## Latency (iPhone 11)
- **YOLOv9-c (CoreML, single-frame, Infer-only):**
  - **CPU-only (rawValue: 0):** mean **1106.23 ms**, p50 **1083.72 ms**, p95 **1325.19 ms** (n=23)
  - **CPU+GPU (rawValue: 1):** mean **1064.34 ms**, p50 **1033.67 ms**, p95 **1360.37 ms** (n=29)
  - **CPU+NeuralEngine (rawValue: 3):** mean **193.67 ms**, p50 **178.18 ms**, p95 **198.20 ms** (n=58)

- **MobileSAM Encoder (CPU/GPU):** _待设备测_
- **MobileSAM Decoder (CPU/GPU):** _待设备测_

# Day 5 — Decode + NMS (ML_Vision)

## Output Tensor Interpretation (YOLOv9-c, CoreML)
- Output shape: **(1, 84, 8400)**
- Interpret as **84 channels × 8400 locations**
- **84 = 4 box + 80 class scores** (no separate objectness channel)

### Channel Layout (per location index i in 0..8399)
- `0`: **cx** (center x)
- `1`: **cy** (center y)
- `2`: **w** (width)
- `3`: **h** (height)
- `4..83`: **class scores** (80 classes)

> This matches YOLOv8/v9 detection head convention for 80-class COCO models.

## Decode Logic (Reference)
1. For each location `i`:
   - Read `cx, cy, w, h` from channels 0..3
   - Find `bestClass = argmax(classScores)` and `bestScore = max(classScores)`
2. **Confidence** = `bestScore` (no objectness multiplier)
3. Filter by confidence threshold
4. Convert box to **xyxy** for NMS:
   - `x1 = cx - w/2`, `y1 = cy - h/2`, `x2 = cx + w/2`, `y2 = cy + h/2`
5. Run class-aware NMS (preferred) or class-agnostic NMS

## Coordinate Assumption
- Boxes are in **input image coordinates** for the **640×640** model input.
- If input is letterboxed, map from 640×640 to original frame using the same scale + padding applied during preprocessing.

## Recommended Thresholds (Starting Point)
- **Confidence threshold:** `0.25`
- **NMS IoU threshold:** `0.45`
- Adjust as needed once you see detection density.

## Notes
- Two outputs exist but both are `(1, 84, 8400)`; treat either output as the detection tensor unless builder confirms a preferred one.
- If detections seem too many or too few, verify if the model was trained with a different class count or requires sigmoid/softmax; generally YOLOv8/v9 uses sigmoid on class scores.
