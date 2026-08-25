# Day 5 — Quantization & Format Route v1 (cold start + ANE path)

Owner: **ML_Vision**  
Target device: **iPhone 11 (A13)**  
Focus: reduce **cold start**, improve chance of **ANE execution**, keep correctness stable.

---

## 0) Current artifacts (baseline)

### YOLOv9
- `models/yolov9-c.mlmodel` (CoreML **neuralnetwork** format)
- Input: ImageType 640×640 RGB with embedded scale=1/255
- Output: `var_3019` float32 (1,84,8400) is the authoritative head

### MobileSAM (split, mlprogram)
- `models/MobileSAM_ImageEncoder.mlpackage`
- `models/MobileSAM_PromptMaskDecoder.mlpackage`
- Both are **mlprogram + MultiArray** (see `shared/model_plan.md` §2.3)

---

## 1) YOLOv9 — recommended format/precision route

### 1.1 Primary recommendation: migrate YOLO to **mlprogram (.mlpackage)**
Rationale:
- **Cold start**: mlprogram models often compile/load more predictably on iOS 16/17 and can unlock better graph optimizations.
- **ANE path**: mlprogram is generally the preferred modern path for ANE eligibility.

Plan:
1) Export PyTorch → (optional ONNX for debug) → coremltools convert **to mlprogram**
2) Use **FP16 compute precision** in conversion (or FP16 weights where supported)

Notes:
- Keep output as float32 if needed for decode stability; compute can still be FP16.
- If mlprogram conversion fails due to unsupported ops, keep neuralnetwork as fallback for bring-up.

### 1.2 Quantization guidance (v1)
- **FP16** is the default first step (low risk, usually big perf win on Apple accelerators).
- **INT8**:
  - Defer until we have a small calibration set and a stable decode/NMS baseline.
  - INT8 can cause accuracy drops and/or export friction; prioritize shipping a stable pipeline first.

### 1.3 Cold-start reductions (iOS-side knobs, non-UI)
(These are integration notes, not architecture changes.)
- Prefer bundling compiled model (`.mlmodelc`) produced by Xcode build step.
- Warm-up: run 1 dummy inference off the main thread after app launch.
- Ensure `MLModel` is cached singleton; do not reload per frame.

---

## 2) MobileSAM — precision + computeUnits recommendations

### 2.1 Precision
Current conversion uses `compute_precision = FLOAT16` (good default).

Recommendations:
- Keep internal compute as **FP16**.
- Keep I/O as **float32 MultiArray** for simplicity and to match current artifacts/spec.
  - If we later want to reduce memory bandwidth: consider changing encoder output embedding to float16, but only after we have stable correctness tests.

### 2.2 computeUnits policy (iPhone 11)
MobileSAM contains transformer-ish ops and custom layers; ANE support can be fragile.

Recommended rollout:
1) **Bring-up default**: `.cpuAndGPU` (most stable)
2) After correctness is confirmed on device: test `.all`
   - If `.all` fails or outputs are unstable, keep `.cpuAndGPU` for v1.

### 2.3 Scheduling implication (ties to Day5 Builder tasks)
- The split design is only worth it if encoder cadence is throttled.
- Suggested cadence starting point:
  - encoder every **N=12~20 frames**
  - decoder every frame or every top-1 update (depending on HUD/perf)

---

## 3) Toolchain risk note (important)

In the export venv we used:
- coremltools == **9.0**
- torch == **2.9.1**

coremltools prints a warning that torch 2.9.1 is untested (torch 2.7.0 most recently tested).

Recommendation:
- For future conversion work (especially YOLO→mlprogram, quantization): consider a dedicated export env pinned to **torch==2.7.x** for better compatibility.
- This does **not** block using the already-generated MobileSAM mlpackages.

---

## 4) Acceptance criteria for Day5 v1

- We can articulate a stable path:
  - YOLO: neuralnetwork (current) → mlprogram (next) with FP16
  - MobileSAM: keep split mlprogram; start `.cpuAndGPU` then attempt `.all`
- Builder/Debugger can run the golden case and collect device timing.
