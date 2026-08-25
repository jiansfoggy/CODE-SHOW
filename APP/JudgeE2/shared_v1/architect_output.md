# Architect Output — Architecture Specification (Day 1)

Project: **JudgeE2 / JudgeEverything** — real-time iOS video **instance segmentation** (rebuild)

Target device: **iPhone 11 (A13)**
- Primary accelerators: **Apple Neural Engine (ANE)** via Core ML when compatible; fallback **GPU/CPU**.

Models
- Detection: **YOLO-v9**
- Segmentation: **MobileSAM** (prompted segmentation)

Python env (for export/conversion only):
- `/Users/jiansun/Documents/Doctor Courses/4455/env1`

Weights:
- `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models`

---

## 0. Goals / Non-Goals

### Goals
- **On-device**, real-time, camera-driven pipeline: **Camera → Detect → Segment → Overlay**.
- Modular architecture to swap:
  - detector backend (Core ML vs other)
  - segmenter backend (Core ML vs fallback)
  - prompt strategy (boxes / points / mask priors)
- Measurable performance targets (see §6).

### Non-goals (explicit Day 1 boundaries)
- No app UI polish; no training pipeline; no cloud inference.
- No implementation details/code in this document.

---

## 1. End-to-End System Overview (Conceptual)

The app runs a **two-stage pipeline** per frame:
1. **Detector** (YOLO-v9) produces candidate objects (class + score + bbox).
2. **Segmenter** (MobileSAM) uses **bbox prompts** (optionally refined with points) to produce an **instance mask per detection**.

A lightweight **Tracking / Temporal Stabilization** layer reduces flicker and avoids running segmentation for every detection on every frame.

### Key architectural constraints
- **Memory bandwidth and ANE scheduling** on iPhone 11 are limiting; segmentation must be throttled.
- Camera frames arrive faster than we can segment everything; pipeline must be **asynchronous** and **drop frames** safely.

---

## 2. Module Decomposition (Responsibilities + Interfaces)

### 2.1 iOS App Modules

#### A) CameraCapture
- Responsibility: capture frames from camera at a configured resolution and deliver pixel buffers with timestamps.
- Inputs: camera configuration (fps target, format, resolution).
- Outputs:
  - `CVPixelBuffer` + `CMTime` timestamp
  - camera intrinsics/metadata if needed later

#### B) Preprocess
- Responsibility: resize/letterbox/normalize frames for the detector & segmenter.
- Inputs: `CVPixelBuffer` (camera frame)
- Outputs:
  - `CVPixelBuffer` or `MLMultiArray` shaped to model inputs
  - transform metadata (scale, padding) for mapping outputs back to original frame coordinates

#### C) DetectorEngine (YOLO-v9)
- Responsibility: run object detection; output bboxes in normalized or pixel coords.
- Inputs:
  - preprocessed frame tensor/buffer
- Outputs (per frame):
  - `Detection[]` where `Detection = {classId, score, bbox (x,y,w,h in original frame coords), frameTime}`
- Notes:
  - should include NMS (either inside model graph or post-process)

#### D) PromptBuilder
- Responsibility: translate detections into segmentation prompts.
- Inputs: `Detection[]` + mapping metadata
- Outputs:
  - `Prompt[]` where `Prompt = {instanceId, bbox, optional points, frameTime}`
- Policy knobs:
  - max instances per frame (top-K by score)
  - class allowlist/denylist
  - box expansion factor (helps SAM capture full object)

#### E) SegmenterEngine (MobileSAM)
- Responsibility: produce instance masks from image + prompt.
- Inputs:
  - image embedding input (either precomputed embedding or raw image depending on export)
  - `Prompt` (bbox / points)
- Outputs:
  - `MaskPrediction = {instanceId, mask (binary/float), maskConfidence, frameTime}`

#### F) TemporalManager (Tracking + Cache)
- Responsibility:
  - maintain association of instances across frames
  - decide **when** to run segmentation
  - cache last good mask per instance and warp/propagate when needed
- Inputs:
  - detections stream
  - mask predictions stream
  - frame timestamps
- Outputs:
  - `ActiveInstance[]` each with stabilized bbox + mask + label

#### G) Renderer / Overlay
- Responsibility: composite masks + labels onto preview.
- Inputs:
  - `ActiveInstance[]` + current frame
- Outputs:
  - rendered overlay (Metal/CoreAnimation)

#### H) Telemetry / Profiler
- Responsibility: measure latency per stage, dropped frames, ANE/GPU usage.
- Outputs:
  - per-stage timings
  - rolling FPS and end-to-end latency

---

## 3. iOS Data Flow (Camera → Vision/CoreML → Overlay)

### 3.1 Primary runtime pipeline (asynchronous)

1. **CameraCapture** produces frames at camera FPS (e.g., 30fps).
2. **FrameBroker** (conceptual component) enqueues frames and applies backpressure:
   - detector runs on most recent frame (drop older frames)
   - segmenter runs at a lower rate and per-instance budget
3. **DetectorEngine** runs on a dedicated inference queue.
4. **PromptBuilder** selects top-K detections and constructs prompts.
5. **SegmenterEngine** runs prompts (potentially batched) under a strict budget.
6. **TemporalManager** merges latest detections + cached masks to produce stable instances.
7. **Renderer** overlays results onto preview.

### 3.2 Vision vs Core ML integration
- Preferred approach:
  - Use **Core ML** models directly for performance.
  - Use **Vision** request wrappers when they improve convenience (pre/post processing) *without* harming latency.
- Constraint:
  - Vision request scheduling can add overhead; for maximum control use direct `MLModel` execution.

### 3.3 Frame dropping policy (required for real-time)
- Detection: always run on **latest frame**; if detector lags, skip frames.
- Segmentation: run on **selected instances only**, and optionally only every N frames per instance.

---

## 4. Python ⇆ iOS “Edge Inference” Design

### 4.1 Separation of concerns
- **Python side** is responsible for:
  - model graph cleanup and export to mobile formats
  - quantization decisions and calibration dataset preparation
  - offline verification (numerical + qualitative)
- **iOS side** is responsible for:
  - real-time capture + preprocessing matching the export
  - scheduling, caching, and UI overlay
  - device-specific performance tuning

### 4.2 Artifact contract (handoff from Python to iOS)
For each model artifact, Python export produces:
- Core ML package(s): `*.mlpackage` (preferred) or `*.mlmodel`
- A **Model Card** (human-readable) containing:
  - input tensor spec (shape, dtype, normalization)
  - output tensor spec and decoding rules
  - expected class list / label mapping
  - version + git hash of export scripts
- A **Golden Test Set**:
  - a small set of images + expected output summaries (not full tensors) for regression.

### 4.3 Runtime compatibility matrix
- If a layer/op is unsupported on ANE:
  - fallback to GPU/CPU via Core ML
  - or restructure export (preferred)
- The architecture assumes we may temporarily ship:
  - detector on ANE/GPU
  - segmenter partially on GPU/CPU if necessary

---

## 5. Model Export / Format Pipeline (High-level)

> This section defines *what* is exported and validated; detailed conversion commands are owned by **ML_Vision**.

### 5.1 YOLO-v9 (Detection)
Target runtime: Core ML (ANE-first)
- Export outcome: a Core ML model whose outputs can be decoded into bboxes/classes.
- Post-processing (NMS/decoding) placement options:
  1) Inside the model graph (fewer CPU steps; sometimes harder to export)
  2) Outside the model in Swift (more flexible; might cost CPU)
- Decision guideline:
  - Start with “outside” for correctness/debuggability; migrate “inside” if CPU becomes bottleneck.

### 5.2 MobileSAM (Segmentation)
MobileSAM conceptually consists of:
- Image encoder (embedding)
- Prompt encoder (bbox/points)
- Mask decoder

Export outcomes (two viable architectures):
- Option A (recommended for performance):
  - Encode image embedding at a reduced cadence or resolution
  - Reuse embedding across multiple prompts in the same frame
  - Then decode masks for each prompt
- Option B (simpler but slower):
  - One monolithic model per prompt that takes image + box and emits mask

Architecture requirement:
- The iOS module boundaries must allow either A or B, because export feasibility may dictate it.

---

## 6. Interface Between Detection and Segmentation (Core Contract)

### 6.1 Coordinate systems
Define a single canonical coordinate system for prompts:
- Canonical: **pixel coordinates in original camera frame** `(xMin, yMin, xMax, yMax)`
- PromptBuilder is responsible for converting detector output into canonical coordinates.

### 6.2 Prompt schema
- Required prompt type for v1: **bounding box**
- Optional additions (future):
  - positive/negative points
  - prior mask for refinement

### 6.3 Instance identity
- Each detection is assigned an `instanceId` via TemporalManager association rules.
- Segmentation outputs attach to `instanceId` to update the cached mask.

### 6.4 Gating policy (budget control)
Segmentation is expensive; define gating:
- `maxInstancesPerFrame` (e.g., 1–3 on iPhone 11)
- `minScore` threshold
- `minIoUChange` threshold to re-segment (if bbox moved significantly)
- periodic refresh (e.g., re-segment each active instance every N frames)

---

## 7. Performance Targets & Constraints (iPhone 11)

### 7.1 Targets (initial, adjustable)
- Preview/render: 30 FPS display if possible
- Detector throughput: ≥ 15–30 FPS depending on input resolution
- Segmentation throughput: budgeted; aim for **interactive** updates:
  - 5–10 Hz mask refresh for the primary object, or
  - 1–3 objects with lower refresh (e.g., 2–5 Hz)
- End-to-end perceived latency (camera → overlay update):
  - detector overlay: < 100 ms typical
  - mask overlay: < 250–400 ms typical (budgeted)

### 7.2 Primary constraints
- Thermal throttling: sustained loads will downclock.
- Memory: mask tensors and embeddings can be large; must reuse buffers.
- Threading: avoid main thread work; keep rendering on GPU path.

### 7.3 Scheduling model
- Separate queues:
  - capture queue
  - detector inference queue
  - segmenter inference queue
  - rendering queue
- Strict rule: never block capture on inference completion.

---

## 8. Risks / Open Questions (for Day 2+ follow-up)

1. **Export feasibility** for YOLO-v9 and MobileSAM ops to Core ML / ANE.
2. Best MobileSAM decomposition (Option A vs B) given Core ML limitations.
3. Optimal camera resolution tradeoff (e.g., 640p vs 720p) for iPhone 11.
4. NMS placement (in-model vs post-process) and its CPU cost.
5. Tracking strategy choice (simple IoU association vs more advanced).

---

## 9. Decisions (Day 1)

- Adopt a **two-stage architecture** with explicit module boundaries to allow model-export-driven changes.
- Define a **canonical prompt coordinate space** (original frame pixels) and require PromptBuilder to map to it.
- Enforce **budgeted segmentation** with caching/temporal stabilization to meet real-time constraints.

---

## 10. Deliverables Produced Today (Architect)

- High-level architecture spec (this document) covering:
  - Python ⇆ iOS division of responsibilities
  - model export pipeline (high-level)
  - detection→segmentation interfaces
  - iOS camera→inference→overlay data flow
  - performance targets and constraints

---

# Day 2 Addendum (Architect)

> 目标：先让 Day 2 “可编译 + 可加载模型 + 单帧推理 + overlay” 顺畅推进。
> 本补充定义 **坐标/几何契约 v1**、**I/O 契约表 v1**，并更新 **风险清单**（按里程碑排序）。
> 2026-02-26（重建复核）：在新一轮 JudgeE2 任务中复核无冲突，契约保持有效。

## A. 坐标系/几何契约 v1 (Coordinate + Geometry Contract)

### A.0 术语与对象
- **Camera Frame**：相机输出的原始图像缓冲（`CVPixelBuffer`），尺寸记为 `Wc × Hc`（单位：px）。
- **Preview View**：屏幕上的显示区域（UIKit view），尺寸记为 `Wv × Hv`（单位：pt）。
- **previewLayer**：`AVCaptureVideoPreviewLayer`，其 `bounds` 与 view 对齐，但其 **videoGravity** 决定画面如何缩放/裁剪。
- **Model Input**：送入 YOLO 的输入尺寸 `Wi × Hi`（单位：px），例如 640×640。
- **Letterbox**：保持宽高比缩放到目标输入尺寸，未覆盖区域用 padding 填充。

### A.1 统一“规范坐标空间”(Canonical Space)
**Canonical 坐标空间 = Camera Frame 像素坐标**：
- 原点：左上角 `(0, 0)`
- x 轴：向右增大
- y 轴：向下增大
- bbox 表达：`(xMin, yMin, xMax, yMax)`，均为 **像素**，可为 Float（但语义是 px）

> 约定：Detector 的 decode、NMS、PromptBuilder、Renderer 中的几何数据**统一使用 Canonical（camera px）**。
> 任何与 preview 显示、与 model input 的转换都必须携带 transform 元数据，并可逆。

### A.2 设备旋转与图像朝向（Orientation）
- 运行时显示通常为 portrait，但相机输出可能为 landscape 缓冲。
- 契约：Preprocess/FrameBroker 必须产出一个 `FrameGeometry` 元数据，显式描述：
  - `cameraWidth=Wc, cameraHeight=Hc`
  - `orientation`（等价于 EXIF / CGImagePropertyOrientation 语义）
  - `isMirrored`（前置摄像头常见）

**规则（v1 简化）**：
- YOLO 推理输入使用与 preview 一致的“用户看到的方向”（即在送入模型前做 rotate/mirror，使 model input 与屏幕方向一致）。
- 因此，模型输出 bbox 先回到“推理前的那张图”的像素空间，然后再通过记录的几何映射回 Canonical camera px（如有必要）。

> 这样做的好处：Builder/Debugger 在 overlay 与点击交互时不会被 orientation 搅乱；缺点是 preprocess 多一步旋转。

### A.3 View(tap) ↔ Preview ↔ Camera Frame(px)
需求：用户点击屏幕某点（未来用于 point prompt），需要映射到 Canonical camera px。

定义 `videoGravity` 两类：
1) **resizeAspectFill（常用）**：保持比例，填满 view，**会裁剪**。
2) **resizeAspect（fit）**：保持比例，完整显示，**会留黑边**。

对任意 gravity，都定义从 camera→view 的仿射映射：
- 计算缩放因子：
  - `sFill = max(Wv/Wc, Hv/Hc)`
  - `sFit  = min(Wv/Wc, Hv/Hc)`
- 计算 scaled 尺寸：`Ws = Wc * s`, `Hs = Hc * s`
- 计算 offset（以 view 左上为原点）：
  - `ox = (Wv - Ws)/2`
  - `oy = (Hv - Hs)/2`

映射：
- camera px → view pt：
  - `xv = ox + xc * s`
  - `yv = oy + yc * s`
- view pt → camera px（点击回投）：
  - `xc = (xv - ox)/s`
  - `yc = (yv - oy)/s`

**裁剪/黑边处理**：
- aspectFill：`ox, oy` 常为负数（表示被裁掉），view→camera 反投后仍需 clamp 到 `[0,Wc)×[0,Hc)`。
- aspectFit：`ox, oy` 常为正数（黑边），点击落在黑边区的点反投会落在 camera 外，应当判定无效或 clamp（推荐：判定无效）。

### A.4 Camera Frame(px) ↔ Model Input(letterbox)
YOLO 常用 letterbox 到 `Wi×Hi`。

定义：
- `r = min(Wi/Wc, Hi/Hc)`
- resized 尺寸：`Wr = Wc*r`, `Hr = Hc*r`
- padding：
  - `px = (Wi - Wr)/2`
  - `py = (Hi - Hr)/2`

映射：
- camera px → model input px：
  - `xm = px + xc * r`
  - `ym = py + yc * r`
- model input px → camera px：
  - `xc = (xm - px)/r`
  - `yc = (ym - py)/r`

bbox 的映射同理对四个边界应用该变换。

**重要：必须记录并复用** `r, px, py, Wi, Hi, Wc, Hc`，作为该帧的 `LetterboxTransform` 元数据。

### A.5 YOLO 输出 decode → Canonical
- 如果 YOLO 输出为 **归一化到 model input**（0..1）或 **以 model input px 表示**：先还原到 model input px，再用 A.4 映射回 camera px。
- 若输出已是 camera px（较少见）：必须在 Model Card 中明确标注，并跳过 A.4。

### A.6 渲染（Canonical → Preview）
Renderer 需要将 camera px 的 bbox/mask 绘制到 previewLayer：
- 使用 A.3 的 camera→view 映射（使用与 previewLayer 相同 gravity/尺寸）。
- mask 若以 camera 尺寸存储：
  - 优先：按 camera→view 的同一映射缩放/裁剪后叠加。
  - 允许：在 GPU（Metal）中做纹理采样映射，避免 CPU resize。

### A.7 v1 约束与测试用例
- v1 只保证：**同一帧内** bbox/mask 的 overlay 与点击回投一致。
- Debugger/Builder 可用以下 sanity check：
  - 在 camera 四角/中心画 5 个点（Canonical），检查落在 preview 的正确位置。
  - 点击 preview 中心点，反投应接近 `(Wc/2, Hc/2)`。

---

## B. I/O 契约表 v1（Swift 侧期望）

> 说明：具体 tensor 名称/shape 以 ML_Vision 导出的 CoreML artifact 为准。本表为 **Swift 侧集成的最小契约**：需要哪些字段、我们如何适配。

### B.1 YOLOv9 Detector — 输入契约
- 输入数量：1（图像）
- 输入类型（首选）：`CVPixelBuffer`（由 CoreML/Vision 负责转张量）
- 输入色彩：RGB 或 BGR（必须在 Model Card 中固定）
- 输入范围：`[0,1]` 或 `[0,255]`（必须固定）
- 预处理：
  - resize：letterbox 到 `Wi×Hi`
  - normalization：`mean/std`（若使用）

**字段（待 ML_Vision 确认）**
- `input.name`: TBD（常见：`image` / `input`）
- `input.shape`: `[1, 3, Hi, Wi]`（NCHW）或 `[1, Hi, Wi, 3]`（NHWC）
- `input.dtype`: FP16/FP32（CoreML 内部可能自动）

### B.2 YOLOv9 Detector — 输出契约
为了让 Builder/Debugger 先跑通，建议 v1 支持两种输出形态之一：

**形态 1：已解码候选（推荐易用）**
- 输出 keys：
  - `boxes`: `[N, 4]`（`xMin,yMin,xMax,yMax`，坐标空间需注明：model input px 或 normalized）
  - `scores`: `[N]`
  - `classIds`: `[N]`
- dtype：FP16/FP32（boxes/scores），Int32（classIds）

**形态 2：原始 head 输出（后处理在 Swift）**
- 输出 keys：一个或多个 feature map（例如按 stride：8/16/32）
- shape：TBD（与 YOLOv9 具体实现强相关）
- 需要 Model Card 提供：
  - stride 列表
  - 维度含义（例如 `[1, C, H, W]`）
  - decode 公式（anchor-free vs anchor-based）、类别数、分数计算方式

> v1 集成建议：若 ML_Vision 可在导出时把 decode/NMS 放入图内，Builder 会更快达成 smoke test；否则至少保证输出 keys/shape 清晰稳定。

### B.3 MobileSAM（后续）— 预留输入/输出契约（不阻塞 Day 2）
为避免后续接口大改，Swift 侧先按“可组合三件套”预留：
- Image encoder 输入：`image`（可能是 `CVPixelBuffer` 或 `MLMultiArray`）
- Prompt encoder 输入：
  - `boxes`: `[K, 4]`（Canonical camera px 或 model input px，必须固定）
  - `points`（可选）
- Mask decoder 输出：
  - `masks`: `[K, 1, Hm, Wm]`（float logits 或 prob）
  - `iou_predictions` / `mask_scores`（可选）

---

## C. 风险清单更新（对齐 Day 2 “先跑通再优化”里程碑）

按里程碑排序（先保证路径通，再谈性能）：

### C.1 编译/工程层风险
- Xcode 工程设置与 iOS deployment target 不一致（CoreML/Vision API 版本差异）。
- 模型资源加入方式不当（把 `.mlmodel` 当运行时文件；未触发编译成 `.mlmodelc` / `.mlpackage` 处理）。

### C.2 相机与预览风险
- orientation/mirror 与 previewLayer 的显示不一致，导致 overlay 错位。
- aspectFill 裁剪未计入坐标映射，导致 bbox/mask 偏移。

### C.3 CoreML 加载/单帧推理风险（Day 2 必过）
- 输入类型不匹配（期望 image 却喂了 multiarray 或尺寸不对）。
- output keys/shape 不稳定或未文档化，导致 Debugger 无法打印并验证。

### C.4 后处理与 decode 风险
- YOLO 输出若为 raw head：Swift 侧 decode/NMS 工作量大且易错；建议尽快由 ML_Vision 给出稳定 decode 说明或内置后处理。

### C.5 Overlay 与性能风险（先正确后快）
- 在主线程做 resize/渲染导致 UI 卡顿。
- mask 张量过大导致内存峰值高；需要缓存与复用策略（Day 3+）。

### C.6 退路策略（v1）
- 若 ANE 不支持：允许 detector 先跑 GPU/CPU（computeUnits 设置），以“能跑通”为先。
- 若输出复杂：允许先只画 bbox（无 mask），确认几何契约正确，再接 MobileSAM。

---

# Day 3 Addendum (Architect)

## D. Review: FrameGeometry / Letterbox / Overlay（与 Day 2 契约一致性）

本节是对 Day 3 目标（真机 camera→preprocess→infer→decode/NMS→bbox overlay）所需几何链路的 **审阅清单** + **边界条件补充**。原则：
- **所有逻辑坐标统一使用 Canonical camera px**（Day 2 §A.1）。
- 任何进入/离开 Canonical 的变换都必须显式、可逆、可复现。

### D.1 FrameGeometry 最小字段（必须）
建议 Builder/Debugger 侧将 `FrameGeometry` 视为“这帧的几何真相”，至少包含：
- `Wc, Hc`：camera buffer 像素尺寸（注意：是 buffer 的宽高，不是 screen）
- `orientation`：与用户看到方向对齐的枚举（等价 EXIF/CGImagePropertyOrientation 语义）
- `isMirrored`：前置摄像头镜像标记
- `timestamp`：用于关联推理/渲染

**边界条件**
- orientation/mirror 处理必须在一个地方“定锚”。推荐：在 preprocess 产出 640×640 输入时完成 rotate/mirror，使 **model input 与 preview 视觉一致**（Day 2 §A.2）。
- 若未来改为“模型吃原始 buffer 不旋转”：必须同时改写 overlay 的映射链路，不可混用。

### D.2 LetterboxTransform（必须可回投）
对每个送入 detector 的帧，记录：
- `Wi, Hi`（当前 YOLOv9-c = 640×640）
- `r, px, py`（Day 2 §A.4）
- `Wc, Hc`（与 FrameGeometry 一致）

**边界条件（数值/取整）**
- `r, px, py` 允许为 Float；但实际实现会涉及像素取整。
- 建议契约：
  - letterbox resize 使用 **同一个 rounding 规则**（例如四舍五入或 floor），并在 debug 日志中打印 `Wr,Hr,px,py`。
  - decode 回投时使用同样的 `px,py,r`，避免 off-by-one。

### D.3 Decode 输出坐标空间（必须明确）
当前 ML_Vision 已确认：
- 主输出：`var_3019` float32 shape `(1,84,8400)`
- 含义：`84 = 4 + 80`（COCO），且 **无 objectness**；cls 已 sigmoid（由 reference decoder 给出）

因此 decode 的坐标空间通常是 **model input 空间**（640×640）上的 `xywh(center)`。

契约要求：Swift decode 后必须输出 Canonical camera px：
1) `xywh(center)` → `xyxy`（仍在 model input 空间）
2) 用 `LetterboxTransform` 执行 model→camera 的逆变换（Day 2 §A.4）
3) clamp 到 `[0,Wc)×[0,Hc)`

### D.4 Preview overlay 映射（Canonical → View）
Renderer 只做一件事：把 Canonical camera px 映射到 preview view。

**默认建议（v0）**
- previewLayer `videoGravity = resizeAspectFill`（更符合相机 App 体验）。
- overlay 使用 Day 2 §A.3 的 camera→view 映射，其中 `s = max(Wv/Wc, Hv/Hc)`。

**边界条件**
- aspectFill 下 `ox/oy` 可能为负：bbox/mask 映射到 view 后会自然落在 view 边界外；渲染时可由 clip 自动裁剪。
- 如果改用 aspectFit：必须处理黑边点击无效区（Day 2 §A.3）。

### D.5 Sanity Check 建议（与 Debugger “5 点测试”对齐）
为快速验证几何闭环（无须看模型对不对）：
- 在 Canonical camera px 上选点：四角 + 中心
- 映射到 view，画点
- 同时在 view 点击这些点位置（或近似），反投回 Canonical，应接近原点（误差来自手指点击与取整）

---

## E. 默认阈值建议（Day 3 v0：先稳定跑通）

### E.1 Detector decode/NMS 默认值（与 ML_Vision reference 对齐）
- `scoreThreshold = 0.25`
- `iouThreshold = 0.65`
- `topK = 100`

**建议补充（便于移动端稳定）**
- `maxDetPerFrame = 20`（进入 overlay 的最大框数；防止极端场景爆炸）
- NMS 策略：优先 **class-aware NMS**；如实现成本高，可临时用 class-agnostic（但类别会互相抑制）。

### E.2 预算与掉帧（v0）
- 推理频率：先确保 detector 可持续运行（掉帧允许，但必须“最新帧优先”）。
- overlay：先 bbox-only；mask 后续再接 MobileSAM。

---

## F. 需要继续确认/对齐的 2 个点（不阻塞 Day 3，但要记录）
1) `var_3022` 的语义与用途：目前不参与 decode，但建议在后续 model plan 中给出解释/是否可删。
2) PixelBuffer 的颜色空间与 CoreML 输入预处理：当前模型 input 说明为 RGB + scale=1/255；需确保 camera→pixelBuffer→model 的颜色通道一致（避免 BGR/RGB 颠倒导致“能跑但全错”）。

---

# Day 5 Addendum (Architect)

## G. Pipeline 调度与缓存策略 review v1（iPhone 11 默认建议）

> 目标：在 **Detector + MobileSAM split**（encoder/decoder 分离）架构下，达到“画面稳定、交互可用、不会卡 UI”的体验。
> 原则：
> - **相机/预览优先**：永不阻塞 capture / preview。
> - **最新帧优先**：detector 追最新帧；segmenter 追“当前关注实例”。
> - **缓存优先**：mask/embedding 以缓存为主，推理为补充。

### G.1 运行时实体与缓存对象（Runtime State）
建议在 TemporalManager（或等价组件）维护以下状态（逻辑概念，不要求具体实现形式）：

- `PrimaryInstance`：当前“主目标”（top-1 或用户选择）
  - `instanceId`
  - `classId` / `label`
  - `bbox_cam`（Canonical camera px）
  - `lastSeenTs`
- `EmbeddingCache`（按“帧/时间”缓存，不按实例）：
  - `embedding`（来自 MobileSAM encoder 输出）
  - `embeddingFrameId` / `embeddingTs`
  - `geometryId`（对应的 FrameGeometry/letterbox/preview 变换版本号）
  - `ttlFrames` / `ttlMs`
- `MaskCache`（按实例缓存）：
  - `mask_lowres_logits`（如 256×256 logits）
  - `maskFrameId` / `maskTs`
  - `maskForBbox_cam`（生成该 mask 时的 bbox）
  - `renderedMaskTexture`（可选：GPU 纹理缓存，用于避免重复 CPU→GPU 上传）

**关键约束**：任何缓存结果必须携带“它对应哪一帧的几何/尺寸语义”（FrameGeometry id）。否则在 aspectFill + rotation 情况下容易出现“mask 贴错帧/贴错方向”。

### G.2 调度策略 v1：三条 cadence（Detector / Encoder / Decoder）

#### 1) Detector cadence（高频，追最新帧）
- 默认：尽可能高频，但采用 **latest-frame-only**（队列长度=1）策略。
- 若 detector 落后：允许丢帧；但必须保持 bbox 更新及时。

#### 2) MobileSAM Encoder cadence（低频，生成 embedding）
- 默认：`encoderEveryNFrames = 12`（约等于在 30fps 下 2.5Hz）
- 动态调整建议：
  - 若设备温度升高或掉帧明显：提高到 18–24
  - 若用户需要更“跟手”：降低到 8–10（代价是算力/功耗上升）

Encoder 触发条件（任一满足则触发）：
- `(frameIndex - embeddingFrameIndex) >= encoderEveryNFrames`
- `PrimaryInstance` 改变（切换主目标）
- `geometryId` 变化（orientation/mirror/输入尺寸变化；或预处理路径切换）
- `EmbeddingCache` 过期（见 G.4）

#### 3) MobileSAM Decoder cadence（中频/事件驱动，更新 mask）
- 默认：`decoderEveryNFrames = 6`（30fps 下约 5Hz）
- 触发条件（任一满足则触发）：
  - `(frameIndex - lastMaskFrameIndex(instanceId)) >= decoderEveryNFrames`
  - bbox 变化显著（见 G.3 的阈值）
  - mask 过期（G.4）

> 注：decoder 运行依赖 embedding。若 embedding 未就绪：
> - 允许先复用旧 mask（如果未过期）
> - 或仅显示 bbox（fallback）

### G.3 top-1 policy / 主目标选择与切换（Primary selection）

默认主目标策略（v1：无交互选择时）：
1. 对检测结果做 NMS 后，按 `score` 降序。
2. 过滤 `score < primaryMinScore` 的候选。
3. 选择 top-1 作为 `PrimaryInstance`。

建议默认参数：
- `primaryMinScore = 0.35`（比 decode 的 0.25 更保守，减少“乱触发分割”）
- `primaryHysteresisFrames = 12`：避免主目标在相近分数间抖动
  - 规则：若当前主目标仍可见且 `score >= 0.8 * bestScore`，则保持不切换

主目标丢失策略：
- 若 `PrimaryInstance` 连续 `primaryLostFrames = 15` 帧未匹配到检测（或 `score` 很低），则清空主目标，回到“bbox-only 或 top-1 重新选”。

实例匹配（用于保持 instanceId 稳定）：
- 使用 IoU 匹配（class-aware 优先）：
  - `matchIoUThreshold = 0.5`
  - 备选：若同类框 IoU < 0.5，但中心点距离很近，也可保持（以减少近距离抖动）

### G.4 缓存/过期策略（Cache TTL）

#### EmbeddingCache TTL
- `embeddingTtlFrames = 36`（约 1.2s @30fps）
- `embeddingTtlMs = 1200`
- 过期条件：
  - 超过 TTL
  - geometryId 不一致（orientation/mirror/输入尺寸变化）

#### MaskCache TTL
- `maskTtlFrames = 24`（约 0.8s）
- `maskTtlMs = 800`
- mask 再计算触发（bbox drift）：
  - `bboxReSegIoU = 0.70`：当前 bbox 与生成 mask 时的 bbox IoU < 0.70 则强制重分割
  - 或 `centerShiftPx = 0.08 * min(Wc,Hc)`（约 50px@640p 级别）

#### Fallback / degrade
当 segmenter 忙/不可用/超时：
- 优先保证 UI：继续显示 bbox（detector-only）
- 若有旧 mask 且未过期：允许继续显示旧 mask（并在 HUD 标注“stale mask”）

### G.5 并发与队列规则（避免互相干扰）

建议严格的“单航道”模型，避免多路并发争用内存带宽：
- `DetectorQueue`：串行（或限制并发=1），latest-frame-only
- `SAMEngineQueue`：串行；encoder/decoder 都在同一队列上排队（避免 encoder/decoder 互相抢资源导致尾延迟爆炸）
- `RenderQueue`：主线程仅提交轻量 draw；重采样/合成尽量在 GPU

必要的背压策略：
- 若 `SAMEngineQueue` backlog > 1：丢弃旧的 decoder request，只保留最新主目标请求。

### G.6 交互体验相关的默认“可调旋钮”（Perf knobs）

**推荐暴露为常量/配置（便于 Day5 A/B）**：
- `encoderEveryNFrames`（默认 12）
- `decoderEveryNFrames`（默认 6）
- `primaryMinScore`（默认 0.35）
- `primaryHysteresisFrames`（默认 12）
- `bboxReSegIoU`（默认 0.70）
- `maskTtlMs`（默认 800）

**建议 HUD 必显示的派生指标**（用于快速验证策略是否生效）：
- embedding cache hit rate（%）
- decoder request drop count（因 backlog 丢弃）
- 当前主目标稳定时长（frames）
- mask age（ms）

### G.7 质量/稳定性边界条件（必须明确）
- 在 `.resizeAspectFill` 下，mask 的显示应当与 bbox 同一条 Canonical→View 映射链路。
- 若 mask 为低分辨率（256×256），必须明确它的坐标语义：
  - 语义 A：对应 encoder 输入（例如 1024×1024）
  - 语义 B：对应 camera frame（Wc×Hc）
  - 语义 C：对应裁剪 ROI

v1 推荐：mask 语义固定为“对应 encoder 输入的整图（1024×1024）”，再通过与 encoder 相同的几何映射回 Canonical camera px → preview。

---

## H. 本 addendum 的落地范围（与现有 Day4/Day5 Builder 任务对齐）
- 若 Builder 已实现 split + embedding cache：以上提供的是 **默认参数与过期/回退规则**，可直接作为 Day5 A/B 的基线。
- 若当前仍是 monolithic MobileSAM：仍可使用本文的 **top-1、TTL、backpressure** 思路，只是 encoder/decoder cadence 合并为单一“segmentEveryNFrames”。

---

# Day 6 Addendum (Architect)

## I. Orientation & Mirroring 契约 v2（D6-A-GEOMETRY-V2）

> 目标：把“方向（orientation/rotation）”与“镜像（mirroring）”的**语义、来源、优先级**一次性讲清楚，避免出现：
> - preview 显示方向正确，但模型输入方向不一致
> - bbox/mask overlay 偶发左右反
> - 前置摄像头（自拍）镜像规则在不同模块各自处理，导致重复镜像或漏镜像
>
> 本节只定义**契约/接口**，不规定具体实现方式。

### I.0 不变项：Canonical 定义（沿用 v1）
- **Canonical 坐标空间不变**：仍然定义为 **Camera Frame 像素坐标**（buffer 的 `Wc×Hc`），原点左上，x→右，y→下。
- 所有检测/分割输出在进入 Temporal/Renderer 前必须归一到 Canonical（camera px）。

### I.1 FrameGeometry（v2）— 必填字段（Required）
`FrameGeometry` 是“这帧几何真相”的唯一载体。对每个 camera frame（`CVPixelBuffer`）必须伴随以下字段：

**1) 尺寸与时间**
- `cameraWidthPx (Wc)`：Int
- `cameraHeightPx (Hc)`：Int
- `timestamp`：用于跨模块关联（CMTime/秒/帧序号均可，但必须单调）

**2) 方向（Orientation）**
- `orientation`：等价于 `CGImagePropertyOrientation` 语义（Up/Down/Left/Right + Mirrored variants 也可，但见 I.3）
- `orientationSource`：枚举，必须记录来自哪一种来源（见 I.2）

**3) 镜像（Mirroring）**
- `isFrontFacing`：Bool（是否前置摄像头；用于默认镜像策略）
- `isPreviewMirrored`：Bool（用户看到的 preview 是否左右镜像）
- `isModelInputMirrored`：Bool（送入模型的输入是否左右镜像）

> 解释：
> - `isPreviewMirrored` 与 UI 体验相关；
> - `isModelInputMirrored` 与 ML 输入相关；
> 两者允许不同，但必须显式声明，任何模块不得“猜”。

**4) 版本与一致性（强烈建议，便于调试）**
- `geometryId`：可用递增 Int 或 hash（orientation/mirror/尺寸任一变化都应变化），用于缓存失效（embedding/mask）

### I.2 Orientation 的允许来源（Allowed sources）与优先级
本项目允许的 orientation 来源仅限三类（必须在 `orientationSource` 中标注）：

1) **AVCaptureConnection / videoRotationAngle（推荐为首选）**
- 来源：`AVCaptureConnection.videoRotationAngle`（或等价 API）
- 语义：描述当前输出与设备/界面的旋转关系，适合实时视频 pipeline。

2) **AVCaptureVideoDataOutput 的 sample buffer attachments / EXIF 方向（兼容路径）**
- 来源：CMSampleBuffer 附带的 EXIF/CGImagePropertyOrientation 信息（若存在）
- 语义：偏“图像文件/元数据”语义，适合静态图或某些系统输出。

3) **强制约定（Fallback / Bring-up）**
- 仅在上述两类不可用或明显错误时使用：例如 bring-up 阶段固定 portrait。
- 必须在 `orientationSource = .forced` 中标记，且 Debug HUD/日志应提示。

**优先级（当多种同时可用时）**
- 运行时视频流：优先用 `connection/videoRotationAngle`。
- 若 videoRotationAngle 缺失或不可信：再用 EXIF。
- forced 仅作为最后退路。

**禁止事项**
- 同一帧不得同时被多个模块“各自决定 orientation”。唯一真相必须来自 `FrameGeometry.orientation`。

### I.3 Mirroring 规则（前置镜像）— 明确且可组合
镜像在 iOS 摄像头链路里会出现两种“镜像语义”，本契约必须同时覆盖：

- **Preview mirroring（UI 镜像）**：用户看到的是“镜子”效果（左右反）。
- **Buffer/model mirroring（数据镜像）**：实际像素数据是否已左右翻转。

#### I.3.1 默认 UI 规则（推荐体验）
- 前置摄像头：`isPreviewMirrored = true`（像镜子一样）
- 后置摄像头：`isPreviewMirrored = false`

> 注：这只是 UI 默认。若产品决定“前置也不镜像”，则必须通过 `isPreviewMirrored=false` 显式表达。

#### I.3.2 模型输入的镜像策略（必须二选一并固定）
为避免左右反与 bbox/mask 错位，模型输入镜像必须选择一条路线并在整个 pipeline 中固定：

- **路线 A（推荐）**：模型输入与用户看到的方向保持一致
  - 规则：`isModelInputMirrored = isPreviewMirrored`
  - 好处：debug 更直观（模型“看见”的就是屏幕上看到的）
  - 约束：所有输出回到 Canonical 时必须记录并应用对应的 mirror/rotation 逆变换（见 I.4）

- **路线 B（可选）**：模型输入永不镜像（保持与原始 buffer 方向一致）
  - 规则：`isModelInputMirrored = false`（无论前后摄）
  - 好处：减少一次数据变换（有时更快/更省事）
  - 代价：UI/交互要额外处理“preview 镜像”与“模型非镜像”的差异，极易出错；只有在性能/兼容性强约束时采用。

**禁止事项**
- Detector 与 Segmenter 不能使用不同的镜像策略。`isModelInputMirrored` 必须对两个模型一致。

### I.4 变换链路的“单一真相”与可逆性（Orientation+Mirror）
为了保持 v1 的“可逆、可回投”原则，v2 强制要求：

- 每帧必须存在一条明确的变换链：
  - **Canonical (camera px)** ⇄ **Oriented/Mirrored Working Space** ⇄ **Model Input (letterbox / 640×640 或 1024×1024)**
- `FrameGeometry` 必须能回答两个问题：
  1) 从 Canonical 到“用户看到的 preview 方向”需要怎样的 rotate/mirror？
  2) 从 Canonical 到“模型输入空间”需要怎样的 rotate/mirror？（由 `orientation` + `isModelInputMirrored` 确定）

> 约束：
> - rotate/mirror 的选择必须集中在一个地方“定锚”（推荐由 Preprocess 产生 working space 与模型输入，并写入 FrameGeometry / Transform 元数据）。
> - Renderer/PromptBuilder/TemporalManager 只消费这些元数据，不再自己推断。

### I.5 与 LetterboxTransform 的关系（不变，但强调顺序）
- LetterboxTransform（`r,px,py,Wi,Hi,Wc,Hc`）仍然是必要元数据。
- v2 强调：Letterbox 发生在哪个空间必须明确：
  - 如果先做 rotate/mirror 再 letterbox：则 `Wc,Hc` 指的是 rotate/mirror 后的 working-space 尺寸（必须在文档/字段中说明）。
  - 如果先 letterbox 再 rotate/mirror：则 letterbox 的 `Wc,Hc` 为原 buffer 尺寸。

> 推荐做法（与 v1 思路一致，降低认知复杂度）：
> - 先把输入变换到“与 preview 一致”的 working space（解决 orientation+mirror）
> - 再对该 working space 做 letterbox 到模型输入

### I.6 最小验收（用于 Debugger/Builder）
v2 契约落地后，必须能用以下最小测试在真机上快速发现左右反/旋转错：
- 后置 portrait：bbox overlay 与目标位置一致；左右不反。
- 前置 portrait：
  - 若 `isPreviewMirrored=true`：人脸举左手时，preview 中应显示为“右手”（镜像），但 bbox/mask 必须仍贴合脸部。
  - 点击 preview 左侧的点，反投到 Canonical 后，再映射回 preview，应回到同一视觉位置（误差仅来自取整/手指）。
- 旋转设备（或切换 UI orientation，如果支持）：bbox/mask 仍正确贴合，且 geometryId 变化能触发缓存失效（embedding/mask 不应“贴在旧方向”上）。

---

## J. Scheduling & Contention Guide v1 (Day 6)

> 目标：基于 Day 6 的争用测试（Contention Mitigation）与 Compute Units AB 结果，固化**默认调度参数**与**并发抑制策略**。
> 核心原则：**重任务互斥**（Heavy-task Exclusion）。不要让两个 CPU/ANE 密集型任务在同一帧或极短窗口内撞车。

### J.1 默认参数（Default Knobs）

| Parameter | Default Value | Context / Rationale |
|---|---|---|
| **YOLO ComputeUnits** | `.all` | 必须启用 ANE 以获得 ~230ms 稳态推理；冷启动（~8s）需通过 UI 掩护。 |
| **SAM Encoder ComputeUnits** | `.cpuAndGPU` | 默认求稳（稳态 ~600ms+）；若追求性能可开 `.all`（需 AB 验证无 mask 异常）。 |
| **SAM Decoder ComputeUnits** | `.cpuAndGPU` | 耗时极短（~35ms），与 Encoder 保持一致或独立均可。 |
| **YOLO Format** | `.mlpackage` | 推荐迁移到 mlprogram（iOS 17+）以获得更好的编译器优化与 ANE 亲和性。 |

### J.2 动态调度与流控（Flow Control）

#### 1) 稳态 cadence（无争用时）
- `detector.infer`: **Always** (subject to capture FPS, e.g., 30fps drop-if-busy)
- `detector.postprocess` (decode+nms): **Every 1 frame** (全速)
- `sam.encoder`: **Every 12 frames** (~2.5 Hz)
- `sam.decoder`: **Every 6 frames** (~5 Hz)

#### 2) 并发抑制（Contention Mitigation）
当系统检测到“重任务”活跃时，必须主动降频其他模块：

**Scenario A: Run Golden (Benchmark/Calibration)**
- **Trigger**: 用户点击 Run Golden，或系统后台运行校准。
- **Action**: **PAUSE Realtime Pipeline**。
  - 停止 Detector 推理与 Postprocess。
  - 停止 SAM Encoder/Decoder。
  - 仅保留 Camera Preview（视频流不断）。
- **Exit**: Golden 完成后恢复。

**Scenario B: SAM Encoder Active (Heavy CPU/ANE load)**
- **Trigger**: `shouldRunEncoder = true` (based on cadence/trigger).
- **Action**: **Throttling Detector Postprocess**。
  - 在 Encoder 运行的当帧（及前后 1 帧窗口），将 `detector.postprocess` 降频或 Skip。
  - 建议：`detector.postprocessEveryNFrames` 临时升至 3（或直接 skip current frame decode）。
- **Rationale**: 避免 Encoder (~600ms) 与 Decode+NMS (~150ms) 叠加导致单帧耗时 >800ms，引发明显掉帧卡顿。

### J.3 Decode/NMS 优化参数
基于 Day 6 AB 结果，推荐以下组合以平衡 Recall 与 Latency：

- `scoreThreshold`: **0.35** (较保守，减少送入 NMS 的框数量)
- `preNmsTopK`: **150** (限制 NMS 计算量)
- `topK`: **50** (最终输出框数上限)
- `classAwareNms`: **ON** (如果性能允许；否则 OFF)

> 调优目标：将稳态 `decode+nms` 控制在 **<80ms** (iPhone 11) 以留出时间片给渲染与调度。

