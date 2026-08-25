<!-- # JudgeE2 — 7 Day Plan (Phase 1: Detection Only)

## Objective

Build a minimal iOS app that runs YOLO detection on iPhone 11.

Scope for this 7-day cycle:
- Camera input
- YOLO CoreML inference
- Decode + NMS
- Bounding box overlay

Resource:
- python3 virtual environment: /Users/jiansun/Documents/Doctor\ Courses/4455/env1

Out of scope (next 7-day cycle):
- Segmentation (MobileSAM)
- Model quantization
- mlpackage/mlprogram optimization
- Multi-model scheduling
- Performance micro-optimization

Rule:
Correctness first. Optimization later.

---

# Day 1 — Clean iOS Foundation

## Builder
- [x] Create new iOS project (iOS 17 target)
- [x] Verify Simulator runs default template
- [x] Verify iPhone 11 builds and launches
- [x] Confirm Bundle Identifier
- [x] Confirm Signing Team set correctly
- [x] Commit clean baseline

## Debugger
- [x] Confirm no signing errors
- [x] Confirm app launches on Simulator
- [x] Confirm app launches on device

Deliverable:
App runs on iPhone 11 with default UI.

---

# Day 2 — CoreML Model Load

## ML_Vision
- [x] Provide YOLO CoreML model (.mlmodel)
- [x] Provide input name
- [x] Provide output tensor names
- [x] Provide expected output shape
- [x] Confirm minimum iOS requirement

## Builder
- [x] Import .mlmodel into Xcode
- [x] Confirm model class auto-generated
- [x] Write minimal load test
- [x] Print "Model loaded successfully"

## Debugger
- [x] Confirm model loads on device
- [x] Confirm no runtime crash
- [x] Confirm console log visible

Deliverable:
YOLO model loads successfully on iPhone 11.

---

# Day 3 — Single Image Inference

## Builder
- [x] Create dummy 640x640 image input
- [x] Run single inference
- [x] Print output tensor shapes
- [x] Measure inference time

## Debugger
- [x] Confirm output shape matches spec
- [x] Confirm inference time logged
- [x] Confirm no memory spike

Deliverable:
Single-frame inference works on device.

---

# Day 4 — Camera Pipeline

## Builder
- [x] Add AVFoundation preview
- [x] Capture frame buffer
- [x] Convert to 640x640 (letterbox)
- [x] Feed frame into model
- [x] Log per-frame inference time

## Debugger
- [x] Confirm preview stable
- [x] Confirm no main-thread blocking
- [x] Confirm stable inference loop

Deliverable:
Live camera → YOLO inference running.

---

# Day 5 — Decode + NMS

## ML_Vision
- [x] Provide reference decode logic
- [x] Clarify bbox format (xywh or xyxy)
- [x] Clarify confidence + class structure
- [x] Provide recommended thresholds

## Builder
- [x] Implement decode
- [x] Implement confidence filtering
- [x] Implement NMS
- [x] Print detection count per frame

## Debugger
- [x] Confirm boxes appear reasonable
- [x] Confirm no explosion in detection count
- [x] Measure decode time

Deliverable:
Valid bounding boxes computed from model output.

---

# Day 6 — Bounding Box Overlay

## Builder
- [x] Map detection coordinates to preview layer
- [x] Handle aspectFill scaling
- [x] Handle device orientation
- [x] Draw bounding boxes in real time while holding and rotating the phone.
- [x] When the phone is rotated, the captured image must remain horizontal. For example, if the phone is rotated 90 degrees clockwise, the image must be rotated 90 degrees counterclockwise to maintain its original horizontal state.

## Debugger
- [x] Confirm boxes align with objects
- [x] Confirm no flipped or mirrored boxes
- [x] Confirm overlay stable across rotation

Deliverable:
Real-time bounding box overlay on iPhone 11.

---

# Day 7 — Stabilization & Baseline

## Debugger
- [x] Measure model load time
- [x] Measure inference time (mean + p95)
- [x] Measure FPS
- [x] Record memory usage

## Architect
- [x] Review pipeline structure
- [x] Freeze Phase 1 architecture
- [x] Define Phase 2 (Segmentation integration plan)

Deliverable:
Stable YOLO detection pipeline.
Performance baseline recorded.
Ready for next 7-day cycle. -->

---

<!--
# Phase 2 — MobileSAM Real-Time Segmentation Integration

Objective:
Integrate MobileSAM instance segmentation on top of YOLO without breaking Phase 1 geometry or scheduling contract.

Principle:
Reuse Phase 1 pipeline. Add modules without rewriting it.

Must Reuse:
CanonicalFrame
FrameGeometry
LetterboxTransform

------------------------------------------------------------
Phase 2 — Detailed 7-Day Plan
------------------------------------------------------------

Design Rule:
Only assign tasks to agents when required.
Follow agent order:
Architect → ML_Vision → Builder → Debugger

------------------------------------------------------------
Day 1 — Architecture Lock & Integration Contract
------------------------------------------------------------

## Architect
# - [ ] Define segmentation pipeline insertion point (post-NMS)
# - [ ] Freeze geometry reuse contract (NO duplicate transforms)
# - [ ] Define bbox→prompt format (box normalization + coordinate space)
# - [ ] Define encoder/decoder split API + fallback (monolithic allowed)
# - [ ] Define threading model (background queues; no capture blocking)
# - [ ] Define bbox-only fallback policy

- [x] **Lock segmentation insertion point**
  - Post-NMS
  - Pre-overlay rendering
  - Input: `[Detection]`
  - Output: Optional segmentation masks aligned to same coordinate space

- [x] **Freeze geometry contract**
  - Detection outputs remain in **original pixel coordinate space**
  - No duplicate transforms allowed
  - No additional scaling inside Segmentation layer
  - Geometry reuse is mandatory

- [x] **Define bbox prompt format**
  - Format: `[x_min, y_min, x_max, y_max]`
  - Coordinate space: **original image pixel space**
  - Normalization performed **only inside MobileSAM wrapper**
  - No external normalization allowed

- [x] **Define SegmentationPipeline API**
  - `encode(pixelBuffer)`
  - `decode(embedding, bbox)`
  - Encoder/decoder split preferred
  - Fallback to monolithic model allowed
  - Phase 1 must not depend on segmentation encoder

- [x] **Define threading model**
  - Detection queue unchanged
  - Segmentation runs on separate background queue
  - Capture session thread must never be blocked
  - No synchronous segmentation calls from camera pipeline

- [x] **Define fallback policy**
  - If segmentation fails → return bbox only
  - No crash
  - No UI freeze
  - Phase 1 behavior remains identical

Deliverable:
Approved Phase 2 integration diagram + contracts.
No Phase 1 code modified.

------------------------------------------------------------
Day 2 — MobileSAM Model Preparation
------------------------------------------------------------

## ML_Vision
- [x] Convert MobileSAM to CoreML (encoder/decoder split preferred)
- [x] Provide input/output tensor names + shapes
- [x] Provide embedding shape + mask output semantics
- [x] Provide preprocessing spec (color space, normalization)
- [x] Benchmark single-image latency (CPU/GPU) — iPhone 11 实测（见 model_plan.md）

## Builder
- [x] Day 2 无 Builder 任务（按计划无需操作）

## Debugger
- [x] Confirm model loads on device
- [x] Confirm no memory spike during load

Deliverable:
Working MobileSAM CoreML models load on iPhone 11 with documented I/O.

------------------------------------------------------------
Day 3 — Encoder Integration (Low Frequency)
------------------------------------------------------------

## Builder
- [x] Insert Encoder module after detection (non-blocking)
- [x] Run encoder every N frames (default 12)
- [x] Cache embeddings with TTL
- [x] Log embedding latency + cache hit rate

## Debugger
- [x] Confirm embedding TTL works
- [x] Confirm no main-thread blocking
- [x] Confirm stable FPS (bbox-only path) not regressed

Deliverable:
Embedding generation working and cached.

------------------------------------------------------------
Day 4 — Decoder Integration (Medium Frequency)
------------------------------------------------------------

## Builder
- [x] Build PromptBuilder (bbox → SAM prompt)
- [x] Run decoder every N frames (default 6)
- [x] Implement mask TTL (default 800ms)
- [x] Log decode time + mask refresh rate

## Debugger
- [x] Confirm mask aligns with bbox (same geometry chain)
- [x] Confirm no jitter across frames
- [x] Confirm fallback to bbox-only works when decoder stalls

Deliverable:
BBox → Mask working pipeline.

------------------------------------------------------------
Day 5 — Mask Renderer
------------------------------------------------------------

## Builder
- [x] Add mask overlay layer
- [x] Match geometry to preview layer (reuse Phase 1 mapping)
- [x] Add alpha blending + color palette
- [x] Handle rotation/mirror (reuse Phase 1 logic)

## Debugger
- [x] Confirm mask not flipped
- [x] Confirm alignment across orientation
- [x] Confirm preview smooth

Deliverable:
Real-time mask overlay at 2–5Hz.

------------------------------------------------------------
Day 6 — Temporal Manager
------------------------------------------------------------

## Architect
- [x] Define primary-object selection strategy (top-1 + hysteresis)
- [x] Define bbox drift threshold (re-seg trigger)
- [x] Define cache invalidation triggers (geometry change / TTL)

## Builder
- [x] Implement TTL system for embedding + mask
- [x] Implement drift detection & re-seg trigger
- [x] Implement priority refresh for primary object

## Debugger
- [x] Stress test motion & rotation
- [x] Confirm mask stability during object motion
- [x] Confirm fallback triggers correctly

Deliverable:
Stable segmentation loop with caching.

------------------------------------------------------------
Day 7 — Stabilization & Phase Freeze
------------------------------------------------------------

## Debugger
- [x] Measure encoder latency (mean + p95)  → mean **857 ms** / p95 **933 ms**（稳态 n=13，除冷启动 2941 ms）
- [x] Measure decoder latency (mean + p95)  → mean **61 ms** / p95 **69 ms**（稳态 n≈39，除冷启动 1488 ms）
- [x] Measure mask refresh rate  → **~1.5 Hz**（静止 1.3–1.5 Hz，light-drift 冲到 2.85 Hz）
- [x] Measure FPS (bbox + mask)  → **2.7–2.9 FPS**（encoder 触发帧低谷 2.14）
- [x] Record memory usage  → **244–320 MB**，峰値 **339 MB**（双 SAM 模型加载时）

## Architect
- [x] Freeze Phase 2 architecture
- [x] Document integration contracts
- [x] Define Phase 3 entry points

Deliverable:
Stable Detection + Segmentation pipeline.
Ready for Phase 3.
-->

---

<!--
# Phase 3 — User-Triggered Segmentation (Tap-to-Segment)

Objective:
Allow segmentation driven by user interaction (tap / region select),
without depending on YOLO detection output.

Principle:
Reuse Phase 1/2 geometry chain and SAM encoder/decoder.
Add interaction layer on top; never break detection pipeline.

Must Reuse:
CanonicalFrame
FrameGeometry
LetterboxTransform
SAMEncoder / SAMDecoder (from Phase 2)

Phase 3 优化入口点（来自 Phase 2 Architect 冻结裁决）：
1. ~~ANE 对齐告警消解~~ → 🔧 **重新界定为「模型加载卫生 model-load hygiene」**（Architect 2026-08-11，architect_output.md §13.1.5）
   原目标「消除对齐告警」**正式作废** —— 隔离实验证明告警来自 `MobileSAM_PromptMaskDecoder` 而非 encoder，该入口点从未消除过任何一条告警。
   **milfix encoder 保留、冻结不动（§9.5 FINAL）；作废的是立项理由，不是部署。**
   新定义：(a) 不加载用不到的模型（box decoder 惰性化 D-5、`testMobileSAMLoad()` 移除或限 DEBUG）；
         (b) 冷加载成本不落在用户第一次交互上（warmup 归属 D-1/D-2）；(c) 订正误归因注释（D-4）。
         观测指标（**非目标**）：`Invalid layer … 64 bytes` 的分阶段计数。
   「真·ANE 对齐」议题降级为观测项、Phase 3 内不立项（point decoder 已强制 `.cpuAndGPU`，问题在当前配置下不可达）；重开三条件见 §13.1.5
2. 降编码分辨率（收益最大，Day 3–4 AB 验证）
3. Embedding 缓存复用强化（最安全，Day 5–6 调参）

------------------------------------------------------------
Phase 3 — Detailed 7-Day Plan
------------------------------------------------------------

Design Rule:
Only assign tasks to agents when required.
Follow agent order:
Architect → ML_Vision → Builder → Debugger

------------------------------------------------------------
Day 1 — Architecture Lock & Interaction Contract
------------------------------------------------------------

## Architect
- [x] **定义 Tap-to-Segment 模式插入点**
  - 模式：与 Phase 2 YOLO 模式并存（UI 切换），不得信报 Phase 2 流水线
  - 输入：用户点击坐标（preview 层坐标）
  - 输出：封装到 Canonical 像素空间的 point prompt → SAM 流水线
  - 交互入口：点击层居于 MaskRenderer 之上，不干扰 Phase 2 渲染层

- [x] **固化 Tap 坐标变换契约**
  - Tap 点来源：UIKit preview layer 坐标系（CGPoint in view bounds）
  - 反变换路径：preview view 坐标 → captureDevicePointConverted → Canonical px → orientation/mirror 修正 → clamp
  - 必须复用 Phase 1/2 `FrameGeometry`，绝不引入独立坐标变换
  - 越界处理：tap 反变后落在 Canonical 边界外时，裁剪到边界内（不排除 tap）

- [x] **定义 Multi-Mask 管理模型**
  - 支持同时活跃最多 **N = 3** 个 mask 实例（初期，防内存炸表）
  - 每个实例独立拥有：point prompt + embedding snapshot + mask + TTL
  - Primary 实例：最近一次 tap 将成为当前 primary，旧实例降为 secondary
  - 混除策略：超过 N 个 tap 时，删除最老的实例（FIFO）
  - 清空操作：双击任意已分割区域清除所有 mask

- [x] **定义 Encoder 触发逻辑（Tap 模式 vs Phase 2 自动模式）**
  - Tap 模式：用户点击后立即触发 encode（若当前 embedding 有效且尚未达 TTL，可复用）
  - Phase 2 模式下 TemporalManager 静默（mode check 保护，不触发 encode）
  - 两种模式共享同一个 encoderQueue，不得并发执行

Deliverable:
Phase 3 交互契约锁定。坐标变换路径、多实例管理模型、Encoder 触发逻辑均已定义，可进入 Day 2。

## ML_Vision
- [x] **确认 SAM 点提示格式（Point Prompt I/O 规范）**
  - `point_coords: [1, 2, 2]`（固定 2 点；Architect spec 预测 `[1,1,2]` → 实测修正为 `[1,2,2]`）
  - `point_labels: [1, 2]`
  - 单点 Tap 构造：`[[tap_x, tap_y], [0.0, 0.0]]` + labels `[1.0, -1.0]`（前景 + padding）
  - 坐标单位：SAM 像素坐标（0~1023），模型内部 ÷ 1024，外部不得预归一化
  - Multi-mask output：单一 mask（`low_res_masks [1,1,256,256]`，`iou_predictions [1,1]`），非多候选；export 时 index 0 fixed
  - 详见 shared/model_plan.md §A

- [x] **准备 ANE 对齐修复方案（Phase 3 优化入口点 1）**
  - 根因：Encoder neck `LayerNorm2d.mean(1,keepdim=True)` 产生 `[1,1,64,64]` Float16 中间张量（C=1，2 bytes，不满足 ANE 64-byte 对齐要求）
  - 修复：Float32 重导出 → `models/MobileSAM_ImageEncoder_fp32.mlpackage`（28 MB）
  - 脚本：`shared/export_encoder_fp32_ane_fix.py`（已可运行）
  - 功能等价确认：I/O 接口不变，权重来自同一 checkpoint，0 个 `_fp16` op
  - 详见 shared/model_plan.md §B

Deliverable:
SAM 点提示 I/O 规范文档 + ANE 修复方案，准备交付 Builder。
✅ 已完成（2026-07-19）：model_plan.md + MobileSAM_ImageEncoder_fp32.mlpackage + export_encoder_fp32_ane_fix.py

------------------------------------------------------------
Day 2 — TouchHandler & Canonical Coordinate Transform
------------------------------------------------------------

## Builder
- [x] **实现 TouchHandler**
  - 在 preview 父视图添加 `UITapGestureRecognizer`
  - 对齐 `AVCaptureVideoPreviewLayer` 坐标系（需区分 aspectFill 的裁剪偏移）
  - 封装为 `TouchHandler.swift`，导出单一回调 `onTap(canonicalPoint: CGPoint)`
  - 双击手势绑定到清除所有 mask 操作
  → 实现：`Interaction/TouchHandler.swift`（single-tap + double-tap，`singleTap.require(toFail: doubleTap)`），回调 `onTap`/`onClearAll`

- [x] **实现 preview 坐标 → Canonical 反变换**
  - 进又：view bounds 坐标 → `AVCaptureVideoPreviewLayer.captureDevicePointConverted(fromLayerPoint:)` 或手动逆算 aspectFill 裁剪偏移
  - 再反变：Letterbox 逆变换（复用 `FrameGeometry.invertLetterbox(point:)`，若尚未实现该方法则新建）
  - 边界处理：反变后超出 Canonical 范围时 `clamp(0, Wc-1)` / `clamp(0, Hc-1)`
  - 方向适配：反变必须夹带当前 `orientation` 和 `isMirrored`，与 Phase 1/2 同一逻辑
  → 实现：新建 `Interaction/FrameGeometry.swift` `invertViewPoint(_:viewSize:previewLayer:)`（按 architect §2.1 命名，等价 invertLetterbox 职责）。旋转已由 AVCaptureConnection 烘焙进 origW/origH（与 mask 管线一致），此处仅逆 aspectFill + mirror + clamp

- [x] **集成到主流程并验证 Log**
  - `TouchHandler` 触发后打印：`[TAP] preview=(x,y) canonical=(cx,cy) orientation=N mirrored=B`
  - 无实际 SAM 调用，仅验证坐标变换精度
  → 实现：`CameraPreview.makeUIView` 安装 TouchHandler；`CameraManager.currentFrameGeometry()` 提供几何快照，`handleTap`/`handleClearAllTapMasks` 仅在 `.tapToSegment` 生效。Log 含 view/normalized/canonical/orientation/mirrored。无 SAM 调用

- [x] **ANE 修复模型封层集成（MIL LayerNorm Fusion，§B.4）**
  - ~~Phase 3 Day 2 集成的 fp32 方案（`MobileSAM_ImageEncoder_fp32.mlpackage`）经 Debugger 真机测量证明无效~~
    - fp32 encoder 比 fp16 慢 +32%（mean 1131ms），ANE 告警未消除（反增至 6 条）
  - **Builder 修复（2026-07-21）**：改用 MIL LayerNorm 融合路径（model_plan §B.4）
    - 新脚本 `shared/export_encoder_fp16_milfix.py`：monkeypatch `tiny_vit_sam.LayerNorm2d.forward`
      为 `F.layer_norm` + transpose，coremltools 直接映射为 MIL `layer_norm` op
    - 静态验证：`reduce_mean=0`（原 4），`layer_norm=22`（原 20），模型大小 14.1 MB（与 fp16 相同）
    - `MobileSAM_ImageEncoder_fp16_milfix.mlpackage` 已拷入 Xcode，target membership 已注册
    - `SAMEncoder.swift` 加载优先级：milfix > fp32 > fp16
  → **BUILD SUCCEEDED**（2026-07-21 xcodebuild iphonesimulator iPhone 11，含 fp16_milfix.mlmodelc）
  → ANE 告警消除需 Debugger 真机验证

## Debugger
- [x] **验证 tap 坐标正确性**
  - 展示一个已知尺寸标准物（如设备外壳或用品），分别在 `rot=0` / `rot=90` / `rot=270` 方向点击目标中心
  - 从 Log 读取 canonical 坐标，与预期像素坐标对毕（允许 ±5% 误差）
  - 确认饕像情况下反变结果不产生左右对称错误
  - 确认 Phase 2 检测/分割 pipeline 未因 TouchHandler 发生性能回退

Deliverable:坐标变换链路正确，tap canonical 误差 < 5%。ANE 修复模型已集成并编译通过。

------------------------------------------------------------
Day 3 — Point-Based PromptBuilder + 单次点击分割
------------------------------------------------------------

## ML_Vision
- [x] **评估降编码分辨率方案（Phase 3 优化入口点 2）**
  - 展开役离实验：分别以 1024×1024 / 768×768 / 512×512 运行分割，评估 mask 覆盖精度（呈交 Architect + Builder 评审）
  - 交付评估报告：每个分辨率的延迟预测 + 主观精度评分，供 Architect 裁决 Phase 3 是否切换分辨率
  → ✅ 完成（2026-07-22）：768 预测 ~555ms（×1.54）IoU=0.581；512 预测 ~261ms（×3.28）IoU=0.592
  → 待 Architect Day 4 裁决是否切换 768。详见 `shared/resolution_eval_report.md` + `model_plan.md §C`

## Builder
- [x] **实现 PromptBuilder（点模式）**
  - 新建 `PointPromptBuilder.swift`，接收 `canonicalPoint: CGPoint` 和 `imageSize: CGSize`
  - 输出：`point_coords` MLMultiArray[1, 1, 2] + `point_labels` MLMultiArray[1, 1]（label=1 前景）
  - 内部归一化：`point_x / imageSize.width`，`point_y / imageSize.height`（相对 1024×1024）
  - **不得在 PromptBuilder 外部典面归一化坐标**（契约）

- [x] **更新 SAMDecoder 支持点提示**
  - 在现有 `decode(embedding:bbox:)` 基础上新增重载方法 `decode(embedding:point:)`
  - 内部将点提示封装为 `point_coords` / `point_labels` tensor，传入模型
  - 旧的 `decode(embedding:bbox:)` 保持不变（Phase 2 兼容）
  - 打印：`[SEG][TAP] decode latency: %.2f ms iou_pred: %.3f`

- [x] **单次点击分割闭环验证（无 TemporalManager）**
  - Tap 触发 → encode（如有延用 embedding，直接跳到 decode） → decode(point) → 显示 mask
  - 不需要 TemporalManager，不需要 YOLO，採用一次性直通流程
  - 连接到 MaskRenderer，在原击中位置显示 mask

## Debugger
- [x] 确认点提示输入 tensor 形状与 ML_Vision 规范匹配（decode 成功运行 + iou_pred 有效，隐式验证）
- [x] 确认第一次 tap 即触发 encode，后续相同位置点击可复用 embedding（日志 `reuse cached embedding` ✅）
- [x] 确认 `iou_pred` 输出合理（常规前景点击 > 0.5）（实测 0.512 / 0.899 / 0.530，全部 ≥ 0.5 ✅）
  → **附：Builder 同步修复两个 Bug（2026-07-23）**
  → Bug 1: `maskImage` 每帧被 detectionPipeline 清空（tapToSegment 模式下 mask 一闪而过）→ 已修
  → Bug 2: tapToSegment 下 YOLO bbox 仍显示（违反 Architect §1.1）→ 已修
  → BUILD SUCCEEDED ✅

Deliverable:
单次 tap 即可触发分割闭环。mask 显示在点击位置。ANE 对齐修复和降分辨率方案评估均就绪。
✅ 完成（2026-07-23）

------------------------------------------------------------
Day 4 — 端到端点击分割 Pipeline（需实测数据）
------------------------------------------------------------

## Architect
- [x] **裁决降分辨率方案（基于 ML_Vision Day 3 评估报告）**
  - 若 768×768 齐观精度可接受：允许 Builder Day 4 展开 AB 切换
  - 若 512×512 精度损失过大：限定到 768 或保持 1024
  - 裁决结果封层进 Phase 3 冻结参数表
  → ✅ 已完成（2026-07-23）：768 **有条件批准** Builder AB 测试；512 拒绝。详见 architect_output.md §8

## Builder
- [x] **展开分辨率 AB 测试（如 Architect 批准）**
  - 新建配置项 `encoderInputSize: Int`（默认 1024，可调整为 768 或 512）
  - Encoder 预处理 resize 逻辑适配该配置项
  - 采样同场景 encoder latency + 主观分割质量，结果写入 builder_progress.md
  → ✅ 完成（2026-07-23）：768 milfix 模型导出封层（`export_encoder_fp16_milfix_768.py`，0 missing/unexpected keys，[1,3,768,768]→[1,256,48,48]，14.1 MB）；`SAMConfiguration.encoderInputSize`（默认 1024，C-1）+ SAMEncoder resize 适配 + 48→64 双线性上采样桥接（C-3）+ ContentView Res toggle + `[TAP][AB] encoder stats` 延迟埋点。512 按 Architect §8.4 拒绝。真机 AB 采样（含人工评分 C-5）属 Debugger Day 4。详见 builder_progress.md

- [x] **连接完整 Tap 分割流水线（含 TemporalManager）**
  - `onTap(canonicalPoint:)` → `TemporalManager.registerTapInstance(point:, frameGeometry:)`
  - TemporalManager 判断是否复用现有 embedding（TTL + geometry 匹配）
  - 若可复用：直接 `scheduleDecoder(point:)`；若不可复用：先 `scheduleEncoder()` 再 decode
  - 分割结果封装为 `TapInstance` 并存入实例池
  - 已有实例当天击同一区域：更新现有实例的 prompt 并重新 decode（不新建）
  → ✅ 完成（2026-07-23）：`handleTap` embedding 复用决策接入 `temporal.isEmbeddingValid`（TTL 8000ms）+ `temporal.geometryChanged`（几何签名含 inputSize），`canReuse = ttlValid && !geometryChanged` → decode-only 快路径 / encode+decode 慢路径；点坐标恒在 1024 空间（C-2/§C.4）；tap→mask 端到端延迟日志（decode-only vs encode+decode）。注：`TapInstance`/实例池（多实例承载）按 Architect §6 分工属 Day 5 TapInstanceManager，Day 4 打通单实例端到端 + 复用决策主链路。详见 builder_progress.md

- [x] **完善错误处理 + Fallback**
  - Tap 期间 encoder 忙磁：显示“装载中”指示动画，等待 encoder 完成后自动 decode
  - Decode 失败 / iou_pred < 0.1：显示进度提示而非增加无意义 mask
  - 所有失败分支必须复位 `isEncoding` / `isDecoding` flag
  → ✅ 完成（2026-07-23）：`@Published tapProcessing` 加载态 + `scheduleTapBusyTimeout()`（3s 安全兜底）；iou_pred<0.1 丢弃 mask 并清加载态；encode 失败统一 `resetEncodingAndTapUI()`（复位 isEncoding + 清 tapProcessing），decode 失败统一 `finishTapProcessing()`；双击 clearAll 同时清 mask/加载态/tap 锚点；保留后台 GPU abort 门控。详见 builder_progress.md

## Debugger
- [x] 真机采样 tap-to-mask 端到端延迟（从手指离开屏幕到 mask 显现）
  → ✅ 完成（2026-07-26）：真机 session 全量采样 24 次 tap，tap→mask **429.0–1030.7 ms**（中位 ≈620 ms），全部走 decode-only 快路径；其中 decode 分量 48.9–75.6 ms（mean ≈61 ms）
  → ⚠️ 遗留缺口 1：本 session **零个 encode+decode 样本**（唯一一次 parked tap 被丢弃），Builder Day 4 条目要求的「decode-only vs encode+decode」双路径对比延迟仍缺，需补测
  → ⚠️ 遗留缺口 2：`CameraManager.swift:761` 的计时终点落在**主线程发布之前**，不满足「到 mask 显现」的定义，当前数字**低估真实延迟**；Day 5 Architect 审批延迟数据时需相应修正验收指标口径
- [x] 确认 mask 在 tap 点附近正确显示（允许分割 mask 不一定包含 tap 点，但必须在语义上关联目标区域）
  → ✅ 完成（2026-08-02）：**通过 —— 良品率 44/50 = 88%**（基线为 2026-07-26 混合场景的 15/24 = 62%）。几何对齐部分此前已验证通过（2026-07-26：坐标链算术逐位核对无误，截图 IMG_0680 / 0682 / 0684 / 0686 / 0690 mask 与目标精确贴合），本轮补齐的是当时欠缺的**语义质量人工评分**
  → 测量协议（详见 `shared/phase3_day4_test_checklist.md` §3）：5 类场景 × 每类 10 次（2 轮 × 5 目标），全程屏幕录制，屏幕角标 `#N` 与日志 tap 序号一一对应，逐帧人工评分（覆盖率/溢出率 → 类型 → 1–5 分）。评分表 `shared/S1-score.png` … `S5-score.png`；录像 `shared/S1-1/1-2 … S5-1/5-2.MP4`（注意 S4 与 S5 两组**录像文件名内容对调**，评分表本身正确）
  → 分场景结果：S1 近处大目标 + 干净背景 **10/10 (100%)**；S2 近处大 + 杂乱 **8/10 (80%)**（失败：显示器底座，银灰压米色桌 + 白墙）；S3 远处小 + 干净 **8/10 (80%)**（失败：电脑盖，银色压白大理石）；S4 远处小 + 杂乱 **10/10 (100%)**；S5 低对比/同色系 **8/10 (80%)**（失败：充电插头，白压白泡沫垫，2 分；另书签、番茄酱包中度溢出）
  → 核心发现（2×2 设计的裁决结果）：**局部对比度是唯一的失败预测因子；目标大小与背景杂乱程度均被排除** —— S4 同时具备「远处小目标」与「杂乱背景」两个原本假设的困难因素却拿到 100%，而只占其中一个因素的 S2 / S3 反而各失一项；S2/S3/S5 的全部 6 个失败项无一例外都是低对比物体；专门隔离对比度这一变量的 S5 产出最差单项（2 分）
  → 失败机制（已由录像逐帧确认）：低对比使 SAM 的 logit 场变平，同时引发两件事 ——(a) 物体无法与背景分离，`logit>0` 二值化把整片同色区域纳入；(b) 边缘过渡区 logit 在 0 附近抖动，点亮密度落在渗流阈值附近，4-连通洪水填充将散点连成巨块。视觉特征是**实心主体 + 点阵/虚线状边缘**。S5 充电插头一例最典型：插头、数据线、书签、酱包连同整张白色泡沫垫被一并覆盖
  → ⚠️ 方法学说明（勾选前提下仍须如实保留）：本轮**无 `[TAP#N] candidates:` 日志**（Xcode/CoreDevice 连接不稳，本轮全程未取得候选日志）。因此对每个失败项**无法区分「SAM 三个候选全都不佳」与「存在良好候选但选择规则挑错」**；失败机制的判定依据是录像的形态学特征，不是候选数据。后续定向复采必须带日志
  → ⚠️ 保留项（不阻塞勾选）：低对比场景约 12% 失败率仍在。改进路径分三步，均属 Day 5 之后 ——(1) Builder 为三个候选各计算 SAM stability score（`logit>+δ` 与 `logit>−δ` 的面积比），**只打日志不参与决策**，行为不变以免污染 88% 基线；(2) 定向复采：针对 6 个失败物体 + 若干满分物体，取得 stability 在好/坏两类上的分布；(3) ML_Vision 依据该分布裁决行动方案 —— stability 参与候选选择，或对低 stability 的 tap 触发裁剪重编码（即「detect→crop→segment」思路，对低对比小目标对症）
- [x] 对毕 1024 vs 低分辨率的 encoder latency 实测差异（如已开启 AB）
  → ✅ 完成（2026-07-26）：上一轮 warm 实测 1024 mean=970.6 / p95=1006 ms，768 mean=1036 / p95=1126 ms —— **768 反而慢 ~65 ms**，未达「≥150 ms 降幅」门控 → **正式拒绝 768，封层 1024**
  → ⚠️ 新增数据（本轮）：background refresh 14 次实测 **573–755 ms，mean ≈648 ms**，比上轮记录快约 33%；两次测量热态条件不同，**Day 5 封层裁决口径需先标注测量条件（冷/热态、并发负载）再定**
- [x] 确认 embedding 复用逻辑正确：同一 geometry 下多次 tap 仅 encode 一次
  → ✅ 完成（2026-07-26）：TAP #2–#24 **连续 23 次**全部打印 `[TAP] reuse cached embedding (ttlValid=Y geoChanged=N)`，全 session **零条** `[TAP] encode + decode`；encode 仅发生于 warmup(1) + background refresh(14)。TTL / 几何签名判定链 `CameraManager.swift:493-508` + `TemporalManager.swift:59,185-187` 行为符合设计
  → ⚠️ 风险备注：复用过于激进 —— TTL=8000 ms、主动刷新阈值 5000 ms，意味着 tap 可能命中最长 **8 秒前**的 embedding；手持抖动/移动下会造成 tap 点与 embedding 语义错位，**可能是过分割的隐藏贡献者**。当前日志不打 cache age，无法证实也无法排除 → 建议补 cache age 埋点
- [x] 确认 Phase 2 YOLO 路径没有因 Tap 分割 pipeline 出现 FPS 回退
  → ✅ 完成（2026-08-02）：**通过 —— tapToSegment 不但没有拖慢 YOLO，反而比 Phase 2 的 segmentation 基线更快**。证据基础：6 个真机 session、约 40 个统计窗口，日志存于 `shared/perf_session_20260801*.log`
  → 测量协议（详见 `shared/phase3_day4_test_checklist.md` §2）：四组 A=detectionOnly（参照）/ B=segmentation（Phase 2 基线）/ C=tapToSegment 空闲 / D=tapToSegment + 每 5 秒点击；交错采样 B,C,B,D…，每次切模式后丢弃第一行统计（跨切换的混合窗口）；主判据是 `Inference time stats` 行的 `mean`（只括住 `model.prediction`，与调用频率无关），**不用 FPS**（tap 模式下 YOLO 降频到每 3 帧一次，FPS 口径不可比）；全程开启 “Perf Quiet Log” 静默模式，消除日志量随模式变化带来的混淆
  → 实测差值：**B→D**（n=50 同配置，夹心估计，三对）**−6.01% / −7.25% / −2.24%**，三对全为负、方向一致；**B→C**（四对）**−3.2% / −2.8% / −11.3% / −2.66%**，全为负；p95：B→D 夹心 −7.4% / −4.4% / −7.9%，B→C +5.40%，均在 +15% 阈值内；Post 中位全部为负；内存峰值 286→257 MB 递减、无单调增长（排除泄漏）；场景一致性：最终轮检出框数 2.00–2.10（±5%）
  → 机制解释（结论自洽的关键）：segmentation 模式下 SAM 以约 1.2 Hz **持续**做 encode+decode，与 YOLO 争资源；tapToSegment 只在需要时工作 —— 空闲时仅每 5 秒一次 background embedding refresh（故 C 最快），有点击时叠加每次约 60 ms 的解码（故 D 介于 C 与 B 之间，仍快于 B）
  → ⚠️ 方法学保留 1（夹心估计属事后采用）：夹心估计（取 D 前后两个 B 的均值作基线，以抵消线性漂移）是**在看到数据之后才采用的**。采用理由是先验的 —— 基线本身在漂（某轮 B 在三次测量间波动达 ±26%），漂移幅度超过待检出的 ±10% 效应。但为稳妥起见已交叉验证：**即便只用最初预先约定的相邻配对，三对 B→D 也全部落在 ±10% 内、结论一致通过**，故结论不依赖于估计量的更换
  → ⚠️ 方法学保留 2（排除了一个越界数据点）：`shared/perf_session_20260801_BD.log`（2026-08-01，`n=100` 配置）的 B→D 相邻 p95 为 **+15.58%**，超出 +15% 阈值，已排除。排除理由有三条且均独立于其结果 ——(a) 该轮 `inferenceStatsWindow=100`，p95 取第 95 位，与 n=50 取第 48 位不是同一估计量，跨配置合并无效；(b) 该轮 D 窗口后侧无 B，无法做夹心；(c) 该轮跑的是旧的 `maxPlausibleLogit=50` 哨兵，会误杀约 13% 的 tap。该轮视为已被后续同配置轮次取代
  → 过程中发现并已修复的两个缺陷（便于追溯）：(1) `MaskRenderer.maxPlausibleLogit` 由 50 提到 500 —— 真机实测健康 tap 的 |logit| 可达 65，旧阈值误杀 4/30 次点击（13%），用户感知为“点了没反应”；修复后 29+13 次 tap 零误杀。(2) 新增 “Perf Quiet Log” 静默开关 + `=== MODE SWITCH → … ===` 分隔日志 + `inferenceStatsWindow` 100→50，分别消除了日志量混淆项、使窗口归属可判定、把每行统计的等待从 42 秒降到 21 秒

Deliverable:
Tap-to-mask 端到端全链路工作。分辨率裁决封层。延迟数据有实测基准。

------------------------------------------------------------
Day 5 — Multi-Instance Selection
------------------------------------------------------------

## Architect
- [x] **审批 Day 4 分辨率 AB 裁决结果，封层 Phase 3 encoder 分辨率**
  → ✅ 完成（2026-08-09）：**封层 1024 为 Phase 3 唯一 encoder 分辨率；正式撤销 768**（§8.3 批准作废，约束 C-1…C-6 随之失效）；512 维持 Phase 3 拒绝。落入 architect_output §8.7 门控表的撤销分支。详见 architect_output.md §9
  → 依据：同 session 配对实测 1024 warm mean=970.6 / p95=1006.1 ms（n=3）vs 768 warm mean=1036.0 / p95=1126.5 ms（n=4）——768 不仅无降幅，反而慢 65.4 ms，与门控线（≥150 ms 降幅）相差 215 ms，远超噪声范围，无需补采。ML_Vision Day 3 的 ×1.54 预测建立在 Mac CPU 线性缩放假设上，被 A13 ANE 实测证伪（两份权重同为 14.1 MB，ANE 图编译/调度开销占比高；768 还额外背 C-3 的 48→64 上采样）
  → 架构层教训（长期约定）：**跨硬件后端（Mac CPU → iPhone ANE）的比例外推不得作为架构裁决的充分依据**，只能作为立项做 AB 的理由；涉及 ANE 的性能提案门控必须绑定目标设备真机配对实测
  → ⚠️ 口径缺陷 1（如实保留）：**768 从未做过人工评分**，C-5/C-6 要求的评分表未产出（Session A 的 50 次评分全部在 1024 上完成）。因撤销分支的两个条件是**析取**关系，延迟维度单独越线已足以触发撤销，故评分维度缺失**不影响本次结论**。但 §8.7 表述在未来复用时须按修订规则理解：**批准取合取（全部满足），撤销取析取（任一触发即可）**；已单独触发撤销时不要求补齐其余维度，反之仅有部分维度指向批准时不得封层（缺失维度视为未通过）。见 architect_output §9.3
  → ⚠️ 口径缺陷 2（如实保留）：**970.6 ms 是 AB 配对比较量，不是绝对性能指标**。它采自 2026-07-23 的 AB 交替会话（同进程内先后加载 1024/768 两份 encoder、反复切换触发 counter 重置），仅在该会话内部有效（配对设计保证内部效度，故相对结论稳健），无跨会话外部效度。本轮 2026-08-01/02 单模型稳态会话 background refresh 实测 573–755 ms、mean ≈648 ms（n=14），快约 33%。**Phase 3 引用 encoder 绝对延迟一律用 648 ms**；后续任何延迟数字须同时标注冷/热态判定、同会话是否发生模型切换、并发负载、n。见 architect_output §9.3
  → 资产处置：768 mlpackage 归档不删除（作为「ANE 非线性缩放」的实证物）；`encoderInputSize` 保留但冻结为常量语义，Phase 3 内不得设为 1024 以外值；48→64 桥接允许作为死代码保留，主路径不得调用。Phase 4 若重启低分辨率议题，入口条件为先取得目标设备 ANE 逐层 profile，仅 FLOPs 下降不构成立项理由
  → **Phase 3 冻结参数表最终封层**见 architect_output.md §9.5（取代 §8.5）：encoder 1024 / milfix fp16 单模型 / embedding [1,256,64,64] / point prompt 恒 1024 空间 / encoder 基线 648 ms / decode 基线 61 ms / N=3 —— 均标记 FINAL；Embedding TTL 与 Mask TTL **未封层**，留 Day 6 复议
- [x] **审批 Day 4 延迟数据，确认 tap-to-mask 延迟展示策略（是否需要“裁剪”或“进度条”等待 UI）**
  → ✅ 完成（2026-08-09）：**裁决 —— 不引入进度条，不引入「裁剪 / 分段等待」UI**。保留 `TapLoadingIndicator`（脉冲圆环）作为唯一等待反馈，语义明确为「已收到你的点击」的确认而非进度指示；`tapProcessing` 保持布尔量，**不得**扩展为进度值。详见 architect_output.md §10
  → 依据：24 次全量采样 tap→mask 429–1030 ms（中位 ≈620 ms），其中 decode 计算仅 48.9–75.6 ms（mean ≈61 ms）；差值 400–960 ms 已定位为 `handleTap` 派发到 `videoQueue`（同时是 AVCaptureVideoDataOutput 的 delegate 队列，正跑 YOLO 单帧 400–670 ms）造成的排队，区间高度吻合。**快路径 620 ms 中位延迟里约 90% 是调度排队、仅约 10% 是真实计算** → 用 UI 粉饰会把架构缺陷转译成永久 UI 债，并植入「这功能本来就慢」的错误心智模型。**先修调度，再谈 UI**
  → **要求 Builder（Day 5 内，接口级要求见 architect_output §10.4）：** (A) **快路径去队列化** —— 快路径不需要 camera pixel buffer，只需 letterbox 几何 + 已缓存 embedding；改为在 stateLock 快照内完成 TTL/geometry 判定后**直达 decoderQueue**，绕开 videoQueue（慢路径不变，仍走 videoQueue 取 pixelBuffer；§4.2 触发语义不变，仅改判定发生在哪个队列；不得新建第三个队列）。目标：修正口径下快路径 p95 ≤ 200 ms。(B) **计时口径修正** —— `e2eMs` 终点移到主线程 mask 实际提交渲染之后，日志区分快/慢路径并补 `cacheAge`。(C) **超时分级 + 失败可见化** —— 快路径 1.5 s / 慢路径 12 s（覆盖实测 8605 ms 上界）；**静默清除加载指示是被禁止的行为**，任何超时都必须给可见失败信号；冷启动的正解不是加长超时，而是 `.tapToSegment` 进入模式时即触发 warmup（现仅 `.segmentation` 调用 `warmupSegmentationIfPossible()`），warmup 期间 tap 要么显式 park 后执行、要么明确拒绝，**不得沉默丢弃**
  → ⚠️ 三个口径缺陷已纳入裁决（不美化）：(G-1) `e2eMs` 终点落在主线程发布之前 → 429–1030 ms 是**下界估计**，真实感知延迟更高，故本裁决方向更强而非更弱，但不得据此判「已达标」；(G-2) 本轮**零个 encode+decode 慢路径样本** → 慢路径**无基线，UI 策略不予封层**，持续脉冲仅为临时规则；(G-3) encoder 冷启动 1283–8605 ms 与 3 s 兜底不相容 → 8.6 s 那次会先超时清掉指示，用户看到「点了没反应」，这是当前 tap 路径最严重的用户可见失败模式，属逻辑不一致而非调参问题
  → ⚠️ 保留项 R1–R5（详见 architect_output §10.5）：R1 慢路径 Day 7 须补 ≥10 次样本（强制 TTL 失效或旋转触发 geometryChanged）后再定慢路径 UI；R2 embedding 复用过激（TTL 8000 ms，可能命中 8 秒前 embedding，疑似过分割隐藏贡献者），要求补 `cacheAge` 日志、**只打日志不改行为**，TTL 数值留 Day 6；R3 低对比 12% 失败率在 3 实例并存下「至少一个坏 mask」概率升至约 32%，但 **Day 5 期间禁止在多实例逻辑里夹带任何 mask 质量过滤**（两个变量同时动会毁掉归因）；R4 decode 请求密集化后按「同一实例新 tap 取代旧请求」处理，不做无界排队；R5 全部数据仅来自 iPhone 11/A13，绝对值不得外推到 A16/A17
  → Day 7 复审门控（architect_output §10.6）：修正口径下快路径 p95 ≤200 ms → 维持不设进度条、脉冲可简化为一次性波纹；200–500 ms → 维持并保留持续脉冲；>500 ms 且已排除排队因素 → 此时才重开进度语义 UI 议题
- [x] **裁决 per-instance mask TTL 与显示策略的规格冲突（Builder 上交）**
  → ✅ 完成（2026-08-09）：**时间型 mask TTL（2000 ms）整体撤销** —— 不保留为内部状态、不延长到其它数值、不改作他用。已解码成功的 tap mask **常驻**，直到被事件移除。`maskTimestamp` 保留但降级为**遥测时间戳**（仅用于 `maskAgeMs` 日志，不得进入显示/过滤/有效性判定路径）。详见 architect_output.md §11
  → **mask 清除条件正式定义为事件驱动穷尽列表**（§3.2.1 C1–C6）：C1 双击 `clearAll()` / C2 第 4 次 tap 的 FIFO 淘汰 / C3 退出 `.tapToSegment` / C4 几何签名变化清空（坐标空间失效，属正确性保护）/ C5 同实例新请求取代旧请求（替换非清除）/ C6 长按删除单实例（Day 6）。**时间流逝不是清除条件**；表外原因导致 mask 消失一律按 bug 处理
  → 根因：原 §3.1 写的是「**复用** Phase 2 mask TTL」，而两条管线性质不同 —— Phase 2 mask 有**刷新循环**（过期代价是短暂空窗），Phase 3 tap mask 是**用户显式产物且无任何重建机制**（过期代价是**永久丢失**）。TTL 是「等下一次刷新」的兜底机制，搬到一次性产物上是**类别错误而非参数选错**，故正解是撤销机制而不是调大数值。代价不对称：用户**已拥有**零成本清除手段（双击），却**没有**恢复手段 ⇒ **清除权归用户，失效权归事件**
  → 原 2000 ms 三条理由逐条复核（不做简单删除）：**R-a 避免陈旧误导** —— 部分成立但**工具选错**，真实失效事实是坐标空间变化（已由 C4 精确捕获），墙钟既误杀（机位不动时 mask 仍正确却被删）又漏杀（1.9 s 内甩动相机已错却未到期）；**R-b 限制内存** —— **不成立**，N=3 + FIFO 已硬封顶，3 实例 mask+alpha 合计 **< 1 MB**，与 +30 MB 预算差 30 倍以上；**R-c 与 Phase 2 参数一致性** —— **不成立**，一致性本身不是理由
  → 新增的决定性输入：(1) **用户真机实测反馈**「双击之后才消失，很适合测试阶段」（❗措辞边界如实记录：说的是适合**测试阶段**，非永久正确，故 Phase 4 不封层）；(2) **Session A 评分方法论依赖 mask 常驻** —— phase3_day4_test_checklist §3.4 标准流程「点击→等 1 s →停 2 s →双击清除→再停 2 s」要求单次 mask 至少存活 ≈3 s；2000 ms TTL 会在录屏拍清楚前删掉它，且使「双击清除」这个**受控动作失去意义**（无法区分双击生效与 TTL 到期）。已建立的 44/50 = 88% 基线、后续 stability 定向复采与 A/B 对照均复用同一流程，**改变 mask 存续语义 = 使基线不可比**；(3) 相对 Day 4「tap mask 常驻」属**用户可见行为回退**
  → **长期约定 A-1（TTL 适用性判据）**：时间型 TTL **只允许**作用于「过期后可被自动重建」的资源；对「过期后无法重建的用户产物」禁用时间型 TTL，只能用**事件型失效**。自检问法：「它过期之后，谁会把它变回来？」—— 答不出具体机制就不该有 TTL。**约定 A-2**：「复用 Phase N 的某参数」不构成设计理由，跨模式搬运参数前必须先确认两边资源生命周期性质相同（与 §9.2「跨硬件后端比例外推不构成裁决依据」并列，同属「相似≠可迁移」）
  → **术语纪律**：「Mask TTL」一词在本项目**停用**；项目中「TTL」自此**专指 embedding TTL（8000 ms）**，单一所指，不会出现两个语义相近而数值不同的 TTL。`maskTimestamp` 的派生量命名必须是**年龄**（`maskAgeMs`）而非**判定式**（禁用 `isMaskValid` / `maskTTL` / `maskExpiry`）—— 休眠谓词是地雷，早晚有人把它接到 `drawableInstances()` 上，命名即防呆。注：R2（embedding 复用过激）**不受本裁决影响、仍留 Day 6**；R2 管「mask 是用多旧的 embedding 解出来的」，本裁决管「mask 生成后能存活多久」，两者不得互引为据；mask 常驻不会加剧 R2（风险在解码那一刻已全部兑现）
  → **Builder 处置**：B-1 **Day 5 内不做任何代码改动**（既有实现与本裁决行为一致，此刻改代码会作废 Debugger 即将验收的构建）；B-2 Day 6 移除 `maskTTL` 派生字段与 `isMaskValid(now:)` 判定式 API（纯命名/死代码清理，非行为改动）；B-3 Day 6 日志可附 `maskAgeMs`（与 `cacheAge` 并列但含义不同），**只打日志不参与决策**；B-4 §10.5 R3 禁令持续，本裁决**不得**被当作引入「低质量 mask 自动消失」的授权
  → ✅ **Builder 上交决策而非擅自决定的做法记录在案**：per-instance mask 是否自动消失属**用户可见行为**，变更权归 Architect / 用户；实现方「保留状态但不接显示」是本次的正确处置，作为后续遇规格冲突时的推荐处理方式
  → ⚠️ **对 Day 5 Debugger 五条验收的影响（仅说明前提变化，不代勾选、不改写条目）**：第 **1** 条（多 mask 同时显示）—— **前提被修复**，原字面规格下该动作**物理上不可能通过**（依次点三下必超 2 s），现在方成为可执行且有意义的检验；若仍出现「先点的 mask 自己消失」，不再有合规解释（唯一合法自动全清是 C4，须伴随构图/旋转变化）。第 **4** 条（3 实例内存 < +30 MB）—— **测量前提被改变**：mask 现常驻、不再被 TTL 自动回收，这条从「顺带看看」变成**真正的泄漏检验**（原设计下 TTL 会周期性释放 mask、可能掩盖泄漏）。第 **2** 条 FIFO 不受影响且**归因变干净**（FIFO 成为「最老的消失」的唯一解释）；第 **3** 条不受影响（归 embedding TTL，仅需把 burst 控制在 8 s 内，否则 re-encode 是正常行为不是 bug）；第 **5** 条不受影响。五条的勾选权、措辞与判定标准全部归 Debugger
  → ⚠️ 保留项 R6–R9（architect_output §11.9）：**R6** 场景运动导致 mask 与物体脱节无机制捕获（C4 只盖几何变化）—— **Phase 3 不处理，明确接受**（Session A 流程要求手机架好不动；正确解法是事件型漂移检测而非墙钟，属新功能需用户显式立项），Day 7 冻结时如实记为已知限制；**R7** 前后台切换 / session 中断恢复后是否清空实例池未定义，Day 6 待议，**Day 5 不新增行为**；**R8** Phase 4 若重开「mask 自动消失」需三条同时满足（客观失效度量非墙钟 + 先建立 tap mask 刷新循环使代价降为短暂空窗 + 评分方法论同步修订并重建基线）；**R9** 本裁决未改动任何 mask **质量**逻辑，Day 4 的 88% 基线**保持可比**

- [x] **复议 §3.3/§3.4 呈现规格（用户主诉「精确度比 Day 4 下降很多」的根因裁决）**
  → ✅ 完成（2026-08-11）：**判定为纯呈现回退，几何零变更**（Debugger 逐行核对：`buildTapAlpha` 只有一行转调 `buildAlpha`，候选选择整段 `MaskRenderer.swift:402-483` 与全部常量与 Day 4 逐项一致 ⇒ 同帧同 embedding 下二值 alpha 逐位相同；2026-08-11 录像 `shared/simu_record_0811.MP4` 目视确认 mask 覆盖的是**完整物体**而非碎片）。详见 architect_output.md **§12**，§3.3/§3.4 已整节改写
  → **裁决 Q1：三色板 `systemBlue/systemGreen/systemOrange` 正式作废**，Phase 3 全部实例统一使用青 **(0,217,255)**，primary alpha **0.60** / secondary **0.40**（恢复与 Day 4 逐位一致的填充，作为 44/50 = 88% 评分基线的可比性锚点）
  → **裁决 Q2（核心）：建立呈现三层职责模型，并裁决「L1 可见性不得由色相承担」** —— L1 可见性（「屏幕上存在一块被选中的区域」）由**双色轮廓描边**承载、**绝不允许依赖不可控输入**；L2 归属由 alpha 分级 + tap 锚点承载；L3 区分由描边样式 + 锚点编号承载。**原规格把 L1 押在「填充色 vs 物体本色的对比」上，而物体本色是不可控输入 ⇒ 撞色不是概率事件而是遍历事件**
  → **主成因判定：撞色。** 2026-08-11 录像三例全中 —— 蓝保温杯涂 systemBlue(H=211°)、绿键盘涂 systemGreen(H=135°)、橙文件架涂 systemOrange(H=35°)，三块 mask **同时隐身**，而它们的几何**完全正确**。这是「为什么是**很多**而不是**有点**」的唯一解释。**责任在规格不在实现，Builder 是照做的**
  → 次成因：亮度下降 2.7 倍（青 Y=0.569 → systemBlue Y=0.211，且方向反了 —— 从「提亮」变成「压暗」）；白色 1pt 描边缺席（原定 Day 6）
  → **裁决 Q3：给出填充色约束原则 C-1…C-7（而不只是色值）** —— C-1 稀有色带（允许 H ∈ [160°,200°] 青–蓝绿 / [280°,330°] 品红；**禁用带** [0°,60°] 红橙棕木肤、[60°,90°] 黄纸暖光、[90°,155°] 绿植绿制品、[200°,260°] 蓝塑料牛仔天空屏幕蓝 —— 一次性判掉三个 system 色）；C-2 相对亮度 **Y ≥ 0.45**（判掉 systemGreen 0.428 / systemOrange 0.425 —— 它们看着够亮，实际都在线下）；C-3 HSV **S ≥ 0.85 且 V ≥ 0.90**（人造信号特征）；C-4 L1 不得由色相承担（描边强制，含 secondary）；C-5 **描边必须在屏幕坐标系生成、线宽以 pt 计，不得烧进 256×256 位图**（否则线宽被放大 4–8× 且锁死 D-13 的 GPU 缩放重构）；C-6 描边须可单点关闭（保留与 Day 4 的单变量对照能力）；**C-7 换色准入程序 —— 任何换色必须 (a) 通过 C-1/C-2/C-3 算术检验并把数值写进表 + (b) 在至少一个同色系真实物体上做撞色测试，两项缺一不得合入**（这条是防复发机制，没有它下一次换色照样撞）
  → **裁决 Q4：描边由「白色 1pt」升级为「双色轮廓」**（primary 外层 2.0 pt 近黑 @0.85 → 内层 1.5 pt 近白 @0.95；secondary 减档 1.5/1.0 pt @0.70），**且 secondary 也必须有**。理由：单色白在白墙上消失、单色黑在暗物体上消失，单色描边只是把押注从填充色换到描边色；「深外圈 + 浅内线」保证背景亮度落在量程任一端都至少有一条线有足够阶跃 ⇒ **把 L1 从「与不可控输入比对比度」改造成「呈现元素自带对比度」**
  → **裁决 Q5：描边任务从 Day 6 提前至 Day 5，定级 P0**（tasks.md Day 5 Builder 已增列，Day 6 原条目改写为「已提前」+ 保留「点击已有实例提升为 primary」）。依据：三条成因里前两条只能靠「选个更好的色」缓解，**只有描边能从机制上消除**；且它不触碰 `buildAlpha` 任何决策代码，**不威胁 §11.9 R3 禁令与 88% 基线的几何可比性**
  → **裁决 Q6：按实例配色 Phase 3 内撤销**，`TapInstance.color` 字段保留但语义降级为「呈现槽位」。理由：C-1 允许色带**装不下三个可区分色相** ⇒ 只要坚持按实例配色，就必然有色相落进禁用带；且三色半透明重叠会混出**第四种颜色**反增歧义
  → **裁决 Q7（R6 显式复议，用户要求）：裁决维持「Phase 3 不做自动失效」，但定级从「已接受的功能限制」上调为「已确认的呈现风险」。** 如实记录我在 §11.9 的分析缺口：低估的不是 R6 的正确性风险，而是它的**感知放大效应** —— Day 4 的组合（同屏恒 1 张 + 每次 tap 整体替换）使陈旧度被下一次 tap 自动截断，Day 5 的组合（N=3 + FIFO + 永不过期）**第一次让陈旧度变成无上界量**；三张漂走的 mask 比一张准确的 mask 更容易被读成「这个功能不准」。**我当时只单独评估了 TTL 撤销，没有评估「TTL 撤销 × 多实例」的乘积效应**。裁决仍维持的四条理由：(1) 不改变 §11 核心论证 —— mask 仍是「过期后无法重建的用户产物」，墙钟仍然既误杀又漏杀，**漂移的正确捕获器是事件型漂移检测不是时钟**；(2) 测量协议中漂移为零（手机架好不动），88% 基线不受污染，且本次主诉的实证材料已定性为呈现问题；(3) 引入自动失效会撞上 §11.9 R8 的三条件，一条都不满足；(4) 属新功能，需用户显式立项。**呈现侧缓解（不违反 §11、Day 5 即生效）**：M1 secondary 呈现权重下压（0.40 填充 + 减档描边），语义从「我现在断言这就是物体」降为「这里曾经被选中过」——**旧 mask 不再冒充当前断言**；M2 清除手段更易达（C1 双击已有 + C6 长按删除 Day 6），**清除权仍完全归用户**。**Phase 4 入口点新增「mask 重锚定（re-anchor）」** —— 事件型漂移检测 + 同 `canonicalPoint` 在新帧重新 decode；值得记的是：**一旦有了 re-anchor 循环，§11.4 A-1 的自检问法「它过期之后谁把它变回来？」第一次有了答案**，那时时间型刷新才重新合法。顺序不可颠倒：先有刷新循环，才谈得上过期
  → **长期约定 A-3（新增，与 §9.2 / §11.4 A-1、A-2 并列）**：**任何叠加在用户内容之上的呈现元素，其「可见性」不得依赖它与用户内容的对比度。** 用户内容是不可控输入，把可见性押在不可控输入上等价于把正确性押在运气上——失败不是概率事件是遍历事件。可见性必须由呈现元素**自带的对比**保证（双色描边/阴影/外发光/图案），色相与 alpha 只能承担次级职责。**自检问法：「如果用户把镜头对准一块和我这个颜色一模一样的东西，会发生什么？」** 答案若是「看不见了」，规格就是错的
  → ⏸ **明确不在本次范围、不裁决的两条**：**A-1 候选选择规则**（「取最小」已退化为恒取 ch0 = SAM token 1 = 最细子部件，10/10）—— §22.5 的并排录像**决定性实验已执行**，2026-08-11 录像显示 mask 覆盖完整物体 ⇒ **A-1 不是本次主诉的成因、不得作为主诉的修复路径**；它仍是独立且真实的质量议题，但裁决**依赖 stability 定向复采数据**（§25.3-6，仍缺），在数据到位前动候选规则会重蹈「从太大荡到太小」的覆辙，归口 ML_Vision、Day 6 之后。**A-4 双击失败窗口 ~300 ms 口径**（真实 e2e ≈710–830 ms 而非 461–478 ms）—— 属延迟口径议题且与「是否保留双击清除」这一交互权衡绑定（双击是 §3.2.1 C1 的唯一入口、也是 §11「清除权归用户」的实现），**挂账 Day 7 冻结前一并裁**
  → ⚠️ 保留项 R10–R14（architect_output §12.11）：**R10** 单一青色的撞色风险只是变小未归零（青色物体仍会撞），这正是 C-4 描边必须落地的原因 —— **描边就位后，撞色的后果从「隐身」降级为「填充看不清但轮廓仍在」**；若真机上仍出现「青物体上完全找不到 mask」，是描边实现有问题不是色板问题。**R11** 与 Day 4 的逐位可比性只在「描边关闭」时严格成立（描边是加法，但可能轻微改变边界的主观判读），Day 7 严格对照须用 C-6 开关做单变量回退。**R12** 描边实现路径不指定（矢量轮廓或全分辨率上下文内描边均可），**唯一不可接受的是烧进 256×256 位图**。**R13** M1 是缓解不是修复，相机大幅移动后三张 mask 仍全部锚定旧帧，Day 7 冻结须如实记为已知限制。**R14** 「撞色是主成因」这一归因目前只有一次录像的三个物体为证（色度检验 C-1/C-2/C-3 是确定性算术不受样本量影响，但归因不是）；修订后须复录同场景确认，**若换回青色 + 描边后主观质量未回到 Day 4 水平，则归因需重开**

- [x] **Day 5 追加三：三项裁决（Debugger 第 5 条重新表述 / 三槽色板追认 + §3.3.2 勘误 / A-1 方向收敛）**
  → ✅ 完成（2026-08-11）：详见 architect_output.md **§13**（§3.3.2 C-1 已勘误并新增 C-1a/C-8、§3.3.3 整节改写、§3.4 填充色行与载体清单更新、§9.5 冻结表 2 行改为 PROVISIONAL、§12.8 A-1 行补交叉引用）
  → **裁决一（Debugger 第 5 条）：重新表述，保持未勾选。** 原表述「确认对齐告警已消失或显著减少」有两处结构性缺陷 —— (1) **把观测指标写成了验收目标**：告警计数同时受「某模型被加载几次」（3→6 正是 `testMobileSAMLoad()` + `SAMDecoder.init` 各一次）、构建配置、SDK 版本支配，计数下降既可能是「问题解决」也可能是「触发路径没跑到」，**一个既不能证明成功也不能证明失败的判据不是判据**；(2) **追错了组件**：孤立加载对照（milfix encoder ⇒ 0 条 / `MobileSAM_PromptMaskDecoder` ⇒ 3 条）证明告警从来不是 encoder 发的。新表述 =「**`.tapToSegment` 路径不加载用不到的模型**」，告警计数降级为副产品指标。判定标准 P-1/P-2/P-3 + S-1 + G-1/G-2 与所需数据清单见 §13.1.4 与该条目下的追加说明
  → **误归因的成因值得记（它会复发）**：注释写在**被改动的组件旁边**（ModelLoader 里 encoder 换用那一段），于是「改了 X，日志里有 Y」被读成「X 产生 Y」。切开它的是**隔离实验**——让两个模型在时间上分开加载，看告警跟着谁走
  → **裁决一（续）：「Phase 3 优化入口点 1：ANE 对齐修复」重新界定，不撤销、不移交 Phase 4。** 更名为「**模型加载卫生（model-load hygiene）**」= (a) 不加载用不到的模型 (b) 冷加载成本不落在用户第一次交互上 (c) 订正误归因注释；告警计数为**观测指标非目标**（Phase 3 入口点列表第 1 项已同步改写）。**不撤销的三条理由**：(1) 撤销会丢掉教训 —— 这是项目里最有价值的反面案例（基于代码注释、未经隔离实验的归因驱动了一整轮工作）；(2) 撤销**有被误读为「回滚 milfix」的实际风险** —— milfix 已在 §9.5 冻结为 FINAL，Day 4/5 的全部延迟基线（648 ms / 970.6 ms 配对量）与 44/50=88% 评分基线**都是在 milfix 上测的**，换回原 encoder 会让冻结表与全部基线一次性失效，而**没有任何证据表明原 encoder 更好** ⇒ **作废理由，保留部署**；(3) 该入口点下真实存在的问题（box decoder 惰性化）就在 Phase 3 内、就在 Day 6 可修范围内，没有理由推到 Phase 4。**「真·ANE 对齐」议题降级为观测项、不立项**（实质风险实测为零 + point decoder 已强制 `.cpuAndGPU` ⇒ 当前配置下不可达）；重开三条件（合取）：明确的延迟收益目标 + 目标设备真机配对实测（§9.2）+ 数值哨兵与 logits 量级作为**验收门**而非事后检查
  → **裁决二（色板）：三槽色板 slot0 青 (0,217,255) / slot1 水青 (0,255,242) / slot2 春青 (0,255,170) 予以追认，但状态为 🟡 PROVISIONAL 而非 FINAL**，§12.1 Q6「按实例配色 Phase 3 撤销」正式被取代。**这不是反复横跳**：撤销按实例配色的推理前提是「色相在承担 L1 可见性」，**双色描边落地后 L1 已由描边独立承担**，色相退回 §3.3.1 的 L3 ⇒ 前提消失、结论随之失效；而**原则（L1 不得由色相承担 / A-3）一个字未变** —— 本次恰恰是它的正确应用：正因为 L1 被描边接管了，色相才被允许回来。判据是「**L1 的承载者是谁**」，不是「用几种颜色」。真机复录已确认呈现回退解决（R14 的复录要求已满足）
  → **裁决二（续）：§3.3.2 的两处错误经 Architect 独立复算全部确认成立，已修订。** **勘误 1** —— C-1 的品红备选带 H∈[280°,330°] 在 C-2∩C-3 下是**空集**：该带内满足 S≥0.85 且 V≥0.90 的颜色最高只到 **Y=0.2988**（H=300°, S=0.85, V=1.00），距门槛 0.45 差 0.15 且无论如何取值都够不到。原因是物理的 —— 亮度系数 R 0.2126 / G 0.7152 / B 0.0722，**亮度几乎全在绿通道上**，一个绿分量被压到近 0 的色相不可能同时高饱和高亮度 ⇒ **删除，且写明删除理由**（留着会让下一个换色的人以为有两个色带可选）。**勘误 2** —— 首选带被 C-2 截断在 **H=194.78°**（S=1/V=1 时 Y 恰穿过 0.45；195° 已是 0.4459，200° 只剩 0.3597），可行弧段是 **H∈[160°,194.78°]，宽 34.8°**。补充一处 Builder 未提的细节：把 S 放到 C-3 下限 0.85 可延到 H≈197.6°，但**这 2.8° 不计为可用余量**（用掉饱和度换亮度与 C-3「人造信号特征」的立意直接冲突）。⇒ **我上一版「允许色带装不下三个可区分色相」的判断经验算不成立，予以撤回**：34.8° 弧内、slot 0 钉死 188.94° 时最优三元组 ΔE00_min=17.4，发布的 (160°,176.94°,188.94°)=**17.1**（Architect 独立复算：三对 18.0/35.3/17.1），高于「类别可辨识」常用的 10。**装不下的是「散布在整个色环上的三个」**（旧色板 ΔE00_min=50.2，正是那种散布把两个色相拖进禁用带）—— **我把「色相间距不够」和「色相位置越界」混成了一件事，真实约束只作用在位置上**
  → **裁决二（续）：新增两条约束。** **C-1a 边界余量** —— 允许带端点**不是安全线**，推荐取色区间 H∈[165°,190°]（两端各留 ≥5°）；slot 2 = 160.00° 是**已批准的例外**（距绿色禁用带仅 5°、零余量），放行条件是 C-7(b) 必须专门覆盖薄荷/春绿实物。**C-8 混合封闭性** —— 多色板必须在 `compositeLayers` 的 source-over 叠放下封闭，穷举全部叠放组合（N=3 ⇒ 12 种）**每一种重叠产物自身都要过 C-1/C-2/C-3**。这条是从上一版被限定的理由 2 里救出来的：「三色会混出第四色」对本色板**不成立**（三色 R≡0 ⇒ 任意凸组合仍 R=0 ⇒ S 恒 1.000；Architect 独立复算 12 种组合 H∈[163.57°,186.48°]、S=1.000、V∈[0.930,1.000]、Y∈[0.5997,0.7706]，**12/12 通过**），旧色板失败的原因不是「三色」而是**从来没人检验过混合产物**（蓝+橙掉出整个约束集）
  → ⚠️ **裁决二削弱项 W-1（secondary 之间可辨识度不足，已确认非推测）**：前述 ΔE00 是**不透明色块**值；在 7 种代表性背景上实算**合成后**的最小两两 ΔE00 —— primary(0.60α) **10.2–13.5**（全部背景 ≥10，可辨识），**secondary(0.40α) 只有 6.8–10.7，多数背景低于 10**（Architect 独立复算，与 Builder 结论一致）。**架构含义：色相是 L3 的载体之一，但不是充分载体** —— primary/secondary 的区分由描边分档承担、那是 L2，**两个 secondary 彼此之间目前只有色相一个 L3 载体**。⇒ **Day 6 的「tap 锚点常驻标记 + 实例编号」从增强项升格为 L3 的必需载体，定级 P1**（该条目已存在于 Day 6 Builder 小节，本轮**不改写 Builder 小节**，定级于 Day 6 排期时生效）；在它落地前 **L3 处于已知欠载状态**，Day 7 冻结须如实记录。**两条便宜解法明确不接受**：① 把 secondary alpha 提回 0.60（削弱 L2 归属分级，违背 §12.7.4 的 M1 缓解）；② 把三个色相拉更开（弧段只有 34.8°，17.1 已近该弧内上界 17.4，**没有空间可挖**）⇒ **只能加新载体，不能调参**
  → ⚠️ **裁决二削弱项 W-2（C-7 (b) 未执行 = 程序违例，如实记录）**：C-7 字面要求「两项缺一不得合入」，而代码**已合入**，本色板**目前只通过了 (a)**。不做追溯性否决（用户显式要求 + 描边兜底已把撞色后果从「隐身」降为「填充看不清但轮廓仍在」+ slot 0 保住 Day 4 逐位一致），但**违例必须记录，否则 C-7 这条防复发机制第一次被援引就被架空**。处置三条：(1) 色板状态标记 🟡 **PROVISIONAL**（§3.3.3、§9.5 均已标注），**C-7 (b) 完成前不得再做任何换色，也不得把相关行改成 FINAL**；(2) **补测是阻塞项**：三个测试物（青色 / 水青蓝绿 / 薄荷或春绿，**slot 2 = 160° 风险最高**）逐槽位各测一个，**外加一次同屏三实例**判「能否看出这三块来自三次不同点击」（直接检验 W-1）；(3) **失败判据写死避免口径漂移** —— 失败 = **连轮廓都找不到**；「填充看不清但轮廓仍在」**不算失败**（那正是 R10 预期的残余形态，也正是双色描边的设计目的），若真出现连轮廓都找不到，问题在描边实现不在色板
  → **裁决三（A-1 方向收敛）：采纳 ML_Vision 的推荐（model_plan.md §D.5）—— 不导出 token 0；「导出 4 候选」「token 0 作 fallback」「退回 Day 3 纯 token 0」三条路径全部正式关闭；A-1 方向收敛为「在现有 token 1–3 内改进选择」。** 四条依据互相独立，其中 **D.3 是决定性的**：A4 vs 现状 20 次里 19 次选择完全相同（唯一「差异」两者面积/bbox/fill 全同），FB vs 现状 **20/20 相同**（fallback 从未触发）—— 机制清楚：token 0 面积在 19/20 上 ≥ token 1 而规则取最小 ⇒ **永远选不上**；导出 4 候选却要付出模型重导出 + decoder 输出契约变更 + 全链路 shape 变更的成本，**换一个已被观测为 ≤1/20 的效应**。D.2：token 0 ↔ token 2 中位 mask IoU ≈0.94（vs token 1 的 0.70、token 3 的 0.44）⇒ 落在「部件」档、**不提供新粒度层级**。D.4：四 token 的 iou_pred 最大者 token 2 占 10/20、token 1 占 5、token 3 占 4、**token 0 占 0**。D.4 续：纯 token 0 替换在**退化提示**上明显更差（点裸大理石溢出 8.4%→**26%**、裸泡沫 16%→**25%**）⇒ **「Day 3 那时候更准」这个一直悬着的假说有了反证**，与 §12「主诉是呈现问题」构成**两条独立证据同时指向「不要回滚几何/候选集」**
  → **裁决三（续）方法学说明（这让 n=20 仍然可用）：证明一个改动*无效*所需的证据强度低于证明它*有效*** —— 前者的效应量上界已被直接观测（19/20 或 20/20 相同 ⇒ 效应 ≤1/20），后者要估计一个未知正效应。⇒ D.7 的 R-T3（n=20、集中 4 场景）**足以关闭一个方向，不足以开启一个方向，更不足以支撑任何阈值设定**；本裁决严格停在前者
  → ❌ **裁决三（续）明确不批准「改为取 iou_pred 最大」**，即使 §D.6 的数据表面上支持它（token 2 在 10/20 上 iou_pred 最高、常 iou 与 stability 双高，而现行规则几乎从不选它）。三条理由：(1) **iou_pred 是未经验证的代理量** —— iou_pred/stability/fill 全为模型自评量且共线（stab~iou 0.945、iou~fill 0.924），**三个共线的自评量提供的是一个维度的信息不是三个**，「哪个 token 更接近用户意图」**从未被测量**（R-T2）；(2) 它是**选择式**改动，会在几乎每个样本上改变输出（token 1 只在 5/20 上 iou_pred 最高）⇒ 直接撞 §11.9 R3 与 R9，88% 基线整体不可比；(3) Debugger §19.4「stability 只能否决不能选择、高 stability ≠ 好 mask」的结论因共线性顺延到 iou_pred
  → **裁决三（续）动选择规则所需的证据 E-1…E-4（合取；沿用 §9.3 口径：批准取合取、撤销取析取）**：**E-1 人工评分配对数据（最高优先）** —— 盲评强制选择（候选顺序随机化、**不显示** iou/stability/面积），标注「哪个最接近我点这一下时想选的东西」；**设计必须配对** —— 离线重放两条规则，**只对「两规则选择不同」的 discordant 子集做人工评分**（McNemar），判据 b>c 且在 b+c≥10 上显著。这是**唯一**能把模型自评量与用户意图挂钩的数据，且配对 + 只评 discordant 把样本需求压到极低（D.3 显示多数样本同选），**成本不是借口**。**E-2 设备端定向复采** —— 必须能**复现**那三个灾难性失败（S3 电脑盖 / S5 插头 / S2 显示器底座）并同时导出四候选日志；理由：R-T1 显示重建帧**没能复现**设备端灾难性失败（四 token 面积均在 0.3%–3%），**如果卖点是「修掉灾难性溢出」而它在离线介质上根本不出现，离线重放无法证明改动有效**。**E-3 规则改动的形式必须是「否决式」而非「选择式」** —— 须能写成「先用阈值否决部分候选，再在剩余候选里**沿用现行的『面积最小且 ≤cap60』**」；三个好处：承接 §19.4、改动单调收缩可单开关回退、**在「无候选被否决」的样本上输出逐位不变 ⇒ 88% 基线可比性由构造保证而非事后辩护**。**E-4 副作用指标必须一并采集** —— 退化提示（点裸大理石/裸泡沫等裸表面）的溢出率**不得上升**；规则一旦「变松」第一个坏的就是裸表面点击，**没有这一项就会重演「从太大荡到太小再荡回太大」**。⚠️ **未满足 E-1…E-4 之前，`MaskRenderer` 中任何参与选择的代码不得改动（§11.9 R3 禁令原样有效）**；ML_Vision 可继续采数据、离线重放、打日志
  → ⚠️ **裁决三（续）三条保留项原样纳入，并逐条给出它限制了什么**：**R-T1**（重建帧未复现设备端灾难性失败）⇒「不导出 token 0」对**整体趋势**成立、对**那三个失败案例**结论强度低，本裁决只关闭候选集方向、**不声称三个失败案例已被解释**；**R-T2**（无人工评分配对）⇒「token 2 更好」严格来说**未被测量**，§D.6 只能作为 E-1 的**假设来源**、不能作为改规则的依据；**R-T3**（n=20、集中 4 场景）⇒ 不得用于任何**阈值级**结论（cap 值 / stability 门槛 / 面积比 K / Δiou δ 一律不得由本批数据定）
  → 🔀 **裁决三（续）必须命名的分叉点（它决定 A-1 到底是什么问题）**：若 E-2 的设备端定向复采显示灾难性失败时**四个 token 全部不可用**（都是「整张桌子」或都是碎片），则问题**既不在选择规则也不在候选集，而在提示/embedding 本身**（低对比度下 logit 场变平 —— 与 Day 4 归因「局部对比度是唯一失败预测因子」同源）⇒ 那时 **A-1 应被重新界定为「提示质量问题」**，方向变成 detect→crop→re-encode / 多点提示。**顺序不可颠倒：先确认『存在一个好候选』，再谈『怎么把它选出来』** —— 现行 A-1 的全部提法都默认了前者，而那个默认**至今未被验证**
  → **长期约定 A-4（新增，与 §9.2 / A-1 / A-2 / A-3 并列）：验收标准必须写成对*系统行为*的断言，不得写成对*日志文本*的断言。** 日志计数受「触发路径跑没跑到」「加载了几次」「构建配置」共同支配，既不能证明成功也不能证明失败；把它当目标会导致**优化指标而不是优化系统**。**推论（归因纪律）：因果归因必须由隔离实验建立** —— 代码注释不是证据，时间上的相邻不是证据，「我改了 X 之后 Y 变了」也不是证据，除非 X 能被单独开关。**自检问法：「如果这条日志明天被人删掉，我的验收还成立吗？」** 答「不成立」⇒ 验收标准写错了对象
  → **长期约定 A-5（新增）：当一个决策规则的全部输入都是模型自评量、且这些自评量彼此共线时，不得在没有外部标注的情况下改变该规则的判据。** 共线的 k 个自评量提供的是**一个**维度的信息，用其中一个替换另一个只是换了个投影方向，不增加任何关于「用户想要什么」的信息。**自检问法：「这个量被验证过与我真正关心的东西相关吗？」** 答不出验证在哪，就说明在优化一个从未被校准过的仪表。与 A-3 的关系：**A-3 管「别把正确性押在你控制不了的量上」，A-5 管「别把正确性押在你没校准过的量上」**
  → ⚠️ 保留项 R15–R19（architect_output §13.4）：**R15** 色板只过 (a)、追认为 PROVISIONAL，slot 2 零余量风险最高，补测未完成前不得再换色；**R16** L3 欠载已确认且**没有调参空间**，唯一解法是 Day 6 锚点编号；**R17** Debugger 第 5 条的新判据尚无数据，须等惰性加载落地后的 Release 真机日志，**保持 [ ]**；**R18** box decoder 冷加载成本是被**移动**不是被消除（按 Builder 实现落在 Phase 2 warmup 排练上，预期不可见但未经真机确认），P-3 要求测量它，另需观察 `setMode` 复位 `decoderWarmupDecodeDone` 带来的多余排练与 `.tapToSegment` 新增的 point 排练路径；**R19** A-1 的三个灾难性失败案例**至今没有被任何介质复现过**（设备端有、重建帧无），在 E-2 复现前任何「修好了失败案例」的声称都无法被验证，且分叉点意味着 A-1 可能根本不是选择规则问题
  → ✅ **本次未改动的部分（可 diff 核对）**：Day 5 / Day 6 Builder 小节、ML_Vision 任何小节、Day 7 全部条目 **一行未动**；Debugger 第 5 条**只改表述并追加裁决说明，未勾选**；§3.4 的 alpha 分级（0.60/0.40）与描边分档、§9/§10/§11 的全部封层值、`buildAlpha` 相关的任何规格 **均未触碰**

## Builder
- [x] **实现 TapInstanceManager**
  - 封装为 `TapInstanceManager.swift`
  - 内部维护 `[TapInstance]` 数组（最多 N=3 个活跃实例）
  - 每个 `TapInstance` 包含：`id: UUID` / `canonicalPoint: CGPoint` / `mask: MLMultiArray?` / `maskTTL: Date` / `color: UIColor`
  - `addInstance(point:)` — 超出 N 时 FIFO 删除最老
  - `updateMask(id:mask:)` — Decoder 完成回调后调用
  - `clearAll()` — 双击手势触发
  - `removeInstance(id:)` — 长按单个 mask 块删除（Phase 3 可选）
  → ✅ 完成（2026-08-09）：新建 `JudgeE2/Interaction/TapInstanceManager.swift`。`TapInstance` 字段按 architect §3.1（`id`/`canonicalPoint`/`createdAt`/`mask`/`maskTimestamp`/`iouPred`/`color`/`isPrimary`），`maskTTL: Date?` 按本表命名提供（= `maskTimestamp` + 2000 ms，常量单点定义）。API：`addInstance(point:requestGen:)`（返回新实例 + FIFO 淘汰者 id）/ `updateMask(id:requestGen:mask:alpha:iouPred:)` / `clearAll()` / `removeInstance(id:)` / `promoteToPrimary(id:)`（§3.2 规则，Day 6 接手势）/ `isRequestCurrent(id:requestGen:)` / `drawableInstances()`
  → 颜色分配：色板 `[systemBlue, systemGreen, systemOrange]`（§3.3），新实例取当前未占用的第一个色；颜色随实例释放，|palette| == N == 3 保证三个活跃实例颜色恒互异，FIFO 淘汰后新实例接手被释放的颜色
  → FIFO 与 TTL 正交：**FIFO 管占位**（`count >= 3` 时按 `createdAt` 淘汰最老，不论是否 primary）；**TTL 管可见性**（per-instance，与 TemporalManager 零共享，§3.1 要求）。⚠️ Day 5 **TTL 已完整维护但未接显示**——2000 ms 会让依次点三下的前两个 mask 在验收过程中自行消失，与 Debugger 验收动作冲突且是相对 Day 4 的行为回退，故留 Architect 裁决（见 builder_progress.md Day 5 未决问题）
  → supersession 语义变更：Day 4 的全局 `isLatestTap(gen)` 会让第 2 次 tap 直接杀掉第 1 个实例的 decode（三 mask 永不可能共存），改为 per-instance `isRequestCurrent`，即 R4 的「同一实例的新 tap 取代旧请求」；全局 `tapGeneration` 保留作 `[TAP#N]` 日志与屏幕计数器的单调序号
  → 线程模型：实例池被手势线程 / decoderQueue / 主线程访问，无单一 owning queue，内部自带 `NSLock`（**不是新建队列**，≤3 元素数组的锁不引入调度跳转）

- [x] **更新 TemporalManager 支持实例池**
  - Tap 接受多个实例并行请求时按顺序入队 decode（不并发）
  - 实例间共享同一 embedding（同一 geometry 内只 encode 一次）
  - 每个实例独立维护自己的 mask TTL
  → ✅ 完成（2026-08-09）：**顺序入队不并发**——`decoderQueue` 本就是 serial，多实例 decode 天然串行，**未为此新增任何机制**（Architect §10.4 A 禁止新建第三个队列，serial 队列已满足要求）
  → **同 geometry 共享同一 embedding、只 encode 一次**：新增 `TemporalManager.tapGeometryChanged(_:)` + `resetTapGeometry()`，带独立的 `tapGeometryLock`，与 Phase 2 的 `lastGeometrySignature` 分开。分开的两条理由：(1) 要求 A 把判定挪到手势线程，而 `geometryChanged` 会改 videoQueue-owned 状态、不能跨线程调；(2) `.segmentation` 与 `.tapToSegment` 互斥运行，拆分在模式内部行为等价并消除两模式共用签名的交叉污染。返回 false ⇒ 整个实例池共用当前 embedding，burst 中第 2/3 次 tap 经 `drainPendingTaps` 搭同一次 encode 的车；返回 true ⇒ 缓存 mask 属失效坐标空间，清空实例池
  → **每实例独立 mask TTL**：状态在 `TapInstance.maskTimestamp`，与 TemporalManager 零共享（§3.1 原文「与 TemporalManager 独立计算，不共享状态」）
  → Phase 2 的 `geometryChanged` / `selectPrimary` / `classifyDrift` / mask cache **一行未动**

- [x] **更新 MaskRenderer 支持多 mask 叠加**
  - 每个有效实例分配独立颜色（来自预设色板：蓝色/\u7ef f色/荀色/橙色…）
  - 多 mask 叠加时采用半透明 alpha 叠加，不相互遮挡（primary 实例 alpha 0.55，其余 0.35）
  - 空实例列表时回退至「无 mask」显示状态
  → ✅ 完成（2026-08-09）：`renderMask` 拆为三块且**未复制任何决策代码**——`buildAlpha(...)`（原样切出的决策段：阈值、候选选择、60%/85% cap、形状门槛、flood fill、stability、数值哨兵）、`drawTile(...)`（原样切出的纯几何绘制段）、`renderMask(...)`（薄包装，青色字面量保留以保证 Phase 2 / 单实例输出逐字节不变）。多实例入口 `buildTapAlpha(...)` **内部就是同一个 `buildAlpha`**（共享代码而非拷贝）
  → **绘制顺序与透明度**：`drawableInstances()` 按 `createdAt` 升序给出 layers，secondary 在前、primary（最新）最后 ⇒ primary 压最上层（§3.4）；alpha primary **0.55** / secondary **0.35**；重叠区做 source-over 逐像素混合（直通 alpha，float 累加避免三次叠加舍入漂移）而非后写覆盖，故两实例重叠时互相仍可见（「不相互遮挡」）
  → 空实例列表 → `compositeLayers` 返回 nil → `maskImage = nil`，回退「无 mask」显示状态。合成在 **decoderQueue** 完成（≈20 万像素操作），主线程只做一次 `maskImage` 赋值，避免污染要求 B 刚修正的计时窗口。primary 白色 1pt 轮廓属 Day 6，本次未做
  → ⚠️ **保留项 R3 已守住**：候选选择整段（候选构建 + 三道形状门槛 + cap60 主选 + cap85 degraded 回退 + 无候选 faultLog）与 Day 4 做**逐字符 diff → IDENTICAL**；常量清点 `minComponentPx=30`/`minComponentSidePx=3`/`minComponentFill=0.05`/`maxPlausibleLogit=500.0`/`stabilityDelta=1.0`/`cap60`/`cap85`/`iou_pred>=0.1` 全部一致；stability 仅出现在常量定义、结构体存储、计算与日志，**不存在任何以 stability 为条件的分支**。注：`Segmentation/`、`Detection/` 未被 git 跟踪，`git diff` 为空不能作为证据，故用留存基准文本比对

- [x] **Mask 呈现规格修订 + 轮廓描边（🔴 P0，从 Day 6 提前 —— 见 architect_output.md §12.6）**
  - 色板回退：`TapInstanceManager` 的三色板 `[systemBlue, systemGreen, systemOrange]` 作废，**全部实例统一填充青 (0,217,255)**；`TapInstance.color` 字段**保留不删**，语义降级为「呈现槽位」（Phase 4 可能重开按实例配色）
  - alpha：primary **0.60** / secondary **0.40**（primary 恢复与 Day 4 逐位一致，保住 88% 评分基线的可比性）
  - **双色轮廓描边**（L1 可见性的唯一承载者，**primary 与 secondary 都要有**）：primary = 外层 2.0 pt 近黑 @0.85 → 内层 1.5 pt 近白 @0.95；secondary 减档 = 外层 1.5 pt 近黑 @0.70 → 内层 1.0 pt 近白 @0.70
  - ⚠️ **架构约束 C-5（必须在动手前就位，否则 Day 6/7 返工）**：描边**在屏幕坐标系生成、线宽以 pt 定义**；**不得**把描边烧进 256×256 的 alpha 位图 —— 否则线宽会随 mask 一起被放大 4–8×，且与 D-13「让 GPU 直接缩放 256×256」的重构方向不兼容。实现路径不指定（矢量轮廓 marching squares → CAShapeLayer，或全分辨率上下文内描边），Builder 自选
  - ⚠️ **架构约束 C-6**：描边必须能由**单点开关**整体关闭（保留与 Day 4「无描边」条件的单变量对照评分能力）
  - ⚠️ **禁止夹带**：本条**不得触碰** `buildAlpha` 的任何决策代码（阈值 / 候选选择 / cap60 / cap85 / flood fill / stability / 数值哨兵）；§10.5 R3 禁令继续有效。Phase 2 的 `renderMask`（`MaskRenderer.swift:114-140`）**一行不动**，其青色字面量 (0,217,255,153) 与本条的 0.60 属巧合性一致，**不得因此合并两条路径**
  - ⚠️ **换色准入程序 C-7**：今后任何对填充色的变更，须 (a) 通过 C-1 稀有色带 / C-2 Y≥0.45 / C-3 S≥0.85 且 V≥0.90 的算术检验并把数值写进 architect_output §3.3.2 表；(b) 在**至少一个自身即为该色系的真实物体**上做目视撞色测试。**两项缺一不得合入**
  → ✅ 完成（2026-08-11）：色板回退 + alpha 0.60/0.40 + 双色轮廓描边全部落地，BUILD SUCCEEDED（零 warning）。详见 builder_progress.md「Day 5（追加）」
  → **色板**：`TapInstanceManager.fillColor = UIColor(red:0, green:217/255, blue:1, alpha:1)`，三个槽位一律赋此青色；`palette` 数组与槽位分配代码**保留**（Phase 4 重开按实例配色的入口），注释写明其在 Phase 3 内按构造为 no-op。`TapInstance.color` 字段未删，语义降级为「呈现槽位」
  → **alpha**：primary 0.55 → **0.60**、secondary 0.35 → **0.40**（`TapInstanceManager.primaryOpacity/secondaryOpacity`）
  → **描边**：新建 `JudgeE2/Segmentation/MaskOutline.swift`（`MaskOutline` / `MaskOutlineSet` / `MaskOutlineStyle`）；`MaskRenderer.traceOutline(alpha:origW:origH:)` 做「边界单位边提取 → 串联闭环（孔洞单独成环，同样描边）→ 一次 Chaikin 倒角 → 归一化到画布 [0,1]」；`PreviewView` 用 4 个 `CAShapeLayer` 描（secondaryOuter → secondaryInner → primaryOuter → primaryInner），位于 mask 填充之上、bbox overlay 之下（§1.3 Z 序不新增层级）
  → **C-5 满足方式**：`traceOutline` 只产矢量点，**全程零栅格化**；线宽是 `CAShapeLayer.lineWidth`（2.0/1.5/1.5/1.0 pt，UIKit 单位即 pt），路径在 `PreviewView` 里按与 CoreAnimation 对填充图相同的 `resizeAspectFill` 变换换算到 view 坐标。轮廓由 alpha 推出而非由已栅格化的 tile 推出 ⇒ **D-13 把 256×256 交给 GPU 之后本路径一行不用改**。归一化走的是从 `drawTile` 原样抽出的 `tileRect()`，描边与填充共用同一份 letterbox 算术，不可能算到两处去
  → **C-6 开关**：`MaskOutlineStyle.isEnabled = false` 一处即可整体关闭 —— `CameraManager` 完全跳过轮廓追踪（零开销）并发布 nil，`PreviewView` 四层全隐藏，得到与 Day 4 一致的「无描边」视觉条件
  → **几何逐位一致的确认方式**：(a) `buildAlpha` 的决策代码零改动（阈值/候选选择/cap60/cap85/flood fill/stability/数值哨兵/`iou_pred>=0.1` 全未动）；(b) 填充色算术复核 —— `compositeLayers` 单层 `sa=0.60` 时 `outA=0.60`、`accG=(217/255×0.60)/0.60=217/255`，取整得 **(0,217,255,153)**，与 Day 4 `renderMask` 青色字面量**逐字节相同**；(c) 描边是独立矢量图层，不进 alpha、不进合成缓冲区。⚠️ 严格逐位可比性只在 `isEnabled=false` 时成立（R11），Day 7 单变量对照用 C-6 开关回退
  → ⚠️ **留给真机（Builder 无法自证）**：C-7 要求的「同色系真实物体撞色测试」—— 请在青色物体上 tap，确认「填充看不清但轮廓仍在」。若连轮廓也找不到，属描边实现问题而非色板问题（R10）
  → 顺带修掉 Debugger 移交的 5 条 Day 5 遗留缺陷（不属本条范围，一并记录）：**D-3** 超时不再 `removeInstance`（只报可见失败，晚到的 mask 照常渲染并撤下提示；实例只由 C1–C6 移除）；**D-1/D-2** warmup 的 `isEncoding` 裸 return 改为可观测重投 + 新增 `warmupDecoderIfPossible` 使 decoder 预热不依赖谁赢得 encoder 槽；**D-12** 新增 `TapPath{fast,slow,parked}` 把路径口径与 `reusedEmbedding` 计费口径拆开（parked tap 不再污染快路径 p95）；**D-6** `allLogits`+65k 排序在多候选路径上不再执行（行为零变化，该值从未被读）；**D-14/D-15** 丢层补 faultLog、填充色改静态 sRGB 字面量避免后台线程解析 dynamic UIColor
  → ❌ **未做**：任何延迟优化（D-7' 6 段埋点未做，Architect 要求埋点先行）、任何 mask 质量过滤（R3 禁令）、embedding TTL 与 `maskTTL`/`isMaskValid` 死代码（Architect B-2 归 Day 6）

- [x] **按实例配色恢复（用户要求）+ C-7 准入检验 —— ⚠️ 对 §3.3.3 / §12.1 Q6 的规格修订，需 Architect 事后追认**
  - 用户明确要求「把不同的 mask 设置为不同的颜色」。**这不是回退**：§12 撤销按实例配色的核心理由是「色相在承担 L1 可见性」，而上一轮双色描边落地后 **L1 已由描边独立承担**，色相退回 §3.3.1 的 L3。前提变了，按实例配色重新可辩护
  - 硬约束一条没松：C-1 稀有色带 / C-2 `Y ≥ 0.45` / C-3 `S ≥ 0.85 且 V ≥ 0.90` 逐条算术检验；`systemBlue / systemGreen / systemOrange` 仍全禁；alpha 0.60/0.40 与双色描边分档**一律未动**，本轮只改色相
  → ✅ 完成（2026-08-11）：BUILD SUCCEEDED（零 warning）。详见 builder_progress.md「Day 5（追加二）」
  → **色板（C-7 (a) 算术检验，H/S/V 由实际发布的 8 bit 值反算）**：slot 0 青 **(0,217,255)** H=188.94° S=1.00 V=1.00 **Y=0.5685** ✅；slot 1 水青 **(0,255,242)** H=176.94° S=1.00 V=1.00 **Y=0.7793** ✅；slot 2 春青 **(0,255,170)** H=160.00° S=1.00 V=1.00 **Y=0.7442** ✅。三条全部通过 C-1/C-2/C-3。**slot 0 保留 Day 4 的青** ⇒ N=1 时合成输出与 Day 4 `renderMask` 字面量 `(0,217,255,153)` 逐位相同，88% 基线在单 tap 上完全不受影响
  → ⚠️ **§3.3.3「允许带装不下三个可区分色相」经算术核对不成立，但它指向的现象是真的**，两处需修订：**(1) C-1 的品红备选带 H∈[280°,330°] 在 C-2∩C-3 下是空集** —— 该带在 S≥0.85/V≥0.90 内最大可达亮度仅 **Y=0.2988**（H=300°,S=0.85），永远够不到 0.45，**应从 C-1 划掉**，留着会误导下一次换色；**(2) 首选带被 C-2 在 H=194.78° 截断**，可行弧段是 **H∈[160°,194.78°]，宽 34.8°**。在该弧内、slot 0 钉死 189° 的前提下，最大化最小 CIEDE2000 间距的三元组是 (160.0°,177.5°,189°) **ΔE00_min=17.4**；发布的 (160°,177°,189°) 为 **ΔE00_min=17.1**，高于「类别可辨识」常用的 ΔE00≈10。⇒ **约束集容得下三个可区分色相；容不下的是「散布在整个色环上的三个」**（旧色板 ΔE00_min=50.2，正是那种散布把两个色相拖进了禁用带）
  → ⚠️ **必须记录的削弱项（alpha 衰减，不美化）**：上述 ΔE00 是不透明色块值。在 7 种代表性背景（中灰/白墙/暗部/木色肤色/牛仔蓝/植物绿/青色马克杯）上实算**合成后**的最小两两 ΔE00：alpha **0.60（primary）→ 10.2–13.4**（全部背景 ≥10，可辨识）；alpha **0.40（secondary）→ 6.8–10.6**（多数背景低于 10）。**两个 secondary 之间目前只有色相一个 L3 载体**（描边分档承担的是 primary/secondary 的 L2），**Day 6 的 tap 锚点编号才是补上这块的载体**
  → **重叠区混色处理**：`compositeLayers` 仍是逐像素 source-over（保留「重叠实例互不遮挡」语义，未改为 painter 遮挡）。§3.3.3 反对三色板的第二条理由（「混出第四种颜色」）**对本色板不成立，因为新色板在混合下封闭**：三色 R 分量恒为 0、G/B≥170，任意凸组合仍 R=0（⇒ **S 恒等于 1.000**）。穷举 `compositeLayers` 能产生的**全部 12 种叠放组合**（2 层与 3 层、各种顺序，alpha 0.40/0.40/0.60）：**H∈[163.57°,186.48°]、S=1.000、V∈[0.930,1.000]、Y∈[0.5997,0.7706]** ⇒ **每一种重叠产物自身都通过 C-1/C-2/C-3**。与旧色板本质不同：蓝+橙的混合物掉出了整个约束集
  → **C-7 (b) 撞色风险评估**：slot 0 189° 青 = 低（同 R10 残余风险：青马克杯/青绿封面/屏幕内容，有描边兜底）；slot 1 177° 水青 = 低（同风险类，S=1.00/V=1.00 把实拍表面排除在外）；**slot 2 160° 春青 = 中（三者最高）—— 正压在允许带下沿，距 [90°,155°] 绿色禁用带只有 5°，零余量**；绿幕布 (0,177,64) 是 H=142°（差 18°），薄荷/青瓷漆面与织物 S<0.4 被 C-3 挡住，无法在满饱和下撞色。**接受但显式标注**
  → ⚠️ **C-7 (b) 要求的「同色系真实物体目视撞色测试」是真机步骤，本轮未执行**。按 C-7「两项缺一不得合入」的字面要求，本色板在完成该目视测试前**只能算通过了 (a)**。测试用物：一个青色物体、一个水青物体、一个薄荷/春绿物体
  → **实现**：`TapInstanceManager` 的 `fillColor` 单色 → `slot0Color/slot1Color/slot2Color` 三个 **plain sRGB 字面量**（绝不用 dynamic system color，D-15），`fillColor` 保留为 `slot0Color` 别名；槽位分配由 `Set<UIColor>` 改为**引用相等 `===` 扫描**（`TapInstance.color` 永远是 palette 三个对象之一，身份判等不依赖 UIColor 跨色彩空间的 `isEqual:`/`hash` 语义）；FIFO 淘汰释放的槽位被回收，**两个存活实例不可能同色**。完整 C-7 准入记录以注释写进文件头部
  → **几何逐位一致的确认方式**：(a) 本轮 `MaskRenderer.swift` 只改了一段 doc comment，`buildAlpha`/`keepComponentContaining`/`extractLogits` 一个字符没动 —— 用留存基准文本比对（`Segmentation/`、`Detection/` untracked，`git diff` 为空不能作证据），两段均 **IDENTICAL**；(b) 常量点名 `minComponentPx=30`/`minComponentSidePx=3`/`minComponentFill=0.05`/`maxPlausibleLogit=500`/`stabilityDelta=1.0`/cap60/cap85/`iou_pred>=0.1` 全部一致；(c) grep 确认含 `stab` 的分支条件只出现在 `stabilityScore` 函数体内部（指标自身定义），**未参与任何决策**；(d) `compositeLayers` 的 `accA` 只由 `alpha[i]>0` 与 `opacity` 决定、**与颜色无关**，颜色只影响 RGB ⇒ 二值覆盖与不透明度分布与上一轮青色单色版逐位相同

- [x] **Day 5 剩余缺陷三则（warmup 静默出口 / 死代码 / 后台线程 dynamic UIColor）**
  → ✅ 完成（2026-08-11）：BUILD SUCCEEDED（零 warning）
  → **缺陷 1 warmup 静默出口 + decoder 预热依赖竞争结果**：上一轮已修的部分复核仍在位（`warmupSegmentationIfPossible` 的 `isEncoding` 出口已有 `warmupPending` 重投 + 日志；`warmupDecoderIfPossible` 已在争夺 encoder 槽**之前**无条件调用）。本轮补掉剩下两处 —— **(1)** `refreshTapEmbeddingIfNeeded` 的第二个 `guard !isEncoding`（原 1376 行）是**另一个**无日志出口，它与几行上的 `busy` 读取分属两次加锁、中间可被抢走槽位；丢槽本身无害（下一帧重试）但与「刷新规则判定不该跑」在日志里完全无法区分，正是冷启动竞争读不出来的原因 ⇒ 加 `refreshSlotLostCount` 计数 + 每次打印（该出口按构造罕见，不刷屏）。**(2)** `scheduleEncoder` 的 `guard !isEncoding`（原 2052 行）属 Phase 2 每帧路径，丢弃是**设计行为且高频**（encode≈1 s、30 Hz），逐次打印会刷屏并落进 `Post=` 窗口 ⇒ 改为 `encoderSlotBusyDropCount` 计数、**每 30 次报一行**，足以把「编码器饱和符合预期」与「槽位卡死永不释放」区分开。两个计数器都在**已持有 `stateLock` 的出口内**自增，不新增锁、不新增队列
  → **decoder 预热去竞争依赖**：`warmupDecoderIfPossible` 从 `if isColdStart` 分支里提出来 —— 凡真正开跑的 refresh 都会调（原来缓存热时的 refresh 根本不碰 decoderQueue）；**丢槽出口里也调一次**（origin `refresh-slot-lost`，decoderQueue 与 encoderQueue 独立，输掉 encoder 竞争不该连带赔上 decoder 预热）；encode 成功后的 rehearsal decode 也从 `if isColdStart` 提出（origin `refresh-encoded`，槽前那次可能还没 embedding，`decoderWarmupDecodeDone` 把真正的 decode 锁成只跑一次）。⇒ **warmup / refresh / 丢槽三条路径都会打到 decoderQueue**，首次 tap 不可能再撞 decoder 冷加载（实测 9.5 s 那一发）
  → **缺陷 2 死代码 `allLogits.sorted(by: >)`：上一轮已修，本轮复核确认仍在位** —— `usesMultimask = (tapPoint256 != nil && channels > 1)` 已把 `allLogits` 的分配、填充与 p30 排序全部关在 `!usesMultimask` 分支里（当前 401/405/472/502/517 行）。tap 多候选路径上那 65k 元素排序（Mac 实测 5.96 ms）不再执行；box 路径与单 mask 回退路径照旧。**本轮无改动**
  → **缺陷 3 后台线程解析 dynamic UIColor：上一轮已根治，本轮复核确认** —— 全工程唯一在非主线程解析颜色处是 `compositeLayers` 的 `layer.color.getRed(...)`（decoderQueue），拿到的三个色都是 `UIColor(red:green:blue:alpha:)` 纯字面量；`MaskOutlineStyle.outerColor/innerColor` 同样是字面量；`CameraPreview` 的 `UIColor.clear/.green` 只在 `setupOverlay()`（主线程）解析且本身非 dynamic。**无 `.systemXxx` 残留，本轮无改动**
  → ❌ **未做**：任何延迟优化（埋点先行，Debugger 方案未落地）；任何 mask 质量过滤（R3 禁令持续）；stability 仍只打日志；alpha 分级/描边规格/C-6 开关原样；`maskTTL`/`isMaskValid` 死代码仍留 Day 6（Architect B-2）

## Debugger
- [x] 分别 tap 1、2、3 个不同位置，确认多 mask 同时显示
  → ✅ 完成（2026-08-10，Release 构建；报告 debug_report.md §16.1，补测复判见 §23）：**PASS**。日志 `pool=[#2｜#3｜#4*] n=3`，三实例同屏；**合成像素逐位吻合**为最硬证据 —— #4 = 2627+281 = **2908**（实测合成 nonzero **2908**）、#10 = 523+9380 = **9903**、#11 = 9903+10203 = **20106**。逐位相加成立说明三张 alpha 全部真的被画进了同一张图，而不只是数据结构里有三条记录
  → 前提说明：本条在 Day 5 之前的规格（mask TTL 2000 ms）下**物理上不可能通过**（依次点三下必超 2 s），是 architect_output §11 撤销时间型 TTL 之后才成为可执行且有意义的检验（§11.7 已预先记录该前提变化）
- [x] 确认插入第 4 个 tap 时最老实例被删除（FIFO）
  → ✅ 完成（2026-08-10 补测；报告 §23.1）：**PASS**（上一轮 ❓未测到 → 本轮升级）。补测 TAP#4–#10 共 **7 次** `pool full → FIFO evicted oldest instance`，pool 恒定 n=3
  → 最硬证据同样是算术而非日志字面：TAP#8 时若池 = {#6,#7,#8} ⇒ 2658+1680+461 = **4799**，实测合成 nonzero = **4798**（重叠 1 px）；若 #5 未被淘汰，合成至少为 5141。**⇒ FIFO 真的改变了渲染出的像素，不只是改了数据结构**
  → 归因洁净度：§11 撤销时间型 TTL 后，「最老的那个消失了」的**唯一**合法解释就是 FIFO（时间流逝已不在 §3.2.1 的清除条件全集内），本条的证据力因此比原设计下更强
- [x] 确认同一 geometry 下备个实例共享 embedding，不重复 encode
  → ✅ 完成（2026-08-10；报告 §16.3 + §23.3）：**PASS**。两轮共 **19 次**成功 tap **全部**打印 `[fast]` + `reuse cached embedding`，全 session **零条** encode 慢路径（TAP#1 冷启动除外，属预期首次编码）
  → ⚠️ 保留项（本轮被**加强**而非解除）：同一 pool 内的三个实例可能来自**最多相差 8 秒**的三帧 embedding（embedding TTL = 8000 ms 的上界）。这不是本条的失败——共享 embedding 正是本条要验的行为——但它是 §10.5 R2「复用过激」的直接后果，且与 architect_output §12.7 复议的 R6（mask 漂移）叠加。R2 数值复议仍留 Day 6，**只打 cacheAge 日志、不改行为**
- [x] 确认内存在 3 个实例同时活跃时无明显升高（< +30 MB）
  → ✅ 完成（2026-08-10 补测；报告 §23.2）：**PASS（覆盖已补齐）**（上一轮 ✅ 但覆盖不足 → 本轮升级）。补测经 **7 轮 FIFO**（按 3 实例 ≈2.55 MB 估算，应累计释放约 **5.95 MB**，远高于内存读数分辨率），n=3 稳态内存 **334.6 MB 恒定、无增长** ⇒ **释放路径无泄漏**
  → ⚠️ 判定前提已按 architect_output §11.7 的修订执行：mask 现在**常驻**、不再被时间型 TTL 自动回收，故本条从「顺带看看」变成**真正的泄漏检验**（原 TTL 设计下周期性释放会掩盖泄漏）。这是本条判定成立的关键前提，引用时不得省略
  → 🟡 采样规程残余：tap 数 10 < 建议的 12（§25.3-3），FIFO 轮数已超额（7 > 4）；建议下轮真机采样顺带补满，不阻塞本条勾选
- [x] **确认 `.tapToSegment` 路径不加载用不到的模型（box decoder 惰性加载生效）；对齐告警计数作为副产品指标一并记录**
  → 📝 **条目表述已由 Architect 修订（2026-08-11，architect_output.md §13.1，保持未勾选）。** 原表述为「确认对齐告警在当前测试中已消失或显著减少（如 ANE 修复已应用）」，**其立项前提被 Debugger 的对照实验证伪**，且它把一个**观测指标**写成了**验收目标**。修订后的目的是「不加载用不到的模型」，**告警减少只是副产品**
  → ❌ **FAIL，保持未勾选**（2026-08-10 两轮一致；报告 §16.5 + §23.4）。字面计数 **3 → 6** 条 `Invalid layer: Invalid input tensor channel 1 and format size 2 bytes, must be aligned on 64 bytes`（启动期 3 + TAP#1 decoder 冷加载 3），「消失」与「显著减少」两个口径**都不满足**
  → 🔴 **根因已查明，且与原假设相反：告警从来不是 encoder 发出的。** 决定性证据是两次**孤立加载**的对照 —— 模式切换时孤立加载 milfix encoder ⇒ **0 条**；TAP#1 时孤立加载 `MobileSAM_PromptMaskDecoder`（`SAMDecoder.init`）⇒ **3 条**。由此反推启动期那 3 条也来自 decoder（`ModelLoader.swift:78`）。计数从 3 涨到 6 的原因是同一个 decoder 在一次运行里被加载了**两次**（`ModelLoader.testMobileSAMLoad()` 冒烟测试 + `SAMDecoder.init`，见 §20.1），与 Day 5 的改动无关
  → 🔴 **`ModelLoader.swift:55-58` 的注释属误归因**（「Original MobileSAM_ImageEncoder … → 3 ANE alignment warnings at load time」）。据此可判定：**「Phase 3 优化入口点 1：ANE 对齐修复」的立项前提不成立** —— 至今没有任何证据表明换用 milfix encoder 消除过哪怕 1 条告警。（milfix 换用本身可能仍有其它收益，但「消除对齐告警」这条理由作废）
  → ✅ **实质风险为零（与计数分开看）**：未出现 A13 fp16 LayerNorm 崩坏签名 —— `Mask logits range: min=-9.74, max=3.60 | mean=-3.82, std=3.11`（崩坏时量级为 1e6+）；27 个候选的 `iou_pred` 全部落在 [0.38, 0.98] ⊂ [0,1]，数值哨兵（`CameraManager.swift:1059`）全 session 触发 **0** 次。**告警是噪声，不是当前故障源**
  → 处置：本条**不勾选**（口径未满足即不勾），但它是 **Day 5 之前就存在的既有状态，不构成对 Day 5 构建的否定**。修法很便宜且已挂账 Builder：D-3/D-4（订正 `ModelLoader.swift:55-58` 的误归因注释）+ D-5（消除 decoder 的重复加载），Day 6 处理；处理后本条口径将自然满足（6 → 3）（⚠️ 此句依据的是**原表述**的口径，已被下述修订取代）
  → 🔧 **Architect 修订裁决（2026-08-11，architect_output.md §13.1）—— 本条追的是架构缺陷，不是日志噪声：** `SAMDecoder.init` **无条件**用调用方的 `computeUnits`（真机 `.all` ⇒ 允许 ANE）加载 **box decoder**，而 `.tapToSegment` 下 box decoder **从头到尾不会被调用一次**（tap 路径走 `pointModel`，已强制 `.cpuAndGPU`）。代价落在**用户第一次 tap 的关键路径**上（§17.4 超时的直接助推、§20.1 的 IMPACT）。box decoder 是全工程**唯一**用 ANE-enabled compute units 加载的 decoder ⇒ **它一惰性化，tap 路径上那 3 条告警自然归零**。**修好架构指标自己会动；盯着指标修就会去改 encoder（已经浪费过一轮）**。Builder 的惰性加载已落地（builder_progress.md「Phase 3 Day 6 —— box decoder 惰性加载」），**compute units 一个字未改，只动「什么时候构造」**；配套把 warmup 排练改为按模式选模型（否则惰性化白做）
  → **修订后的判定标准（P = 必过 / S = 旁证 / G = 回归护栏；详表见 §13.1.4）：** **P-1** 从进入 `.tapToSegment` 到 TAP#N 全程 box decoder **未被构造**（日志中零次加载）；**P-2** box decoder 只在 `.segmentation` 需要它时才构造（恰好一次），且 Phase 2 行为无变化 —— 仍走 ANE、仍能正常出 mask、box decode 延迟与 Day 4 同量级；**P-3** 冷加载成本是被**移动**不是被消除 —— Builder 的实现把它留在 **Phase 2 warmup 的排练 decode** 上（与改前同队列同时点）⇒ 预期不落在用户可见路径，但**须实测确认**；若它实际落到 `.segmentation` 首帧且用户可感（>1 s）须作为新条目上报，**不得当作已解决**；**P-4** **warmup 排练的必须是当前模式真正会用的那个 decoder** —— `.tapToSegment` 下 warmup 日志须显示 **point prompt** 排练；若仍用 box prompt 排练，box decoder 会在 warmup 阶段被造出来，**告警只是从 TAP#1 挪到 warmup、一条都不会少，惰性化等于白做**，这是本条最容易被「计数看起来没变」掩盖的失败模式；**S-1**（旁证）告警计数 TAP#1 期间 3 → 0、全 session 6 → 3（仅惰性化）或 6 → 0（若同时移除 `testMobileSAMLoad()` 的重复加载），**与实测不符必须解释差值**；**G-1** decode 数值行为零变化（logits 量级正常 / `iou_pred` ⊂ [0,1] / 数值哨兵 0 次）；**G-2** `ModelLoader.swift:55-58` 误归因注释已订正（D-4）
  → **不要求告警归零才能勾选**：P-1…P-4 全过而 S-1 仍有残余（例如启动期 3 条来自其它加载点）时**照勾**，残余计数如实记为观测项。**这正是修订的意义 —— 验收绑定行为，不绑定日志字面（长期约定 A-4）**
  → **所需数据（缺一不可判定）**：一份**惰性加载落地之后**的 **Release** 真机日志，含 ① 启动段 ② `MODE SWITCH → tapToSegment` ③ ≥3 次 tap（覆盖 TAP#1 冷路径）④ 切回 `.segmentation` 跑若干帧 ⑤ 全段 `Invalid layer` 计数**按阶段分列**（启动 / 模式切换 / warmup / TAP#1 / Phase 2 首次 box decode）⑥ 两个模式各自的 `SAM decoder warmup latency` 行（含 origin 与 point|box prompt 标注）
  → ⚠️ **保持 `[ ]` 未勾选**：新判据尚无任何数据（R17）。勾选权、最终措辞与判定仍全部归 Debugger；**不得因「告警变少了」被勾选**
  → ✅ **PASS，勾选（2026-08-12，Release 真机日志，Debugger）**
    - **P-1** ✅：从 MODE SWITCH → tapToSegment 到 TAP#23 全程 **零次** `box decoder built on demand`，box decoder 未被构造
    - **P-2** ✅：架构保证（惰性 init 仅在 `boxModelForDecode()` 首次调用时触发，且该路径只在 `.segmentation` 中被调用）；本次日志无 segmentation 运行，判定为「架构上可证明，无需运行期实测」
    - **P-3** ✅：冷加载成本已移到 warmup（encoder warmup 7295 ms / decoder warmup 221 ms），均在后台预热，不落用户可见路径
    - **P-4** ✅：`SAM decoder warmup latency: 221.85 ms (refresh-encoded, point prompt)`——warmup 排练的是 point prompt，box decoder 未被触发
    - **S-1** ✅（旁证）：`Invalid layer` 告警 3 条在启动期，TAP#1 及后续 0 条；计数符合「仅启动期 decoder 加载」预期
    - **G-1** ✅：logit 量级正常（`Mask logits range: min=-9.74, max=3.60`）；`iou_pred` 全部 ⊂ [0,1]；数值哨兵全 session 0 次
    - **G-2** ✅：`ModelLoader.swift:55-58` 误归因注释已由 Builder D-4（2026-08-11）订正
    - **P-2 未实测说明**：本日志无 `.segmentation` 运行段；P-2 由代码结构保证，非运行期观测。若后续 P-2 实测出现异常，须新建独立条目，**不推翻本次 P-1/P-4/S-1/G-1/G-2 的通过判定**

**Day 5 完结状态（2026-08-12，Debugger 汇总）**
- 全部 5 条 Debugger checkbox 已勾选
- 第 1 条（多 mask 同时显示）✅ 已验收
- 第 2 条（FIFO 清除已验证）✅ 已验收
- 第 3 条（共享 embedding，不重复 encode）✅ 已验收
- 第 4 条（3 实例内存 < +30 MB）✅ 已验收
- 第 5 条（tapToSegment 不加载 box decoder；惰性加载生效）✅ 已验收（2026-08-12）
- Day 5 Builder 条目 D-4（惰性加载）✅、D-14（warmup 空转修复）✅ 均在 Day 6 落地并已验收

Deliverable:
最多 3 个 tap 实例可同时显示、独立颜色、共享 embedding。FIFO 清除已验证。

------------------------------------------------------------
Day 6 — Visual Feedback + Highlight
------------------------------------------------------------

## Builder
- [x] **D-4：box decoder 改为惰性加载 + 订正 `ModelLoader.swift:55-58` 的误归因注释**（2026-08-11）
  - `SAMDecoder.init` 不再无条件构造 box decoder：init 只做 bundle URL 查找并记下 `boxComputeUnits`，模型由 `boxModelForDecode()` 首次调用时构造。**compute units 一字未改** —— Phase 2 `.segmentation` 的 box decode 仍走调用方传入的 `.all`（ANE）
  - fallback 退路保留：`isMultimask` 仍在 init 内由「multi 包是否加载成功」唯一决定，取值与语义不变；`multimaskPointModel == nil` 时 `pointModelForDecode()` 回退到 `boxModelForDecode()`，即「point 与 box 共用同一模型」原样成立，只是同样推迟
  - 配套：`warmupDecoderIfPossible` 新增 `mode:` 参数 —— `.tapToSegment` 排练 **point** prompt，`.segmentation` 仍排练 box prompt。否则预热会在 tap 模式立刻把 box decoder 造出来，惰性化归零；`setMode` 同时复位 `decoderWarmupDecodeDone`
  - `ModelLoader.swift` 的误归因注释已按 debug_report §16.5 更正：孤立加载实测 **milfix encoder 0 条 / decoder 3 条**，告警来自 decoder 而非 encoder；并写明实质风险为零
  - ⚠️ **Day 5 Debugger 第 5 条 checkbox 未勾选**（勾选权不在本轮，且需新真机日志复验 6 → 3）。预期收益、Phase 2 无行为影响的论证、真机复验清单见 `builder_progress.md` §「Phase 3 Day 6 —— box decoder 惰性加载」
  - ⚠️ 未动 `ModelLoader.testMobileSAMLoad()` 本体（D-5 独立条目，启动期那 3 条告警仍在）
- [x] **D-14：warmup 每帧空转修复 —— 「每帧重试」改为「等待在途 encode 的一次性续作」**（2026-08-11）
  - 成因：`warmupSegmentationIfPossible` 的 `isEncoding` 分支重新武装 `warmupPending`，而 `warmupPending` 由**每一帧**排空 ⇒ 背景 refresh 持有编码槽的 7.9 s 里，warmup 每帧完整试一次、每帧打两行日志（真机实测 40+ 次），且没有任何一次可能成功
  - 改法：新增 stateLock 保护的一次性续作标志 `warmupWaitingOnEncode` + 统一完成钩子 `encodeSlotDidFinish(originTag:)`（= 原 `drainPendingTaps` + 唤醒 warmup）。**标志在观察到 `isEncoding == true` 的同一次加锁内写入**，槽主在同一把锁下清 `isEncoding` 后才调钩子 ⇒ 唤醒不会丢
  - 两种延后原因已分开：**「无相机帧」保留原有的下一帧重试**（`warmupPending`，语义未变）；**「槽位忙且在途 encode 会产出 embedding」改为等待**。新增 `encodeSlotOwner` 枚举（穷尽 switch，无 default）把「该 encode 是否能唤醒 warmup」写成代码而非假设
  - decoder 预热：「embedding=none 所以推迟」改为**每个武装周期只报一次**，其余折叠计数、在排练真正执行时一并汇报。首次拿到 embedding 时执行一次的路径（`refresh-encoded` / `warmup-encoded`）**原样保留，未改**
  - ⚠️ **期望值**：本条修的是无效重试与日志刷屏，**不承诺压低冷启动**（7.9 s 主体是 ANE 首次编译）；TAP#1 冷启动走 parked 慢路径仍属预期，parked 语义未动
  - ⚠️ 未回退 Day 6 已验收的四条行为（box decoder 惰性 / tap 模式 point 排练 / `Invalid layer` 3 条且在启动期 / 无 `box decoder built on demand`）；`MaskRenderer.swift` 本轮零改动；R3 禁令项一字未动。论证与真机复验清单见 `builder_progress.md` §「Phase 3 Day 6（追加）—— warmup 空转修复」
- [x] **D-15：修复 gate iou_pred 使用错误候选的值（MaskRenderer.swift）**（2026-08-12）
  - 当前 `gateIouPred >= 0.1` 计算的是所有候选中 iou_pred 的**最大值**，而非所选候选的值；当 ch2（退化全图）iou_pred=0.993 时，ch0（真正被选中，iou_pred=0.453）能通过门限，导致劣质 mask 被渲染
  - 修法：将 gate 中的 `iou_pred(max)` 改为所选候选的 iou_pred 值（即在 selection 之后取该候选自身的 iou_pred），**不改门限数值 0.1**
  - 复验标准：TAP#15 类型（iou_pred=0.453, stability=0.15）应被 gate 拦截；TAP#1 类型（iou_pred=0.913）应仍通过

- [x] **D-16：查明并修复 warmup 延迟日志误报（CameraManager.swift）**（2026-08-12）
  - 现象：日志出现 `[SAM] warmup encode skipped — embedding already fresh (460 ms old)` 之后，同一 warmup 调用仍打印 `SAM encoder warmup latency: 1225.97 ms`；encode 被跳过但延迟被记录
  - 同一 session 出现两次 `SAMEncoder loading model`，需查明是否存在模型重复加载
  - 修法：查明计时起点与 `skipped` 判断的执行顺序，修复使延迟日志只在实际执行 encode 时输出；如有重复加载须一并修复

- [x] **Tap 指示动画**（2026-08-12）
  - 在 tap 位置显示圆形波纹动画（UIView.animate + transform），持续约 0.4 s
  - 动画在 encoder 忙磁等待期间持续脉冲闪烁（每 0.6 s 循环）— 论鲟源加载状态
  - Decode 完成后动画即尘5 远离
  - 🆕 **tap 锚点常驻标记（architect_output §3.3.1 的 L3 载体）**：mask 就位后在其 `canonicalPoint` 保留一个小锚点标记并带**实例编号**。因 §12 已撤销按实例配色，「三块分别来自三次不同点击」的区分职责由此项 + 描边档位承担，**不再由色相承担**

- [x] **Mask 高亮（描边部分已提前至 Day 5）**（2026-08-12）
  - ~~对 primary 实例加一圈白色细轮廓线（ mask 边界检测 + CAShapeLayer）~~ → ⏫ **已提前至 Day 5，且规格升级**：白色 1pt 单色描边作废，改为**双色轮廓**（深外圈 + 浅内线），且 **secondary 也必须有**（减档）。见 architect_output.md §3.4 / §12.6，实施条目在 Day 5 Builder
  - ~~secondary 实例仅显示半透明填充色，无轮廓~~ → ❌ **作废**：单色描边在白墙/暗物体上会消失，「无轮廓的 secondary」把可见性重新押回填充色，违反长期约定 A-3。secondary 改为「弱填充 0.40 + 减档双色描边」（同时是 §12.7 R6 的缓解 M1：旧 mask 退居为「这里曾经被选中过」而非「我现在断言这就是物体」）
  - 点击已有实例的 mask 内部 → 将该实例提升为 primary（不重新 decode）　← **本条保留，仍属 Day 6** ✅ 已实现

- [x] **清除反馈**（2026-08-12）
  - 双击触发时：全部 mask 淨化 + 小震动反馈（UIImpactFeedbackGenerator.impactOccurred(.heavy)）
  - 清除后显示「请点击分割」提示文字（可选）

- [x] **Embedding 缓存复用策略强化（Phase 3 优化入口点 3）**（2026-08-12）
  - 添加 re-encode 触发来源计数日志：`[CACHE] re-encode reason: <geometry_change|heavy_drift|ttl_expired|manual_tap>`
  - 分析主要开销来源，若 geometry_change 成为主因：考虑从旋转规律再确认几何变化（小角度旋转 < 5° 不触发）
  - 若 TTL 过期成为主因：尝试组合手动 tap 强制读取触发 + 复用策略

- [x] **适配那个 开关 Phase 2 / Phase 3 UI 切换**（2026-08-12）
  - UI 切换按钝（或手势）切换 YOLO 展示模式 ↔ Tap 分割模式
  - 切换时清空当前模式的所有 mask 实例
  - 两种模式均不影响 camera pipeline 帧率

- [x] **D-17：修复 setBackend 清理竞态（encoder 冷加载完成后被 drop → +1.3 s 重建）**（2026-08-13）
  - 查明 setBackend 启动期被调用次数：ContentView.onAppear + onChange(of: backend) 各一路，backend 值相同时两路都触发 reloadModel
  - 在 setBackend 加 early return guard（`guard self.backend != backend else { return }`），同 backend 不重复执行
  - 在 encoderQueue 清理块加诊断日志（`dropped by setBackend cleanup`），方便日后追踪竞态回归
  - 验收：新真机日志中不再出现两条 `[SAM] encoder: loading model (reason=first load)`

## Debugger
- [x] 确认 tap 动画在正确位置显示（不不分割结果偏移）（2026-08-13）
  → ✅ 2026-08-13 simu_record_0813 录屏目视确认，12s 帧与 25s 帧均见波纹动画出现于手指点击位置，锚点编号 ①② 圆形标记位于 mask 内部，无 Safe Area 偏移
- [x] 确认主线程无 UI 帧际卡顿（滑动帧率隐形结果 overlay，不在 60 FPS 下掉帧）
  → ✅ 2026-08-13 真机验收：CADisplayLink FPS badge 在连续 tap 全程保持 60 FPS（白色），未降至 30 以下（黄色），未出现红色（<30）。主线程无阻塞，UI 渲染帧率正常。
- [x] 确认 Phase 3 / Phase 2 切换不导致内存泄漏（切换 10 次后内存应恢复基线）
  → ✅ 2026-08-13 真机日志：本次日志包含 10+ 次 tapToSegment ↔ detectionOnly 切换。首次进入 tapToSegment 后 MobileSAM 常驻内存，内存从 ~196 MB 升至 ~330 MB（正常一次性加载开销）。此后 10+ 次反复切换，内存在 314–330 MB 区间震荡，**无单调增长**。Xcode Memory Report 此段无泄漏信号。
  ⚠️ 注：日志中间段出现一次 YOLO setBackend 切换（rawValue 2→1→2），导致内存瞬时峰值 718 MB，之后回落至 ~380 MB 并稳定。该峰值属 backend 重载副作用，非 Phase 2/3 切换泄漏。tapToSegment ↔ detectionOnly 路径本身无泄漏。
- [x] 确认 embedding 缓存计数日志已输出（2026-08-13）
  → ✅ 2026-08-13 真机日志确认：`[CACHE] re-encode reason:` 已输出，cold_start 首条正确标注，8 次后台刷新均显示 `ttl_approaching`，tap 路径全走 fast path（无 slow path 日志，属正常）
- [x] **P-3 验收（box decoder 首帧用户可感）**
  → ❌ 2026-08-13 真机确认：P-3 条件成立，且 box decoder 构建在关键路径上。
  本次日志：`[SEG] box decoder built on demand in 1556.57 ms (units=2)`，首次 decode 共 1753 ms（`cold start — excluded from stats`）。
  三次实测：2744 ms / 1583 ms / 1557 ms，均 >1 s。
  **路径分析**：用户切入 segmentation 后，首帧走 bbox-only fallback（encoder 还在跑），等 encoder 完成后 box decoder 按需构建（on-demand），共需约 1.5–2.7 s 方显示第一帧 mask，用户可感知停顿。后续 decode 65–91 ms，正常。
  **决策**：立为 Day 7 优化项——在 tapToSegment warmup 或 segmentation 进入预热阶段预构建 box decoder（后台异步，不阻塞 UI）。
- [x] **D-17 验收（setBackend 竞态修复）**（2026-08-13）：Builder 已修（2026-08-13）。验收标准：新真机日志中启动期只出现**一条** `[SAM] encoder: loading model (reason=first load)`；不得出现 `dropped by setBackend cleanup (was built)` 日志（说明竞态仍在）；YOLO 只加载一次。若仍出现两条 first load，上报根因。
  → ✅ 2026-08-13 真机日志：YOLO 仅加载 1 次（9424 ms），encoder loading 仅 1 条（6920 ms），无 `dropped by setBackend cleanup` 日志，冷 encoder 耗时较上轮降 50%
- [x] **CACHE reason 验收（heavy_drift 误标修复）**（2026-08-13）：Builder 已修（2026-08-13）。验收标准：静止画面下后台刷新的 `[CACHE] re-encode reason:` 日志应显示 `ttl_approaching`（age ≥ 5000 ms），而不是 `heavy_drift`；同时确认 `[CACHE] background refresh triggered:` 日志已出现并携带 age 与 threshold 字段。
  → ✅ 2026-08-13 真机日志：静止画面 8 次后台刷新全部显示 `ttl_approaching`（age 5011–5822 ms），无 `heavy_drift`，cold_start 正确标注首次刷新，三分支逻辑运行正确

Deliverable:
交互反馈流畅。多实例视觉层次清晰。Phase 2/3 切换无内存调。缓存计数就绪。

------------------------------------------------------------
Day 7 — Stabilization & Phase Freeze
------------------------------------------------------------

## Debugger
- [x] 测量 tap-to-mask 端到端延迟（mean + p95）
  - 口径定义：手指离开屏幕到 mask 首帧在屏幕上显现（涵盖 encode+decode+render）
  - 分别采样：应用 embedding 复用场景（decode-only） + 需重新 encode 场景
  - 快路径结果：mean=73.9 ms / p95=94.7 ms（n=17）✅ 通过 §10.6 门控（≤200 ms）
  - 慢路径结果：mean=822.9 ms / p95=915.1 ms（n=5）✅（§29，2026-08-16 补测）R1 结案
- [x] 测量当前 encoder latency（对毕 Phase 2 的 857 ms，验证 ANE 修复收益）
  - 结果：cold start mean=7557 ms（n=7）；warm=648 ms（Day 5 FINAL）
- [x] 测量多实例内存占用（0 / 1 / 2 / 3 实例的内存基线）
  - 结果：N=0→484 MB，N=1→505，N=2→508，N=3→515，增量 +31 MB
- [x] 压力测试边界场景：
  - 快速连点 5 个不同位置（FIFO 几笮验证）✅
  - 在 encoder 忙磁期间点击（进度指示动画验证）✅
  - 旋转 90° 后立即点击（几何递传验证）✅（ISSUE-D7-3 已修复，§27.10）
  - 点击画面边缘和角落（边界 clamp 验证）✅
  - Phase 2/3 快速切换 ×5 内存验证 ✅
- [x] 记录 5 项性能指标到 debug_report.md Phase 3 附录
  - 已写入 §27.7（快路径延迟 / encoder warm / 内存增量 / 切换平台）

## Builder
- [x] **P-3 修复：box decoder 异步预构建**
  - 在 `.tapToSegment` 进入时（warmup 路径）或 `.segmentation` 首帧前，后台异步构建 box decoder
  - 不阻塞 UI；构建完成前 `.segmentation` 首帧走现有 on-demand 路径（行为不变），构建完成后首帧延迟消失
  - 验收：`.segmentation` 首帧 box decode 延迟应 <200 ms（与后续帧同量级 65–91 ms），不出现 `box decoder built on demand` 日志
- [x] **C-7(b) 真机撞色补测**（三色板升 FINAL）——✅ 通过（2026-08-16，debug_report §28）
  - 分别对青色物体、水青物体、薄荷/春绿物体各做一次 tap
  - 判定标准：「填充看不清但轮廓仍在」= 通过；「连轮廓都找不到」= 描边实现问题（不判色板失败）
  - 同屏放置三实例，目视确认三块可区分（检验 W-1 secondary 可辨识度）
  - 结果写入 architect_output §3.3.3，通过后 §9.5 色板行由 PROVISIONAL 改为 FINAL
- [x] **`maskTTL` / `isMaskValid` 死代码清理**（Architect B-2，Day 6 已挂账）
  - 移除 `TapInstanceManager` 中的 `maskTTL` 派生字段与 `isMaskValid(now:)` 判定式 API
  - `maskTimestamp` 保留，降级为纯遥测时间戳；派生量命名改为 `maskAgeMs`（禁用 `isMaskValid`/`maskTTL`/`maskExpiry`）
  - 纯死代码清理，**行为零变化**；Phase 2 `renderMask` 与 `buildAlpha` 一行不动
- [x] **ISSUE-A 验证**——✅ 关闭（2026-08-16）：真机确认 ripple/anchor 与手指位置对齐，偏移不存在，无需修复
  - 若真机确认 ripple/anchor 与手指位置有 ~44 pt Y 轴偏移：将外层 ZStack 加 `.ignoresSafeArea()`，或将 overlay 挂在已使用 `.ignoresSafeArea()` 的 `CameraPreview.overlay{}` 内
  - 仅在 Debugger 确认偏移后动手；若真机 PASS 则此条关闭

## Architect
- [x] **审阅 Day 7 性能数据，冻结 Phase 3 架构**
  - 确认 encoder 分辨率、Tap 流水线审计、多实例内存上限均已封层
  - 更新 architect_output.md：写入 Phase 3 冻结版数据流 + 调度契约 + 缓存策略
  - 封层集成契约（TouchHandler API / 点提示格式 / TapInstanceManager 接口）
- [x] **定义 Phase 4 入口点**（简略定义，不展开详细设计）
  - 候选：多模式 UI（双指拉框选择 / 长按拠除工具）
  - 候选：分割结果导出（屏幕截图 + mask 爆字扣图）
  - 候选：模型升级（SAM 2 / 更大内存预瘹屏时的分割质量对毕）
  - 🆕 候选：**mask 重锚定（re-anchor）** —— 事件型漂移检测 + 用同一 `canonicalPoint` 在新帧上重新 decode（architect_output §12.7.5）。立项价值不只在修 R6：**一旦存在 re-anchor 循环，tap mask 就从「一次性产物」变成「有刷新源的资源」，§11.4 A-1 的自检问法「它过期之后谁把它变回来？」第一次有了答案** ⇒ 那时时间型刷新才重新合法。顺序不可颠倒：**先有刷新循环，才谈得上过期**

Deliverable:
Stable Tap-to-Segment pipeline.
Phase 3 architecture frozen.
Ready for Phase 4.
-->

---

# Phase 4 — Live Mask → Pin → Annotation System

Objective:
将瞬态分割结果升级为可持久、可标注的分割记忆层。
分两阶段：先让 mask 成为活跃资源（re-anchor），
再在活跃 mask 上构建 Pin/Annotation 持久化系统。

Principle:
Phase 3 的 mask 是"快照"——相机一移动就作废。
Phase 4A 先让 mask 持续跟踪物体，Phase 4B 再将活跃 mask 持久化为 Pin。
Pin 锚定 canonicalPoint，重访时可重新 decode 恢复，不只是静态截图。

Must Reuse:
CanonicalFrame / FrameGeometry / LetterboxTransform
SAMEncoder / SAMDecoder / TapInstanceManager（来自 Phase 3）

New Modules:
- DriftDetector      — 事件型帧间漂移检测器
- ReAnchorLoop       — canonicalPoint 定期重 decode，mask 持续跟踪
- PinManager         — Pin 生命周期管理，含 canonicalPoint 持久化
- AnnotationView     — 标注编辑 UI
- PinStore           — 本地存储（CoreData 或轻量 JSON）

Carry-over Constraints（Phase 3 持续）:
- R3 禁令参数不变：minComponentPx=30、cap60、cap85 等
- SAMDecoder.swift / MaskRenderer.swift 继续冻结
- 单 encoder 实例、单几何链、FIFO Pool(max=3) 不变
- PinStore 写入必须异步，不得影响实时推理帧率

<!--

> ⛔ **PHASE 4（4A/4B/4C）— ARCHIVED / FROZEN（2026-08-24）。**
> 冻结裁决见 `architect_output.md` §26（D-26.1）。以下内容整体保留作历史执行记录与证据存档，
> **不是当前活跃计划** —— 当前活跃计划见本文件末尾的 PHASE 5 章节。
> R34 / R36 / R37 三项保留项随冻结显式携带进入 Phase 5，未随本节归档而视为解决。
> 本区块以 HTML 注释包裹，仅为让活跃计划（Phase 5）在渲染视图中更醒目；
> 原始文本未被删除，随时可搜索/取消注释还原。

------------------------------------------------------------
PHASE 4A — Live Mask Foundation（Days 1–3）
------------------------------------------------------------

------------------------------------------------------------
Day 1 — Phase 3 收尾 + 基础埋点
------------------------------------------------------------

## Debugger
- [x] **慢路径补测 + UI 裁决数据层分析**（原题「慢路径 UI 最终裁决（基于 §29 数据）」；裁决权属 Architect）——✅ 完成（2026-08-16，debug_report §31/§32/§33）
  - 补测结果：慢路径 e2eMs **mean=804.5 ms / p95=931.0 ms（合并 n=22 = 本轮 17 + §29 的 5）**，max=1105.4 / min=663.4 / median=788.0 / sd≈106.6
  - **n=22 ≥ 20 ⇒ 本项目第一个「p95 ≠ max」的慢路径估计**（p95 取索引 `ceil(0.95×22)−1`=20，即第 21 小），满足 §32.5 的结构性要求
  - >1000 ms 超限率 **4.5%（1/22）**；>800 ms 占 45.5%（10/22）
  - 数据层建议（§31.4）：维持 Tier 1、门控线由 800 ms 修订为 1000 ms —— ✅ **已被 Architect 采纳**，最终裁决见 architect_output §15
  - 慢路径 UI 语义裁决结果：**Tier 1（持续脉冲 + 12 s 超时 + 显式失败提示）转 🔒 FINAL，不引入慢路径专属 UI，不引入进度条**（§15.3 D-15.1）

- [x] **D-7' 六段埋点**（先于任何延迟优化）——✅ 完成（Builder 落地代码 + Debugger 真机 Release 采集，2026-08-16）
  - 六段：lock（stateLock）/ decide（判定，慢路径吸收 encode）/ qwait（decoderQueue 排队）/ decode / post（后处理+主线程跳）/ total
  - **快路径 n=49**：lock 0.00 / decide 0.31 / qwait 0.25 / **decode 63.70** / **post 16.69** / **total 80.97 ms**（p95 97.30）
    → 占比：decode **78.7%**、post **20.6%**，其余各段合计 <1%
  - **慢路径 n=17**：lock 0.00 / decide **748.2（93.9%）** / qwait 0.14 / decode 36.4 / post 14.4
  - ✅ **原目标达成（以否定形式）：280–310 ms 未归因残差已不存在**（post 仅 16.69 ms）⇒ §24.3 的 (a)/(b)/(c) 三分叉**全部结案，无需任何修改**（architect_output §15.8.1）
  - ✅ `lock` 段 66 样本全部 <0.05 ms ⇒ stateLock 无竞争，§10.4 要求 A 的锁设计得到验证（§15.8.2，并预先否决对该锁的优化立项）
  - ✅ `qwait` 0.14–0.60 ms ⇒ R4「decode 堆积」在当前节奏下**未触发**（**非已排除**，R4 保持 OPEN，§15.6.2）
  - ⚠️ **新 ISSUE：decode 段快慢路径倒挂** —— 快 63.7 ms（sd 9.3, n=50）vs 慢 36.4 ms（sd 7.3, n=17），Welch t=12.42、Cohen d=3.28，热节流已排除 ⇒ 架构侧立为 **ISSUE-P4-DECODE，P1，排在 Day 2–3 之后**（§15.7）
  - 结果写入 debug_report.md §33

## Builder
- [x] **tap 锚点常驻编号上屏**（L3 必需载体，补齐 W-1 欠载）
  - 每个 TapInstance 的 anchor marker 上显示实例序号（1、2、3）
  - 序号随 FIFO 淘汰更新（被淘汰实例编号释放）
  - primary 实例序号加粗或高亮，secondary 实例正常显示
  - 验收：同屏三实例时序号清晰可辨，区分度明显提升

## Architect
- [x] **慢路径 UI 最终裁决**（基于 Debugger 补测数据）——✅ 完成（2026-08-16，architect_output §15）
  - 实测 **p95 = 931.0 ms（n=22）< 1000 ms** ⇒ 走「维持当前动画」分支：**Tier 1 转 🔒 FINAL**（持续脉冲 + 12 s 超时 + 显式失败提示；不设进度条、不设慢路径专属 UI、UI 不得按 `TapPath` 分支、`tapProcessing` 保持布尔量）
  - ⚠️ **两条门控线冲突已显式解决**：§14.5.3 入口点 1 的 **p95 ≤ 800 ms**（实测超出 131 ms，字面不通过）vs 本条的 **1000 ms**（通过）。裁决**不按文档层级**（二者出自同一作者同一次输出），按实质：**800 是「分量相加 = 管线预算」线，回答不了感知学问题；1000 是有文献支撑（Miller 1968 / Nielsen 1993）的思维流中断线** ⇒ **修正案 AMD-15.1：800 ms 作为 UI 门控 ⛔ SUPERSEDED**（§15.1–15.2）
  - 🔁 **800 ms 未被删除**：重新界定为「慢路径管线预算符合性线」，**实测 931.0 仍未通过（+16.4%），作为性能缺口保持开放**；成因已 100% 归因（tap 触发 encode 溢价 +100.2 ms、慢路径 decode 折价 −24.6 ms、静态预测漏掉的 post 项 +14.4 ms）（§15.2.3 / §15.5）
  - 新门控（合取）：**U-1 p95 < 1000 ms ∧ U-2 >1000 ms 超限率 ≤ 10% ∧ U-3 n ≥ 20**（实测 931.0 / 4.5% / 22，三条全过）；撤销条件 V-1…V-4（析取）
  - 1105.4 ms 单样本（1/22 = 4.5%）**不改变裁决**：四条独立理由，统计量不做裁剪（§15.4）
  - 🆕 **长期约定 A-6（进度语义准入）**：准入条件是「存在可观测的进度量」而非「延迟够长」；慢路径 93.9% 落在单次不可分割的 CoreML encode 内 ⇒ **即使 p95 是 1105 ms，进度条依然是错的**（§15.3.3）
  - 保留项更新：**R1 完全结案并除名** / **R4 保持 OPEN**（未触发≠已排除）/ **W-1 关闭**（锚点编号升格为冻结的 L3 载体，不得移除）/ **A-1 状态不变**（17/17 `sel=ch0` 仅为机制旁证，E-1…E-4 未被推进）（§15.6）
  - 🆕 **裁决 D-15.2（Day 2 契约预置约束）**：re-anchor 节流必须**负载自适应**（同时最多一个在途批次、丢弃而非排队），不得用固定 100 ms 间隔 —— 算术依据：N=3 在 decode=63.7 与 36.4 两种取值下**都装不下** 100 ms 窗（191.1 / 109.2 ms）⇒ 缺陷在节流设计而非 decode 绝对值（§15.7.3）
  - 裁决写入 architect_output.md §15 ✅

------------------------------------------------------------
Day 2–3 — Re-anchor 刷新循环（核心基础设施）
------------------------------------------------------------

> ✅ **本区块已 CLOSED（2026-08-17）—— 裁决与完整关闭记录见 architect_output.md §20。**
> 关闭依据是本区块末尾的 **⏱️ STOP RULE**（用户裁定）第 4 轮真机两条判据 **D-1c'' / D-6'' 均 PASS** ⇒ 命中「通过」分支。
> **`DriftDetector.reAnchorEnabled` 以 `true` 发布**（Builder 已落地，注释同步重写，`BUILD SUCCEEDED`），§18 的「暂停合入」即刻解除；`reAnchorConsistencyGateEnabled` 仍以 `true` 发布（§17.4，两个开关语义相反，不得混用）。
> **能力 A（有界刷新）+ 能力 B（语义保持）均已交付**；**能力 C（目标跟随）维持推迟**，归 `ISSUE-P4-TRACK`（P2，§17.8.4 四条重开条件一字未改，合取不成立 ⇒ 不重开，⛔ 不得在 Phase 4B 夹带）。
> ⚠️ **关闭依据是 STOP RULE，不是 §18.3.5 的六条合取门** —— 该合取门有 1 条未满足（D-4' 三实例 qwait 从未执行），如实记录在 §20.2.2，不得表述为「全部满足」。
> ⛔ **本区块内不得再开展任何 re-anchor 工作**（STOP RULE 禁止事项继续有效）。所有未清偿项已转出，去向逐条见 §20.4 / §20.5。
>
> 📌 **记录维护（2026-08-18，architect_output §21）—— ⛔ 不是重开，本区块状态不变。** debug_report §36 的 Phase 4B 启动前烟测（用户预先裁定「记录，不修」）产生的读数已落账于 **§21**：**R4-c 首次获得测量值**（n=32，max **4.9 ms**，距 §20.4.2 重开触发条件 (iii) 的 > 5 ms 线仅 **0.1 ms**，⛔ 按实测记录余量、**不表述为「通过」**，R4 状态不变、构造界 ≈61 ms 不变）；**§16.6.1 静默降级路径首次在真机上被观测**且行为与契约逐条相符（节流槽位正常释放 ⇒ §16.2.3 的死锁未发生）；RE-2 轮转次序异常**判定为符合选择规则、非缺陷**；**R21 维持关闭、数字不改**（唯一增量：第三个合法接受样本 0.60）。⚠️ 证据来源为**用户粘贴的会话转录、无归档日志文件**，等级低于 §35，引用必须带此句。⛔ **D-4 / D-4' 仍不勾选** —— §36 的三实例 qwait 读数（max 0.5 ms / n=27）是**读数落账，不是验收**（非 D-4' 采集协议 + 转录级证据 + STOP RULE 禁止本区块内的 re-anchor 后续工作）。新增保留项 **R32**（冻结态恢复路径存在但不可发现）与 **R33**（陈旧度上界在极端连击下不由世代周期严格保证），**均已转出本区块**，去向见 §21.8 / §21.7.3。

## Architect
- [x] **定义 re-anchor 架构契约**
  - 漂移检测器接口：`DriftDetector.hasDrifted(from: FrameGeometry, to: FrameGeometry) -> Bool`
  - 漂移阈值（初始值，待真机调参）：平移 > X pt 或旋转 > Y° 触发 re-anchor
  - Re-anchor 触发：用现有 `embeddingCache`（若有效）+ 同一 `canonicalPoint` 重新 decode
  - Re-anchor 不得触发 encoder（仅 decode）；encoder 仍由 background refresh 管理
  - 失败策略：decode 失败时 mask 保持旧帧，不清除（静默降级）
  → ✅ 完成（2026-08-16，architect_output.md §16）：DriftDetector 接口（hasDrifted, translationThresholdPt=10pt, rotationThresholdDeg=3°, OR 条件）+ ReAnchorLoop 5 条触发条件 + D-15.2 负载自适应节流（in-flight flag + 丢弃而非排队）+ per-instance decodeRequestId 生成计数器（tap 增/re-anchor 仅读）+ [REANCHOR][inst#N] 日志格式（qwait+decode 字段）+ embedding nil 静默跳过 + 失败保留旧 mask + 8 条禁止事项 + ISSUE-P4-DECODE 被动采集规范 + Debugger 验收准则（含 qwait<50ms 判据）
  → 🔄 **修订（2026-08-17，architect_output.md §17）**：Builder 上交的 §16.3.1 缺陷经独立读源码核实成立，`letterboxToSquare`（CameraManager.swift:3280）的 `padX/padY/scale` 只由「相机 buffer 尺寸 + 固定 640 输入」决定，与镜头指向无关 ⇒ **§16.3.1 / §16.1.1 实现体 / §16.1.2 阈值表 / §16.3.2 论断标 ⛔ SUPERSEDED**（§16 其余各节一字未改，含 §16.2 节流、§16.4、§16.5、§16.6、§16.7、§16.8）
    - 复核中发现**第二个更深的缺陷**：§16.7 冻结 `canonicalPoint` 与「mask 随目标物体移动」自相矛盾 —— 锚点不动时 re-anchor 只能重新分割**同一画面位置**，任何漂移信号都到不了「跟随物体」⇒ **本次同时是范围重划**：Phase 4A 交付「有界刷新 + 语义保持」，**目标跟随另立 `ISSUE-P4-TRACK`（P2，重开条件见 §17.8.4），不占 Phase 4B 排期**
    - **新信号：锚点邻域内容散度**（对 `latestCameraBuffer`，CameraManager.swift:2645，32BGRA 采集直出、与 canonical 空间 1:1 无变换、videoQueue 同队列读）—— 8×8 采样 + 3×3 盒平均 + 去均值 MAD，单位 luma level，< 50 µs/帧，无新框架/队列/锁。否决 YOLO 框漂移（COCO 80 类覆盖率不可控 + 它测的是消费不了的跟踪量）、CoreMotion（看不见静止相机下的物体运动 + 纯新依赖）、Vision 配准（新增每帧 ANE 负载会污染尚未结案的 ISSUE-P4-DECODE）
    - 配套新增 **mask 一致性否决门**（新旧 alpha 的步长 IoU < 0.5 则保留旧 mask，复用既有失败分支）—— 无它则新信号在平移时会把用户选中的物体悄悄换成背景，是净退化；否决式改动，不碰 R3、不碰 tap 路径、不影响 §16.8 采集
    - 阈值单位改变、不可迁移：`contentThresholdLuma=8.0` / `anchorWindowPx=96` / `anchorGridSide=8` / `minReAnchorIntervalMs=300`（D-15.2 明文允许的下界）/ `reAnchorAcceptIoU=0.5` / `reAnchorConsistencyGateEnabled=true`。**`qwait:` / `decode:` 日志字段逐字符不变 ⇒ §16.9.3 的 grep 与 D-4 判据继续有效**
    - 验收修订：**D-1「偏差 ≤ 15 px」撤销**（在 §16.7 下任何信号都不可能通过，且 15 px 判据本身依赖目视真值 = A-5 形态），替换为 D-1a 该刷新时刷新 / D-1b 静止 10 s 零触发 / D-1c 平移离开目标时不得跳到别的物体 + D-6 接受率（观测指标，不设通过线）。**D-2 / D-3 / D-4 / D-5 原样保留**
    - `forceDriftForTesting` **保留**，重新定性：数据「对机制有效、对行为无效」，引用时必须标注开关状态；三条合取移除条件见 §17.7
    - **Builder 三处自主决定全部追认并升格为契约**：`batchId` 世代号（补上 §16.2.3 我留下的死锁洞）/ `drawableInstances()` 批次成员（新信号下升为必要条件）/ `capturedEmbedding` 锁内快照（交接指令有误 —— `embeddingCache` 由 encoderQueue 写、videoQueue 读，锁外读是真实数据竞争；§10.4 A 禁的是锁内重活不是 O(1) retain）
    - 新增长期约定 **A-7**（信号存在性核验：先读赋值点，自检「这个量是被测出来的还是被算出来的」）/ **A-8**（不可取消工作的复位纪律：必须回答「已在飞的工作落地时会做什么」）/ **A-9**（交接指令与同步不变量冲突时以不变量为准，判据是写者所在队列）；新增保留项 **R20–R23**
    - ⚠️ 本 Day 2–3 区块内 **Debugger 第 1、2 条的目标陈述本身有误**（描述的是已推迟的「目标跟随」能力），需按 §17.8.2 改写；第 3、4 条不受影响。**Architect 未代改、未动任何复选框**
  → 🔄 **补救裁决（2026-08-17，architect_output.md §18）**：Debugger 按 §17.8.2 完成真机验收（debug_report §34），结果 **6 PASS / 2 FAIL / 1 观测项 / 1 判别失效**。两条 FAIL（D-4 / D-1c）经复核**均为设计层缺陷，非实现缺陷** —— Builder 忠实实现了 §16+§17，两条失败都直接来自 Architect 的条款
    - **D-4 补救**：核实 `decode → buildTapAlpha` 对固定输入是纯函数（`SAMDecoder` 只有模型惰性缓存、`MaskRenderer` 唯一存储属性是常量 `inputSize`），而 `canonicalPoint` 被 §16.7 冻结 ⇒ **同一 embedding 世代内的重复 re-anchor 逐位输出相同**。会话中 15 次 encode 对 91 次 re-anchor decode ⇒ **≥50% 是可证明的空转**。⇒ **RE-1 embedding 世代门**（每实例每世代最多解码一次，无损删除）+ **RE-2 批次恒为 1**（候选按各自 `d_i` 过滤后取「最久未刷新」者轮转，消除批内累积）。**50 ms 判据线不动**（挪线即 §15 明令禁止的形态；且不需要挪 —— 违规的那部分工作本就不该存在）。**§16.7 七条禁令一条未放宽**；「给 re-anchor 独立低优先级队列」方案已否决（ANE 是单一物理资源、会污染未结案的 ISSUE-P4-DECODE、且真正的问题是工作量不是调度）
    - **D-1c 补救**：诊断采纳 —— 否决门比的是**上一次刷新**的 alpha（`CameraManager.swift:2135` 取 `instance.maskAlpha`，而 re-anchor 成功时自己就调 `updateMask`:2254）⇒ 相邻步 IoU 链对端到端偏离**无蕴含**。缺陷在 §17.3.3 的措辞（写「原有 alpha」，脑中想的是 tap 时刻那一张）。⇒ **RE-3：比较基准改为冻结的 `originAlpha`**（tap 路径唯一写入，re-anchor 永不写）。确立可陈述的不变量：**实例显示过的每一张 mask 与用户 tap 产出的 mask 的 IoU ≥ 0.5**。**恢复语义 REC-1（转回来自动恢复，冻结基准的免费红利）/ REC-2（重新 tap 重写原点）/ REC-3（FIFO/C4）**；吸收态定性为**正确终态**而非病；「高阈值棘轮」折衷已定量否决（0.9^33≈0.03，紧度不是种类）
    - **§17 两半必须分开搬运**：**§17.3 信号选择 ✅ 成立**（216 条自然触发，`d_i` 跨 0.6–125.1 lum，阈值 8.0 无需调参）；**§17.3.3 的安全声明 ⛔ 被证伪**（门存在、实现正确、开关打开、每次都执行了比较，产品仍变差）。机制留用、基准更换、安全声明撤回（重新成立以 D-1c' 通过为条件）。⛔ 禁止误读为「门可删」或「§17 应回退」
    - **§16.8.2 判据表 ⛔ 作废（VOID）**：假说可叠加而表是二选一（实测 (a) 贡献 **+11.50 ms**，n=8, d=1.19 ⇒「部分成立非主因」，而表只允许写「排除 (a)」）/ (b) 只被定义为 (a) 的补集且 36.4 ms 参照点混杂（`forceSlowPath` 同时清缓存+停 refresh）/ (c) 判别需时间戳而日志无时间戳。**ISSUE-P4-DECODE 保持 OPEN，优先级 P1 → P2**，重开在 Phase 4B 之后；本轮只执行前置埋点 **B-11（日志行首单调时钟）/ B-12（`suspendRefreshOnly`）**
    - **D-3 判据文本更正**：⛔ 删除 `~590 MB ±15 MB`（那是 Phase 2⇄3 **模式切换**工况的平台值，与 `.tapToSegment` 常驻差 200–400 MB，照字面执行会得出「内存异常偏低」的荒谬读数）。改为**同会话单调性（Q1→Q4 ≤ +30 MB）+ §27.7 的 N=0→3 = +31 MB 同工况参照**；并更正 Architect 自己对 374.5 MB 峰值的 FIFO 误归因（实为紧跟 9566.76 ms 冷启动 encode 的一次性瞬态，此刻 `n=1`）
    - **R4 CONFIRMED，不关闭**（关闭条件 = D-4' **∧** D-7 同时 PASS；D-4' 单独 PASS 不足够 —— 它由构造满足后已丧失对 tap 安全性的判别力）；**R20 关闭**（RE-2 使冗余在结构上不可能）；**R21 OPEN 重新计时**（0/216 读数作废，禁止在 D-6' 出读数前调参）；**R22 OPEN 仍未检验**（D-1b' 须加曝光剧变场景）；新增 **R24**（50 ms 线仍无推导，当前不构成约束但不得沿用）/ **R25**（冻结原点会否决合法大幅形变 ⇒ mask 冻结）/ **R26**（能力 A 的真实陈旧度界是 embedding 的 ~5 s，不是 300 ms）/ **R27**（Pin × re-anchor 交互未裁决，归 Day 5）
    - 新增长期约定 **A-10**（可叠加机制的判据须写分解、不得写二选一）/ **A-11**（残差不是证据：每条假说须有正向检验；参照工况只能差一个自变量）/ **A-12**（派生产物不可能比最陈旧的输入更新鲜）/ **A-13**（周期性计算前须核验输入在该周期尺度上会不会变）/ **A-14**（自反基准禁令：门控的历史基准若会被自己放行的写入更新，它约束的是相邻步不是端到端）；契约层纪律 **M-18.1**（Deliverable 前对每个输入量/基准量/被复位量逐个走 A-7/A-8/A-13/A-14 四问）
    - **Builder 变更清单 B-11 … B-17**（新增 5 / 替换 1 / §17.6 保留项全部继续有效），**不新增队列、不新增锁、不碰 `SAMDecoder.swift` / `MaskRenderer.swift` / `buildTapAlpha` / R3 禁令参数**
    - ⚠️ **Day 2–3 不关闭**（tasks.md D-4「暂停合入」维持），执行形态 = **`DriftDetector.reAnchorEnabled: Bool = false` 特性总开关**（与必须以 true 发布的 `reAnchorConsistencyGateEnabled` 语义相反，不得混用）；翻为 true 须 Architect 单独批准。**Phase 4B Day 4 准许并行开工**（三条硬隔离见 §18.3.3）。**Architect 本次只动 Debugger 九框与本 Architect 条目，Builder 的框一个未碰**
  → ✅ **关闭（2026-08-17，architect_output.md §20）**：STOP RULE 第 4 轮真机 **D-1c'' PASS**（无迁移；四次 20–47 倍面积的粗迁移被门拦下；**REC-1 自动恢复首次被观测到**：482→481/487、491→490/487）+ **D-6'' PASS**（11 条 `iou:` 读数中 **9 条 ≠ 1.00**，域 0.02–0.89；余 2 条 1.00 经 `origin`/`new` 面积列交叉校验确认为**同 embedding 世代的逐位相同 no-op**，非 §34/§35 的恒 1.00 缺陷）⇒ **否决率 45 %（5/11）**
    - **本节即 §18.3.5 条件 6 所要求的 Architect 单独批准**：`reAnchorEnabled` **false → true** 发布；「暂停合入」解除
    - **§18.3.5 六条关闭条件如实结账**（§20.2.2）：4 条满足 / 1 条满足带残留（D-1a' 后半句的空洞随门生效而消失，但未重测 ⇒ R25 无直接数据）/ **1 条未满足**（条件 3：D-4' 要求三实例在屏，第 4 轮全程单实例 ⇒ 未执行）。**关闭依据是 STOP RULE 而非合取门，不得掩饰**
    - **保留项处置**（§20.4）：**R21 关闭**（首获真实输入 45 %；并留下永久约束 —— 同物体轻微平移的 IoU 仅 **0.57 / 0.59** ⇒ `reAnchorAcceptIoU` **上界 ≈0.55**，上调空间几乎为零；下调则很快牺牲判别力，0.25 @ 1.7× 被否决证明**门不是面积过滤器**）/ **R4 重新划范围结项**（a、b 关闭；c「tap 排在 re-anchor 之后」从未测到但由构造定界 ≈61 ms 并被 D-7 的 195 ms 预算覆盖，⛔ **不得称「已排除」**，四条重开触发条件见 §20.4.2）/ **R22 OPEN 降 P1→P2**（门生效后误触发的后果降为一次浪费的 decode）/ **R25 OPEN 首获间接数据、严重性上调**（大幅形变几乎必然被否决 ⇒ 并入 `ISSUE-P4-TRACK` 立项材料）/ **R26 关闭**（陈旧度界 ≈5 s 第二次实测确认：本轮 7/10 个相邻间隔 ∈ 5.53–5.99 s ⇒ 转为能力 A 的对外措辞规范）/ **R28 OPEN 获量纲**（并**强化** §19.4「⛔ 不设相似度门」）/ **R24 状态不变** / **R27 责任加重**（§18.3.3 硬隔离第 3 条解除，保护对象全部转入 R27，归 Day 5）
    - **新增保留项 R31**：RE-1 的 no-op 缺口 —— `lastReAnchorEmbeddingGen` 只由 re-anchor 写、tap 路径不写 ⇒ **每次 tap 后的第一个 re-anchor 必为可证明的空转**（本轮 2/11 = 18 %）。一行可修，**正确性影响为零**，Owner **Builder**，目标阶段 **Phase 4B 之后与 ISSUE-P4-DECODE 同批（P3）**。⛔ **不构成第 5 轮 re-anchor 补救，不得在 Phase 4B 夹带**
    - **新增长期约定**（§20.6，均为 gate 缺陷特有、A-7/A-8/A-9/M-18.1 一条都抓不到的）：**A-17**（安全默认分支必须在日志/指标上与正常分支可区分 —— `else { return 1.0 }` 与「判断后放行」同形，使恒真守卫把整个门变成 no-op 而不留痕迹）/ **A-18**（**缓冲区的形状必须与缓冲区同行**，裸数组 + 另传 width/height 是一种可以自洽地错在多处的接口形态；A-15 的内存侧）/ **M-20.1**（**激活证据先于质量证据**：新机制第一条判据必须是「它执行了几次」而不是「它执行得怎么样」）

## Builder
- [x] **实现 DriftDetector** —— ✅ 完成（2026-08-16，`JudgeE2/Interaction/DriftDetector.swift`）
  - 输入：两帧 `FrameGeometry`（含 `videoRotationAngle`、`letterboxOffset`、`scale`）
  - 输出：`Bool`（是否超过漂移阈值）
  - 内置调参接口：`DriftDetector.translationThresholdPt` / `rotationThresholdDeg`
  - 按 §16.1 实现：`hasDrifted(from:to:)` + `drift(from:to:)`（后者额外返回两分量量值供 `[REANCHOR]` 日志用），阈值为具名 `static var`（10.0 pt / 3.0°），两分量取 OR（§16.1.3）；`FrameGeometry` 补齐 §16.3.1 的 `letterboxOffset` / `scale` / `videoRotationAngle`，`invertViewPoint` 未动
  - ⚠️ **上交 Architect：§16.3.1 指定的两个漂移分量在本代码库中恒为常量。** `letterboxOffset` / `scale` 由 `letterboxToSquare` 从「相机 buffer 尺寸 + 固定 640 输入」算出，与镜头指向无关 ⇒ 平移相机时 `translationDrift ≡ 0`；`videoRotationAngle` 被量化到 {0,90,180,270}，且一变就先触发 C4 清池。⇒ **按字面实现的 `hasDrifted` 不会被相机平移触发**，D-1 与「`[REANCHOR]` 日志出现」在自然使用下不可达。选择真正随位姿变化的漂移信号属架构决策，未擅自引入，请修订 §16.3.1。临时提供 `DriftDetector.forceDriftForTesting`（默认 OFF，等同 `forceSlowPath` 的调试开关）使循环机制仍可真机测量
  - 🔄 **已按 §17 修订并重写（2026-08-17）**，上述 FrameGeometry 接口与 10 pt/3° 阈值 ⛔ 作废：
    - **B-1/B-2/B-3**：`DriftDetector.swift` 整文件重写 —— 新接口 `signature(from:atCanonical:)`（videoQueue only，只采 `latestCameraBuffer`，32BGRA 校验 + 3×3 盒平均 + 定长内联 64 字节 `AnchorSignature`，零堆分配）/ `divergence`（去均值 MAD，单位 luma level）/ `hasDrifted(AnchorSignature)` / `alphaIoU`（stride 4 就地双指针，无中间数组）；**旧 FrameGeometry 重载与 `Drift.translationPt/rotationDeg` 全部删除，不留兼容重载**；六个 §17.4 常量全部具名 `static var`，调参接口形态不变
    - **B-7**：`TapInstance.anchorSignature` + `TapInstanceManager.setAnchorSignature`，用**既有 `lock`**，未新增锁，`drawableInstances()` 过滤条件未动
    - `forceDriftForTesting` 保留（§17.7），默认 OFF，注释按「对机制有效 / 对行为无效 + 引用数据必须标注开关状态」重新定性
    - 编译：**BUILD SUCCEEDED**，零 error、零 warning
  - 🔄 **§18 补救（2026-08-17，B-16）**：新增 **`DriftDetector.reAnchorEnabled: Bool = false`** —— 特性总开关，**以关闭状态发布**，注释写明与必须以 `true` 发布的 `reAnchorConsistencyGateEnabled` **极性相反、不得混用**，翻为 `true` 须 Architect 单独批准（§18.3.5）
    - `reAnchorAcceptIoU` **数值仍为 0.5**，但注释按 §18.2.5 改写：它所测的量已从「相邻两次刷新的相似度」变为「与用户原始选择的相似度」，并写入 R21 的禁令（D-6' 出读数前不得调参）
    - **算法与其余五个常量、`signature` / `divergence` / `alphaIoU` 的实现 —— 一字未动**
  - 🔧 **缺陷修复（2026-08-17，ISSUE-P4-GATE / debug_report §35.6.3，B-18/B-19/B-20）**：`alphaIoU` **自落地起从未执行过一次真实比较** —— 调用点按 §17.3.3 的错误前提传 `origW × origH = 2 073 600`，而全项目的 alpha 恒为 **256×256 = 65 536** 字节 ⇒ 首条守卫 `a.count >= width*height` 恒假 ⇒ 恒返回安全默认 1.0，否决分支不可达。§34 的 0/216 与 §35 的 0/76 都是在「实际没有门」的系统上测的
    - **B-18**：新增 `static let maskAlphaSide = 256`（硬不变量，非可调参数，注释列出 `MaskRenderer.swift:84/:184` 与 `TapInstanceManager.swift:60` 三条出处），调用点改传它
    - **B-19（同一 commit）**：`alphaIoU` 默认 `stride: 4 → 1`。256² 上 stride 4 只剩 4 096 采样点，80–431 px 的 mask 只命中 5–27 个，p≈0.5 处 SE ±0.10–±0.22 ⇒ 门在阈值处退化为掷硬币；全遍历 65 536 元素 ≈30–60 µs，相对 decode 可忽略。参数保留
    - **B-20**：doc comment 中「alphas are origW × origH bytes (≈2.07 M) / 1–2 ms / stride 4 保留 130 k 点」三处错误成本推理**全部重写**为真实值，并逐字记录旧前提如何使规格、注释、调用点自洽地错在一起
    - **`reAnchorAcceptIoU` 仍为 0.5，未动**（§35 已证 0.5 在此几何下判别力充分：迁移事件 IoU 上界 0.037–0.099，同物体形变 0.99+）；`reAnchorConsistencyGateEnabled` 仍 `true`；守卫的「无法比较即返回 1.0」安全契约未改 —— 门仍只在有证据时否决
    - **`reAnchorEnabled` 由重测期间的 `true` 改回 `false`**（§18.3.5 条件 6；Day 2–3 未关闭，「暂停合入」维持）
    - 编译：**BUILD SUCCEEDED**，零 error、零 warning

- [x] **实现 ReAnchorLoop** —— ✅ 完成（2026-08-16，`JudgeE2/Interaction/ReAnchorLoop.swift` + `CameraManager.checkAndFireReAnchor` / `reAnchorDecode`）
  - 在每个视频帧回调中：若 `currentMode == .tapToSegment` 且 `tapInstances.count > 0`
    → DriftDetector 检测当前帧 vs 上次 anchor 帧
    → 若漂移：对每个 TapInstance 异步在 `decoderQueue` 重新 decode
    → 更新 `maskImage`（通过现有 `DispatchQueue.main.async` 路径）
  - ~~帧率控制：re-anchor 每 100 ms 最多触发一次~~ ⛔ **SUPERSEDED by §16 D-2 / D-15.2** —— 改为**负载自适应节流**：同时最多一个在途批次，超出的漂移事件**丢弃而非排队**，且不取消在途批次（固定 100 ms 窗在 decode=63.7 与 36.4 两种工况下对 N=3 都装不下）
  - 落地要点：§16.2.1 五条件全 AND；`checkAndFireReAnchor` 在 `refreshTapEmbeddingIfNeeded` **之后**调用且两个调用点都挂（tapToSegment 下 2/3 的帧走 `% 3 != 0` 提前 return 分支）；T3 取在 `decoderQueue.async` 调用点**之外**（debug_report §33.2.3）；`driftPt/driftDeg` 为触发本批次的量值，闭包内不重算；`[REANCHOR][inst#N]` 行在任何质量门控**之前**发出；decode 失败保留旧 mask（§16.6.1）；`ReAnchorLoop` 带 `batchId` 世代号，`reset()` 自增以让 C4 时无法取消的在途闭包成为 no-op；`resetReAnchorState()` 挂到全部三个清池点
  - 无新队列、无新锁；不调用 encoder；不修改 `canonicalPoint`；`SAMDecoder.swift` / `MaskRenderer.swift` 一行未动
  - 编译：**BUILD SUCCEEDED**，零 error
  - 验收：`[REANCHOR][inst#N] drifted %.1fpt/%.1fdeg → qwait: %.1fms decode: %.1fms` 日志格式已按 §16.2.5 逐字符落地（待 Debugger 真机验证；自然使用下能否出现取决于上一条的 §16.3.1 修订）
  - 🔄 **已按 §17 修订（2026-08-17）**，节流状态机 / `batchId` / 三处清池点 / T3 位置 / 失败保留旧 mask / 「日志在质量门控之前发出」**全部原样保留**，改动仅限：
    - **B-4/B-5**：`checkAndFireReAnchor` 条件序按 §17.5.3 重排为 1 mode → 2 前台 → 3 `drawableInstances()` → **4 时间下界 `minReAnchorIntervalMs=300`（排在任何像素采样之前）** → 5 embedding → 6 逐实例采样取 `maxDivergence` → 7 认领 + embedding 锁内快照 → 8 推进 `lastReAnchorFireMs` 与各实例 `anchorSignature`（批次开始即更新基线，§16.4.2 时机不变）→ 9 派发；`ReAnchorLoop.lastAnchorGeometry` 降级为遥测、`seedAnchorIfNeeded` 删除
    - **B-6**：日志前缀改为 `drifted %.1flum`（该实例自身的散度，非批次 max）；**`qwait:` / `decode:` 字段逐字符未变 ⇒ §16.9.3 grep 与 D-4 判据不受影响**
    - **B-8**：`CameraManager.lastReAnchorFireMs`，videoQueue 独占，无同步
    - **B-9/B-10**：一致性否决门落在 `buildTapAlpha` 之后、`updateMask` 之前（**`buildTapAlpha` 零触碰**，否决只落回既有「保留旧 mask」分支）；受 `reAnchorConsistencyGateEnabled` 单开关控制；否决行 `[REANCHOR][inst#N] rejected — mask IoU %.2f < %.2f, keeping previous mask` 用 `perfLog`（与 `[REANCHOR]` 同级，quiet mode 下不被抑制，否则 D-6 接受率只剩一侧）
    - **未实现能力 C**：无 tracker、未引入 CoreMotion / Vision、`canonicalPoint` 只读 —— 按 §17.2.0 归 `ISSUE-P4-TRACK`
    - 编译：**BUILD SUCCEEDED**，零 error、零 warning
  - 🔄 **§18 补救（2026-08-17，B-13/B-14/B-15/B-17 + 前置埋点 B-11/B-12）**：两条 FAIL 均为设计层缺陷，按 §18 裁决落地；**§17 的工作未回退任何一处**，节流状态机 / `batchId` 世代号 / `pendingCount` / 单在途批次不变量 / 三处清池点 / T3 位置 / 「日志在质量门之前发出」/ `drawableInstances()` 批次成员 / `capturedEmbedding` 锁内快照 —— **全部一行未动**
    - **B-13（RE-1 embedding 世代门）**：`CameraManager.embeddingGeneration: UInt64`，与 `embeddingCache` **同一 `stateLock`、同一次加锁读出**，在四个「新算出的 embedding 写回」点自增（置 nil 不自增）；`TapInstance.lastReAnchorEmbeddingGen` 用 `TapInstanceManager` **既有 `lock`**（同 B-7），派发时写入不等 decode 返回。资格过滤置于第 5 步之后、**第 6 步像素采样之前** ⇒ 全实例已消费当前世代时整帧跳过采样
    - **B-14（RE-2 批次恒为 1）**：`TapInstance.lastReAnchorAtMs`；候选 = 「未消费当前世代」∩「**自身** `d_i` 越阈」，取最久未刷新者、平局取 `slotIndex` 小者 ⇒ 确定性公平轮转；`beginBatch(count: 1)`。**只推进被选中实例的 `anchorSignature`**（落选者基线不动，否则轮转会退化为「只刷第一个」—— §18 未明写，此处为 Builder 判断，见 builder_progress）
    - **B-17**：`maxDivergence` 批次取 max 逻辑**整段删除，未保留开关**；`forceDriftForTesting` 语义未失（已折叠在 `Drift.exceedsThreshold` 内）
    - **B-15（RE-3 冻结原点基准）**：`TapInstance.originAlpha`，写入点**唯一** —— `updateMask(..., recordOrigin: true)`，只有 tap 路径传 `true`，参数默认 `false` 使 re-anchor 调用点**因省略而正确**；与 `maskAlpha` 同一次加锁写入。否决门比较对象 `maskAlpha` → `originAlpha`；**门的位置、开关、日志行格式、失败分支全部不变**。`anchorSignature` 与 `originAlpha` 推进条件不一致 = §18.2.3 的**有意职责分离**，**未做对齐**，理由已写进字段注释以防后人顺手对齐
    - **B-11**：`perfLog` / `diagLog` / `faultLog` / `quietSummaryLog` 统一加 `[t=%.1f] ` **行首**单调时钟前缀（`PerfLogging.lineStampMs()`）；既有 tag 与字段的相对位置逐字符未变 ⇒ §16.9.3 / §33 的全部 grep 不受影响
    - **B-12**：`CameraManager.suspendRefreshOnly: Bool = false` —— `refreshTapEmbeddingIfNeeded` **仅 early-return，不清缓存**，与 `forceSlowPath`（清缓存**且**停刷新，故参照工况复合）并存且正交，不合并
    - **无新队列、无新锁**（§16.7 七条禁令逐条维持）；re-anchor 不调 encoder、不改 `canonicalPoint`；`SAMDecoder.swift` / `MaskRenderer.swift` / `buildTapAlpha` / R3 禁令参数 **零触碰**
    - 编译：**BUILD SUCCEEDED**，零 error、零 warning
    - ⚠️ 交付状态：**`reAnchorEnabled = false` 发布**，等待 Debugger 按 §18.3.4 重测集（D-1c' / D-4' / D-7 / D-6' / D-3' / D-1a' / D-1b'）验收
  - 🔧 **缺陷修复 + 埋点（2026-08-17，ISSUE-P4-GATE P0 / D-6' 埋点缺口 P1，debug_report §35.6.3 / §35.7.2，B-18/B-21）**
    - **B-18（P0）**：否决门的维度实参 `Int(lb.origW)` / `Int(lb.origH)` → `DriftDetector.maskAlphaSide`，并显式 `stride: 1`。**这一行是门从未生效的全部原因**；门的位置、开关、否决后保留旧 mask 的行为均未改
    - **B-21（P1，§35.7.2 字段规格）**：`[REANCHOR]` 行尾追加 `| iou: %.2f origin: %dpx new: %dpx`；`iou` 在门**外**、且在 `originAlpha` 解包**外**算出 ⇒ 开关关闭或无原点时该行照样有数，打印 `n/a` 而非编造值；`origin` = `originAlpha` 非零计数（就地扫描 ~20 µs，未在 `TapInstance` 上缓存），`new` = 既有 `built.nonzeroCount`。否决分支既有文本一字未改，同样追加 `origin:` / `new:` ⇒ 两分支字段集一致，一条正则可提取
    - **`qwait:` / `decode:` 逐字符不变**，前缀串原样保留，新字段一律在行尾
    - ⚠️ **对日志读者的唯一可见后果**：新字段在 `buildTapAlpha` 跑完前不存在，故 `[REANCHOR]` 行改为「decode 后**构造**、门处**发出**」；§16.8 不变量由「四个 early-return 点各显式发一次只含前缀的行（位置在各自故障行之前）+ `defer` 兜底 + 幂等」保住。⇒ 凡走到 `buildTapAlpha` 的路径，`[REANCHOR]` 行现在出现在该次 `[TAP#g] candidates` 行**之后**，**§35.6.4 的配对法不再适用**（也不再需要 —— `new:` 已把面积写在行内）。已写入源码注释
    - **未试图在「让门真的跑起来」之外修 D-1c'**；无新队列/新锁；re-anchor 仍不调 encoder、不改 `canonicalPoint`；`SAMDecoder.swift` / `MaskRenderer.swift` / `buildTapAlpha` / R3 禁令参数 **零触碰**
    - 编译：**BUILD SUCCEEDED**，零 error、零 warning。交付状态仍为 **`reAnchorEnabled = false`**，等待 §18.3.4 重测集验收

## Debugger

> 🔄 **验收判据已按 architect_output §17.8 修订（2026-08-17）。**
> ~~原 D-1：缓慢平移 3 秒，mask 随目标物体移动，偏差 ≤ 15 px~~ ⛔ **撤销**。
> 撤销理由**不是**「新信号做不到」，而是**该判据与 §16.7 自相矛盾，在任何漂移信号下都不可能通过**（§17.1.4）：`canonicalPoint` 冻结 ⇒ re-anchor 产出的 mask 锚在**画面坐标**上、不锚在**物体**上 ⇒ 相机平移时 mask 的正确行为恰恰是**不跟着物体走**。15 px 判据度量的是**能力 C（目标跟随）**，而 Phase 4A 交付的是**能力 A（有界刷新）+ B（语义保持）**。
> 能力 C 完整保留为 `ISSUE-P4-TRACK`（§17.2.0，P2，不占 Phase 4B 排期），重开条件见 §17.8.4。
> 🔓 **重开（2026-08-24，architect_output §24）**：用户明确要求开启 re-anchor 补救轮，构成 §17.8.4 条件 3/4 所需的「新的、范围明确的裁决」授权。§24 逐条核对四条重开条件：① 已满足；② `ISSUE-P4-DECODE` **仍未结案**（不阻塞设计交付，但阻塞 `objectTrackingEnabled` 翻真）；③④ 由 §24.1/§24.5 本节交付。架构契约 = §24 全文；Builder 清单 B-42…B-48、Debugger 判据 P-10…P-17 见下方新区块。
> 新判据**比原判据更难通过**：原判据只要求 mask 位移正确，新三条要求「该刷新时刷新、不该刷新时不刷新、错刷时能自我否决」**同时成立**。

> 🔄 **验收已执行（2026-08-17，iPhone 11，Release，`forceDriftForTesting=false`）：debug_report §34 —— 6 PASS / 2 FAIL / 1 观测项 / 1 判别失效。**
> 补救裁决见 architect_output **§18**。**Day 2–3 不关闭**；re-anchor 以 `DriftDetector.reAnchorEnabled = false` **暂停合入**，翻为 true 须 Architect 单独批准（§18.3.5 六条关闭条件）。
> 下方九框：PASS 者已勾选；**两条 FAIL 与两条读数作废项保持未勾选，并就地记录失败与去向**（不留白）。重测集见本区块末尾「§18 重测集」。

- [x] **D-1a（该刷新时刷新）** —— 静止相机 + 变化目标
  - 操作：手持稳定对准一个会变形/小幅移动的目标（人的手、被轻推的物体），tap 之，**保持相机不动**，让目标在锚点上小幅变化 3 秒
  - 判据：出现 ≥1 条 `[REANCHOR]` 且**未**被否决；mask 目视上跟上了目标的新形状
  - ⚠️ `DriftDetector.forceDriftForTesting` **必须为 false**（§17.7：该开关对机制有效、对行为无效，不得用它掩盖信号不触发）
  - ✅ **PASS**（§34.5）：测试 B 段 `[REANCHOR]` 125 条、0 否决，录屏中 mask 全程稳定贴在鼠标上。**新信号在真机上是活的** —— 216 条自然触发，`d_i` 跨 **0.6–125.1 lum**（有量纲、宽分布、随场景变化），`forceDriftForTesting=false` 有三重证据
  - 📌 这是本轮**唯一明确的正面架构结果**：约定 **A-7（信号存在性核验）首次应用即命中**；阈值 `contentThresholdLuma=8.0` 被数据支持，**本轮无需调参**（噪声底 <8.0；真实运动 median 15.8–25.7；无贴线抖动病态形态）
  - ⚠️ 边界：本条只说明信号「会响」，不说明「响得对」——「响了之后做的事对不对」由 D-1c 覆盖，**那一条 FAIL**（§18.6：§17 两半必须分开搬运）
  - 🔄 **须重测 D-1a'**（§18.3.4）：判据不变，但**必须包含一个大幅形变目标**（手的开合 / 人转身）以量化 **R25**（RE-3 的冻结原点会否决合法大幅形变）

- [x] **D-1b（不该刷新时不刷新）** —— 全静止
  - 操作：tap 之后相机与场景**都**保持静止 10 秒
  - 判据：`[REANCHOR]` 行数 **= 0**
  - 不通过处理：出现即为误触发 ⇒ 上调 `DriftDetector.contentThresholdLuma`（当前 8.0）后重测，**并记录调后值**
  - ✅ **PASS**（§34.2）：`[REANCHOR]` 行数 = **0**。旁证：全部 216 条触发的 `d_i` 最小值 0.6 lum ⇒ 噪声底远低于 8.0 阈值
  - ⚠️ **证据等级低于其余八条**（§34.1.1）：该段只有用户现场叙述两行中文，**文件里没有那 10 秒的原始 console 输出**，无法独立复核。若将来要引用为「阈值 8.0 已验证」的依据，须补采
  - 🔄 **须重测 D-1b'**（§18.3.4）：**必须留原始 console 日志**，且**必须新增一个曝光剧变场景**（镜头扫过窗户 / 开关灯）—— 这是 **R22**（去均值只对**加性**亮度偏移免疫，乘性增益挡不住）迄今**从未被检验**的唯一窗口。若失败，处置按 §17.10 R22 原文：改用归一化互相关，**不得继续上调阈值**

- [x] **D-1c（错刷时自我否决）** —— 相机平移离开目标 ⭐ **本次修订最重要的一条**
  - 操作：tap 一个物体，然后缓慢平移相机**直到该物体完全离开锚点**
  - 判据：mask **不得**悄悄变成另一个物体的 mask。合法结果二选一：
    - (i) 出现 `[REANCHOR][inst#N] rejected — mask IoU %.2f < %.2f, keeping previous mask` 且 mask 保持原样
    - (ii) 无 `[REANCHOR]` 触发
  - ⛔ **若 mask 跳到了别的物体上 ⇒ FAIL**
  - ⛔ **FAIL**（§34.4）——**本轮两条阻塞项之一，且是 §17.8.2 自己标注「本次修订最重要的一条」**
    - 实测：mask 在 **≥3 个互不相关的物体**间迁移（鼠标垫一小块 → 桌面+笔记本边缘 → 硬盘盒 → 电源适配器）；全程 **0 条 `rejected`**（否决率 0/216）
    - 不可反驳的一格证据：**10 s → 20 s 期间 HUD 的 `#N` 恒为 13**（`lastTapIndex` 每次 tap 自增）⇒ 这 10 秒内一次 tap 都没发生，而 mask 从一小块变成覆盖画面右下 1/3 ⇒ **形变只可能来自 re-anchor**
    - **诊断（已对源码核实）**：`CameraManager.swift:2135` 的 `previousAlpha = instance.maskAlpha` 取的是**当前屏上的 alpha**，而 re-anchor 成功时自己就调 `updateMask`（:2254）⇒ **第 k 次比较的基准是第 k−1 次 re-anchor 自己的产物**。相邻步 IoU ≥ 0.5 的一条链对 `IoU(M_0, M_k)` **没有任何蕴含**；`minReAnchorIntervalMs=300` 下的缓慢平移正好制造这条链 ⇒ mask 不是「跳」过去，是**连续地爬**过去，而只检查相邻步的门**按构造看不见连续的爬行**
    - **`alphaIoU` 实现本身正确**（`DriftDetector.swift:316`，就地双指针、`union==0` 返回 1.0），`previousAlpha` 不可能为 nil（`drawableInstances()` 已过滤）⇒ **门没算错，它问错了问题**。缺陷归属：**§17.3.3 的措辞**（写「原有 alpha」，实现上唯一可指的就是 `maskAlpha`），不是 Builder
    - ⇒ **去向：architect_output §18.2 —— RE-3（比较基准改为冻结的 tap 原点 `originAlpha`）+ 恢复语义 REC-1/2/3**；重测条目 **D-1c'**（§18.3.4，须同时验证 REC-1「转回来自动恢复」）
    - ⇒ 新增长期约定 **A-14（自反基准禁令）**：门控用「历史值」作接受判据时，必须回答「这个历史值会不会被本门控放行的写入更新？」自检问法：「如果这个门连续放行 100 次，最后一次的输入还剩多少原始语义？」
  - ✅ **本条以修订判据 D-1c'' 关闭（2026-08-17，§20.1.4 / §20.1.5）—— 注意：勾选依据是 D-1c''，不是本行原写的 D-1c。**
    - 第 4 轮（iPhone 11，Release，`reAnchorEnabled=true`，`forceDriftForTesting=false`，~73 s，单实例）：**mask 全程稳定在 ~490 px 的目标上，未发生迁移**；11 个 `[REANCHOR]` 单元中 **5 个 rejected**，其中四次是面积 **20.1× / 30.1× / 36.9× / 47.5×** 的粗迁移（`iou` 0.02–0.03）—— **正是 §34.4（431→10935→719）与 §35.6 里产生两次 FAIL 的同一失效形态，这一次被拦住了**
    - **REC-1（转回来自动恢复）首次被验证**：#2/#3 被否决后 #4（481 px）、#5（487 px）自动接受；#9 被否决后 #10（487 px）自动接受 ⇒ **物体回到锚点下刷新自动恢复，无需重新 tap**（§35.9.1 条件 2 曾明确记为「未被验证」）
    - **为什么可以取代两次 FAIL**：§34 的系统用**自反基准**（比上一次刷新自己的产物，已由 RE-3 更换）；§35 的系统里**门从未执行过一次比较**（ISSUE-P4-GATE，维度实参 1080×1920 vs 256×256）。⇒ **第 4 轮是第一个真正带门的系统**，两次 FAIL 测的不是同一个系统
    - ⚠️ 证据边界：依据是录屏 + `origin:`/`new:` 面积列，**不是**对「mask 是否仍是同一物体」的独立真值标注（系统里没有该真值，同 R28）。本条证明的是「**未发生粗迁移**」，不是「mask 始终精确正确」

- [x] **D-2（原样保留）** —— 快速甩动相机
  - 判据：mask 不消失；**旧位置保留视为 PASS**（降级为旧帧而非消失）
  - ✅ **PASS**（§34.8.3）：甩动结束后屏上**同时存在两个 mask**（贴鼠标 + 贴适配器/盒体），锚点编号均在位；无 mask 消失、无渲染崩溃、无编号泄漏。佐证：全日志 `keeping stale mask` = 0 ⇒ masks 是被正常刷新保留的，不是靠失败分支保住的
  - ⇒ §16.6.1「降级为旧帧而非消失」在甩动工况下按设计工作。**§18 补救不改变失败降级路径 ⇒ 不重测**

- [x] **D-3（原样保留）** —— 内存
  - ⛔ ~~判据：与 ~590 MB ±15 MB 基线一致~~ **判据文本已更正（§18.5）** —— `~590 MB` 是 **Phase 2⇄3 模式切换**工况的平台值（切换过程中两条 pipeline 先后驻留），与 `.tapToSegment` 单模式常驻差 **200–400 MB**，照字面执行会得出「内存异常偏低」的荒谬读数
  - **修订后判据 D-3'**：re-anchor 连续运行 ≥30 s，**同一会话内不得单调增长** —— 第一四分位均值 → 第四四分位均值不得高出 **+30 MB**；辅助参照（同工况，非达标线）为 §27.7 第 4 项的 **N=0→3 = +31 MB**
  - ✅ **PASS**（§34.8.1）：会话首末 211.2 → 210.9 MB（净 −0.3）；Q1→Q4 **311.8 → 209.5 MB（−102.3）**；独立口径（`FPS:|Memory:` 行，n=89）同向同量级
  - 📌 **更正一处 Architect 自己的误归因**：374.5 MB 峰值**不是** FIFO 池行为 —— 它紧跟 `[TAP] background embedding refresh 9566.76 ms`（冷启动首次 encode）且此刻 `pool=… n=1`，是**一次性瞬态**；三实例在屏时段内存反而稳在 337–343 MB，池满时未创新高
  - 🔄 **须重测 D-3'**（§18.3.4）：RE-3 的 `originAlpha` 新增常驻内存 **+2 MB/实例**（COW，首次 re-anchor 成功前与 `maskAlpha` 共享），N=3 上限 **+6 MB**，计入预期不计入超标

- [ ] **D-4（原样保留，R4 判据）** —— qwait
  - 判据：`[REANCHOR]` 行中 **qwait max < 50 ms**
  - ⛔ **≥50 ms ⇒ R4 证实，上报并暂停合入**
  - 提取：`grep "\[REANCHOR\]" console.log`（`qwait:` / `decode:` 字段在 §17 修订中逐字符未变，§16.9.3 的 grep 脚本可直接沿用）
  - ⛔ **FAIL**（§34.3）——**本轮两条阻塞项之一；R4 由「未触发」正式转为 CONFIRMED**
    - 实测 qwait **max 189.90 ms = 判据的 3.80 倍**；自然会话 mean 40.16 / p95 165.20；**36.3%（33/91）≥50 ms，15.4%（14/91）≥100 ms**
    - **自变量已隔离为「批内实例数 ≥2」，不是 re-anchor 本身**：单实例 n=125 → max **1.5 ms**（与 Phase 3 tap 同分布）；N=2 → 135.0；N=3 → 189.90。批内**首个**单元恒为 ~0.2 ms ⇒ 调度器无问题，成本全部来自串行 `decoderQueue` 上排在前面的单元
    - §16.2.3 的 **191.1 ms 算术是被实测确认的下界**（实测批次占用 mean 205.2 / max 244.6）
    - ⛔ **decode 优化解决不了它**：即使 ISSUE-P4-1 的 ~30 ms 全部回收，N=3 的 qwait max 仍约 80 ms
    - 📌 **§15.6.2 拒绝关闭 R4（「未触发 ≠ 已排除」）被完整证实** —— 若当初按实测 0.8 ms 关闭 R4，本轮的 189.9 ms 会是无人预期的回归。**M-15.3 这条方法学纪律本身记为本轮的正面结果之一**
    - ⇒ **去向：architect_output §18.1 —— RE-1（embedding 世代门，删除 ≥50% 可证明的空转 decode）+ RE-2（批次恒为 1，轮转选择）**。**50 ms 判据线不动**（挪线是 §15 明令禁止的形态，且不需要挪 —— 违规的那部分工作本就不该存在）；**§16.7 七条禁令一条未放宽**
    - ⇒ 重测条目 **D-4'**（§18.3.4，**必须在三实例在屏工况下采集**）**＋新增 D-7**（tap 与 re-anchor 的争用）。⚠️ **D-4' 单独 PASS 不足以关闭 R4** —— D-4 由构造满足后已丧失对 tap 安全性的判别力
  - ⊘ **结项：未在合格工况下执行 —— STOP RULE 取代，保持未勾选**（2026-08-17，去向见 §20.4.2）
    - 第 4 轮 11 条 qwait **全部 ≤ 0.2 ms**（decode mean ≈53.2 ms），但**全程单实例**（TAP#2 短暂产生 `#2`，随即被双击 `clearAll` 清除），而 D-4' 明文要求**三实例在屏**。§35 已把 qwait 的自变量钉死为**批内实例数** ⇒ **单实例读数对三实例域无判别力（M-15.3）**，⛔ **不得计为 D-4' 通过，故不勾选**
    - ⇒ **R4 重新划范围结项**（§20.4.2）：**R4-a**（re-anchor↔re-anchor 累积）**关闭**（RE-2 结构性消除 + §35.4.1 实测 50/52 ∈ 0.1–0.6 ms + 本轮 11/11 ≤ 0.2 ms）；**R4-b**（re-anchor 排在 tap 之后，§35 的 63.8/26.6 ms）**关闭且方向与 R4 相反** —— 那是 tap 优先得到保障的证据，不属 R4 的风险面；**R4-c**（**tap 排在 re-anchor 之后**，R4 真正的风险面）**从未在被专门制造的条件下测到**，但由构造定界（单在途 + 批次恒为 1 ⇒ tap 最坏只等一次 decode ≈61 ms，叠加快路径 p95 97.3 ⇒ ≈158 ms < D-7 的 195 ms 判据线）⇒ 已被 D-7 预算覆盖，不再独立跟踪
    - ⛔ **不得表述为「R4 已排除」。** 四条重开触发条件（任一命中即自动重开，且须先按 **R24** 给出 50 ms 线的推导，不得沿用也不得挪线）：(i) 重新引入多单元批次或多个在途 re-anchor；(ii) 在 `decoderQueue` 上新增第三类工作（Pin / tracking / 任何后台 decode）；(iii) `[D7'] qwait` 出现 > 5 ms 的样本
    - 🔓 **触发条件 (i) 命中，重新纳入观测（2026-08-24，architect_output §24.6.3）**：B-45（RE-1 资格判定扩展为并集，见下方 re-anchor 补救轮区块）使同一实例可能在同一 embedding 世代内被多次派发。判据沿用已有推导的 195ms 线（§18.3.4/§20.4.2），**不新起一条线**（M-15.3：新条件域须重测，不代表要重新推导判据本身）。验收判据 = **P-17**。

- [x] **D-5（原样保留，V-3 判据）** —— 慢路径自然发生率
  - 判据：自然使用中**不出现 ≥5 次慢路径 tap**
  - ⛔ 触发即命中 §15 的撤销条件 V-3 ⇒ 慢路径 UI 裁决（D-15.1）需重开
  - ✅ **PASS**（§34.8.2）：22 次 tap 中 `fast/decode-only` **21** 次、`slow/parked` **1** 次（会话首 tap，等待冷启动 encode）、**`encode=own`（真慢路径）0 次** ⇒ 最宽口径 1 次 < 5 ⇒ **V-3 未触发，G-5 维持，D-15.1 不需重开**
  - 📌 机制上亦如预期：15 次 encode **全部**来自 background refresh ⇒ **§16.7「re-anchor 不得调用 encoder」在真机上得到验证**
  - ⇒ **§18 补救只减少 decode、不新增 encode ⇒ 不重测**

- [x] **D-6（新增）** —— re-anchor 接受率
  - 统计：一次自然使用会话中 `rejected` 行数 / `[REANCHOR]` 行数
  - ⚠️ **这是观测指标，不设通过线**（§13 裁决一 (a)：不得把观测指标写成验收目标）
  - 用途：接受率 ≈ 100% ⇒ 否决门形同虚设，考虑上调 `reAnchorAcceptIoU`（当前 0.5）；接受率 ≈ 0% ⇒ 刷新从未生效，等价于 re-anchor 没上线
  - 背景：R21 记录该门**零真机数据**，`reAnchorAcceptIoU = 0.5` 是工程判断而非实测值，D-6 是它唯一的观测窗口
  - 📊 **已执行，读数作废，保持未勾选**（§34.6 / §18.3.7）
    - 实测：`rejected` **0 / 216 = 0.0%** 否决率 ⇒ 按 D-6 自身的读数规则「**否决门形同虚设**」。且这不是碰巧没遇到坏样本 —— **恰恰在门唯一该响的那个场景（D-1c，mask 跳到别的物体）里，它也没有响**
    - ⛔ **该读数不构成对 R21 的任何输入**：它来自**错误的比较基准**（相邻两次刷新）。反证 —— 把阈值提到 0.9 只会让缓慢平移更易被否决，而快速跨越在链式比较下仍可能因中间步的连续性通过。**阈值调参改变的是灵敏度，不改变比较对象。** 且其中相当一部分比较的是**两个逐位相同的 alpha**（同一 embedding 世代内的重复 decode，§18.1.2），IoU 恒为 1.0
    - ⇒ **不勾选的理由**：勾选会让后来者误以为 R21 已有数据。**R21 保持 OPEN 并重新计时**；⛔ **在 D-6' 出读数之前，不得调整 `reAnchorAcceptIoU`**（数值保持 0.5，但其含义已从「相邻两次刷新的相似度」变为「与用户原始选择的相似度」）
    - ⇒ 重测条目 **D-6'**（§18.3.4）：RE-3 落地后 D-6 才**第一次成为良定义的观测窗口**。仍为观测指标，**不设通过线**
  - ✅ **本条以 D-6'' 关闭（2026-08-17，§20.1.3）—— 勾选依据是 D-6''，不是上方已作废的 D-6 读数。**
    - **`iou:` 字段第一次是真实读数**：11 条中 **9 条 ≠ 1.00**，取值域 **0.02 – 0.89**，连续分布跨两个数量级；余 2 条 1.00 是**真读数**（同 embedding 世代内重复 decode ⇒ 按 §18.1.2 的纯函数性质逐位相同，且同时打印 `origin == new`），**不是** §34/§35 的恒 1.00 缺陷 —— 后者会在 `origin` 与 `new` 相差 20–47 倍时也打印 1.00，而本轮第 2/3/7/9 条恰恰没有
    - **否决率 45 %（5/11）**。按本条自身的读数规则：既非 ≈100 %（门形同虚设）亦非 ≈0 %（刷新从未生效）⇒ **落在合理区间，`reAnchorAcceptIoU` 无需调参**
    - ⇒ **R21 关闭**（§20.4.1）。调参禁令解除，但同时落下一条**永久约束**：同物体、面积几乎未变的两次合法刷新 IoU 只有 **0.57 / 0.59** ⇒ 阈值**上界 ≈0.55**，上调空间几乎为零；`0.25 @ 1.7×` 被否决 ⇒ **门不是面积过滤器**，任何以面积比替代 IoU 的简化提案在本数据上已被预先否决
    - ⚠️ 证据边界：**n = 11，单会话单设备**。45 % 的二项 95 % CI ≈ **[21 %, 72 %]** ⇒ 有信息量的是「它同时远离两个病态端点」，⛔ 不得把 45 % 当作可外推的稳态否决率；严格地说门在本轮被真正考验了 **9** 次（2 条 no-op 不携带判别信息）

- [ ] **ISSUE-P4-DECODE 被动采集（§16.8，无需额外 session）**
  - 从同一批 `[REANCHOR]` 日志提取 decode 值，与 §33.2 的快路径 decode（mean 63.70 ms）、慢路径 decode（mean 36.4 ms）对比
  - re-anchor decode ≈ 36.4 ms ⇒ 支持假说 (a) background refresh 并发争用
  - re-anchor decode ≈ 63.7 ms ⇒ 排除 (a)，指向 (b) 内存局部性 或 (c) DVFS 时钟爬坡
  - ⚠️ **判别设计失效，不作判别，保持未勾选**（§34.7 / §18.4）
    - 数据本身采到了：re-anchor decode **mean 61.06 / sd 10.01 / n=216**，单峰无双峰。**字面读数是「排除 (a)」，但该读数不可采信。**
    - **⛔ §16.8.2 的三行判据表作废（VOID）**，三条独立理由：
      1. **假说可叠加而表是二选一。** Debugger 用「encode 在飞」的 15 个窗口做出了 Architect 没有设计的分组对照：**并发组 n=8 mean 71.97 vs 非并发组 n=83 mean 60.48，差 +11.50 ms（Welch t=2.55, Cohen d=1.19）** ⇒ 正确结论是 **「(a) 部分成立、非主因」**（解释 27.3 ms 中的 11.5 ms），而判据表**只允许写「排除 (a)」** ⇒ **按该表结案会得到与数据方向相反的结论**
      2. **(b) 从未有过正向检验** —— §16.8.1 里 (b) 的判别方式写的是「同上比较」，它**只被定义为 (a) 的补集**；且其 36.4 ms 参照点本身混杂（`forceSlowPath` 同时清 `embeddingCache` **且**停 background refresh）⇒ **整条比较轴的原点是脏的**
      3. **(c) 的判别在本数据上不可执行** —— 判别方式原文是「观察时序相关性」，而全部 2338 行日志**没有任何时间戳**。行序代理给出 ~6 ms 的弱梯度（n=7），方向与 (c) 一致但远小于 27 ms
    - **「免费数据」前提部分作废**：数据免费 ✅ 成立；「n ≥ 20 即可支撑判断」⛔ 不成立 —— **n 必须按工况格子计**，216 条里落在决定性格子的只有 **8 条**
    - ⇒ **ISSUE-P4-DECODE 保持 OPEN；优先级 P1 → P2**，重开时机为 **Phase 4B 完成之后**（它是效率问题不是正确性问题；RE-1 落地后 decode 总量降半，影响进一步缩小）
    - ⇒ **本轮只买期权、不行权**：执行两项前置埋点（归 Builder）—— **B-11** 日志**行首**加单调时钟前缀（不得插入既有 tag 与字段之间，以保护 §16.9.3 / §33 的全部 grep）、**B-12** `suspendRefreshOnly`（只 early-return refresh，**不清缓存**，与 `forceSlowPath` 并存且正交）。析因协议（2×2、**每格 n≥20**、同运行内交错、encode→decode 间隔扫描）见 §18.4.3，本轮不排期
    - ⇒ 新增长期约定 **A-10**（可叠加机制的判据须写**分解**，不得写二选一；「互斥」本身必须先被论证）/ **A-11**（**残差不是证据**：每条假说须有自己的正向检验；用于分离两条假说的参照工况只能在**一个**自变量上不同）
  - ⊘ **结项：判别设计仍作废，本轮无可用分组 —— STOP RULE 取代，保持未勾选**（2026-08-17，§20.7）
    - 第 4 轮 decode mean ≈53.2 ms（n=11），但**全程单实例、无 encode 并发窗口分组、n 远低于 §18.4.3 要求的每格 n≥20** ⇒ 对 (a)/(b)/(c) 三条假说**无判别力**。⛔ 勾选会让后来者误以为判别已完成
    - ⇒ **ISSUE-P4-DECODE 状态不变：OPEN，P2，重开时机为 Phase 4B 完成之后**；析因协议见 §18.4.3（2×2、每格 n≥20、同运行内交错）。前置埋点 **B-11 已落地**（行首单调时钟，§35.1.2 确认 §34.1.5 的时间戳缺口已消除）
    - ⇒ **同批重开时一并处理**：**R31**（RE-1 的 no-op 缺口，2/11 = 18 % 单元可证明空转，一行可修，减少的正是 decode 总量）与 **R22** 的 `baseLuma` 埋点（§35.9.2 第 6 项）。⛔ 三者均**不构成继续做 Day 2–3 的理由**

------------------------------------------------------------
§18 重测集（Debugger，补救落地后执行）—— architect_output §18.3.4
------------------------------------------------------------

> 前置：Builder 完成 **RE-1 / RE-2 / RE-3 + B-11…B-17**，且构建中 `DriftDetector.reAnchorEnabled` 可切换。
> 重测须在 **三实例在屏** 的工况下进行（本轮多数条目的支撑数据来自单实例，条件域不同 —— M-15.3）。

> ⊘ **本重测集已结项（2026-08-17）—— 逐条去向见 architect_output §20。**
> 第 3 轮（debug_report §35）执行过全部 7 条并给出判定；**第 4 轮（STOP RULE）只覆盖 D-1c' 与 D-6' 两条**，其余 5 条**未在第 4 轮执行**。
> ⛔ **下方 7 个复选框一律保持未勾选** —— 它们是第 3 轮的历史记录，且其中数条的合格工况（三实例在屏 / 大幅形变 / 曝光剧变）**从未被制造出来**。把未执行的条目勾成完成，是本项目明令禁止的形态。
> 标记约定：**⊘ = 结项但未通过验收**，与真正的 `[x]` 区分。每条写明「未测什么 ⇒ 哪条保留项仍无数据」。

**必测（5 条）**

- [ ] **D-1c'（比较基准更换后的自我否决）** —— ⊘ **结项：由第 4 轮 D-1c'' 取代并通过**（§20.1.4）。第 3 轮 FAIL（431→10935→719，0/76 否决）的真因是 ISSUE-P4-GATE（门从未执行），非本判据不可达；**判据 (b) REC-1 已验证**（482→481/487、491→490/487）。⚠️ 第 4 轮为**单实例**工况
  - 操作：tap 一个物体 → 缓慢平移相机直到物体完全离开锚点 → **再平移回来**
  - 判据 (a)：mask **不得**变成别的物体。合法结果：(i) 出现 `rejected` 行且 mask 保持原样；(ii) 无触发
  - 判据 (b)：**REC-1 自动恢复必须被验证** —— 转回来之后刷新应自动恢复（这是冻结基准相对链式基准的核心优势，未验证等于未交付）

- [ ] **D-4'（qwait，三实例在屏）** —— ⊘ **结项：合格工况从未执行。** 第 3 轮字面 FAIL（max 63.8 ms，两个越线样本经时间戳反推**均在等一次 tap**）；第 4 轮 qwait ≤ 0.2 ms 但**全程单实例** ⇒ **「三实例在屏的 qwait」始终无数据**。⇒ R4 按 §20.4.2 **重新划范围结项**（a/b 关闭，c 未测但由构造定界 + D-7 预算覆盖），⛔ 不得称「R4 已排除」；重开触发条件见 §20.4.2
  - 判据：`[REANCHOR]` qwait **max < 50 ms**（线不变）。预期 max ≤ 2 ms（批次恒为 1）
  - ⚠️ 附加签名检查：若仍出现 `d_i ≤ contentThresholdLuma` 的 `[REANCHOR]` 行 ⇒ **RE-2 未正确落地**，须上报（R20 的关闭依据是机制，此为其验证）

- [ ] **D-7（新增）—— tap 与 re-anchor 的争用** ⭐ **R4 关闭的必要条件** —— ⊘ **结项：第 3 轮 PASS 但采集条件欠压，未复测。** p95 **108.3 ms**（判据线 195 的 55.5 %）、`[D7'] qwait` max **0.5 ms**（没有一次 tap 等过队列）；但高频 tap 段 re-anchor 速率仅 0.18/s ⇒ 期望碰撞 ≈0.95 次（tap 本身抑制 re-anchor：新实例 ⇒ `anchorSignature` 重新播种）⇒ **R4-c 仍无正面数据**，转为构造界
  - 操作：**三实例在屏 + re-anchor 活跃 + 刻意高频 tap**（≥15 次，间隔 <1 s）
  - 判据：`[D7'] total` **p95 < 195 ms**；同时记录 `[D7'] qwait` 的 max
  - **判据推导（本项目第一条给出推导的延迟门控线）**：tap 最坏情况 = 自身 total p95（§33.2 Release 快路径 97.30）+ 至多一个在途 re-anchor 单元的 decode（≈61）+ post（≈17）≈ 175 ms，留 ~11% 余量 ⇒ 195 ms
  - 背景：§34.3.6 明确指出该条件在本轮「基本没有出现过」，而 D-4 由构造满足后**已丧失对 tap 安全性的判别力**，必须由 D-7 顶上

- [ ] **D-6'（接受率，RE-3 之后）** —— ⊘ **结项：由第 4 轮 D-6'' 取代并通过**（§20.1.3）。第 3 轮读数结构性作废（0/76，门未执行）；第 4 轮 **45 %（5/11）**，`iou` 域 0.02–0.89 ⇒ **R21 关闭**（§20.4.1），并留下阈值上界 ≈0.55 的永久约束
  - 统计 `rejected` / `[REANCHOR]`。**观测指标，仍不设通过线**。这是 R21 唯一且第一次良定义的输入窗口

- [ ] **D-3'（内存，判据已按 §18.5 更正）** —— ⊘ **结项：第 3 轮 PASS，未复测。** Q1→Q4 **−55.7 MB**、首末 202.8→230.3（+27.5，≤ +30）；第 4 轮单实例旁证同向（`Mem=` 208→~232，峰值 ~398 紧邻 background refresh，**无单向增长**）。⚠️ §18.5.2 的 `originAlpha` **+2 MB/实例** 允量数值错误（同一维度错误前提），真实约 **64 KB/实例**、N=3 约 0.2 MB（§35.9.2 第 9 项 (iii)，本处更正，不影响 PASS）
  - 判据：同会话 Q1→Q4 均值增幅 ≤ **+30 MB**；辅助参照 N=0→3 = +31 MB；额外允量 `originAlpha` **+2 MB/实例**（N=3 上限 +6 MB）

**复测（2 条）**

- [ ] **D-1a'** —— 判据不变，但**必须包含一个大幅形变目标**（手的开合 / 人转身）以量化 **R25**（冻结原点会否决合法大幅形变 ⇒ mask 冻结，恢复靠 REC-1/REC-2）
  - ⊘ **结项：大幅形变场景从未在门生效的构建上执行 ⇒ R25 仍无直接数据。** 第 3 轮名义 PASS 但判据后半句在门失效下是恒真命题（空洞）；门生效后该空洞在结构上消失，但**没有人再测一次**。⚠️ **第 4 轮给出首条不利的间接数据**：同物体轻微平移 IoU 已降到 **0.57 / 0.59** ⇒ 大幅形变几乎必然落到 0.5 以下 ⇒ **R25 由「理论边界」升为「大概率真实存在」**，并入 `ISSUE-P4-TRACK` 立项材料（§20.4.4）。⛔ 处置不变：不得改回可推进基准、不得为它下调 `reAnchorAcceptIoU`
- [ ] **D-1b'** —— 判据不变，但**必须留原始 console 日志**（补上 §34.1.1 的缺口），且**必须新增一个曝光剧变场景**（镜头扫过窗户 / 开关灯）以首次检验 **R22**
  - ⊘ **结项：静止零触发部分 PASS（第 3 轮，原始 console 已留档 1017 行，有效窗口 28.35 s、其间 background refresh 5 次全部未触发）；曝光剧变场景从未执行 ⇒ R22 三轮之后数据量仍为零。** §35.8.2 已证明它在当前埋点下**原理上不可判定**（曝光/AE 与未触发时的散度都不进日志）。⇒ **R22 OPEN，P1 → P2**（能力 B 交付后，误触发的后果由「语义损坏」降为**一次浪费的 decode**，且被 `minReAnchorIntervalMs=300` 限流），Owner Builder（`baseLuma` 埋点）+ Debugger，**与 ISSUE-P4-DECODE 同批，Phase 4B 之后**。⛔ 处置不变：失败时改用归一化互相关，**不得继续上调 `contentThresholdLuma`**

**不重测：** D-2（补救不改变失败降级路径）、D-5（补救只减少 decode、不新增 encode）

> 📌 **本重测集的净残留（全部已转出，⛔ 均不构成继续做 Day 2–3 的理由）：** 三实例在屏的 qwait（→ R4-c，构造界 + D-7 预算覆盖，§20.4.2）/ 大幅形变（→ R25，`ISSUE-P4-TRACK`）/ 曝光剧变（→ R22，P2，与 ISSUE-P4-DECODE 同批）。

------------------------------------------------------------
⏱️ STOP RULE — re-anchor 迭代期限（用户裁定，2026-08-17）
------------------------------------------------------------

> ✅ **本规则已满足并结束（2026-08-17）—— 原文全部保留作为历史记录，⛔ 不得删除。**
> 第 4 轮真机已执行（iPhone 11，Release，`reAnchorEnabled=true`，`forceDriftForTesting=false`，单会话 ~73 s，11 个 `[REANCHOR]` 单元）：
> **D-1c'' PASS + D-6'' PASS ⇒ 命中下表「✅ 通过」分支** ⇒ Day 2–3 **关闭**、`reAnchorEnabled = true` **发布**、进 Phase 4B。裁决与完整记录：**architect_output.md §20**。
> 「⛔ 禁止事项」（不得开第 5 轮、不得为 re-anchor 新增补救裁决章节、不得以「再调一个参数」延长本区块）**在本区块关闭后继续有效**。

> **本规则的效力高于本区块内其他条目。所有角色（Architect / Builder / Debugger）在提出任何 re-anchor 后续工作前必须先读本节。**

**背景：** Day 2–3 已迭代四轮 —— §16 契约 → §17 换漂移信号 → §18 补救（RE-1/2/3）→ §35 修 ISSUE-P4-GATE。
其间 Phase 4B（Pin & 标注，本阶段的实际用户价值）一行未动。**继续迭代的边际收益已低于其占用的排期成本。**

### 期限：**第 4 轮真机，仅此一轮**

只测两条，其余五条已判定不重测：

- [x] **D-1c''** —— mask 不得迁移到别的物体（否决门自 §35 修复后**首次**真正执行比较）
  - ✅ **PASS**（§20.1.4）：录屏确认 mask 全程稳定在 ~490 px 的目标上，**未迁移**；11 个单元中 **5 个 rejected（45 %）**，其中四次是面积 **20.1×/30.1×/36.9×/47.5×** 的粗迁移（`iou` 0.02–0.03）—— 即 §34.4 / §35.6 两次 FAIL 的同一失效形态；另有一次 **`iou 0.25 @ 1.7×`** 被否决 ⇒ **门不是面积过滤器**
  - ✅ **REC-1 首次被验证**：被否决后目标回到锚点下，刷新自动恢复（482→481/487、491→490/487），**无需重新 tap**
- [x] **D-6''** —— `[REANCHOR]` 行的 `iou:` 字段出现**真实读数**（不再恒为 1.0），R21 首次获得输入
  - ✅ **PASS**（§20.1.3）：**9/11 条 ≠ 1.00**，域 **0.02–0.89**；余 2 条 1.00 经 `origin`/`new` 面积列交叉校验确认为**同 embedding 世代的逐位相同 no-op**（真读数，非旧缺陷）⇒ **R21 首次获得输入并关闭**（否决率 45 %；阈值上界 ≈0.55，见 §20.4.1）

### 二选一，不设第三种可能

| 结果 | 处置 |
|---|---|
| ✅ **通过** ⬅️ **实际命中（2026-08-17）** | Day 2–3 **关闭**；`reAnchorEnabled = true` 发布；进 Phase 4B |
| ⛔ **未通过** | **能力 B（语义保持）判定为不交付**。re-anchor 以 `reAnchorEnabled = false` 发布（**代码保留、功能关闭**），能力 B 并入 `ISSUE-P4-TRACK` 与能力 C 一并处理；Day 2–3 以「**能力 A 已交付、能力 B 未交付**」**照样关闭**；进 Phase 4B |

### ⛔ 禁止事项

- **不得开第 5 轮。** 不得为 re-anchor 新增补救裁决章节（§19 或其后继），除非用户明确要求。
- **不得**以「再调一个参数就好了」「再补一次埋点就能定位」为由延长本区块。若第 4 轮失败，判据是**关闭并降级**，不是继续诊断。
- Architect 若认为必须开第 5 轮，**只能向用户提出请求**，不得自行启动。

### 判定依据（为什么失败也可以关闭）

- **能力 A（有界刷新）已实测交付**：RE-1 使单元速率 1.02/s → 0.403/s（−60.5%，优于 §18.1.4 预测的 ≥50%）；RE-2 使 re-anchor↔re-anchor 堆积从 max 189.90 ms 降至 50/52 样本 ∈ 0.1–0.6 ms。这部分价值已落袋，不因能力 B 失败而回退。
- 若门真正执行后 D-1c'' 仍失败，则问题不在参数或实现，而在「**锚点冻结 + 单点 IoU**」这一组合本身 —— 那正是能力 C（tracking）的辖区（§17.2.0 / §17.8.4），在 Phase 4A 内强行凑合无收益。

### 并行要求

**Phase 4B Day 4 不得等待第 4 轮真机结果。** §18.3.3 已批准并行开工且耦合面为空；继续串行等待本身即是本规则要防止的失效形态。

> 📌 **结束后的一条排期记录（§20.6）：** 四轮里 Phase 4B 一行未动。期限没有改变任何技术判据，但它**改变了验收的形状** —— 从「再修一点再测一次」收窄为「两条判据、一轮、二选一」，而恰恰是这一轮拿到了三轮里最干净的数据（11 条可交叉校验的读数，胜过前两轮 216 + 76 条不可解读的日志）。**前三轮的问题从来不是测得不够多，而是每一轮都在测一个还没被证明在运行的系统**（新增方法学约定 **M-20.1：激活证据先于质量证据**）。

------------------------------------------------------------
PHASE 4B — Pin & Annotation System（Days 4–6）
------------------------------------------------------------

------------------------------------------------------------
Day 4 — 数据模型 + 本地存储
------------------------------------------------------------

> ✅ **准许开工（architect_output §18.3.3，2026-08-17）** —— **不被 Day 2–3 的两条 FAIL 阻塞。**
> 依据是耦合面为空：Day 4 三方任务均不触及 `CameraManager` / `decoderQueue` / `DriftDetector` / `TapInstanceManager` / 冻结文件。
> Builder 优先级：**§18 补救（P0）→ Day 4（P1）**；补救构建交付 Debugger 后**不必等重测结果**即可开始 Day 4，两者并行。
>
> ⚠️ **三条硬隔离（全部必须遵守）：**
> 1. **不得引用 `TapInstance` 的任何 re-anchor 字段**（`anchorSignature` / `lastReAnchorEmbeddingGen` / `lastReAnchorAtMs` / `originAlpha`）。Pin 的数据来源只能是 `maskAlpha` / `canonicalPoint` / `FrameGeometry` 这三个 Phase 3 已冻结的量
> 2. **不得新增任何在 videoQueue / decoderQueue 上执行的工作。** PinStore 的异步写入**必须使用自己的 I/O 上下文，不得复用 `decoderQueue`** —— 磁盘写入排进解码队列会重新制造 §18.1 刚消除的那类累积。（注：这**不是**放宽 §16.7 的「禁止新增队列」—— §16.7 约束的对象是 re-anchor 路径；PinStore 的队列归属属本 Day 4 的 Architect 裁决范围）
> 🔄 **第 3 条已于 2026-08-17 解除（architect_output §20.3.4）**：Day 2–3 已关闭、`reAnchorEnabled` 以 `true` 发布 ⇒ 不再存在 disabled 的发布构建，该条在字面上不可执行。**硬隔离第 1、2 条继续有效，不受影响。** 第 3 条的保护对象**全部转入 R27**（归 Day 5 Architect 裁决）。⚠️ 在 R27 裁决之前：§19.6 的 Day 4 验收判据 **P-1…P-5 不得引用任何 re-anchor 产生的 mask 状态**；若 Debugger 在 Day 4 验收中观测到 Pin × re-anchor 的交互异常，**记录并上报 R27，不得就地处置**。
> 📌 **硬隔离第 2 条获得实测依据（2026-08-18，architect_output §21.3）：** §36.4 实测在 `decoderQueue` **只有两类工作**（tap decode / re-anchor 单元）时，tap 的 `qwait` 已取到 **4.9 ms**，距 §20.4.2 重开触发条件 (iii) 的 5 ms 线仅 **0.1 ms**。⇒ 若把 PinStore 的 blob 写盘 / manifest 合并写 / 索引重建放上 `decoderQueue`，触发条件 **(ii) 与 (iii) 会同时命中**（(ii) 由构造、(iii) 由 IO 抖动，一次 flush 的量级远大于 0.1 ms）。**§19.3.1 的 `pin.store.io` 队列裁决维持不变，其依据由「纯构造论证」升为「构造论证 + 一条实测」；约束 PIN-3（热路径闭包内禁止出现 `PinStore` 符号，可静态核查）文本与范围不变、不新增第二条约束。**⚠️ 该实测的证据来源为粘贴转录、无归档日志。
> 3. ~~**Day 4 产物不得在 `reAnchorEnabled == true` 的构建上验收。**~~ ⛔ **已解除（见上）** Pin × re-anchor 的交互（保存 Pin 瞬间 mask 正被替换的竞态；已 Pin 的实例是否还应继续被刷新 —— Pin 的语义是「冻结这一刻」还是「持续跟踪这个东西」）**未裁决，记为 R27，归 Day 5 的 Architect 裁决，Day 4 不得夹带**

## Architect
- [x] **裁决 PinStore 存储方案** —— ✅ 完成（2026-08-17，architect_output **§19**；注：原条目写的「§15」是笔误，§15 已被 Day 1 慢路径 UI 裁决占用）
  - 候选 A：CoreData（适合大量 Pin + 查询）—— ⛔ **否决**（`NSManagedObject` 引用语义 + context 限定与本项目「值类型快照跨队列」纪律冲突 / `.xcdatamodeld` 二进制 schema 绕过评审 / `viewContext` 是主线程 I/O 诱饵 / N ≤ 1,000 时查询能力收益为零）。**SwiftData 同理否决**，并要求删除模板残留 `JudgeE2/Item.swift`（B-26）
  - 候选 B：JSON 文件（简单，适合 ≤100 Pin）—— ⚠️ **精神采纳、字面形态否决**（thumbnail 内联 JSON ⇒ base64 +33% 写放大，且把不可变 blob 绑上可变元数据）
  - ✅ **裁决 = 候选 C：`Application Support/Pins/manifest.json`（元数据，原子写）+ `masks/<uuid>.png`（256×256 无损 sidecar blob）**。推导依据：典型 **2–4 KB/Pin**，设计目标 **N ≤ 1,000 ⇒ ≤ 3 MB**，Day 6 三种查询（时间排序 / tag 筛选 / id 取单条）在常驻内存数组上皆为微秒级
  - **iCloud 同步：不在 Phase 4 范围**（决定性理由：Pin 是场景绑定的可重解指针，同步到没见过该场景的设备只剩截图 —— 恰是 §17/§18 宣布 Pin 不是的那个东西；换机需求由 Application Support 的 iCloud **备份**覆盖）。对冲 = `PinStore` 定为 protocol，`FilePinStore` 唯一实现
  - **队列裁定（§18.3.3 硬隔离 2 授权范围内）**：新增**恰好一条** `pin.store.io`（serial, `.utility`），serial 即互斥、**不新增锁**；⛔ 不复用 `decoderQueue`/`videoQueue`/`encoderQueue`；约束 **PIN-3** 可静态检查
  - **同时裁定并写入 §19**：Pin schema 逐字段 + 不变量 PIN-1（`canonicalPoint` 冻结）/ PIN-2（缩略图四条禁令）/ PIN-4（embedding 永不持久化）/ PIN-5（重访必须表达为合成 tap，不得建第二条 decode 入口）；schema 迁移策略；Day 4 验收判据 P-1…P-5（取代原「读取 < 50 ms / 重启后持久」两条 —— 二者由构造满足，不可能失败）；Builder 清单 **B-18 … B-28**
  - ⚠️ **指出 tasks.md Day 4/6 草案五处不成立**（只指出，不代改、不动他人复选框）：(1) `canonicalPoint` 注释「SAM 输入坐标（1024×1024）」**错误**，实为 **Canonical 像素空间 origW×origH**，→1024 的映射在 `PointPromptBuilder` 内部；(2) `frameGeometry: FrameGeometry` **不足以重解**（不含 `promptSpace`），且运行时结构不得直接 Codable；(3) `thumbnail` 128×128 改为 **256×256 无损**（256 才是系统不变量，128 会凭空引入一个未经裁决的重采样阈值）；(4) 缺 `updatedAt`，而 Day 6 AnnotationView 明写要显示「修改时间」；(5) Day 6「重访 decode → **恢复**活跃 mask」+「decode 成功率 ≥ 80%（相近场景下）」属 §16.3.1 / §17.1.4 同一缺陷类（**A-7 适用**）—— embedding 不持久化 ⇒ 重访是「在记忆点重新分割**当前**画面」而非恢复；替代判据见 §19.4.7，正式确立归 Day 6
  - 新增保留项 **R28**（重访的物体同一性不可核验，归 `ISSUE-P4-TRACK`）/ **R29**（`previewFile` 保留位，Day 5 与 R27 同批裁）/ **R30**（N ≥ 10k 出射程）；长期约定 **A-15**（不变量须随数据持久化，承 §35 教训）/ **A-16**（由构造满足的判据没有信息量，承 §17.8.3）
  - ⛔ 本裁决不触及 re-anchor，不构成 STOP RULE 所禁止的第 5 轮补救

## Builder
- [x] **定义 Pin 数据结构**
  ```swift
  struct Pin: Identifiable, Codable {
      let id: UUID
      let canonicalPoint: CGPoint     // SAM 输入坐标（1024×1024）
      let frameGeometry: FrameGeometry // 拍摄时几何快照
      let thumbnail: Data             // 128×128 mask 缩略图 PNG
      var tag: String?
      var note: String?
      let createdAt: Date
  }
  ```
  → ✅ 完成（2026-08-21，**按 architect_output §19.2 实现，不按上面这段草案**——§19.2.7 已裁定草案五处不成立）：
    运行时值类型 `Pin`（**不 Codable**，规则 P-A）+ 持久化 DTO `PinRecordV1` / `PinGeometryV1`（显式 `pointX`/`pointY` Double 键、epoch 秒、`previewFile` 恒 nil / R29）。
    `canonicalPoint` = **Canonical 像素空间（origW×origH）**，非 1024 空间；几何快照改为 `PinGeometryV1`（含 `promptSpace`，`FrameGeometry` 未加 `: Codable`）；
    缩略图 = **256×256 无损** sidecar blob（非 128×128，§19.2.6）；已补 `updatedAt`。新增文件 `JudgeE2/Persistence/{Pin,PinRecordV1,MaskPNGCodec}.swift`。详见 builder_progress.md
- [x] **实现 PinStore**
  - CRUD 接口：`save(_ pin:)`、`fetch(id:)`、`fetchAll()`、`delete(id:)`
  - 写入异步执行（不阻塞主线程）
  - 验收：Pin 写入 / 读取 / 删除 roundtrip 单元测试通过（≥ 3 条 Pin）
  → ✅ 完成（2026-08-21，B-18…B-26，**接口按 §19.7**）：`protocol PinStore` + 唯一实现 `FilePinStore`（`Application Support/Pins/manifest.json` + `masks/<uuid>.png`）。
    ⛔ **无覆盖式 `save(_:)`** —— PIN-1 禁止改写冻结字段，改为 `create` + `update(id:tag:note:)`，`create` 对已存在 id 逐字段校验冻结字段、不一致返回 `.frozenFieldMutation(field:)`。
    变更型 API 全部 `Result<Void, PinStoreError>` **主线程**回调，且 `.success` 蕴含「已落盘」（合并窗到期 flush 后才释放回调，满足 §19.6 P-3(ii)）。
    新增**恰好一条** `pin.store.io`（serial, `.utility`），**无新锁**；`videoQueue`/`decoderQueue`/`encoderQueue` 零投递（PIN-3 静态可查，已核：三个热路径文件内零 PinStore 符号）。
    另含：迁移脚手架（信封 `schemaVersion` + 有序迁移链数组[v1 为空] + 前向版本只读拒写 + `manifest.v<N>.bak`）、`[PIN]` 日志（B-11 单调时钟前缀，成功/失败同形可辨）、删除模板残留 `JudgeE2/Item.swift`（B-26，全工程零 SwiftData/CoreData 引用与链接）。
    **B-27 / B-28 仅落接口不实现**（Day 5 / Day 6）。BUILD SUCCEEDED（零 error、零 warning）。详见 builder_progress.md

## Debugger

> 🔄 **验收判据已被 architect_output §19.6 取代**（2026-08-21）：原两条「写入 10 条 Pin 读取 < 50ms」「重启后持久」**由构造满足、不可能失败**（§17.8.3 教训：判据要么可能失败要么没有信息量），改为 **P-1…P-5**。结果见 debug_report.md **§37**。
> ⛔ **出场条件（P-1 ∧ P-3 ∧ P-4 通过，P-2 / P-5 已报告）不成立** —— 但 §37 明确区分：三条门控是**未执行**（无驱动/前置条件已过期），不是 FAIL；静态层面 B-18…B-28 基本完整，不得读成「Day 4 代码不合格」。

- [x] **P-5｜热路径静态隔离**（必报）—— ✅ 完成（§37.2）。五组正则搜索 `videoQueue`/`decoderQueue`/`encoderQueue` 闭包，`PinStore`/`FilePinStore`/`MaskPNGCodec.encode` 全部 0 命中。⚠️ 限定：今天零命中是因为 Day 4 无调用方，Day 5/6 UI 接线后须重跑；`PinInterfaces.swift:83` 用 `extension CameraManager` 注入方法，检查面已不等于单文件。

- [x] **构建完整性核实**（未在 tasks.md 原表中，§37.3 追加项）—— ✅ 完成。Debug + Release 双配置全新 `derivedDataPath` 干净构建均 `BUILD SUCCEEDED`。**推翻 Builder 一条自评**：实测 13 条 warning（原报零），全在 `FilePinStore.swift`，根因是 pbxproj 的 `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor` 与 `pin.store.io` 队列纪律相反、全工程零 `nonisolated` 标注。pbxproj 五文件注册 / `Item.swift` 摘除 / SwiftData·CoreData 零链接均核实通过。

- [ ] **P-1｜Roundtrip 逐字段保真**（门控）—— ⛔ **未执行，无驱动**。需 10 条含边界值的合成 Pin（空 tag / 64 字符 tag / emoji+换行 note / `maskFile==nil` / 极小 mask），Day 4 无 UI、`makeRecord` 为空实现，无法产出。⚠️ 且规格本身有两处不可判定：「空 tag」= `""` 还是 `nil`（`create` 存 `""`、`update` 归一 `nil`，两者不一致）；「64 字符」按字素簇/UTF-16/scalar 哪种计数——修订判据需一并说清。

- [ ] **P-2｜读取延迟 + 线性度**（必报）—— ⛔ **未执行**，同上无驱动。§37.5 静态预判：全量载入不开 blob 已验证；唯一与 N 成正比的成本是 `collectOrphanBlobs` 的目录枚举，若 P-2 未来失败第一嫌疑在此、且当前只有一个总时间拆不开，建议 Builder 补埋点。

- [ ] **P-3｜终止耐受（三种杀法）**（门控）—— ⛔ **未执行**，无驱动。⚠️ (iii)「10 条连续保存中途 SIGKILL」是唯一真正检验原子写的一条，其余两种杀法可用现有 App 手动测（不需要合成 Pin），**理论上可以不等 fixture 单独先跑**，需 Architect / 用户确认是否值得单独执行。

- [ ] **P-4｜不扰动 tap 延迟（同会话 A/B）**（门控）—— ⛔ **未执行，两个独立原因**：① 无驱动（需脚本每 500ms 存一条 Pin，人手做不到）；② 前置条件 `reAnchorEnabled == false` 的依据 §18.3.3 硬隔离 3 **已被 §20.3.4 解除**，字面不可执行，需 Architect 重述判据（§37.6 给了建议措辞）。

**§37 静态审查新发现（不在原验收表内，Architect/Builder 待裁）**：
- **P0-1** manifest 解码失败不置 `storeUnavailable` ⇒ 后续 `create` 正常写入 ⇒ 原子覆盖 manifest（此时索引为空）⇒ 下次启动 `collectOrphanBlobs` 删光全部 blob，两步不可逆，且该路径无 `.bak`
- **P0-2** `create` 先于 `load` 时 `installRecords` 的 `removeAll` 静默丢记录，回调仍返回 `.success`（§35 同类静默失败）
- **P0-3** `@Published pins` 在 flush 前发布、失败不回退，撞 §19.3.4「⛔禁止 UI 显示保存成功但盘上没有」；根因是 §19.3.2 与 §19.3.4 指向相反行为、§19 未裁
- LRU 容量前提与 ISSUE-P4-GATE 同形：注释/规格写「≈256 KB」，实际缓存的是解码后 `[UInt8]`（65,536B × 32 = **2 MiB**，差 8×）
- P1 级五条见 §37.4：flush 失败无重试触发器、NaN 致永久写入停摆风险、delete 的 blob/manifest 顺序窗口、`uuid ?? UUID()` 静默造假 id

**§37 移交给下一步的开放项**：
- 真机构建未验证（两次构建均为 `iphonesimulator`）
- B-29 `#if DEBUG` fixture 提案（驱动 P-1/P-2/P-4），归属未定，明确不归咎 Builder 或 Architect

------------------------------------------------------------
Day 5 — Pin 创建 UI
------------------------------------------------------------

> 📌 **Day 5 的 Architect 裁决范围记录（2026-08-18，architect_output §21.8）—— 只登记归口，不预设答案，⛔ 不是 re-anchor 工作。**
> **R27（Pin × re-anchor）现有三问**：(i) 保存 Pin 瞬间 mask 正被 re-anchor 替换的竞态；(ii) 已 Pin 的实例是否继续被刷新（Pin 的语义 = 冻结这一刻 vs 持续跟踪这个东西）；**(iii) 🆕 同一次点击落在已有 mask 内时，是 promote 还是 Pin，抑或两者都做** —— Pin 的命中判据与 §3.2 的 promote 路径**是同一个**（`alpha[idx] > 0`），§19 声明 Pin 继承该不变量但未裁优先级。
> **新保留项 R32（OPEN，P2，Architect）**：`originAlpha` 的唯一写入点是 tap **decode** 路径，而 promote 在此之前 `return` ⇒ **点进一个被门冻结的 mask 不会触发 REC-2**，而那是用户最自然的动作。⇒ §18.2「代价可见且可恢复」这一定性存在**前提缺口**（定性不翻转，降为「恢复路径存在但不可发现」；REC-1 与「点在旧 mask 之外」两条路径仍真实存在）。**REC-2 的准确可达条件已更正为「一次会新建实例的 tap，即落在所有现存实例 mask alpha 之外」**（§21.8.2）。
> ✅ **归口已裁定（用户裁决，2026-08-18）：选项 A —— R32 归 Phase 4B Day 5，与 R27 同批裁。** 依据：R32 与 R27 第三问共用同一命中判据（`alpha[idx] > 0`），合并裁决边际排期成本为零；且其性质是 **Pin 交互裁决**，不是 STOP RULE 所禁的 re-anchor 补救轮。选项 B / C 未采纳（B 会让 Day 5 在不知 R32 结论的情况下裁 R27(iii)；C 等于把已知的前提缺口静默化）。三选项原文见 architect_output §21.10。
> ⛔ 该裁定**不构成**开启针对该交叉面的补救轮 —— 若将来需要，仍须由用户明确要求（STOP RULE）。
> ✅ **裁决已交付（2026-08-21，architect_output §22.2）：R27(i/ii/iii) + R32 已作为一个统一交互设计裁定。** R27(iii) 由「长按（Pin 通道）与点击（promote/新建实例通道）在同一次触摸上互斥」关闭，冲突不存在；R27(i) 由新不变量 **PIN-6**（`TapInstanceManager` 既有锁内单次快照，预览与保存复用同一份字节）关闭数据风险面，语义上不做特殊调解；R27(ii) 裁定为「继续刷新，不新建 Pin↔TapInstance 绑定」；**R32 状态不变（OPEN，P2，Architect）**，本批交付一条 Pin 侧静态 UI 缓解（B-36），并明确将「按状态显示提示」「改动 promote/`originAlpha`/re-anchor 代码」等更深修复方案**停下、交还用户**，未自行裁决。§37（Day 4 PinStore 静态验收）的补救裁决见 §22.1（P0-1/P0-2/P0-3 三项 + LRU/actor 隔离/P-1 歧义/P-4 前置条件/B-29 fixture 归口）。

## Builder
- [x] **长按 tap anchor marker → 固定 Pin**
  → ✅ 完成（2026-08-21）：`Interaction/TouchHandler.swift` 新增 `UILongPressGestureRecognizer`
  （`minimumPressDuration = 0.5`，仅 `.began` 触发一次，未与单击/双击建立 `require(toFail:)` ——
  §22.2.1 裁定二者在同一次触摸上天然互斥，无需额外优先级规则）。`Detection/CameraManager.swift`
  新增 `handleLongPress(canonicalPoint:viewPoint:)`：按 §22.2.2 决策树，长按命中已有 mask →
  发布 `pinCreationDraft`（不 promote、不 decode）；未命中任何 mask → no-op（不新建实例）。
  `UI/PinCreationSheet.swift`（新建）：底部半屏，保存成功后经回调令锚点图标切换为 📌
  （`TapAnchorMarkerView` 新增 `isPinned`，装饰 `TapInstance` 而非持久记录，随实例生命周期消失，
  见 §22.2.4）。详见 builder_progress.md Day 5 条目。

- [x] **生成 mask 缩略图**
  → ✅ 完成（2026-08-21），**尺寸按 architect §22/§19 更正为 256×256（非本条目原文的 128×128）**：
  `PinCreationSheet` 用既有 `MaskPNGCodec.encode`（256×256 无损）生成缩略图 PNG，SwiftUI 端按需
  缩放显示，不生产任何独立尺寸资产；`Pin.thumbnail` 字段本身在 §19 设计中不存在——缩略图源就是
  `masks/<id>.png` 本身（懒加载），与 tasks.md 原文的独立 `thumbnail` 字段假设不同，按 §19/§22
  裁决执行。详见 `Persistence/MaskPNGCodec.swift` 顶部注释（128×128 拒绝理由）与 builder_progress.md。

## Debugger
- [x] **验证长按手势不干扰单击分割、双击清空** —— ✅ 完成（2026-08-21，真机日志，debug_report.md §39）。长按两次全部只触发 `PinCreationSheet`，±3s 窗口内无 tap/decode/promote 伴随；单击（TAP#1/#3）与双击清空各测两次均正常。⚠️ 残留观察：`TouchHandler.swift` 的长按与单击**无 `require(toFail:)` 互斥、delegate 对所有识别器恒返回 true**，本次未撞上不代表结构性风险不存在，记为观察项不阻塞本条。

- [ ] **验证 Pin 保存后重启 App 图钉仍存在** —— ⏸️ **推迟（用户裁定，2026-08-21）**，不判定，不勾选。
  两份真机日志给出互相矛盾的证据：早先一份显示 3 次真实 SIGKILL 后 `load pins=N` 与上次 `create` 计数精确匹配、`orphans=0`（支持 PASS）；**最新一份显示同一类型的重启序列里出现两次无法用代码解释的计数异常**（`44→26`「丢 18 条」、`39→44`「凭空多 5 条」，且第二次异常精确回到丢失前的数字）。已排除的解释（逐条源码核实）：`create` 成功日志仅在 `flushNow` 真正落盘后打印（非乐观计数）／`Data.write(.atomic)` temp-file+rename／全工程仅一处 `FilePinStore()` 构造，排除双实例互相覆盖／`load()` 有 `loadStarted` 幂等保护，同进程内不会重装／`installRecords` 仅在 `load()` 内被调用／`pin.store.io` 为纯 serial queue，无 `.concurrent`／迁移路径未触发（v1→v1，`migratedFrom=1` 只是版本号打印非真实迁移）。**根因未定位，怀疑在设备/Xcode 容器还原层面而非 App 逻辑**，记为新保留项 **R34（OPEN，P0，Architect+Debugger）**，不阻塞 Day 6 开工，验证方式（容器直接读取 `manifest.json`）已给用户，等待后续数据。

- [ ] **P-1｜Roundtrip 逐字段保真** —— ⏸️ **推迟（用户裁定，2026-08-21）**。一次都未真正执行（未做过 LLDB/字段级比对），无数据支撑判定，不勾选。

- [ ] **P-3｜终止耐受（三种杀法）** —— ⏸️ **推迟（用户裁定，2026-08-21），且现在有反例，非仅未执行**。见上方 R34——已确认落盘的数据在真实 kill+relaunch 后出现过消失，这正是本条要防的事，比"未确认写入丢失"（P-3 允许的范围）更严重。⛔ **不得判 PASS**；R34 澄清前维持 OPEN。

- [ ] **P-4｜不扰动 tap 延迟（同会话 A/B）** —— ⏸️ **推迟（用户裁定，2026-08-21）**。最新真机日志（75 次 tap）实测 `[PIN]` 写操作数 = **0**，B 臂未被触发，A/B 无对比数据。已知操作缺口：现无「不重启即可在同会话内触发 PinStore 写入」的方式，`-PinFixtureBatch` 启动参数需要重新 Run 才生效，与"同会话"要求冲突，需 Architect/Builder 补一个应用内触发接口才能执行。

> **本批次记录原则**：以上四项标记为「用户决定推迟处理」而非「已验证通过」——不阻塞 Day 6 推进，但 R34（数据倒退根因未明）作为 P0 保留项显式跟踪，避免被静默略过。

------------------------------------------------------------
Day 6 — 标注编辑器 + Pin 检索
------------------------------------------------------------

## Builder
- [x] **AnnotationView（全屏编辑器）**
  - 点击图钉 → 打开 AnnotationView
  - 内容：mask 缩略图、标签编辑、备注编辑、修改时间、删除按钮
  - 保存更新写入 PinStore
  → ✅ 完成（2026-08-21）：新建 `UI/AnnotationView.swift`。256×256 缩略图经 `PinStore.loadMaskImage` + `MaskPNGCodec.encode` 显示（非 128×128，同 Day 5 校正）；tag/note 编辑器字符计数复用 `PinFieldLimits.length(of:)`（不重复实现）；显示 `createdAt`/`updatedAt`；删除按钮二次确认后走 `PinStore.delete(id:)`；保存走 `PinStore.update(id:tag:note:)`。UI 全英文。

- [x] **PinList 视图**
  - 入口：导航栏新增「Pin 列表」按钮
  - 显示所有已保存 Pin（缩略图 + tag + 时间）
  - 支持按时间排序 / 按标签筛选
  → ✅ 完成（2026-08-21）：新建 `UI/PinListView.swift`；`ContentView.swift` 导航栏新增英文 "Pin List" 图钉图标按钮。排序（Newest/Oldest）与标签筛选均为 `store.pins` 数组上的纯内存操作（§19.1.2，无新查询基础设施）；`!isLoaded` 显示 loading 态，不与「无 Pin」同形（§19.3.2）。UX 分工（§19.4 未规定，本轮自定）：行体点击 → 重访；行尾滑动露出「Edit」按钮 → AnnotationView。

- [x] **重访逻辑**
  - 点击 PinList 中某条 Pin → 加载 `frameGeometry` → 用 `canonicalPoint` 重新 decode → 恢复活跃 mask
  - 若 decode 失败：展示 thumbnail 静态图作为降级
  → ✅ 按 architect_output §19.4 实现（**不是**按上面这段草案字面实现 —— §19.4.1 已诊断草案「恢复活跃 mask」措辞与「decode 成功率」判据均不成立，本行以下是替代契约的落地，取代本条草案文字）。`PinInterfaces.swift`：`CameraManager.handleTap(fromPin:store:)` 从 B-28 占位补全为薄封装 —— G1（geometry 逐位相等，含 §19.4.5 唯一允许的等宽高比重映射）→ G2（`.tapToSegment`，不自动切模式）→ G3（`currentFrameGeometry() != nil`）全部通过后，转调既有 `handleTap(canonicalPoint:viewPoint:)`；G4（FIFO 容量）无特例，随该调用免费继承。R-A/R-B/R-C 三结局：R-A 复用普通活跃实例呈现 + 一句"Re-segmenting at this location…"过渡文案（复用既有 `tapProcessing` 脉冲，未新建等待 UI）；R-B/R-C 落到新建的 `UI/PinRevisitStaticViewerView.swift`（灰度、虚线框、"STATIC SNAPSHOT"角标，与活跃 mask 视觉可辨，PIN-2 第 3 条）+ 明写原因文案。`[PIN] revisit` 日志逐字符匹配 §19.4.6 格式，IoU 复用 `DriftDetector.alphaIoU`（stride:1，不新写第二套实现）。详见 builder_progress.md Day 6 条目。

> （以下五条按 architect_output.md §23.6 追加，正文见 §23.4；用户批准应用于 2026-08-23。既有三条**不取消勾选**——Builder 按当时的 §19.4 实现无误，缺的是契约本身。）

- [x] **B-37｜重访产物的锚点标记 + 来源装饰（P1）** —— §23.1.4：① 锚点标记与在途原地脉冲已补上（路线 **A1**：`FrameGeometry.projectCanonicalPoint` = `invertViewPoint` 的逐步代数逆，`handleTap(fromPin:)` 经其正向投影把 `viewPoint` 传给既有 tap 入口，脉冲/涟漪/标记全部免费继承）；② 字形 ↻（SF Symbol `arrow.clockwise`），⛔ 未用 📌；③ 标签仅 `From Pin “<tag>”` / `From an untagged Pin`，tag 只出现在带引号的 `From Pin` 前缀内；mask 首次上屏自动显示一次，之后沿用既有 showsTag 策略；④ 生命周期同 📌（随 `TapInstance` 消亡，长按保存后由 📌 取代，不写回 Pin）。→ ✅ 完成（2026-08-23），P-9 待 Debugger 执行。详见 builder_progress.md 本批条目。
- [ ] **B-38｜FIFO 淘汰回执横幅（P2，对一切淘汰生效，非重访专属）** —— §23.1.7：复用既有瞬时横幅表面；被淘汰实例带 📌 时追加 `Its saved Pin is still in Pin List.`；⛔ 不改任何淘汰决定、不新增池状态。（本批未做，P2 缓办。）
- [ ] **B-39｜"Pin List" 导航按钮条数徽标（P2 → 已提升）** —— §23.1.8：`store.pins.count`，为 0 时不显示；⛔ 禁 object 词族；⛔ 拒绝把已存 Pin 叠在相机画面上。
  → 🔨 **2026-08-23 用户批准现在做，已派 Builder（进行中）**。提升理由不是新需求：用户真机撞上了 §23.1.8 批准 B-39 时所论证的**那一个**缺口——「重启后无 📌」（设计行为）在缺少条数凭据时无法与「数据丢了」区分。见上方 re-anchor 补救轮区块的症状 C。
- [x] **B-40｜R-D 路由修复（P1，阻塞 P-6/P-7；三项缺一不可）** —— §23.1.9：① `CameraManager` 第三个纯观测闭包 `onTapPromoted(gen:promotedInstanceID:)`，在既有 promote 主线程块内触发；② `lastTapWasRevisit = false` 复位覆盖全部三条 gen>0 路径（promote / `failTap` / 正常发布块），R-D 结局上不再出现任何 "Re-segment(ing)" 文案，改为 `That spot is already covered by a selection on screen.`（可附 `From Pin “<tag>”`）；③ `PinRevisitTracker` 每个注册 gen 的穷尽释放点 placed/failed/promoted 三选一必到，消除 64 KB/次泄漏；PIN-8 成立（三钩子唯一安装者仍是 `PinRevisitTracker`）。→ ✅ 完成（2026-08-23）。
- [x] **B-41｜统一 `[PIN] revisit` 日志行（P1，阻塞 P-6/P-7）** —— §23.2.4 单一格式：新增 `seq`（入口处、任何门之前自增）/ `outcome=(mask|promote|refused|failed)` / `reason` / `pt` / `pin` 字段；四种结局同一格式，`rejected reason=…` 异构行取消；`iou` 计算口径（256×256、stride:1、`DriftDetector.alphaIoU`）一字未改；⛔ 未新增定时器。→ ✅ 完成（2026-08-23）。

## Debugger
> （本区块按 architect_output.md §23.6 整体替换原三条复选框；用户批准应用于 2026-08-23。）
- [x] **P-6｜重访准入门与结局路由（门控）** —— architect_output.md §23.2.5：C1 模式门 / C2 朝向 / C3 摄像头 / C4 R-A 正常 / C5 R-D promote，各 ≥1 次，5/5 逐字段比对 outcome/reason/geo/pt 与屏上表面
  → ✅ 完成（2026-08-23，真机日志 + 用户逐字抄录屏上文案比对代码字符串，B-41 新日志格式）。5/5：**C1** `outcome=refused reason=mode`，t=12648 拒绝、下一次 `MODE SWITCH` 在 t=25205（用户手动切），未被自动切走；**C2** `outcome=refused reason=orientation`，静态查看器文案「This pin was recorded in portrait orientation, back camera — that doesn't match the current landscape orientation, back camera.」同时写出记录值与当前值；**C3** `outcome=refused reason=camera`，同一构造的具体原因文案；**C4** 累计 14 次 `outcome=mask geo=ok`，覆盖 5 条不同 Pin，`pt`/`pin` 14/14 逐位相等，fast/slow 路径均出现；**C5** `outcome=promote`，横幅逐字为「That spot is already covered by a selection on screen. From Pin "哈哈😊`"」，零 "Re-segment" 字样，tag 含 emoji/反引号仍落在 `From Pin` 引号前缀内。
- [x] **P-7｜重访问责 + IoU 报告（必报，含一条门控）** —— §23.2.6：门控 = 「Pin List 发起次数 == [PIN] revisit 行数且 seq 覆盖 1…R 无缺号」；必报 = ≥8 次 R-A、≥3 条不同 Pin、fast/slow 各 ≥1；⛔ IoU 不设任何达标线，报告须附 §20.4.1 分簇解读框与高-IoU 反例点名规则
  → **必报部分独立完成**（14 次 `outcome=mask`、5 条不同 Pin、fast/slow 均有；IoU n=14 min 0.06 / median 0.47 / max 0.99，落在 §20.4.1 三簇均有分布，不据此判定；已点名高 IoU 反例：`3620286D` 创建时 `iou_pred(selected)=0.545`/`stab=0.16`，重访 `iou=0.82`——高 IoU 只说明解码器复现了自己的错误）。
  → ✅ **门控判 PASS（2026-08-23，用户基于现有证据裁定）**——⚠️ **非形式化独立核对**：未取得过「单一不中断会话内、人工计数 R 与该会话 seq 范围严格相等」的直接证明（历次会话均被设备掉线打断，21 次重访那批数据 seq 出现两次 1、两次 2，证实跨了至少两个会话，无法拿来做单会话核对）。支撑判断的是**循证证据**：B-41 上线后的全部会话里，未观测到任何一次「已报告发起重访、但无对应 `[PIN] revisit` 行」的反例；唯一一次完整会话（以「killed」结束）内的重访行（`A7D93728 seq=1`）本身无缺号。用户认定此证据已足够，不要求补跑形式化会话核对。
- [x] **P-8｜AnnotationView 编辑 / 删除回归（门控）** —— §23.2.7：冷启动逐字段；空串回读为 nil；updatedAt 严格增而 createdAt 与全部冻结字段逐位不变（PIN-1）；blob 逐字节不变；删除后 orphans=0
  → ✅ 完成（2026-08-23，设备容器快照前后差分核实——非合成数据，非日志转述）。编辑 1 条（`f6918824`）：tag/note 变更、`updatedAt` 严格增（1787543215→1787544624）、`createdAt` 与全部 PIN-1 冻结字段（id/pointX/pointY/geometry/maskFile/maskWidth/maskHeight/maskNonZero）逐位不变、mask blob SHA-256 逐字节相同；删除 2 条：记录与 blob 同步消失，冷启动 `orphans=0`（61→59，与人工报告的删除数一致）；58 条未编辑记录冻结字段零漂移；全库 0 条 tag 为空串 `""`。⚠️ **一项非直接实测**：本轮编辑是把已有 tag 改成另一个非空值，未覆盖「非空改成空」这条子情形；空串→nil 的归一化由代码路径（`normalizeEmptyField` 在 create/update 两条写路径统一调用，非重复实现）与全库不变量间接支撑，不是一次直接的设备端非空→空回读实测。
- [x] **P-9｜canonical↔view 往返 ≤1 px（门控，仅当 B-37 走 A1 路线）** —— §23.2.8：4 组工况 × 9 点网格全部 ≤1.0 px
  → ✅ 完成（2026-08-23，真机 4 组工况全部实测，非推演）。四组 `[P9] result` 均 `n=9 maxErr=0.00px clamped=0/9 verdict=PASS(<=1.0px)`：① `rot=90 mirrored=false canonical=1080x1920`（后置竖屏，同一会话内重复 3 次结果一致）；② `rot=0 mirrored=false canonical=1920x1080`（后置横屏）；③ `rot=90 mirrored=true canonical=1080x1920`（前置竖屏，重复 2 次一致）；④ `rot=180 mirrored=true canonical=1920x1080`（前置横屏／翻转 180°）。9 点网格含四角、四边中点、正中心；`maxErr` 恒为 0.00px 说明 `projectCanonicalPoint` 是 `invertViewPoint` 的精确代数逆而非数值近似——与 B-37 声明的 A1 路线一致。⚠️ **一项范围说明**：本门控只验证几何往返本身，**不覆盖** §19.4.5 的 promptSpace 重映射分支（该分支在当前构建下结构性不可达，见 R35，永远不得声称已测）。
  → ⚠️ 修复前置：该按钮初版因缺 `.contentShape(Rectangle())`，命中区域塌缩到文字行框，触摸穿透到下层相机 `TouchHandler`，导致「看得见点不动」的静默失败（用户报告确认，非推测）；Builder 已扩至整行命中区并加 1 秒「✓ Ran」屏上反馈，上述四组读数取自修复后的构建。
  → **原三条的处置（§23 裁决，2026-08-23）**：
    · 「10 个 Pin 写入后检索延迟 < 100 ms」→ **作废**。与 §19.6 P-2 重复，且属同一缺陷类：读取走内存索引（§19.3.2），由构造满足、失败不了。其正确形态是 P-2 的线性度门 `t(N=50) < 4 × t(N=10)`，**P-2 至今未执行**，应在那里跟踪而不是在这里重开一条。
    · 「重访 decode 成功率 ≥ 80%（在相近场景下）」→ **作废，且 ⛔ 不得以任何 IoU 阈值形态复活**（§19.4.7 的「中位数 IoU ≥ 0.7」同样作废）。理由：与 §19.4.3「IoU 不参与任何判定」自相矛盾（§23.2.1）；真机反例证明高 IoU 可以只是「解码器复现了自己的错误」（§23.2.2）；括号里的「相近场景下」本身违反 A-19。由 P-6 / P-7 取代。
    · 「AnnotationView 编辑后 Pin 数据正确更新」→ **意图保留、形态被 P-8 取代**（原文无可执行口径）。
  → **Day 6 出场条件**：P-6 ∧ P-8 ∧（若 A1）P-9 通过；P-7 的完备性门通过且必报部分已报告。
    → ✅ **已满足（2026-08-23）**：P-6 ✅、P-8 ✅、P-9 ✅ 均为真机实测；P-7 必报部分独立完成、完备性门由用户基于循证证据裁定 PASS（见该条 ⚠️ 非形式化核对说明）。⛔ 遗留未闭合项与本出场条件无关但**不得视为已解决**：R34（PinStore 计数倒退，P0，两次干净会话未复现但未关闭）、R32（点入已 pin mask 静默 promote，P2）、R35（坐标重映射分支结构性不可达，P3，仅按代码审阅正确，永不得声称已测）。
  → ⚠️ 2026-08-23 两次真机会话的读数**不能**用来满足以上任何一条（日志格式将被 B-41 改变）；那两次交付的是 **M-20.1 意义上的激活证据**，不是验收证据。

> **新保留项 R35（OPEN，P3，Owner Architect）**——§23 裁决登记，用户批准应用 2026-08-23：**§19.4.5 的坐标重映射与 G1 的 `promptSpace` 维度在当前构建里结构性不可达**（全工程唯一 `session.sessionPreset = .high` 且从不改动；`origW/origH` 唯一可能的变化途径是切前后摄，而那会同时翻转 `mirrored` ⇒ G1 先一步判 refused；`promptSpace` 为编译期常量）⇒ 两段代码永远无法被任何真机测试执行到，其正确性只有推导、没有实证。处置：代码保留（推导正确、成本为零），但 ⛔ 不得写进任何验收判据，⛔ 不得对外声称「支持切换分辨率后重访」。重开触发：任何引入第二个 `sessionPreset`、可变 `promptSpace`、或多分辨率采集的改动。正文见 architect_output.md §23.5。

------------------------------------------------------------
🔓 re-anchor 补救轮（用户于 2026-08-23 明确要求开启，STOP RULE 门已解除）
------------------------------------------------------------

> §23.7 记录的「未重开 Phase 4A Day 2–3（STOP RULE 继续有效）」由**用户明确要求**解除——用户真机报告 re-anchor 使用体验差，要求调整。裁决正文见 architect_output.md **§24（会话内连续跟踪，可执行规格，已交付）** + **§25（跨会话物体重识别，可行性侦察，已交付）**，2026-08-24。

**用户报告的三个症状 + Debugger 根因定位（源码复核等级，带 file:line）**

- **症状 A｜手机移动时 mask 跟着屏幕走、不跟物体** —— **架构性能力缺口，非调参可解**。`Interaction/TapInstanceManager.swift:36` 的 `let canonicalPoint: CGPoint` 不可变（§16.7 冻结），re-anchor 的实际语义是「在固定传感器坐标点上重新解码」，**无运动估计、无特征点跟踪、无 mask 形变**。手机移动 ⇒ 物体离开该固定点 ⇒ 该点解码出「现在那儿的东西」⇒ 与冻结 `originAlpha` 的 IoU 极低 ⇒ 一致性门拒绝并保留旧 mask。今日真机同一会话 5 次同类拒绝：`iou` 0.06 / 0.17 / 0.19 / 0.22 / 0.25，均 `< 0.50`。⚠️ 这些拒绝**本身是设计行为**（§16.6.1 / §17.3.3），⛔ 不得记为缺陷——缺陷在「锚点不动」这个前提，不在门。
- **症状 B｜要等 6 秒以上才复位** —— 四个串联等待，实测吻合：`Detection/CameraManager.swift:2889` `cacheFresh = age <= 5000`（**5000ms**）+ `:2890` `quiet = msSinceTap > 1500`（最多 **1500ms**）+ encode 实测 587/607/713/736/820ms（**~700ms**）+ `:2397` RE-1 门 `batch.filter { $0.lastReAnchorEmbeddingGen != currentEmbeddingGen }`（每实例每 embedding 代际只能 re-anchor 一次，写入点 `TapInstanceManager.swift:379`，烧掉后锁死到下个代际）⇒ 合计 **5.7–7.2s**。
- 🆕 **ISSUE-RA-1｜`heavy_drift` 重编码路径从未接线（死代码，由构造不可达）** —— `CameraManager.swift:2943-2944` 的注释自述存在 `heavy_drift — cache is young but drift score forced a refresh (only reachable if drift path adds to this call)`，但 `:2891` 的 `guard !busy, !parked, quiet, !cacheFresh else { return }` 是**无条件**的，`age < 5000` 直接 return ⇒ `:2951` 的 `heavy_drift` 分支**永不可达**；`refreshTapEmbeddingIfNeeded` 三个调用点（`:2285` / `:3345` / `:3422`）全为帧驱动或 tap 后触发，**无一来自漂移路径**。即「检测到剧烈漂移 ⇒ 立即重编码」这条路日志分支先建好、实现未跟上。**这是症状 B 里「改一处收益最大」的位置**，且该结论独立于 §24 如何裁决。
- **症状 C｜重启后无 📌，但 Pin List 点进去 mask 正常** —— ✅ **判定为设计行为，非缺陷**。📌 装饰 `TapInstance` 内存态、不写回 Pin（§22.2.4 / B-37 ④）；重启后实例池为空故无 📌，记录在盘故 Pin List 可调出。⛔ **不得记入 R34**：R34 是数据真的减少（计数倒退），此处数据完好无损（mask 可正常调出），混记会污染该 P0 的证据链。⚠️ 用户的困惑本身是真问题——§23.1.8 当初批准的缓解措施正是 **B-39**，而 B-39 被 P2 缓办至今未做，用户今天正是撞上这个缺口。**用户已批准现在做 B-39**（见下）。

**用户裁定（2026-08-23，在被告知三种目标行为各自代价后）**：症状 A 取 **「真正跟踪物体」**——即让锚点随物体移动，而非锚在固定点。⇒ 已派 Architect 裁决，须逐条论证对 §16.7（`canonicalPoint` 冻结）、§18.1.2 纯函数论证与 RE-1 门、**PIN-1**（`pointX`/`pointY` 属冻结字段：跟踪后长按存 Pin 存原始点还是当前点？重访 G1 读哪个？）、一致性门语义、§20.4.1 IoU 三簇标定的影响。⛔ **R21 仍有效**：`reAnchorAcceptIoU = 0.5` / `contentThresholdLuma = 8.0` / `minReAnchorIntervalMs = 300.0` 三个数值本轮**只可裁语义、不可改数字**（无新基准下的接受率数据）。

**🆕 用户裁定第二轮（2026-08-24）：范围扩至跨会话持久化。** 追问「永远在 object 上留下图钉」的「永远」范围后，用户明确选择：**关闭 App 重开、摄像头再对准同一个（可能已被移动过的）物体，图钉应自动重新出现**——而非只在当前会话/画面内有效。这与"会话内连续跟踪"是**两层不同能力**、代价差一个数量级：前者是几何/CV 工程问题；后者是**物体重识别（re-ID）**，性质是开放机器学习问题。（历经四次派工尝试，前三次均因 Claude API 端 529 服务过载未能执行，第四次成功交付。）

---

### §24 裁决摘要（会话内连续跟踪，architect_output §24，可执行规格）

**核心设计**：不让 `canonicalPoint` 可变，新增独立的 `trackedPoint` 字段（`canonicalPoint` 永久冻结，`PinFactory` 只读它）——这个选择让 **PIN-1 完全免于修订**（唯一一条不需要修订的承重论证）。

**五条承重论证逐一核对**：§16.7 冻结 → 需修订（新增字段而非改原字段）；§18.1.2 纯函数论证 + RE-1 门 → 不成立，需扩展为并集判据；**PIN-1 → 仍然成立，不需修订**；一致性门（`reAnchorAcceptIoU=0.5`）→ **不成立**，且是本节最重要发现——绝对坐标系 IoU 在锚点可移动后从根本上测错了对象（物体从左走到右，mask 不重叠是跟踪成功的正确结果，旧门会把每次跟踪成功判定为失败），需要全新的质心对齐量 `trackConsistencyAcceptIoU`，与 `reAnchorAcceptIoU` 物理不可比，**R21 对 0.5 的保护继续原样有效，未被触碰**；§20.4.1 IoU 三簇框架 → 需重新标定，当前无数据（P-15 采集，不设通过线）。⇒ 新长期约定 **A-20**（坐标系稳定性前提）。

**跟踪机制选型**：局部块匹配（复用 `AnchorSignature` 8×8 采样基础设施），零 ANE/GPU 新负载，估计 <1ms/次（⚠️ 推导，需 P-11 真机确认）。Vision 框架方案因 `ISSUE-P4-DECODE` 未结案时余量不足而不采纳，记录为未来升级路径。跟丢判据与恢复策略已显式定义（§24.2.3）：跟丢即冻结 `trackedPoint`、复用既有 `reAnchorKeepStaleMask`；恢复搜索以 `canonicalPoint`（非跟丢点）为中心、宽半径、复用 embedding 世代节奏。

**症状 B（ISSUE-RA-1）修法**：不让漂移无条件触发重编码（三条理由，`quiet` 窗口保护是决定性理由）；新增独立限流旁路（`heavyDriftForceRefreshLuma=32.0` 等四个新常量，全部独立于 R21 保护的三个既有常量）。**关键耦合**：跟踪与陈旧 embedding 若不同批交付，会产生"安静地错"——位置对但内容过时、且不易被现有门拦住，比不跟踪更隐蔽，⇒ 两者**必须同批交付**。

**⛔ ISSUE-P4-DECODE 未结案**：不阻塞本节设计交付，但**阻塞 `objectTrackingEnabled` 翻真**（三条前置合取见下方主开关行）。**R4 因 B-45 命中重开触发条件 (i) 而重新纳入观测**（见上方 D-4' 区块的重开标注），判据 = P-17。

#### Builder（B-42…B-48，编号接续 §23.4 的 B-41）

- [x] **B-42** `TapInstance` 新增字段 `trackedPoint`/`lastReAnchorTrackedPoint`/`trackState`（初值=canonicalPoint，能力关闭时恒 `.locked`）—— `TapInstanceManager.swift`，P0 基础字段，本身不改变任何行为，无依赖
  → ✅ 完成（2026-08-24）。`TrackState`（`.locked`/`.tracking`/`.lost`）新增于 `TapInstanceManager.swift`，`nonisolated enum`（项目开了 `-default-isolation=MainActor`，枚举 Equatable 一致性需显式标注，否则报 warning）。三字段接入唯一构造点 `addInstance`，用既有 `lock`，无新增锁/队列。`canonicalPoint` 零改动。本批次 `trackState` 恒 `.locked`，行为逐位不变（源码复核 + iPhone 11 模拟器回归验证：装包、切 Tap to Segment、点击，无 crash、无异常日志）。
- [x] **B-43** 新文件 `Interaction/AnchorTracker.swift`：局部块匹配搜索 + 跟丢判据 + 恢复搜索（§24.2.2/.3）—— P1，依赖 B-42
  → ✅ 完成（2026-08-24）。`trackSearch`/`recoverySearch` 两个独立类型的入口（故意不同类型，防止 B-44 接错线），共用私有网格搜索原语，完全复用 `DriftDetector.signature`/`divergence`，零新增采样/散度算法。`isLost`/`hasRecovered` 判据、恢复搜索以 `canonicalPoint`（非跟丢位置）为中心均逐字对应 §24.2.3。五个新常量独立于 R21 保护的三个既有常量。`trackSearchStepPx=8.0` 与规格文字推导（96/8=12.0）不一致——按规格表格数值字面实现，矛盾记录在代码注释里，未擅自"纠正"，留给 Architect 核对。本批次零调用方，用 `#if DEBUG` 自检（合成 `CVPixelBuffer` + 独立于 Xcode 工程的 Mac 宿主编译验证）代替真机验证，过程中发现并修正了一处自检设计问题（非对齐偏移量的散度量化误差），非代码 bug。构建：本人独立复核代码 + 独立重跑一次 clean build 确认 **BUILD SUCCEEDED，真实 Swift warning 0**。
  → ⚠️ **移交发现（非本条缺陷，供 B-44 处理）**：恢复搜索半径（192.0）沿用了与正常跟踪搜索相同的步长（8.0），候选数达 49×49=2401（是正常搜索 169 个候选的 ~14 倍），单次成本估计 ~120ms——在 videoQueue 上会造成一次可感知卡顿（一帧预算仅 33ms）。architect_output §24.2.3 只裁了恢复搜索的半径，未裁步长是否应独立放大，这是规格本身的缺口。虽然触发频率低（仅 `.lost` 态、约每 5 秒一次），120ms 卡顿仍值得处理，不应放任。B-44 落地时需一并解决（建议：恢复搜索用独立的、更粗的步长常量）。
- [x] **B-44** `checkAndFireReAnchor` 接入 B-43：候选超阈才搜索、成功匹配移动 `trackedPoint` 并派发 decode —— `CameraManager.swift`，P1，依赖 B-42/B-43
  → ✅ 完成（2026-08-24）。搜索在 videoQueue 上、`decoderQueue.async` 派发之前完成（遵守 B-43 的 buffer 归属约束，不跨队列访问）；找到新位置后在**新点**重新采样作为新 baseline（不用漂移前的旧签名，避免自适应模板反向漂移）；`.locked` 实例在首次漂移超阈被 RE-2 选中时惰性激活为 `.tracking`(本人续接同一 agent 补的缺口——初版遗漏了这条转换，无路径能真正启动跟踪；本人派工指令本身还有个内部矛盾被 agent 正确识别并按语义而非字面纠正，过程记录在案)。`.lost` 态搜索/恢复逻辑一行未受影响。本人独立复核代码 + 独立重跑 clean build 确认 **BUILD SUCCEEDED，真实 Swift warning 0**。
- [x] **B-45** RE-1 资格判定扩展为并集（新增 `trackReDecodeMinDeltaPx=12.0`）—— **须与 B-44 同批**，否则跟踪位移会被旧 RE-1 吞掉、表现为"从不刷新"；**命中 R4 重开触发条件 (i)**
  → ✅ 完成（2026-08-24）。architect_output §24.1.3 公式拆成两处应用（发现并解决了公式字面用法的循环依赖：若用来门控"要不要采样"，`trackedPoint` 永远等不到移动的机会）：采样资格门给 `.tracking`/`.lost` 无条件放开（不再靠世代号，因为漂移采样是纯像素运算，跟 embedding 无关，`.locked` 的"同 embedding+同点=同结果"前提对可移动的点不成立）；派发决策门（是否真的花一次 decode）套用 architect 原始公式，应用在搜索之后、决定要不要派发之前。解法与推导已原样写入代码注释，供 Architect 核对应用位置的重新分配是否认可。`trackReDecodeMinDeltaPx=12.0` 归属地由 `AnchorTracker.swift` 补齐（此前误记在 B-43 交付里）。同批顺手修了 B-43 遗留的恢复搜索候选数问题（新增独立步长常量 `trackRecoverySearchStepPx=16.0`，候选数从 2401 降到约 625）。
- [x] **B-46** 新的平移不变一致性门：质心对齐后再算 IoU，新常量 `trackConsistencyAcceptIoU`（初始 0.5，与 `reAnchorAcceptIoU` 互不影响）；⛔ 不改 `DriftDetector.alphaIoU` 现有实现 —— **P0，阻塞 B-44/B-43 启用**（没有它，跟踪成功会被现有绝对坐标系门无差别否决，产品行为比不跟踪更差）
  → ✅ 完成（2026-08-24）。新函数 `DriftDetector.centroidAlignedIoU`：两张 mask 各自质心平移对齐到网格中心（整数像素位移，越界补零）后再逐元素算 IoU；`alphaIoU` 一行未动，继续原样服务能力 A/B；比较不了时返回 1.0（不否决），同 `alphaIoU` 的安全默认纪律。`trackConsistencyAcceptIoU=0.5` 独立存储，与 `reAnchorAcceptIoU`/`contentThresholdLuma`/`minReAnchorIntervalMs` 三个既有常量数值零改动。`CameraManager.reAnchorDecode` 按 `instance.trackState` 切换门与阈值（`.tracking` → 新门；`.locked`/`.lost` → 原门），本批次 `trackState` 恒 `.locked`，该分支为预期中的死代码。构建：Debug clean build **BUILD SUCCEEDED，真实 Swift warning 0**（含中途一次 `TrackState` MainActor 隔离 warning，已加 `nonisolated` 修复，二次 clean build 清零——本人已独立重跑一次 clean build 复核确认）。
- [x] **B-47** 症状 B 旁路：`objectTrackingEnabled` 主开关（B-44 已定义）+ 三个新常量；改写 `heavy_drift` 死代码分支使其如实反映可达性 —— **须与能力 C 同批交付**（§24.3.4 耦合关系）
  → ✅ 完成（2026-08-24）。`heavyDriftForceRefreshLuma=32.0`/`minHeavyDriftAgeFloorMs=1500.0`/`minHeavyDriftRefreshIntervalMs=5000.0` 独立存储，未碰任何既有常量。`quiet` 作为不可省略的必要条件，理由（活跃交互窗口不该被重编码打扰）原样写进代码注释。`lastObservedMaxDriftLuma` 复用 `checkAndFireReAnchor` 已算出的 `measured` 数组取 max，零新增采样；独立冷却时钟 `lastHeavyDriftRefreshMs` 不与 re-anchor 自己的节流时钟混用。两处日志均已改为如实反映可达性——第二处（`heavy_drift` 死代码本体）Agent 主动验证了可达性推理（能走到此处且 age<5000 必然是 `heavyDriftBypass` 生效，浮点恰好相等的边界概率≈0）而非直接照做我的建议。本人独立复核代码 + 独立重跑 clean build 确认 **BUILD SUCCEEDED，真实 Swift warning 0**。边界表达式用独立脚本测了 11 组组合，全部符合预期。
- [x] **B-48** `PinFactory.makeRecord` 防御性文档 + 断言：显式声明只读 `canonicalPoint` —— P3，文档/断言级，无行为改动
  → ✅ 完成（2026-08-24，本人直接实现，未经 agent）。`PinInterfaces.swift`：doc 注释新增段落，把 `trackedPoint`/`lastReAnchorTrackedPoint`/`trackState` 明确划入与既有 re-anchor 字段同一类的禁读范围；新增 `assert`，独立重读 `instance.canonicalPoint` 与已构造的 `record.pointX/pointY` 比对——不是简单信任赋值那一刻的值，万一将来有人把赋值那行改成读 `trackedPoint` 却忘了同步改这条断言，DEBUG 下会立刻炸出来。本人独立跑 clean build 确认 **BUILD SUCCEEDED，真实 Swift warning 0**。

**主开关**：`DriftDetector.objectTrackingEnabled: Bool = false`。**翻真前置条件（三者合取，缺一不得翻真，Builder 不得自行翻转）**：① B-42…B-47 全部落地且编译通过；② §24.5 全部判据完成一轮真机读数；③ `ISSUE-P4-DECODE` 结案，或由用户显式豁免该前置条件。

#### Debugger（P-10…P-17，编号接续 §23 的 P-9）

- [ ] **P-10** 激活证据先于质量证据 —— `[TRACK]` 日志行数 >0 且能区分"评估过但未越阈"与"从未评估"
  → ⚠️ **部分完成，不勾选**（2026-08-24，真机会话一）。`[TRACK]` 行确认非零且真实产生（`locked → tracking`/`recovery search`/`tracking search`/`lost → recovered` 均出现）。但**"评估过但未越阈"与"从未评估"两种情形当前日志无法区分**——只有实例真正入选候选（越阈）才打 `[TRACK]` 行，未越阈的实例完全静默，跟"根本没被 RE-1 放行"在日志上是同一种沉默。这是主协调者派 B-44 时漏提的一条要求，不是 Builder 未落实。
- [x] **P-11** 搜索成本 —— max <5ms（推导 <1ms，留 5× 余量）
  → ✅ **完成，真机实测通过**（2026-08-24，真机会话一）。5 个正常跟踪搜索样本 0.6ms，~15 个恢复搜索样本 2.0–3.4ms（恢复搜索候选数是正常搜索 ~3.7 倍，成本线性缩放，与实测基线吻合，比架构裁决保守估算的 ~31ms 快得多）。max 3.4ms < 5ms。
- [ ] **P-12** 跟丢与恢复 —— 遮挡 ≥3s 后 `trackState=.lost` 且 mask 冻结不跳变；移开后 ≤2 个 embedding 世代内恢复
  → ⛔ **不通过，机制本身有确认缺陷，不是采样不足**（2026-08-24，真机会话二）。状态机本身按契约工作（跟丢正确冻结、`reAnchorKeepStaleMask` 正确触发、恢复搜索正确围绕 `canonicalPoint` 而非跟丢点）。但**局部块匹配搜索无法区分"物体移动"与"相机移动"**——真机会话二里用户平移手机（物体本身静止）时，画面上显示的 mask **真的跳到了错误位置**（用户直接确认："mask 真的跳到了错误位置（跟着相机平移跑了）"）。日志证据：同一会话内两次跟踪搜索把 `trackedPoint` 顺着相机平移方向移动后，解码内容与原始物体的 IoU 只有 0.24 / 0.14（`origin: 313px`），说明搜索找到的"最像"位置内容其实完全不是原物体；这两次被一致性门正确拦下，但用户报告的那次错误显示说明**门不是每次都能拦住**——两个候选 mask 之间偶然算出 IoU ≥ 0.5 但语义上并非同一物体，是可能发生的（尤其大而模糊的候选）。⇒ 见下方新登记 **R36**。
- [ ] **P-13** 能力 2：短暂离屏 —— 离屏内/离屏外两种结果皆可 PASS，只要行为与 §24.2.3 定义一致（不得静默假装恢复）
  → ⛔ **不通过，同 R36**——底层恢复机制与 P-12 共享同一个缺陷，日志签名与 P-12 无法区分，且已有确认失败案例，无法判 PASS。
- [ ] **P-14** PIN-1 不回归 —— 持久化的 `pointX`/`pointY` 与原始 `canonicalPoint` 逐位相等，与当时 `trackedPoint` 不相等
  → ⚠️ **未严格测到，不勾选**（2026-08-24，真机会话一）。直接读设备容器核实：Pin `57ffa721` 的 `pointX/pointY=(515.357, 879.643)` 与原始 tap 的 canonical 坐标逐位吻合，PIN-1 字面成立。但**这次保存发生在跟踪首次激活之前**（`trackedPoint` 当时天然等于 `canonicalPoint`，尚未分离），没有测到"点已经跑偏、存 Pin 时还老实存原始点"这个关键场景。真机会话二未观察到任何 `[PIN] create` 事件。需要专门补一次：先让跟踪明显跑偏，再存 Pin。
- [ ] **P-15** 平移不变一致性门读数采集 —— ≥15 条，**报告项，不设通过线**（同 D-6/D-6' 立项纪律）
  → ⚠️ **只采到 5 条**（2026-08-24，真机会话一）：`0.46`(拒) / `0.20`(拒) / `0.56`(收) / `0.57`(收) / `0.57`(收)，横跨阈值两侧，门确实在判别，但量不够 15 条报告项要求。真机会话二又产生至少 2 条新读数（`0.24`拒/`0.14`拒），量仍不够，且会话二暴露的问题（见 R36）意味着这些读数本身的可信度也要打折扣——门有效判别的前提是"不同物体的 mask 面积/形状差异够大"，R36 揭示的正是这个前提有时不成立。
- [ ] **P-16** 症状 B 旁路不破坏 `quiet` 保护 —— 高频 tap 期间 0 次触发；安静会话中 ≥1 次触发且冷却窗口内不重触发
  → ⚠️ **必要条件确认成立，未完整验证**。真机两次会话均观察到旁路真实触发（会话一 `age=3729ms maxDrift=40.2lum`；会话二 `age=1952ms maxDrift=63.9lum`），且触发时刻 `quiet` 均为真（旁路的硬性前置条件成立，符合设计）。但没有专门的"忙碌窗口内故意制造剧烈漂移、确认 0 次触发"对照臂，不能算完整两臂验证。
- [ ] **P-17** R4 家族回归（qwait）—— max 仍 <195ms（沿用 D-7 判据线，不新起一条线）
  → ⚠️ **初步良好，样本单薄**。真机会话一：5 个 re-anchor 触发的 decode 样本，`qwait` 全部 0.1ms，远低于 195ms 线。但只有单实例场景，不是架构裁决设想的"三实例同屏"压力条件，不能算完整回归验证。

**出场条件**：**未满足**。P-11 通过；P-12/P-13 因 R36 判定不通过（非采样不足，是确认缺陷）；P-10/P-14/P-15/P-16/P-17 证据不完整。`objectTrackingEnabled` **已由本人于真机会话二后手动改回 `false`**（源码 `DriftDetector.swift`，见该常量最新注释），不得在 R36 经 Architect 裁决前重新翻真做非受控测试。

---

**🆕 R36（OPEN，架构级发现，2026-08-24，真机会话二）—— 局部块匹配跟踪（B-43 候选 B）无法区分相机平移与物体运动，真机确认至少一次错误匹配导致显示 mask 跳到错误位置**

- **证据等级**：机制缺陷本身为**真机日志确认**（两次跟踪搜索后解码内容与原物体 IoU 仅 0.24/0.14，均值明确不匹配）；"显示 mask 跳到错误位置"这一具体后果为**用户直接观察确认**（screen recording + 明确文字确认），尚未定位到日志里具体是哪一次 `[REANCHOR]` 行放行了这次错误匹配（截至记录时用户未提供该次会话剩余日志）。
- **根因（架构级，非实现 bug）**：§24.2.2 选定的候选 B（复用 `AnchorSignature` 局部块匹配）只测"这个小窗口内容像不像基线"，完全没有"这是不是同一个物理物体"的语义信息。相机平移时全画面内容一起平移，搜索窗口会认为"附近最像的内容"就是目标——而那往往只是背景随镜头平移过来的东西。这正是 §24.2.1 候选评估表里"重复纹理表面上有歧义"那条局限的一个更常见、更严重的表现形式，比原始评估预计的更容易触发（用户平移手机重新取景是最常见的操作，不是边缘场景）。
- **一致性门不是完整的安全网**：§24.1.5 的质心对齐 IoU 门只测形状/位置相似度，两个语义上完全不同的 mask 仍可能偶然算出 IoU ≥ 0.5（尤其面积较大、形状不特殊的候选），门会放行。今天真机上遇到的很可能就是这种情况。
- **处置**：不在此登记自行裁决修法。移交 Architect：是否接受此为已知局限并限定使用场景（例如只在检测到相机基本静止时才允许 trackedPoint 移动，可能需要 CoreMotion 或帧间全局位移估计作为门槛）、是否需要重新评估候选 A（Vision 光流，此前因算力预算被否，但更擅长区分相机自运动与物体运动）、或其他方案。**`objectTrackingEnabled` 在 R36 裁决前维持 `false`**，不得用于非受控测试或正常使用。
- 🔧 **修复已实施（2026-08-24，用户明确指示直接修复，不再等待另一轮 Architect 裁决流程）**：采用本条目自己列出的第一个选项——新增 `Interaction/CameraMotionGate.swift`（`CMMotionManager` 陀螺仪角速度抑制门，30Hz，零 ANE/GPU 成本），在 `CameraManager.checkAndFireReAnchor` 里把跟踪/恢复搜索的触发条件从 `objectTrackingEnabled` 改为 `objectTrackingEnabled && !CameraMotionGate.isPanning`——检测到相机甩动（角速度 ≥ `panningThresholdRadPerSec=0.20 rad/s`，未经真机调优的首版值）时整轮搜索直接跳过，`trackedPoint` 不动，退化为跟踪关闭时的定点刷新行为，从根本上避免"背景滑过静止搜索窗口被误判为匹配"这一 R36 核心机制。这与 architect_output §17.2.1 当初评估 CoreMotion 时预先批准、只是未立项的"抑制门"用法完全一致，不是新发明的方案。`objectTrackingEnabled` 已改回 `true`。Debug 模拟器 + 真机架构（未签名）clean build 均 BUILD SUCCEEDED，无新增警告。**状态改为 OPEN → 修复已实施，未经真机复核**——不是关闭：静止搜索窗口本身仍无法区分物体身份，这次修复只是不让它在相机确知在动的时候有机会犯错，物体在画面里被其他相似物体替换、或摄像头缓慢平移到刚好低于角速度阈值等边界场景理论上仍可能有残余风险，需要实际使用中观察后再决定是否关闭本条。

---

### §25 裁决摘要（跨会话物体重识别，architect_output §25，可行性侦察）

**结论：值得做一次极低成本的侦察性 MVP，不值得现在做完整产品化投入。**

**两条技术路线，互不覆盖对方**：
- **ARWorldMap 世界地图重定位**：能覆盖"同一物理空间、物体未被移动"的静态场景，技术上**几乎零风险**（Apple 标准框架用法，不需要新模型或离线评估）。**⛔ 但与用户"动的物体也要认出来"的要求直接冲突**——它锚定空间坐标，不是物体本身，物体一旦被移动，重定位 100% 成功也只会精确指向"物体不在的地方"。iPhone 11 无 LiDAR，重定位成功率预期明显低于有 LiDAR 机型（方向性推导，非本项目实测）。
- **物体级视觉重识别（re-ID）**：可复用 MobileSAM 图像编码器已有的 `image_embeddings`（无需新模型、无需额外推理），但**判别力完全未知、未经验证**——MobileSAM 训练目标是类别无关分割，不含任何度量学习目标，用它的池化特征做相似度比较是与训练目标无关的挪用。**诚实结论：通用场景（任意物体、可能有相似候选、可能被挪动）大概率会让用户失望**；YOLOv9-C 无实例级判别信号，帮不上忙。库变大后比对开销不是瓶颈，判别力才是。

**MVP 路径（若要做，按此顺序，每步都是决策点）**：
1. 只做 ARWorldMap 静物子集——直接满足用户诉求的一半（物体未挪动的情况），挪动过的物体诚实告知"未找到，请重新框选"。
2. **纯离线**评估 MobileSAM 池化特征判别力（同物体多角度 vs 不同物体相似度分布是否可分）——不涉及任何 App 内实现，零真机/ANE 成本，是决定 re-ID 值不值得继续投入的唯一关键数据。
3. 仅在第 2 步显示有信号时，才做人工确认型辅助（候选列表 + 用户确认，不自动认定，同 PIN-7"不确定的东西不能呈现成确定的"纪律）。

**触发重议条件**：若离线评估显示两组分布不可分，记录"现有管线不足以支撑 re-ID，需专门训练模型"，重议需用户对训练/获取新模型的成本给出明确授权（超出当前"仅用本地已有权重"范围）。

**待补数据**（architect_output §25.5）：① iPhone 11 ARWorldMap 重定位成功率（Debugger，真机）；② MobileSAM 特征判别力离线评估（ML_Vision，离线，**唯一当前就值得做的一项**，成本极低）；③ 若②有信号，需多天真实使用数据（Debugger+用户配合）。

⛔ **本节不产生 Builder/Debugger 编号任务**——按 Architect 纪律，"值得做"的部分（ARWorldMap 静物子集）若要推进，需要一次单独的、范围明确的裁决轮出可执行规格，不在本次侦察范围内直接派工。

------------------------------------------------------------
PHASE 4C — 稳定性测试 + 可选扩展（Day 7+）
------------------------------------------------------------

------------------------------------------------------------
Day 7 — 稳定性测试
------------------------------------------------------------

## Debugger
- [x] 50 个 Pin 写入 / 读取压力测试（无崩溃，延迟 < 200 ms）
  → ✅ **PASS（2026-08-24，真机实测，`devicectl --console` 直接采集控制台输出）**。清空到 `pins=0` 干净基线后，用 `-PinFixtureBatch 50:100` 启动参数驱动 50 次真实 `PinStore.create`（非模拟）：50/50 全部 `ok`，0 个 `FAILED`，全程无崩溃。写入耗时 t=1123.4ms→6151.7ms（~5.0s，与 49×100ms 的人为间隔吻合，逐次开销可忽略）。落盘核实：设备容器读出 `manifest.json` 50 条记录、`masks/` 目录 50 个 blob，一一对应，0 孤儿。**读取延迟**：强制冷启动重新加载这 50 条，`[PIN] load ok pins=50 orphans=0 ... ms=11.9`——11.9ms，远低于 200ms 判据线。测试用的临时诊断代码（打印 `ProcessInfo.arguments` 排查一次启动参数传递问题）已撤销并重新验证 clean build（Debug 模拟器 0 warning + 真机 BUILD SUCCEEDED）；原始 Pin 数据（9 条，含之前几轮真机测试的记录）已从备份完整恢复，`mtime` 与原始写入时间逐位吻合。
- [x] PinStore 并发写入安全验证（sessionQueue / main thread 混合写入）
  → ✅ **PASS（由构造满足，2026-08-24，源码审查）**。`pin.store.io` 确认为 serial queue（`FilePinStore.swift:82`，无 `.concurrent`），`create`/`update`/`delete`/`loadMaskImage` 全部状态读写逐方法核查过均在此队列内完成，调用方无论主线程还是别处发起，最终都在此排队串行。专门排查了"写在队列、读绕过队列"这个最容易出问题的点：`fetch(id:)`/`fetchAll()` 读的是独立的 `mainIndex` 镜像，唯一写入点 `publishSnapshot` 与唯一读取点均被限定在主线程（`@MainActor`），靠主线程串行性同步，不是绕过队列的裸读，不构成数据竞争。`TapInstanceManager` 是另一套独立保护面（`NSLock`），与本判据不相关。未发现真实竞态窗口，不需要压测——架构本身已从结构上排除并发写竞态。
- [x] Phase 2/3/4 模式切换下 PinManager 内存不泄漏（Instruments 确认）
  → ✅ **用户裁决通过（2026-08-24）**：「我的使用体验是通过的，而且联系起两个视频，也证明通过了」——用户结合自己的真机使用体验与两段录屏，直接认可本项通过，采信用户的最终裁决，覆盖此前"未用 Instruments、工具缺口"的保留意见（该保留意见仍如实保留在下方证据记录中，供以后查阅，但不再阻塞本项勾选）。
  → ⚠️ **部分证据，判据文本要求的工具与实际使用的工具不一致**（2026-08-24，真机 `--console` 采集，约 237s 连续会话，Detection→Segmentation→TapToSegment→Detection 循环约 8–9 轮，覆盖判据要求的量级）。**证据来源是 App 自身的 `Mem=`/`Memory:` 日志字段，不是 Instruments Allocations**——判据文本明确写"Instruments 确认"，这里未使用 Instruments，是否可作为等效替代未经 Architect/用户认可，如实标注为方法论缺口而非直接判定通过。
  → 观测到的模式：`detectionOnly` 阶段的内存基线在会话前 ~90s 内从 ~217–233 MB 抬升到 ~250–271 MB，此后在剩余 ~140s（覆盖之后约 6 轮完整模式切换）内**维持在该区间波动，未见持续单调爬升**——与 `SAM encoder cache hit rate` 日志（26/30→53/60→80/90→105/120，命中率稳定在 87–89%）吻合：缓存样本数随时间线性增长符合有界缓存的预期行为，不是无界增长的信号。⚠️ 但早段的一次性抬升本身未获解释（缓存预热 vs. 轻微泄漏后趋于饱和，两种假设当前数据无法区分），且 237s 仍短于典型泄漏检测所需时长，一次缓慢泄漏可能被这个时间窗口掩盖。**结论：未观测到无界增长的直接证据，但不满足判据要求的验证工具，且时长/时段划分不足以排除慢速泄漏**——不计为通过。
  → **第二组会话补充证据（2026-08-24，新进程 pid=54213，~162s，5 轮完整模式切换）**：同样的模式再现——`detectionOnly` 冷启动基线 ~185–200 MB，进入 segmentation/tapToSegment 后升到 ~230–290 MB，此后在剩余约 140s（覆盖 4 轮以上模式切换）内稳定在 ~240–285 MB 波动，**同样未见持续爬升**，会话结束时（262.8 MB）与会话中段水平相当。两组独立会话给出一致的"抬升后趋于平台"模式，进一步降低"未见增长"是采样窗口偶然性的可能，但仍未使用 Instruments，工具缺口结论不变。
- [ ] re-anchor 运行 5 分钟无 FPS 下降（YOLO 推理不受影响）
  → ⛔ **未按判据设计的方式验证，且发现一项新的、值得跟踪的性能信号**（2026-08-24）。会话被 Xcode 调试器手动终止（`Message from debugger: killed`）于 t≈237.1s，**未连续跑满 5 分钟**；且判据原意是"专门的、独立的 5 分钟 re-anchor 窗口"，实际这 237s 混合了三种模式循环切换，`tapToSegment` 单段最长连续窗口远不足 5 分钟。`objectTrackingEnabled` 现为 `false`（R36 未结案），本轮 `[REANCHOR]` 全部是原始"内容刷新"语义，不含物体跟随，与判据标题字面意义（"re-anchor 跑 5 分钟"）部分不符，但与我给用户的测试说明一致（已提前告知只测内容刷新）。
  → **意外发现**：`detectionOnly` 阶段的 `Inference time stats` 均值随会话推进持续变差——t=52283 时 mean=190.16ms/p95=231.36ms，t=115063 时 mean=220.78ms/p95=239.00ms，t=169333 时 mean=228.77ms/p95=248.09ms，t=200864 时 mean=254.45ms/p95=281.31ms，t=231258 时 **mean=277.85ms/p95=397.64ms**——从会话开始到结束推理均值上涨约 **+46%**，且末段（t≈216000–220000）`tapToSegment` 单帧推理多次冲到 300–420ms（FPS 瞬时降到 2.3–2.8），远差于会话前半段同模式下的 FPS 10–14。**登记为 R37（OPEN）**：现象是"持续 ~4 分钟摄像头+CoreML+录屏负载下推理延迟渐进恶化"，首要假设是 iPhone 11 热节流（ANE/GPU 持续高负载 + 屏幕录制本身的编码开销），但未经独立验证（没有温度/热状态 API 日志，没有"不录屏"对照组）——不排除是其他资源竞争。不计为软件缺陷登记到 Builder，仅记录观测，处置留给下一次专门复测（建议：同机同姿势、关闭录屏、单独跑 tapToSegment 5 分钟连续窗口，对比 `Inference time stats` 首尾两段）。
  → **qwait/R4 侧面数据反而更充分**：本轮 [D7'] 采到 14 个 tap 样本，`pool` 多次达到 n=3 上限并正确触发 FIFO 淘汰（如 t=142148.9 `[TAP#10] pool full → FIFO evicted oldest instance`），除已知的首次冷启动 parked 场景（qwait=163.0ms，标注为 slow/parked，非队列竞争）外，其余全部样本 `qwait` 落在 **0.1–0.4ms**，远低于 195ms 判据线——首次在三实例满载条件下拿到 qwait 数据，为 **P-17** 补上此前"仅单实例负载"的证据缺口，可将 P-17 的证据等级从"partial"上调。
  → **第二组会话（新进程 pid=54213，紧接第一组之后）为 R37 提供更强佐证**：本次是全新 App 进程（重新冷启动），但启动即异常慢——YOLO `Model loaded in 14187.31 ms`（对比第一组会话首次冷启动的 9821ms，进一步变慢）、MobileSAM `models loaded in 5873.08 ms`，随后 t=25739.5 出现单次 **`SAM encoder latency: 8364.84 ms`（cold start）**，比此前任何一次观测到的编码器冷启动都严重数倍。`Inference time stats` 均值全程在 **261–317ms** 区间波动，**从未回落到第一组会话开局时 ~190ms 的冷态水平**——即两组会话背靠背运行、设备中途没有冷却窗口，第二组从"已经偏热"的状态开始，与 R37 的热节流假设一致（比"重启进程会重置延迟"的反证假设更支持热节流而非软件状态残留）。**同一批次 qwait 数据不受影响**：本组 [D7'] 又采到 11 个样本，`qwait` 仍全部 0.1–0.3ms——说明"整体推理变慢"与"decode 队列排队时间"是两个独立现象，后者对前者不敏感，R4/P-17 的结论不受 R37 影响。R37 状态维持 OPEN，处置建议不变（关闭录屏、给设备充分冷却时间后单独复测）。
  → 🔓 **用户裁决（2026-08-24）：本项不再要求复测，转为直接修复。** 用户原话：「这个不用反复测了，就是没实现，你从Day 4开始测，每次我都配合，全都失败，你也没修改好，还让我在这里漫无目的的检测，我不想再浪费时间了。这一项你直接修改吧。re-anchor以前还有点，现在完全不follow。」——用户明确表示：①不再进行更多轮真机测试；②问题不是"证据不足"而是"功能没做对"（R36 描述的相机平移/物体运动混淆是根因）；③要求直接实施修复，不要求先经过又一轮 Architect/Debugger 流程再回来找用户配合测试；④额外报告了一个新的退化信号——"以前还有点跟随效果，现在完全不 follow"，需要在修复时一并核实是否存在比 R36 更严重的回归。**整体系统变慢（R37）用户明确排除在外**，不影响本项判定。
  → 🔧 **修复已实施（2026-08-24）**。先核实了"完全不 follow"这个新信号的成因：源码审查（`AnchorTracker.swift`/`TapInstanceManager.swift`）确认 `objectTrackingEnabled == false` 时每个实例的 `trackState` 永远停在 `.locked`，`trackedPoint` 终生不再写入第二次——这是 `false` 这个值本身按构造造成的零跟随，不是另一处新退化；这两个文件不在 git 跟踪范围内，无法做 diff 级别的回归排查，如实记录为证据边界，而非"确认无回归"。
  → **修复方案**：不是自行发明的新设计，是 architect_output.md §17.2.1 早已为这个失效模式预先批准、只是当时未立项的"抑制门"方案——"若 §17.4 的阈值在真机上被手抖误触发困扰，CoreMotion 的角速度可以作为抑制门（角速度高于阈值时判定为甩动，跳过 re-anchor）"，与 R36 处置记录本身列的第一个选项（"只在检测到相机基本静止时才允许 trackedPoint 移动"）一致。新增 [`Interaction/CameraMotionGate.swift`](JudgeE2/Interaction/CameraMotionGate.swift)：用 `CMMotionManager`（零 ANE/GPU 成本，纯 CPU 传感器融合，不占用 `ISSUE-P4-DECODE` 争用的那部分预算，也是候选 A/Vision 光流被否决的理由本身不适用于这个方案的原因）以 30Hz 读取陀螺仪角速度模长，`isPanning`（阈值 `panningThresholdRadPerSec=0.20 rad/s`，未经真机数据调优的首版取值）为真时判定相机正在甩动。在 [`CameraManager.checkAndFireReAnchor`](JudgeE2/Detection/CameraManager.swift:2629) 里，原来的 `if DriftDetector.objectTrackingEnabled { ... }` 改为 `if DriftDetector.objectTrackingEnabled && !CameraMotionGate.isPanning { ... }`——甩动期间整个跟踪/恢复搜索本轮直接跳过，`trackedPoint` 原地不动，该实例退化为与"跟踪关闭"完全一致的能力 A/B 定点刷新行为，不会决策去匹配错误物体。这只是一个否决项，从不触发跟踪、也不改动 `DriftDetector`/`AnchorTracker`/一致性门的任何既有逻辑——静止持机、物体自己移动的场景（陀螺仪读数本就接近零）不受影响，跟踪该场景本来就工作正常。`CameraMotionGate.start()/stop()` 接入 `CameraManager.start()/stop()`，与相机会话生命周期一致；无陀螺仪硬件时静默不生效（`isPanning` 恒 `false`），不会误关闭能力。`DriftDetector.objectTrackingEnabled` 改回 `true`，文档注释已更新说明这是本次修复而非又一轮未经复核的翻转。
  → **构建验证**：Debug 模拟器 clean build（`xcodebuild ... -destination 'generic/platform=iOS Simulator'`）与 Debug 真机架构 clean build（`-destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO`）均 **BUILD SUCCEEDED**，新文件已正确加入 `project.pbxproj`（`Interaction` 分组为普通 PBXGroup、非同步文件夹，手动补齐 PBXFileReference/PBXBuildFile/组成员/Sources 编译阶段四处），无新增 Swift 警告。
  → ⚠️ **尚未经真机复核** —— 这是修复实施记录，不是通过判定。角速度阈值是未调优的首版取值，理论上可能偏松（甩动时仍误放行）或偏紧（正常持机小幅移动时过度抑制跟踪），需要实际使用中观察。**不要求用户按协议专门测试**，按用户原话的精神，正常使用中如果再遇到 mask 跟错物体或完全不跟随，直接反馈即可，无需重新走一遍真机取证流程。
  → 📋 **首次真机日志片段（2026-08-24，修复后）——用户口头报告"re-anchor 确实没激活"，但日志本身显示跟踪确有运行**：`[TRACK][inst#0] locked → tracking`（t=62799.3）、多次 `tracking search:*ms best divergence:*lum → (x,y)`、一次 `tracking → lost`（t=72691.6，divergence 25.4lum ≥ lost 阈值）随即 `lost → recovered`（t=72697.0）、`inst#1` 也有对称的 `locked→tracking`/`lost`/多次 `recovery search ... still lost` 序列——这些都是 `objectTrackingEnabled=true` 分支下才可能出现的日志格式，说明跟踪机制本身在跑，不是完全零激活。**用户所指的"没激活"更可能是别的现象**（例如启动早期出现的一条 CoreMotion 权限相关系统日志：读取 `com.apple.CoreMotion.plist`（Managed Preferences）报 `Operation not permitted`——这是 iOS 系统对"受管设备偏好"文件的常见噪音日志，不特定于本次改动，也不代表 `CMMotionManager.startDeviceMotionUpdates` 本身失败，但未在这份日志里找到确认 `CameraMotionGate` 拿到过陀螺仪样本的直接证据，`isPanning` 是否真的被计算过尚未证实）。经用户裁决**暂不深挖，搁置**，状态维持"修复已实施、未经真机复核"不变，待用户后续视情况决定是否继续排查 CoreMotion 数据流本身。
- [x] 录制演示视频：tap → 长按固定 → 标注 → 重访完整流程（不含 re-anchor 跟踪——`objectTrackingEnabled` 因 R36 维持 `false`，已提前告知用户此次只测原始内容刷新，不测物体跟随）
  → ✅ **PASS（2026-08-24，`ScreenRecording_08-24-2026 14-07-24_1.MP4`，`ffprobe` 核实时长 148.6s，828×1792 HEVC）**。人工抽帧核实覆盖的环节：① Detection/Segmentation/Tap to Segment 三模式切换，且 Tap to Segment 下确认多实例并存（instance marker `1`/`2` 同屏，pool n=2，与 P-17 的 3-实例池观测一致）；② **Pin List** 界面（"Pins" 列表，含多条历史 Pin，创建时间跨 09:09–14:07，逐条可见 tag 文本）——满足"重访"环节的界面证据；③ **Edit Pin** 界面：Tag 输入框处于聚焦状态、拼音键盘弹出、正在输入内容（`, h阿里`，5/64 字符计数），随后同一 Pin 以更新后的 tag 重新出现在列表顶部（"Created 8/24/26, 14:07"）——直接可见"编辑→保存→列表反映更新"这条链路，满足"标注"环节。⚠️ 未在抽样帧中直接捕捉到"长按 mask 触发 PinCreationSheet"这一具体手势瞬间（抽帧间隔 10–15s，可能落在手势之间）；该动作本身在此前多个会话的真机日志里已反复出现且行为一致（如第一组会话 t=37694.8 `[PIN] long-press hit existing mask (gen #1) — opening PinCreationSheet`），本次视频未覆盖到帧不代表功能缺失，只是抽样未命中这一时刻。综合视频的直接画面证据 + 历史日志的功能一致性，判定本项满足。⚠️ 本视频无配套控制台日志，无法做时间戳级别的日志-画面互证，仅作视觉演示证据使用。

## Architect
- [x] 审阅 Phase 4 性能数据，冻结 Phase 4 架构
  → 见 architect_output.md §26（Day 7 稳定性数据整合 + 冻结裁决 D-26.1，R34/R36/R37 显式携带进入 Phase 5，冻结表面 vs Phase 5 可改表面已列出）
- [x] 定义 Phase 5 入口点
  → 见 architect_output.md §27（五条 Focus 映射到现有 UI 文件、范围边界裁决、Day 1 入口点裁决 D-27.1：新建 UI/BottomControlBar.swift）

-->

------------------------------------------------------------
可选扩展（Optional，Day 8+ 按需决定，Phase 4 遗留，未随归档处理）
------------------------------------------------------------

## Optional A — 双指框选（Box Prompt 交互模式）
- [ ] 双指拖拽框选目标区域 → box prompt → SAM decode
- [ ] 与单指点击并存（手势识别器区分）
- [ ] 框选 prompt 格式复用 Phase 2 `SAMBoxDecoder`

## Optional B — 分割结果导出
- [ ] 单 Pin 导出：PNG（含 mask alpha 通道）+ JSON（canonicalPoint + tag + note）
- [ ] 批量导出：ZIP 压缩包，含所有 Pin 数据
- [ ] 分享入口：系统 `UIActivityViewController`

## Optional C — MobileSAM 模型升级评估
- [ ] 评估 EfficientSAM / SAM 2 在 iPhone 11 上的 CoreML 可行性
- [ ] AB 测试：当前 MobileSAM fp16_milfix vs 候选模型（encoder latency + 分割质量）
- [ ] 不替换现有模型，仅评估；替换须通过完整 Phase 3 验收门控

## Optional D — A-1 候选选择规则改进（ML_Vision）
- [ ] 定向复采 stability 数据（含 `candidates` 详细日志）
- [ ] 分析「取最小」退化为 ch0 的频率和场景
- [ ] ML_Vision 给出候选选择改进方案，提交 Architect 裁决

Deliverable:
Phase 4A：mask 从快照升级为活跃跟踪资源
Phase 4B：活跃 mask 可被"钉住"、标注、持久化、重访
Phase 4C：系统级稳定性 + 性能基线冻结
Optional：交互扩展 / 导出 / 模型升级（按需执行）

------------------------------------------------------------
PHASE 5 — UI Redesign & App Polish
------------------------------------------------------------

Objective:
Transform prototype into product-like app.

Focus:
- Bottom control bar
- Mode switching (Detect / Segment / Annotate)
- Clean animation
- Improved overlay style
- Settings panel

Result:
Complete functional MVP app.

Scope Guard（继承自 architect_output.md §26.5 / §27.3，本阶段全程有效，不是本节新设规则）:
- 本阶段是**纯 UI/UX 层工作**。冻结表面（`Detection/CameraManager.swift` 核心管线与 `checkAndFireReAnchor` / `Interaction/DriftDetector.swift` / `Interaction/AnchorTracker.swift` / `Interaction/CameraMotionGate.swift` / `Interaction/TapInstanceManager.swift` / `Persistence/`(PinStore 系列) / `Segmentation/SAMEncoder.swift` / `SAMDecoder.swift` / `MaskRenderer.swift` 的算法与判定逻辑 / `TemporalManager.swift`）**不得随手修改**。
- 若某项 UI 需求"看起来"必须触碰以上任一文件的判定逻辑（例如把 "Annotate" 做成第四个独立行为模式、或把 `TapLoadingIndicator` 改造成进度条），**必须先回 Architect 走一次范围明确的裁决**，不得由 Builder 在 UI 分支里顺手实现。
- 遗留保留项 **R34（P0，PinStore 计数异常，未闭合）/ R36（跟踪抑制门修复已实施，未经真机复核）/ R37（推理延迟渐进恶化，机制未证实）** 随 Phase 4 冻结一并携带进入本阶段。本阶段遇到与三者相关的新证据（尤其 R36：`CameraMotionGate` 是否真的拿到陀螺仪样本；R37：录屏/持续负载下的延迟趋势）应**记录并上报**，不得就地处置，不得被静默视为已解决。
- Mask 呈现的**数值**（填充色/alpha/描边宽度）改动须走既有 C-7 准入程序（算术检验 + 撞色目视测试，§12/§14.4）；候选选择**算法**（R3 六项参数，`minComponentPx=30` 等）INVIOLABLE。

------------------------------------------------------------
Day 1 — 底部控制栏 + 模式命名裁决
------------------------------------------------------------

## Architect
- [ ] 裁决 "Annotate" 命名问题（architect_output.md §27.2 已给出倾向性意见，未代为最终决定）：`.tapToSegment` 的 `displayName` 是否改呈现为 "Annotate"（纯字符串改动），或需要新增独立语义模式（行为语义变更，需另行裁决）——**须在本日 Builder 动工前给出**，避免夹带进底部栏任务

## Builder
- [ ] 新建 `UI/BottomControlBar.swift`（裁决 D-27.1）：合并 `ContentView.swift:256-261` 的 App Mode 分段选择器与 `:414-441` 的悬浮"模式快切"按钮为一条底部常驻栏，绑定 `ContentView` 已有的同一个 `$mode: AppMode` 状态——不新建状态、不复制状态
- [ ] 不改 `AppMode` 枚举本身（三个 case 不变）；若本日 Architect 裁决改名，仅改 `displayName` 字符串，不改 case 名或下游判定逻辑
- [ ] 移除旧的悬浮快切按钮与折叠面板里的 App Mode picker，避免同一状态出现两个重复入口

## Debugger
- [ ] 真机验收：新底部栏可见即可切换三种 `AppMode`，切换后行为与切换前完全一致（同一 `mode` 状态、同一套下游逻辑，无重复触发）
- [ ] 确认冻结表面未被触碰（核对本日 diff 范围仅限 `UI/` 目录）

------------------------------------------------------------
Day 2 — Settings 面板职责分离
------------------------------------------------------------

## Builder
- [ ] 新建产品化 Settings 界面（复用现有 `.sheet()` 呈现模式，同 `PinListView`/`AnnotationView`），只暴露用户需要的项
- [ ] 现有 `Compute Units` 选择 / `Encoder Res` AB 测试 / `Force Slow Path (testing)` / P-9 几何往返自检按钮等纯工程调试项，收敛到明确标记的"开发者/调试"分区或 `#if DEBUG` 条件编译——**保留，不删除**
- [ ] `Perf Quiet Log` 开关按用户可感知程度判断归属（用户设置 vs 调试项）；不确定时按调试项处理

## Debugger
- [ ] Release 构建下调试项不可见/不可达；Debug 构建下全部原有功能逐项行为等价

------------------------------------------------------------
Day 3 — Overlay 视觉样式改版
------------------------------------------------------------

## Builder
- [ ] Mask 呈现改版（填充色 / alpha / 描边宽度等**数值**）——先过 C-7 准入（算术检验 + 同色系撞色目视测试）再合入，非本阶段新规则
- [ ] R3 六项候选选择**算法**参数（`minComponentPx=30` 等）INVIOLABLE，不得触碰

## Debugger
- [ ] C-7 准入验收记录（算术检验结果 + 撞色测试截图/结论）
- [ ] 三模式 + 多实例场景（同屏 2–3 实例）下视觉回归检查，确认颜色/描边可辨识度不下降

------------------------------------------------------------
Day 4 — 交互动画与反馈
------------------------------------------------------------

## Builder
- [ ] 复用/精修现有动画基元：`TapRippleEffect`（一次性涟漪）、`TapLoadingIndicator`（脉冲圆环）、`TapFailureIndicator`、面板展开/收起 spring 动画
- [ ] ⛔ `TapLoadingIndicator` 的"无进度条"语义是 §10 FINAL 裁决——本阶段可重新设计视觉表现，**不得**改造成进度指示（撞 §10.4，需重开裁决才能改）
- [ ] Pin 创建 / 编辑 / 重访流程的过渡动画（长按固定 → sheet 弹出 → 列表刷新）

## Debugger
- [ ] 真机验收动画流畅度；顺带留意 `debug_report.md` §40 已登记的 `PinCreationSheet.thumbnailImage` 主线程 PNG 重编解码问题（P2，修复与否待 Architect 裁决）在本日新动画下是否可感知——**仅记录，不代为修复**

------------------------------------------------------------
Day 5 — UX 细节打磨 + 可用性走查
------------------------------------------------------------

## Builder
- [ ] 空状态 / 错误状态文案与呈现（Pin List 为空、mask 生成失败等）
- [ ] 手势可发现性改进（长按固定 Pin 的首次使用提示，若现有 UI 缺失）

## Debugger
- [ ] 端到端可用性走查：tap → 长按固定 → 标注 → 重访 完整流程，按"第一次使用的新用户"视角逐步验证
- [ ] 记录发现的 UX 问题；架构层问题不在本日就地处置，按 Scope Guard 上报

------------------------------------------------------------
Day 6 — 性能清理（UI 层）
------------------------------------------------------------

## Builder
- [ ] UI 层性能清理（视图重绘范围、不必要的 `@Published` 触发等），范围限定在 `UI/` 目录，不触碰冻结表面
- [ ] ⛔ 不得"顺手修一下" R36 或 R37——二者均非 UI 层问题（architect_output.md §27.4），触碰需先走冻结表面流程（§26.5），不属于 UI polish 的自然延伸

## Debugger
- [ ] 执行 `debug_report.md` §40.7 给出的 R37 复测协议（基线 / sheet-only / sheet+keyboard 三组，各 ≥5 分钟，关闭 quiet log，按时间戳精确划窗，n≥20，首尾基线核对热漂移）——**在本日 UI 改动之前先摸清 R37 背景噪声基线**，为 Day 7 基准测量提供参照，避免把"UI 改动的锅"和"R37 本就存在的锅"混为一谈
- [ ] 本日若在正常使用中观察到 R36/R37 新证据（例如是否有更多设备日志佐证 `CameraMotionGate` 拿到过陀螺仪样本），记录并上报，不就地处置

------------------------------------------------------------
Day 7 — 基线基准 + Demo 构建
------------------------------------------------------------

## Debugger
- [ ] Baseline benchmark：对照 Day 6 摸清的 R37 背景噪声基线，评估 Phase 5 全部 UI 改动是否引入新的性能回归
- [ ] 补做遗留的 Phase 4C Day 7 "re-anchor 5 分钟无 FPS 下降" 验收项——该项此前被用户明确重定向为直接修复（R36 修复），未完成原判据本身的验证，见 tasks.md Phase 4C 记录
- [ ] Demo build：Release 配置，真机验证核心用户旅程（tap→分割→长按固定→标注→重访）完整可用

## Architect
- [ ] 审阅本阶段是否出现任何触碰冻结表面的例外申请，逐项裁决（若有）
- [ ] 视 Day 7 判据结果，决定 Phase 5 冻结裁决

Deliverable:
Complete functional MVP app —— 底部常驻控制栏、职责分离的 Settings 面板、经 C-7 准入的视觉改版、精修动画、可用性打磨、UI 层性能清理、Day 6/7 性能基线与 Demo build。R34/R36/R37 状态原样携带进入下一阶段，不因 Phase 5 完成而自动视为解决。
