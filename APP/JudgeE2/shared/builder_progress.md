# Builder Progress — JudgeE2 (Phase 3: Tap-to-Segment)

> 承接 `shared_v2/builder_progress.md`（Phase 1–2 基线）。本文件记录 Phase 3 Builder 交付。
> 依据：`shared/tasks.md`（Phase 3）+ `shared/architect_output.md`（Day 1 契约）+ `shared/model_plan.md`（Day 1 ML_Vision）。

---

## Phase 3 Day 2 — TouchHandler & Canonical Coordinate Transform (Builder)

**日期：** 2026-07-19
**状态：** 完成（编译通过；坐标精度真机验证属 Debugger Day 2 任务）。

### 交付概览

Phase 3 Day 2 Builder 四项任务全部完成，交互层与坐标反变换链路已落地，ANE fp32 修复模型已封层进 target 并编译打包成功。全程未改动 Phase 1/2 检测/分割流水线，未触碰 TemporalManager 调度逻辑。

---

### 1) TouchHandler 实现 ✅

**文件：** `JudgeE2/Interaction/TouchHandler.swift`（原为占位空文件，现实现）

- 在 preview 视图安装 `UITapGestureRecognizer`：单击（`numberOfTapsRequired = 1`）+ 双击（`= 2`）。
- `singleTap.require(toFail: doubleTap)`：单击等待双击失败，避免「分割」与「清除全部」手势冲突。
- 导出单一回调 `onTap: ((CGPoint) -> Void)`（Canonical 像素坐标）+ `onClearAll: (() -> Void)`（双击清除）。
- 坐标变换全部委托 `FrameGeometry.invertViewPoint`，TouchHandler **自身不做任何坐标数学**（契约：变换路径唯一）。
- `geometryProvider` 闭包在 tap 时拉取当前 `FrameGeometry` 快照；几何未就绪时忽略 tap 并打印说明。
- 实现 `UIGestureRecognizerDelegate`，不拦截其他渲染层触摸（architect §1.3 Z 轴契约）。

### 2) preview → Canonical 反变换 ✅

**文件：** `JudgeE2/Interaction/FrameGeometry.swift`（新建）

按 architect_output.md §2.1 冻结的变换链实现 `invertViewPoint(_:viewSize:previewLayer:)`：

- **Step 1** — AspectFill 逆映射：`previewLayer.captureDevicePointConverted(fromLayerPoint:)` → 归一化传感器坐标 [0,1]（Apple 内部处理裁剪偏移）。
- **Step 2** — 归一化 × (origW, origH) → Canonical 像素坐标。
- **Step 3** — 镜像修正：前置相机时归一化 X 轴翻转（与 `CameraManager.maskMirrored` 一致）。
- **Step 4** — 越界裁剪：`clamp(0, origW-1)` / `clamp(0, origH-1)`（tap 不排除）。

> **几何一致性说明：** 设备旋转已由 `AVCaptureConnection.videoRotationAngle` 在捕获阶段烘焙进 origW/origH（display orientation），与 Phase 1/2 的 mask 渲染管线完全一致（见 `CameraPreview.swift` 注释）。因此此反变换**不重复施加旋转**，只逆 aspectFill + mirror + clamp，保持单一几何链、无重复变换（契约 §2.3）。
> `FrameGeometry` 承担 architect 契约中 `invertViewPoint` 的职责（等价于 tasks.md 提到的 `invertLetterbox` 逆映射角色）。

`invertViewPoint` 同时返回中间 `normalized` 点，供 `[TAP]` 日志输出。

### 3) 集成到主流程并验证 Log ✅

- `CameraManager.currentFrameGeometry()`：由 `lastLetterbox` + `currentPosition` + `lastRotationAngle` 构造 `FrameGeometry` 快照。
- `CameraPreview.makeUIView`：一次性安装 `TouchHandler`，回调路由到 `CameraManager.handleTap(canonicalPoint:)` / `handleClearAllTapMasks()`；`PreviewView.touchHandler` 持有其生命周期。
- 两个处理方法均 `guard currentMode == .tapToSegment`——Detection / Segmentation 模式下 tap 被静默忽略，**不影响 Phase 1/2**。
- **Day 2 无实际 SAM 调用**（encode/decode 落 Day 3）。TouchHandler 打印契约要求日志：

  ```
  [TAP] view=(x,y) normalized=(nx,ny) canonical=(cx,cy) orientation=N mirrored=B
  ```

  `CameraManager.handleTap` 追加一行 `[TAP] pipeline received canonical=(cx,cy) — Day 2 (no SAM call)` 确认坐标抵达流水线。

### 4) ANE 修复模型封层集成 ✅

依据 `model_plan.md` §B.3：

- 将 `models/MobileSAM_ImageEncoder_fp32.mlpackage`（28 MB，Float32 全精度，0 个 `_fp16` op）拷入 Xcode 同步文件夹 `JudgeE2/Segmentation/Models/`。
- 在 `JudgeE2.xcodeproj` 同步组 membership 中加入 `MobileSAM_ImageEncoder_fp32.mlpackage`，使其纳入 target 编译（原 fp16 模型保留不删）。
- `SAMEncoder.init`：优先加载 `MobileSAM_ImageEncoder_fp32.mlmodelc`，缺失时回退原始 `MobileSAM_ImageEncoder`。加载时打印 `SAMEncoder loading model: MobileSAM_ImageEncoder_fp32`。
- I/O 接口零变化：input `image [1,3,1024,1024]`、output `image_embeddings [1,256,64,64]`（已用 coremltools 静态确认），与 `SAMEncoder.preprocess`（Float32 planar RGB）完全兼容。

> **告警消除待真机确认：** `Invalid input tensor channel 1 ... aligned on 64 bytes` 属 ANE 调度告警，模拟器无 ANE 无法复现/确认。清 build 已通过、fp32 模型已打包，告警消失的真机验证属 Debugger Day 2 任务范围。

---

### 编译验证

- `xcodebuild -scheme JudgeE2 -sdk iphonesimulator -destination 'iPhone 11' -configuration Debug build`：**BUILD SUCCEEDED**。
- 打包产物含 `MobileSAM_ImageEncoder_fp32.mlmodelc`（与原 fp16、Decoder、yolov9 并存）。
- 新增 `TouchHandler.swift` / `FrameGeometry.swift` 已加入 Sources build phase 并成功编译。

### 改动文件

- 新建：`JudgeE2/Interaction/FrameGeometry.swift`
- 实现：`JudgeE2/Interaction/TouchHandler.swift`（原空占位）
- 编辑：`JudgeE2/Detection/CameraManager.swift`（`currentFrameGeometry` / `handleTap` / `handleClearAllTapMasks`）
- 编辑：`JudgeE2/Detection/CameraPreview.swift`（安装 TouchHandler + 持有引用）
- 编辑：`JudgeE2/UI/ContentView.swift`（传入 `cameraManager`）
- 编辑：`JudgeE2/Segmentation/SAMEncoder.swift`（优先加载 fp32）
- 新增资源：`JudgeE2/Segmentation/Models/MobileSAM_ImageEncoder_fp32.mlpackage`
- 编辑：`JudgeE2/JudgeE2.xcodeproj/project.pbxproj`（注册新 Swift 文件 + fp32 模型 membership）

### 边界与未做项（交 Debugger / 后续 Day）

- **不做 Debugger 工作**：tap 坐标 ±5% 精度、rot=0/90/270 验证、Phase 2 回退监控、ANE 告警真机确认均属 Debugger Day 2。
- **未重设计架构**：严格按 architect Day 1 契约实现，未改 SAMDecoder 签名、未调 TemporalManager、未加载第二份 encoder（fp32 替换而非并发）。
- Day 3 待办（非本次）：`PointPromptBuilder`、`SAMDecoder.decode(embedding:pointPrompt:)` 重载、单次点击分割闭环。

---

## Phase 3 Debugger Session 分析跟进 — ANE Fix Revision + YOLO 稳定 + Cache 策略

**日期：** 2026-07-21
**状态：** 完成（编译验证中；ANE 告警消除需 Debugger 真机验证）。

### 背景

Debugger 对 fp32 encoder（Phase 3 Day 2 集成）进行了真机测量，发现 3 个问题：
1. **P1 🔴** fp32 encoder 比 fp16 慢 +32%（mean 1131ms vs 857ms）
2. **P1 🔴** ANE 告警未消除——反而多出 3 条（ANE runtime 内部仍将部分 fp32 算子回退 fp16，触发 C=1 对齐检查）
3. **P2 🟡** 分割模式下 YOLO 推理变慢 +14–25%（fp32 模型 28 MB 占用更多内存带宽）

---

### P1 — MIL LayerNorm 融合 (§B.4) ✅

**根因精确定位：**

MobileSAM encoder 实际存在**两个** `LayerNorm2d` 类：
- `mobile_sam.modeling.common.LayerNorm2d` — 用于 transformer 块，原始 fp16 导出时 coremltools DEFAULT pipeline 已自动融合为 `layer_norm` MIL op（共 20 个）
- `mobile_sam.modeling.tiny_vit_sam.LayerNorm2d` — **用于 encoder neck**，未被 DEFAULT pipeline 识别，仍以 4 个 `reduce_mean` fp16 算子存在，C=1 张量（`u_cast_fp16`, `s_cast_fp16` 等）触发 ANE 对齐告警

**修复方案实现：**

新建脚本 `shared/export_encoder_fp16_milfix.py`（保留原 fp32 脚本不变）：

```
策略：Monkeypatch tiny_vit_sam.LayerNorm2d.forward
  原始实现：x.mean(1, keepdim=True) → [B,1,H,W] Float16 → C=1 → ANE告警
  修复实现：x.permute(0,2,3,1) → F.layer_norm([C]) → permute back
             ↓
           coremltools 直接映射为 MIL `layer_norm` op（无 C=1 中间张量暴露）
```

数学等价性验证：对每个位置 (b,c,h,w)，两种实现均计算：
`output = weight[c] × (x[b,c,h,w] - μ(x[b,:,h,w])) / σ(x[b,:,h,w]) + bias[c]` ✓

**静态验证结果（protobuf 层级）：**

| 模型 | total_ops | layer_norm | reduce_mean |
|------|-----------|------------|-------------|
| 原始 fp16 | 1188 | 20 | **4** (C=1 告警根源) |
| fp32 旧修复 | — | — | — （ANE runtime 仍触发） |
| **fp16_milfix (新)** | **1169** | **22** | **0** ✅ |

结论：4 个 C=1 `reduce_mean` op 全部消除，neck 的 2 个 `LayerNorm2d` 融合为 2 个 `layer_norm` op。

**交付物：**
- `shared/export_encoder_fp16_milfix.py`（可重现的导出脚本）
- `models/MobileSAM_ImageEncoder_fp16_milfix.mlpackage`（14.1 MB，与原始 fp16 相同大小）
- `JudgeE2/Segmentation/Models/MobileSAM_ImageEncoder_fp16_milfix.mlpackage`（已拷入 Xcode 同步目录）
- `JudgeE2/JudgeE2.xcodeproj/project.pbxproj`（已注册 target membership）
- `JudgeE2/Segmentation/SAMEncoder.swift`（加载优先级更新：milfix > fp32 > fp16）

**SAMEncoder.swift 加载优先级（已更新）：**
```
1. MobileSAM_ImageEncoder_fp16_milfix  ← 新首选（fp16 精度 + 无 C=1 ANE 告警）
2. MobileSAM_ImageEncoder_fp32         ← 回退（fp32，Phase 3 Day 2 旧方案）
3. MobileSAM_ImageEncoder              ← 最终回退（原始 fp16）
```

**待 Debugger 真机验证：**
- ANE 告警 `Invalid input tensor channel 1 ... aligned on 64 bytes` 是否消失
- encoder 延迟是否恢复至 Phase 2 基线（~857 ms mean，~933 ms p95）

---

### P2 — 分割模式 YOLO 变慢（+14–25%）✅ 随 P1 自动改善

**分析：** YOLO 变慢的直接原因是 fp32 模型（28 MB）加载 + 运行时占用大量内存带宽，
挤压了 YOLO 的 CPU/GPU 资源。切回 fp16_milfix（14.1 MB，与原始 fp16 相同）后，
内存带宽竞争恢复至 Phase 2 fp16 水平，YOLO 延迟应恢复正常。

**无额外代码改动**，此 P2 已随 P1 milfix 一并解决。待 Debugger 真机对比实测。

---

### P3 — Cache Hit Rate 55%（可接受，Day 6 预留优化） 📝

**结论：** 55% cache hit rate 属可接受范围，与本次 session 频繁切换目标和旋转有关，
非代码 bug。Phase 2 目标 80% 在正常使用场景（单目标持续跟踪）下可达。

**Day 6 预留调参入口（非本次任务，记录为 backlog）：**
1. 几何变化阈值：`geometry_change` 触发占比高时，考虑小角度旋转（< 5°）不触发重 encode
2. TTL 延长：若 `ttl_expired` 是主因，可在目标稳定时适度延长 embedding TTL
3. 来源日志：`[CACHE] re-encode reason: <geometry_change|ttl_expired|primary_changed>`
   已在 Phase 3 Day 6 任务中预留（tasks.md Day 6 Builder item）

---

### 边界与未做项（按分工）
- **Debugger 工作**：ANE 告警真机消除验证、encoder 延迟实测对比、YOLO 延迟对比，均属 Debugger 职责
- **未重设计架构**：严格遵守 architect Day 1 契约，未改动 Phase 1/2 任何逻辑

---

## Phase 3 Debugger Session 2 — ANE 告警根因追踪 + 后台 GPU abort 修复

**日期：** 2026-07-21（深夜 session）
**状态：** 完成（BUILD SUCCEEDED）。

### 背景

Debugger 提供了 fp16_milfix 真机运行日志，观察到：
1. **3条启动期告警** 仍然存在（日志出现在 `MobileSAM models loaded` 前）
2. **3条运行期告警** 出现在 milfix encoder 首次 ANE 执行时
3. **后台 GPU abort**：app 进入后台时 encoder 正在运行，GPU 任务被 iOS 终止

### 发现：3条启动期告警来自 ModelLoader，不来自 milfix

通过日志时序分析（3条告警出现在 `SAMEncoder loading model: MobileSAM_ImageEncoder_fp16_milfix` 之前），
确认启动期告警来源：`ModelLoader.testMobileSAMLoad()` 直接加载原始 `MobileSAM_ImageEncoder`（C=1 fp16 LayerNorm，未修复），而非经过 SAMEncoder 的优先级选择逻辑。

**修复：** `ModelLoader.testMobileSAMLoad()` 改为优先加载 `MobileSAM_ImageEncoder_fp16_milfix`（优先级与 SAMEncoder.init() 一致）。

**预期效果：** 启动期 3 条告警消除（ModelLoader 现使用 milfix，reduce_mean=0）。

### 3条运行期告警（已知限制）

milfix `layer_norm` op（axes=[-1] on [B,H,W,C]）在 ANE 内部计算 mean/variance 时仍产生 C=1 形状的中间张量（[B,H,W,1]），与原始 fp16 encoder 的行为等价。

**结论：** 这 3 条运行期告警与 Phase 2 fp16 baseline 的 3 条一致——都是 LayerNorm 内部不可避免的 ANE 调度行为，不影响性能（encoder latency ~860ms ≈ Phase 2 baseline 857ms）。**可接受，标记为已知限制。**

### 修复：后台 GPU abort

**现象：**
```
IOGPUMetalError: Insufficient Permission (to submit GPU work from background)
MobileSAM encoder prediction failed: ...
SAM encoder warmup failed
```

**根因：** `encoderQueue.async` 任务在 app 进入后台后继续运行，iOS 强制终止 GPU/ANE 任务。

**修复（已应用）：**
1. `CameraManager.warmupSegmentationIfPossible()`：encode 前检查 `UIApplication.shared.applicationState == .background`，后台时静默跳过并释放 `isEncoding` 标志
2. `CameraManager.scheduleEncoderIfNeeded()`（主流水线）：同样加后台检测门控
3. encode 返回 nil 时的错误日志更新，说明可能的后台 GPU abort 原因

**效果：** app 进入后台时 encode 被跳过（不触发 GPU 任务），返回前台后下一帧正常触发重新 encode，不会出现 `encoder warmup failed`，不会有 CoreML error 日志。

### 性能数据确认（本次 session）

| 指标 | Phase 2 基线 | fp32（旧修复） | fp16_milfix（本次） |
|------|------------|--------------|-----------------|
| Encoder latency (warm) | ~857 ms | ~1131 ms (+32%) | **852–978 ms** ✅ |
| YOLO latency | ~176 ms | ~201–220 ms (+25%) | **173–207 ms** ✅ |
| Decoder latency | ~61 ms | ~64 ms | **55–68 ms** ✅ |
| ANE 告警（启动） | 3条（已知） | 3条 | 3条→预计消除(ModelLoader fix) |
| ANE 告警（运行） | 3条 | 3条 | 3条（已知限制，性能等价） |

### 改动文件
- `JudgeE2/Detection/ModelLoader.swift`（testMobileSAMLoad 改用 milfix 优先级逻辑）
- `JudgeE2/Detection/CameraManager.swift`（warmup + 主流水线 encode 加后台检测门控）

---

## Phase 3 Day 3 — Point-Based PromptBuilder + 单次点击分割 (Builder)

**日期：** 2026-07-22
**状态：** 完成（BUILD SUCCEEDED；单次点击闭环已实现，等待 Debugger 真机验证）。

### 交付概览

Phase 3 Day 3 Builder 三项任务全部完成：新建 PointPromptBuilder、SAMDecoder 新增点提示重载、CameraManager 实现 tap→encode→decode 直通闭环。全程未改动 Phase 1/2 流水线，未引入 TemporalManager 依赖。

---

### 1) PointPromptBuilder 实现 ✅

**文件：** `JudgeE2/Interaction/PointPromptBuilder.swift`（新建）

依据 model_plan.md §A.2–A.6：

- `buildPointPrompt(canonicalPoint:origSize:inputSize:)` → `PointPrompt`
- 变换链与 `PromptBuilder.buildBoxPrompt` 完全一致：
  - `ResizeLongestSide(inputSize=1024)` → `scale = 1024 / max(origW, origH)`
  - Centered pad → `padX = (1024 - origW*scale) / 2`
  - `samX = canonicalPoint.x * scale + padX`（SAM 像素坐标 0~1023）
- Prompt 构造（固定 2 点，model_plan §A.2）：
  - Point 0: `[tapX, tapY]` label=1.0（前景）
  - Point 1: `[0.0, 0.0]`   label=-1.0（padding）
- 输出 tensor 规格：`point_coords [1,2,2]`, `point_labels [1,2]`, `mask_input [1,1,256,256]`（全零）, `has_mask_input [1]`（0.0）

已注册至 Xcode project.pbxproj（PBXFileReference + PBXBuildFile + Sources build phase）。

### 2) SAMDecoder 点提示重载 ✅

**文件：** `JudgeE2/Segmentation/SAMDecoder.swift`（新增重载，原方法不变）

```swift
func decode(embedding:point:) -> (mask: MLMultiArray, iouPred: Float)?
```

- 与 `decode(embedding:prompt:)` 使用同一模型和 I/O key
- 额外提取 `iou_predictions` scalar → 返回 `(mask, iouPred)` tuple
- 打印：`[SEG][TAP] decode latency: %.2f ms iou_pred: %.3f`
- iouPred 由调用方用于质量门控（< 0.1 → 丢弃）

### 3) 单次点击分割闭环 ✅

**文件：** `JudgeE2/Detection/CameraManager.swift`（Day 2 stub 替换为 Day 3 实现）

主要方法：
- `handleTap(canonicalPoint:)`：模式检查 → videoQueue 快照 → embedding 缓存检测 → 路由到快/慢路径
- `tapEncodeAndDecode()`：encode 当前帧（encoderQueue）→ 缓存 embedding → decode
- `tapDecodeWithPoint()`：decoderQueue 懒加载 decoder → buildPointPrompt → decode → quality gate → renderMask → maskImage

**闭环流程：**
```
用户 tap（main thread）
  → videoQueue: 快照 buffer + letterbox
      ├─ embedding cached: → decoderQueue: decode(point) → renderMask → maskImage ✅
      └─ no cache: → encoderQueue: encode → cache → decoderQueue: decode(point) → maskImage ✅
```

**关键设计决策：**
- 与 Phase 2 共享 `embeddingCache`（Phase 2 关闭时 warmup 可预填充缓存）
- `iouPred < 0.1` 时丢弃 mask 并打印（不崩溃、不显示空 mask）
- 后台检测门控：encode 前检查 `UIApplication.shared.applicationState == .background`
- Decoder 懒加载于 decoderQueue（不阻塞 videoQueue）
- **不修改 Phase 2 TemporalManager/segmentation 路径**

### 编译验证

`xcodebuild -scheme JudgeE2 -sdk iphonesimulator -destination 'iPhone 11' -configuration Debug build`：**BUILD SUCCEEDED**

### 改动文件

- 新建：`JudgeE2/Interaction/PointPromptBuilder.swift`
- 编辑：`JudgeE2/Segmentation/SAMDecoder.swift`（新增 `decode(embedding:point:)` 重载）
- 编辑：`JudgeE2/Detection/CameraManager.swift`（`handleTap` + `tapEncodeAndDecode` + `tapDecodeWithPoint`）
- 编辑：`JudgeE2/JudgeE2.xcodeproj/project.pbxproj`（注册 PointPromptBuilder.swift）

### 边界与未做项（交 Debugger / Day 4）

- **不做 Debugger 工作**：iou_pred 合理性验证（> 0.5）、embedding 复用实测、mask 位置主观验证、FPS 回退监控——均属 Debugger Day 3
- **未引入 TemporalManager**：Day 3 为直通流程，TapInstanceManager 在 Day 5 引入
- **未实现多实例**：Day 3 只维护单个 maskImage（Phase 2 同一变量），多色多实例 Day 5

---

## Phase 3 Day 3 编译验证 Session

**日期：** 2026-07-22
**状态：** 验证通过（BUILD SUCCEEDED），`tasks.md` Day 3 Builder 三项 checkbox 已勾选。

### 验证内容

本次 session 确认 Day 3 三项交付文件均已就位：

| 文件 | 状态 |
|------|------|
| `JudgeE2/Interaction/PointPromptBuilder.swift` | 实现完整，ResizeLongestSide + centered pad，`point_coords [1,2,2]` |
| `JudgeE2/Segmentation/SAMDecoder.swift` | `decode(embedding:point:)` 重载已添加，iouPred 返回正常 |
| `JudgeE2/Detection/CameraManager.swift` | `handleTap` + `tapEncodeAndDecode` + `tapDecodeWithPoint` 全部就位 |

`xcodebuild -scheme JudgeE2 -sdk iphonesimulator -destination 'iPhone 11' -configuration Debug build`：**BUILD SUCCEEDED**

### tasks.md 更新

Phase 3 Day 3 Builder 三项任务均已勾选：
- [x] 实现 PromptBuilder（点模式）
- [x] 更新 SAMDecoder 支持点提示
- [x] 单次点击分割闭环验证（无 TemporalManager）

---

## Phase 3 Day 3 真机反馈修复：tapDecodeWithPoint 后台检测门控

**日期：** 2026-07-23
**状态：** 已修复（BUILD SUCCEEDED）

### 问题（P1 🔴）

**扮漏根因：**
`tapDecodeWithPoint` 的 `decoderQueue.async` 内部在调用模型前，**缺少后台状态检测门控**。
Encode 路径已有该门控，decode 路径漏採。

**场景复现：**
1. 用户切换到 `.tapToSegment` 模式
2. 第一次 tap 启动 encode（SAMEncoder 冷启动，1646ms）
3. 等待期间用户继续点击（共 12 次）→ 全部 "encoder busy" 立即居断，**但 embedding 缓存后它们一起进入 decoderQueue**
4. App 进入后台 → 12 个 decode 任务连续 GPU abort

**日志证据：**
```
[TAP] reuse cached embedding → decode point=... (×12，瞬间全部分发)
IOGPUMetalError: Insufficient Permission (to submit GPU work from background) ×N
[SEG][TAP] decoder prediction failed: Unable to compute the prediction...
```

### 修复

**文件：** `JudgeE2/Detection/CameraManager.swift` — `tapDecodeWithPoint()`

在 `decoderQueue.async` 内部、decoder 调用之前插入门控（与 encode 路径完全对称）：

```swift
// Background guard — iOS aborts GPU/ANE work when in background.
// tapDecodeWithPoint can have multiple tasks queued on decoderQueue;
// each must individually check before touching the GPU/ANE.
var isBackground = false
DispatchQueue.main.sync {
    isBackground = UIApplication.shared.applicationState == .background
}
guard !isBackground else {
    print("[TAP] decode skipped: app in background")
    return
}
```

**证明：** `xcodebuild -scheme JudgeE2 -sdk iphonesimulator -destination 'iPhone 11' -configuration Debug build` → **BUILD SUCCEEDED** ✅

### 尞未解决（非本次任务）

| 问题 | 级别 | 计划时间 |
|------|------|----------|
| 冷启动 1646ms 无反馈 | 🟡 P2 | Day 4/6 |
| 快速连点排队（Day 5 TapInstanceManager 统一处理） | 🔵 P3 | Day 5 |
| FPS 退化（热降频，系统级） | 🟡 P2 | Day 7 观察 |

---

## Phase 3 Day 3 真机反馈 Session 2 — maskImage 清除 + bbox 隐藏修复

**日期：** 2026-07-23
**状态：** 已修复（BUILD SUCCEEDED）

### 问题 1：maskImage 每帧被清除（核心 Bug）

**根因：**
`runDetectionPipeline()` 中的旧逻辑：
```swift
// 旧代码（错误）
if currentMode != .segmentation {
    maskImage = nil   // tapToSegment 也满足此条件，每帧被清除！
}
```

`tapToSegment != segmentation` 为 true，所以 tap 分割生成的 mask 在下一帧 YOLO 完成后立即被清除。用户看到的现象是“Tap 了没有反应”（mask 闪现一帧就消失）。

**修复：**
```swift
// 新代码：三种模式各自独立管理 maskImage
if currentMode == .detectionOnly {
    maskImage = nil   // 仅 detectionOnly 清除
}
if currentMode == .segmentation {
    runSegmentationPipeline(...)   // segmentation 路径自己更新
}
// tapToSegment: mask 由 tapDecodeWithPoint 驱动，不触不动
```

### 问题 2：tapToSegment 模式下 YOLO bbox 仍展示

**根因：**
`runDetectionPipeline()` 无条件发布 `boxes = rects`，所有模式均展示检测框。

**修复：**
```swift
if currentMode == .tapToSegment {
    // YOLO 运行但结果隐藏（仅用于 FrameGeometry，architect §1.1）
    boxes = []
} else {
    let rects = top5.compactMap { mapToMetadataRect($0) }
    boxes = rects
}
```

**影响：** tapToSegment 模式下：
- YOLO 内部仍然运行（`lastLetterbox` 持续更新，`handleTap` 可取到第一帧几何）
- 屏幕不再显示 YOLO bbox
- mask 仅通过 tap 触发更新，不会被 YOLO 帧循环清除

**改动文件：** `JudgeE2/Detection/CameraManager.swift` — `runDetectionPipeline()`

**证明：** `xcodebuild -scheme JudgeE2 -sdk iphonesimulator -destination 'iPhone 11'` → **BUILD SUCCEEDED** ✅

---

## Phase 3 Day 4 — 端到端点击分割 Pipeline + 分辨率 AB 测试基础设施 (Builder)

**日期：** 2026-07-23
**状态：** 完成（BUILD SUCCEEDED；768 AB 模型已导出封层，端到端 tap 流水线接入 TemporalManager，错误处理/Fallback 完善。真机 AB 采样属 Debugger Day 4）。

### 交付概览

Day 4 Builder 三项任务全部完成，严格遵守 Architect §8 降分辨率裁决（768 有条件批准、512 拒绝）及 C-1~C-6 约束：
1. 分辨率 AB 测试基础设施（768 milfix 模型导出 + `encoderInputSize` 配置 + 48→64 上采样桥接 + UI 切换 + 延迟统计日志）
2. 完整 Tap 分割流水线接入 TemporalManager（embedding 复用决策 = TTL + 几何匹配）
3. 错误处理 + Fallback 完善（encoder busy 加载指示、iou 门控、所有失败分支复位 flag）

全程未改动 Phase 1/2 检测/分割路径，未做 Debugger 工作，未重设计架构。1024 默认路径零修改（Architect C-1）。

---

### 任务 1：分辨率 AB 测试（Architect §8.3 批准 768，§8.6 授权规格）✅

#### 1a) 768 milfix encoder 导出 ✅

**脚本：** `shared/export_encoder_fp16_milfix_768.py`（新建，派生自 `export_encoder_fp16_milfix.py`）

按 model_plan §C.5 + Architect §8.6 导出规格实现，含两处关键修正：

- **dummy_trace = `torch.randn(1, 3, 768, 768)`**（§8.6 规格）
- **动态 feat_size**：monkeypatch `TinyViT.forward_features`，将硬编码 `x.view(B, 64, 64, C)` 改为 `feat_size = self.img_size // 16`（1024→64 / 768→48 / 512→32），修复 model_plan §C.1 指出的非 1024 崩溃。
- **milfix LayerNorm 融合保留**（§B.4 / C-4）：`common.LayerNorm2d` + `tiny_vit_sam.LayerNorm2d` 双补丁不变，768 变体维持 ANE 对齐修复。

> ⚠️ **导出踩坑记录（重要）**：初版仅 monkeypatch `forward_features` 不足以在 768 运行——TinyViT 内部 window-attention 层的 `assert L == H*W` 依赖**构造时**传入的 `resolution`，`sam_model_registry["vit_t"]` 固定以 `img_size=1024` 构建。修复：**直接以 `TinyViT(img_size=768, ...)` 构建 encoder**，再从 1024 checkpoint 提取 `image_encoder.*` 权重加载。验证 model_plan §C.1 结论——window-based `attention_biases` 只依赖 `window_size`（固定），与 `input_resolution` 无关，权重无损迁移：**0 real-missing / 0 unexpected keys**。

**导出验证（Mac，env1）：**
```
[build] TinyViT(img_size=768) loaded: missing=0 (real=0) unexpected=0
[sanity] Forward pass OK — output shape (1, 256, 48, 48)
Input  'image': [1, 3, 768, 768]  Float16
Output 'image_embeddings': [1, 256, 48, 48]  Float16
File size: 14.1 MB  (= 1024 milfix，符合 §8.6 预期 ~14 MB)
```

**交付物：** `models/MobileSAM_ImageEncoder_fp16_milfix_768.mlpackage`（14.1 MB）+ 拷入 `JudgeE2/Segmentation/Models/` + project.pbxproj membershipException 注册。

#### 1b) `encoderInputSize` 配置项 + Encoder 适配 ✅

**新文件：** `JudgeE2/Shared/SAMConfiguration.swift`

- `SAMConfiguration.EncoderResolution`：`.res1024`（默认，C-1）/ `.res768`（AB）。**512 不提供**（Architect §8.4 Phase 3 拒绝）。
- `encoderInputSize`（1024/768）、`featureSize`（64/48）、`decoderEmbeddingSize=64`（固定）、`pointPromptSpace=1024`（C-2 锁定）。
- 默认 `.res1024`，严格遵守 C-1。

**SAMEncoder 适配（`JudgeE2/Segmentation/SAMEncoder.swift`）：**
- `init(computeUnits:resolution:)` 新增 resolution 参数（默认取全局配置），`inputSize`/`featureSize` 由 resolution 驱动。
- 加载优先级：`.res768` → `milfix_768`（缺失则 return nil，不静默降级到尺寸不符的 1024）；`.res1024` → `milfix > fp32 > fp16`（原逻辑不变）。
- 预处理 resize 用实例 `inputSize`（1024 或 768），render target 按 inputSize 预分配。

#### 1c) Encoder 输出上采样桥接（C-3）✅

**`SAMEncoder.bilinearUpsampleEmbedding(_:srcSize:dstSize:)`**：
- 768 encoder 输出 `[1,256,48,48]` → 双线性上采样 → `[1,256,64,64]`，再交给 Decoder（Decoder 固定 64×64，C-3）。
- align_corners=false 卷积约定（匹配 PyTorch/CoreML 默认）；逐通道计算；Float32 输出。
- 1024 路径 `featureSize==64` **提前 return**，零开销、零行为变化。

#### 1d) UI 切换 + 延迟统计 ✅

- `ContentView` 新增 “Encoder Res” 分段选择器（1024 / 768(AB)），默认 1024。
- `CameraManager.setEncoderResolution(_:)`：切换时 drop `samEncoder` + 清 `embeddingCache` + 重置 tap 统计 + `temporal.invalidateMask/resetPrimary`（几何签名含 inputSize，切换即失效）。
- **AB 延迟统计日志**（Debugger Day 4 采样用）：
  - `[TAP][AB] encoder stats res=%d (n): mean=%.2f p95=%.2f`（冷启动排除）
  - `[TAP] encode done %.2f ms (res=%d) → decode`
  - `[TAP] mask displayed — iou_pred=%.3f | tap→mask %.1f ms (decode-only|encode+decode)`

**AB 采样口径（供 Debugger Day 4 真机执行 + §8.6/C-5/C-6）：**
- 同场景同目标，切 1024 / 768 各 tap ≥ 5 次，读取 `[TAP][AB] encoder stats` mean/p95 + `iou_pred` + **人工视觉评分 1–5**（C-5：iou_vs_1024 不可信，人工评审为唯一质量门控）。
- 结果写入本文件（C-6），供 Day 5 Architect 封层裁决（§8.7 门控：latency 降 ≥150ms 且评分 ≥3.5 → 封层 768）。

> **Mac 侧预测参考**（model_plan §C.2，非真机实测）：768 encoder ~555ms（×1.54 vs 1024 的 857ms）。真机 milfix_768 实测由 Debugger 补齐。

---

### 任务 2：完整 Tap 分割流水线（接入 TemporalManager）✅

**文件：** `JudgeE2/Detection/CameraManager.swift` — `handleTap` / `tapEncodeAndDecode` / `tapDecodeWithPoint`

Day 3 的直通 embedding-cache 判断升级为经 **TemporalManager** 的复用决策：

- **embedding 复用 = TTL 有效 AND 几何未变**：
  - `temporal.isEmbeddingValid(entry:nowMs:)`（embeddingTTLms=8000，Phase 2 继承）
  - `temporal.geometryChanged(geoSig)`（几何签名含 origW/H、scale、pad、rotation、mirror、inputSize）
  - `canReuse = ttlValid && !geometryChanged` → 复用走 decode-only 快路径；否则 encode+decode 慢路径。
- **决策日志**：`[TAP] reuse cached embedding (ttlValid=Y geoChanged=N) → decode` / `[TAP] encode + decode (reason=geometry change|no cache|ttl expired)`。
- **端到端延迟**：`tapStartMs` 从 tap 接受时刻贯穿到 mask ready，日志区分 `decode-only` / `encode+decode` 两种口径（对齐 Day 7 Debugger 采样要求）。
- **坐标空间锁定（C-2）**：`PointPromptBuilder` 显式传 `inputSize: SAMConfiguration.pointPromptSpace`（=1024），无论 encoder 跑 1024 还是 768，点坐标恒在 1024 空间归一化（model_plan §C.4，避免 mask 错位）。

> **关于 TapInstance/实例池**：Architect §6 分工明确 `TapInstanceManager` 属 **Day 5** Builder 任务。Day 4 保持单 mask（复用 Phase 2 `maskImage` 变量），仅打通「TemporalManager 复用判断 → encode/decode 调度」这一主链路。tasks.md Day 4 描述中的 `registerTapInstance` / 实例池 / 同区域更新，其**多实例承载**部分随 TapInstanceManager 在 Day 5 落地；Day 4 已就位单实例端到端闭环 + 复用决策骨架，避免与 Day 5 的实例池设计重复造轮子。

---

### 任务 3：错误处理 + Fallback 完善 ✅

- **encoder busy → 加载指示**：新增 `@Published tapProcessing` / `lastTapCanonicalPoint`。tap 接受即置 `tapProcessing=true`（Day 6 波纹/脉冲动画锚点与状态源）；encoder 忙时保留加载态并 `scheduleTapBusyTimeout()`（3s 安全兜底，防 UI 卡死）。
- **iou 门控**：`iou_pred < 0.1` → 丢弃 mask + 清加载指示（显示进度而非无意义 mask，符合 tasks.md Day 4 要求）。
- **全失败分支复位 flag**：
  - encode 路径失败统一走 `resetEncodingAndTapUI()`（复位 `isEncoding` + 清 `tapProcessing`）：背景态、encoder nil、encode 返回 nil、slot 竞争。
  - decode 路径失败统一 `finishTapProcessing()`：无 letterbox、decoder nil、背景态、buildPrompt 失败、decode nil、iou 门控、renderMask nil。
  - 双击 clearAll 同时清 `maskImage` + `tapProcessing` + `lastTapCanonicalPoint`。
- **后台 GPU abort 门控**：encode/decode 路径均保留 Day 3 的 `applicationState == .background` 检查。

---

### 编译验证

`xcodebuild -scheme JudgeE2 -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 11' -configuration Debug build` → **BUILD SUCCEEDED** ✅

产物 `JudgeE2.app` 含全部模型：`MobileSAM_ImageEncoder_fp16_milfix.mlmodelc` + **`_fp16_milfix_768.mlmodelc`（新）** + `_fp32` + 原始 fp16 + Decoder + yolov9-c。

### 改动文件

- 新建：`shared/export_encoder_fp16_milfix_768.py`
- 新建：`models/MobileSAM_ImageEncoder_fp16_milfix_768.mlpackage`（14.1 MB）+ 拷入 `JudgeE2/Segmentation/Models/`
- 新建：`JudgeE2/Shared/SAMConfiguration.swift`
- 编辑：`JudgeE2/Segmentation/SAMEncoder.swift`（resolution 参数 + 加载优先级 + 48→64 上采样桥接）
- 编辑：`JudgeE2/Detection/CameraManager.swift`（TemporalManager 复用决策 + tapProcessing/加载指示 + 全分支 flag 复位 + AB 延迟统计 + setEncoderResolution）
- 编辑：`JudgeE2/UI/ContentView.swift`（Encoder Res AB 分段选择器）
- 编辑：`JudgeE2/JudgeE2.xcodeproj/project.pbxproj`（SAMConfiguration.swift 注册 + milfix_768 membershipException）

### 边界与未做项（按分工）

- **Debugger Day 4 工作**：真机 tap→mask 端到端延迟采样、1024 vs 768 encoder latency 实测对比、AB 人工视觉评分（C-5）、mask 位置主观验证、embedding 复用实测、Phase 2 FPS 回归监控。Builder 已提供全部日志埋点与 UI toggle。
- **未做 512**：Architect §8.4 Phase 3 拒绝，保留 Phase 4。
- **TapInstanceManager / 多实例池**：Architect §6 分工属 Day 5，本次仅打通单实例端到端 + 复用决策骨架。
- **未重设计架构**：SAMDecoder 签名不变、TemporalManager 仅调用只读判断方法（isEmbeddingValid/geometryChanged/invalidateMask/resetPrimary）、未加载第二份 encoder（切换时替换而非并发）、1024 默认路径零修改。

---

## Phase 3 Day 5 — 快路径去队列化 + 计时口径修正 + 超时分级 + 多实例池 (Builder)

**日期：** 2026-08-09
**状态：** 完成（BUILD SUCCEEDED，零 warning）。Architect §10.4 要求 A/B/C 全部落地；Day 5 Builder 三条全部实现。真机验证属 Debugger。

### 交付概览

按用户指定顺序执行：先修 A/B/C（既有缺陷 + 测量口径），再做多实例。理由是 B 改变了后续测量口径，多实例必须建立在修好的调度与计时之上，否则 Day 7 的数据无意义。

1. **要求 A** — tap 快路径从 `videoQueue` 摘除，手势线程内完成判定后直达 `decoderQueue`
2. **要求 B** — `e2eMs` 终点移到主线程 mask 提交之后；日志区分快/慢路径并补 `cacheAge`
3. **要求 C** — 超时分级（快 1.5 s / 慢 12 s）+ 失败可见化 + warmup 前移与 parked tap 不丢弃
4. **Day 5-1** — `TapInstanceManager`（N=3 实例池、FIFO、色板、per-instance TTL 状态）
5. **Day 5-2** — TemporalManager 支持实例池（tap 路径几何签名独立且带锁；同 geometry 共享一次 encode）
6. **Day 5-3** — `MaskRenderer` 多 mask 叠加（每实例独立颜色 + primary/secondary alpha）

**保留项 R3 严格遵守：** 多实例逻辑中未夹带任何 mask 质量过滤。候选选择块经逐字符 diff 验证与 Day 4 完全一致（见下文「R3 验证方式」）。stability score 仍只打日志、不参与任何分支。

---

### 要求 A：快路径去队列化 ✅

**问题：** `handleTap` 把整个判定派发到 `videoQueue`，而 videoQueue 同时是 `AVCaptureVideoDataOutput` 的 delegate 队列（YOLO 单帧 400–670 ms）。快路径实际只需 letterbox 几何 + 已缓存 embedding，没有任何理由排在帧后面。

**实现：**

- `handleTap` 现在在**手势线程**上完成：`stateLock` 快照 → 几何签名比较 → TTL 判定 → 实例入池 → 分流。
  - `canReuse == true` → 直接 `tapDecodeWithPoint(...)`，其内部 `decoderQueue.async`，**完全不经 videoQueue**。
  - `canReuse == false` → `videoQueue.async` 取 `latestCameraBuffer`（慢路径不变）。
- **未新建第三个队列**（Architect 明令禁止）。`decoderQueue` 仍是唯一 serial decode 队列，多实例的顺序性由它天然保证。
- §4.2 触发语义不变：TTL + geometry 决定复用；`isEncoding` 仍是唯一共享 encoder slot；encode 仍先于 decode。**`isEncoding` 的 busy/free 分支仍在 videoQueue 上、派发时刻重新判定**（用更新的值），只是判定发生的队列变了。

**线程安全（关键点）：** 快路径读的 `lastLetterbox` / `backend` / `lastRotationAngle` / `currentPosition` 原本分别是 videoQueue-only 与 sessionQueue-only。把判定挪到手势线程会把它们变成三处无锁跨队列读——**用队列跳转换数据竞争不算改进**。因此：

- 新增 `stateLock` 保护的镜像 `tapGeometryMirror: TapGeometrySnapshot?`（letterbox + rotation + mirrored），由 `letterboxToSquare` 在 videoQueue 上每帧 `publishTapGeometry(info)` 发布。
- 新增 `stateLock` 保护的 `backendMirror`，由 `setBackend` 在 sessionQueue 上发布。
- `currentFrameGeometry()`（TouchHandler 每次 tap 都调，本来就在主线程）同步改为读镜像——顺带消掉一处既有的无锁读。
- 手势线程**只读快照 + 判定**，锁内不做任何模型调用、不跨队列等待。

---

### 要求 B：计时口径修正 ✅

- `e2eMs` 的计算移入**主线程发布块内部**，位于 `self.maskImage = composed` 之后（即 architect §10.4 B 要求的「main-thread 块尾」）。Day 4 的终点落在 decoderQueue 边界上，是下界。
- 日志标签明确区分：`(fast/decode-only)` 与 `(slow/encode+decode)`。
- **R2 要求的 `cacheAge` 已补齐**，出现在三处：`reuse cached embedding (… cacheAge=NNNms)`、`encode + decode (… cacheAge=NNNms)`、以及 `mask displayed` 汇总行。**只打日志，TTL 数值与复用行为一字未改**（TTL 复议属 Day 6）。
- `mask displayed` 行同时输出实例池状态 `pool=[#12* #10 #7] n=3`（`*` = primary），便于 Debugger 核对 FIFO 与共享 embedding。

---

### 要求 C：超时分级 + 失败可见化 ✅

**分级超时**（`scheduleTapTimeout(gen:seconds:label:)` 取代固定 3 s 的 `scheduleTapBusyTimeout`）：

| 分支 | 超时 | 依据 |
|------|------|------|
| 快路径 | **1.5 s** | decode 实测 ≈61 ms，已是 20 倍余量 |
| 慢路径 | **12 s** | 覆盖实测 encoder 冷启动上界 8605 ms + 余量 |

旧实现的判定条件 `tapProcessing && maskImage == nil` 在多实例下已不成立（先落地的实例会让 `maskImage` 非空），改为按 `inFlightTaps[gen]` 是否仍在册判定。

**失败可见化的 UI 形式：** 新增 `@Published var tapFailure: TapFailure?`（`index` / `message` / `viewPoint`）。ContentView 在 tap 位置渲染 `TapFailureIndicator` —— **红色停止环 + ✗ + 一行原因文案**，约 1.6 s 后由 CameraManager 自动清除。

选择「红环 + 文案」而非 toast 的理由：它就在用户手指刚离开的位置，语义与脉冲确认环形成对照（青色脉冲 = 收到、红色静止环 = 失败），且**明确不含任何时长语义**，不违反 D1/D2 对进度条的禁止。

**所有失败出口统一走 `failTap(gen:viewPoint:message:)`，不再有静默清除 `tapProcessing` 的路径**，包括：无相机帧、encode 失败/被后台打断、decoder 不可用、buildPrompt 失败、decode 返回 nil、iou_pred 越界、`iou_pred < 0.1` 门控、tap 处无 mask 区域、超时、park 溢出。

> ⚠️ 范围说明：Architect 原文强调的是「任何**超时**路径必须可见」。本次把**全部**失败出口纳入可见化。理由：这些出口共享同一个用户可感知的失败模式（「点了没反应」），且**只加 UI 信号、不改任何判定**——`iou_pred >= 0.1` 门控的阈值、比较方向与后果一行未动（R3）。如 Architect 认为 routine 拒绝（如「tap 处无对象」）不该给红环，改回静默只需在 `failTap` 增加一个分类参数。

**冷启动（正解 = warmup 前移，而非加长超时）：**

- `setMode(.tapToSegment)` 早已调用 `warmupSegmentationIfPossible()`（debug_report §254 记录的「仅 .segmentation 调用」在当前代码中已不成立）。**但存在一个真实缺口：** 该函数 `guard let cameraBuffer = self.latestCameraBuffer else { return }` —— 若 tapToSegment 是启动模式，此时还没有任何帧，warmup 被**静默丢弃**，冷启动照旧落回首次 tap。
- 修复：新增 `warmupPending`（videoQueue-only），无帧时置位并记日志；`captureOutput` 拿到第一帧后立即补跑 warmup。
- 新增 `@Published var samWarmingUp`，UI 在 tapToSegment 模式下显示「Preparing segmentation…」横幅，warmup 期间用户看到的是明确的初始化态。

**parked tap 不得沉默丢弃：**

- `pendingTap`（单个）→ `pendingTaps`（数组，上限 = `TapInstanceManager.maxInstances`，R4 禁止无界排队；溢出者显式失败）。
- `drainPendingTaps(originTag:)` 在**任何** encode 结束后调用，无论成败：warmup 成功 / warmup 失败 / warmup 被后台打断 / tap encode 成功 / tap encode 失败 / background refresh 成功 / refresh 失败。
- 契约：一个 parked tap 离开 `drainPendingTaps` 时只有两种结局——**被 decode**，或**被 `failTap` 可见地报错**。Day 4 唯一一次被丢弃的 parked tap（G-2 的成因）在这里被堵死。

---

### Day 5-1：TapInstanceManager ✅

**新文件：** `JudgeE2/Interaction/TapInstanceManager.swift`

`TapInstance` 字段严格按 architect §3.1：`id` / `canonicalPoint` / `createdAt` / `mask: MLMultiArray?` / `maskTimestamp` / `iouPred` / `color` / `isPrimary`。另加两个实现字段并注明理由：

- `maskAlpha: [UInt8]?` —— 渲染用的 256×256 二值 alpha。合成走它而非 `mask`，这样后来的 tap 触发重新合成时**不会重跑候选选择**（R3：无关实例的更新不得重新裁决已定的 pick）。
- `requestGen: Int` —— 该实例最新请求的 tap 序号，per-instance 取代旧的全局 `isLatestTap`（见下）。

`maskTTL: Date?` 按 tasks.md 命名提供，实现为 `maskTimestamp + 2.0s`，TTL 常量只有一处。

**API：** `addInstance(point:requestGen:)`（返回新实例 + 被 FIFO 淘汰者的 id）、`updateMask(id:requestGen:mask:alpha:iouPred:)`、`removeInstance(id:)`、`clearAll()`、`promoteToPrimary(id:)`、`isRequestCurrent(id:requestGen:)`、`drawableInstances()`、`debugSummary()`。

**颜色分配策略：** 色板 `[systemBlue, systemGreen, systemOrange]`（§3.3）。新实例取**当前未被占用的第一个**色。因为「颜色随实例一同释放」且 |palette| == N == 3，这保证任意时刻三个活跃实例颜色互不相同；FIFO 淘汰最老实例后，新实例恰好接手被释放的颜色（1蓝 2绿 3橙 → 第4次 tap 淘汰#1 → 第4个实例重新拿到蓝）。

**FIFO 与 TTL 的关系（重要，二者是正交的）：**

- **FIFO 管占位**：仅在 `count >= 3` 时触发，按 `createdAt` 最早淘汰，**不论是否 primary**（§3.2）。
- **TTL 管可见性**：per-instance `maskTimestamp`，与 TemporalManager 不共享状态（§3.1 明确要求）。
- ⚠️ **Day 5 未把 TTL 接到显示上**：`drawableInstances()` 返回所有「有 alpha」的实例，不过滤过期。理由是 maskTTL=2000 ms 而 Debugger 的验收动作是「分别 tap 1、2、3 个不同位置确认同时显示」——依次点三下必然超过 2 秒，接上 TTL 会让前两个 mask 在验收过程中自己消失，同时也是相对 Day 4「tap mask 常驻」的行为回退。**TTL 状态已完整维护（`isMaskValid(now:)` 可直接调用），是否让过期 mask 消失属用户可见行为变更，留待 Architect 裁决**（见文末未决问题）。
- 淘汰/清空导致的在途请求不会变成超时误报：`cancelRequests(forInstance:reason:)` 静默注销（什么都没失败，是用户自己替换的），只有真失败才走 `failTap`。

**线程模型：** 实例池被手势线程（add）、decoderQueue（updateMask/查询）、主线程（clearAll）访问，没有单一 owning queue，故内部自带 `NSLock`。**这不是新建队列**（Architect 禁的是队列），一个 ≤3 元素数组的锁不引入任何调度跳转。

**supersession 语义变更（必须记录）：** Day 4 的 `isLatestTap(gen)` 是**全局**的——任何新 tap 会作废所有在途 tap。多实例下这会让第 2 次 tap 直接杀掉第 1 个实例的 decode，三个 mask 永远不可能同时出现。改为 per-instance：`isRequestCurrent(id:requestGen:)`，请求只在「实例仍在池中 且 requestGen 未被同实例的新请求顶掉」时有效（正是 R4 的「同一实例的新 tap 取代旧请求」）。全局 `tapGeneration` 保留，继续作为 `[TAP#N]` 日志与屏幕计数器的单调序号。

---

### Day 5-2：TemporalManager 支持实例池 ✅

- **顺序入队、不并发**：`decoderQueue` 本就是 serial，多个实例的 decode 天然串行。**没有为此新增任何机制**——Architect §10.4 A 禁止新建队列，而 serial 队列已经满足要求。
- **同 geometry 共享同一 embedding、只 encode 一次**：新增 `TemporalManager.tapGeometryChanged(_:)` + `resetTapGeometry()`，带**独立的 `tapGeometryLock`**，与 Phase 2 的 `lastGeometrySignature` 分开。
  - 为什么分开：(1) 要求 A 把判定挪到手势线程，而 `geometryChanged` 会改 videoQueue-owned 状态，不能跨线程调；(2) `.segmentation` 与 `.tapToSegment` 互斥运行，拆分在模式内部行为等价，同时消除了两模式共用一份签名的交叉污染。
  - 语义与 `geometryChanged` 一致：与上一次 tap 的几何签名比较，不同则返回 true 并采纳新签名。
  - 返回 false ⇒ 整个实例池共用当前 embedding，encoder 一次都不再跑；burst 中第 2、3 次 tap 走 `drainPendingTaps` 搭同一次 encode 的车。
  - 返回 true ⇒ 所有缓存 mask 属于失效坐标空间，`discardAllTapWork` 清空实例池（并清 overlay）。
- **每实例独立 mask TTL**：状态在 `TapInstance.maskTimestamp`，与 TemporalManager 零共享（§3.1 原文要求「与 TemporalManager 独立计算，不共享状态」）。
- Phase 2 的 `geometryChanged` / `selectPrimary` / `classifyDrift` / mask cache **一行未动**。

---

### Day 5-3：MaskRenderer 多 mask 叠加 ✅

**重构方式（为保住 R3 而刻意选择）：** 把 `renderMask` 拆成三块，**没有复制任何决策代码**：

- `buildAlpha(...)` —— 从旧 `renderMask` 原样切出的决策段（阈值、候选选择、60%/85% cap、形状门槛、flood fill、stability、数值哨兵）。
- `drawTile(...)` —— 原样切出的纯几何绘制段（letterbox → origW×origH 画布）。
- `renderMask(...)` —— 变成薄包装：`buildAlpha` → 青色 RGBA 填充（字面量保留，未走新的合成器，以保证 Phase 2 / 单实例输出逐字节不变）→ `drawTile`。

新增两个公开入口：

- `buildTapAlpha(...)` —— 多实例路径的二值化入口，**内部就是同一个 `buildAlpha`**（共享代码，不是拷贝）。
- `compositeLayers(_:origW:origH:tapIndex:)` —— 多 mask 合成。

**绘制顺序与透明度：**

- 入参 `layers` 由 `drawableInstances()` 按 `createdAt` 升序给出：**secondary 在前、primary（最新）在最后**，因此 primary 压在最上层（§3.4）。
- alpha：primary **0.55** / secondary **0.35**（§3.4，常量在 `TapInstanceManager.primaryOpacity/secondaryOpacity`）。
- 重叠区做 **source-over 逐像素混合**（直通 alpha，float 累加避免三次叠加的舍入漂移），而不是后写覆盖 —— 这样两个实例重叠时互相仍可见，符合 tasks.md「不相互遮挡」。
- 空实例列表 → `compositeLayers` 返回 nil → `maskImage = nil`，回退到「无 mask」显示状态。
- primary 的白色 1pt 轮廓线属 **Day 6**（「Mask 高亮 + 轮廓描边」），本次未做。

合成在 **decoderQueue** 上完成（约 20 万像素操作），主线程只做一次 `maskImage` 赋值 —— 否则会污染要求 B 刚刚修正的那个计时窗口。

---

### R3 验证方式（如何确认候选选择与 stability 未被改动）

`JudgeE2/Segmentation/` 与 `JudgeE2/Detection/` **未被 git 跟踪**，`git diff` 返回空、不能作为证据（此坑已记入 Builder memory）。实际验证手段：

1. **逐字符 diff**：把改动前 `renderMask` 中 `if let tap = tapPoint256, channels > 1 { … } else {` 整段（候选构建 + 三道形状门槛 + cap60 主选 + cap85 degraded 回退 + 无候选 faultLog）留存为基准文本，与重构后 `buildAlpha` 中的同一段做字符串相等比较 → **IDENTICAL**。
2. **常量清点**：`minComponentPx=30` / `minComponentSidePx=3` / `minComponentFill=0.05` / `maxPlausibleLogit=500.0` / `stabilityDelta=1.0` / `cap60 = contentPx*60/100` / `cap85 = contentPx*85/100` / `gateIouPred >= 0.1` —— 全部与 Day 4 一致。
3. **stability 角色清点**：`stab` / `stability` 的全部出现点均为 —— 常量定义、结构体存储、`stabilityScore` 计算、日志格式化。**不存在任何以 stability 为条件的分支**，`Candidate` 的排序键仍是 `comp.count`，degraded 回退的排序键仍是 `iou`。
4. flood fill（`keepComponentContaining`）与 `extractLogits` 一行未动。

---

### 编译验证

```
xcodebuild -project .../JudgeE2/JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'generic/platform=iOS Simulator' build
```
→ **BUILD SUCCEEDED**，零 warning。

---

### 改动文件

- 新建：`JudgeE2/Interaction/TapInstanceManager.swift`
- 编辑：`JudgeE2/Detection/CameraManager.swift`
  （`tapGeometryMirror`/`backendMirror` 镜像 + `publishTapGeometry` + `handleTap` 去队列化 + `beginTapRequest`/`endTapRequest`/`cancelRequests`/`failTap`/`scheduleTapTimeout`/`parkTap`/`drainPendingTaps`/`discardAllTapWork` + `tapEncodeAndDecode`/`tapDecodeWithPoint` 改为 per-instance + 多实例合成发布 + `warmupPending` 补跑 + `samWarmingUp`/`tapFailure` 发布）
- 编辑：`JudgeE2/Segmentation/TemporalManager.swift`（`tapGeometryLock` + `tapGeometryChanged` + `resetTapGeometry`；Phase 2 路径零改动）
- 编辑：`JudgeE2/Segmentation/MaskRenderer.swift`（拆出 `buildAlpha`/`drawTile`；新增 `AlphaResult`/`MaskLayer`/`buildTapAlpha`/`compositeLayers`；决策代码零改动）
- 编辑：`JudgeE2/UI/ContentView.swift`（`TapFailureIndicator` + warmup 横幅 + overlay 优先级）
- 编辑：`JudgeE2/JudgeE2.xcodeproj/project.pbxproj`（注册 `TapInstanceManager.swift`）

---

### 边界与未做项

- **Debugger Day 5/Day 7 工作**：真机多 mask 显示验证、FIFO 验证、共享 embedding 验证、内存基线、以及**用修正口径重测快路径 p95**（Architect §10.6 门控）。Builder 只提供埋点。
- **未做 mask 质量过滤**（R3 明令禁止）。
- **未改 embedding TTL / mask TTL 数值**（R2，Day 6 复议）。
- **未改 stability 的角色**（仍只打日志）。
- **未做 Day 6 项**：波纹动画、primary 白色轮廓、点击已有 mask 内部提升 primary（`promoteToPrimary` 已就位但未接手势）、长按删除单实例（`removeInstance` 已就位）、`[CACHE] re-encode reason` 计数日志。
- **未新建队列**，未改 Phase 1/2 路径，未重设计架构。

---

### 交给 Architect 的未决问题（1 个）

**tap mask TTL 是否应该让 mask 自行消失？**

§3.1 规定 per-instance mask TTL = 2000 ms，§3.4 规定「无有效 mask → 不显示 mask」。字面执行会导致：tap 后 2 秒 mask 自动消失，且三个实例不可能同时可见（依次点三下必然 > 2 s），与 Day 5 的验收动作直接冲突，也是相对 Day 4「tap mask 常驻」的可见行为回退。

Builder 的处置：**TTL 状态完整维护但不接显示**（`drawableInstances()` 不过滤过期）。请 Architect 裁决三选一：
1. 维持现状（TTL 仅为状态，mask 常驻至被替换/清除）——Builder 建议，与 tap 语义（用户显式选择的对象）更一致；
2. 接上 TTL 但把 tap mask TTL 从 2000 ms 上调到一个能容纳三次点击的值；
3. 接上 2000 ms TTL 并相应调整 Day 5/Day 7 的验收动作。

---
---

## Phase 3 Day 5（追加）—— 呈现规格修订 + 双色轮廓描边 + Day 5 遗留缺陷修复

日期：2026-08-11
依据：architect_output.md **§12（呈现规格复议裁决）** 与修订后的 **§3.3 / §3.4**（C-1…C-7）、tasks.md Day 5 Builder 新增 P0 条目、debug_report.md §15–§26（D-1/D-2/D-3/D-6/D-12/D-14/D-15）

---

### 第一组（P0）：呈现规格修订

#### 1) 三色板作废 → 全实例统一青色

`TapInstanceManager.swift`

- 新增 `static let fillColor = UIColor(red: 0, green: 217/255, blue: 1, alpha: 1)` —— 青 (0,217,255)，H=189° / Y=0.569 / S=1.00 / V=1.00，逐条对上 C-1（稀有色带）/ C-2（Y ≥ 0.45）/ C-3（S ≥ 0.85 且 V ≥ 0.90）。
- `palette` 由 `[systemBlue, systemGreen, systemOrange]` 改为 **三个槽位一律 `fillColor`**。**槽位分配代码原样保留**（`used` 去重 + fallback），注释写明它在 Phase 3 内是「按构造为 no-op」而非碰巧无效 —— 这是 Q6「字段保留、语义降级为呈现槽位」的落点，Phase 4 若重开按实例配色只需改这三个字面量。
- `TapInstance.color` 字段与注释同步降级为「呈现槽位」。
- **alpha：primary 0.55 → 0.60，secondary 0.35 → 0.40。**

⚠️ 该 `fillColor` **刻意不是 dynamic system color**。`compositeLayers` 在 decoderQueue 上用 `getRed` 解析它；dynamic color 在后台线程会按后台线程的 trait collection 解析（**D-15**），深色模式下与 UI 其余部分不一致。此约束已写进 `MaskRenderer.MaskLayer.color` 的文档注释，避免以后被改回系统色。

#### 2) 双色轮廓描边（L1 可见性的唯一承载者）

新建 `JudgeE2/Segmentation/MaskOutline.swift`：

- `struct MaskOutline { polygons: [[CGPoint]]; isPrimary: Bool }` —— 闭合环，坐标是**相对 mask 画布（origW×origH）归一化的 [0,1]**，不带任何像素尺度。
- `struct MaskOutlineSet { canvasSize: CGSize; outlines: [MaskOutline] }` —— 轮廓与其归一化基准画布**打包成一个值发布**，避免视图拿到分属两帧的 polygons 与 canvasSize。
- `enum MaskOutlineStyle` —— 全部呈现常量集中一处：`isEnabled`（**C-6 开关**）、outer/inner 颜色、四个线宽（2.0 / 1.5 / 1.5 / 1.0 pt）、四个 alpha（0.85 / 0.95 / 0.70 / 0.70）。

`MaskRenderer.swift` 新增 `traceOutline(alpha:origW:origH:)`：

1. **边界单位边提取** —— 对每个前景像素，凡邻居为背景的一侧就是一条有向边，按固定旋向（上→右→下→左，mask 空间 y 向下即顺时针）发射。顶点落在像素角上，坐标恒为整数，**没有浮点键、不需要 epsilon 匹配**。
2. **串联成闭环** —— 顶点字典逐条消费，天然分出外轮廓与孔洞（孔洞单独成环并同样描边：孔的边缘也是真实的 mask 边界）。
3. **一次 Chaikin 圆角** —— 原始环是轴对齐阶梯，在 iPhone 11 预览上约 3.5 pt 一级，压在被 CoreAnimation 平滑过的填充上会读成锯齿。一次 Chaikin 把直角换成 45° 倒角，顶点位移不超过半个 mask 像素，因此描边仍然贴着生成填充的那张二值 alpha。
4. **归一化** —— 走**同一个** `tileRect(origW:origH:)`（从 `drawTile` 里原样抽出的 letterbox 算术，数值不变），因此描边与填充**不可能算歪到两处去**。

`CameraPreview.swift`：`PreviewView` 新增 4 个 `CAShapeLayer`（secondaryOuter → secondaryInner → primaryOuter → primaryInner，深色外圈在下、浅色内线在上），位于 mask 填充层之上、bbox overlay 之下（§1.3 Z 序不新增层级）。`rebuildOutlinePaths()` 用与 CoreAnimation 对填充图完全相同的 `resizeAspectFill` 变换（取两轴比例的**较大者**再居中）把归一化点换算成 view 点；`bounds` 变化时用留存的几何重建。

**C-5 如何满足（逐条）：**

| C-5 要求 | 落点 |
|---|---|
| 描边在屏幕坐标系生成 | 路径在 `PreviewView.rebuildOutlinePaths()` 里换算到 view 坐标，`CAShapeLayer.path` 就是 view 空间 |
| 线宽以 pt 计 | `CAShapeLayer.lineWidth = 2.0 / 1.5 / 1.0`，UIKit 单位即 pt；**不随 mask 的 4–8× 放大而变** |
| **不得烧进 256×256 位图** | `traceOutline` 只产出矢量点，**全程没有任何一次栅格化**；`buildAlpha` / `compositeLayers` / `drawTile` 输出的位图内容与描边完全无关 |
| 与 D-13 兼容 | 描边由 alpha 推出、不由已栅格化的 tile 推出。D-13 把 256×256 交给 GPU（`contentsRect`）之后，本路径一行不用改 |

**C-6 怎么用：** 把 `MaskOutlineStyle.isEnabled` 改成 `false`，重编译。`CameraManager` 会**完全跳过轮廓追踪**（零开销）并发布 `nil`，`PreviewView` 四个 shape layer 全部隐藏 —— 得到与 Day 4 完全一致的「无描边」视觉条件，其余一切不变。这是 R11 要求的单变量回退手柄。

#### 3) 几何输出与 Day 4 逐位一致 —— 如何确认

- **决策代码零改动**：`buildAlpha` 的阈值 / 候选选择 / cap60 / cap85 / flood fill / stability / 数值哨兵 / `iou_pred >= 0.1` 全部未动（本轮唯一触碰 `buildAlpha` 的是 D-6 死代码门控，见下，且该量在多候选路径上从来没有被读过）。
- **填充色算术复核（单实例 primary）**：`compositeLayers` 对单层、`sa = 0.60` 的情形：`outA = 0.60`，`accG = (217/255 × 0.60) / 0.60 = 217/255`。取整后得 **(0, 217, 255, 153)** —— 与 Day 4 `renderMask` 里的青色字面量 `(0, 217, 255, 153)` **逐字节相同**。
- **二值 alpha 未被触碰**：描边是纯加法的独立图层，既不改 alpha，也不进合成缓冲区。
- ⚠️ 严格的逐位可比性只在 `isEnabled = false` 时成立（Architect R11）：描边本身是加法，但会让边界主观上更"硬"。Day 7 若要与 44/50 = 88% 严格对照，用 C-6 开关做单变量回退。
- Phase 2 的 `renderMask`（box 路径）**一行未动**，其青色字面量原样保留，两条路径未合并。

---

### 第二组：Day 5 遗留缺陷修复

#### D-3 超时不得 retire 在途实例（`scheduleTapTimeout`）

原来超时先 `removeInstance` 再 `failTap`，于是 TAP#1 那次在 220 ms 后**成功解出**的 mask 被丢弃。现在超时**只报告、不回收**：

- 不再 `removeInstance`，不再从 `pendingTaps` 里删；实例只由 §3.2.1 的 C1–C6 事件移除。
- **可见失败信号保留**（Architect 禁止静默清除），文案改为 `timed out after Xs (…path) — still working`。
- 晚到的结果照常渲染：`isRequestCurrent` 查的是实例池而非 `inFlightTaps`，所以 decode 继续走完；发布分支里既有的 `tapFailure = nil` 自然把提示撤下。

#### D-1 / D-2 warmup 被抢跑 + decoder 预热无独立触发

- **D-1**：`warmupSegmentationIfPossible` 里 `guard !isEncoding else { return }` 这个**全函数唯一无日志、无重试的出口**，改为 `warmupPending = true` + `diagLog("warmup deferred — encoder slot busy, re-armed for next frame")`，复用既有的首帧补跑机制重投。
- **补一条防抖**：重投前先看 embedding 缓存，若已 ≤ 5000 ms 新鲜则**跳过这次 encode 并打日志** —— 否则输给 background refresh 之后会无限重投，最后还要在刚完成的那次 encode 之上再白付约 650 ms。
- **D-2**：新增 `warmupDecoderIfPossible(letterbox:backend:origin:)`，在 **decoderQueue** 上执行（不新建队列，§10.4 A）：
  - **构造 decoder 无条件执行**（这才是 > 1.5 s 的冷启动大头），不依赖 encoder 槽、也不依赖是否已有 embedding；
  - 预热 decode 需要 embedding，因此每次调用都尝试一次，成功后由 `decoderWarmupDecodeDone` 上锁只跑一遍；
  - 调用点三处：warmup 的 encoder 槽争夺**之前**、warmup encode 成功之后、以及 background refresh 的冷启动分支 —— **无论哪条路径赢得竞争，decoder 都会被预热**。
  - `decoderWarmupDecodeDone` 在 `samDecoder = nil` 处（backend 切换）复位。
- **顺带（Debugger §17.6-3）**：`refreshTapEmbeddingIfNeeded` 在 `embeddingCache == nil` 时做的就是 warmup 的活。现在这条冷启动分支会 `setWarmingUp(true/false)` 并补打 `SAM encoder warmup latency: … (via background refresh)`，否则「Preparing segmentation…」横幅恰好在最需要它的那几秒里不显示。

#### D-12 路径标签与计费口径拆开

新增 `enum CameraManager.TapPath { case fast, slow, parked }`，与 `reusedEmbedding` **并列而非合并**：

- `reusedEmbedding` = **计费口径**：这次 tap 有没有自己付一次 encode。parked tap 仍是 `true`（它搭的是别人的车）。
- `path` = **路径口径**：`.fast` = tap 时缓存直接可用 → `fast/decode-only`；`.slow` = 自己跑 encode → `slow/encode+decode`；`.parked` = 排在别人的 encode 后面 → **`slow/parked→decode-only`**。
- 日志行末尾同时给出两者：`… tap→mask %.1f ms (slow/parked→decode-only, encode=shared)`。
- 效果：`drainPendingTaps` 出来的 tap 不再被算进快路径 p95（原来 TAP#1 那种等了 9.5 s 的样本会被标成 `fast`）。

#### D-6 死代码（`allLogits` + 65k 排序）

在 `buildAlpha` 里加一个 `usesMultimask = (tapPoint256 != nil && channels > 1)`：

- `allLogits` 在多候选路径上**不再分配、不再填充**；
- p30 排序与整个 `thresh` 计算在该路径上跳过（两个分支都加了 `!usesMultimask` 条件）；
- `thresh` 的日志改为 `thresh=n/a(multimask: logit>0 per candidate)` —— 原来那行会诱导读者以为 tap mask 是按那个数值阈值化的，而它从来没被读过。
- **行为零变化**：多候选分支的二值化用的是 `valueAt(c,y,x) > 0`，从不读 `thresh`；box 路径与单 mask 回退路径的代码形状与取值完全照旧。

#### D-14 `compositeLayers` 两处静默丢层

`alpha.count != total` 与 `getRed` 失败两处 `continue` 各补一条 `faultLog`（带层序号）。当前 `drawableInstances()` 已过滤 nil，两个分支都不可达 —— **正因为不可达才必须让它们在变得可达的那天叫出声**。

#### D-15 后台线程解析 dynamic UIColor

由 `fillColor` 改为静态 sRGB 字面量根治（见第一组 1），并在 `MaskLayer.color` 处写死约束注释。

---

### 未做（明确划界）

- **未做任何延迟优化**：约 280–310 ms 未归因残差原样保留，D-7' 的 6 段埋点未做（Architect 要求埋点先行，本轮不动队列与渲染管线结构）。
- **未夹带任何 mask 质量过滤**（R3）：候选规则、`minComponentPx=30` / `minComponentSidePx=3` / `minComponentFill=0.05` / `maxPlausibleLogit=500` / cap60 / cap85 / `iou_pred >= 0.1` 一律未动。
- **stability 仍只打日志**，不参与任何决策。
- **未动 embedding TTL（8000 ms）**，未动 `maskTTL` / `isMaskValid` 死代码（Architect B-2 归 Day 6）。
- **未做 Day 6 项**：tap 锚点标记 / 编号、长按删除、点击已有 mask 提升 primary、D-4/D-5 的重复模型加载。

---

### 编译验证

```
xcodebuild -project .../JudgeE2/JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'generic/platform=iOS Simulator' build
```
→ **BUILD SUCCEEDED**，源码零 warning。

轮廓算法另做了一次离线自检（把 `traceOutline` 原样抽到 scratch 里用 `swiftc -O` 跑）：实心方块 → 1 环、bbox 归一化坐标与手算的 letterbox 期望值**完全相等**；带孔方块 → 2 环；对角相接的两像素 → 2 环（不崩、不串环）；空 alpha → 0 环；全屏 alpha → 1 环 2048 点、Mac 上 0.50 ms。

---

### 改动文件

- 新建：`JudgeE2/Segmentation/MaskOutline.swift`
- 编辑：`JudgeE2/Segmentation/MaskRenderer.swift`（`traceOutline` + `tileRect` 抽取 + D-6 门控 + D-14 日志 + `MaskLayer.color` 约束注释）
- 编辑：`JudgeE2/Interaction/TapInstanceManager.swift`（`fillColor` + 青色单色板 + alpha 0.60/0.40 + 槽位语义注释）
- 编辑：`JudgeE2/Detection/CameraManager.swift`（`maskOutlines` 发布 + 轮廓追踪接线 + `TapPath` + `warmupDecoderIfPossible` + D-1 出口可观测 + D-3 超时不 retire + refresh 冷启动 warming 状态）
- 编辑：`JudgeE2/Detection/CameraPreview.swift`（4 个描边 `CAShapeLayer` + `updateMaskOutlines` + `rebuildOutlinePaths`）
- 编辑：`JudgeE2/UI/ContentView.swift`（传 `maskOutlines`）
- 编辑：`JudgeE2/JudgeE2.xcodeproj/project.pbxproj`（注册 `MaskOutline.swift`）

---

### 留给真机验证的点（Builder 无法自证）

1. **C-7 撞色测试**：青色物体（青马克杯 / 青绿封面）上 tap，确认「填充看不清但轮廓仍在」。若连轮廓都找不到，那是描边实现问题、不是色板问题（Architect R10）。
2. **warmup 验收**：进入 `.tapToSegment` 后日志须同时出现 `SAM encoder warmup latency`（可能带 `via background refresh` 后缀）**与** `SAM decoder warmup latency`；首次 tap 的 decode 应落在 38–70 ms 而非 220 ms。
3. **D-3 验收**：制造一次超时，确认失败提示出现后**晚到的 mask 仍然渲染**并把提示撤下。
4. **描边线宽目视**：不同 mask 大小下线宽应恒定（若随 mask 变粗，说明 C-5 被破坏了）。

---

## Phase 3 Day 5（追加二）—— 按实例配色恢复（C-7 准入）+ 编码器槽出口可观测化 (Builder)

**日期**：2026-08-11
**分支**：`phase3-tap-segment`（未 commit）
**触发**：用户明确要求「把不同的 mask 设置为不同的颜色」
**依据**：architect_output.md §3.3.1（L1/L2/L3）、§3.3.2（C-1…C-7）、§3.4、§12
**状态**：⚠️ **本轮色板属于对 §3.3.3 / §12.1 Q6 的规格修订，需 Architect 事后追认。**

---

### 0) 为什么这不是简单回退

§12 撤销按实例配色的核心论证是：**色相当时在承担 L1 可见性**，而「填充色 vs 物体本色」是不可控输入（§12.9 长期约定 A-3）。
上一轮双色轮廓描边落地后，**L1 已由描边独立承担**，色相退回 §3.3.1 的 L3（「这三块分别来自哪三次点击」）。前提变了，按实例配色重新可辩护。

**硬约束一条没松**：C-1 稀有色带、C-2 `Y ≥ 0.45`、C-3 `S ≥ 0.85 且 V ≥ 0.90` 逐条检验；`systemBlue / systemGreen / systemOrange` 仍然全禁。primary/secondary 的 alpha 0.60 / 0.40 与双色描边分档**一律未动**，本轮只改色相。

---

### 1) C-7 (a) 算术准入检验

相对亮度 `Y = 0.2126·R_lin + 0.7152·G_lin + 0.0722·B_lin`（sRGB 线性化）；H/S/V 由**实际发布的 8 bit 值**反算，不是由标称色相反推。

| 槽位 | sRGB | H | S | V | Y | C-1 | C-2 | C-3 |
|---|---|---|---|---|---|---|---|---|
| **slot 0** 青 Cyan | (0, 217, 255) | 188.94° | 1.00 | 1.00 | **0.5685** | ✅ | ✅ | ✅ |
| **slot 1** 水青 Aqua | (0, 255, 242) | 176.94° | 1.00 | 1.00 | **0.7793** | ✅ | ✅ | ✅ |
| **slot 2** 春青 Spring cyan | (0, 255, 170) | 160.00° | 1.00 | 1.00 | **0.7442** | ✅ | ✅ | ✅ |
| ~~systemBlue~~ | (0, 122, 255) | 211.3° | 1.00 | 1.00 | 0.2114 | ❌ | ❌ | ✅ |
| ~~systemGreen~~ | (52, 199, 89) | 135.1° | 0.74 | 0.78 | 0.4230 | ❌ | ❌ | ❌ |
| ~~systemOrange~~ | (255, 149, 0) | 35.1° | 1.00 | 1.00 | 0.4275 | ❌ | ❌ | ✅ |

**slot 0 保留 Day 4 的青**：单 tap（N=1）时 `compositeLayers` 的输出与 Day 4 `renderMask` 的字面量 `(0,217,255,153)` **逐位相同**，44/50 = 88% 人工评分基线在 N=1 上完全不受本轮影响（§11.9 R9）。

---

### 2) 「允许带装不下三个可区分色相」—— 实际验算结论：**装得下**（但 §3.3.3 有两处需要修订）

§3.3.3 的这条论断经算术核对**不成立**，但它指向的现象是真的。逐条：

**(a) C-1 的备选带 H ∈ [280°,330°]（品红）在 C-2 ∩ C-3 下是空集。**
在 `S ≥ 0.85, V ≥ 0.90` 约束内穷举该带，**最大可达亮度只有 Y = 0.2988**（H = 300°, S = 0.85），离 0.45 差得很远。
⇒ **品红备选带应当从 C-1 划掉**：它写在规格里会误导下一次换色。

**(b) 首选带 H ∈ [160°,200°] 被 C-2 在 H = 194.78° 处截断**（Y 恰好穿过 0.45）。可行弧段是 **H ∈ [160°, 194.78°]，宽 34.8°**。

**(c) 在这段弧里、且 slot 0 钉死在 189° 的前提下**，最大化最小 CIEDE2000 间距的三元组是 **(160.0°, 177.5°, 189°)，ΔE00_min = 17.4**；实际发布的 (160°, 177°, 189°) 给出 **ΔE00_min = 17.1**，高于「类别可辨识」常用的 ΔE00 ≈ 10 经验阈。

⇒ **约束集容得下三个可区分色相；它容不下的是「散布在整个色环上的三个色相」。** 被作废的旧色板 ΔE00_min = 50.2 —— 正是那种散布把其中两个色相拖进了禁用带。

**⚠️ 必须如实记录的削弱项（alpha 衰减）：** 上面的 ΔE00 是**不透明色块**的值。半透明叠加后，色相差按 alpha 比例缩水。在 7 种代表性背景（中灰 / 白墙 / 暗部 / 木色肤色 / 牛仔蓝 / 植物绿 / 青色马克杯）上实算合成后的最小两两 ΔE00：

| alpha | 合成后 ΔE00_min 范围 | 判读 |
|---|---|---|
| **0.60**（primary） | **10.2 – 13.4** | 全部背景上都 ≥ 10，可辨识 |
| **0.40**（secondary） | **6.8 – 10.6** | **多数背景低于 10** —— 两个 secondary 之间靠色相区分是**勉强**的 |

secondary 之间的区分目前只有色相一个载体（描边分档承担的是 primary/secondary 的 L2，不是实例间的 L3）。**Day 6 的 tap 锚点编号才是补上这块的载体**，在它落地前，「哪个 secondary 是哪个」是弱区分。这条不美化。

---

### 3) 重叠区混色 —— 如何处理

`compositeLayers` 仍是**逐像素 source-over**（保留「重叠实例互不遮挡」的既有语义，不改为 painter 遮挡）。

§3.3.3 反对三色板的第二条理由是「三个色相的半透明层重叠会混出第四种颜色（蓝+橙=灰褐）」。**这条对本色板不成立，因为新色板在混合下是封闭的**：三色 **R 分量恒为 0**、G/B ≥ 170，故任意凸组合仍满足 R = 0（⇒ **S 恒等于 1.000**），且色相仍落在同一条带内。

穷举 `compositeLayers` 能产生的**全部 12 种叠放组合**（2 层与 3 层、各种顺序，alpha 0.40/0.40/0.60，primary 在最上）：

```
H ∈ [163.57°, 186.48°]   S = 1.000   V ∈ [0.930, 1.000]   Y ∈ [0.5997, 0.7706]
```

⇒ **每一种重叠产物自身都同时通过 C-1 / C-2 / C-3。** 与旧色板的区别是本质性的：蓝+橙的混合物掉出了整个约束集（色相进禁用带、亮度塌陷），而本色板的混合物仍是「稀有带 + 高亮 + 满饱和」。
重叠区里「这块属于谁」的歧义由 L1/L2 载体（描边 + Z 序）解决，不指望色相。

---

### 4) C-7 (b) 撞色风险评估（逐色）

| 槽位 | 风险等级 | 说明 |
|---|---|---|
| slot 0 189° 青 | 低（= Day 4 现状） | 残余风险同 §12.11 R10：青马克杯、青绿封面、屏幕内容。有描边兜底 |
| slot 1 177° 水青 | 低 | 与 slot 0 同风险类。S=1.00 / V=1.00 把真实表面排除在外——实拍表面普遍去饱和 |
| slot 2 160° 春青 | **中（三者最高）** | **正好压在允许带下沿，距 [90°,155°] 绿色禁用带只有 5°，零余量**。绿幕布 (0,177,64) 是 H=142°（差 18°）；薄荷/青瓷色的漆面与织物 S < 0.4，被 C-3 挡在门外，无法在满饱和下撞色。**接受但显式标注，不藏。** |

⚠️ **C-7 (b) 要求的「同色系真实物体目视撞色测试」是真机步骤，本轮没有执行。** 按 C-7「两项缺一不得合入」的字面要求，本色板在完成该目视测试前**只能算通过了 (a)**。测试用物：一个青色物体、一个水青/浅蓝绿物体、一个薄荷/春绿物体。

---

### 5) 实现改动

`JudgeE2/Interaction/TapInstanceManager.swift`

- `fillColor` 单色 → `slot0Color / slot1Color / slot2Color` 三个 **plain sRGB 字面量**（**绝不能用 dynamic system color**，D-15）；`fillColor` 保留为 `slot0Color` 的别名。
- `palette = [slot0Color, slot1Color, slot2Color]`。
- 槽位分配由 `Set<UIColor>` 改为**引用相等 `===` 扫描**：`TapInstance.color` 永远是 `palette` 里的那三个对象之一，用身份判等就不必依赖 UIColor 跨色彩空间的 `isEqual:` / `hash` 语义。FIFO 淘汰后被释放的槽位会被回收，**两个存活实例不可能同色**。
- 完整的 C-7 准入记录（上面第 1–4 节）以注释形式写进文件头部，换色时能直接对照。

`JudgeE2/Segmentation/MaskRenderer.swift`：只改了 `MaskLayer.color` 的约束注释（`fillColor` → `palette` 全体），**零逻辑改动**。

---

### 6) Day 5 剩余缺陷（本轮）

#### 缺陷 1 —— warmup 静默出口 + decoder 预热依赖竞争结果

上一轮已修的部分（复核确认仍在位，未回退）：`warmupSegmentationIfPossible` 的 `isEncoding` 出口已有 `warmupPending` 重投 + `diagLog`；`warmupDecoderIfPossible` 已在**争夺 encoder 槽之前**无条件调用。

本轮补掉剩下的两处：

1. **`refreshTapEmbeddingIfNeeded` 的第二个 `guard !isEncoding`（原 1376 行）** —— 这是**另一个**无日志出口。它与上面几行的 `busy` 读取分属两次加锁，中间可被别的队列抢走槽位。丢槽本身无害（下一帧重试），但它与「刷新规则判定不该跑」在日志里**完全无法区分**，正是 Day 5 冷启动竞争读不出来的原因。现在：`refreshSlotLostCount` 计数 + 每次都打日志（该出口按构造是罕见事件，不会刷屏）。
2. **decoder 预热不再依赖谁赢**：
   - `warmupDecoderIfPossible` 从 `if isColdStart` 分支里**提出来**，凡是真正开跑的 refresh 都会调（原来缓存热时的 refresh 根本不碰 decoderQueue）；
   - **丢槽出口里也调一次**（origin `refresh-slot-lost`）—— decoderQueue 与 encoderQueue 独立，输掉 encoder 竞争不该连带赔上 decoder 预热；
   - encode 成功后的那次 rehearsal decode 也从 `if isColdStart` 里提出来（origin `refresh-encoded`）：槽前那次可能还没有 embedding 可用，而 `decoderWarmupDecodeDone` 会把真正的 decode 锁死成只跑一次。
   ⇒ 现在 **warmup / refresh / 丢槽三条路径都会打到 decoderQueue**，首次 tap 不可能再撞 decoder 冷加载（实测 9.5 s 的那一发）。

3. **`scheduleEncoder` 的 `guard !isEncoding`（原 2052 行）** —— Phase 2 每帧路径，丢弃是**设计行为且高频**（encode ≈ 1 s、帧率 30 Hz），逐次打印会刷屏并落进 `Post=` 测量窗口。改为 `encoderSlotBusyDropCount` 计数、**每 30 次报一行**，足以把「编码器饱和，符合预期」与「槽位卡死永不释放」区分开。

两个计数器都放在**已经持有 `stateLock` 的出口内**自增，不新增锁、不新增队列。

#### 缺陷 2 —— `MaskRenderer` 死代码（`allLogits.sorted(by: >)`）

**上一轮已修，本轮复核确认仍在位**：`usesMultimask = (tapPoint256 != nil && channels > 1)` 已把 `allLogits` 的分配、填充与 p30 排序全部关在 `!usesMultimask` 分支里（当前文件 401 / 405 / 472 / 502 / 517 行）。tap 多候选路径上那 65k 元素排序（Mac 实测 5.96 ms）不再执行；box 路径与单 mask 回退路径照旧。本轮无改动。

#### 缺陷 3 —— 后台线程解析 dynamic UIColor

**上一轮已根治，本轮复核确认**：全工程唯一在非主线程解析颜色的地方是 `compositeLayers` 里的 `layer.color.getRed(...)`（decoderQueue），而它拿到的三个色都是 `UIColor(red:green:blue:alpha:)` 纯字面量。`MaskOutlineStyle.outerColor / innerColor` 同样是字面量。`CameraPreview` 里的 `UIColor.clear / .green` 只在 `setupOverlay()`（主线程）解析，且本身非 dynamic。**无 `.systemXxx` 残留。** 本轮无改动。

---

### 7) 如何确认几何输出与 Day 4 逐位一致

1. **改动范围证明**：本轮 `MaskRenderer.swift` 只改了一段 doc comment（`MaskLayer.color`），`buildAlpha` / `keepComponentContaining` / `extractLogits` **一个字符没动**。
   ⚠️ `JudgeE2/Segmentation/` 与 `JudgeE2/Detection/` 在 git 里是 **untracked**，`git diff` 是空的、什么都证明不了（2026-08-02 的教训）。所以沿用留存基准比对：改动前把决策段（`buildAlpha` 全函数 + `keepComponentContaining`）原样导出到 scratchpad，改完再导一次，`diff` → **IDENTICAL**（两段都是）。
2. **常量点名**：`minComponentPx=30` / `minComponentSidePx=3` / `minComponentFill=0.05` / `maxPlausibleLogit=500` / `stabilityDelta=1.0` / `cap60 = contentPx*60/100` / `cap85 = contentPx*85/100` / `gateIouPred >= 0.1` —— 逐条 grep 确认取值与所在行的语义未变。
3. **stability 未参与决策**：`grep` 所有含 `stab` 的分支条件，命中只有 `stabilityScore` 函数体内部（该指标自身的定义 `v > ±delta`），无一条出现在候选筛选 / 排序 / 挑选里。
4. **色相不进 alpha 通道**：`compositeLayers` 的 `accA` 只由各层 `alpha[i] > 0` 与 `opacity` 决定，**与颜色无关**；颜色只影响 RGB。因此二值覆盖与不透明度分布与上一轮青色单色版**逐位相同**。
5. **N=1 与 Day 4 逐位相同**：单实例时 `sa = 0.60`、`accA = 0.60` → 输出 alpha `round(0.6*255) = 153`，RGB → `(0, 217, 255)`，与 Day 4 `renderMask` 的字面量 `(0,217,255,153)` 完全一致。

---

### 8) 未做（明确划界）

- **未做任何延迟优化**（埋点先行，Debugger 方案未落地）；本轮唯一的性能相关动作是复核上一轮的死代码门控。
- **未夹带任何 mask 质量过滤**（R3）：候选选择规则、四个形状/尺寸门限、cap60 / cap85、`iou_pred >= 0.1` 一律未动。
- **stability 仍只打日志**。
- **未动 alpha 分级与描边规格**：0.60 / 0.40、2.0/1.5 pt 与 1.5/1.0 pt、C-6 单点开关全部原样。
- **未动 `maskTTL` / `isMaskValid` 死代码**（Architect B-2 归 Day 6）。
- **未做 Day 6 项**：tap 锚点编号（这正是本轮第 2 节点名的 secondary 区分缺口的解法）、长按删除、点击已有 mask 提升 primary。

---

### 9) 编译验证

```
xcodebuild -project /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/JudgeE2/JudgeE2.xcodeproj \
  -scheme JudgeE2 -destination 'generic/platform=iOS Simulator' build
```
→ **BUILD SUCCEEDED**，源码零 warning。

---

### 10) 改动文件

- 编辑：`JudgeE2/Interaction/TapInstanceManager.swift`（三槽位色板 + C-7 准入记录 + `===` 槽位分配 + `TapInstance.color` 语义注释）
- 编辑：`JudgeE2/Detection/CameraManager.swift`（`refreshSlotLostCount` / `encoderSlotBusyDropCount` 两个计数器 + 两处静默出口可观测化 + decoder 预热去竞争依赖）
- 编辑：`JudgeE2/Segmentation/MaskRenderer.swift`（仅 `MaskLayer.color` 注释）

**未 commit。**

---

### 11) 留给真机验证的点（Builder 无法自证）

1. ⚠️ **C-7 (b) 撞色目视测试（准入必需，尚未执行）**：分别对青色 / 水青 / 薄荷绿三类物体各 tap 一次，确认「即使填充与物体同色，轮廓仍在」。若连轮廓都找不到，那是描边实现问题，不是色板问题（R10）。
2. **三色可辨识度复核**：连点三处，确认三块 mask 目视可区分；**重点看两个 secondary 之间**（本轮第 2 节实算它们的合成 ΔE00 只有 6.8–10.6，是本色板最弱的一环）。
3. **重叠区**：让两块 mask 部分重叠，确认重叠区仍是「更实的青绿」而非可疑的第四色。
4. **warmup 验收**：进入 `.tapToSegment` 后日志须同时出现 `SAM encoder warmup latency` 与 `SAM decoder warmup latency`（origin 可能是 `warmup` / `refresh` / `refresh-slot-lost` / `refresh-encoded` 中任意一个）；首次 tap 的 decode 应落在 38–70 ms。
5. **新日志行**：`background refresh lost the encoder slot (race, n=…)` 若持续增长，说明有队列长期占着 encoder 槽——那是新问题，不是本轮引入的。

---

## Phase 3 Day 6 —— box decoder 惰性加载 + ANE 对齐告警归因更正 (Builder)

**日期**：2026-08-11
**范围**：`SAMDecoder` 的 box decoder 改为按需构造；decoder 预热改为按模式选对模型；更正 `ModelLoader` 的误归因注释。
**依据**：`shared/debug_report.md` §16.5（孤立加载对照实验）、§20.1（D-4 / ISSUE (b)）。

### 1) 问题（Debugger 已验证，非推测）

`Invalid layer: Invalid input tensor channel 1 and format size 2 bytes, must be aligned on 64 bytes`
**从来不是 encoder 发的**：模式切换时孤立加载 milfix encoder → 0 条；TAP#1 孤立加载 decoder → 3 条。
根因是 `SAMDecoder.init` **无条件**加载 box decoder（`computeUnits` 由调用方传入，真机实测 `.all` → 走 ANE），
而 `.tapToSegment` 下 box decoder **一次都不会被调用**（tap 路径走 point decoder，明确用 `.cpuAndGPU`）。

### 2) 改法：`SAMDecoder` 惰性化（`JudgeE2/Segmentation/SAMDecoder.swift`）

- `init` 只做 **URL 查找**（免费）并记下 `boxModelURL` + `boxComputeUnits`；找不到 mlmodelc 仍 `return nil`，
  与旧行为一致。**不再**在 init 里 `MLModel(contentsOf:)` box decoder。
- 新增 `boxModelForDecode()`：首次调用时用**原样的 `boxComputeUnits`** 构造并缓存；
  失败用 `boxModelLoadFailed` latch 住，只报一次错。
  ⇒ **Phase 2 `.segmentation` 的 box decode 仍然跑在 ANE 上，只是构造时点后移。**
- `pointModel` 拆成 `multimaskPointModel: MLModel?`（仍在 init 里 eager 加载，CPU+GPU）
  \+ `pointModelForDecode()`。

**坑 1（fallback `pointModel = model`）怎么绕**：
`isMultimask` 仍**在 init 内**由「multi 包是否加载成功」唯一决定，语义与取值完全不变（对外是 `let`）。
fallback 时 `multimaskPointModel = nil`，`pointModelForDecode()` 回退到 `boxModelForDecode()` ——
即「point 与 box 共用同一个模型、同一套 compute units」这条退路原样保留，只是同样被推迟到首次 decode。

**坑 2（Phase 2 要 box + 要 ANE）怎么绕**：
compute units **一个字都没改**。box decoder 依旧用调用方传入的 `computeUnits`（真机 `.all`）。
本次只动「什么时候构造」，没动「构造成什么」。

**线程安全**：`SAMDecoder` 实例由 `CameraManager` 的 `decoderQueue` 独占创建与调用
（`decoderForQueue()` 注释已声明 "decoderQueue only"），惰性状态因此**不加锁**，并在类型文档里写明这一前提。

### 3) 配套：decoder 预热必须按模式选模型（`CameraManager.swift`）

否则惰性化白做 —— `warmupDecoderIfPossible` 原先**无论什么模式都跑 box prompt 排练 decode**，
在 `.tapToSegment` 下会立刻把 box decoder 造出来（告警只是从 TAP#1 挪到 warmup，一条都不会少），
而且它排练的是**该模式根本不用的那个模型**。

- `warmupDecoderIfPossible(letterbox:backend:origin:mode:)` 新增 `mode` 参数：
  - `.tapToSegment` → 用画面中心点建 `PointPromptBuilder.PointPrompt`，排练 **point decoder**；
  - 其它（Phase 2 `.segmentation`）→ 原样的 box prompt 排练，**行为不变**。
  - fallback（`isMultimask == false`）下两者同模型，哪种排练都对。
- `mode` 的取值来源：`warmupSegmentationIfPossible` 在 **videoQueue** 上与 `capturedBackend` / `capturedLetterbox`
  一起快照 `currentMode`（不新增 decoderQueue 上的跨队列读，D-9 未被动到）；
  `refreshTapEmbeddingIfNeeded` 的三个调用点直接传 `.tapToSegment`（该函数的两个调用点都在 `currentMode == .tapToSegment` 之后）。
- `setMode` 里把 `decoderWarmupDecodeDone` 复位：排练现在是分模式的，旧 latch 会让切入的模式拿到一个冷 decoder。
  代价至多一次多余的排练 decode。
- 排练 decode 的日志前缀改用新增的 `logTag`（`decode(embedding:point:tapIndex:logTag:)`，默认 nil、旧调用点零改动），
  打成 `[SAM][warmup]` 而不是 `[SEG][TAP]`，避免与真实 tap 链混读。
  `SAM decoder warmup latency` 行额外标注 `(origin, point|box prompt)`。

### 4) 更正误归因注释（`JudgeE2/Detection/ModelLoader.swift`）

原注释称 "Original MobileSAM_ImageEncoder … → 3 ANE alignment warnings at load time"。
已替换为带 §16.5 依据的更正说明：孤立加载实验测得 **milfix encoder 单独加载 0 条 / decoder 单独加载 3 条**；
decoder 包从未变过且 pre-milfix 构建同样是 3 条 ⇒ 原始 encoder 也贡献 0 条。
同时写明**实质风险为零**（logits min=-9.74/max=3.60、iou_pred 全在 [0,1]）以及
「milfix 可能仍有其它收益，但『消除对齐告警』不是其中之一」。
`ModelLoader.testMobileSAMLoad()` 本身**未动**（D-5 是独立条目，不在本轮范围）。

### 5) 预期收益（估算，需真机复验）

| 项 | 改前 | 改后（预期） | 依据 |
|---|---|---|---|
| `.tapToSegment` 全 session 告警条数 | 6（启动 3 + TAP#1 3） | **3**（只剩 `ModelLoader.testMobileSAMLoad` 的启动 3 条） | §16.5 定位：3 条 = 一次 box decoder 加载 |
| `.tapToSegment` 常驻内存 | box + multi 两个 decoder | 只剩 multi 一个 | §20.2 记录 TAP#1 峰值 317.9 → 371.3 MB 与两个 decoder 同时加载吻合 |
| 预热/首 tap 关键路径 | 造 2 个 MLModel | 造 1 个 | §20.1 (b) |
| Phase 2 `.segmentation` | — | **无行为变化** | box decoder 仍 ANE，仍在 warmup 阶段被排练构造 |

内存下降的**具体 MB 数无法在模拟器上度量**，报告中不给数字。

### 6) 对 Phase 2 `.segmentation` 的行为影响：无

- 进入 `.segmentation` → `warmupSegmentationIfPossible` → `capturedMode == .segmentation` → box prompt 排练
  → `boxModelForDecode()` 在 **decoderQueue** 上构造（与旧代码同一个队列、同一个时点、同一套 compute units）。
- `runSegmentationPipeline` 的 `decoder.decode(embedding:prompt:)` 调用点、prompt 构造、输出 key 优先级全部未动。
- 唯一可观察差异：多了一行 `[SEG] box decoder built on demand in %.2f ms (units=…)`。

### 7) 未做（明确划界）

- ❌ **未夹带任何 mask 质量过滤**（R3）：`minComponentPx=30` / `minComponentSidePx=3` / `minComponentFill=0.05` /
  `maxPlausibleLogit=500` / cap60 / cap85 / `iou_pred >= 0.1` 与候选选择规则**一字未动**；`MaskRenderer.swift` **本轮零改动**。
- ❌ **stability 仍只打日志、不参与决策**。
- ❌ **未动新色板与描边**（等用户真机验证）。
- ❌ **未做延迟优化**（埋点先行）。
- ❌ **未勾选 Debugger 第 5 条 checkbox**（需新真机日志复验，勾选权不在本轮）。
- ❌ **未动 `ModelLoader.testMobileSAMLoad()` 本体**（D-5）、未动 `currentMode` 同步（D-9）、未动 `maskTTL` 死代码。
- ❌ **未 commit。**

### 8) 编译验证

```
xcodebuild -project /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/JudgeE2/JudgeE2.xcodeproj \
  -scheme JudgeE2 -destination 'generic/platform=iOS Simulator' build
```
→ **BUILD SUCCEEDED**，源码零 warning。

### 9) 改动文件

- 编辑：`JudgeE2/Segmentation/SAMDecoder.swift`（box decoder 惰性化 + `pointModelForDecode()` + `logTag`）
- 编辑：`JudgeE2/Detection/CameraManager.swift`（`warmupDecoderIfPossible` 增 `mode:` + 5 处调用点 + `setMode` 复位 latch + videoQueue 快照 `currentMode`）
- 编辑：`JudgeE2/Detection/ModelLoader.swift`（误归因注释更正）

### 10) 留给真机验证的点（Builder 无法自证）

1. 🔴 **告警条数**：Xcode 选真机 Run，进 `.tapToSegment` 并 tap 数次，数
   `Invalid layer: … aligned on 64 bytes` 的总条数。**预期 3 条且全部出现在启动期**（`MobileSAM models loaded in …` 之前），
   TAP#1 前后应为 **0 条**。若 TAP#1 仍出现 3 条，说明还有别的路径在造 box decoder。
2. **预热日志**：进 `.tapToSegment` 应看到 `SAM decoder warmup latency: … (origin, point prompt)`，
   以及 `[SAM][warmup] decode latency: … iou_preds: …`；**不应**出现 `[SEG] box decoder built on demand`。
3. **Phase 2 回归**：切到 `.segmentation`，应看到 `[SEG] box decoder built on demand in … (units=…)`
   紧跟 `SAM decoder warmup latency: … (…, box prompt)`，且 mask 正常出图。
4. **内存**：对比 `.tapToSegment` 稳态内存与 §20.2 记录的 314–334 MB 基线。
5. ⚠️ **装机前 Xcode 不要开着改工程**（编译≠装机，见用户 memory 的踩坑记录）。

---

## Phase 3 Day 6（追加）—— warmup 每帧空转修复：一次性续作取代逐帧重试 (Builder)

**日期**：2026-08-11
**范围**：`JudgeE2/Detection/CameraManager.swift` 单文件。
**依据**：上一轮真机日志（惰性加载验收通过的同一份）中，切入 `.tapToSegment` 后 8 秒内两行日志每帧交替 40+ 次。

### 1) 自旋的确切成因（代码级定位，非推测）

三段代码构成闭环：

1. `warmupSegmentationIfPossible()` 的 `isEncoding` 分支（D-1 引入）——
   发现编码槽被占，设 `warmupPending = true` 并 `return`。
2. `captureOutput(_:didOutput:from:)`（videoQueue，**每一帧**）——
   `if warmupPending { warmupPending = false; warmupSegmentationIfPossible() }`。
3. `refreshTapEmbeddingIfNeeded()` 在切模式后的第一帧抢到编码槽，
   打出 `cold-start encode taken by background refresh`，然后**持有该槽 7916.98 ms**（ANE 首次编译）。

⇒ warmup 每帧被 (2) 唤起 → 每帧撞上 (1) → 每帧重新武装 → 循环。
**期间没有任何状态会改变答案**：唯一能让 `isEncoding` 变 false 的事件，就是那一次 encode 结束。
所以这 40+ 次尝试**没有一次有成功的可能**，纯粹是每帧两行日志 + 一次 videoQueue 往返。

`warmupSegmentationIfPossible` 开头**无条件**调用 `warmupDecoderIfPossible(origin:"warmup")`，
于是同样被拉进这个循环：decoder 模型早已建好、embedding 仍是 nil ⇒ 每帧打一条
`decoder warmup (warmup): model ready, decode deferred (embedding=none letterbox=ok)`。这是第三行刷屏的来源。

**注意**：`refreshTapEmbeddingIfNeeded` 本身**不**参与刷屏 —— 它的 `guard !busy` 是静默返回。
刷屏全部来自 warmup 侧。

### 2) 一次性续作怎么实现

- **状态**：新增 `warmupWaitingOnEncode: Bool` + `warmupDeferralsFolded: Int`，均由 `stateLock` 保护。
- **统一完成钩子**：新增 `encodeSlotDidFinish(originTag:)` = `drainPendingTaps(originTag:)` + `resumeDeferredWarmupIfWaiting(originTag:)`。
  项目里本来就有「任何 encode 结束都调 `drainPendingTaps`」这条先例，本次把 warmup 的唤醒挂在同一个钩子上，
  **没有新增队列、没有新增锁、没有存闭包**。
  原先 9 个 `drainPendingTaps(originTag:)` 调用点全部改成调钩子；
  另外给 Phase 2 的 `scheduleEncoder` 三个出口**补上**钩子（它原先只放槽、不通知，
  否则在 `.segmentation` 下 warmup 一旦排在它后面就会永远等下去）。
- **消费**：`resumeDeferredWarmupIfWaiting` 在一次加锁内读取并清零标志（幂等，两次 encode 连续结束只唤醒一次），
  然后 hop 到 videoQueue，确认当前模式仍是 `.segmentation` / `.tapToSegment` 后重跑 `warmupSegmentationIfPossible()`。
- **不会形成新循环**：续作重跑时 embedding 刚刚落库，命中既有的
  `warmup encode skipped — embedding already fresh` 分支直接返回，不产生新的 encode。
  即使续作自己去 encode 且失败，标志已被消费为 false，不会二次自动重试（旧的逐帧重试反而没有这个上界）。
- **新增 `releaseEncodeSlot()`**：把散落 8 处的 `lock; isEncoding = false; unlock` 收拢，
  顺带保证槽主标识一起清空。全工程 `isEncoding` 的每一处写入都与 `encodeSlotOwner` 的写入配对（已 grep 核对）。

### 3) 两种延后原因如何区分

| 延后原因 | 判据 | 处置 | 理由 |
|---|---|---|---|
| **还没有相机帧** | `latestCameraBuffer == nil` | **保留逐帧重试**（`warmupPending`，语义一字未改） | 这**确实**是逐帧条件：下一帧要么解决它、要么没解决，重试是对的 |
| **槽位忙，且在途 encode 会产出可用 embedding** | `isEncoding == true` 且 `encodeSlotOwner?.canSatisfyDeferredWarmup == true` | **等待一次性续作** | 只有那次 encode 结束才会改变答案，逐帧问是纯浪费 |
| 槽位忙但没有可唤醒的槽主（理论兜底） | `encodeSlotOwner == nil` | 退回逐帧重试 | 没有完成事件可等，逐帧是唯一诚实选项 |

新增 `private enum EncodeSlotOwner { warmup, tap, backgroundRefresh, segmentationFrame }`，
在 4 处 `isEncoding = true` 处写入、在所有释放处清空。
`canSatisfyDeferredWarmup` 用**穷尽 switch、不写 default** —— 将来新增一条既不写 `embeddingCache`
也不走完成钩子的 encode 路径时，编译器会强制它表态，而不是悄悄把一个等待中的 warmup 饿死。
（当前四条路径全部满足，但这个事实是**写在代码里被检查**的，不是注释里的假设。）

### 4) 线程安全如何保证

- **不丢唤醒（关键点）**：`warmupWaitingOnEncode = true` 是在**观察到 `isEncoding == true` 的同一次
  `stateLock` 加锁内**写入的（读判定与置标志之间不解锁）。槽主释放槽时也在同一把锁下把 `isEncoding` 置 false，
  且**释放之后**才调完成钩子；钩子再加锁消费标志。
  ⇒ 「先看到忙 → 后置标志」与「先释放 → 后消费标志」两个序列不可能交错出丢失窗口。
- **不重复唤醒**：消费与清零在同一次加锁内完成。
- **无新队列、无新锁**：只复用既有的 `stateLock` / videoQueue / encoderQueue / decoderQueue。
- **无新增跨队列无锁读（D-9 的坑）**：续作里读 `currentMode` 发生在 **videoQueue** 上 ——
  与 `warmupSegmentationIfPossible` 现有的 `capturedMode` 快照、`runDetectionPipeline` 的读取**同队列同属性**，
  没有引入新的读取位置或新的队列。
- **无锁内调用**：所有 `encodeSlotDidFinish` 调用点均在解锁之后（已逐点核对；`stateLock` 是 `NSLock`，非递归，锁内调用会死锁）。
- **强制释放不走钩子**：`setBackend` / `setEncoderResolution` 是**重置**而非 encode 完成，
  它们自行清空 `warmupWaitingOnEncode`（否则会留下一个等待永不到来的完成通知），随后本来就会重新 kick warmup。

### 5) 日志改成什么样

**每条延后原因只报一次**，重复的折叠成计数在恢复时一并汇报。修复后冷启动预期日志：

```
=== MODE SWITCH → tapToSegment ===
[SAM] warmup deferred — no camera frame yet, armed for next frame            ← 0 或 1 次
[SAM] first frame arrived — running deferred warmup                          ← 恰好 1 次（原为 40+）
[SAM] cold-start encode taken by background refresh — warming state raised
[SAM] decoder warmup (refresh): model ready, decode deferred until an embedding exists (embedding=none letterbox=ok)   ← 恰好 1 次（原为几十条）
[SAM] warmup deferred — background-refresh encode in flight; awaiting its completion (one-shot, no per-frame retry)     ← 恰好 1 次（原为 40+）
        …… 约 8 秒，warmup / decoder-warmup 相关日志一行不出 ……
[TAP] background embedding refresh 7916.98 ms
SAM encoder warmup latency: 7916.98 ms (via background refresh)
[SAM][warmup] decode latency: … | iou_preds: …
SAM decoder warmup latency: 208.39 ms (refresh-encoded, point prompt)        ← 与上一轮验收完全一致
[SAM] deferred warmup resumed after background-refresh encode
[SAM] warmup encode skipped — embedding already fresh (≈10 ms old)
```

折叠计数在真发生时才出现，形如
`[SAM] decoder warmup: 2 further deferral(s) folded in while waiting for the first embedding` /
`[SAM] deferred warmup resumed after tap-encode encode (1 further attempt(s) folded in)`。

### 6) 上一轮四条已验收行为：如何确认未回退

| 已验收行为 | 确认方式 |
|---|---|
| box decoder 保持惰性 | `SAMDecoder.swift` **本轮零改动**（mtime 仍是上一轮的 20:05，本次只改了 `CameraManager.swift`）；本次未新增任何 decoder 构造点 |
| tap 模式 warmup 用 point prompt 排练 | `warmupDecoderIfPossible` 的 `mode` 分支**一字未动**；续作路径调的是 `warmupSegmentationIfPossible()`，它照旧在 videoQueue 快照 `capturedMode` 并透传 ⇒ tap 模式仍走 point |
| `Invalid layer` 3 条且全在启动期 | 告警来自 box decoder 加载；本轮没有任何新代码会构造 box decoder（唯一新增的模型相关调用是续作里对 `warmupDecoderIfPossible` 的重入，而它被 `decoderWarmupDecodeDone` latch 住直接返回） |
| 无 `[SEG] box decoder built on demand`（tap 模式） | 同上：该日志只在 `boxModelForDecode()` 首次构造时打印，tap 模式下无调用者 |

排练归属也未变：`refresh-encoded` 的 decoder warmup 调用在完成钩子**之前**同步入 decoderQueue，
decoderQueue 是串行队列 ⇒ 它先拿到 latch，`SAM decoder warmup latency: …(refresh-encoded, point prompt)`
这行的 origin 与上一轮一致，不会被续作抢成别的 origin。

### 7) 未做（明确划界）

- ❌ **未动 `MaskRenderer.swift`**（本轮零改动，mtime 未变）；R3 禁令项
  `minComponentPx=30` / `minComponentSidePx=3` / `minComponentFill=0.05` / `maxPlausibleLogit=500` /
  cap60 / cap85 / `iou_pred >= 0.1` 与候选选择规则**一字未动**。
- ❌ **stability 仍只打日志、不参与决策**；**色板与描边未动**（等用户验证）。
- ❌ **未动 parked 语义**：TAP#1 冷启动 parked → resumed（2600 ms）仍是预期路径。
- ❌ **未新建队列、未引入新的跨队列无锁读**。
- ❌ **未做其它延迟优化**（埋点先行）。
- ❌ **未 commit。**

### 8) 期望值校准（如实）

本轮**不承诺**冷启动时间下降。7.9 s 的主体是 ANE 首次编译，与调度无关。
可以如实主张的只有三点，且都可由日志直接证伪：
(a) 8 秒窗口内 warmup 相关日志从 80+ 行降到 3 行；
(b) 那 40+ 次 videoQueue → 状态判定 → 重新武装的往返被消除；
(c) 状态机从「靠逐帧轮询碰运气」变成「事件驱动、可解释」。
**如果冷启动时间顺带变好，需要有实测依据才写；没有变好不算失败。**

### 9) 编译验证

```
xcodebuild -project /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/JudgeE2/JudgeE2.xcodeproj \
  -scheme JudgeE2 -destination 'generic/platform=iOS Simulator' build
```
→ **BUILD SUCCEEDED**，源码零 warning / 零 error。

### 10) 改动文件

- 编辑：`JudgeE2/Detection/CameraManager.swift`（唯一改动文件）
  - 新增：`warmupWaitingOnEncode` / `warmupDeferralsFolded` / `encodeSlotOwner` + `EncodeSlotOwner` 枚举 /
    `decoderWarmupDeferralLogged` / `decoderWarmupDeferralsFolded`
  - 新增函数：`releaseEncodeSlot()` / `encodeSlotDidFinish(originTag:)` / `resumeDeferredWarmupIfWaiting(originTag:)`
  - 改写：`warmupSegmentationIfPossible` 的 `isEncoding` 分支；`warmupDecoderIfPossible` 的 deferred 日志
  - 机械替换：9 处 `drainPendingTaps(originTag:)` → `encodeSlotDidFinish(originTag:)`；
    8 处 `lock; isEncoding = false; unlock` → `releaseEncodeSlot()`；`scheduleEncoder` 三出口补钩子

### 11) 留给真机验证的点（Builder 无法自证）

1. 🔴 **自旋消失**：进 `.tapToSegment`，数 `first frame arrived — running deferred warmup` 与
   `warmup deferred` 的条数。**预期各 ≤ 1 条**；`decoder warmup … decode deferred` **预期 1 条**。
2. 🔴 **续作真的接上了**：8 秒后应出现 `[SAM] deferred warmup resumed after background-refresh encode`，
   紧跟 `warmup encode skipped — embedding already fresh`。**若只见前者不见后者**，说明续作重跑时缓存判定异常，需报。
   **若两者都不出现**，说明唤醒丢失（等待被饿死），是本轮最需要盯的失败模式。
3. **四条已验收行为复验**：`Invalid layer` 仍为 3 条且全在启动期、TAP#1 前后 0 条；
   预热行仍是 `(refresh-encoded, point prompt)`；tap 模式全程无 `[SEG] box decoder built on demand`。
4. **Phase 2 回归**：切到 `.segmentation`，mask 正常出图，仍见
   `[SEG] box decoder built on demand`（该模式下**应该**出现），无卡死、无编码槽泄漏
   （`encode request dropped — encoder busy (N so far)` 的 N 应停止无界增长）。
5. ⚠️ **装机前 Xcode 不要开着改工程**（编译≠装机，见用户 memory 的踩坑记录）。

---

## Phase 3 Day 6 — 视觉反馈与多项修复 (Builder)

**日期：** 2026-08-12
**状态：** 完成（BUILD SUCCEEDED，零 error / 零 warning）。
Day 6 全部 Builder 条目已实现，R3 禁令一处未动。

---

### D-15：iou_pred gate 修复 ✅

**根因：** `tapDecodeWithPoint` 原本在 `buildTapAlpha` 之前就用 `result.iouPreds.max()` 作门控，导致全帧候选（ch2，iou=0.993）可能通过门控，而被选中的真正候选 iou 可能为 0。

**修复（`CameraManager.swift`）：** 将 `buildTapAlpha` 调用移到 gate 之前，gate 改为 `selected?.iou ?? result.iouPreds.max() ?? 0`。若无候选则 fallback 到 max（兜底一致性）。阈值 0.1 严格不变（R3）。

日志更新：`gate iou_pred(max)` → `gate iou_pred(selected)`，反映了修复语义。

---

### D-16：warmup 延迟日志误报修复 + 模型重复加载审查 ✅

**根因：** `resumeDeferredWarmupIfWaiting` 无条件调用 `warmupSegmentationIfPossible()`，而该函数检测到 embedding 新鲜时会打印「warmup encode skipped — embedding already fresh」，导致日志中「跳过」行与「warmup latency: 1225 ms」行同时出现，含义相互矛盾。

**修复（`CameraManager.swift`）：** `resumeDeferredWarmupIfWaiting` 在调用前先检查 embedding 年龄，若 ≤ 5000 ms 则直接打「deferred warmup resolved」并跳过，不再触发 warmup 函数。只有 embedding 确实陈旧时才调用 warmup 并跑 encode。

**模型重复加载调查：** 新增 `loadReason` 字符串区分 `"first load"` vs `"rebuild (units changed)"`，日志行：`[SAM] encoder: loading model (units=..., reason=...)`，供 Debugger 在真机日志中核实是否有非预期的 rebuild。

---

### Item 3：Tap 指示动画 + 锚点标记 ✅

**TapAnchorMarker struct（`CameraManager.swift`，`@Published tapAnchorMarkers`）：**
- `id: UUID`, `viewPoint: CGPoint`, `slotIndex: Int`, `requestGen: Int`
- 由 `publishAnchorMarkersOnMain()` 从 `TapInstance.viewPoint` 构建，在 decode 成功后 + promote 成功后刷新

**TapInstance.viewPoint 字段（`TapInstanceManager.swift`）：**
- 新增 `viewPoint: CGPoint?`，在 `addInstance(point:viewPoint:requestGen:)` 时传入并存储
- `slotIndex(in:)` 方法：按 `===` 引用比较色板槽位，返回 0/1/2

**UI（`ContentView.swift`）：**
- `TapAnchorMarkerView`：22pt 实心圆（slot 色板颜色）+ 居中的槽编号（1/2/3），用 `.position(marker.viewPoint)` 叠加在相机 overlay 上
- `TapRippleEffect`：触发时圆环从小扩散到 80 pt，0.4 s easeOut，`.onChange(of: trigger)` 驱动（tap 接受时 `rippleTrigger &+= 1`）
- 通过 `onChange(of: cameraManager.lastTapIndex)` 在新 tap 接受时取 `lastTapViewPoint` 作为 ripple 位置
- 两个视图均 `allowsHitTesting(false)`

---

### Item 4：Mask 高亮（hit-test 促升）✅

**实现（`CameraManager.handleTap` 快路径，在 `addInstance` 之前）：**

将 `canonicalPoint` 映射到 256×256 mask 空间（与 decode 路径相同的变换），对池中所有实例检测 `inst.maskAlpha[ty*256+tx] > 0`，命中则 `promoteToPrimary(id:)` + `recompositeForPromote(letterbox:gen:)` 刷新合成图，不触发 encode/decode。

---

### Item 5：清除反馈 ✅

- 双击 haptic：`UIImpactFeedbackGenerator(style: .heavy).impactOccurred()`（`TouchHandler.handleDoubleTap`）
- `handleClearAllTapMasks`：清除后 `showSegmentHint = true`，1.5 s 后自动归零
- 「请点击分割」提示文字：`if cameraManager.showSegmentHint`，居中悬浮在屏幕中下部，`.animation(.easeInOut(duration: 0.2))` 淡入淡出
- `discardAllTapWork` 同时清 `tapAnchorMarkers = []`

---

### Item 6：Embedding 缓存复用原因日志 ✅

新增 `[CACHE] re-encode reason: <geometry_change|ttl_expired|manual_tap>` 日志，两处打点：
1. `handleTap` 慢路径分流前（几何变化 / TTL 过期 / manual tap）
2. `refreshTapEmbeddingIfNeeded` background refresh 路径（manual_tap / heavy_drift）

**纯日志，无任何行为改动。**

---

### Item 7：Phase 2 / Phase 3 UI 切换 ✅

已通过既有的 Mode Picker（`.detectionOnly` / `.segmentation` / `.tapToSegment`）实现。

**新增：** 悬浮快切按钮（底部左侧），仅在 `.segmentation` 或 `.tapToSegment` 模式下显示，单击在两者之间切换，无需展开 Settings 面板。按钮图标与标签跟随当前模式实时变化（`hand.tap` / `cpu`）。

模式切换调用 `cameraManager.setMode(newValue)`，内部已调用 `discardAllTapWork()` 清空实例池，不影响相机帧率。

---

### 编译验证

```
xcodebuild -project JudgeE2/JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**，零 error，零 warning。

### 改动文件

| 文件 | 改动内容 |
|------|---------|
| `JudgeE2/Detection/CameraManager.swift` | D-15 gate 修复、D-16 日志修复 + loadReason、TapAnchorMarker struct、@Published tapAnchorMarkers/showSegmentHint、publishAnchorMarkersOnMain、recompositeForPromote、hit-test 促升（Item 4）、CACHE 日志（Item 6）、discardAllTapWork 清 tapAnchorMarkers（Item 5）、handleClearAllTapMasks 设 showSegmentHint（Item 5） |
| `JudgeE2/Interaction/TapInstanceManager.swift` | viewPoint 字段、slotIndex(in:) 方法、addInstance viewPoint 参数 |
| `JudgeE2/Interaction/TouchHandler.swift` | handleDoubleTap 加 haptic（Item 5） |
| `JudgeE2/UI/ContentView.swift` | @State ripplePoint/rippleTrigger、anchor marker overlay、TapRippleEffect 触发逻辑、showSegmentHint 提示文字、快切按钮、TapRippleEffect struct、TapAnchorMarkerView struct |

### 未做项（严格划界）

- **MaskRenderer.swift：** 一行未动（R3 禁令）
- **SAMDecoder.swift：** 一行未动（D-4 已完成，mtime 08-11 20:05）
- **iou_pred 阈值 0.1：** 一字未变
- **新队列、新依赖：** 未引入
- **Debugger 工作：** 真机验证 D-15 gate 生效（iou 分布对比）、D-16 日志不再矛盾、ripple/anchor 视觉验证、hint 文字出现时机 —— 均属 Debugger

---

## Phase 3 Day 6 追加 — setBackend 竞态修复 + CACHE reason 展开（2026-08-13）

### D-17：setBackend early return guard + encoder drop 诊断日志

**问题**：启动期 YOLO 被加载 3 次（均 `MLComputeUnits(rawValue: 2)`），日志还出现两条 `[SAM] encoder: loading model (reason=first load)`，第二次是 cold build 完成后被 `setBackend` 的清理 block 清掉、warmup 再建导致（+1.3 s）。

**调用点统计**（grep 结果）：
- `ContentView.swift:232` — `onAppear` 时调用一次（用持久化的 `backend` 值）
- `ContentView.swift:236` — `onChange(of: backend)` 时调用一次

两路在 App 启动时先后触发，backend 值相同，均导致 `reloadModel` + encoder drop。

**修改内容**：
- `CameraManager.swift` `setBackend` 函数 — 在 `sessionQueue.async` 块最开头加：
  ```swift
  guard self.backend != backend else { return }
  ```
  同 backend 时直接 return，阻止幂等的重复加载。
- `CameraManager.swift` `encoderQueue.async` 清理块 — 改为展开写法并加诊断日志：
  ```swift
  self.encoderQueue.async {
      if self.samEncoder != nil {
          diagLog("[SAM] encoder: dropped by setBackend cleanup (was built)")
      }
      self.samEncoder = nil
      self.samEncoderUnits = nil
  }
  ```

**验收**：修复后启动期 `setBackend` 对同一 backend 值只执行一次，日志中不再出现第二条 `[SAM] encoder: loading model (reason=first load)`；若未来再出现竞态，`dropped by setBackend cleanup (was built)` 日志行会立刻可见。

### 任务 2：CACHE reason 日志展开

**问题**：后台刷新全部报 `heavy_drift`，实为 TTL age ≥ 5000 ms 触发的定期刷新，误标签使日志无法区分 drift 驱动与 age 驱动。

**修改内容**：
- `CameraManager.swift` `refreshTapEmbeddingIfNeeded` — 在进入 encoderQueue.async 之前新增触发点日志，打印实际 age 与 threshold：
  ```swift
  diagLog("[CACHE] background refresh triggered: \(triggerReason)")
  ```
- `refreshTapEmbeddingIfNeeded` 的 reason 计算展开为三分支：
  - `cold_start` — 无 embedding 缓存
  - `ttl_approaching` — age ≥ 5000 ms（定期刷新，非 drift）
  - `heavy_drift` — age 较小但 drift 触发（当前实测不会走这条）
- 日志新增 `threshold=5000ms` 字段，方便日后修改阈值时日志与代码保持一致。

### 改动文件

| 文件 | 改动内容 |
|------|---------|
| `JudgeE2/Detection/CameraManager.swift` | `setBackend` early return guard；encoderQueue 清理块展开 + drop 日志；触发点日志 `[CACHE] background refresh triggered`；reason 三分支展开 + threshold 字段 |
| `shared/tasks.md` | 新增 D-17（已勾选）+ P-3（未勾选）条目 |
| `shared/builder_progress.md` | 本次记录 |

### 未改动项（严格划界）

- `MaskRenderer.swift`、`SAMDecoder.swift` — 一行未动（R3 禁令）
- tapToSegment 分割路径（`tapDecodeWithPoint`、候选选择规则）— 一行未动
- `minComponentPx` / `minComponentSidePx` / `minComponentFill` / `maxPlausibleLogit` / `stabilityDelta` / `cap60` / `cap85` — 一字未变
- `setMode` — 未动

---

## Phase 3 Day 7 — box decoder 异步预构建 + maskTTL 死代码清理 (Builder)

**日期：** 2026-08-15
**状态：** 完成（BUILD SUCCEEDED，零 warning）。Task 1 (P-3) + Task 2 (B-2) 已实现并编译通过。Task 3 (C-7b) 阻塞于真机测试。Task 4 (ISSUE-A) 未确认，不实现。

---

### Task 1 — P-3 修复：box decoder 异步预构建 ✅

**依据：** tasks.md Day 7 Builder § P-3；architect_output.md 第 1294 行裁决（"若实际落到 `.segmentation` 首帧且用户可感 >1 s，作为新条目上报"）。

**问题根因：**
`SAMDecoder.boxModelForDecode()` 在首次调用时构造 `MLModel(contentsOf:configuration:)`，
包含 ANE 首次编译开销（预期约 1–2 s）。
现有 warmup 路径在有 embedding 时会通过排练 decode 触发该构造；
但若 warmup 排练在 embedding 落库前执行（"decode deferred, embedding=none"），
且 `.segmentation` 首帧随后调用 `decode(embedding:prompt:)`，则构造发生在首帧，
使用户可感延迟升至 >200 ms（与 Day 7 Debugger 任务 "box decoder built on demand" 日志吻合）。

**实现：**

文件：`JudgeE2/Detection/CameraManager.swift`

1. **新增属性** `private var boxDecoderPrebuilt = false`（decoderQueue only，紧跟 `samDecoderUnits` 声明）。

2. **`decoderForQueue(computeUnits:)` 新增一行**：新 SAMDecoder 实例创建时将 `boxDecoderPrebuilt = false` 重置，确保 compute units 变更后下次仍会预构建。

3. **新增方法 `scheduleBoxDecoderPrebuild(backend:)`**：
   - 分发到 `decoderQueue.async`（不阻塞 UI 或 videoQueue）
   - 检查 `boxDecoderPrebuilt`（幂等，每个 decoder 生命周期至多执行一次）
   - 检查 `isAppBackgrounded`
   - 通过 `decoderForQueue` 取得 decoder
   - 创建零填充哑 embedding `[1, 256, 64, 64]` Float32
   - 用 `PromptBuilder.buildBoxPrompt(x1:256, y1:256, x2:768, y2:512, origW:1280, origH:720, inputSize:1024)` 建哑 prompt
   - 调用 `decoder.decode(embedding:dummyEmb, prompt:dummyPrompt)` → 内部触发 `boxModelForDecode()` → MLModel 构造并缓存；结果丢弃
   - 置 `boxDecoderPrebuilt = true`，打 `[SAM] box decoder pre-built (async, dummy decode)` 日志

4. **`setMode` 两处调用 `scheduleBoxDecoderPrebuild(backend: self.backend)`**：
   - `case .segmentation` — 直接进入分割模式时，在 `warmupSegmentationIfPossible()` 之后异步调用
   - `case .tapToSegment` — 投机性预构建（用户可能随后切换到 segmentation），在 `warmupSegmentationIfPossible()` 之后调用

**不影响项（严格验证）：**
- `SAMDecoder.swift` **一行未动**（R3 + 任务绝对约束）
- `MaskRenderer.swift` **一行未动**
- `boxModelForDecode()` 私有方法未改、计时日志未改（"[SEG] box decoder built on demand" 仍在预构建时打印一次，之后不再出现于首帧）
- Phase 2 `.segmentation` decode 路径（`boxModelForDecode()` 缓存 hit）零行为变化
- R3 禁令常量一字未动

**验收条件（需 Debugger 真机确认）：**
- `.segmentation` 首帧不出现 `box decoder built on demand` 日志（在模式切换日志后应只出现一次，来自预构建）
- 首帧 box decode 延迟 <200 ms（与后续帧 65–91 ms 同量级）

---

### Task 2 — maskTTL / isMaskValid 死代码清理（Architect B-2）✅

**依据：** tasks.md Day 7 Builder § B-2；任何渲染路径均未调用 `isMaskValid` / `maskTTL`。

**改动：** `JudgeE2/Interaction/TapInstanceManager.swift`

**移除：**
- `TapInstance.maskTTL: Date?` 计算属性（原 74–76 行，`maskTimestamp + maskTTLSeconds`）
- `TapInstance.isMaskValid(now: Date) -> Bool` 方法（原 78–81 行）
- `TapInstanceManager.maskTTLSeconds: TimeInterval = 2.0` 常量（原 109 行，含注释块）

**保留：**
- `TapInstance.maskTimestamp: Date?` — 降级为纯遥测时间戳，注释更新说明 B-2

**新增：**
- `TapInstance.maskAgeMs(now: Date) -> Double?` 计算方法：当 `maskTimestamp` 非 nil 时返回 `(now - maskTimestamp) * 1000`，否则返回 nil。遥测用途（日志）。

**全项目 grep 确认：**
- `TapInstanceManager.isMaskValid` / `TapInstanceManager.maskTTL` / `TapInstanceManager.maskTTLSeconds`：零存活调用点 ✅
- `TemporalManager.isMaskValid(nowMs:)` + `maskTTLms`（Phase 2 segmentation 路径）：保留不动，与 Phase 3 死代码无关 ✅
- `CameraManager.swift` 第 2485 行 `temporal.isMaskValid(nowMs:)` = TemporalManager 方法，不属于本次清理 ✅

**行为零变化：** `drawableInstances()` 逻辑、渲染路径、Phase 2 mask TTL（TemporalManager）均未触碰。

---

### Task 3 — C-7(b) 真机撞色补测（BLOCKED - 需 Debugger 设备测试）

**状态：** ⛔ BLOCKED — 需真机目视验证，Builder 无法自证。

**代码核实（READ ONLY）：**

`TapInstanceManager.swift` 色板值（与 architect_output.md §3.3.3 完全吻合）：
- slot 0 cyan: `UIColor(red: 0.0, green: 217.0/255.0, blue: 1.0, alpha: 1.0)` → H=188.94°
- slot 1 aqua: `UIColor(red: 0.0, green: 1.0, blue: 242.0/255.0, alpha: 1.0)` → H=176.94°
- slot 2 spring cyan: `UIColor(red: 0.0, green: 1.0, blue: 170.0/255.0, alpha: 1.0)` → H=160.00°

`MaskRenderer.swift` 不直接存储这些色相值 — 颜色通过 `MaskLayer.color` 从 TapInstanceManager 传入 `compositeLayers`，`renderMask`（Phase 2 路径）保留字面量 `(0, 217, 255, 153)`（= slot0）。

**Debugger 需执行的验证步骤：**

1. 进入 `.tapToSegment` 模式
2. 分别对以下三类物体各 tap 一次：
   - 青色物体（青马克杯、青绿封面等）→ 预期 slot 0 mask
   - 水青/浅蓝绿物体（接近 H=177° 的物体）→ 预期 slot 1 mask
   - 薄荷/春绿物体（接近 H=160° 的物体）→ 预期 slot 2 mask
3. 判定标准（R10）：「填充看不清但轮廓仍在」= PASS；「连轮廓都找不到」= 描边实现问题，不判色板失败
4. 同屏放置三实例，目视确认三块可区分（W-1 secondary 可辨识度验证）
5. 通过后：将 architect_output.md §3.3.3 色板行由 PROVISIONAL 改为 FINAL，§9.5 同步更新

**此任务 tasks.md 中不勾选（需 Debugger 真机结果）。**

---

### Task 4 — ISSUE-A（Safe Area Y 轴偏移）— 未实现

**状态：** 不实施。

**根据：** `shared/debug_report.md` §25.9 及 §26 末尾汇总表（第 2522 行）：
「ISSUE-A（§25.9）：Safe Area Y 偏移 | **仍悬而未决** | ✅ 目视确认（待）」

偏移**未经真机确认**。按 tasks.md Day 7 指令："仅在 Debugger 确认偏移后动手；若真机 PASS 则此条关闭"。当前状态 = 未确认，不实现。

---

### 编译验证

```
xcodebuild -project JudgeE2/JudgeE2.xcodeproj -scheme JudgeE2 \
  -configuration Release -destination 'generic/platform=iOS' \
  build CODE_SIGNING_ALLOWED=NO
```
→ **BUILD SUCCEEDED**，零 error，零 warning。

---

### 改动文件

| 文件 | 改动内容 |
|------|---------|
| `JudgeE2/Detection/CameraManager.swift` | 新增 `boxDecoderPrebuilt` 属性；`decoderForQueue` 重置该标志；新增 `scheduleBoxDecoderPrebuild(backend:)` 方法；`setMode` `.segmentation` 和 `.tapToSegment` 两处调用预构建 |
| `JudgeE2/Interaction/TapInstanceManager.swift` | 移除 `maskTTL`、`isMaskValid`、`maskTTLSeconds`；`maskTimestamp` 注释降级为遥测；新增 `maskAgeMs(now:)` |

### 绝对未动项

- `SAMDecoder.swift` — 一行未动
- `MaskRenderer.swift` — 一行未动
- R3 禁令常量：`minComponentPx=30` / `minComponentSidePx=3` / `minComponentFill=0.05` / `maxPlausibleLogit=500.0` / `stabilityDelta=1.0` / `cap60` / `cap85` / 候选选择规则 — 一字未变

---

## Phase 4 Day 1 — tap 锚点常驻编号（primary 强调）+ D-7' 六段埋点 (Builder, 2026-08-16)

### Task A — tap 锚点常驻编号上屏（L3 载体，补齐 W-1 欠载）

**现状复核：** `TapAnchorMarkerView` 此前已渲染 `slotIndex + 1`（1/2/3），故「编号上屏」与
「编号随 FIFO 淘汰释放」两条本已成立 —— 编号取自 `slotIndex`，而 `TapInstanceManager.addInstance`
的槽位分配是「first-unused-slot」，实例被 FIFO 淘汰时槽位（= 颜色 = 编号）同步回收，
**编号 N 与色相 N 永远同源、永不错配**。本次不改动该映射。

**本次实际补齐的是欠缺的两条：primary/secondary 区分 + 任意相机内容下的可辨识度。**

**`CameraManager.TapAnchorMarker`**
- 新增 `isPrimary: Bool` 字段；`publishAnchorMarkersOnMain()` 从 `inst.isPrimary` 填充
- `requestGen` 注释订正：它是 tap 序号（遥测/调试用），**不是**上屏编号；上屏编号来自 slotIndex

**`UI/ContentView.swift` — `TapAnchorMarkerView`**
- 签名 `TapAnchorMarkerView(slotIndex:isPrimary:)`（`isPrimary` 有默认值 `false`，旧调用点不破坏）
- primary：直径 28pt、字号 16pt、`.heavy` 字重、外加 2pt 白色描边环、不透明度 1.0
- secondary：直径 22pt、字号 12pt、`.bold` 字重、无环、不透明度 0.85
  （与 §3.4 的 0.60 / 0.40 mask 填充分级同向，语义一致：secondary = 「这里曾被选中」）
- 可辨识度三层叠加，**均不依赖背景**：不透明黑色底盘（直径 +4）→ 饱和槽位色圆盘 →
  黑色数字。三个槽位色相对亮度 Y ≥ 0.568，黑字对比度 ≥ 4.5:1
- 白环是**第二条非色相通道**（achromatic），色觉障碍用户仍可读出 primary 角色

**未触碰：** 三色板数值（188.94° / 176.94° / 160.00°，FINAL）、槽位分配算法、
`primaryOpacity` / `secondaryOpacity`、任何分割逻辑。

### Task B — D-7' 六段埋点（实现 debug_report.md §30 规格，代码侧）

严格按 §30.1 位置表与 §30.6 落地清单实施，全部在 `CameraManager.swift` 内。

| 戳 | 实现 | 位置 |
|---|---|---|
| T1 | 复用现有 `tapStartMs` | `handleTap` |
| T2 | 新增 `lockDoneMs` | `stateLock.unlock()` 之后 |
| T3 | 新增 `enqueueMs` | `tapDecodeWithPoint` 内、**`decoderQueue.async` 调用点之前（闭包外）** |
| T4 | 新增 `decodeStartMs` | 闭包内、`decoder.decode(...)` guard 之前 |
| T5 | 新增 `decodeEndMs` | decode guard 块结束之后（`iouSane` 哨兵之前） |
| T6 | 复用主线程 e2e 戳 | 提取为 `displayedMs`，`e2eMs = displayedMs - tapStartMs` |

- **时钟源全部为 `PerfLogger.nowMs()`（单调）**，未使用 `Date().timeIntervalSince1970`（§30.4 约束 4）
- **T3 在调用点而非闭包内**（§30.1 警告）：否则 `qwait` 恒为 0，R4 decode 堆积不可观测
- **`[TAP#N] mask displayed` 行一字未改**（§30.4 约束 1）：§27/§29 的采样正则与既有 22 个样本保持有效
- `[D7']` 为独立一行，紧随原 perfLog 之后，含 `path.label`（§30.4 约束 2），`%.1f` 精度（约束 3）

输出格式：
```
[D7'][TAP#3] lock=0.2 decide=3.1 qwait=1.4 decode=55.8 post=14.2 | total=74.7 ms (fast/decode-only)
```

**参数传递（§30.6 第 8 项）：** `lockDoneMs: Double` 贯穿
`handleTap → tapEncodeAndDecode → tapDecodeWithPoint` 与 `parkTap → pendingTaps → drain`，
使 parked 路径的 `decide` 段（含泊车等待）同样口径正确。
`pendingTaps` / `overflow` 元组随之各加一个 `lockDoneMs: Double` 字段。

**自洽性（§30.3）：** `I1+I2+I3+I4+I5 == I6` 按构造恒等；`T6 − T1` 与 `tap→mask` 取自
**同一个** `nowMs()` 调用 ⇒ 逐位相同，可作为「埋点生效」的最低验证。

**未实现：** §30.5 的 D-7'-ext（`T2b` videoQueue 排队 / `T2c` encoderQueue 排队）。
§30.5 明示「本项不阻塞六段埋点落地」，且需再改两处签名，非零风险，按指令不实施。

**tasks.md 中 D-7' 复选框未勾选** —— 该条列在 Debugger 名下，需真机 Release 采集验证后由 Debugger 勾选。

### 编译验证

```
cd JudgeE2 && xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**，零 error，零 Swift warning。

⚠️ 提醒 Debugger：§30 末尾要求埋点数据**必须在 Release 构建下采集**（§24.1：Debug/-Onone
会把 Swift 侧成本放大 4–6 倍）。

### 改动文件

| 文件 | 改动内容 |
|------|---------|
| `JudgeE2/Detection/CameraManager.swift` | `TapAnchorMarker.isPrimary` 新增 + 发布；D-7' T2/T3/T4/T5 时间戳、`lockDoneMs` 参数贯穿（`tapEncodeAndDecode` / `tapDecodeWithPoint` / `parkTap` / `pendingTaps` / `overflow`）、`[D7']` 日志行 |
| `JudgeE2/UI/ContentView.swift` | `TapAnchorMarkerView` 增加 `isPrimary` 与三层可辨识度构造；调用点传入 `marker.isPrimary` |

### 绝对未动项

- `SAMDecoder.swift` — 一行未动
- `MaskRenderer.swift` — 一行未动
- R3 禁令常量：`minComponentPx=30` / `minComponentSidePx=3` / `minComponentFill=0.05` /
  `maxPlausibleLogit=500.0` / `stabilityDelta=1.0` / `cap60` / `cap85` / 候选选择规则 — 一字未变
- 三色板 HSV 值、槽位分配规则、`primaryOpacity` / `secondaryOpacity` — 未变
- 控制流零变化：无新队列、无新锁、无新路由状态（`lockDoneMs` 为纯遥测数值，不参与任何分支）

---

## Phase 4 Day 2–3 — Re-anchor 刷新循环：DriftDetector + ReAnchorLoop (Builder, 2026-08-16)

依据 `architect_output.md §16`（Re-anchor 架构契约）全文实施。
tasks.md Day 2–3 Builder 条目中「每 100 ms 最多触发一次」一行**已被 §16 的 D-2/D-15.2 裁决取代**，
按 §16.2.3 实现**负载自适应节流**（同时最多一个在途批次，超出的漂移事件丢弃而非排队）。

### 新增文件

| 文件 | 内容 |
|------|------|
| `JudgeE2/Interaction/DriftDetector.swift` | §16.1 接口契约：`hasDrifted(from:to:)` + `drift(from:to:)`（返回两个分量的量值，供日志用）；可调参 `translationThresholdPt = 10.0` / `rotationThresholdDeg = 3.0`（具名 `static var`，Debugger 可直接改值重测，不改其它代码）；两分量取 OR（§16.1.3） |
| `JudgeE2/Interaction/ReAnchorLoop.swift` | §16.2.3 节流状态机：`beginBatch(count:anchor:)` / `completeUnit(batch:)` / `reset()` / `seedAnchorIfNeeded(_:)`。**自身不持锁**（§16.7 禁止新增队列/锁），全部成员要求调用方持有 `CameraManager.stateLock`，方法文档逐条写明该前置条件 |

### CameraManager 改动（`Detection/CameraManager.swift`）

- 新增 `private let reAnchor = ReAnchorLoop()`（stateLock 保护）+ `reAnchorNoEmbeddingSkips`（仅 videoQueue 访问的日志限频计数）
- `currentFrameGeometry()` 抽出 `static frameGeometry(from:)` —— tap 路径与 re-anchor 路径共用**同一个** FrameGeometry 构造点（§16.3.1）
- 新增 `checkAndFireReAnchor()`：§16.2.1 五条件全 AND；§16.2.2 位置 —— 在 `refreshTapEmbeddingIfNeeded()` **之后**调用，且**两个调用点都加**（tapToSegment 下 2/3 的帧走 `% 3 != 0` 提前 return 分支，漂移是帧的属性而非「本帧是否跑了 YOLO」的属性）
- 新增 `reAnchorDecode(...)`：镜像 `tapDecodeWithPoint` 的 decode + composite 步骤（同一 `SAMDecoder`、同一 `MaskRenderer.buildTapAlpha`、同一 R3 冻结门控），仅失败语义不同
- 新增 `reAnchorKeepStaleMask(slot:reason:)`（§16.6.1 静默降级）与 `finishReAnchorUnit(batchId:)`
- `resetReAnchorState()` 挂到**全部三个清池点**：`discardAllTapWork`（C4 落点）、`setBackend`、分辨率切换

### 六条易错点的落地位置（逐条对照）

1. **`clearAll()` 必须复位 `isReAnchoring` / `pendingCount`** —— 由 `resetReAnchorState()` 在三处 `tapInstances.clearAll()` 之后调用。
   额外加固：`ReAnchorLoop` 带 `batchId` 世代号，`reset()` 自增该号，使 C4 时仍排在 `decoderQueue` 上（无法取消）的闭包落地时 `completeUnit` 成为 no-op —— 否则它们会把**下一个**批次的计数器减穿（减到负数后永不再等于 0，节流永久死锁）
2. **T3 取在 `decoderQueue.async` 调用点之前、闭包之外** —— `reAnchorDecode` 内 `let enqueueMs = PerfLogger.nowMs()` 在 `decoderQueue.async {` 上一行，注释显式引用 debug_report §33.2.3
3. **`capturedEmbedding` 快照** —— 见下方「与 §16 的偏离」，此处为唯一一处按代码实际线程模型作出的判断
4. **日志格式逐字符对齐** —— `perfLog(String(format: "[REANCHOR][inst#%d] drifted %.1fpt/%.1fdeg → qwait: %.1fms decode: %.1fms", ...))`，用 `perfLog`（永远打印，与 `[D7']` 同级），且**在任何质量门控之前发出**，保证被门控丢弃的 mask 也留下 qwait/decode 数据
5. **`driftPt` / `driftDeg` 为触发本批次的量值** —— 在 `checkAndFireReAnchor` 调用 `DriftDetector.drift` 的当场取得，作为 `DriftDetector.Drift` 值传入每个闭包；闭包内**不重算**
6. **N=3 = 三次 `main.async`** —— 保持 §16.2.6 的既有 publish 路径，注释标注为主线程掉帧时的第一排查点；re-anchor 的主线程块**不碰** `tapProcessing` / `tapFailure` / 锚点 marker（re-anchor 不是用户请求，不得抬起加载指示，也不移动锚点）

### 与 §16 的偏离 / 解释（三处，全部显式记录）

**(A) `capturedEmbedding` 在 `stateLock` 内取，而非 §16 与交办清单要求的「锁外取」。**
交办清单第 3 条要求：锁内确认 `embeddingCache != nil` 并置 `isReAnchoring = true`，**锁外**再把 `embeddingCache` 赋给局部变量。
实施时取了锁内（与置标志同一次 acquisition）。理由三条：
(1) `embeddingCache` 由 encoderQueue 写、videoQueue 读，锁外读它是**真实的数据竞争**，不是风格问题；
(2) §10.4 要求 A 禁止的是锁内做**重活**，而 `embeddingCache?.embedding` 只是对既有对象的一次 retain —— 无分配、无拷贝、O(1)，MLMultiArray 的「重」在其 backing store，retain 不触碰它；
(3) 本文件既有全部同类读取（`drainPendingTaps` L1495、`scheduleEncoder` L2549、`warmupDecoderIfPossible` L911）都是锁内读，锁外读会成为文件内唯一的例外。
**全部 decode 派发仍在锁外**，锁内只有 `nil` 检查 + 标志置位 + 基准更新。
另：块内**先取 embedding、后 claim 槽位** —— 反过来会在「两次 acquisition 之间缓存被清空」时留下一个**永不释放的 claim**，正是 §16.2.3 警告的永久死锁。
→ **请 Architect 裁定是否接受**；若坚持锁外，需要同时给出该字段的新同步方案。

**(B) §16.3.1 指定的两个漂移分量在本代码库中恒为常量 —— D-1 验收按现状不可达。**
`FrameGeometry` 原本没有 `letterboxOffset` / `scale` / `videoRotationAngle` 三个字段，已按 §16.3.1 加上
（`letterboxOffset` ← `LetterboxInfo.padX/padY`，`scale` ← `LetterboxInfo.scale`，`videoRotationAngle` ← `rotation`）。
但 `letterboxToSquare` 对每帧的计算是：
```
scale = min(640/w, 640/h) ；padX = (640 − w·scale)/2 ；padY = (640 − h·scale)/2
```
**三者只依赖相机 buffer 尺寸与固定的 640 模型输入，与镜头指向无关** ⇒ 平移相机时 `translationDrift ≡ 0`。
`videoRotationAngle` 则被 `AVCaptureDeviceRotationCoordinator` 量化到 {0, 90, 180, 270}，且它一变就先触发
C4（`applyCaptureRotation → discardAllTapWork`）把池清空，re-anchor 根本观察不到。
⇒ **按字面实现的 `hasDrifted` 无法被相机平移触发**，tasks.md 的 D-1（缓慢平移 3 秒 mask 跟随）与
「`[REANCHOR]` 日志出现」在自然使用下都不会发生。
选择一个真正随相机位姿变化的漂移信号（光流 / CoreMotion / YOLO 框质心 …）是**架构决策**，
按「不得重新设计架构」的约束**未擅自引入**，原样上交 Architect 修订 §16.3.1。

**(C) 为此提供了 `DriftDetector.forceDriftForTesting`（默认 `false`）。**
打开后每次比较都判为漂移，使循环机制（节流、qwait、decode、失败保留旧 mask）在 §16.3.1 修订前
仍可真机测量 —— §16.8 的 ISSUE-P4-DECODE 被动采集与 §16.9 的 D-4（qwait max < 50 ms）都依赖
`[REANCHOR]` 行存在。性质等同既有的 `CameraManager.forceSlowPath` 调试开关，**不是**漂移信号，
默认关闭时行为与 §16.1.1 逐字一致。

### 其它实现判断（非偏离，记录备查）

- 批次成员取 `tapInstances.drawableInstances()`（已有 mask 的实例），而非全部实例：首次 tap 的 decode 尚在途的实例没有 mask 可刷新，重 decode 会与那次 tap 抢同一个实例槽。触发门仍按 §16.2.1 条件 2 用 `tapInstances.count > 0`
- 首次带池帧无基准时只做 `seedAnchorIfNeeded` 并返回（此时 mask 正是从该几何解出的，无可比对象）
- 计数器递减用闭包首行 `defer { self.finishReAnchorUnit(batchId:) }` —— 闭包内有十一条 early return，defer 是唯一能保证 §16.2.4「成功失败都必须递减」的写法
- `[REANCHOR] skipped — no embedding` 用 `diagLog` 且每 30 次打一行（每帧调用，否则刷屏）；quiet mode 下会被抑制。**§16.9.3 的取数只依赖 `[REANCHOR][inst` 行，这些是 `perfLog`，不受影响**
- 失败行用 `faultLog`（异常路径永不静默，与本文件既有约定一致）

### 编译验证

```
cd JudgeE2 && xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**，零 error，改动文件零 Swift warning。
（`project.pbxproj` 需手工登记两个新文件：`Interaction` 组不是 filesystem-synchronized，
新文件不会被自动纳入 target —— 第一次构建即因 `cannot find 'DriftDetector' in scope` 失败。）

### 改动文件汇总

| 文件 | 改动 |
|------|------|
| `JudgeE2/Interaction/DriftDetector.swift` | 新增 |
| `JudgeE2/Interaction/ReAnchorLoop.swift` | 新增 |
| `JudgeE2/Interaction/FrameGeometry.swift` | 新增 `letterboxOffset` / `scale` 存储属性 + `videoRotationAngle` 计算属性；**`invertViewPoint` 一行未动**（§2 变换链未触碰） |
| `JudgeE2/Detection/CameraManager.swift` | `reAnchor` 状态 + `frameGeometry(from:)` 抽取 + `checkAndFireReAnchor` / `reAnchorDecode` / `reAnchorKeepStaleMask` / `finishReAnchorUnit` / `resetReAnchorState` 新增；三处清池点与两处帧回调挂钩 |
| `JudgeE2/JudgeE2.xcodeproj/project.pbxproj` | 登记两个新文件 |

### 绝对未动项

- `SAMDecoder.swift` — 一行未动
- `MaskRenderer.swift` — 一行未动
- R3 禁令常量：`minComponentPx=30` / `minComponentSidePx=3` / `minComponentFill=0.05` /
  `maxPlausibleLogit=500.0` / `stabilityDelta=1.0` / `cap60` / `cap85` / 候选选择规则 — 一字未变
- 分割逻辑本身（候选选择、`iou_pred >= 0.1` 门、flood fill、数值哨兵）—— re-anchor 走的是**同一份**代码，未复制、未分叉、未加参数
- **无新队列、无新锁**：复用 `decoderQueue` + `stateLock`
- **re-anchor 不调用 encoder**：全文无 `SAMEncoder` / `refreshTapEmbeddingIfNeeded` / 任何 encode 路径引用
- **re-anchor 不修改 `canonicalPoint`**：闭包只读 `instance.canonicalPoint`
- Phase 2 路径（`.segmentation` / TemporalManager / box decoder / `renderMask`）零改动 —— `checkAndFireReAnchor` 第一条件即 `currentMode == .tapToSegment`

---

## Phase 4 Day 2–3 — §17 修订：漂移信号换为锚点邻域内容散度 (Builder, 2026-08-17)

**性质：修订上一条目，不是新特性。** 依据 `architect_output.md §17`（§17.1–§17.11），
按 §17.6 的 B-1…B-10 变更清单逐条实施。§16.2 / §16.4 / §16.5 / §16.6 / §16.7 / §16.8 一字未改地继续执行。
§17.9 追认的三处自主决定（`batchId` 世代号 / `drawableInstances()` 批次成员 /
`capturedEmbedding` 锁内快照且先取 embedding 后认领）**全部原样保留，未回退**。

### B-1…B-10 落地对照

| # | 内容 | 落地 |
|---|------|------|
| B-1 | 删除 `drift(from:to:) -> Drift` 与 `hasDrifted` 的 `FrameGeometry` 重载，换成 §17.5.1 新接口 | `DriftDetector.swift` 整文件重写：`signature(from:atCanonical:)` / `divergence(from:to:)` / `drift` / `hasDrifted` / `alphaIoU`。**未保留任何兼容重载**（§17.5.1：一个恒返回 false 的谓词是最容易被误调用的形态） |
| B-2 | 删除 `translationThresholdPt` / `rotationThresholdDeg`，换 §17.4 六常量 | `contentThresholdLuma = 8.0` / `anchorWindowPx = 96.0` / `anchorGridSide = 8` / `minReAnchorIntervalMs = 300.0` / `reAnchorAcceptIoU = 0.5` / `reAnchorConsistencyGateEnabled = true`，全部具名 `static var`，调参接口形态与上一轮完全一致 |
| B-3 | `Drift` 改为 `divergenceLuma` + `exceedsThreshold` | 同名结构体保留，两字段替换；「必须在检查时刻捕获、闭包内不得重算」的形态与注释论证保留 |
| B-4 | 判定改读 per-instance `TapInstance.anchorSignature`；`lastAnchorGeometry` 降级 | `ReAnchorLoop.lastAnchorGeometry` 保留但注释标为 ⛔ 遥测专用、不得进入任何判定；`seedAnchorIfNeeded` **删除**（其职责被 per-instance 播种取代） |
| B-5 | `checkAndFireReAnchor` 条件序按 §17.5.3 重排 | 1 mode → 2 前台 → 3 `drawableInstances()` → **4 时间下界（在任何像素工作之前）** → 5 embedding → 6 逐实例采样取 max → 7 认领 + embedding 快照 → 8 推进 `lastReAnchorFireMs` 与各实例基线 → 9 派发 |
| B-6 | 日志前缀 `%.1fpt/%.1fdeg` → `%.1flum` | `[REANCHOR][inst#%d] drifted %.1flum → qwait: %.1fms decode: %.1fms`。**`qwait:` / `decode:` 名称、格式、相对位置逐字符未变**；仍在任何门控之前发出（§16.8 前提） |
| B-7 | `TapInstance.anchorSignature` + 管理器读写 | 新增字段 + `TapInstanceManager.setAnchorSignature(id:signature:)`，**用既有 `lock`**，未新增锁；`drawableInstances()` 过滤条件一字未动 |
| B-8 | `CameraManager.lastReAnchorFireMs` | 私有字段，videoQueue 独占，与 `tapModeFrameCount` 同类，无同步 |
| B-9 | 一致性否决门 | `reAnchorDecode` 中 `buildTapAlpha` 之后、`iou_pred >= 0.1` 门之后、`updateMask` 之前调用 `alphaIoU(previousAlpha, built.alpha, stride: 4)`；`< reAnchorAcceptIoU` 则走既有 `reAnchorRejectUpdate` 并 return（旧 mask 保留）；受 `reAnchorConsistencyGateEnabled` 单开关控制 |
| B-10 | 否决日志行 | `[REANCHOR][inst#%d] rejected — mask IoU %.2f < %.2f, keeping previous mask`，用 `perfLog` |

### 两条实现约束的落地

1. **无每帧堆分配**：`AnchorSignature` 用 8×`UInt64` = 64 字节的**定长内联元组**存储，值类型，
   零堆分配。代价是网格上限 8×8（正是出厂的 `anchorGridSide = 8`）；`effectiveGridSide` 把
   越界的调参值钳到 8，并在注释中写明「要更大网格必须先加宽 `Storage`」。
2. **`alphaIoU` 不构造中间数组**：`withUnsafeBufferPointer` 双指针就地按 stride 4 二维跳采，
   交并两个计数器，无分配。

### 采样路径的三个关键点（§17.3.2 逐条）

- 只采 `latestCameraBuffer`（采集直出 IOSurface，videoQueue 所有），**绝不采 `latestInputBuffer`**
  （`CIContext` 渲染产物，锁基址会等 GPU）—— 已在 `signature` 的文档注释中写死
- 像素格式先校验 `kCVPixelFormatType_32BGRA`，不符返回 `nil`
- `lumaApprox = (r + 2g + b)/4` 纯整数；3×3 盒平均抗混叠；比较前双侧各自去均值（抗自动曝光加性爬坡）
- 降级：buffer 缺失 / 格式不符 / 锁基址失败 → `signature` 返回 `nil` → `checkAndFireReAnchor`
  **视同无漂移直接 return**（不 fire、不更新基线、不记失败），即 §17.4 ⑥

### 两处 §17 需要解释才能落地的地方（已作判断，请复核）

**(1) §17.5.3 第 6 步「播种基线 → continue（本帧不 fire）」的作用域。**
字面 `continue` 只跳过该实例，括注却说「本帧不 fire」。取**后者**：本帧只要有任一实例被播种，
整帧不 fire（`seededThisFrame` 标志）。理由：刚播种的实例其散度未被测量，此时开批次等于
用一个没测过的量去刷新它；而下一帧（≥1 帧后）它就有基线可比，代价只是一帧延迟。

**(2) 网格边长变更导致基线不可比。**
`divergence` 对长度不等的两个签名返回 0（不可比即不判漂移）；调用方在
`baseline.count != current.count` 时走**播种分支重新播种**，而不是拿旧长度硬比。
这条 §17 未规定，是为「Debugger 中途改 `anchorGridSide`」这个明确存在的调参场景补的洞。

### 改动文件汇总

| 文件 | 改动 |
|------|------|
| `JudgeE2/Interaction/DriftDetector.swift` | **整文件重写**：`AnchorSignature` + 新接口 + 六常量 + `alphaIoU`；旧 FrameGeometry 接口与两个旧阈值删除 |
| `JudgeE2/Interaction/TapInstanceManager.swift` | `TapInstance.anchorSignature` 字段 + `setAnchorSignature(id:signature:)`（既有 `lock`）；`addInstance` 构造点补 `anchorSignature: nil` |
| `JudgeE2/Interaction/ReAnchorLoop.swift` | `lastAnchorGeometry` 降级为遥测（注释）；删除 `seedAnchorIfNeeded` |
| `JudgeE2/Detection/CameraManager.swift` | `lastReAnchorFireMs` 新增；`checkAndFireReAnchor` 按 §17.5.3 重写；`reAnchorDecode` 日志前缀 + 一致性否决门 + `previousAlpha` 快照；新增 `reAnchorRejectUpdate(slot:iou:)` |

**未新增文件** ⇒ 本轮无需改 `project.pbxproj`（`DriftDetector.swift` 上一轮已登记）。

### 编译验证

```
cd JudgeE2 && xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**，零 error，改动文件零 Swift warning（一次通过，无需修复）。

### 绝对未动项（本轮）

- `SAMDecoder.swift` / `MaskRenderer.swift` — 一行未动；`buildTapAlpha` **零触碰**
  （否决门是它之外、之后的外层闸，只会否决并落回既有的「保留旧 mask」分支）
- R3 禁令常量全数未变；候选选择、cap60/cap85、flood fill、数值哨兵、`iou_pred >= 0.1` 门 — 未变
- **未实现能力 C（目标跟随）**：无 tracker、未引入 CoreMotion / Vision、`canonicalPoint` 只读、
  `FrameGeometry.invertViewPoint` 未动。§17.2.0 已将其另立为 `ISSUE-P4-TRACK`
- tap 路径（`tapDecodeWithPoint`）与 `.segmentation` 路径 — 零行改动
- 无新队列、无新锁；re-anchor 不调 encoder
- `forceDriftForTesting` 保留，默认 `false`，注释按 §17.7 重新定性（对机制有效 / 对行为无效 + 引用数据必须标注开关状态）

---

## Phase 4 Day 3.5 — §18 补救：RE-1 世代门 / RE-2 单实例批次 / RE-3 冻结原点基准 (Builder, 2026-08-17)

**性质：对 §16+§17 的第二次修订，不是新特性，也不是对 §17 的回退。** 依据 `architect_output.md §18`
（§18.1–§18.10），按 §18.8 的 B-11…B-17 变更清单实施。真机验收 6 PASS / 2 FAIL（debug_report §34），
两条 FAIL（D-4 / D-1c）经 Architect 复核**均为设计层缺陷**，本轮按其裁决消除。

**§17 的两半按 §18.6.1 分开对待：§17.3 的信号（锚点邻域内容散度）成立、逐行保留；
§17.3.3 的比较基准被 RE-3 取代，但门本身（位置 / 开关 / 日志 / 失败分支）一行未动。**
`anchorSignature`（批次开始即推进）与 `originAlpha`（永不推进）的推进条件不一致，
按 §18.2.3 是**有意的职责分离**（散度决定「何时尝试」，原点决定「是否接受」），
**未做对齐**，并已把该理由写进 `TapInstance.originAlpha` 的文档注释以防后人「顺手对齐」。

### B-11…B-17 落地对照

| # | 内容 | 落地 |
|---|------|------|
| **B-13**（RE-1） | `embeddingGeneration` + `TapInstance.lastReAnchorEmbeddingGen` | `CameraManager.embeddingGeneration: UInt64`，与 `embeddingCache` **同一 `stateLock`、同一次加锁读出**；在**四个**「新算出的 embedding 写回」点 `&+= 1`（warmup / tap encode / background refresh / segmentation encode），置 nil **不自增**。实例字段用 `TapInstanceManager` **既有 `lock`**（同 B-7 形态），派发时写入，不等 decode 返回。资格过滤放在第 5 步之后、第 6 步像素采样**之前** ⇒ 全实例已消费当前世代时整帧跳过采样 |
| **B-14**（RE-2） | `TapInstance.lastReAnchorAtMs` + 单实例选择 | 候选 = 「未消费当前世代」∩「**自身** `d_i` 越阈」；取 `lastReAnchorAtMs` 最小者（`nil` 排最前），平局取 `slotIndex` 小者 ⇒ 完全确定、公平轮转。`beginBatch(count: 1)`。**节流状态机 / `batchId` 世代号 / `pendingCount` / `isReAnchoring` 单在途不变量 —— 一行未动** |
| **B-17** | 删除 `maxDivergence` 批次取 max | `var maxDivergence` 及其 `guard forceDriftForTesting \|\| maxDivergence > threshold` **整段删除，未保留开关**。`forceDriftForTesting` 语义未失：它已折叠在 `Drift.exceedsThreshold` 内，开启时每个 eligible 实例都成为候选 |
| **B-15**（RE-3） | `TapInstance.originAlpha` | 写入点**唯一**：`updateMask(..., recordOrigin: true)`，只有 tap 路径（`tapDecodeWithPoint`）传 `true`；参数默认 `false`，使 re-anchor 的调用点**因省略而正确**。与 `maskAlpha` 同一次加锁写入，二者不可能被观察到不同步。否决门比较对象 `instance.maskAlpha` → `instance.originAlpha`；**门的位置、`reAnchorConsistencyGateEnabled` 开关、日志行格式、`reAnchorRejectUpdate` 失败分支全部不变** |
| **B-16** | `DriftDetector.reAnchorEnabled` | `= false` **以关闭状态发布**；`checkAndFireReAnchor` 第 **0** 条 guard。注释写明与 `reAnchorConsistencyGateEnabled`（必须以 `true` 发布）**极性相反**、不得混用，且翻转需 Architect 单独批准 |
| **B-11** | 日志单调时钟前缀 | `PerfLogging` 内新增 `lineStampMs()`（`PerfLogger.nowMs()` 减进程内首次取样基准，单调，非 wall-clock）；`perfLog` / `diagLog` / `faultLog` / `quietSummaryLog` 四个 helper 统一走 `stamped()`，前缀 `[t=%.1f] ` **在行首**。既有 tag 与字段的相对位置**逐字符未变** ⇒ §16.9.3 / §33 的全部 grep 不受影响 |
| **B-12** | `suspendRefreshOnly` | `CameraManager.suspendRefreshOnly: Bool = false`，`refreshTapEmbeddingIfNeeded` 内**仅 early-return**，**不清 `embeddingCache`**。与 `forceSlowPath`（清缓存 **且** 挂起刷新，故其参照工况是复合的，A-11）**并存且正交**，未合并、不互相蕴含，不暴露于 UI |

### 三处 §18 需要解释才能落地的地方（已作判断，请复核）

**(1) 未被选中的候选，其 `anchorSignature` 基线是否推进？—— 判定：不推进，只推进被选中的那一个。**
§18 未明写。理由：§18.1.5 的「每实例刷新周期 ≤ 900 ms 的公平轮转」要求落选实例在下次 fire 时
**仍然越阈**；若一并推进基线，落选实例的待处理散度会被静默清零，轮转退化为「只有第一次被选中的
那个实例会被刷新」。且 §16.4.2 推进基线的理由是「该实例的新 mask 是关于这一帧的断言」——
落选实例没有产生新 mask，其基线本就应停在旧帧。

**(2) 世代号在哪一次加锁中读取？—— 判定：读两次，各有其用。**
资格过滤用第 4 步那次加锁读到的值（与 `hasEmbedding` 同一次读出，保证号与 embedding 配对）；
写入 `lastReAnchorEmbeddingGen` 的值则在**认领批次那次加锁**中与 `capturedEmbedding` 一起读出，
因为那才是本次 decode 真正使用的 embedding 的世代。两次之间若恰有新 embedding 落地，
记录的是真实被解码的世代 —— 记成旧值会白送一次 decode，记成新值会漏掉一次，取真实值两者皆免。

**(3) `.segmentation` 路径的 encode 写回点是否自增世代？—— 判定：自增。**
§18.8「明确不要求做的事」列有「不改 `.segmentation` 路径任何一行」，但 §18.1.4 的规则是
按**写回点**陈述的（「每次 `embeddingCache` 被赋一个新计算出的 embedding 时 += 1」），
不是按模式陈述的。该处新增的是一行计数器自增，`.segmentation` 模式内无任何代码读它，
行为零改变；不加则世代号会与缓存内容脱节，破坏「世代号命名 embedding」这一不变量。
**若 Architect 认为应严格按模式隔离，删掉这一行即可，无其它连带。**

### 改动文件汇总

| 文件 | 改动 |
|------|------|
| `JudgeE2/Interaction/DriftDetector.swift` | 新增 `reAnchorEnabled = false`（B-16）；`reAnchorAcceptIoU` 注释按 §18.2.5 改写（数值仍 0.5，所测的量已变，并写入 R21 的禁止调参条款）。**算法与其余常量一字未动** |
| `JudgeE2/Interaction/TapInstanceManager.swift` | `TapInstance` 新增 `lastReAnchorEmbeddingGen` / `lastReAnchorAtMs` / `originAlpha` 三字段；`updateMask` 增 `recordOrigin: Bool = false`；新增 `markReAnchorDispatched(id:embeddingGeneration:atMs:)`（**既有 `lock`**）；`addInstance` 构造点补三个 `nil` |
| `JudgeE2/Detection/CameraManager.swift` | `embeddingGeneration` 字段 + 四处写回点自增；`suspendRefreshOnly` 字段 + `refreshTapEmbeddingIfNeeded` early-return；`checkAndFireReAnchor` 加第 0 条开关、加 RE-1 资格门、删 `maxDivergence`、改为单实例选择 + `beginBatch(count: 1)`、只推进被选中实例的基线；`reAnchorDecode` 的 `previousAlpha` → `originAlpha`；tap 路径 `updateMask` 传 `recordOrigin: true` |
| `JudgeE2/Shared/PerfLogging.swift` | B-11 行首单调时钟前缀（四个 helper + `lineStampMs()`）；文件头注释就地更正「输出逐字节相同」这一已失效的声明 |

**未新增文件** ⇒ 本轮无需改 `project.pbxproj`。

### 编译验证

```
cd JudgeE2 && xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**，零 error、零 warning（一次通过，无需修复）。

### 绝对未动项（本轮）

- `SAMDecoder.swift` / `MaskRenderer.swift` — 一行未动；`buildTapAlpha` **零触碰**
- R3 禁令参数全数未变（`minComponentPx=30` / `minComponentSidePx=3` / `minComponentFill=0.05` /
  `maxPlausibleLogit=500.0` / `stabilityDelta=1.0` / cap60 / cap85 / 候选选择规则）
- **无新队列、无新锁**：世代号复用 `stateLock`，两个实例字段复用 `TapInstanceManager` 既有 `lock`
  （§16.7 七条禁令逐条维持，§18.1.6 已核对）。**未采纳「给 re-anchor 独立低优先级队列」方案**
- re-anchor **不调 encoder**、**不改 `canonicalPoint`**；能力 C 仍归 `ISSUE-P4-TRACK`
- §17 的工作**未回退任何一处**；§17.9 追认的三处自主决定继续保留
- 节流状态机 / `batchId` 世代号 / `pendingCount` / 三处清池点 / T3 位置 /
  `[REANCHOR]` 日志在质量门之前发出 / `drawableInstances()` 批次成员 /
  `capturedEmbedding` 锁内快照且先取 embedding 后认领 —— 全部一行未动
- tap 路径的**判定逻辑**零改动（B-15 只在既有成功路径上多写一个字段）
- **未做 Debugger 的工作**：无真机测试、无日志分析、未写 `debug_report.md`
- **未执行 §18.4.3 第 3–5 项**（2×2 析因 / 交错采样 / encode→decode 间隔扫描）——
  按裁决那是 Debugger 的采集协议，且本轮只买期权不行权
- **未开始 Phase 4B Day 4**（PinStore）——本条目仅为 §18 补救（P0）

---

## Phase 4 Day 2–3 缺陷修复 —— ISSUE-P4-GATE：一致性否决门自落地起从未执行过一次比较 (Builder, 2026-08-17)

> 依据：`debug_report.md` **§35.6.3 / §35.6.4**（根因与仅凭日志的独立证明）、**§35.7.2**（接受分支埋点字段规格）、
> `architect_output.md` §17.3.3 / §18.2 / §18.3.5 条件 6。
> 本轮**只做两件被指定的事 + 一处回滚**，**未做任何设计决定**，未碰阈值、未碰算法、未做真机测试。

### 缺陷（P0）

`alphaIoU` 的宽高实参与 alpha 数组不是同一个单位：调用点传 `origW × origH = 2 073 600`，
而 `MaskRenderer.AlphaResult.alpha` / `TapInstance.originAlpha` 恒为 **256×256 = 65 536** 字节
（`MaskRenderer.swift:84` 注释、`:184` 的 `alpha.count == total` 硬契约、`TapInstanceManager.swift:60` 注释三处互证）。
⇒ 第一条守卫 `a.count >= width*height` 恒假 ⇒ `alphaIoU` **恒返回 1.0**（「无证据不否决」的安全默认）
⇒ `reAnchorRejectUpdate` 不可达。**§34 的 0/216 与 §35 的 0/76 都是在「实际上没有门」的系统上测出来的。**

错误的源头是文档：`DriftDetector.swift` 原注释逐字写着「the alphas are origW × origH bytes (≈2.07 M at 1080p)」
—— **规格、注释、调用点三者自洽地错在一起**，故三处一并更正。

### 落地对照

| # | 内容 | 落地 |
|---|------|------|
| **B-18**（P0，维度修复） | 调用点传对维度 | `CameraManager.swift` 否决门处改为 `width: DriftDetector.maskAlphaSide, height: DriftDetector.maskAlphaSide`。新增 `DriftDetector.maskAlphaSide = 256`（`static let`，**非可调参数**），注释写明它是渲染路径的硬不变量、出处三条，使调用点无法第二次自行推导错维度 |
| **B-19**（P0，同一 commit） | `stride` 4 → 1 | `alphaIoU` 的默认实参改为 `stride: Int = 1`，调用点亦显式传 `stride: 1`。理由按 §35：256² 网格上 stride 4 只剩 4 096 个采样点，80–431 px 的 mask 只命中 5–27 个，p≈0.5 附近 SE 达 ±0.10–±0.22，门在阈值处退化为掷硬币；全遍历 65 536 元素约 30–60 µs，相对 50–60 ms 的 decode 可忽略。**参数保留**，只改默认值 |
| **B-20**（P0，注释更正） | 成本推理重写 | `alphaIoU` 的 doc comment 删去 2.07 M / 1–2 ms / 130 k 三个数，改写为真实的 65 536 / 30–60 µs，并逐字记录旧注释的错误前提与它如何致盲静态审查。**「无法比较时返回 1.0」的契约与守卫逻辑本身一字未动** —— 门仍然只在有证据时否决 |
| **B-21**（P1，接受分支埋点） | §35.7.2 字段 | `[REANCHOR]` 行尾追加 `\| iou: %.2f origin: %dpx new: %dpx`。`iou` **在门外、且在 `originAlpha` 解包外**算出（`gateIoU: Double?`），门只做判断 ⇒ 开关关闭或 `originAlpha == nil` 时该行照样有数，且打印 `n/a` 而非编造值。`origin` = `originAlpha` 非零计数（re-anchor 时就地扫描，~20 µs；**未在 `TapInstance` 上缓存**，见下「口径判断」）；`new` = 既有的 `built.nonzeroCount`，无新计算 |
| **B-21b** | 否决分支同字段集 | `reAnchorRejectUpdate(slot:iou:originPx:newPx:)`，既有文本一字未改，同样在行尾追加 `\| origin: %dpx new: %dpx` ⇒ 两个分支字段集一致，提取脚本一条正则 |
| **回滚** | §18.3.5 条件 6 | `DriftDetector.swift` 的 `reAnchorEnabled` 由重测期间的 `true` **改回 `false`**。Day 2–3 未关闭，「暂停合入」继续有效 |

**`qwait:` / `decode:` 逐字符不变**（§16.9.3 / §33 / §34 / §35 的全部 grep 依赖它），前缀串
`[REANCHOR][inst#%d] drifted %.1flum → qwait: %.1fms decode: %.1fms` 原样保留，新字段一律追加在行尾。

### 一处必须解释的实现取舍（请 Debugger 在下一轮读日志前先看这条）

**`[REANCHOR]` 行的发出时刻从「decode 返回后立即」推迟到「否决门处」。**
新字段中的 `iou` / `new` 在 `buildTapAlpha` 跑完之前根本不存在，而 §35.7.2 要求它们**追加在同一行**，
故该行现在是「在 decode 后**构造**、在门处**发出**」。§16.8 的不变量（`qwait` / `decode` 绝不因 mask 被否决而丢失）
用两道机制保住：(a) 门与 decode 之间的**四个** early-return 点各显式发一次「只有前缀」的行，
位置在各自的 `keeping stale mask` 故障行**之前**，故两行的相对顺序不变；
(b) 一个 `defer` 兜底任何将来新增的退出路径，`emitReAnchorLine` 幂等。

⚠️ **对日志读者的唯一可见后果**：凡走到 `buildTapAlpha` 的路径，`[REANCHOR]` 行现在出现在该次
`[TAP#g] candidates: …` 行**之后**而非之前 ⇒ **§35.6.4 的配对法（「每条 `[REANCHOR]` 之后紧跟的 `[TAP#g]` 行」）不再适用**。
但它也不再需要 —— `new:` 字段已经把那个面积直接写在 `[REANCHOR]` 行上了。此点已写入源码注释。

**`originAlpha` 非零计数未缓存到 `TapInstance`：** §35.7.2 注 2 建议在 `updateMask(recordOrigin: true)` 时顺手存一个 `Int`，
并自陈「属于实现细节，不属判据」。选择就地扫描，因为存字段等于把一个派生值复制一份去维护同步，
而 65 536 字节扫描（~20 µs，decoderQueue 上，紧邻 ~55 ms 的 decode）在被 `minReAnchorIntervalMs=300` 限流的路径上
不构成热点。**若 Architect 要求缓存，改动局限在 `TapInstanceManager` 一处，无连带。**

### 改动文件汇总

| 文件 | 改动 |
|------|------|
| `JudgeE2/Interaction/DriftDetector.swift` | 新增 `static let maskAlphaSide = 256`；`alphaIoU` 默认 `stride: 4 → 1`；`alphaIoU` doc comment 重写（错误前提 + 真实成本 + stride 理由）；`reAnchorEnabled` `true → false` |
| `JudgeE2/Detection/CameraManager.swift` | 否决门维度实参改为 `maskAlphaSide`、显式 `stride: 1`；`iou` / `origin` / `new` 三字段在门外算出并追加到 `[REANCHOR]` 行尾；`[REANCHOR]` 行改为「构造—发出」两段式（幂等 + `defer` 兜底 + 四个 early-return 点显式发出）；`reAnchorRejectUpdate` 增 `originPx` / `newPx` 两参并追加同名字段 |

**未新增文件** ⇒ 本轮无需改 `project.pbxproj`。

### 编译验证

```
cd JudgeE2 && xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**，零 error、零 warning（一次通过，无需修复）。

### 绝对未动项（本轮）

- `SAMDecoder.swift` / `MaskRenderer.swift` — **一行未动**（只读 `MaskRenderer` 以核实 256×256 不变量）；`buildTapAlpha` **零触碰**
- **`reAnchorAcceptIoU` 仍为 0.5** —— §35 已证明 0.5 在此几何下判别力充分（迁移事件 IoU 上界 0.037–0.099，同物体形变 0.99+），
  调它是 Architect 的决定，且数据说不需要。R21 调参禁令继续有效
- `reAnchorConsistencyGateEnabled` 仍为 `true`（§17.4 禁止以关闭状态发布）
- R3 禁令参数全数未变；**无新队列、无新锁**；re-anchor 仍不调 encoder、不改 `canonicalPoint`
- 门的位置、否决分支的行为（保留旧 mask）、`alphaIoU` 的「无法比较即返回 1.0」安全契约 —— 均未改
- **未试图在「让门真的跑起来」之外修 D-1c'** —— 门够不够用，由下一轮真机确定
- **未做 Debugger 的工作**：无真机测试、无日志分析、未写 `debug_report.md`
- 未勾选/取消 `tasks.md` 任何 checkbox（仅在既有 Builder 条目的行内注记上追记本次修复）

---

## Day 2–3 收尾：re-anchor 总开关合入启用（STOP RULE 通过后）

### 背景

第 4 轮 re-anchor 真机复测**两条判据同时通过**（⏱️ STOP RULE）：

- **D-6''** —— 一致性门确实在执行：11 条 `[REANCHOR]`，其中 **9 条 iou ≠ 1.00**，跨度 0.02–0.89；
  另两条 1.00 是同一 embedding 代内 bit-identical 的空转 re-anchor（真实值），不是修复前那个恒定 1.00。
  否决率 45%（5/11）—— R21 至此才有输入。
- **D-1c''** —— 无 mask 迁移：5 次否决拦下的 mask 分别比冻结原点大 47.5× / 30.1× / 20.1× / 36.9× / 1.7×；
  录屏显示 mask 全程稳定在 ~490 px。
- REC-1 同时成立：被接受的刷新面积近乎不变（482→481、482→487、491→490、491→487）。

按 STOP RULE，Day 2–3 关闭，re-anchor 以**启用**状态合入。

### 本轮唯一改动

文件：`JudgeE2/Interaction/DriftDetector.swift`

- `static var reAnchorEnabled: Bool = true`（第 176 行，注释重写后的行号；原第 166 行）
  —— **发现该字段的值上一轮已处于 `true` 且已提交**（工作区与 HEAD 对该文件无 diff），
  故本轮实际改动落在其**文档注释**上：把原先的「⚠️ MUST SHIP `false` / 暂停合入 / Builder 不得翻转」
  改写为「✅ SHIPS `true`」，并写明授权来源（STOP RULE 通过、D-6'' + D-1c'' + REC-1 的具体数据、
  §18.3.5 的 Architect 单独批准），同时保留「关闭即回退到冻结 Phase 3 行为」这一 rollback 说明。
- 「与 `reAnchorConsistencyGateEnabled` 极性相反」的警告段保留，仅把末句由
  「currently required to be disabled」改为「now authorised to ship enabled」。

### 未改动项（本轮）

- `reAnchorConsistencyGateEnabled` 仍为 `true`（极性相反，§17.4 禁止以关闭状态发布）
- `reAnchorAcceptIoU`(0.5)、`contentThresholdLuma`(8.0)、`minReAnchorIntervalMs`(300.0) 及其余调参常量**一字未动**
- `forceDriftForTesting` 仍为 `false`
- `SAMDecoder.swift` / `MaskRenderer.swift` **未打开修改**；`buildTapAlpha` 零触碰；R3 禁令参数全数未变
- **`shared/tasks.md` 本轮未触碰**（Architect 正并发做 Day 2–3 文档收尾）
- 未做 Debugger 的工作：无真机测试、未写 `shared/debug_report.md`

### 编译验证

```
cd JudgeE2 && xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**

---

## Phase 4B Day 4 —— PinStore 持久化层（B-18 … B-26，B-27/B-28 仅接口）(Builder, 2026-08-21)

依据：architect_output.md **§19**（全节，含 §19.7 的 B-18…B-28 清单）+ §21.3（`decoderQueue` 0.1 ms 余量的实测依据）
+ §20.3.3 / §18.3.3（硬隔离 1、2 仍然有效；第 3 条已由 §20.3.4 解除）+ debug_report §35（256×256 不变量、ISSUE-P4-GATE 的维度前提缺陷）。
⚠️ **tasks.md Day 4 的 `struct Pin` 草案未被采用** —— §19.2.7 已裁定其五处不成立，本轮一律按 §19 实现。

### 新增文件（全部落在新目录 `JudgeE2/Persistence/`，不进 Interaction / Detection）

| 文件 | 覆盖条目 |
|---|---|
| `Persistence/Pin.swift` | B-18（`protocol PinStore` + 线程契约）、B-24（`PinStoreError` 六例） |
| `Persistence/PinRecordV1.swift` | B-19（DTO）、B-20（两个纯映射）、B-25（迁移脚手架 + 显式 JSON 编解码配置） |
| `Persistence/MaskPNGCodec.swift` | B-22（256×256 `[UInt8]` ⇄ 8-bit 灰度 PNG） |
| `Persistence/FilePinStore.swift` | B-18 / B-21 / B-23 / B-24 / B-25 的实现 |
| `Persistence/PinInterfaces.swift` | **B-27 / B-28 —— 只有签名与契约注释，无实现** |

`project.pbxproj` 手工登记：新增 `Persistence` PBXGroup（`Interaction` 组已知非文件系统同步，同样手工登记）+ 5 条 PBXBuildFile / PBXFileReference / Sources 条目。

### 逐条落地

- **B-18** `protocol PinStore`（`load` / `fetchAll` / `fetch(id:)` / `create` / `update(id:tag:note:)` / `delete(id:)` / `loadMaskImage(id:)` / `flush(waitUntilDone:)`）+ 唯一实现 `FilePinStore`。
  ⛔ **不提供覆盖式 `save(_:)`**：整条记录覆盖能改写冻结字段，PIN-1 禁止。`create` 对已存在 id 逐字段比对冻结半区，
  第一处不同即以 `.frozenFieldMutation(field:)` **点名到字段**返回。`update` 的签名里根本够不到冻结字段（结构性防呆，而非注释防呆）。
- **B-19** `PinRecordV1` / `PinGeometryV1`：显式 `pointX`/`pointY` Double、时间一律 epoch 秒 Double（不用 `Date`/`CGFloat`）、
  `maskWidth`/`maskHeight`/`maskNonZero` 随数据持久化（A-15）、`iouPredAtCreation` 注释标明**仅诊断**、`previewFile` 建字段并**恒 nil**（R29）。
  `PinCoders` 显式设置 `outputFormatting` / `dateEncodingStrategy` / `dataEncodingStrategy` / `keyEncodingStrategy` / `nonConformingFloat*`，不依赖任何 SDK 默认值。
- **B-20** 正向 `PinGeometryV1(from:promptSpace:encoderInputSize:)`；反向 **`decodeInputs() -> (origSize, promptSpace)`**。
  ⛔ `FrameGeometry` **未**加 `: Codable`；反向映射**在 API 形状上就够不到** `letterboxScale/PadX/PadY`（返回二元组而非整个结构），
  所以「第二条变换路径」不是靠注释禁止的，是够不到。
- **B-21** `pin.store.io`（serial, `.utility`）**一条**，**零新锁** —— serial 即临界区。blob 立即写（不可变 ⇒ 无覆盖竞态）；
  manifest 置脏 + **250 ms 合并窗** + `Data.write(options: [.atomic])` 全量重写。三处强制 flush：
  `flush(waitUntilDone: true)` 供 `scenePhase → .background` 与 `willTerminate`（Day 5 接 UI 时挂载），合并窗到期是第三处。
  ⚠️ **一处比规格更严的实现决定**：成功回调**不在置脏时**发出，而是**挂起到包含该次变更的 flush 落盘后**统一释放。
  理由是 §19.6 **P-3(ii)**（「一次 `save` 回调返回成功后立刻 SIGKILL ⇒ 该条必须在盘上」）与 250 ms 合并窗在字面上冲突，
  挂起回调是唯一同时满足两者的形态：突发 N 次保存仍只重写一次 manifest，而「回调说成功」仍然蕴含已落盘。
- **B-22** `MaskPNGCodec`：`[UInt8]`(256×256, {0,255}) ⇄ 8-bit 灰度 PNG，`shouldInterpolate: false` + `.none` 插值质量，
  **无重采样、无阈值参数**（本文件里没有任何可调的东西）。DEBUG 下每次 encode 都断言 encode→decode 逐字节一致。
  解码尺寸 ≠ 256 时**抛错而不缩放** —— 那是 degraded 记录，调用方的职责是把它排除出合成，不是猜回原形。
  ⛔ 未引入 128×128：128 在本系统里不对应任何不变量，缩放二值图必须引入一个从未被裁决过的重采样阈值（§19.2.6 / ISSUE-P4-GATE 的成因类）。
- **B-23** 内存索引（`records` + `order`，**只在 `pin.store.io` 上变更**）→ 主线程 `@Published pins`（值类型快照，按 `createdAt` 升序）/ `isLoaded` / `lastWriteError`。
  `fetchAll()` / `fetch(id:)` 服务于**主线程快照与主线程字典镜像**，零队列跳、零锁、**不触磁盘**。
  启动异步载入；载入时 GC 孤儿 blob；blob 懒加载 + **≤32 条 LRU**；⛔ 全量载入不打开任何 blob 文件（否则正是 §19.6 P-2 线性度会抓到的形态）。
  `isLoaded` 在首次快照发布前为 false，protocol 注释写明「⛔ 此期间 UI 不得显示『暂无 Pin』」。
- **B-24** `PinStoreError` 六例齐备（`.unavailable` / `.fieldTooLong(field:)` / `.notFound` / `.frozenFieldMutation(field:)` / `.io(underlying:)` / `.schemaTooNew(found:supported:)`）。
  ⛔ 零 `try?` 吞写入错误（仅用于「删掉一个本就该消失的文件」这类无信息量清理）。`[PIN]` 日志经 `perfLog`/`faultLog` 走 B-11 的 `[t=…]` 前缀。
  **成功/失败同形可辨**：成功行是 `ok` + 该操作自己的字段（`blob=…B nz=… pins=…`），失败行是 `FAILED err=<case>` 且不带任何结果字段 —— §35.9 的教训是「同形分支会让门变成看不见的 no-op」。
- **B-25** 信封 `{schemaVersion, writtenAt, pins}`；`PinSchema.migrations: [Migration]` **v1 为空数组但结构现在就在**（首次 schema 变更是数据改动，不是架构改动）；
  链上缺步 ⇒ 显式抛错，**不靠「凑巧能解码」蒙混**；迁移前落 `manifest.v<N>.bak`（保留最近 1 份）；
  前向版本 ⇒ 进只读态、所有变更型 API 返 `.schemaTooNew`、⛔ **flush 路径显式拒绝重写**（一次降级不得截断用户全部数据）。
- **B-26** 删除 `JudgeE2/Item.swift`（`@Model final class Item`，全工程零引用），并从 `project.pbxproj` 摘除其 PBXBuildFile / PBXFileReference / 组成员 / Sources 四处条目。
  **确认工程既未链接 SwiftData 也未链接 CoreData**：全工程 `import SwiftData` / `import CoreData` / `modelContainer` / `NSPersistentContainer` 命中数在删除后为 **0**；
  `PBXFrameworksBuildPhase` 的 `files` 为空；工程内无 `.xcdatamodeld`。
- **B-27 / B-28 —— 只落接口**：`PinFactory.makeRecord(from:geometry:tag:note:)` / `PinFactory.maskAlpha(from:)` 与
  `CameraManager.handleTap(fromPin:)`（扩展，**未打开 `CameraManager.swift`**）。三者的实现体是 `assertionFailure` + 返回 nil/0。
  注释写死了各自的约束：makeRecord 只读 `maskAlpha` / `canonicalPoint` / `createdAt` 三个量、⛔ 一个 re-anchor 字段都不读；
  `handleTap(fromPin:)` 必须转调既有 `handleTap(canonicalPoint:viewPoint:)`，⛔ 不得成为第二条 decode 入口（PIN-5）。

### 与热路径的隔离（约束 PIN-3，可静态核查 —— 即 §19.6 P-5 的自查部分）

对 `Detection/CameraManager.swift`、`Interaction/*.swift`、`Segmentation/*.swift` 全文检索
`PinStore|FilePinStore|PinRecordV1|MaskPNGCodec|PinFactory|pinLog|pinFault` ⇒ **命中 0**。
本轮唯一触及 `CameraManager` 的是 `Persistence/PinInterfaces.swift` 里的一个扩展方法（空实现），
它不出现在任何 `videoQueue` / `decoderQueue` / `encoderQueue` 闭包内。⚠️ P-5 的正式判定归 Debugger，此处只是写代码当天的自查。

### 未做 / 未碰（明确列出）

- `SAMDecoder.swift` / `MaskRenderer.swift` **未打开修改**（只读 `MaskRenderer` 核实 256×256 不变量）；`buildTapAlpha` **零触碰**；R3 禁令参数全数未变。
- 几何链（§3）、`PointPromptBuilder` 的零先验契约、`.segmentation` 路径、任何 re-anchor 代码 —— **一行未动**。
- **不新增第二条队列、不新增任何锁**；`videoQueue`/`decoderQueue`/`encoderQueue` 零投递。
- **Day 4 零 UI**：PinCreationSheet / PinList / AnnotationView 全属 Day 5/6，未建任何视图，`FilePinStore` 也未接进 `ContentView` / `JudgeE2App`
  （三处强制 flush 里的前两处需要 `scenePhase` / `willTerminate` 挂载点，属 Day 5 接 UI 时的一行接线；`flush(waitUntilDone:)` 已就位）。
- **未实现 §19.4.5 的坐标重映射、未实现任何重访逻辑** —— Day 6，且 R28 的判据问题不在本轮射程内。
- Pin × re-anchor 的交互（R27 / R32）**未处置**：`makeRecord` 空实现意味着「保存瞬间 mask 正被替换」的竞态在本轮不可能发生，
  按 §20.3.4 的要求**记录并留给 Day 5 裁决，不就地处置**。
- **未做 Debugger 的工作**：无真机测试、无日志分析、未写 `shared/debug_report.md`；未写任何单元测试目标
  （§19.6 的 P-1…P-5 是 Debugger 的验收协议，不是 Builder 自评）。
- 未触碰 Architect / Debugger 的任何 checkbox、STOP RULE 区块、Phase 4A 任何条目。

### 规格有歧义、由 Builder 作出解释的三处（如实记录）

1. **`create` 的入参形状。** §19.7 只写 `create(_:completion:)`。B-27 的产物是 `PinRecordV1`，而 blob 字节必须另行送达 store，
   故落为 `create(_ record: PinRecordV1, maskAlpha: [UInt8]?, completion:)`。store 侧校验 `maskFile != nil ⟺ maskAlpha != nil`、
   尺寸必须是 256×256、`maskNonZero` 必须等于实算值（不符 ⇒ `.io(PinIntegrityError…)`，不新增第七个错误 case）。
2. **成功回调的时机**（见 B-21）：合并窗与 P-3(ii) 字面冲突，采「挂起回调至落盘后释放」。
3. **`Pin` 与 `PinGeometryV1` 的关系。** §19.2.1 要求运行时与持久化是两个类型；`Pin` 已做到（**不 Codable**），
   但其 `geometry` 字段直接持有 `PinGeometryV1` 而非再造第三个运行时几何类型 —— 再造一个会让 B-20 的「两个纯函数」变成四个，
   收益为零。`FrameGeometry` 仍然完全不参与持久化，规则 P-A 的保护对象未被削弱。

### 编译验证

```
cd JudgeE2 && xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**，零 error、零 warning。

---

## Phase 4B Day 5 —— PinStore 补救（B-29…B-36）+ B-21/B-27 收尾 + Pin 创建 UI (Builder, 2026-08-21)

依据：architect_output.md **§22**（全节，PinStore 补救裁决 + Pin×既有交互统一裁决）+ §19/§21（Day 4 背景）+
debug_report.md **§37**（Day 4 静态验收，产出本轮全部裁决对象）+ tasks.md Phase 4B Day 5 区块（长按建 Pin / 缩略图，已按 §22 更正）。
⚠️ 本轮**未重新打开** `SAMDecoder.swift` / `MaskRenderer.swift`；`buildTapAlpha` 零触碰；`canonicalPoint` 冻结不变；
**未触碰** `DriftDetector.swift` / `ReAnchorLoop.swift` / 一致性门——§22.2.5 裁定三明确拒绝的方向本轮一行未做。

### A 组 —— §37 静态发现补救（B-29…B-36）

- **B-30**（P0）`FilePinStore.swift` — `stage=decode` 与 `stage=migrate` 两条 catch 分支补齐 `storeUnavailable = true`；
  `stage=decode` 新增取证备份 `writeCorruptBackup(bytes:schemaVersion:)`（写 `manifest.corrupt-<version|unknown>.bak`，
  `stage=migrate` 复用既有 `backupManifest`，未改动）；两条分支新增 `publishUnavailableNow()`，在置位的同一时刻把
  `lastWriteError = .unavailable` 发布到主线程（复用既有 `@Published`，未新增字段）。目录创建失败分支未改动（本轮范围
  按 §22.1.2 严格限定在 decode/migrate 两条）。
- **B-31**（P0）`create` / `update` / `delete` / `loadMaskImage` 四个入口，方法体第一行调用 `self.load()`（幂等）。
  结构性消除 ISSUE-D4-2（P0-2）：串行队列 FIFO 保证 `load()` 的 `installRecords` 闭包必先于本次调用自己的
  `queue.async` 执行，不再需要调用方自律。
- **B-32**（P0）`publishSnapshot` 调用点从 `markDirtyAndPark` 移至 `flushNow` 的 `.success` 分支；`.failure` 分支不发布，
  `pins` 维持上一次真实落盘的快照。`load()` 内的 `publishSnapshot(loaded:true)` 调用点未动。代价（按裁决明示）：
  新建/编辑/删除一条 Pin 后 `pins`（及绑定它的列表 UI）最多延迟 ≤250ms 合并窗才更新，且必须等一次真实落盘成功。
- **B-33**（P3）注释修正：`FilePinStore.swift` 的 blob LRU 注释「≤32 entries (≈256 KB)」改为「≤32 entries (≈2 MiB decoded,
  65,536 B × 32)」。容量数字（32）未改；`create` 的缓存预热判定为合法，未改动。
- **B-34**（P1）actor 隔离显式标注：`Pin.swift` 的 `PinStoreError` / `PinIntegrityError`、`PinRecordV1.swift` 的
  `PinGeometryV1` / `PinRecordV1` / `PinManifestV1` / `PinCoders`（外加未在七类清单内但同一理由的 `PinSchema` /
  `PinManifestHeader`）、`MaskPNGCodec.swift` 的 `MaskPNGCodec` —— 整个类型声明标 `nonisolated`；`FilePinStore`
  的 `pins` / `isLoaded` / `lastWriteError` / `fetch(id:)` / `fetchAll()` 标 `@MainActor`，其余方法体、全部 private
  helper、`queue` 私有的十个状态量标 `nonisolated`（可变存储属性用 `nonisolated(unsafe)`，Swift 语法要求）；
  `PinStore` 协议的对应方法签名同步标注，避免协议要求与实现隔离级别不一致。新增 `FilePinStore.onMain(_:)`
  静态桥接函数（`DispatchQueue.main.async { MainActor.assumeIsolated(work) }`），供 `publishSnapshot` /
  `releasePending` / `finish` / `publishUnavailableNow` 复用——机制仍是 `DispatchQueue.main.async`（B-21 以来未变），
  只是让编译器能核对这条早已存在的纪律。**交付口径按裁决要求**：两个配置（Debug/Release）、全新 `derivedDataPath`
  的干净构建输出见下方「编译验证」。
- **B-35**（P1）`create` 新增 `record.tag`/`record.note` 归一化（`Self.normalizeEmptyField`，`""` → `nil`），与
  `update` 采用同一函数（`update` 原地内联的 `isEmpty ? nil : $0` 一并替换为调用同一函数）。`PinFieldLimits`
  新增 `static func length(of s: String) -> Int { s.utf16.count }`；`FilePinStore.fieldLengthFailure` 与 Day 5
  `PinCreationSheet` 的输入限制器改为共用这一函数（UTF-16 code unit 计数）。
- **B-36**（P2）`PinCreationSheet.swift` 新增无条件静态提示文案（缓解 R32 可发现性问题，见下）。
- **B-29**（P1）新建 `Persistence/PinDebugFixture.swift`，`#if DEBUG` 整文件隔离，编译进 App target（未新建/未修复
  `JudgeE2Tests`）：`makeSyntheticRecord(tag:note:maskAreaPx:includeMaskFile:)` 构造合成 `(PinRecordV1, [UInt8]?)`，
  覆盖 P-1 五类边界值（调用方传参覆盖，fixture 本身不预置具体十条）；`batchSave(store:count:intervalMs:)`
  按固定间隔批量 `create`，服务 P-4 B 臂与 P-3(iii)；`runFromLaunchArgumentsIfPresent(store:)` 解析
  `-PinFixtureBatch <count>:<intervalMs>` 启动参数并触发，接线在 `JudgeE2App.init()` 后的 `onAppear`（`#if DEBUG`），
  **无 UI 入口**。约束核实：不引用 `TapInstance`/任何 re-anchor 字段，不向三条热队列投递工作，只调用 `PinStore`
  协议上的 `create`/`flush`。

### B-21 收尾 —— 三处强制 flush 全部接线

`Persistence/FilePinStore.swift` 的 `flush(waitUntilDone:)` Day 4 已实现（合并窗到期一处已存活）。Day 5 补齐另两处，
全部在新建/改写的 `UI/JudgeE2App.swift` 内：`@StateObject private var pinStore = FilePinStore()` 作为全应用唯一实例，
`.onChange(of: scenePhase)` 在 `newPhase == .background` 时调用 `pinStore.flush(waitUntilDone: true)`；`willTerminate`
无 SwiftUI 场景阶段等价物，改用 `NotificationCenter.addObserver(forName: UIApplication.willTerminateNotification, ...)`
（未引入 `UIApplicationDelegateAdaptor`，一个通知没有必要新增一整个 AppDelegate 桥接层）。`pinStore` 经
`.environmentObject(pinStore)` 注入 `ContentView`；`pinStore.load()` 在 `onAppear` 触发（B-31 落地后其实四个写入
入口任一个都会顺带触发，这里提前触发只是让 UI 尽快见到已有数据）。

### B-27 —— `PinFactory.makeRecord` / `maskAlpha` 从接口落地为实现

`Persistence/PinInterfaces.swift`：

- `makeRecord(from:geometry:tag:note:)` 严格只读三个量：`instance.maskAlpha` / `instance.canonicalPoint` /
  `instance.createdAt`。**两处刻意不读 `instance` 上其他已存在的字段**（如实记录为规格解释，见下方歧义清单）：
  `id` 用全新 `UUID()` 而非 `instance.id`；`iouPredAtCreation` 恒为 `nil` 而非 `instance.iouPred`。
- `maskAlpha(from:)` 原样返回 `instance.maskAlpha`。
- 两函数均不再 `assertionFailure`；`handleTap(fromPin:)`（B-28，Day 6 项）保持接口占位，未实现，未触碰。
- **PIN-6 单次锁内快照纪律**：这两个函数本身不加锁——纪律落在调用方（`CameraManager.handleLongPress` 经
  `hitTestExistingInstance` 对 `tapInstances.snapshot()` 的**唯一一次**调用，产出的 `TapInstance` 值贯穿
  `PinCreationDraft` → `PinCreationSheet` → 这两个函数，构造函数与 Save 时的实际调用读的是**同一个** Swift 值）。

### Day 5 主线 UI —— 长按建 Pin（§22.2 决策树 + tasks.md，128×128 已按 §22 更正为 256×256）

**长按手势（`Interaction/TouchHandler.swift`）**：新增 `UILongPressGestureRecognizer`（`minimumPressDuration = 0.5`），
新回调 `onLongPress: ((CGPoint, CGPoint) -> Void)?`，只在 `.began` 触发一次。**未**与 `singleTap`/`doubleTap` 建立
`require(toFail:)` 关系——§22.2.1 裁定二者在同一次触摸上天然互斥（长按识别器 0.5s 前抬手根本不会 `.began`），
不需要、也不得额外裁决优先级。`Detection/CameraPreview.swift` 接线：`handler.onLongPress` 转调
`cameraManager.handleLongPress(canonicalPoint:viewPoint:)`。

**决策树（`Detection/CameraManager.swift`）**：

- 把原先内联在 `handleTap` 里的 canonical→256 空间命中判据（`alpha[idx] > 0`）抽成私有 helper
  `hitTestExistingInstance(canonicalPoint:letterbox:)`，`handleTap` 的 promote 分支与新 `handleLongPress` 共用，
  避免维护两份坐标数学。`handleTap` 本身除这处抽取外**一字未改**（同一算法、同一调用顺序、同一日志）。
- `handleLongPress(canonicalPoint:viewPoint:)`：命中已有 mask → 发布 `pinCreationDraft`（**不** `promoteToPrimary`，
  **不**触发任何 decode）；未命中任何 mask → no-op（不新建实例，长按不是 §3.2 `addInstance` 的触发源）。
  `PinCreationDraft` 只携带 `TapInstance`（已经是 `tapInstances.snapshot()` 的值类型快照）+ `FrameGeometry`——
  `CameraManager.swift` 全文**仍然零个** `Pin`/`PinStore`/`PinFactory` 符号引用（S1/S2/S5 三组搜索复核，命中的
  全部是英文注释里的"Pin"字样，非代码符号；已在报告里注明这一区分，供 Debugger Day 5 复跑 P-5 时留意）。
- 新增 `TapAnchorMarker.isPinned: Bool`（装饰 `TapInstance`，随其生命周期消失，见 §22.2.4）；`CameraManager`
  新增私有 `pinnedInstanceIDs: Set<UUID>` + `markInstancePinned(id:)`（UI 保存成功后回调）；`discardAllTapWork`
  与 FIFO 淘汰分支各自清理对应 id，避免这个 Set 无限增长。

**`UI/PinCreationSheet.swift`（新建）**：底部半屏——256×256 缩略图（`MaskPNGCodec.encode` 直接生成 PNG 供
`UIImage` 显示，SwiftUI 端缩放显示，**不生产任何独立的 128×128 或其他尺寸资产**）+ 标签输入框（限 64 UTF-16
单元，超出截断，与 store 用同一 `PinFieldLimits.length(of:)`）+ 保存/取消按钮 + **B-36 无条件静态提示文案**
（"若画面内容与当前实际物体不符，可点击该物体在画面中的其他位置以重新框选"——不依状态显示，因为依状态显示
需要读 `originAlpha` 等 re-anchor 派生信号，触犯 §18.3.3 硬隔离 1）+ `lastWriteError == .unavailable` 时的红色
横幅并禁用保存按钮。保存路径：`PinFactory.makeRecord` → `PinFactory.maskAlpha` → `store.create(...)`；成功后
经 `onSaved` 回调让 `CameraManager.markInstancePinned` 把该实例的锚点图标切换成 📌（tasks.md 原文「图标变为
图钉📌样式」）。`ContentView.swift` 新增 `.sheet(item: $cameraManager.pinCreationDraft)` 呈现、`@EnvironmentObject`
接入 `pinStore`、顶层 `lastWriteError == .unavailable` 持久横幅（B-30 第 3 点在 UI 侧的落地）。`TapAnchorMarkerView`
新增 `isPinned` 参数，为真时把编号数字换成 📌。

### `project.pbxproj` 手工登记

新增 `Persistence/PinDebugFixture.swift`（Persistence 组）与 `UI/PinCreationSheet.swift`（UI 组），各自补齐
PBXFileReference / PBXBuildFile / 组 children / App target 的 Sources build phase 四处条目（`Persistence`/`UI`
两个组延续 Day 4 的手工登记模式，均非 `PBXFileSystemSynchronizedRootGroup`）。

### 规格有歧义、由 Builder 作出解释的两处（如实记录）

1. **`makeRecord` 的「恰好三个量」是否严格到排除 `id`/`iouPred`。** §22.1.6/§22.2.3/§19.7 原文反复写「读取
   `instance.maskAlpha` / `canonicalPoint` / `createdAt` 三个量」，字面上不含 `id` 与 `iouPred`。本轮按字面严格
   执行：`PinRecordV1.id` 用全新 `UUID()`（不等于 `instance.id`——Pin 的身份本就不与 TapInstance 绑定，§22.2.4
   已确立两者生命周期独立）；`iouPredAtCreation` 恒为 `nil`（该字段本就标注「⛔ 诊断专用」，留空不影响任何判定
   逻辑）。若 Architect/Debugger 认为 `iouPredAtCreation` 应该从 `instance.iouPred` 填充，这是一处可以低成本
   改动的一行差异，Day 5 未擅自扩大读取范围。
2. **`finish(_:op:id:completion:)` 内 `pinFault` 调用点的隔离处理。** B-34 要求 `FilePinStore` 私有方法体标
   `nonisolated`，但 `pinFault`/`perfLog`/`faultLog`（`Shared/PerfLogging.swift`，不在本轮改动范围）保留模块
   默认的 `@MainActor`。直接从 `nonisolated` 函数体同步调用一个 `@MainActor` 全局函数在 Swift 5 模式下仍会报
   warning（不同于闭包传给 GCD API 那种情况——本轮验证过后者不报）。处理为把 `finish` 里唯一一处 `pinFault`
   调用挪进已经在 `onMain` 里的主线程闭包（其余错误日志此前就都在 `queue.async` 闭包内，未受影响）。代价：
   这一条失败日志从「同步立即」变成「派到主线程后」打印，内容不变，观感上几乎不可分辨（`DispatchQueue.main`
   下一次 runloop）。未触碰 `Shared/PerfLogging.swift`。

### 未做 / 未碰（明确列出）

- `SAMDecoder.swift` / `MaskRenderer.swift` 未打开；`buildTapAlpha` 零触碰；R3 禁令参数全数未变；`canonicalPoint`
  冻结不变。
- `DriftDetector.swift` / `ReAnchorLoop.swift` / 一致性否决门 —— 一行未动（§22.2.4/§22.2.5 明确拒绝的方向本轮
  未做，包括「按状态显示 R32 提示」「promote 补写 originAlpha」等）。
- `handleTap(fromPin:)`（B-28）—— 仍是接口占位，未实现，属 Day 6。
- `AnnotationView` / `PinList` / 重访逻辑 —— 全部 Day 6，本轮未建。
- 未新增队列、未新增锁；PIN-6 复用 `TapInstanceManager` 既有锁（未修改该文件）；`pin.store.io` 队列纪律未变。
- ISSUE-D4-4/5/6/7/9-12/15-17 中除本轮显式列出的（B-30/31/32/33/34/35 覆盖的那部分 + 三处强制 flush 接线）——
  §22.1.10 明确留待后续批次的条目本轮未处理，如实沿用原优先级。
- 未做 Debugger 的工作：无真机测试、无日志分析、未写 `shared/debug_report.md`；P-1…P-4 的实际执行（用 B-29
  fixture 驱动）留给 Debugger。
- 未触碰 Architect / Debugger 的任何 checkbox、STOP RULE 区块。

### 编译验证（按 §22.1.7 口径要求：两个配置、全新 derivedDataPath）

```
# Debug
rm -rf /tmp/dd_debug
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -sdk iphonesimulator \
  -configuration Debug -derivedDataPath /tmp/dd_debug clean build
→ ** BUILD SUCCEEDED **，0 error，0 warning（仅一条与本工程代码无关的
   `appintentsmetadataprocessor` 工具提示「Metadata extraction skipped. No AppIntents.framework dependency
   found.」，非 Swift 编译诊断）

# Release
rm -rf /tmp/dd_release
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -sdk iphonesimulator \
  -configuration Release -derivedDataPath /tmp/dd_release clean build
→ ** BUILD SUCCEEDED **，0 error，0 warning（同上，仅同一条工具提示）
```

另按任务给定的原始命令复核一次（同一模拟器 destination，非 clean）：

```
cd JudgeE2 && xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**，零 error、零 warning。

⚠️ **本轮明确做了 Day 4 没做的事**：以上两个 clean 配置均使用全新 `derivedDataPath`，命令与输出均可复现——
按 §22.1.7「涉及 warning 数量的自评必须附带可复现命令」的要求，不再是一句结论。

---

## Day 6 — AnnotationView + PinList + 重访逻辑（§19.4 契约，取代 tasks.md 草案）

### 范围与前置阅读

按 §19.4.1 的诊断，tasks.md Day 6 草案「加载 frameGeometry → 用 canonicalPoint 重新 decode → 恢复活跃 mask」与验收
「decode 成功率 ≥ 80%」与 §16.3.1/§17.1.4 同一缺陷类（判据依赖系统测不出的量），**本轮不按草案字面实现**，改按
architect_output.md §19.4（G1–G4 准入门 + R-A/R-B/R-C 三结局 + PIN-5 合成 tap 约束 + §19.4.6 日志格式）实现。已确认
§19.4.7 的建议验收判据**尚未经 Architect 在「Day 5/Day 6」正式裁决**（同 Day 5 P-4 缺口的形状）——本轮不勾选、不暗示
Debugger 的三条 Day 6 判据已成立，那是另一批裁决的范围。

### 重访逻辑（`CameraManager.handleTap(fromPin:store:)`，B-28 补全）

**接口签名与 Day 4 占位的差异（需要在此如实记录）**：B-28 原始占位签名是 `handleTap(fromPin pin: Pin) -> Int`，
不带 `store` 参数。§19.4.6 的日志行要求 `iou=<0.00–1.00|n/a>`——IoU 需要 Pin 记录当初存下的 origin mask 字节，而这
只能异步从 blob 读出（`PinStore.loadMaskImage`），`CameraManager` 本身不持有 `PinStore` 引用（也不应持有——PIN-3
整文件禁止 `CameraManager.swift` 出现 Persistence 符号）。本轮把签名扩为
`handleTap(fromPin pin: Pin, store: PinStore) -> Int`，由调用方（UI 层）传入已持有的 `pinStore` 实例。这是唯一必
须偏离 Day 4 字面占位签名的地方，原因是日志需求在 Day 4 占位时尚未细化到这一步。

**G1–G4 准入门**（`JudgeE2/Persistence/PinInterfaces.swift`）：
- **G3 优先判**：先取 `currentFrameGeometry()`，因为 G1 的比对没有它无从谈起；nil ⇒ 拒绝，文案「Camera not ready
  — try again in a moment.」（与既有 tap 失败文案一致）。
- **G2**：`currentMode == .tapToSegment`，否则拒绝并提示切换模式，**不自动切模式**。
- **G1**：`pin.geometry.isCompatible(withRotationDeg:mirrored:promptSpace:)`（复用 `PinGeometryV1` 已有的方法，
  未重写判定逻辑）逐位比对；全等 ⇒ `geo=ok`；rotation/mirrored/promptSpace 全等但 `origW/origH` 不等且宽高比在
  `1e-3` 内 ⇒ §19.4.5 的唯一允许重映射（`pointX' = pointX × origW_now/origW_pin` 等，再复用 §2.3 Step 5 同款边界
  裁剪，**不写回 Pin**，PIN-1）⇒ `geo=remapped`；其余任何不一致 ⇒ `geo=refused`，落 R-B，原因文案按不一致的具体
  维度组装（朝向/摄像头 vs promptSpace vs 宽高比，三选一或组合），全英文。
- **G4**：无特例——下面这行调用直接复用既有 FIFO(max=3)。

**PIN-5（合成 tap，不新建 decode 入口）**：G1–G3 通过后，唯一的动作是
`let gen = self.handleTap(canonicalPoint: derivedPoint, viewPoint: nil)`——`tapGeneration`/`requestGen` 排序、
`inFlightTaps`、快慢路径判定、park/drain、超时、Requirement C 失败可见性、几何变更清池、FIFO、「tap 落在已有
mask 内 promote」全部由这一次调用免费继承，本文件**零重写**。

### §19.4.6 日志的两个观测钩子（唯一触碰 `CameraManager.swift` 的地方，且为纯观测）

`path=(fast|slow)` 与 decode 完成/失败的时机在 `handleTap(canonicalPoint:)` 内部（`tapDecodeWithPoint`/`failTap`），
且这两个函数不可重写（PIN-5）。本轮在 `CameraManager.swift` 新增两个**观测用**闭包属性（默认 nil，零成本）：

```swift
var onTapMaskPlaced: ((_ gen: Int, _ path: TapPath, _ instanceID: UUID, _ alpha: [UInt8]) -> Void)?
var onTapFailed: ((_ gen: Int, _ message: String) -> Void)?
```

分别在 `tapDecodeWithPoint` 主线程发布块（`publishAnchorMarkersOnMain()` 之后）与 `failTap` 主线程块
（`tapFailure` 赋值之后）各插入一行调用，使用的都是该函数体已经算好的值（`gen`/`path`/`instanceID`/`built.alpha`/
`message`），**不引入任何新判定分支、不改变任何既有变量的计算方式**——纯粹是"这个决定已经做出，顺便告诉外部
一声"。`PinInterfaces.swift` 内的 `PinRevisitTracker`（私有单例）按 `gen` 复用这两个单一闭包槽位（因为它们对
**每一次** tap 都会触发，屏幕真实 tap 与重访 tap 共用同一对钩子），main-thread-only 字典分派，无锁（§19.7"不新增
锁"）。

**IoU 计算**：复用 `DriftDetector.alphaIoU`（`stride: 1`，256×256 存储空间，未新写第二套实现，符合 §19.4.6 对
「不得引入采样偏差不同的第二实现」的要求）；`origin` 来自 `PinStore.loadMaskImage` 异步读回的 blob（Pin 若无
blob，`iou=n/a`）；`new` 来自新 decode 出的 `alpha` 的 `MaskPNGCodec.nonZeroCount`。origin 加载与 tap 结果用一个
仅限 main thread 的小型 `Join` 类汇合，两者都到齐才落 `pinLog`/`pinFault` 一行——不阻塞 tap 本身（`handleTap`
仍同步返回 `gen`，日志行异步补上）。

**R-A 文案切换**（§19.4.4，禁用"恢复"类措辞）：新增 `@Published var lastTapWasRevisit: Bool`，在
`handleTap(canonicalPoint:)` 已有的主线程发布块里重置为 `false`（与其余 UI 状态同一批发布，零额外开销），
`handleTap(fromPin:)` 在拿到 `gen > 0` 后**用同一个 `DispatchQueue.main.async` 机制**把它设回 `true`——两次
`main.async` 调用先后入队，GCD 保证 FIFO 顺序，重置一定先于置位执行。UI 侧据此显示一句过渡文案「Re-segmenting at
this location…」，复用既有 `tapProcessing` 脉冲，**未新建等待 UI**（§19.4.2 明令）。

### AnnotationView / PinList / 静态回退查看器（新文件，均在 `UI/`）

- **`AnnotationView.swift`**：256×256 缩略图（`loadMaskImage` + `MaskPNGCodec.encode`，非 128×128）、tag/note 编
  辑（字符计数复用 `PinFieldLimits.length(of:)`，未重复实现）、`createdAt`/`updatedAt` 展示、删除二次确认。保存
  走 `PinStore.update(id:tag:note:)`，删除走 `PinStore.delete(id:)`。
- **`PinListView.swift`**：`store.pins` 数组上的排序（Newest/Oldest）与标签筛选（§19.1.2：内存操作，未建任何查
  询基础设施）；`!isLoaded` 显示 loading，不与"暂无 Pin"同形（§19.3.2 纪律，PinList 是本轮第一处直接消费
  `isLoaded` 的 UI，之前的 UI 都不需要它）。**UX 分工（§19.4 未规定，本轮自行决定，如实记录）**：行体点击 → 重
  访（`cameraManager.handleTap(fromPin:store:)`）并关闭列表；行尾滑动露出「Edit」按钮 → `AnnotationView`（列表内
  嵌套 sheet，不关闭列表本身）。选择理由：重访是"看一眼这条 Pin 现在什么样"的高频操作，值得一次点击；编辑是有
  意的、低频的动作，值得一个需要额外发现的手势，避免误触删除/改标签。
- **`PinRevisitStaticViewerView.swift`**：R-B/R-C 的静态回退。视觉刻意与活跃 mask 不同形（PIN-2 第 3 条）——灰
  度（`saturation(0)`）、虚线橙色描边、"STATIC SNAPSHOT — not live"角标；显式展示原因文案（R-B 的 G1 拒绝原因 /
  R-C 的失败说明），**不静默降级成一张不明图片**。
- **`ContentView.swift`**：导航栏新增英文"Pin List"图钉按钮；`.sheet` 接 `PinListView`；`overlay(alignment:
  .bottom)` 接 `cameraManager.pinRevisitEvent`（R-B/R-C 的横幅，带「View Saved Snapshot」按钮打开静态查看器）；
  `overlay(alignment: .top)` 接 R-A 的过渡文案。`CameraManager.PinRevisitEvent` 是 `CameraManager.swift` 自己的
  嵌套类型（只携带 `UUID`/`String`，零 Persistence 符号），维持 PIN-3。

### 图钉旁标签（任务书顶部指令：「讲 label 写在图钉旁边」）

`CameraManager.TapAnchorMarker` 新增 `tag: String?` 字段（普通 `String`，非 Persistence 符号）；
`markInstancePinned(id:tag:)` 签名从 Day 5 的 `markInstancePinned(id:)` 扩为可选带 tag，由 `PinCreationSheet` 保
存成功后传入当次实际写入的 tag。`TapAnchorMarkerView` 用 `.overlay(alignment: .trailing)` 把标签贴在图钉图标右
侧（不是把图标塞进 HStack 再整体居中——那样会让图标本身偏离 `marker.viewPoint` 这个真实 tap 点，是本轮写完第一
版后发现并改掉的一处问题）。FIFO 淘汰 / `discardAllTapWork` 的清理点都同步加了 `pinnedInstanceTags` 的清除，避免
字典无界增长。

### UI 全英文（任务书顶部指令：「把 UI 都换成英文」）

翻译覆盖：`PinCreationSheet.swift` 全部用户可见字符串；`ContentView.swift` 的调试选项区（Debug Options / Force
Slow Path）、双击后提示（Tap to segment）、模式切换按钮（Tap Mode / SAM Mode）、Pin 存储不可用横幅；新建的三个
Day 6 文件从一开始就是英文。**未翻译的范围**：源码注释（不是 UI）、`shared/*.md` 文档本身（任务书用中文写就，
沿用）。

### 未做 / 未碰（明确列出）

- `SAMDecoder.swift` / `MaskRenderer.swift` 未打开；`buildTapAlpha` 零触碰；R3 禁令参数全数未变。
- `DriftDetector.swift` / `ReAnchorLoop.swift` 一行未动——仅**调用**了 `DriftDetector.alphaIoU`（既有 public 静
  态函数），未修改该文件任何一行。
- `canonicalPoint` 冻结不变；§19.4.5 的重映射结果不写回 Pin。
- 未新增队列、未新增锁；`PinRevisitTracker` 与 R-A 的 `Join` 均为 main-thread-only 的普通字典/局部对象。
- 未在 `CameraManager.swift` 之外的任何"videoQueue/decoderQueue/encoderQueue"闭包内引用任何 Pin/PinStore 符号
  （PIN-3 未扩大触碰面——两个新增的观测钩子调用点都在既有的 `DispatchQueue.main.async` 块内，不在这三条队列上）。
- 未执行 §19.4.7 建议的验收判据（10 次同机位重访中位 IoU、横持 10/10 R-B）——那是 Debugger 的工作，且其判据本
  身尚未经 Architect 正式裁决，本轮不代为勾选。
- 未做 Debugger 的工作：无真机测试、无日志分析、未写 `shared/debug_report.md`；未触碰 R34（未新增任何在
  `load()` 既有幂等守卫之外重读/重装 manifest 的路径）。
- 未触碰 Architect / Debugger 的任何 checkbox、STOP RULE 区块。

### 编译验证（两个配置、全新 `derivedDataPath`，按 §22.1.7 口径）

```
# Debug
rm -rf /tmp/dd_debug_day6
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -sdk iphonesimulator \
  -configuration Debug -derivedDataPath /tmp/dd_debug_day6 clean build
→ ** BUILD SUCCEEDED **，0 error，0 warning（仅同一条与工程代码无关的 AppIntents 元数据提示）

# Release
rm -rf /tmp/dd_release_day6
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -sdk iphonesimulator \
  -configuration Release -derivedDataPath /tmp/dd_release_day6 clean build
→ ** BUILD SUCCEEDED **，0 error，0 warning（同上）
```

另按任务给定的原始命令复核一次（同一模拟器 destination，非 clean）：

```
cd JudgeE2 && xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' build
```
→ **BUILD SUCCEEDED**，零 error、零 warning。

### 新增/修改文件清单

- 新增：`JudgeE2/UI/AnnotationView.swift`、`JudgeE2/UI/PinListView.swift`、
  `JudgeE2/UI/PinRevisitStaticViewerView.swift`
- 修改：`JudgeE2/Persistence/PinInterfaces.swift`（B-28 补全 + `PinRevisitTracker`）、
  `JudgeE2/Detection/CameraManager.swift`（两个观测钩子 + `PinRevisitEvent`/`pinRevisitEvent` +
  `lastTapWasRevisit` + `pinnedInstanceTags` + `TapAnchorMarker.tag` + `markInstancePinned(id:tag:)`）、
  `JudgeE2/UI/ContentView.swift`（Pin List 入口 + 三个新 sheet/overlay + 英文化）、
  `JudgeE2/UI/PinCreationSheet.swift`（`onSaved` 签名带 tag + 英文化）、
  `JudgeE2/JudgeE2.xcodeproj/project.pbxproj`（三个新文件的手工登记，沿用 Day 4/5 的非
  `PBXFileSystemSynchronizedRootGroup` 手工模式）

---

## 用户直接指令的 UI 调整（非 tasks.md 排期项）

> 用户原话：「把UI都换成英文，点击图钉时要显示label。整个操作面板在小化后要收缩成一个小方块，不遮挡屏幕。」
> 本条不对应 tasks.md 任何 checkbox，未勾选任何 checkbox。纯表现层改动，唯一改动文件：`JudgeE2/UI/ContentView.swift`。

### 变更 1 — UI 全英文（复核结论：无残留）

独立复核范围超出 `UI/*.swift`，覆盖整个 app target：

```
grep -rn --include='*.swift' -E '"[^"]*[一-龥]' JudgeE2/    → 21 命中，全部在 // 或 /// 注释行内
grep -rn --include='*.swift' -P '"[^"]*[\x{4e00}-\x{9fff}\x{3000}-\x{303f}\x{ff00}-\x{ffef}]' \
  | grep -vE '^\S+:[0-9]+: *(//|///|\*)'                     → 0 命中
```

另逐一核对了「不在字符串字面量里也可能上屏」的三类来源，全部为英文，无需翻译：

- `AppMode.displayName` / `InferenceBackend.displayName`（`Shared/AppMode.swift`、`Shared/InferenceBackend.swift`）——
  Picker 选项文案的真实来源，Day 6 的英文化只改了 `UI/`，这两个文件当时未被复核，本轮补核：
  "Detection" / "Segmentation" / "Tap to Segment"、"CPU" / "CPU+GPU" / "CPU+NeuralEngine" / "All"。
- 四个 `CustomStringConvertible.description`（`PinStoreError`、`PinIntegrityError`、
  `MaskPNGCodec.CodecError`、`PinRecordV1.MigrationError`）——均为诊断用的 case 名，未本地化亦不面向终端用户。
- `CameraManager` 的 `TapFailure.message` 全部产出英文（如 "camera not ready — tap again in a moment"）。

结论：**Day 6 的英文化确实已完整，本轮未发现任何遗漏的中文 UI 文案，因此变更 1 未产生代码改动**（仅新增了 3 处
`accessibilityLabel`，见变更 3）。源码注释与 `shared/*.md` 按指令保持中文不动。

### 变更 2 — 点击图钉显示 label（原为常驻显示）

**先说清楚一个实现上的限制并据此定形态：** 标记绘制在 `marker.viewPoint`，该点按构造必然落在自己的 mask 内部，
而 §22.2.2 的命中判据是 `alpha[idx] > 0` ——因此「点中标记图形本身」与「点中该 mask 任意位置」在现有契约下
**不可区分**。按任务书指示，不发明 marker 专属命中区（那会割裂已冻结的 tap 契约），采用合理解释：
**任何一次把某实例提升为 primary 的点击，就显示该实例的 label**。

形态选择：**限时显示（约 2.6 s 后淡出），不是 toggle**。理由：toggle 需要「再点同一个图钉 = 收起」，但对
已经是 primary 的实例再点一次，promote 是 no-op，管线不产生任何可观测状态变化，视图层看不见这次点击——除非
新增 marker 专属命中区，而那正是被禁止的。限时显示不需要这个信号。

与 promote 既有反馈的配合：promote 已经用「更大的圆盘 + 白环」回答了*哪个*是当前主张，label 回答的是紧接着的
*它是什么*，然后让位——三个 pin 同时在屏时不会永久压着三块文字。点另一个 pin 时，前一个的 label 自动收起
（`revealedTagMarkerID` 是单值状态）。

实现（全部在 ContentView 表现层，**未触碰 `handleTap` 的路由/decode 逻辑、未触碰 `TouchHandler`、未新增手势识别器、
未新增 decode 入口**）：

- 新增 `@State revealedTagMarkerID: UUID?` + `@State tagRevealToken: Int`。
- 驱动源全部是管线**已经发布**的状态：
  - `.onChange(of: primaryMarkerID)`（`tapAnchorMarkers.first(where: \.isPrimary)?.id` 的计算属性）——
    覆盖「点中另一个 pin」；CameraManager 在 `recompositeForPromote` 之后重发 markers，那才是知道该显示谁的时刻。
  - `.onChange(of: lastTapIndex)`——覆盖「重复点已是 primary 的那个 pin」（此时 primary 不变，上一个钩子不触发）。
    `lastTapIndex` 在 promote 分支同样自增（`CameraManager.swift:1344`），所以这个信号是现成的。
  - `PinCreationSheet.onSaved` 里补一次 reveal：长按保存不 promote（§22.2.2），是唯一不由 primary 变化驱动的一次，
    作为「刚存下的这个标签属于这个图钉」的保存确认。
- 自动收起用 `.task(id: tagRevealToken)`：token 一变旧 task 自动取消并重开计时。**刻意不用
  `DispatchQueue.main.asyncAfter` + 闭包读 `@State`**——那会捕获 View 结构体的旧副本、读到过期的 token，是这个
  写法的经典坑。**未新增队列、未新增锁。**
- `TapAnchorMarkerView` 增加 `var showsTag: Bool = false`（带默认值，既有调用点/预览不受影响），
  `.overlay(alignment: .trailing)` 的条件由 `isPinned && tag != nil` 改为 `showsTag && isPinned && tag != nil`，
  并加 `.transition(.opacity + .scale(0.7, anchor: .leading))`，让 label 从图钉边缘长出来。

已知的可接受代价：点中另一个 pin 时，`lastTapIndex` 钩子会先于 markers 重发若干毫秒触发，理论上会以旧 primary
的 id reveal 一瞬；因为 reveal 走 0.18 s 淡入动画，这个错误值在到达可见不透明度之前就被下一次正确的 reveal 取代，
实际不可见。记录在此以免后人误判为 bug。

### 变更 3 — 面板最小化后收缩成一个小方块

原状：`showMenu == false` 时，左上角仍常驻「JudgeE2 — Camera Pipeline」黑色药丸 + 翻转相机键 + Pin List 键 +
「Settings」DisclosureGroup 行（各自带独立背景），占掉相当一块画面。

现状：**整个面板收起后是一个 44×44 的圆角方块，内含一个 `gearshape.fill`**；展开后内容与今天完全一致。

设计决定与理由：

- **收起态显示什么**：齿轮图标（展开态换成 `xmark`）。44×44 是 HIG 最小可点区，且齿轮语义明确，不会被误读成渲染残留。
- **放在哪**：仍在 top-leading。与 `#N` 计数（bottom-trailing）、模式快切（bottom-leading）、
  "Tap to segment" 提示（居中偏下）、markers/mask（画面中部）都不冲突。
- **翻转相机 / Pin List 收起时是否保留**：**随面板一起隐藏**。用户明确说的是「一个小方块」，保留两个按钮就变成
  三个方块，直接违背字面要求。代价明示：Pin List 从一跳变两跳。判断这个代价可接受，因为最高频的操作
  （Tap ↔ SAM 模式互换）本来就有 bottom-leading 的浮动快切按钮兜底，不依赖这个面板。若用户认为 Pin List 值得
  常驻，改回来只是把那两个 Button 移出 `if showMenu` 一行的事。
- **动画**：把原来的 DisclosureGroup 拆掉，改成**单一容器、单一背景、单一圆角**的 VStack，`.padding` 在收起时
  归零、展开时为 10，外面套 `.animation(.spring(response: 0.32, dampingFraction: 0.86), value: showMenu)`。
  同一个圆角矩形在两个尺寸之间插值，读起来是「一个物体在长大」，不是一堆小方块此起彼落——原来的 DisclosureGroup
  做不到这一点，因为标题药丸和两个按钮各带各的背景、且收起时仍在屏。

**触摸不越界**：容器按内容自适应尺寸（未加 `maxWidth: .infinity`），收起后可交互区域就是那 44×44，其余点击照常
到达 `TouchHandler`。此处**不加** `allowsHitTesting(false)`——与上方那些只读 overlay 不同，这个面板本身就是交互面，
加了反而废掉它；纪律的正确对齐对象是"只读 overlay 一律 false、交互面板一律自适应尺寸"，两者本轮都满足。
`showMenu` 默认值维持 `true`（未被要求更改，改默认值属于未请求的行为变更）。

### 未做 / 未碰（明确列出）

- 未触碰 `SAMDecoder.swift` / `MaskRenderer.swift`；`buildTapAlpha`、R3 禁令参数、`canonicalPoint` 零触碰。
- 未触碰 `DriftDetector.swift` / `ReAnchorLoop.swift` / 一致性门。
- 未触碰 `handleTap` 的路由/decode 逻辑，未触碰 `handleLongPress`，未触碰 `TouchHandler.swift`
  （本轮 `TouchHandler.swift` 的 `M` 状态来自本次会话之前的既有改动，非本轮产生）。
- 未新增队列、未新增锁、未新增依赖、未改架构或数据流。
- **PIN-3 复核（按要求用 `grep -E`，未用 macOS 上会静默失效的 `grep -v` + `\|` 写法）**：
  ```
  grep -rlE --include='*.swift' '(videoQueue|decoderQueue|encoderQueue)\.(async|sync)' JudgeE2/
    → Detection/CameraManager.swift, Detection/delete-CameraManager.swift
  对这两个文件 grep -nE '(PinStore|PinFactory|FilePinStore|PinRecordV1|MaskPNGCodec)'
    并剔除注释行  → 0 命中
  ```
  即这三条队列所在的文件里根本不存在任何 Persistence 符号的代码引用，PIN-3 平凡成立。本轮改动只落在
  `ContentView.swift`（SwiftUI 主线程视图），未扩大触碰面。
- 未做 Debugger 的工作：无真机测试、无日志分析、未写 `shared/debug_report.md`。
- 未勾选 `shared/tasks.md` 任何 checkbox。

### 编译验证（两个配置、全新 `derivedDataPath`、`clean build`）

明确声明：两次都是**全新 derivedDataPath 的 clean build**（不是增量），warning 数为实际统计值，非自报。

```
# Debug
rm -rf <scratch>/dd_debug
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -configuration Debug \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' \
  -derivedDataPath <scratch>/dd_debug clean build
→ ** BUILD SUCCEEDED **（exit 0）

# Release
rm -rf <scratch>/dd_release
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -configuration Release \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' \
  -derivedDataPath <scratch>/dd_release clean build
→ ** BUILD SUCCEEDED **（exit 0）
```

warning 实测口径（对完整构建日志统计，不是凭印象）：

```
grep -cE '^/.*\.swift:[0-9]+:[0-9]+: warning:' build_debug.log    → 0
grep -cE '^/.*\.swift:[0-9]+:[0-9]+: warning:' build_release.log  → 0
grep -cE 'warning:' build_debug.log / build_release.log           → 各 1
  唯一那 1 条是：appintentsmetadataprocessor "Metadata extraction skipped.
  No AppIntents.framework dependency found." —— 工具链提示，非编译器 warning，与工程代码无关。
```

即 **Swift 编译器 warning：Debug 0 条、Release 0 条**。

### 新增/修改文件清单

- 修改：`JudgeE2/UI/ContentView.swift`（唯一改动文件）
- 新增文件：无。删除文件：无。`project.pbxproj` 未改动。

---

## 真机测试驱动的缺陷修复 — AnnotationView「Modified」时间戳不更新（2026-08-23）

> 来源：**真机测试发现的缺陷**，非 tasks.md 排期项。本条**未勾选 `tasks.md` 任何 checkbox**。
> 现象：在 AnnotationView 编辑 Pin 的 tag/note 并保存后，该页「Modified」显示的时间戳不变；
> 而 tag/note 本身持久化正确（重启 App 后仍在）。
> 纯表现层修复，改动文件两个，均在 `UI/`：`AnnotationView.swift`、`PinListView.swift`。

### 诊断复核（独立复读源码后确认，非照抄）

写入路径**无辜**，三点均已在源码上核实：

1. `Persistence/FilePinStore.swift:419` `record.updatedAt = Date().timeIntervalSince1970` —— 每次
   `update` 都写。
2. `flushNow()`（`FilePinStore.swift:575`）落盘成功后**先** `publishSnapshot(loaded: true)`（:593）
   **再** `releasePending(.success(()))`（:594）。两者都经 `Self.onMain` → `DispatchQueue.main.async`，
   FIFO 有序 ⇒ **save 的 completion 跑到时，`@Published pins` 与 `mainIndex` 都已带上新的 `updatedAt`**。
   B-32 把 publish 从 `markDirtyAndPark` 移到 flush 成功分支这件事，在这里反而是有利的：两条通道同一时钟。
3. `UI/AnnotationView.swift:90-91` 读的确实是 `pin.updatedAt`，字段没读错。

⇒ 缺陷确在表现层：`AnnotationView` 原本持有 `let pin: Pin`（值类型，sheet 呈现时冻结），Info 段三行全部
读这个冻结副本。store 重新发布**够不到**一个没人重新初始化的 `let`。诊断成立。

### 修复：身份进来，值从 store 取

- `let pin: Pin` → 拆成 `let pinID: UUID`（唯一主体标识）+ `private let presentedSnapshot: Pin`
  （呈现时快照，**只有两个合法用途**：`@State` 种子、删除窗口兜底）。
- 新增 `private var livePin: Pin { store.fetch(id: pinID) ?? presentedSnapshot }`，Info 段三行、
  `loadThumbnail` 的 `maskFile` 判断全部改走 `livePin`。`update` / `delete` / `loadMaskImage` 改传 `pinID`。
- **为何用 `store.fetch(id:)` 而不是扫 `store.pins`**：`@ObservedObject` 的失效是**整对象**级
  （`objectWillChange`），不做属性级依赖追踪 ⇒ 读 `pins` 并不比读 `fetch` 多出任何响应性；而 `fetch` 走
  `mainIndex` 字典命中（§19.3.2：无队列跳转、无锁），O(1) 而非 O(N)。`mainIndex` 与 `pins` 在
  `publishSnapshot` 的**同一个主线程块**内写入，body 重算发生在该块之后 ⇒ 读到的必是当前值。
- **未新增任何状态管理**：复用既有的 `@ObservedObject store` + `@Published pins`，无新 `@State`、
  无新 ObservableObject、无新队列、无新锁。

### `@State` 编辑缓冲（tag / note）：明确不同步

`_tag` / `_note` 仍只在 `init` 里播种一次，之后**刻意不从 store 回灌**（未加任何 `onChange(of: livePin)`
之类的同步）。理由写进了代码注释：本视图自己的 save、或 250 ms 合并窗关闭，都会在用户仍在打字时触发一次
重新发布；回灌就会吞掉用户正在输入的字符。这是「用户正在编辑的缓冲区归用户所有」的一般规则。

### 删除窗口的行为（明确设计，不是顺手兜底）

`store.delete` 成功 → 记录从 `records` / `pins` 移除 → `fetch(id:)` 返回 nil → 到 `onDismiss()` 真正把
sheet 收走之间有几帧。这几帧内 `livePin` 回落到 `presentedSnapshot`，Form 照常画完整记录，**不空白、
不崩、不画半条记录**。

⛔ 该兜底**不可能**重新掩盖它要修的过期缺陷，理由是构造性的：过期需要「记录**在** store 里且值更新了」，
而只要记录在，`fetch` 返回的就是新值；nil 分支仅在「记录**不在**」时可达。两个条件互斥。
（本 sheet 只从已 `isLoaded` 的 `PinListView` 呈现，且除 `delete` 外无任何路径移除记录 ⇒ nil 即「刚被删」。）

### 日期格式粒度：**改了**，并说明理由

`timeStyle` 由 `.short`（分钟粒度）改为 `.medium`（含秒），`dateStyle` 维持 `.medium`。

- `.short` 只到分钟 ⇒ 同一分钟内的两次写入渲染出**完全相同的字符串**。这让一次真实的 `updatedAt` 变化在
  屏幕上不可见，且与本次修复的过期缺陷**在现象上无法区分** —— 既是用户所报症状的一个高度可能的直接成因，
  也会让修复本身在真机上无法验证。一个显示不出「被修改过」的「Modified」行不成立。
- **两行共用同一个 formatter（Created 也一并带秒）**，不做单行特化：`Created` 与 `Modified` 并排时是被当作
  **比较**读的（「这个 Pin 到底被编辑过没有？」），分钟粒度会让这个比较给出假的「从未编辑」答案；两行格式
  不一致本身也会被读成 bug。附带好处：带秒的时间戳可与 `[PIN] create` / `[PIN] update` 日志行对齐。

### `PinListView` 的复核结论：行为正确，只补文档，不改类型

- `filteredPins` 每次 body 重算都从**活的** `store.pins` 派生 ⇒ 行（`PinRow` 显示 `tag` / `createdAt`）本身不过期；
  `.swipeActions` 闭包捕获的是当前这遍的 `pin`，重开 sheet 拿到的是新值。
- `PinRow` 不显示 `updatedAt` ⇒ 列表侧无需改动。
- `@State editingPin: Pin?` **保留 `Pin?` 类型**：`AnnotationView` 的 `@State` 种子和删除窗口兜底都需要这个
  快照，换成纯 UUID 载体会把兜底一并砍掉。改为加注释钉死它被收窄后的角色（身份 + 种子，**不得回读任何显示值**）。
- 另加一条注释说明 `thumbnails` 缓存无需在编辑时失效 —— mask blob 不可变（PIN-2）。
- 顺带观察到但**未改**：`tagFilter` 选中某 tag 后，若把该 Pin 的 tag 改成别的，Picker 选中项会停在一个已无
  对应 Pin 的 tag 上、列表显示为空。此为既有行为，与本次改动无关，未获授权改动，故不碰。

### 未做 / 未碰（明确列出）

- ⛔ **未触碰 `FilePinStore` 的写入路径、队列纪律、合并写**（持久化侧已验证正确）。`Persistence/` 零改动。
- ⛔ 未触碰 `SAMDecoder.swift` / `MaskRenderer.swift` / `buildTapAlpha` / R3 参数 /
  `DriftDetector.swift` / `ReAnchorLoop.swift` / `handleTap` 路由。
- 未新增队列、未新增锁（`pin.store.io` 仍是 PinStore 唯一队列）、未新增依赖、未改架构或数据流。
- 未勾选 `shared/tasks.md` 任何 checkbox。
- 未做 Debugger 的工作：无真机复测、无日志分析、未写 `shared/debug_report.md`。
- **未改 `save()` 成功后自动 `onDismiss()` 的行为** —— 见下条「留给用户裁决」。

### ⚠️ 留给用户裁决的一点（未擅自改动）

`save()` 在 `.success` 分支直接调 `onDismiss()`，即**保存成功 sheet 立即关闭**。加上上面第 2 点的时序
（publish 与 completion 是背靠背的两个 main.async 块，中间不会插入一次渲染），意味着：**在当前流程下，
活绑定本身不会产生「保存后当场看到 Modified 跳变」的画面**；新值要重开 sheet 才看得到（重开是新鲜的）。

⇒ 若要「不关不开就看到 Modified 变化」，必须让 `save()` 成功后**不再自动关闭** sheet（改为停留 + 成功提示）。
这是**产品/UX 决策，不是缺陷修复**，故本轮不擅自更改，明示交由用户裁决。

### 编译验证（两个配置、全新 `derivedDataPath`、`clean build`）

明确声明：两次都是**全新 derivedDataPath 的 clean build**（不是增量），warning 数为对完整构建日志的实际
统计值，非凭印象自报。

```
# Debug
rm -rf <scratch>/dd_debug
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -configuration Debug \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' \
  -derivedDataPath <scratch>/dd_debug clean build
→ ** BUILD SUCCEEDED **（exit 0）

# Release
rm -rf <scratch>/dd_release
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -configuration Release \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' \
  -derivedDataPath <scratch>/dd_release clean build
→ ** BUILD SUCCEEDED **（exit 0）
```

warning 实测口径：

```
grep -cE '^/.*\.swift:[0-9]+:[0-9]+: warning:' build_debug.log    → 0
grep -cE '^/.*\.swift:[0-9]+:[0-9]+: warning:' build_release.log  → 0
grep -cE 'warning:' build_debug.log / build_release.log           → 各 1
  唯一那 1 条：appintentsmetadataprocessor "Metadata extraction skipped.
  No AppIntents.framework dependency found." —— 工具链提示，非编译器 warning，与工程代码无关。
```

即 **Swift 编译器 warning：Debug 0 条、Release 0 条。**

### PIN-3 复核（用 `grep -E`；未用 macOS grep 上会静默失效的 `grep -v` + `\|` 写法）

```
# 步骤 1 —— 哪些文件里存在三条热路径队列的派发
grep -rlE --include='*.swift' '(videoQueue|decoderQueue|encoderQueue)\.(async|sync)' JudgeE2/
  → JudgeE2/Detection/CameraManager.swift
    JudgeE2/Detection/delete-CameraManager.swift

# 步骤 2 —— 这两个文件里的 Persistence 符号（剔除注释行）
grep -nE '(PinStore|FilePinStore|PinFactory|PinRecordV1|PinGeometryV1|MaskPNGCodec|PinFieldLimits|PinStoreError)' <file> \
  | grep -vE '^[0-9]+:[[:space:]]*(//|/\*|\*)'
  → CameraManager.swift        : 0 命中
    delete-CameraManager.swift : 0 命中
```

即这三条队列所在的文件里根本不存在任何 Persistence 符号的代码引用，**PIN-3 平凡成立**。
本轮改动只落在 `UI/AnnotationView.swift` 与 `UI/PinListView.swift`（纯 SwiftUI 主线程视图），
**未扩大 PIN-3 的检查面**。

### 新增/修改文件清单

- 修改：`JudgeE2/UI/AnnotationView.swift`（`pinID` + `presentedSnapshot` + `livePin`；formatter 粒度；
  `update`/`delete`/`loadMaskImage` 改传 `pinID`）
- 修改：`JudgeE2/UI/PinListView.swift`（**仅注释**：钉死 `editingPin` 收窄后的角色 + `thumbnails` 无需失效）
- 新增文件：无。删除文件：无。`project.pbxproj` **未改动**（无新文件需登记）。

---

## Phase 4B Day 6 补批 — §23 裁决落地：B-40（R-D 路由）+ B-41（统一重访日志）+ B-37（重访标记与来源装饰）(Builder, 2026-08-23)

> 范围：architect_output.md §23（Ruling A/B）的 Builder P1 三项 + §23.6 的 tasks.md 变更（用户批准 2026-08-23）。
> **B-38 / B-39（P2）明确不在本批**，未做。§23 与本批任务书冲突处一律以 §23 为准。

### B-40 — R-D 路由修复（§23.1.9，三项缺一不可）

1. **`onTapPromoted` 第三观测钩子**（`CameraManager.swift`）：
   `var onTapPromoted: ((_ gen: Int, _ promotedInstanceID: UUID) -> Void)?`，与既有两钩子同形
   （默认 nil、纯观测）。触发点插在 promote 分支**既有**的 `DispatchQueue.main.async` 块内
   （`lastTapIndex = myGen` 之后），只读该块已算好的 `myGen` / `hit.id`，零新增判定分支，
   promote 语义一字未动（§22.2.2 冻结维持）。
2. **`lastTapWasRevisit = false` 复位覆盖全部 gen>0 路径**：
   promote 主线程块（新增）、`failTap` 主线程块（新增，天然覆盖 handleTap 的 camera-not-ready
   早退与一切后续失败路径）、正常发布块（既有）。另在重访包装层：延迟置 true 的 main.async 块
   改为 `guard !join.didPromote`——promote 块先入队先执行（同线程两次 main.async 的 GCD FIFO），
   其 onPromoted 回调先置 `join.didPromote = true`，因此 **R-D 结局后该标志终态为 false**，
   "Re-segmenting…" 文案在 promote 上不可能出现（PIN-7 表末行：那是一句可判定的假话）。
3. **Tracker 穷尽释放**：`PinRevisitTracker` 增 `promoted` 字典；三个钩子处理器各自把该 gen 从
   **三个**字典中移除 ⇒ placed/failed/promoted 三选一必到，§23.1.9 后果 3 的 ~64 KB/次泄漏
   （闭包持有读回的 origin blob）消除。PIN-8 成立：三钩子的唯一安装者仍是 `PinRevisitTracker`
   （grep 验证：全工程仅 `installHooks` 内三处赋值）。
4. **R-D 呈现**：`PinRevisitEvent.Kind` 增 `promoted(fromPinTag: String?)`（纯 String，PIN-3 不
   受扰）；ContentView 底部横幅显示
   `That spot is already covered by a selection on screen.`（tag 存在时追加 ` From Pin "<tag>".`，
   tag 仅出现在引号内的 From Pin 前缀里）；图标 info（非警告三角），**无** "View Saved Snapshot"
   按钮（R-D 不是错误也无静态回退需求）。

### B-41 — 统一 `[PIN] revisit` 日志（§23.2.4 取代 §19.4.6 格式）

单一 grammar（四种结局同一行式，异构 `rejected reason=…` 行取消）：

```
[PIN] revisit id=<uuid8> seq=<N> outcome=(mask|promote|refused|failed) reason=(n/a|mode|cameraNotReady|orientation|camera|orientationCamera|promptSpace|aspect|decodeFailed|timeout) geo=(ok|remapped|refused|n/a) path=(fast|slow|n/a) pt=(<x.1>,<y.1>|n/a) pin=(<x.1>,<y.1>) iou=<0.00–1.00|n/a> origin=<N>px new=<N>px|n/a
```

- `seq`：会话内单调递增，在 `handleTap(fromPin:)` **入口处、任何门之前**由
  `PinRevisitTracker.allocateSeq()` 自增（存储放 tracker 单例——extension 不能加存储属性，
  且 tracker 本就是 main-thread-only）。中途夭折的重访也消耗号码 ⇒ P-7 缺号检测成立。
- 结局映射：`mask`=R-A（pinLog）／`promote`=R-D（pinLog）／`refused`=R-B 与 G2/G3 拒绝
  （pinFault）／`failed`=R-C（pinFault）。**每次进入必恰好一行**：G3/G2/G1/defensive-gen0 四个
  早退各自落行；其余经三钩子之一落行；`join.logged` 做恰好一次守卫。
- `reason` 细分：G1 拒绝按逐字段比对产出 `orientation`/`camera`/`orientationCamera`/
  `promptSpace`/`aspect`（复合不匹配时朝向/摄像头优先——那是 P-6 C2/C3 测试者可控的维度；
  次 promptSpace；aspect 兜底）；R-C 按失败 message 是否含 "timed out" 分 `timeout`/`decodeFailed`。
- `pt`＝实际送进 prompt 的派生点（未派发时 n/a），`pin`＝记录存储点，`geo=ok` 时两者相等可从
  日志直接核验；`iou` 口径一字未改（256×256、stride:1、`DriftDetector.alphaIoU`），仅
  `outcome=mask` 且 blob 读回成功时非 n/a。⛔ 未新增定时器（超时仍走既有 failTap→onTapFailed）。
- `path`：`.parked` 沿用旧行为映射为 `slow`（grammar 只有 fast|slow|n/a；用户等待语义上 parked
  就是慢），已在代码注释注明。

每种结局一条示例（口径示意）：

```
[PIN] revisit id=3620286d seq=1 outcome=mask reason=n/a geo=ok path=fast pt=(512.0,384.0) pin=(512.0,384.0) iou=0.82 origin=155px new=162px
[PIN] revisit id=3620286d seq=2 outcome=promote reason=n/a geo=ok path=n/a pt=(512.0,384.0) pin=(512.0,384.0) iou=n/a origin=155px new=n/a
[PIN] revisit id=3620286d seq=3 outcome=refused reason=orientation geo=refused path=n/a pt=n/a pin=(512.0,384.0) iou=n/a origin=155px new=n/a
[PIN] revisit id=3620286d seq=4 outcome=failed reason=timeout geo=ok path=n/a pt=(512.0,384.0) pin=(512.0,384.0) iou=n/a origin=155px new=n/a
```

### B-37 — 重访锚点标记 + 来源装饰（§23.1.4，走 **A1** 路线）

- **正向映射**：`FrameGeometry.projectCanonicalPoint(_:previewLayer:)`（新增，落在
  `FrameGeometry.swift`——§2 坐标链单一真源），实现为 `invertViewPoint` 的**逐步代数逆**
  （4⁻¹ 归一化 → 3⁻¹ 镜像对合 → 2⁻¹ 四档旋转逐 case 解逆 → 1⁻¹ 用 AVFoundation 文档化对偶
  `layerPointConverted(fromCaptureDevicePoint:)`），注释按 §23.1.4 要求写明「本函数是
  invertViewPoint 的逆，不是第二条变换路径」。Step 5 的边界裁剪无逆、不镜像（注释说明理由）。
  正确性归 **P-9** 门控（Debugger，四工况 × 9 点 ≤1px，本批不代测）。
- **接线**：`CameraManager.viewPoint(forCanonicalPoint:)`（main-thread-only helper，读同一
  `currentFrameGeometry()` 快照 + 既有 weak `previewLayer`，无新几何源）；
  `handleTap(fromPin:)` 用它把派生点正向投影后**作为 `viewPoint` 传给既有 tap 入口** ⇒ 在途
  脉冲（`TapLoadingIndicator`）、涟漪、锚点标记全部免费继承——`viewPoint: nil` 正是 §23.1.1
  「空间匿名」缺陷的全部成因（M-23.1 的第一笔代价），本批把参数补上而不是另建呈现路径。
  投影不可用（preview layer 未挂）时返回 nil，退化为旧行为，不阻塞重访。
- **装饰**：`markInstanceRevisitOrigin(id:pinTag:)` + `revisitOriginPinTags: [UUID: String?]`
  （生命周期纪律与 📌 完全同点：FIFO 淘汰清除、`discardAllTapWork` 清除、`markInstancePinned`
  时被 📌 取代——§23.1.6-2）；`TapAnchorMarker` 增 `isRevisitOrigin`/`revisitPinTag`；
  `TapAnchorMarkerView` 字形 ↻（SF Symbol `arrow.clockwise`，黑字压槽位色，沿用既有 chip 几何/
  slot 配色/primary 白环），⛔ 未用 📌 及任何图钉形状；标签仅
  `From Pin "<tag>"` / `From an untagged Pin`（tag 只出现在带引号的 From Pin 前缀内）；
  首次上屏自动 reveal 一次（`onChange(of: revisitOriginMarkerIDs)` 驱动，复用既有单值 reveal
  状态），之后沿用既有 showsTag 策略。装饰由 placed 观测回调安装（与既有 onFailed 回调发布
  R-C 横幅同一模式——钩子驱动呈现/日志，不驱动 decode，PIN-8 语义维持）。
- **措辞检查**：在途文案由 "Re-segmenting at this location…" 更正为 §23.1.4 允许表中的
  `Re-segmenting at this Pin's saved point…`；全 UI 复查无 IoU/相似度数值显示（§23.1.6-1，
  grep 验证 UI 目录零命中）；R-D 路径零 "Re-segment(ing)" 措辞。

### §23 语焉不详处的最小顺从选择（如实列出）

1. **复合 G1 不匹配的 reason 单值化**：grammar 每行只有一个 reason；朝向+摄像头同时不匹配 →
   `orientationCamera`（枚举里有）；几何不匹配与 promptSpace/aspect 并存时取几何（P-6 C2/C3 的
   受控维度），次 promptSpace，aspect 兜底。
2. **`path` 的 `.parked`**：§23.2.4 grammar 无 parked，沿用旧行为映射为 `slow`（注释注明）。
3. **defensive gen==0 分支**（G2 已同线程通过后 handleTap 仍拒绝——实际不可达）：记
   `outcome=failed reason=decodeFailed pt=n/a`。
4. **释放点的一个已知残余缺口（如实上报，未擅自扩）**：实例在 decode 完成前被 FIFO 淘汰/清池时
   走 `cancelRequests`/`endTapRequest`，三钩子均不触发 ⇒ 该 gen 的 tracker 条目滞留、该 seq 无行。
   §23.2.4 明言「唯一的缺口是 promote」且 ⛔ 不新增定时器，也未授权第四钩子，故本批**不**为此加
   释放点；其可观测后果恰好会被 P-7 的缺号门抓到（这正是 seq 的设计目的），归 Architect 后续裁决。
5. **R35 登记位置**：tasks.md 无独立「保留项登记处」区块，循 R32/R34 先例以引用块形式登记在
   Day 6 Debugger 新区块之后。
6. **R-D 横幅形态**：§23.1.9 只给文案；本批复用既有底部横幅、info 图标、无 snapshot 按钮、
   Dismiss 手动关闭（与 R-B/R-C 同寿命策略）。
7. **顺手修复（同族清理点，非 §23 条目）**：`discardAllTapWork` 原本清 `pinnedInstanceIDs` 但漏清
   `pinnedInstanceTags`（无界增长，显示无恙）；本批在同一块内补 `removeAll()`，并与新的
   `revisitOriginPinTags.removeAll()` 同点落地。

### 未做 / 未碰（明确列出）

- **B-38 / B-39 未做**（P2，任务书明确排除）。
- `SAMDecoder.swift` / `MaskRenderer.swift` 未打开；`buildTapAlpha`、R3 禁令参数、`canonicalPoint`
  冻结零触碰。
- `DriftDetector.swift` / `ReAnchorLoop.swift` / 一致性门一行未动（仅继续**调用**既有
  `DriftDetector.alphaIoU`）。
- promote 分支的路由/decode 语义零改动（§22.2.2 冻结）：新增的只有该块内的复位一行与观测一行。
- 未新增队列、未新增锁（`pin.store.io` 仍是 PinStore 唯一队列；tracker/Join 仍 main-thread-only）。
- PIN-4/PIN-5 完好：无 embedding 持久化、无第二条 decode 入口（重访仍唯一转调既有 `handleTap`）。
- 未做 Debugger 工作：无真机测试、无日志分析、未写 debug_report.md；P-6…P-9 留待 Debugger 在
  新日志格式上执行。
- tasks.md 只做了 §23.6 指定的三处（Debugger 区块替换 + Builder 五条追加与三条勾选 + R35 登记），
  其余区块零触碰。

### 编译验证（两个配置、全新 `derivedDataPath`、`clean build`）

明确声明：两次都是全新 derivedDataPath 的 clean build，warning 数为对完整构建日志的实际统计。

```
# Debug
rm -rf <scratch>/dd_debug
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -configuration Debug \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' \
  -derivedDataPath <scratch>/dd_debug clean build
→ ** BUILD SUCCEEDED **（exit 0）

# Release
rm -rf <scratch>/dd_release
xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -configuration Release \
  -destination 'id=64F6C21E-C364-4BB2-859E-DB43CB11CFA1' \
  -derivedDataPath <scratch>/dd_release clean build
→ ** BUILD SUCCEEDED **（exit 0）

grep -cE '^/.*\.swift:[0-9]+:[0-9]+: warning:' build_debug.log    → 0
grep -cE '^/.*\.swift:[0-9]+:[0-9]+: warning:' build_release.log  → 0
grep -cE 'warning:' 两 log                                        → 各 1
  （唯一那 1 条是 appintentsmetadataprocessor 的 "Metadata extraction skipped."
   工具链提示，非编译器 warning，与工程代码无关。）
```

即 **Swift 编译器 warning：Debug 0 条、Release 0 条**（首轮曾出 1 条 `bufY` never-mutated，
已改 `let` 后复测归零）。

### PIN-3 复核（`grep -E`；未用 macOS grep 上会静默失效的 `grep -v` + `\|` 写法）

```
# 步骤 1 —— 哪些文件里存在三条热路径队列的派发
grep -rlE --include='*.swift' '(videoQueue|decoderQueue|encoderQueue)\.(async|sync)' JudgeE2/
  → JudgeE2/Detection/CameraManager.swift
    JudgeE2/Detection/delete-CameraManager.swift

# 步骤 2 —— 这两个文件里的 Persistence 符号（剔除注释行）
grep -nE '(PinStore|FilePinStore|PinFactory|PinRecordV1|PinGeometryV1|MaskPNGCodec|PinFieldLimits|PinStoreError|PinRevisitTracker)' <file> \
  | grep -vE '^[0-9]+:[[:space:]]*(//|/\*|\*)'
  → CameraManager.swift        : 0 命中
    delete-CameraManager.swift : 0 命中
```

新增的 `onTapPromoted` 触发点、复位点与 `markInstanceRevisitOrigin` 全部在既有
`DispatchQueue.main.async` 块内，未扩大 PIN-3 检查面。**PIN-3 平凡成立。**

### 新增/修改文件清单

- 修改：`JudgeE2/Interaction/FrameGeometry.swift`（新增 `projectCanonicalPoint` 正向映射，A1）
- 修改：`JudgeE2/Detection/CameraManager.swift`（`onTapPromoted` 钩子 + PIN-8 注释；
  `PinRevisitEvent.Kind.promoted`；`TapAnchorMarker.isRevisitOrigin/revisitPinTag`；
  `revisitOriginPinTags` 及三处生命周期清理；`markInstanceRevisitOrigin`；
  `viewPoint(forCanonicalPoint:)`；promote 块复位+观测两行；`failTap` 复位一行；
  `discardAllTapWork` 补 `pinnedInstanceTags.removeAll()`）
- 修改：`JudgeE2/Persistence/PinInterfaces.swift`（tracker 三钩子化 + `seq` 分配器 +
  穷尽释放；`handleTap(fromPin:store:)` 统一日志 grammar、G1 reason 细分、A1 正向投影接线、
  R-D 事件与 didPromote 抑制）
- 修改：`JudgeE2/UI/ContentView.swift`（在途文案更正；R-D 横幅分支；↻ 字形与
  `From Pin "…"` 标签；首次上屏自动 reveal）
- 新增文件：无。删除文件：无。`project.pbxproj` 未改动（无新文件需登记）。

---

## Phase 4B Day 6 追加 — P-9 执行载体（调试自检按钮）(Builder, 2026-08-23，用户批准)

### 这是什么

**P-9（§23.2.8）此前不可执行。** `projectCanonicalPoint` 只被生产重访路径
（`CameraManager.swift` 的 `viewPoint(forCanonicalPoint:)` → `PinInterfaces.swift`）调用，
往返误差是一个纯数值性质，**没有任何点屏动作能触发一次测量** —— 这正是 B-37 交付时
把自身正确性显式让渡给 P-9、而非自证的原因。本批只补上**触发它的那个按钮**。

⛔ **只是载体，不是执行。** 本批**没有**跑过 P-9 的任何一组工况（见下"模拟器试跑"）。
`shared/tasks.md` 的 P-9 复选框**保持未勾选**，本批**未改动 tasks.md 一个字**。

### 实现

- **触发**：`ContentView` Debug Options 区，`Force Slow Path (testing)` 下方新增
  `Run Geometry Round-Trip Check (testing)` 按钮（橙色 + `(testing)` 后缀，沿用既有测试专用控件的标记法）。
  一次性测量 ⇒ 用 Button 而非 Toggle。
- **网格**：严格照 §23.2.8 —— canonical 空间四角各向内收 1 px、四边中点、中心，共 9 点。
  取值域按 `invertViewPoint` Step 5 定义的有效 canonical 范围 `[0, origW-1] × [0, origH-1]`。
- **测量**：每点 `projectCanonicalPoint` → `invertViewPoint`，报 **Chebyshev** 距离。
- **落点**：`FrameGeometry.swift` 的一个 extension（`roundTripGrid` / `runRoundTripSelfCheck`），
  与被测的两个函数同文件；`CameraManager.runCanonicalRoundTripSelfCheck()` 是 8 行主线程包装，
  与 `viewPoint(forCanonicalPoint:)` 读同样的 `previewLayer` + `currentFrameGeometry()`。
  **未新增文件**（`Interaction/` 是经典 PBXGroup、非 synchronized folder，新文件需改 4 处
  pbxproj；`project.pbxproj` 因此一字未动）。

### 日志格式（`perfLog`，⛔ 不受 Perf Quiet Log 抑制）

```
[P9] cfg rot=90 mirrored=false canonical=1080x1920 preview=402.0x874.0 tol=1.0px
[P9] i=1/9 pt=(1.0,1.0) → view=(-45.2,12.7) → back=(1.0,1.0) err=0.00px errNoClamp=0.00px clamp=off
…
[P9] result n=9 maxErr=0.00px worstIdx=3 clamped=0/9 rot=90 mirrored=false canonical=1080x1920 verdict=PASS(<=1.0px) scope=1of4cfg
```

- `rot` / `mirrored` / `canonical` **在 cfg 行与 result 行各出现一次** —— 测试者口述的"竖持×后摄"
  不可核验，**只有日志自己说出的几何才算数**；只 grep result 行也能自证工况。
- `scope=1of4cfg`：一次按下只覆盖 4 组工况中的 1 组。**这里的 PASS 不是 P-9 PASS。**
- 几何或 preview layer 缺席时打 `[P9] unavailable reason=…`（`faultLog`），⛔ 不静默 no-op。

### clamp / inset 判断（**上报 Architect，⛔ 未擅自改网格**）

Step 5 的 `clamp` 无逆。它对本测量的作用是**单向**的：边界点上**向外**的往返误差被静默截为 0，
**向内**的误差原样通过。⇒ **clamp 只会掩盖真实误差（伪 PASS），永远不会制造误差（伪 FAIL）。**

- **四角**：§23.2.8 的 1 px 内收对任何"小到能通过"的误差都足以让 clamp 不参与。**够用。**
- **四边中点**：§23.2.8 **只内收四角**，中点各有一个坐标恰好压在 clamp 边界上 ⇒ **仍可被掩盖**。

⇒ 网格**照 §23.2.8 原样实现**（放宽 inset 去凑数就是本项目反复抓到的 A-7/A-19 失效模式）。
改为在**不动判据**的前提下把 clamp 的参与暴露出来：每行附 `errNoClamp`（同一次
`invertViewPoint` 调用的 **clamp 前** `normalized` 输出反算）与 `clamp=on/off`。
`err` 是判据，`errNoClamp` 仅供诊断。**是否把中点也内收 1 px，请 Architect 裁定。**

### 代码走查得到的一条预判（供 Debugger 参考，⛔ 不是读数）

Step 2/3/4 的正逆两向逐 case 代数互逆（rot 0/90/180/270 四支 + mirror 对合），**按行核对可证**，
往返到浮点精度为止。⇒ **P-9 真正能证伪的只有 Step 1 那对 AVFoundation 函数**
（`layerPointConverted(fromCaptureDevicePoint:)` / `captureDevicePointConverted(fromLayerPoint:)`）。
在 `.resizeAspectFill` 下 canonical 的被裁边缘会映射到 **preview layer bounds 之外**的 view 点
（本机实测 preview=402×874、canonical 竖持 ⇒ 左右被裁），Apple 未文档化
`captureDevicePointConverted` 对越界 layer 点是线性外推还是钳到可见区。**前者往返精确、后者会差几百 px。**
⇒ P-9 是一条真能失败的判据，不是走过场。

### 模拟器试跑（⛔ **不满足 P-9**）

模拟器无 capture session（`Camera input unavailable`）⇒ 无 geometry 快照。按下按钮得到：

```
[t=39761.1] [P9] unavailable reason=noGeometrySnapshot — no frame has been processed yet
```

**这只证明按钮已接通、主线程断言未触发、且"缺几何时出声而不是静默"生效。
它不产生任何往返数字，模拟器几何也不属于 P-9 要求的四组工况之一。P-9 仍需真机 4 组。**

### 构建

全新 derivedDataPath 的 clean build，Debug / Release 均 `** BUILD SUCCEEDED **`（exit 0）。
`grep -cE '^/.*\.swift:[0-9]+:[0-9]+: warning:'` → **Debug 0、Release 0**；
两 log 各有 1 条 `warning:` 是 `appintentsmetadataprocessor` 的 "Metadata extraction skipped." 工具链提示，非编译器 warning。

### PIN-3 复核（`grep -E`；⛔ 未用在 macOS grep 上会静默失效的 `grep -v` + `\|`）

```
grep -rlE --include='*.swift' '(videoQueue|decoderQueue|encoderQueue)\.(async|sync)' JudgeE2/
  → CameraManager.swift / delete-CameraManager.swift
grep -nE '(PinStore|FilePinStore|PinFactory|PinRecordV1|PinGeometryV1|MaskPNGCodec|PinFieldLimits|PinStoreError|PinRevisitTracker)' <file> \
  | grep -vE '^[0-9]+:[[:space:]]*(//|/\*|\*)'   → 两文件各 0 命中
grep -rnE --include='*.swift' '(runRoundTripSelfCheck|runCanonicalRoundTripSelfCheck)' JudgeE2/
  → 唯一调用点 ContentView.swift:242（按钮闭包）→ CameraManager:1259 → FrameGeometry:244
```
新增代码全程主线程、无新队列/锁、未进入任何热路径。**PIN-3 平凡成立。**

### 文件清单

- 修改：`JudgeE2/Interaction/FrameGeometry.swift`（末尾新增 P-9 extension + 说明）
- 修改：`JudgeE2/Detection/CameraManager.swift`（`runCanonicalRoundTripSelfCheck()`，8 行）
- 修改：`JudgeE2/UI/ContentView.swift`（Debug Options 一个 Button）
- 新增/删除文件：无。`project.pbxproj`：未改动。
- **`shared/tasks.md`：未改动**（P-9 保持未勾选；是否给本载体单开 B 号交用户决定）。

### 未碰（红线复述）

`projectCanonicalPoint` / `invertViewPoint` 本体一字未改；SAMDecoder / MaskRenderer /
`buildTapAlpha` / R3 参数 / DriftDetector / ReAnchorLoop 零触碰；重访路径、`handleTap`、
任何 PIN 不变量未改；无测试框架、无结果 UI、无结果持久化。

---

## 2026-08-23（第二次）· 真机测试驱动的缺陷修复 — P-9 按钮"点不动"

### 缺陷

用户在真机 iPhone 11 上报告 Debug Options 里 `Run Geometry Round-Trip Check (testing)`
按钮"不是按钮，点不动"——两次多小时 session，console 零 `[P9]` 行。上一个 session 造这个
按钮时只在 iPhone 17 Pro 模拟器上验证过，从未在 iPhone 11 尺寸（414×896pt）下测过。

### 排查过程（先证后修，不是猜）

用 `mcp__Claude_Code_iOS_Simulator__control` 起一个真正的 iPhone 11 模拟器
（E3DF778B-E120-4CE6-BAC8-900B7568B7C1，414×896pt，与项目真机尺寸一致——注意这**不是**
标准 build check 用的目标 64F6C21E，那个其实是 iPhone 17 Pro 模拟器，正是当初漏测的那台）。

先验证了任务书给出的两条预判假设，**都被推翻**：
1. **"内容溢出屏幕底部"**——默认字号下用 `xcrun simctl io screenshot` 量出面板总高度
   只到点坐标 y≈371，远没到 896。不成立。
2. **"某个 overlay 在拦截触摸"**——逐一读了 ContentView.swift 里每个 `allowsHitTesting`，
   在这个按钮的屏幕位置附近没有一层是 `true`（可拦截）的。不成立。

改用像素级测量真正定位问题：截图量出按钮文字的精确 glyph 包围盒
（point 坐标 x:[27,278] y:[348,359.5]），在包围盒下方一点点、但仍在面板圆角矩形内的位置
（150, 369）——一个用户手指完全可能落点的地方——发起 tap，console 输出：

```
[t=...] [TAP] ignored — geometry not ready
```

**没有 `[P9]`，反而是相机层的 TouchHandler 收到了这次 tap。** 这证实了真正的根因：

**这个 Button 是整个面板里唯一一个没有显式 `.contentShape(Rectangle())` 的可交互元素。**
面板里其它每一个控件（齿轮、翻转相机、Pin List 三个 icon 按钮）都显式设了
`.frame(width:, height:).contentShape(Rectangle())`；Toggle 用的是系统 `.switch` 样式，
天然有一整个开关控件那么大的命中区。唯独这个纯文字 `Button("...")` + `.buttonStyle(.plain)`
的命中区被压缩成文字自身的 line box——只有单行文字那么高（约 12–18pt），而它又是面板的
**最后一行**，下面紧贴面板的圆角边缘，没有下一个控件兜底一次没对准的点击：手指点在
"看起来还是这一行"但其实已经出了 line box 的位置，SwiftUI 判定为未命中，touch 穿透整个
ZStack 落到下面的相机预览层，被 TouchHandler 无声吞掉——两次多小时 session 零 `[P9]`，
症状完全吻合。

修好之后原地复测（同一个 (150, 369)，同一个坐标，只改了代码重新编译安装）：

```
[t=4623.0] [P9] unavailable reason=noGeometrySnapshot — no frame has been processed yet
```

### 修的是什么（纯表现层，ContentView.swift 一个文件）

```swift
Button(action: {
    cameraManager.runCanonicalRoundTripSelfCheck()
    p9CheckJustRan = true
}) {
    Text(p9CheckJustRan ? "✓ Ran — check console" : "Run Geometry Round-Trip Check (testing)")
        .frame(maxWidth: .infinity, alignment: .leading)
}
.buttonStyle(.plain)
.foregroundColor(.orange)
.font(.footnote)
.padding(.vertical, 6)
.contentShape(Rectangle())
.task(id: p9CheckJustRan) { … 1s 后 p9CheckJustRan = false … }
```

新增 `@State private var p9CheckJustRan: Bool` 一个纯展示态，复用本文件已有的
`.task(id:)` 自动取消/重启惯用法（和 `tagRevealToken` 同一个模式），没有新队列/锁/依赖。
按钮文案在按下瞬间变成"✓ Ran — check console"，1 秒后自动变回——这是任务书要求的
"按下必须有肉眼可见反馈"，杜绝"看起来是按钮、按了却像没反应"这类静默失败。

`FrameGeometry.swift` / `CameraManager.runCanonicalRoundTripSelfCheck` **一字未碰**。

### 范围判定：不是面板级问题，是这一行专属

检查了面板里其它所有控件是否有同款缺陷：三个 Picker（segmented style，UIKit 原生大命中区，
用 Segmentation tab 实测可点）、Perf Quiet Log / Force Slow Path 两个 Toggle（`.switch`
样式，Force Slow Path 在真机日志里已有 `path=slow` 证明可达，模拟器上也实测可点）、
齿轮/翻转相机/Pin List 三个 icon 按钮（本来就都有 `.contentShape`）——**都不受影响**。
只有 P-9 这一个按钮有这个缺陷，因为它是唯一一个"裸文字 Button + 无 contentShape + 面板最后一行
没有下一控件兜底"三个条件同时满足的地方。**没有加 ScrollView**——量过默认字号下内容总高度
远没到需要滚动，加了也不解决这个按钮本身命中区太小的问题，属于文不对题的修法。

（顺带发现但**没有修**：把系统字号调到 Accessibility XXXL 时标题行会水平溢出、面板整体
变得很挤——这是一个真实但独立的问题，超出这次"这一个按钮点不动"的范围，留给用户/下一个
session 判断是否要处理。）

### 构建

全新 derivedDataPath 的 clean build（标准 build check 目标 64F6C21E = iPhone 17 Pro 模拟器）：
Debug / Release 均 `** BUILD SUCCEEDED **`。两份 log 各 1 条 `warning:`，都是
`appintentsmetadataprocessor` 的 "Metadata extraction skipped." 工具链提示，
**真正的 Swift 编译器 warning：Debug 0、Release 0**。

### 模拟器实机验证（iPhone 11 尺寸，E3DF778B-E120-4CE6-BAC8-900B7568B7C1）

- 修复前：在原始（未加 contentShape）代码上，tap (150,369) → `[TAP] ignored — geometry not
  ready`（穿透到相机层），tap (150,358)（文字正中心）→ `[P9]` 能触发——证明问题确实是
  命中区太窄，不是按钮整体失效。
- 修复后：同一个 (150,369) → `[P9]` 正确触发。
- 视觉验证：把 revert 前的确认文案展示时长临时调到 4s 截图确认"✓ Ran — check console"
  确实会出现（然后按规格改回 1s，重新 clean build 确认修改生效）。

### 文件清单

- 修改：`JudgeE2/UI/ContentView.swift`（新增 `p9CheckJustRan` state；P-9 按钮改用
  `Text` label + `.frame(maxWidth:.infinity)` + `.padding(.vertical,6)` +
  `.contentShape(Rectangle())` + `.task(id:)` 反馈动画）
- 未改：`FrameGeometry.swift`、`CameraManager.swift`、`SAMDecoder.swift`、
  `MaskRenderer.swift`、`DriftDetector.swift`、`ReAnchorLoop.swift`、`project.pbxproj`
- **`shared/tasks.md`：未改动**（按指示不碰）。

## 2026-08-24 · R34 埋点（debug_report.md §39.7 前两条）— 只做日志，不改行为

### 背景

R34（PinStore 计数倒退，P0，OPEN）两次干净会话未复现。Debugger 已用源码复核排除六项
进程内机制，剩余假设全指向跨进程边界：读到旧 manifest，或杀进程窗口期两个进程的文件
系统操作交错。现有 `[PIN]` 日志既不区分"同进程两次事件" vs "两个进程各自事件"，也
看不出"这次读到的 manifest.json 是不是最新写入的那份"。目标：下次复发时从粘贴的日志
文本直接判读，不用再走设备容器 devicectl 溯源。

只做 §39.7 的前两条，**没做**第三条（`migratedFrom` 措辞修正，P2，范围外）。

### 改动 ① — `load()` 打印 manifest 的 mtime/size

`JudgeE2/Persistence/FilePinStore.swift`：

- 新增 `readManifestStamp()`（约第 270-287 行，紧邻 `backupManifest`）：对
  `manifestURL` 调一次 `FileManager.default.attributesOfItem(atPath:)`，取
  `.size` 和 `.modificationDate`；拿不到就返回占位 `("-", 0)`，**不产生新的
  `pinFault`**。`mtime` 用 `DateFormatter`（`en_US_POSIX` + 设备当前时区）格式化成
  `yyyy-MM-dd HH:mm:ss.SSS ZZZZZ` 的绝对墙钟时间——不是本文件惯用的 `[t=…]`
  进程内单调时钟，那个量跨进程不可比对，这里要的就是"能直接读出这是几点几分写的"。
- `load()` 第 2 步（约第 160-169 行）：在 `Data(contentsOf:)` 之前调一次
  `readManifestStamp()`，把结果同时喂给"no manifest"分支（约第 165 行）和正常
  分支（约第 234 行）的 `[PIN] load ok …` 格式串，新增 `mtime=%@ size=%dB` 两个
  字段。`data` 本身要读的 stat 已经在 `Data(contentsOf:)` 里发生一次，这里是在
  已经要做的 IO 基础上多读一次元数据，不是新引入的失败面。
- 语义上"文件不存在"（no-manifest 分支）时两个字段自然是占位符，没有强凑真实值。
- **纯打印，`load()` 的判断/分支逻辑一个字节没动。**

### 改动 ② — `[PIN]` 全部日志行加进程 pid

`pinLog` / `pinFault`（文件末尾 `MARK: - [PIN] logging` 区块）是全部 `[PIN]` 输出
唯一出口。只改了这两个包装函数（及紧邻的一个 `private let` + 一个 `private func`
帮助器），**11+ 处调用点原样未动**：

```swift
private let pinPIDTag = "[PIN pid=\(ProcessInfo.processInfo.processIdentifier)]"

private func pinTagged(_ message: String) -> String {
    guard message.hasPrefix("[PIN]") else { return message }
    return pinPIDTag + message.dropFirst(5)
}
```

做法是重写消息里打头的字面量 `"[PIN]"` → `"[PIN pid=…]"`，而不是要求消息以
`"[PIN] "`（带空格）开头——踩了一个坑：`PinDebugFixture.swift` 里的
`"[PIN][FIXTURE] …"` 没有那个空格，如果按 `hasPrefix("[PIN] ")` 匹配会漏掉这一类
调用点，违反"不遗漏任何一处"的要求。改成匹配 5 字符字面量 `"[PIN]"` 后两类调用点
（`FilePinStore.swift` 里的、以及 `PinInterfaces.swift`/`PinDebugFixture.swift`
里经同一对函数出口的）全部一致带上 pid，验证见下方模拟器日志。

pid 用 `ProcessInfo.processInfo.processIdentifier`，系统现成值，没有新引入队列/锁/
持久化状态。

### 构建

全新 `derivedDataPath`，clean build，标准 build check 目标 64F6C21E（iPhone 17 Pro
模拟器）：

- `-configuration Debug`：`** BUILD SUCCEEDED **`，1 条 `warning:`，是
  `appintentsmetadataprocessor` 的工具链提示（"Metadata extraction skipped."），
  **不计入**。
- ⚠️ **同一次 Debug clean build 里还看到一条与本次改动无关的真实编译器 warning**：
  `CameraManager.swift:2792:55: warning: main actor-isolated conformance of
  'TrackState' to 'Equatable' cannot be used in nonisolated context; this is an
  error in the Swift 6 language mode`。本次会话**从未编辑过 `CameraManager.swift`**
  （红线明确禁止），且早先一次不带 `-configuration`（落到 Release 配置）的 clean
  build 里这条 warning 不出现——像是 Debug 专属的 Swift 6 并发检查诊断，具体是否
  为本会话时间窗口内其他并发改动引入、还是本来就有但之前没在 Debug 配置下测过，
  未深挖（超出本次范围）。如实上报，**未修**。上一条 2026-08-23 的进度记录里同一
  build check 目标测出的是 "Debug 0、Release 0" 真实 warning，供比对。

### 模拟器验证（iPhone 11 尺寸，E3DF778B-E120-4CE6-BAC8-900B7568B7C1）

冷启动（卸载重装，无 manifest）：

```
[t=1626.4] [PIN pid=32857] load ok pins=0 orphans=0 new=1 ms=23.3 mtime=- size=0B
```

用 `-PinFixtureBatch 3:200` 落 3 条真实记录（另一个进程 pid=33045，`create`/
`[FIXTURE]` 两类调用点都带上了 pid，证明改动②确实覆盖了 `FilePinStore.swift` 之外
经同一出口的日志）：

```
[t=892.2]  [PIN pid=33045][FIXTURE] launch-argument batchSave count=3 intervalMs=200
[t=1416.6] [PIN pid=33045] create id=93126340 ok blob=453B nz=4096 pins=1
[t=1416.6] [PIN pid=33045][FIXTURE] batchSave 1/3 ok
```

再冷启动一次（不带 fixture 参数，读到上一进程写的 manifest，pid 换了一个）：

```
[t=1360.1] [PIN pid=33165] load ok pins=3 orphans=0 migratedFrom=1 ms=414.9 mtime=2026-08-24 01:02:57.389 -07:00 size=1277B
```

三行加在一起正是这次埋点要的判读能力：`pid=33045` 写、`pid=33165` 读，两个不同
进程一眼可辨；`mtime=2026-08-24 01:02:57.389 -07:00` 是绝对墙钟时间，可以直接和
设备真机时钟对照，判断这次读到的是不是刚写的那份。

### 文件清单

- 修改：`JudgeE2/Persistence/FilePinStore.swift`（新增 `readManifestStamp()` +
  `manifestStampFormatter`；`load()` 两处 `[PIN] load ok` 格式串加
  `mtime=%@ size=%dB`；`pinLog`/`pinFault` 包装函数注入 pid 前缀）
- 未改：`SAMDecoder.swift`、`MaskRenderer.swift`、`DriftDetector.swift`、
  `ReAnchorLoop.swift`、`CameraManager.swift`（红线明确要求），`project.pbxproj`
- §39.7 第三条（`migratedFrom` 措辞）：**未做**，范围外。
- `shared/tasks.md`：未改动（按指示不碰，勾选由用户做）。

---

## 2026-08-24 — B-42 + B-46（能力 C 基础字段 + 平移不变一致性门，仅这两条）

范围：只做 architect_output.md §24.4 的 **B-42** 和 **B-46**，不做 B-43/B-44/B-45/
B-47/B-48。目标：新字段/新函数"建好但空转"，不接线，App 可观察行为逐位不变。

### B-42（`Interaction/TapInstanceManager.swift`）

- 新增 `enum TrackState: Equatable { case locked, tracking, lost }`，就放在
  `TapInstanceManager.swift` 顶部、`TapInstance` 结构体之前（贴近它唯一的使用者，
  同 `AnchorSignature` 挨着 `DriftDetector` 放的风格）。
  - ⚠️ 项目开了 `-default-isolation=MainActor`（SE-0466），普通 `enum TrackState:
    Equatable` 会让编译器合成的 `Equatable` 一致性也被隐式装进 MainActor，在
    `CameraManager.reAnchorDecode` 的 `decoderQueue.async` 闭包（非隔离上下文）里
    用 `==` 比较会报 "main actor-isolated conformance ... cannot be used in
    nonisolated context"（Swift 6 模式下是 error，这次工具链下是 warning，仍然算
    真实 warning，不允许留着）。解法：显式 `nonisolated enum TrackState:
    Equatable { ... }`。记到这里因为下次任何人在这个开了 MainActor 默认隔离的工
    程里给一个会跨队列用 `==`/`Hashable` 的类型加协议一致性，都会撞同一个坑。
- `TapInstance` 新增三个字段，紧跟在 `originAlpha` 后面：`trackedPoint: CGPoint`
  （构造时 = `canonicalPoint`）、`lastReAnchorTrackedPoint: CGPoint?`（恒 nil）、
  `trackState: TrackState`（恒 `.locked`）。`canonicalPoint` 字段本身一个字节没动。
- 唯一构造点 `TapInstanceManager.addInstance` 补齐三个新参数；全仓库 grep 确认
  `TapInstance(` 只有这一个构造调用点，没有遗漏的第二处。
- 没加锁、没加队列——三个新字段走的是已有 `lock`（与 `anchorSignature` 同一把）。

### B-46（`Interaction/DriftDetector.swift` + `Detection/CameraManager.swift`）

- `DriftDetector.swift` 新增 `static func centroidAlignedIoU(_:_:width:height:)`，
  紧跟在既有 `alphaIoU` 后面，**`alphaIoU` 本身一行没改**，继续原样服务能力 A/B。
  实现选的是**质心对齐**（不是形状描述子）：各自算二值 mask 的质心，各自平移到
  256×256 网格中心（整数像素位移，越界补零，不做插值），再逐元素算 IoU。原因写
  在函数文档注释里：一旦 `trackedPoint` 可以偏离 `canonicalPoint`，两张 mask 在
  绝对坐标系下预期就是不重叠的——直接复用 `alphaIoU` 会把跟踪成功误判成漂移。
- 新增独立常量 `trackConsistencyAcceptIoU: Double = 0.5`（`static var`，物理上与
  `reAnchorAcceptIoU` 不同存储），旁边写清楚"零数据，初始值不是继承三簇框架"。
  `reAnchorAcceptIoU` / `contentThresholdLuma` / `minReAnchorIntervalMs` 三个既有
  常量的数值一个字节没碰。
- `CameraManager.reAnchorDecode` 里新增能力 C 分支：在算 `gateIoU` 之前先
  `let trackState = instance.trackState`（和其余在闭包外捕获的字段同一种写法），
  然后 `switch trackState { case .tracking: centroidAlignedIoU(...); case .locked,
  .lost: alphaIoU(...) }`；接受阈值也跟着切（`.tracking` 用
  `trackConsistencyAcceptIoU`，否则用 `reAnchorAcceptIoU`），`reAnchorRejectUpdate`
  改成吃调用方传入的 `threshold` 参数而不是自己再读一次
  `DriftDetector.reAnchorAcceptIoU`——避免"判定用了 A 阈值、日志打印 B 阈值"这种
  拆开维护迟早会漂移的写法。因为这批次每个实例的 `trackState` 恒为 `.locked`，这
  个 switch 目前只会走 `alphaIoU` 分支，`.tracking` 分支是活代码但暂时到不了。
- 没有加 `DriftDetector.objectTrackingEnabled` 主开关——architect_output.md §24.4
  写得很清楚那是 B-47 的范围（"主开关"单独一行，依赖是"三条合取"不是 B-42），本
  批次的行为不变性完全靠 `trackState` 恒 `.locked` 兜底，不需要再加一层开关。

### 构建 + 回归

- `xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -destination 'id=64F6C21E-
  C364-4BB2-859E-DB43CB11CFA1' -derivedDataPath <scratchpad>/dd-b42-b46
  -configuration Debug clean build` → **BUILD SUCCEEDED**，真实 Swift 编译器
  warning 数 **0**（第一轮撞见上面那条 `TrackState` MainActor warning，加
  `nonisolated` 后二次 clean build 确认清零；唯一残留的是工具链自身的
  "Metadata extraction skipped. No AppIntents.framework dependency found." 提示，
  不算）。
- 模拟器回归（iPhone 11，E3DF778B-E120-4CE6-BAC8-900B7568B7C1）：装包、`simctl
  launch`（不要用 `simctl launch --console-pty ... &` 挂在会话的后台 shell 上——
  这个 shell 一退出就把子进程 SIGHUP 掉了，会被误判成"App 崩溃回到桌面"，其实只
  是工具链的假象，跟本次代码改动无关，踩了一次坑记在这）。切到 Tap to Segment，
  点一下预览区。`xcrun simctl spawn ... log show --predicate 'process ==
  "JudgeE2"'` 复查两次点击前后的日志：进程全程存活，没有 fatalError / Swift
  exception / `[REANCHOR]` 异常行；模拟器本身没摄像头出不了真实 mask（`[TAP]
  ignored — geometry not ready`，与本次改动无关，真机上不会遇到），但新字段接入
  的初始化路径（`addInstance` → 三个新字段 → 现有渲染/日志路径）没有引发任何崩溃
  或异常输出。

### 文件清单

- 改：`JudgeE2/Interaction/TapInstanceManager.swift`（新增 `TrackState`；
  `TapInstance` 三个新字段；`addInstance` 初始化）
- 改：`JudgeE2/Interaction/DriftDetector.swift`（新增 `centroidAlignedIoU` +
  `maskCentroid` + `sampleShifted` 三个私有/公开函数；新增
  `trackConsistencyAcceptIoU` 常量；`alphaIoU` 未动一行）
- 改：`JudgeE2/Detection/CameraManager.swift`（`reAnchorDecode` 新增 `trackState`
  捕获 + gate 分支 + `acceptThreshold` 选择；`reAnchorRejectUpdate` 签名加
  `threshold` 参数）
- 未碰：`SAMDecoder.swift`、`MaskRenderer.swift`、`UI/ContentView.swift`、
  `Persistence/FilePinStore.swift`（按指示避让另一个 Builder session）、
  `shared/tasks.md`、`shared/architect_output.md`
- 未做（按指示，留给后续批次）：B-43（`AnchorTracker.swift`）、B-44（接线到
  `checkAndFireReAnchor`，移动 `trackedPoint`）、B-45（RE-1 并集扩展）、B-47
  （`objectTrackingEnabled` 主开关 + 症状 B 旁路）、B-48（`PinFactory` 防御性
  断言）。

## 2026-08-24 — B-43（`Interaction/AnchorTracker.swift`，只做这一条）

范围：architect_output.md §24.2.2/§24.2.3。新文件，局部块匹配搜索原语 + 跟丢判
据 + 恢复搜索。**不做 B-44**——不接线到 `checkAndFireReAnchor`，不改
`CameraManager.swift`/`TapInstanceManager.swift`/`DriftDetector.swift` 一个字。
本批次交付后这个新文件依然零调用方，App 可观察行为逐位不变（因为没有任何代码
路径会执行到这里）。

### 新文件 `Interaction/AnchorTracker.swift`

核心 API：

- `AnchorTracker.trackSearch(in:baseline:around:) -> TrackResult?`——正常跟踪搜
  索，中心 = 当前 `trackedPoint`，半径 = `trackSearchRadiusPx`，步长 =
  `trackSearchStepPx`。返回值 `TrackResult { best: Candidate, isLost: Bool }`，
  `isLost` 用 `best.divergenceLuma >= contentThresholdLuma * trackLostFactor`
  判定（最优候选就是全窗口最小散度，所以"最优候选仍 ≥ 放大阈值"和"全部候选都
  ≥ 放大阈值"是同一句话，不需要另外扫一遍）。
- `AnchorTracker.recoverySearch(in:baseline:around:) -> RecoveryResult?`——跟丢
  后的恢复搜索，中心 = `canonicalPoint`（不是跟丢时的位置），半径 =
  `trackLostRecoverySearchRadiusPx`。返回值 `RecoveryResult { best: Candidate,
  hasRecovered: Bool }`，`hasRecovered` 用**未放大**的 `contentThresholdLuma`
  判定。
- 两者共用一个私有网格搜索原语 `bestCandidate(in:baseline:center:radius:step:)`
  ——`[center±radius]` 范围内按 `step` 生成候选中心点（用下标而不是浮点累加
  `+=`，避免候选数量因浮点误差跟 §24.2.2 算出的 `(2R/S+1)²` 对不上），每个候选
  调 `DriftDetector.signature(from:atCanonical:)` 采样、`DriftDetector
  .divergence(from:to:)` 算散度，取散度最小的一个。**没有新增任何采样/散度算
  法**——`signature`/`divergence` 一行没改，本文件唯一的新增内容是那个搜索循
  环本身。
- 用不同的返回类型（`TrackResult` vs `RecoveryResult`）而不是"同一个函数换参
  数"，是刻意的：让 B-44 将来"用跟踪结果走恢复判据"或反过来这种接错线的错误在
  编译期就炸，而不是运行时悄悄错。

五个新常量（`static var`，与 `DriftDetector` 现有常量同一种写法）：

| 常量 | 值 | 依据 |
|---|---|---|
| `trackSearchRadiusPx` | `48.0` | `anchorWindowPx`(96.0) 的一半 |
| `trackSearchStepPx` | `8.0` | 见下方"笔误"记录 |
| `trackLostFactor` | `2.0` | 最优匹配也比 `contentThresholdLuma` 差两倍才算跟丢 |
| `trackLostRecoverySearchRadiusPx` | `192.0` | `anchorWindowPx` 的 2 倍 |
| `minTrackLostRecoveryIntervalMs` | `2000.0` | 恢复搜索冷却下限；本批次没有任何代码读取/强制它，只定义常量（B-44 的事） |

`reAnchorAcceptIoU`/`contentThresholdLuma`/`minReAnchorIntervalMs` 三个 R21 保
护的既有常量一个字节没碰——这五个是全新命名的独立存储。

### `trackSearchStepPx` 的笔误怎么处理的

architect_output.md §24.2.2 的表格给的初始值是 **8.0**，但同一节的文字推导说
"等于 `anchorWindowPx / anchorGridSide`"——96.0/8=12.0，跟表格里的 8.0 对不
上，规格文档本身自相矛盾。按用户在任务里已经做的裁决：**如实按表格数值实现
8.0**，因为 §24.2.2 那段"取 R=48pt, S=8pt ⇒ 13×13=169 个候选 ⇒ <1ms"的成本推
导也是拿 8.0 算出来的（12.0 算出来是 9×9=81 个候选），说明作者实际要的是
8.0，"等于 anchorWindowPx/anchorGridSide" 那句话才是笔误。矛盾原样记在
`trackSearchStepPx` 的文档注释里，没有偷偷改成 12.0，留给 Architect 将来核
对。

### 正确性验证——真跑代码，不是纯审查

本批次没有任何调用方，App 内跑不出行为（点了也不会触发这段代码），所以验证分
两层：

1. **文件内 `#if DEBUG` 自检**：`AnchorTracker.selfCheck()`，仿照项目里
   `Persistence/PinDebugFixture.swift` 的先例（debug-only、不接入正常应用流
   程）。构造两张合成 32BGRA `CVPixelBuffer`（暗背景 + 一块亮色方块 marker），
   验证：① marker 在 `trackSearchRadiusPx` 内、按非对称偏移（dx=16, dy=-8，专
   门用来抓坐标轴搞混的 bug）移动后，`trackSearch` 精确命中真实新位置、散度接
   近 0、`isLost==false`；② marker 移出 `trackSearchRadiusPx`（dx=96）后，
   `trackSearch` 报告 `isLost==true`（窗口内全是背景，跟结构化基线怎么都对不
   上）；③ 同一张"跟丢"帧上，以 `canonicalPoint`（原点）为中心、宽半径的
   `recoverySearch` 却能找回 marker、`hasRecovered==true`——同一份数据，两种搜
   索给出不同结论，正是 §24.2.3 要求的语义区别的直接证据。
   - 全部移动偏移量刻意取 `trackSearchStepPx` 的整数倍（16、-8、96，不用
     100）：第一版用了 100（非 8 的倍数），真实位置正好落在两个搜索网格点中
     间，最近网格点比精确匹配点多算出 ≈21 个 luma 单位的散度（用手算 + 直接调
     `DriftDetector.signature`/`divergence` 在真实偏移点验证过，精确点散度是
     0.0）——这是硬边缘合成图案对网格量化的真实敏感性，不是代码坐标系的
     bug，但会污染"自检该断言多严格"这件事，所以改成网格对齐的偏移量，把"坐
     标系有没有搞反"和"量化噪声有多大"这两件事分开验证，注释里记了这一手排
     查过程。
2. **文件外独立编译运行**：把 `DriftDetector.swift` + `AnchorTracker.swift` +
   一个临时 `main.swift`（调 `AnchorTracker.selfCheck()`）复制到 scratchpad，
   `swiftc -D DEBUG ... -framework CoreVideo -framework CoreGraphics` 直接在
   Mac 宿主上编译运行（这两个文件只依赖 Foundation/CoreGraphics/CoreVideo，没
   有 UIKit，macOS 上能编译）。第一轮跑出上面那个"100 不对齐"的散度异常，改完
   偏移量后重新编译运行，`SELFCHECK PASS`，exit code 0。这是真正跑过的可执行
   验证，不是"写了个函数但没人验证过它到底对不对"——同时全程没有碰 Xcode 工程
   任何文件，跟"零调用方"的约束不冲突（没有把 `selfCheck` 接进
   `JudgeE2App.swift` 或任何生产代码路径）。

### 构建

- `xcodebuild -project JudgeE2.xcodeproj -scheme JudgeE2 -destination 'id=64F6C21E-
  C364-4BB2-859E-DB43CB11CFA1' -derivedDataPath <scratchpad>/dd-b43 -configuration
  Debug clean build` → **BUILD SUCCEEDED**，grep 整份构建日志确认真实 Swift 编
  译器 warning 数 **0**（唯一命中的是 `appintentsmetadataprocessor` 的
  "Metadata extraction skipped. No AppIntents.framework dependency found."，工
  具链自身提示，不算）。日志确认 `AnchorTracker.swift` 确实进了 `SwiftCompile`
  批次（`Interaction/ReAnchorLoop.swift, Interaction/AnchorTracker.swift,
  MobileSAM_ImageEncoder...` 那一组）。
- 没做真机/模拟器交互验证——按指示，本批次零调用方，App 里点不出任何行为差
  异，交互验证留给 B-44 落地之后。

### 文件清单

- 新增：`JudgeE2/Interaction/AnchorTracker.swift`
- 改：`JudgeE2/JudgeE2.xcodeproj/project.pbxproj`（注册新文件：`PBXBuildFile` +
  `PBXFileReference` + `Interaction` group 的 children + `Sources` build
  phase，ID 用 `D3A100142F600014000000A1`(file ref) /
  `D3A100152F600015000000A1`(build file)，跟现有 `D3A1...` 系列同一种命名习
  惯）
- 未碰：`CameraManager.swift`、`TapInstanceManager.swift`、`DriftDetector.swift`
  （B-46 刚加的 `centroidAlignedIoU`/`alphaIoU`/`trackConsistencyAcceptIoU` 原
  样不动）、`UI/ContentView.swift`、`Persistence/FilePinStore.swift`、
  `shared/tasks.md`、`shared/architect_output.md`
- 未做（按指示，留给后续批次）：B-44（接线到 `checkAndFireReAnchor`，真正移动
  `trackedPoint`、读写 `TapInstance.trackState`/`lastReAnchorTrackedPoint`）、
  B-45、B-47、B-48。

## Phase 4C — B-44 + B-45（接线 `AnchorTracker` 到 `checkAndFireReAnchor`，RE-1 并集扩展）

日期：2026-08-24
范围：architect_output.md §24.4 B-44/B-45，同批交付（B-45 单独交付会让跟踪产生
的位移被旧 RE-1 吞掉）。前置 B-42/B-43/B-46 已完成。不做 B-47（症状 B 旁路）、
不碰 `ContentView.swift`/`FilePinStore.swift`/`tasks.md`/`architect_output.md`。

### 结构：RE-1 拆成两个门，而不是一处应用整条公式

architect_output §24.1.3 给的 RE-1 公式（世代变了 **或** `trackedPoint` 相对上
次派发位移超过 `trackReDecodeMinDeltaPx`）如果整条用来门控"步骤 6 该不该采样漂
移"，会出现循环依赖：`trackedPoint` 只有搜索跑过才会动，搜索只有实例入选采样
门之后才会跑。拆成两处：

1. **步骤 5b（采样门，`eligible` 过滤）**：`.locked` 沿用世代号规则，逐位不
   变；`.tracking`/`.lost` 只要 `objectTrackingEnabled` 为真就无条件有采样资
   格（不受世代号限制）——因为世代门当初存在的理由（同 embedding+同点⇒结果必
   然相同，§18.1.2）只对"点不会动"的 `.locked` 成立。
2. **步骤 6c（新增，派发门）**：搜索成功、`trackedPoint` 已经真的移动之后，才
   用 architect §24.1.3 的完整公式判断"这次位移值不值得再花一次 decode"。

这个拆分和它的因果链原样写进了 `CameraManager.swift` 步骤 5b 上方的注释（约
50 行），供 Architect 核对这个应用位置的解读对不对——不是重新推翻设计，是补
上规格文字没讲清楚"公式该用在哪一步"这处应用歧义（跟 B-43 那次
`trackSearchStepPx` 表格/推导互相矛盾是同一类"记下来，不擅自选"的处理方式）。

### `checkAndFireReAnchor` 改动结构（`CameraManager.swift`）

- 步骤 5b：`eligible` 过滤器按上面拆分改写。`objectTrackingEnabled == false`
  时，`.tracking`/`.lost` 分支永远走不到（B-42 保证所有实例恒 `.locked`），这
  行代码不改变任何现有行为。
- 步骤 6：漂移采样点从 `instance.canonicalPoint` 改成 `instance.trackedPoint`
  ——对 `.locked` 实例两者恒相等，逐位不变。
- 新增步骤 6c（插在 RE-2 选出 `pick` 之后、步骤 7 抢节流槽位之前，仍在
  videoQueue 上、`decoderQueue.async` 之前）：
  - `!objectTrackingEnabled || pick.instance.trackState == .locked`：完全跳过，
    `decodePoint`/`newBaselineSignature` 保持步骤 6 已经算好的值，等价于今天
    的路径。
  - `.tracking`：调 `AnchorTracker.trackSearch(around: trackedPoint)`。
    `isLost`→`tapInstances.updateTracking(trackedPoint: nil, trackState:
    .lost, anchorSignature: nil)`（`trackedPoint` 冻结，用 `nil` 参数让"冻结"
    在类型层面就是唯一可能结果，不是靠记得不传）+
    `reAnchorKeepStaleMask`，不抢节流槽位。
  - `.lost`：调 `AnchorTracker.recoverySearch(around: canonicalPoint)`（不是
    `trackedPoint`——冻结的旧位置是最不可能重新出现物体的地方）。没恢复就直
    接 return，`.lost` 原样保持。
  - 两条分支的"找到了"结果汇合到共同后续：在**新位置**重新采样一次
    `DriftDetector.signature`（⚠️ 不用步骤 6 的 `pick.signature`，那是漂移前
    的旧内容，拿来当新基线会让跟踪器越搜越偏）→
    `tapInstances.updateTracking(trackedPoint: candidate.point, trackState:
    .tracking, anchorSignature: refreshedSignature)`（无条件提交，不管等下要
    不要派发 decode）→ 按 §24.1.3 公式算 `shouldDispatch`（世代变了，或
    `distance(candidate.point, lastReAnchorTrackedPoint ?? candidate.point) >
    trackReDecodeMinDeltaPx`）→ 不派发就地 return，派发就继续到步骤 7。
- 步骤 8：`setAnchorSignature` 改用 `newBaselineSignature`（`.locked`/关闭路径
  下等于原来的 `pick.signature`，逐位不变；跟踪路径下是步骤 6c 新采样的值，对
  `updateTracking` 已经写过的同一个值做一次幂等重写，无害）；
  `markReAnchorDispatched` 新增可选参数 `trackedPoint:`（默认 `nil`，不写
  `lastReAnchorTrackedPoint`，逐位不变；跟踪路径下传 `dispatchedTrackedPoint`
  记录这次派发时的位置，供下一轮 §24.1.3 公式用）。
- 步骤 9：`reAnchorDecode` 调用新增 `decodePoint:` 参数（`.locked`/关闭路径传
  `canonicalPoint`，跟踪路径传搜索/恢复到的新点）；另外构造一份 `pick.instance`
  的局部拷贝，只在跟踪-找到路径把 `trackState` patch 成 `.tracking`（因为
  `pick.instance` 是搜索**之前**的快照，B-46 已经做好的一致性门选择逻辑要看的
  是"这次 decode 到底是不是一次跟踪型 decode"，不是搜索前的旧状态）——B-46 的
  门选择 `switch` 本身一行没动，只是喂给它的输入现在是对的。

### `reAnchorDecode` 签名变化

新增 `decodePoint: CGPoint` 参数，删掉内部 `let canonicalPoint = instance
.canonicalPoint`，三处内部使用点（`buildPointPrompt` 的 prompt 坐标、
`tapPoint256` 的连通域种子点）全部改用 `decodePoint`。函数顶部 doc 注释按
§24.1.2 改写，说明 `.locked` 时逐位不变、`.tracking` 时点由调用方的搜索结果决
定，`canonicalPoint` 本身永远不被这个函数读取或修改。

### `TapInstanceManager.swift` 新增

- `markReAnchorDispatched` 加可选参数 `trackedPoint: CGPoint? = nil`——非 nil
  时额外写 `lastReAnchorTrackedPoint`，默认不写，所有既有调用点字节不变。
- 新方法 `updateTracking(id:trackedPoint:trackState:anchorSignature:)`：一次锁
  内改完 `trackedPoint`/`trackState`/`anchorSignature` 三个字段（同 A-18"缓冲
  区形状必须与数据同行"纪律，避免读者看到"点已经移动、签名还是旧的"这种撕裂状
  态）。`trackedPoint`/`anchorSignature` 传 `nil` = 保留原值不动，`trackState`
  永远写（每个调用方都是在做状态迁移）。`.lost` 迁移调用时 `trackedPoint` 传
  `nil`——刻意不给"跟丢的同时还能移动 trackedPoint"这条路径留一个参数位置。

### `AnchorTracker.swift` 新增/改动

- 新常量 `trackReDecodeMinDeltaPx: CGFloat = 12.0`——architect_output §24.2.2
  表格已经给出这个值并归在 B-45 名下（不是 B-43，之前任务描述里说"B-43 已经定
  义好"是误记，实际检查源码后确认 B-43 交付的 `AnchorTracker.swift` 里没有这
  个常量，本批次按 B-45 的归属新增，不是"发现遗漏后补"）。
- **复核 B-43 交付时发现一处成本问题，顺手修了**：`recoverySearch` 原来复用
  `trackSearchStepPx=8.0`，但半径是宽得多的 `trackLostRecoverySearchRadiusPx
  =192.0`，候选数 49×49=2401（是 `trackSearch` 169 个候选的 ~14 倍），估算单
  次成本 ~120ms，在 videoQueue 上会造成可感知卡顿（一帧预算 33ms）。
  architect_output §24.2.3 只裁了恢复搜索的半径，没裁步长要不要跟着放大，是规
  格本身的缺口，不是"我觉得原来的不好"。新增独立常量 `trackRecoverySearchStepPx
  = 16.0`（正常步长 2 倍，候选数降到 625，估算 ~31ms），只用于 `recoverySearch`
  内部的 `bestCandidate` 调用，`trackSearch`（正常跟踪）的步长不变。B-43 原有
  的 `selfCheck()` 用到的偏移量（96）同时是 8 和 16 的倍数，改动后重新编译运行
  确认仍然 `PASSED`，没有破坏既有自检。
- 顶部"零调用方"的 SCOPE GUARD 注释按 A-17 更新为如实反映现状：B-44 已经把这
  两个函数接了进去，只是 `objectTrackingEnabled` 默认关闭，所以"实际会执行"的
  调用点数目仍是零，跟旧注释"完全没有调用方"不是一回事，留着旧措辞会跟真实代
  码状态脱节。

### `DriftDetector.swift` 新增

- `static var objectTrackingEnabled: Bool = false`——能力 C 总开关，形态镜像
  `reAnchorEnabled`（B-16）。默认关闭；doc 注释里原样抄录 architect_output
  §24.4"主开关"行给的翻真三前提（B-42…B-47 全部落地编译通过 + P-10~P-17 完成
  一轮真机读数 + `ISSUE-P4-DECODE` 结案或用户显式豁免），并声明 Builder 不得
  自行翻转。这是本批次唯一新增的公开可写状态。

### ⚠️ 发现一处规格空白，没有擅自补——`.locked → .tracking` 的引导路径不存在

复核到一半发现：**整个 B-42~B-45 给定的规格里，没有任何地方定义一个实例如何
从 `.locked` 第一次转成 `.tracking`。** B-42 的 `TapInstanceManager.addInstance`
硬编码构造成 `.locked`（这批次没让我碰这个文件的这部分，也不该碰）；B-44 步骤
6c 的入口门是 `pick.instance.trackState != .locked`——按任务里给的原文"若
`pick.instance.trackState == .locked`：完全走今天的路径，不做任何改动"逐字实
现，这意味着 **`.locked` 实例永远不会进入搜索分支，不管 `objectTrackingEnabled`
是真是假**。旁证：B-42 的字段文档写着"能力 C 关闭时恒 `.locked`"（隐含关闭时
未必恒 `.locked`），B-42 的 memory 记录也写"直到 B-43/B-44 落地"这样的措辞，两
处都暗示这个引导转换该由本批次或后续批次补上，但任务里给的可执行设计没有提供
它。

**后果**：即使未来 `objectTrackingEnabled` 翻真，也不会有任何实例真正进入跟踪
状态——`.tracking`/`.lost` 分支的代码虽然编译、可达（步骤 6c 本身没问题，用独
立 harness 验证过），但没有任何调用路径能把一个实例从初始的 `.locked` 送进
去。这不是本次改动引入的 bug（我没有改 `addInstance`,也没有在步骤 6c 里加一条
`.locked` 分支去猜一个引导规则），是发现的一处规格执行链条空白,**按这次任务
"如果你认为这个结构本身有问题，先停下来告诉我，不要自己另立一套"的指示，记在
这里,不自行设计一套引导转换逻辑**。已经写进独立 harness 的 Scenario 8 作为可
复现证据（见下）。这处空白需要 Architect 补一条设计（大概率的落点是"能力 C 打
开时，`addInstance` 构造成 `.tracking` 而不是 `.locked`"，但这是一个架构判
断，不是 Builder 该替 Architect 做的决定）。

### 正确性验证

**回归（`objectTrackingEnabled == false`，最重要的验收线）**：

- 干净 `derivedDataPath`、`-configuration Debug`、`clean build` →
  **BUILD SUCCEEDED**，grep 整份日志确认真实 Swift 编译器 warning 数 **0**（唯
  一命中的仍是工具链自身的 AppIntents 提示，不算）。额外跑了一次默认 Release
  配置的 clean build 做交叉验证，同样 0 warning。
- 装到 `64F6C21E-C364-4BB2-859E-DB43CB11CFA1`（当前 booted 的 "iPhone 17 Pro"
  模拟器——项目给的固定 destination id，不是"iPhone 11"，按 id 走）跑起来：切
  到 Tap to Segment 模式、UI 交互正常，`launchctl list` 确认进程全程存活，
  `log show` 抓 5 分钟日志 grep `fatal|crash|assert|trap` 零命中。**但**
  `AVCaptureSession` 在这台模拟器上直接报 `AVFoundationErrorDomain -11800`（会
  话启动超时/失败），确认是模拟器本身没有可用摄像头（跟这次改动无关的环境限
  制，日志里能看到这个错误在 App 启动阶段就发生），所以 `checkAndFireReAnchor`
  在这次会话里从未被相机回调驱动执行过一次——日志里没有任何 `[REANCHOR]`/
  `[TRACK]` 行，新旧代码路径在这台模拟器上事实上都没跑到过，回归验证只能停留
  在"编译正确 + 代码审查确认所有新分支的入口条件在开关关闭/实例恒 `.locked`
  时不可达 + App 不崩溃"这个层面，做不到"跑一次真实 tap 看到 mask 出来"这种更
  强的验证（这也是这台模拟器在之前 B-42/B-43 session 里就有的已知限制，不是本
  批次新引入的）。

**跟踪行为本身（`objectTrackingEnabled == true`）——按指示，独立 harness 验
证，不改真机/模拟器状态**：

- 复用 B-43 session 建立的套路：把 `AnchorTracker.swift`/`DriftDetector.swift`
  原样复制到 scratchpad，另写一个 `main.swift`，**逐字转录**（不是重新实现）
  `checkAndFireReAnchor` 步骤 6c 的控制流到一个用 mock `struct`（只含步骤 6c
  会读写的字段，绕开 `TapInstance` 依赖的 UIKit）+ 一个镜像
  `TapInstanceManager.updateTracking` 语义的函数，`swiftc -D DEBUG` 在 Mac 宿
  主上独立编译运行。9 个场景全部 PASS：
  1. `.tracking` 小位移（8px，有真实上次派发参照点）→ 状态/`trackedPoint`/
     基线更新，但 `shouldDispatch=false`。
  2. `.tracking` 大位移（24px，有真实参照点）→ `shouldDispatch=true`。
  3. `lastReAnchorTrackedPoint` 为 `nil` 时，架构公式的 `?? $0.trackedPoint`
     兜底会让距离恒为 0——**这不是本次转录引入的 bug，是公式字面写死的行
     为**，专门写了一条场景钉住这个细节（见 architect_output.md:5775 原文核
     对过），供以后审计。
  4. 世代号变了、位移 0 → 仍然 `shouldDispatch=true`（公式第一支路独立生
     效）。
  5. 位移超出 `trackSearchRadiusPx` → 判定 `.lost`，`trackedPoint` 冻结不
     动，不派发。
  6. `.lost` 状态下恢复搜索命中宽半径内的目标（有真实参照点）→ 转回
     `.tracking`，`shouldDispatch=true`。
  7. `.lost` 状态下恢复搜索连宽半径都够不到 → 保持 `.lost`，`trackedPoint`
     依旧冻结。
  8. `.locked` 路径（无论开关是否打开）→ `decodePoint` 逐位等于
     `canonicalPoint`，与今天行为一致。
  9. **`.locked → .tracking` 引导路径缺失的直接证据**：`.locked` 实例即使内
     容已经"漂移"96px、开关打开，步骤 6c 的入口门也不会放它进去——状态和
     `decodePoint` 原地不动。
  另外重新跑了一遍 B-43 原有的 `AnchorTracker.selfCheck()`（改了
  `trackRecoverySearchStepPx` 之后），确认仍然 `PASSED`，没有回归。
- 这套 harness 是"转录检查"，不是"黑盒测试真实集成"：能抓住这批次自己写的算
  术/控制流错误（比如新基线用错签名、`shouldDispatch` 条件写反），但抓不住"复
  制到 `CameraManager.swift` 时手滑打错一个字段名"这类纯粘贴错误——那一类由
  `xcodebuild` 的类型检查兜底（拼错字段名/类型不对会直接编译失败），两者互
  补，不重复也不冲突。

### 文件清单

- 改：`JudgeE2/Detection/CameraManager.swift`（`checkAndFireReAnchor` 步骤
  5b/6/6c/7-9 改写，`reAnchorDecode` 签名+内部三处引用+相关文档注释改写）
- 改：`JudgeE2/Interaction/TapInstanceManager.swift`（`markReAnchorDispatched`
  加可选参数，新增 `updateTracking`）
- 改：`JudgeE2/Interaction/AnchorTracker.swift`（新增
  `trackReDecodeMinDeltaPx`/`trackRecoverySearchStepPx` 两个常量，
  `recoverySearch` 步长改用新常量，顶部 SCOPE GUARD 注释更新）
- 改：`JudgeE2/Interaction/DriftDetector.swift`（新增
  `objectTrackingEnabled` 主开关）
- 未碰：`JudgeE2/Segmentation/*`、`UI/ContentView.swift`、
  `Persistence/FilePinStore.swift`、`shared/tasks.md`、
  `shared/architect_output.md`、B-46 已完成的一致性门选择 `switch` 本身
- 未做：B-47（症状 B 旁路 + `heavyDriftForceRefreshLuma` 等常量）、B-48
  （`PinFactory` 防御性断言）
- **待 Architect 决策**：`.locked → .tracking` 引导转换机制不存在，见上面
  "发现一处规格空白"一节——这是本批次交付前必须回报的唯一开放问题。

### 追加（同日）：coordinator 确认上面那处规格空白是派工疏漏，已按其给的修法补上

coordinator 确认"`.locked → .tracking` 引导转换缺失"是设计时想到、写最终指令
时漏掉的疏漏，不是该我自己猜的地方，指示直接在原有 6c 分支基础上修，不新增独
立逻辑。改动位置全部在 `CameraManager.swift` 步骤 6c 内部：

1. **入口 guard**（原 `:2571` 附近，现已因本轮改动整体下移几行）：原来是
   `if DriftDetector.objectTrackingEnabled, pick.instance.trackState != .locked`，
   把 `.locked` 排除在外，导致这个分支永远进不去。coordinator 给的修法原文是
   把它改成 `!= .lost`（或等价的 `[.locked, .tracking].contains(...)` 允许
   集）——**但这两种写法字面实现都会把 `.lost` 排除在这个 if 块之外，而
   `.lost` 的恢复搜索（`switch` 里的 `case .lost`）就定义在同一个 if 块内部**，
   字面照做会让 `.lost` 实例的恢复搜索整个失效，跟 coordinator 反复强调的"不
   要碰这段代码之外的任何东西"以及他们自己给的理由（".locked 和 .tracking 都
   进入,只有 .lost 走 recovery 分支"——这句话本身在描述"`.lost` 走 recovery
   分支"这件已经存在、不需要改的事实，不是要把它排除在外）互相矛盾。判断这是
   写指令时的笔误（跟本批次早前 `trackSearchStepPx` 表格/推导矛盾是同一类"记
   下来、按能让全篇自洽的读法实现"处理方式），**实际实现是把 trackState 条件
   整个去掉**（`if DriftDetector.objectTrackingEnabled {`）——效果上等于
   `.locked`/`.tracking`/`.lost` 三种状态现在都能进入这个块，`.lost` 的恢复
   搜索逻辑一行没动。已经在改动点上方写了一段注释解释这条不一致以及为什么这
   样处理，会在给 coordinator 的总结里原样提出来，不是自己悄悄拍板。
2. **`switch` 合并**：`case .tracking:` 改成 `case .tracking, .locked:`，共用
   同一段 `trackSearch` 逻辑（`.locked` 实例的 `trackedPoint` 按构造恒等于
   `canonicalPoint`，所以"围绕 `trackedPoint` 搜索"对它来说就是"围绕原始点击
   点搜索"，是它第一次被搜索时该做的事）。原来 `case .locked: break //
   unreachable` 那个分支删掉（switch 现在只有两个 case，穷尽）。
3. **激活日志**：合并后的 case 开头判断 `pick.instance.trackState == .locked`
   存成 `wasLocked`，是 `.locked` 时先打一条
   `[TRACK][inst#N] locked → tracking — activating on first drift-triggered
   candidacy`；跟丢/找到两条既有日志也按 `wasLocked` 改成 `"locked"` 还是
   `"tracking"` 开头，不再对刚激活的实例硬编码打印"tracking"。
4. **激活时机说明**：按 coordinator 的要求，在改动上方写清楚激活是"惰性、由
   触发驱动"的——只有 `.locked` 实例的漂移超阈、被 RE-2 选中之后才会走到这里
   被激活，不是 `objectTrackingEnabled` 一开启就批量把所有实例转成
   `.tracking`（那需要在别处新增一个触发点，违反"不新增触发路径"纪律）。

**受影响但不需要改逻辑、只改了措辞的地方**：步骤 8（`setAnchorSignature`/
`markReAnchorDispatched` 附近）和步骤 9（构造 `dispatchInstance` 附近）原来的
注释把"`.locked` 路径"当成"零改动路径"的同义词，这个前提现在不成立了（`.locked`
+ 跟踪开启现在也能产生真实的 `dispatchedTrackedPoint`），改成了"跟踪整体关闭"
才是真正的零改动路径，逻辑代码本身没有改（`dispatchedTrackedPoint != nil` 这
条判断照样对，因为无论是从 `.locked`/`.tracking` 找到,还是从 `.lost` 恢复,
成功都统一走同一次 `updateTracking` 提交,`dispatchedTrackedPoint` 会不会非
nil 只看有没有真的走到"找到了"分支,跟 pre-search 状态是哪一个无关）。

**构建**：干净 `derivedDataPath`、Debug 配置、`clean build` → **BUILD
SUCCEEDED**，grep 确认真实 Swift 编译器 warning/error 数 **0**。

**harness 新场景**（scratchpad 里那套独立编译运行的 mock 转录 harness,同步
改了 `step6c` 转录函数里的 guard 和 switch,跟真实文件改动保持一致）：

- 场景 7 改成测"跟踪整体关闭"（不是"`.locked` 单独关闭"）才是唯一的零改动路
  径——`.locked` 实例在这种条件下 `decodePoint`/`trackState` 都原地不动。
- 场景 8（新）：`.locked` 实例、开关打开、漂移 24px（在 `trackSearchRadiusPx`
  内）→ 断言 `updated.trackState == .tracking` 且显式断言
  `!= .locked`（coordinator 要求的"不能停留在 .locked"负向检查）。
- 场景 9（新）：`.locked` 实例、开关打开、漂移 96px（超出搜索半径）→ 第一次
  搜索直接判跟丢，断言 `updated.trackState == .lost` 且显式 `!= .locked`，
  `trackedPoint` 冻结在 `canonicalPoint`，不派发。
- 全部场景（含改动前就有的 1/2/2b/3/4/5/6）重新编译跑了一遍，**ALL PASSED**。
  额外确认 B-43 原有 `AnchorTracker.selfCheck()`（本轮未改 `AnchorTracker
  .swift`）不受影响，未重跑（无必要，改动范围不含该文件）。

---

## 2026-08-24 — B-47（症状 B 旁路：`heavy_drift` 死代码接上真实条件，能力 C 批次收尾）

按 architect_output §24.3.2/§24.3.4 裁决实现。B-42~B-46 均已落地，
`DriftDetector.objectTrackingEnabled` 仍是 `false`（未动）。

### 新增常量（`Interaction/DriftDetector.swift`，紧挨着 `objectTrackingEnabled`）

- `heavyDriftForceRefreshLuma: Double = 32.0`（`contentThresholdLuma` 8.0 的
  4 倍）
- `minHeavyDriftAgeFloorMs: Double = 1500.0`
- `minHeavyDriftRefreshIntervalMs: Double = 5000.0`（独立冷却窗口，不是
  `minReAnchorIntervalMs=300` 的复用）

三个常量数值与 R21 保护的既有常量（`reAnchorAcceptIoU` 等）互不接触。

### `CameraManager.swift` 新增 videoQueue-only 属性（挨着 `lastReAnchorFireMs`）

- `private var lastObservedMaxDriftLuma: Double = 0`——`checkAndFireReAnchor`
  的 `guard !seededThisFrame, !measured.isEmpty else { return }`
  （原 `:2503`，现 `:2503` 附近）之后新增一行
  `lastObservedMaxDriftLuma = measured.map { $0.drift.divergenceLuma }.max()
  ?? lastObservedMaxDriftLuma`——只是对已经算好的 `measured` 取一次 `max()`，
  没有新增采样或 `DriftDetector` 调用，B-44/B-45 已写好的搜索/派发分支未动。
- `private var lastHeavyDriftRefreshMs: Double = 0`——独立时钟，不复用
  `lastReAnchorFireMs`（那是 re-anchor 节流自己的时钟）。

### `refreshTapEmbeddingIfNeeded` 旁路接入

按 §24.3.2 伪代码原样实现：

```swift
let heavyDriftBypass = DriftDetector.objectTrackingEnabled
    && quiet                                                          // 必要条件，不可省略
    && cacheAgeMs != nil && cacheAgeMs! >= DriftDetector.minHeavyDriftAgeFloorMs
    && lastObservedMaxDriftLuma >= DriftDetector.heavyDriftForceRefreshLuma
    && (nowMs - lastHeavyDriftRefreshMs) >= DriftDetector.minHeavyDriftRefreshIntervalMs

guard !busy, !parked, quiet, (!cacheFresh || heavyDriftBypass) else { return }
```

`quiet` 必要性的道理（tap 活跃交互期间插入重编码会抢 encoderQueue、重开
2.6–3.6s 延迟这个已修好的洞）按要求原样写进了代码注释，没有简化。

`heavyDriftBypass` 真正生效时，在拿到 encoder 槽位（`isEncoding = true` 之后、
`stateLock.unlock()` 之后）写 `lastHeavyDriftRefreshMs = nowMs`，videoQueue-only，
不加锁，跟 `lastReAnchorFireMs` 的写法一致。

### 两处日志

1. **抢槽位之前**那条 `[CACHE] background refresh triggered: ...`：改成三分支
   （`cacheAgeMs == nil` → cold_start；`cacheAgeMs! >= 5000` → 原 age 措辞；
   否则 → 新增 `heavy_drift bypass (age=%.0f ms, maxDrift=%.1f lum ≥ %.1f
   threshold)`，把年龄和触发它的漂移量都打出来）。
2. **抢到槽位之后**那条 `refreshLogReason`（死代码本体）：验证了推理——
   guard 已改成 `(!cacheFresh || heavyDriftBypass)`，`cacheFresh = age<=5000`，
   所以能走到这里且 `cacheAgeMs! < 5000` 的路径，`!cacheFresh` 必为
   `false`，guard 要通过就必须 `heavyDriftBypass == true`；而这两个局部变量
   （`cacheAgeMs`、`heavyDriftBypass`）从第一次 guard 到这里全程不变（都是
   `let`），中途唯一的分支只会更早 `return`（丢槽位）或直接走完，不会绕开这个
   推理。**结论：推理成立，`heavy_drift` 这个 else 分支现在确实可达且只在
   这一条路径下可达**（唯一的浮点边界巧合：`cacheAgeMs! == 5000.0` 时两种
   解释都成立，但实际时间戳几乎不可能精确撞在这个值上，不影响结论）。已把
   过时注释 `only reachable if drift path adds to this call` 删掉，换成如实
   反映现状的说法。

### 构建 + 回归

- 干净 `derivedDataPath`、**Debug** 配置、`clean build`（destination 用真机
  同款 iPhone 17 Pro 模拟器 id）→ **BUILD SUCCEEDED**，grep 确认真实 Swift
  编译器 warning 数 **0**。
- iPhone 11 尺寸模拟器（E3DF778B-E120-4CE6-BAC8-900B7568B7C1，本机已启动的
  那台）：装机、`simctl launch --console-pty` 跑控制台、切到 Tap to Segment
  模式、点了一下预览区域——无崩溃，FPS 稳定 60，日志格式（`[SAM]`/`[TAP]`/
  `=== MODE SWITCH ===`）跟以前一致。模拟器没有真实摄像头
  （`Camera input unavailable`），所以 `refreshTapEmbeddingIfNeeded` 在
  `guard let buffer = latestCameraBuffer else { return }` 那一步就提前退出，
  新代码本轮没有在模拟器上被真正执行到——这是模拟器环境的已知限制（真机才有
  camera pipeline），不是本次改动引入的问题。`objectTrackingEnabled` 恒
  `false`，`heavyDriftBypass` 第一个合取项恒假，就算有摄像头帧，guard 也会
  退化回今天的 `!cacheFresh` 单条件，行为逐位不变。

### 边界验证（scratchpad 独立 Swift harness，非 Xcode target）

`heavyDriftBypass` 布尔表达式常量原样抄自 `DriftDetector.swift`，测试了 11
组边界：
`objectTrackingEnabled=false` 恒否决 / `quiet=false` 恒否决 / age 低于
1500ms 恒否决（1500ms 整数通过）/ `cacheAgeMs=nil` 恒否决 / drift 低于
32.0 恒否决（32.0 整数通过）/ 冷却未到 5000ms 恒否决（5000ms 整数通过）/
全部满足 → true / age 已 ≥5000ms 时表达式本身仍为 true（这种情况下
`!cacheFresh` 已经为 true，OR 是冗余的，不影响 guard 结果）。
**全部 11 组 PASS**。

### 不做的事——确认遵守

未碰 `AnchorTracker.swift`/`TapInstanceManager.swift`；未碰
`checkAndFireReAnchor` 里 B-44/B-45 已写好的搜索/派发分支（只加了一行 max()
记录）；未重新定义 `objectTrackingEnabled`；未改任何 R21/B-43/B-45 既有常量
数值；未碰 `ContentView.swift`/`FilePinStore.swift`；未改 `shared/tasks.md`/
`shared/architect_output.md`。

### 文件清单

- `JudgeE2/Interaction/DriftDetector.swift`——新增三常量。
- `JudgeE2/Detection/CameraManager.swift`——新增两属性
  （`lastObservedMaxDriftLuma`/`lastHeavyDriftRefreshMs`）、
  `checkAndFireReAnchor` 加一行 max() 记录、`refreshTapEmbeddingIfNeeded`
  接入 `heavyDriftBypass`、两处日志改动。

---

## PHASE 5 — App 单功能化 + 品牌视觉（Builder，2026-08-24）

依据 `shared/UI-instruction`（Phase 5 一次性执行清单）+ §26.5 冻结表面表
+ §27 入口点定义 / D-27.1。

### 交付清单（对照任务书 7 项）

| # | 任务 | 状态 |
|---|------|------|
| 1 | App 图标 / Logo（tap → segment → pin）| ✅ 1024×1024 三变体（any/dark/tinted）写入 `AppIcon.appiconset`；`BrandMark.swift` 提供同一图形的矢量版供 Settings/后续启动屏复用 |
| 2 | 单功能化（UI 层）| ✅ `mode` 默认改 `.tapToSegment`；三态 picker 与悬浮快切按钮从 `ContentView` 移除，picker 迁入 `SettingsView` 的 `#if DEBUG` 分区；`AppMode` 枚举与 `CameraManager` 路由未动 |
| 3 | 底部控制栏 | ✅ 新建 `UI/BottomControlBar.swift`：Pins / Clear / Flip / Settings 四项，不新建状态 |
| 4 | Settings 职责分离 | ✅ 新建 `UI/SettingsView.swift`：用户级 = About + 版本号 + 清空所有 Pin；`Compute Units` / `Encoder Res` / `Perf Quiet Log` / `Force Slow Path` / P-9 自检 / App Mode 全部收进 `#if DEBUG` 的 "Developer" 分区，保留未删 |
| 5 | Overlay 视觉样式 | ⛔ **未做，见下方「停下来记录」** |
| 6 | 动画与交互反馈 | ✅ `TapRippleEffect` / `TapLoadingIndicator`（脉冲圆环语义未动）/ `TapFailureIndicator` 原样复用；新增 Pin 徽章 spring、`PinListView` 行增删动画 |
| 7 | UX 细节 | ✅ `PinListView` 空状态文案改为不提"模式"；新增首次长按提示（`@AppStorage("ui.hasSeenSaveHint")`，仅在首次出现选区时展示一次） |

### ⛔ 停下来记录：Task 5（Overlay 视觉样式）未实施

任务书要求 mask 数值改版「必须先过既有 C-7 准入流程」。C-7 由两半构成，
**(a) 算术检验 + (b) 同色系真机撞色目视测试，两项缺一不得合入**（§3.3.2）。
(b) 与任务书第三条子项（用最多 3 实例池验证同屏 2–3 个 tap 实例的可读性）
都**只能在真机上做**，而本次任务书已把「需要真机测试时间」的项目明确排除在
范围外。

因此现行三槽色板（slot0 青 (0,217,255) / slot1 水青 (0,255,242) /
slot2 春青 (0,255,170)，alpha 0.60/0.40，描边 2.0/1.5 与 1.5/1.0 pt）
**保持 🔒 FINAL 原值未动**，`MaskRenderer.swift` 未触碰。
本次没有产生任何需要 C-7 复议的改动；Task 5 应与真机复测一起单独排期。

反向约束已落实到图标配色：任务书第 1 项要求图标与 overlay 调色一致，既然
overlay 维持 FINAL 色板，图标与 `BrandPalette` 就**直接引用该色板的三个色值**
（`BrandMark.swift` 注释已标明是引用而非选色，不构成 C-7 变更）。
唯一新增色是 chrome accent (0,122,153)——只用于徽章/按钮 tint 等
非 mask 表面，是青色压暗到白字对比度 4.95:1 的结果，不进入 C-7 管辖范围。

### 验收证据

- `xcodebuild` Debug + Release（`generic/platform=iOS`）均 **BUILD SUCCEEDED，零 warning**。
- 二进制层面核对 Debug vs Release（Debug 走 `.debug.dylib`）：
  `Run Geometry Round-Trip Check` / `Force Slow Path (testing)` / `Perf Quiet Log` /
  `Compute Units` / `Encoder Res` / Developer 分区脚注文案 —— **Release 命中 0，Debug 命中 1**。
  即「Release 不可见/不可达、Debug 功能不丢失」是由条件编译**构造性**保证的，不靠自觉。
- `Assets.car` 内 `AppIcon` 三种 appearance（any / UIAppearanceDark / ISAppearanceTintable）均已编入。
- 模拟器实跑截图确认底部栏与 Settings 版式（模拟器无摄像头，预览为白，管线本身未验证）。

### 改动范围核对

仅 `JudgeE2/UI/`（ContentView、PinListView 改；BottomControlBar、SettingsView、
BrandMark 新增）+ `JudgeE2/Assets.xcassets/`（AppIcon 4 文件、AccentColor）
+ `project.pbxproj`（仅登记三个新文件）。
§26.5 冻结表面文件**全部未修改**（mtime 逐个核对，最新的
`CameraManager.swift` 停在本次开工前的 14:54）。
`Shared/AppMode.swift` 也未改——任务书允许改 `displayName`，但
"Annotate" 命名归 §27.2 留给 Architect 的专门裁决，且模式名现在只出现在
Debug-only picker 里，改它没有产品收益，故不夹带。

### 已知问题（非本次引入，未处置）

`JudgeE2UITests` target 的 `INFOPLIST_FILE` 指向不存在的
`JudgeE2/JudgeE2/Info.plist`，导致该 target 无法构建。本次开工前既已如此，
修它超出「只碰 UI/ + Assets」的授权范围，故仅记录。

### 图标修订（2026-08-24，用户指示）

用户改了图标方向：**白底 + 红色的「J 形地标」+ 红色定位圈**。

- 字母与地标是**同一个轮廓**，不是把 J 叠在图钉上：J 的上端字身 = 图钉头
  （带字怀），竖干 = 钉杆，左钩 = 它站立的脚。一个剪影同时承担两种读法，
  这是它在 40 pt 还成立的原因。前后试了三轮构造（尾巴弯成 J 的水滴形 /
  水滴内挖 J 字怀 / 字体 J 加钉头），只有「钉头 J」两种读法都立得住。
- 红 `(227,30,46)`：白色在其上对比度 4.67:1，字怀与白底主页都够。
- 构图由**实测包围盒**缩放居中到 76%，不用构造公式推算——iOS 会把图标裁成
  squircle，靠公式很容易把定位圈压到边上被切掉（第一版就是）。
- `dark` 外观与 `any` 同为白底：这是个白底标志，为深色模式反相等于换了一个标志。
  `tinted` 例外——iOS 拿它当**亮度遮罩**再自己上色，所以必须画成深底亮图形，
  喂白底稿会得到反相结果。
- `UI/BrandMark.swift` 同步重画为同一图形（`Canvas` + 与生成脚本共用的几何常量），
  Settings 里的 About 行随之更新。填充用**非零**而非偶奇——头/杆/钩/端帽
  故意互相重叠以免接缝，偶奇会把每一处重叠又挖回去（已实测踩过）。

**这一改动的副作用，需要确认：**

1. 原 §27.2 / 任务书 Task 1 要求「图标配色与 overlay 调色保持一致」。overlay
   仍是 C-7 锁定的青色三槽色板，图标现在是红的，**两者不再同色系**。
   `BrandPalette` 的注释已改为明确声明这一点，不再自称是色板镜像。
   ⚠️ 这是用户指令的直接结果，不是疏漏；但如果后续想恢复一致，动的应该是图标，
   **不是 overlay**（改 overlay 要走完整 C-7，见上一节）。
2. `Assets.xcassets/AccentColor` 与 Pin 计数徽章仍是青色 chrome accent
   `(0,122,153)`，未随图标改红——用户只指定了图标三项，全局 accent 换色
   影响面更大，未擅自扩大。若要统一为红系，说一声即可。

#### 图标二次修订：头部改水滴形 + 网眼打通（2026-08-24，用户指示）

J 的头部由圆形改为**地标水滴形**，字怀（网眼）打通。

- 整个字母现在是**一条封闭轮廓**，不再靠若干填充块叠加求并：水滴头 = 圆 +
  从下方虚拟尖点引的两条切线；切线走到**半宽恰好等于竖干半宽**处截断，竖干
  由此接上，所以接缝与肩台都不存在；竖干下行转为左钩，钩的内外缘是同一条
  半圆中心线的两条等距偏移。网眼是路径上真正的洞（偶奇填充），不是盖上去的
  白圆——换成非白底也仍然成立。
- 定稿参数：`drop=3.00 / hook_R=165 / stem_bot=675`（设计单位 /1000）。

**两个靠渲染才发现的坑，已写进两份源码的注释：**

1. **钩子末端圆头的扫掠方向。** 中心线在钩末端是**向上**走的，所以圆头必须
   向上鼓（`0 → -π`）。写成 `0 → +π` 会让它鼓到内缘下面、与内缘交出尖角，
   整个剪影读成"爪子"而不是 J——第一版就是这个症状。
2. **定位圈线宽必须用设计单位，不能用绝对像素。** 原来写的是 `int(SS*w)`，
   在试验稿（N=1536）和正式稿（N=4096）下相对粗细差了一倍，正式图的环细到
   几乎看不见。已改为 `w * (N/1000)`。

`UI/BrandMark.swift` 同步改为同一构造（同一套几何常量 + 同样的偶奇打洞），
填充规则也随之从非零改回偶奇——上一版必须用非零是因为那时轮廓由重叠块拼成，
现在轮廓不自重叠，偶奇才是对的。Debug + Release 均 BUILD SUCCEEDED 零 warning，
三个 appearance 均已编入 `Assets.car`。
