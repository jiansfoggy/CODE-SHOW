# Day 6 — MobileSAM Encoder Stage Profile (D6-D-SAM-PROFILE)

日期：2026-02-19
设备：iPhone 11 (A13) / iOS 17
目标：用 Instruments 抓 **MobileSAM encoder 阶段**热点，并把总耗时拆分为：
- preprocess（1024 resize+pad）
- MLMultiArray pack（BGRA → NCHW float32）
- CoreML encoder 执行（`MobileSAM_ImageEncoder` prediction）
- mask 后处理/渲染（threshold/upsample/crop + createCGImage + overlay）

> 现状说明：当前 repo 里 encoder 只对 `prediction(from:)` 做了计时（`encMs`）。preprocess/pack/post/render 没有细分计时字段。
> 因此本文包含两部分：
> 1) **已确认的 Instruments/CoreML 证据（数值级）**
> 2) **基于代码路径的热点归因（可复现的采集方法 + 明确的下一步打点）**

---

## 0) Evidence sources（证据来源）

### 0.1 Instruments（`.all`） — CoreML Aggregation
- 截图：`/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/record3_coreML_all.png`

关键数值（截图中 Model Activity Aggregation）：
- `Prediction → MobileSAM_ImageEncoder`：Avg **~637.83 ms**（Count=34，Max ~1.14 s）
- `Input Copy → MobileSAM_ImageEncoder`：Total **~123.40 ms**（Count=35，Avg ~3.53 ms）
- `Output Transformation → MobileSAM_ImageEncoder`：Total **~2.56 ms**（Count=34，Avg ~75 µs）

结论（严格基于证据）：
- 在 `.all` 下，encoder 的 `prediction` 平均耗时约 **638ms**（这通常是“CoreML 调用 + backend compute + 框架侧 copy/validation/transform”的合计）。
- `Input Copy/Output Transformation` 在 aggregation 中占比不大（ms 级别），说明主要时间在 backend compute / 调度。

### 0.2 App 内 console 计时（SegmentationEngine）
- 代码：`swift_app/JudgeEverythingApp/Sources/SegmentationEngine.swift`
- 当前日志：`[SegmentationEngine] seg ... (enc ..., dec ...)`

注意：
- `encMs` 只包了 `encBox.model.prediction(from:)`（不含 preprocess + pack）。
- `segMs` 是端到端（含 preprocess/pack/decoder/mask 后处理/渲染）。

---

## 1) Code-path decomposition（实际执行路径拆解）

以 `SegmentationEngine.tick()`（以及 `runPending`）为准，encoder 阶段的关键步骤如下：

### 1.1 Preprocess（1024 resize+pad）
- 文件：`Sources/MobileSAMPreprocess.swift`
- 实现：
  - `CIImage.oriented(exif)`
  - `CIImage.transformed(scale)` → `composited(over:black 1024×1024)`
  - `CIContext.render(composed, to: CVPixelBuffer(1024×1024 BGRA))`

**预期热点**：`CIContext.render` + resize filter（当前写法是 transform + render，实际仍会触发 GPU/CPU 采样）。

### 1.2 MLMultiArray pack（BGRA → NCHW float32）
- 文件：`Sources/SegmentationEngine.swift`
- 函数：`makeCHWFloat32Input(pixelBuffer:)`
- 实现：
  - 双重 for-loop 遍历 1024×1024（≈1,048,576 像素）
  - 每像素读取 BGRA 4 bytes，并写入 3 个 float32 plane（R/G/B）

**预期热点**：这是典型的 CPU 带宽/循环热点（1M 次 * 多次内存访问），很可能是 encoder 调用中“最可优化”的部分之一。

### 1.3 CoreML encoder 执行
- `encBox.model.prediction(from:)`
- 计时字段：`encMs`

**已知事实**：CoreML aggregation 给出 `.all` 下 encoder prediction avg ~638ms。

### 1.4 mask 后处理/渲染（严格来说属于 decoder 后半段）
- `thresholdMask256`（256×256 float32 → UInt8）
- `makeCameraMaskImage`：
  - 256→1024 upsample（CIImage scale）
  - crop active region
  - scale back to camera
  - `CIBlendWithAlphaMask`
  - `CIContext.createCGImage`

**预期热点**：`createCGImage` 可能在峰值时拉高 segMs；你们已把 CIContext 做成 static（正确）。

---

## 2) What Instruments should show（建议的 Instruments 采集方式）

> 目标是把一次 segmentation 的时间窗口切成 4 段，并分别看热点栈。

### 2.1 推荐 Instruments 配置
- Time Profiler（必须）
- Core ML（必须，用于区分 encoder/decoder 与 copy/transform）
- Points of Interest（强烈建议；需要在代码加 signpost）

### 2.2 采集时的控制变量（避免噪声）
1) Profile 模式下暂停 camera（你们已有 `cameraEnabled=false` 策略）
2) 只跑一次 encoder（强制 embedding cache miss）：
   - 临时把 `encoderEveryNFrames=1`（或清掉 cachedEmbedding），触发一次 encoder
   - 然后停止，避免长时间录制引入热/调度变化
3) 关闭 Run Golden，避免 contention 污染

### 2.3 必须加的 signpost（否则 Time Profiler 很难对齐窗口）
建议在 `SegmentationEngine` 的 worker queue 内加 4 个 signpost 区间：
- `SAM.preprocess_1024`
- `SAM.pack_CHW_float32`
- `SAM.encoder_prediction`
- `SAM.mask_post_and_render`

并把 `frameIndex`、`camW/H`、`newW/H`、`computeUnits` 写入 signpost metadata。

这样在 Instruments 的 POI 时间轴上能精确框选每段，再看 Time Profiler 的 call tree。

---

## 3) Expected hotspot ranking（当前最可能的热点排序：基于代码结构的推断）

> 这是“可验证假设”，不是最终结论；一旦加了 signpost + 细分计时，就能证伪/证实。

1) **CoreML encoder backend compute（~638ms avg @ .all）**
   - 已被 CoreML instrument 证实是大头。
2) **MLMultiArray pack（双重 for-loop 1024×1024）**
   - 虽然当前没有独立计时，但极可能是 CPU 侧第二大头（尤其在 `.cpuAndGPU` 更慢时）。
3) **CI preprocess render（1024 resize+pad）**
   - 取决于是否走 GPU/是否发生额外拷贝。
4) **mask 后处理/渲染（createCGImage + blend）**
   - 通常是峰值来源（p95/p99），均值未必最大。

---

## 4) Actionable optimizations（按收益/风险排序）

### 4.1 低风险：先把“pack 时间”单独打点并优化
- 现在 `encMs` 不含 pack；建议新增：
  - `sam_preprocess_ms`
  - `enc_input_pack_ms`
  - `enc_ms`（保留）
  - `mask_post_ms`
  - `mask_render_ms`

**优化方向（pack）**：
- 用 Accelerate/vImage 做 BGRA → planar float 的解交织/转换（避免 Swift per-pixel loop）。
- 或者把 encoder 输入改成 **ImageType/CVPixelBuffer**（需要重新导出/包装模型，收益大但需要 ML_Vision 支持）。

### 4.2 中风险：encoder 输入 dtype 从 float32 → float16（若模型允许）
- 现在 encoder 输入 `MLMultiArray float32`，内存带宽压力大。
- 若能改为 float16：
  - pack 写入量减半
  - CoreML 可能更容易走高效路径

### 4.3 中风险：把 1024 preprocess 从 CI 迁到 vImage 或 Metal
- CIContext.render 在某些设备/场景下会引入隐式同步/拷贝。
- 使用 Metal compute shader 做 resize+pad，并直接产出 CHW buffer（一步到位）通常更快。

### 4.4 高风险/中期：进一步降低 encoder 调用频率
- 这不是“加速 encoder”，而是“少跑 encoder”。
- 当前策略已经正确：embedding cache + encoder cadence。
- 建议再加“错峰/互斥”：YOLO decode+nms 与 SAM encoder 不在同一帧窗口执行（降低 contention tail latency）。

---

## 5) Deliverable status（本任务交付状态）

- ✅ 已输出本文件：`shared/day6_sam_encoder_profile.md`
- ⚠️ 需要进一步补齐的“硬证据”（下一轮采集建议）：
  1) 加 POI signpost 后，分别截图/记录四段（preprocess/pack/enc/post-render）的平均与 p95
  2) Time Profiler 在 `SAM.pack_CHW_float32` 窗口内的 Top stacks（应能看到 `makeCHWFloat32Input`）
  3) CoreML instrument 对 `Input Copy / Output Transformation` 的占比在 pack 优化前后对比

---

## Appendix A — 当前实现的关键函数列表（便于在 Instruments 中搜索）

- `MobileSAMPreprocess.makeInput(...)`
- `SegmentationEngine.makeCHWFloat32Input(pixelBuffer:)`
- `MLModel.prediction(from:)` for `MobileSAM_ImageEncoder`
- `SegmentationEngine.thresholdMask256(...)`
- `SegmentationEngine.makeCameraMaskImage(...)`
- `CIContext.render(..., to: CVPixelBuffer)`
- `CIContext.createCGImage(..., from: ...)`
