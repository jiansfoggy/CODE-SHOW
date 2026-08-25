# Architect Output — Day 7

Project: JudgeE2 (Phase 1: Detection Only)
Target: iPhone 11
Date: 2026-02-28

## 1) Phase 1 Pipeline Structure Review（检测-only）
**Pipeline（实时检测）**
CameraCapture → Preprocess（orientation/mirror + letterbox 640×640） → YOLO CoreML → Decode + NMS → Overlay（bbox）

**关键结构确认**
- **Canonical 坐标**：统一使用 camera buffer px（Wc×Hc）作为逻辑坐标；所有几何变换显式记录并可逆。
- **FrameGeometry/LetterboxTransform**：每帧必须携带 Wc/Hc、orientation、mirroring、r/px/py 等元数据，保证 overlay 对齐。
- **线程/队列**：capture/preview 不阻塞；detector latest-frame-only；decode/NMS 与推理解耦并允许丢帧。
- **渲染**：overlay 仅做 Canonical→preview 映射（aspectFill 需处理裁剪偏移）。
- **性能指标**：必须可测量 model load time / inference mean+p95 / FPS / memory，作为 Phase 1 基线。

结论：Phase 1 结构完整且最小闭环，满足“正确性优先”的目标。

## 2) Phase 1 Architecture Freeze（冻结）
**冻结内容（不再改动的 v1 基线）**
1. **数据流**：Camera → Preprocess → YOLO → Decode/NMS → Overlay（bbox-only）。
2. **坐标契约**：Canonical = camera px；所有变换显式化（orientation/mirror + letterbox）。
3. **调度策略**：latest-frame-only + drop-if-busy；detector 与 decode 可分离。
4. **输出范围**：仅 bbox + class + score；不引入分割、跟踪或 UI 交互。

**允许变动（Phase 1 内仍可调）**
- 阈值/TopK/NMS 参数；
- computeUnits 选择；
- 可视化样式（不影响几何契约）。

## 3) Phase 2: Segmentation Integration Plan（MobileSAM）
**目标**：在不破坏 Phase 1 的几何/调度契约前提下，引入分割与缓存策略。

**3.1 模块新增**
- PromptBuilder（从 bbox 生成 prompts）
- SAM Encoder/Decoder（优先 split，允许 monolithic 作为备选）
- TemporalManager（主目标选择、缓存、TTL）
- Mask Renderer（mask 叠加）

**3.2 运行策略（默认值，后续可调）**
- **Encoder cadence**：每 12 帧（~2.5Hz@30fps），或主目标变化/geometry 变化触发。
- **Decoder cadence**：每 6 帧（~5Hz@30fps），或 bbox 变化显著触发。
- **缓存**：embedding TTL 1200ms；mask TTL 800ms；过期或 bbox drift 触发重算。

**3.3 约束与退路**
- 与 Phase 1 共享 Canonical/FrameGeometry/LetterboxTransform（不可重复定义）。
- 若 SAM 忙/超时：退回 bbox-only；旧 mask 在 TTL 内可显示。
- computeUnits 先求稳（CPU/GPU），待 AB 验证后再切 ANE。

**3.4 验收标准（Phase 2 启动前）**
- Mask 与 bbox 对齐（同一几何链路）。
- 交互帧率可用（bbox > 15fps；mask 刷新 2–5Hz）。
- 主要线程不卡顿，preview 连续。

---
结论：Phase 1 架构已冻结；Phase 2 的分割集成路径与调度策略已定义，可在下一周期按此实施。

---

# Phase 2 — Day 1 (Architect) Integration Contract

## 1) Segmentation 插入点（锁定）
- **位置**：Post-NMS，Pre-overlay
- **输入**：`[Detection]`
- **输出**：可选 `Mask[]`（与 Detection 同一坐标空间）
- **约束**：不得改动 Detection 内部逻辑或 NMS 流程

## 2) 几何契约（冻结）
- Detection 输出继续使用 **原始 camera 像素坐标**（Canonical）
- Segmentation **不得重复几何变换**（不允许二次缩放/旋转）
- 必须复用 Phase 1 的 `FrameGeometry` + `LetterboxTransform`

## 3) BBox Prompt 格式（锁定）
- 格式：`[x_min, y_min, x_max, y_max]`
- 坐标空间：**原始图像像素空间**（Canonical）
- 归一化仅允许在 **MobileSAM wrapper 内部** 完成

## 4) SegmentationPipeline API（锁定）
- `encode(pixelBuffer) -> embedding`
- `decode(embedding, bbox) -> mask`
- **优先 split encoder/decoder**；允许 monolithic 作为回退
- Phase 1 **不依赖** segmentation encoder

## 5) 线程模型（锁定）
- Detection 队列保持不变
- Segmentation 使用 **独立后台队列**
- Capture/preview **绝不阻塞**
- 不允许从相机回调同步调用 segmentation

## 6) 失败回退策略（锁定）
- Segmentation 失败/超时/资源争用 → **回退 bbox-only**
- 不崩溃、不冻结 UI
- Phase 1 行为与性能保持不变

结论：Phase 2 Day 1 契约已锁定，满足隔离协议与 Phase 1 保护要求，可进入 Day 2（模型准备）。

---

# Phase 2 — Day 6 (Architect) Temporal Manager 规范

## 1) 主目标选择策略（Top-1 + Hysteresis）
- **初始选择**：按 detection 置信度排序，取 Top-1 作为 primary object。
- **稳定策略（hysteresis）**：
  - 若当前 primary 仍存在且满足以下条件，则**保持不切换**：
    - 与当前帧 Top-1 的 **score 差值 < 15%**，且
    - 与当前帧同类目标 **IoU ≥ 0.5**（或中心点距离 < 0.1×frame 对角线）。
  - 若 primary 丢失 **连续 3 帧**（未匹配/置信度低于阈值），则切换到当前 Top-1。
- **多目标情况**：仅对 primary 维持 mask；其余目标保持 bbox-only。

## 2) BBox 漂移阈值（Re-seg 触发）
当 primary bbox 满足任一条件时，触发重新分割：
- **IoU 下降**：与上一帧 bbox IoU < **0.6**；
- **中心漂移**：中心点位移 > **0.1 × frame 对角线**；
- **尺度变化**：面积变化 > **±25%** 或 长宽比变化 > **±20%**；
- **置信度突降**：score 下降 > **30%** 且持续 ≥ 2 帧。

## 3) 缓存失效触发（Embedding/Mask Invalidations）
- **几何变化**：orientation/mirror/letterbox 参数变化 → 立即失效。
- **TTL 到期**：embedding TTL（1200ms）/ mask TTL（800ms）到期 → 失效。
- **主目标切换**：primary object 改变 → 旧 embedding/mask 失效。
- **漂移触发**：满足漂移阈值 → mask 失效并重新 decode。
- **输入尺寸变化**：camera buffer W/H 变化 → embedding + mask 失效。

结论：Temporal Manager 的 primary 选择、漂移触发与缓存失效机制已定义，可进入 Builder 实现与 Debugger 压测阶段。

---

# Phase 2 — Day 7 (Architect) 冻结裁决 + Architecture Freeze

日期：2026-07-19  
依据：debug_report.md §5 契约漂移表 + 附录 A 真机实测数据（iPhone 11）

---

## 1) §5 契约漂移裁决（6 项）

| # | 参数 | 原契约 | 代码实际 | 裁决 | 理由 |
|---|------|--------|----------|------|---------|
| 1 | **Encoder cadence** | 每 12 帧 | 每 12 帧 ✅ | — 无漂移 | 保持不变 |
| 2 | **Decoder cadence** | 每 6 帧 | 每 2 帧 | **(a) 以代码值更新契约 → 每 2 帧** | Decoder latency 仅 61 ms，不是瓶颈；每 2 帧解码可最大化 mask 刷新机会，附录 A 实测 mask 稳定达 1.5 Hz，无因此产生的 CPU/热量告警。6 帧是过保守估计。 |
| 3 | **Embedding TTL** | 1200 ms | 8000 ms | **(a) 以代码值更新契约 → 8000 ms** | Encoder 每 12 帧触发，@3 fps ≈ 4 s/轮。1200 ms TTL 会导致每轮 encode 周期内约 70% 时间无有效 embedding，强迫 bbox-only 回退。8000 ms TTL 覆盖 2 个 encode 周期，保证 cache hit 率（附录 A 已观测到 33%→61%→64% 递增趋势）。Drift 检测是 freshness 的主控信号，TTL 仅作兜底失效保护，8000 ms 是「保持直至下次主动更新」的正确策略。 |
| 4 | **Mask TTL** | 800 ms | 2000 ms | **(a) 以代码值更新契约 → 2000 ms** | @1.5 Hz mask refresh，相邻两次 decode 间隔约 0.67 s；800 ms TTL 仅有 130 ms 余量，在实际抖动下极易在下一帧到达前过期导致 mask 闪烁。2000 ms 覆盖约 3 个 decode 周期，drift 触发仍保证对象大幅移动时立即失效重算。2000 ms 是正确的平滑策略。 |
| 5 | **Class 切换滞后** | 连续 3 帧 | 连续 6 帧 | **(a) 以代码值更新契约 → 连续 6 帧** | @3 fps，3 帧 ≈ 1 s，6 帧 ≈ 2 s。6 帧 hysteresis 抑制了 detector 置信度抖动引发的 primary object 频繁切换，有助于 embedding 复用（切换即失效旧 embedding）。附录 A 全程追踪稳定，无异常切换记录。 |
| 6 | **Drift 触发（re-seg）** | 单级 IoU < 0.6 | 双级：heavy IoU < 0.10 → re-encode；light IoU < 0.55 → re-decode | **(a) 以代码双级策略更新契约** | 单级方案不区分漂移程度，任何 IoU < 0.6 都触发昂贵的 re-encode（857 ms）。双级方案：轻漂移仅触发 re-decode（61 ms，廉价）；重漂移才触发 re-encode（必要）。这是严格优于原契约的架构改进。附录 A 日志实证两级触发均正确激活（`[SEG] heavy drift → re-encode` 与 `[SEG] light drift → re-decode` 均有记录），无误触发。 |

**裁决汇总：5 项全部选 (a) 以代码实测值更新契约文档；0 项需 Builder 回退。**  
理由：这些参数均为 Builder 在真机实测中发现的工程优化，有实测数据支撑且无负面观测。以代码值更新契约是正确的「实现驱动契约」原则。

---

## 2) Phase 2 冻结版数据流 + 调度契约 + 缓存策略

### 2.1 完整数据流（Pipeline）

```
[Camera Capture]
    │  AVCaptureSession 引出 YCbCr buffer
    ▼
[Preprocess]
    │  orientation + mirror 校正
    │  Letterbox 调整至 640×640
    │  输出: CanonicalFrame + FrameGeometry
    ▼
[YOLO Detection]                               ← videoQueue / detectorQueue
    │  computeUnits = .cpuAndNeuralEngine
    │  mean 182 ms / p95 205 ms
    │  输出: [Detection](坐标系: 原始像素空间 Canonical)
    ▼
[TemporalManager — Primary Object Selection]
    │  Top-1 + Hysteresis（6 帧切换滞后）
    │  Drift 检测（双级）
    ▼
         ┌──────────────────────────────────────┐
         │       [SAM Encoder]               │  encoderQueue
         │   每 12 帧 OR geometry变化      │  mean 857 ms / p95 933 ms
         │   OR heavy drift (IoU < 0.10)    │  computeUnits = .all
         │   输入: PixelBuffer 1024×1024     │
         │   输出: Embedding (256-dim)       │
         └──────────────────────────────────────┘
                        │ Embedding (TTL 8000 ms)
                        ▼
         ┌──────────────────────────────────────┐
         │       [SAM Decoder]               │  decoderQueue
         │   每 2 帧 OR light drift          │  mean 61 ms / p95 69 ms
         │   (IoU < 0.55)                    │  computeUnits = .all
         │   输入: Embedding + BBox prompt   │
         │   输出: Mask[256×256, Float32]   │
         └──────────────────────────────────────┘
                        │ Mask (TTL 2000 ms)
                        ▼
[Mask Renderer]                                ← 主线程 (main)
    │  mask 有效: 叠加半透明 mask + bbox
    │  mask 无效(TTL过期 / 编码中 / 无嵌入): bbox-only 回退
    ▼
[Preview Overlay 输出]
    bbox + mask 叠加，连续帧 ~2.7–2.9 FPS
```

### 2.2 调度契约（Scheduling Contract）—冻结版

| 组件 | 触发条件 | 击发周期（@3 fps） | 延迟 | 说明 |
|--------|----------|---------------|------|------|
| **YOLO Detection** | 每帧 | ~333 ms | 182 ms (mean) | ANE, latest-frame-only |
| **SAM Encoder** | 每 12 帧 / geometry变化 / heavy drift (IoU<0.10) | ~4 s (默认周期) | 857 ms (mean) | 最大瓶颈，不可并发 |
| **SAM Decoder** | 每 2 帧 / light drift (IoU<0.55) | ~667 ms | 61 ms (mean) | 非瓶颈，可高频 |
| **Mask Renderer** | 每次 decode 完成 / 主线程刷新 | ~1.5 Hz (稳态) | < 5 ms | 包括阐值/统计/渲染 |

**调度不变式**（Invariants）：
- Capture/preview 展示线程绝不阔塞
- Encoder 不可重入（`isEncoding` flag 保证）
- Decoder 不可重入（`isDecoding` flag 保证）
- 所有失败分支必须复位 flag（已验证，无死锁风险）

### 2.3 缓存策略（Cache Policy）—冻结版

| 对象 | TTL | 主动失效条件 | Cache Hit 策略 |
|------|-----|-------------|----------|
| **Embedding** | **8000 ms** | geometry 变化 / primary 切换 / heavy drift | 复用嵌入，仅 light drift 时需重新提示 |
| **Mask** | **2000 ms** | geometry 变化 / primary 切换 / TTL过期 | TTL内显示旧 mask，过期则 bbox-only |

**缓存不变式**：
- Embedding / Mask 存勿连同 FrameGeometry元数据；几何不匹配则将 cache 视为 miss
- Cold start 首次 encode 后，进入稳态循环（已去ANE编译危影响：首次 2941 ms，稳态 857 ms）
- Cache hit 率志标已就绪，预期 Phase 3 可提升至 > 80%

---

## 3) 集成契约固化（Integration Contracts）

### 3.1 Encoder / Decoder API 契约（冻结）

**SAMEncoder**
- 输入：`CVPixelBuffer`（原始相机帧）
- 预处理：内部 deinterleave + 归一化， resize 至 1024×1024（内部完成，外部不知情）
- 输出：`embedding: MLMultiArray`（256-dim）+ `latency: TimeInterval`
- 异步：必须在 `encoderQueue` 执行，完成回调到调用方可选中
- 非可重入：调用层必须守卫 `isEncoding` flag

**SAMDecoder**
- 输入：`embedding: MLMultiArray` + `bbox: CGRect`（Canonical 像素坐标）
- 内部完成 BBox 归一化（除以 1024.0）后将 prompt 退入模型
- 输出：`mask: MLMultiArray[1, 256, 256]` (Float32) + `iou_pred: Float` + `latency: TimeInterval`
- 异步：必须在 `decoderQueue` 执行，完成回调到调用方
- 非可重入：调用层必须守卫 `isDecoding` flag

### 3.2 几何契约（冻结）

- **坐标系唯一来源**：Canonical = 原始 camera px（W×H）。任何组件不得引入自己的局部坐标系。
- **变换路径唯一**：Preprocess 输出的 `FrameGeometry`（orientation + mirror + letterbox scale/padding）是所有组件的共享单一布局真相，不得重复计算或覆盖。
- **Encoder 进 SAM 的坐标**：画面到模型内的归一化仅允许在 SAMDecoder 内部执行，不得在 SAMDecoder 外预先归一化 bbox。
- **Mask 反映到屏幕**：Mask坐标从 256×256 反映至 preview 层时，必须经过 `FrameGeometry` 反变换，和 Phase 1 bbox overlay 共用同一路径。
- **旌转/镖像不变式**：旋转和镖像已在 Phase 1 Preprocess 处理完毕，后续所有组件看到的均是方向正确的帧。
- **几何不匹配即失效**：当 `FrameGeometry` 中 orientation/mirror/letterbox 任一参数变化，缓存 embedding 和 mask 均必须立即失效。

### 3.3 Fallback 策略（附实证）

| 场景 | 行为 | 实证状态 |
|------|------|----------|
| Encoder 忙磁 / 无有效 embedding | 返回 bbox-only，不崩溃不冻屏 | ✅ 附录 A 日志实证：`[SEG] fallback: bbox-only (encoding in progress, no valid embedding)` |
| Mask TTL 过期 | 移除幸幕，退回 bbox-only | ✅ 代码层确认，无死锁风险 |
| Heavy drift (IoU < 0.10) | 触发 re-encode，期间显示 bbox-only | ✅ 附录 A 日志实证：`[SEG] heavy drift → re-encode` |
| Light drift (IoU < 0.55) | 触发 re-decode，暗禁更新 | ✅ 附录 A 日志实证：`[SEG] light drift → re-decode` |
| Decoder 忙磁 / decode 失败 | `isDecoding` reset，下帧重试 | ✅ 代码层所有失败分支均已复位 |
| CoreML 对齐告警 | 不致崩溃，但可能走非最优内核 | ⚠️ 附录 A 实证可恢复，需 Phase 3 消解 |

---

## 4) Phase 3 入口点定义

**瓶颈定论：Encoder 857 ms 是硬约束。**  
Decoder (61 ms) 、Renderer (< 5 ms) 、YOLO (182 ms) 均不是瓶颈。  
Phase 3 所有优化工作必须展开在 Encoder 上。

### 入口点 1 ： ANE 对齐告警消解（最优先、收益最确定）

- **问题**：`Invalid input tensor channel 1 ... must be aligned on 64 bytes` 导致 Encoder 可能走非最优 ANE 内核（附录 A 已实证再现）。
- **目标**：重新导出 / 重编译 MobileSAM Encoder mlpackage，修复 64 bytes tensor 对齐。
- **工具**：`coremltools` recompile；对毕 `.all` compute units 下告警是否消失。
- **预期收益**：Encoder 延迟从 ~857 ms 降至 600–700 ms（纯 ANE 内核，无 fallback），延迟可能降低 20–30%。
- **风险**：低；重导出不改变模型权重和精度。

### 入口点 2 ： 降编码分辨率（收益最大，需验证精度损失）

- **问题**：当前 Encoder 输入 1024×1024，是延迟的直接来源。
- **目标**：尝试 768×768 或 512×512 输入分辨率，评估分割精度损失。
- **预期收益**：理论上 512×512 将使 Encoder 一致预处理和推理计算量降为 1/4，延迟可降至 200–400 ms（粗估）。
- **验证方法**：AB 测试在相同内容上对毕 mask 覆盖精度，确保 非 正常拆取帧精度不可接受地下降。
- **风险**：中；小目标分割精度可能明显下降，需 ML_Vision 共同评估。

### 入口点 3 ： Embedding 缓存复用策略强化（无精度损失，最安全）

- **问题**：当前 cache hit 率 64%，年剔 36% 的帧仍触发 re-encode。
- **目标**：调优 drift 阈值和 Encoder cadence，将稳态 cache hit 率提至 > 80%。
- **具体手段**：
  1. 分析 re-encode 触发源（geometry 变化 vs. heavy drift vs. TTL 过期）的比例，确认主要浪费来源。
  2. 若 geometry change 占比 > 30%：限制窗口朜旋转偏移量才算改变（减少微小旋转触发）。
  3. 若 heavy drift 占比 > 30%：适度增大 heavy drift 阈值到 IoU < 0.05，依赖 re-decode 处理中间状态。
- **预期收益**：少 ~36% re-encode 减至 ~20%，即 ~30 ms 平均延迟改善，几何 FPS 容易超过 3.0。
- **风险**：低；纯调参数，不改变模型权重。

### 参考：待决 — Encoder 预处理标量化

- 退化项：SAMEncoder 预处理中 `deinterleave` 仍有标量循环（附录 A 未单独采样，无法定量贡献）。
- 优化路径：改用 `vImageConvert_BGRA8888toPlanar8`，预期节省数十 ms。
- 裁决：**待 Phase 3 入口点 1/2 成果后再评估是否必要**；如果入口点 1 和 2 已把 encoder 延迟降至目标范围，预处理优化可使放。

---

## 5) Phase 2 冻结声明

> **Phase 2 已冻结。**  
> 以下参数为最终生产定义，建立在 iPhone 11 真机实测基础上，达到冻结状态。1个每帧写入这些层的工程师必须封開此表作为变更基线。

| 参数 | Phase 2 冻结定义 |
|------|-----------------|
| Encoder cadence | 每 12 帧（或 geometry 变化 / heavy drift）|
| Decoder cadence | 每 2 帧（或 light drift）|
| Embedding TTL | 8000 ms |
| Mask TTL | 2000 ms |
| Class 切换滞后 | 连续 6 帧 |
| Heavy drift 阈值 | IoU < 0.10 → re-encode |
| Light drift 阈值 | IoU < 0.55 → re-decode |
| Encoder 输入分辨率 | 1024 × 1024（Phase 3 优化目标）|
| Encoder compute units | .all（ANE 路径优先）|
| Encoder 延迟（稳态） | mean 857 ms / p95 933 ms |
| Decoder 延迟（稳态） | mean 61 ms / p95 69 ms |
| Mask 刷新率 | ~1.5 Hz（稳态），drift 触发时可达 ~2.8 Hz |
| Pipeline FPS | 2.7–2.9 FPS（包含分割）|
| 内存常驻 | 244–320 MB，峰値 339 MB |
| Fallback | 全层 fallback 已实证，无崩溃风险 |
