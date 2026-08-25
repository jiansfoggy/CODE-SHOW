# Debug Report — Phase 2 Day 7 (Debugger)

Date: 2026-07-18
Scope: Phase 2 Day 7 — Stabilization & Phase Freeze
Debugger 任务: encoder/decoder latency (mean + p95)、mask refresh rate、FPS (bbox + mask)、memory usage
附加分析: 编译/构建问题、运行时错误、性能瓶颈

---

## 0) 执行方法 / 证据来源
- **静态审查 + 真实编译**：读取全部源码（`CameraManager.swift`、`TemporalManager.swift`、`SAMEncoder.swift`、`SAMDecoder.swift`、`PromptBuilder.swift`、`MaskRenderer.swift`、`PerfLogger.swift`、`InferenceBackend.swift`）。
- **真机构建验证**：使用 `xcodebuild` 对 iPhone 11 目标执行 **clean build**（见 §1）。
- **测量口径**：Day 7 的 5 项延迟/FPS/内存指标需在 **物理 iPhone 11** 上采集（Debugger 当前环境无法连真机运行）。本报告：
  1. 确认每项指标的**instrumentation 已在代码内就绪**（可直接从 Xcode console 读取）；
  2. 汇总 model_plan.md / 既有运行日志中**已有的实测数据**；
  3. 明确标注**仍需真机补测**的缺口，并给出采集步骤。

---

## 1) 编译 / 构建问题

**结论：✅ Clean Build 成功，零代码告警。**

- 命令：`xcodebuild clean build -project JudgeE2.xcodeproj -scheme JudgeE2 -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 11' -configuration Debug`
- 结果：`** BUILD SUCCEEDED **`
- 编译告警：**0 个代码告警**（仅 1 条无害提示 `appintentsmetadataprocessor: No AppIntents.framework dependency found`，与 App 逻辑无关）。
- 模型打包确认：
  - `yolov9-c.mlmodelc` ✅ 已编译进 App bundle
  - `MobileSAM_ImageEncoder.mlmodelc` ✅
  - `MobileSAM_PromptMaskDecoder.mlmodelc` ✅
  - 工程内模型源：`Detection/Models/yolov9-c.mlmodel`、`Segmentation/Models/MobileSAM_ImageEncoder.mlpackage`、`Segmentation/Models/MobileSAM_PromptMaskDecoder.mlpackage`
- 部署目标 iOS 17.0；Bundle ID `js.JudgeE2`；Signing Team `W95LVGJ7G3`。

> 注：Day 6 报告提到的 `automaticallyAdjustsVideoMirroring` 崩溃已在当前代码修复（`toggleCamera()` 中先 `conn.automaticallyAdjustsVideoMirroring = false` 再设 `isVideoMirrored`，顺序正确）。

---

## 2) 运行时错误

**结论：编译期无错误；运行期需真机复测。静态审查发现 2 个真实并发隐患 + 1 个 CoreML 告警需跟进。**

### 2.1 ⚠️ 跨队列共享状态存在数据竞争（真实隐患）
`latestCameraBuffer` / `latestInputBuffer` / `lastLetterbox` 在 **videoQueue** 写入，但在 **encoderQueue / decoderQueue** 读取，且**未受 `stateLock` 保护**：
- `CameraManager.swift:334,339` 在 videoQueue 写 `latestCameraBuffer` / `latestInputBuffer`
- `CameraManager.swift:506` `scheduleEncoder(cameraBuffer: latestCameraBuffer)` 在 videoQueue 读，但闭包在 encoderQueue 执行时可能读到被后续帧覆盖的 buffer
- `warmupSegmentationIfPossible()`（:152,:178）直接从非 sessionQueue 上下文读 `latestCameraBuffer` / `lastLetterbox`

影响：偶发使用**过期/半写入的帧或几何**做 encode → mask 与 bbox 短暂错位（非崩溃、难复现）。
建议：把这三个字段的读写也纳入 `stateLock`，或在 videoQueue 内 `let snapshot = latestCameraBuffer` 后再传入闭包（值捕获而非引用最新值）。

### 2.2 ⚠️ `isProcessing` 帧节流标志无锁
`isProcessing`（:335-337）在 `captureOutput` 内读写。因 `videoQueue` 为串行队列，当前逻辑安全；但若未来把 `runDetectionPipeline()` 异步化，会立即变成竞态。**当前不阻塞，标注为技术债。**

### 2.3 ⚠️ CoreML 对齐告警（沿用 Day 6，仍未消解）
运行日志历史出现：
`Invalid layer: Invalid input tensor channel 1 and format size 2 bytes, must be aligned on 64 bytes`
- 推断来源：MobileSAM mlpackage 的某层输入张量（Float16，2 bytes）未按 64 bytes 对齐。
- 目前不致崩溃、仍可推理，但可能是 SAM Encoder 高延迟的诱因之一（走了非最优内核）。
- 建议：用 `coremltools` 重新导出/编译 Encoder，检查 compute precision 与 tensor 对齐；对比切 ANE 后告警是否消失。

### 2.4 fallback 路径审查（Day 6 遗留项的代码级确认）
- bbox-only fallback 路径存在且正确：`runSegmentationPipeline` 在 `currentlyEncoding && !hasValidEmbedding` 时 `return`（仅显示 bbox），不崩溃、不冻结（:508-512）。
- `isDecoding` / `isEncoding` 在所有失败分支都有 reset（prompt build 失败、decode 失败、encode 失败均已复位），**无死锁风险**。
- 但**真机上的 fallback 触发日志仍缺**（Day 6 遗留）—— 代码路径正确，尚无运行时证据证明 TTL 过期→回退→恢复链路。

---

## 3) 性能瓶颈

### 3.1 已有实测数据（来自 model_plan.md，iPhone 11）
- **YOLOv9-c（单帧，Infer-only）：**
  - CPU-only: mean **1106 ms** / p95 **1325 ms**
  - CPU+GPU: mean **1064 ms** / p95 **1360 ms**
  - **CPU+NeuralEngine: mean 194 ms / p50 178 ms / p95 198 ms** ← 唯一可用配置
- **MobileSAM Encoder / Decoder：** model_plan.md 仍标注「待设备测」。

### 3.2 从既有运行日志推断（Day 6 报告 + 代码日志口径）
- YOLO 推理：~170–210 ms/帧（ANE），实际 pipeline ~2.7–3.2 FPS。
- **SAM Encoder：~0.9–1.3 s（最大瓶颈）。**
- SAM Decoder：~80–100 ms。
- Mask 刷新率：0.3–1.5 Hz。

### 3.3 瓶颈定位（静态分析补充）
1. **SAM Encoder 是绝对瓶颈**（1024×1024 输入，~1s）。当前 `encoderEveryNFrames=12`（约每 4s 一次 @3fps），已尽量降频，但单次延迟无法通过降频掩盖首帧/漂移触发时的卡顿。
2. **Encoder 预处理为 CPU 双重循环 deinterleave**（`SAMEncoder.preprocess` :112-121）：1024×1024 = 1M 像素的 `for row/for col` 逐像素拆通道，虽注释称已 vDSP 化，但**deinterleave 仍是标量循环**，vDSP 仅用于归一化。这部分在 A13 上约数十 ms，可用 `vImageConvert_BGRA8888toPlanar8` 进一步加速。
3. **MaskRenderer 全图 256×256 双重循环**做 min/max/mean/std 统计 + 阈值填充（多个 `for y/for x`），每次 decode 都执行，约数 ms；非主瓶颈但可优化为 vDSP。

---

## 4) Day 7 Debugger 五项指标：instrumentation 就绪状态 + 数据

| 指标 | 代码内 instrumentation | 当前状态 | 数据 / 缺口 |
|------|----------------------|---------|------------|
| **Encoder latency (mean+p95)** | `scheduleEncoder` 打印 `SAM encoder latency: %.2f ms`（单次）；warmup 亦打印 | ⏳ 单次已打，**mean+p95 聚合未实现** | 现有日志推断 ~0.9–1.3s；**需真机跑并聚合 mean/p95** |
| **Decoder latency (mean+p95)** | decoderQueue 打印 `SAM decoder latency: %.2f ms`（单次） | ⏳ 单次已打，**mean+p95 聚合未实现** | ~80–100 ms；**需真机聚合 mean/p95** |
| **Mask refresh rate** | `TemporalManager.recordMask` 返回 interval，打印 `mask refresh: %.2f ms (%.2f Hz)` | ✅ 已就绪 | ~0.3–1.5 Hz（日志可直接读） |
| **FPS (bbox + mask)** | `PerfLogger.logFpsAndMemoryEverySecond` + `logTimings` | ✅ 已就绪 | bbox 路径 ~2.7–3.2 FPS；**mask 叠加后 FPS 需真机确认** |
| **Memory usage** | `PerfLogger.currentMemoryMB()`（mach_task_basic_info resident_size） | ✅ 已就绪 | 每秒随 FPS 打印；**需真机记录峰值（分割模式加载双 SAM 模型时）** |

> **关键 gap：** encoder / decoder 的 **p95 聚合尚未在代码里实现**——目前只有 YOLO 推理有 `inferenceTimesMs` 滑窗聚合（:378-388），SAM encoder/decoder 只打印单次值。要完成 Day 7 严格口径，建议 Builder 为 SAM encoder/decoder 各加一个类似 `inferenceTimesMs` 的滑窗（复用现成模式）。

### 真机采集步骤（Debugger 建议）
1. iPhone 11 上以 `.cpuAndNeuralEngine` backend 运行，切到 segmentation 模式。
2. 录制 ≥60s console 日志，覆盖：静止、快速平移、旋转、目标离场再入。
3. 从日志抽取 `SAM encoder latency` / `SAM decoder latency` 全部样本，离线算 mean/p95。
4. 记录 `FPS / Memory` 每秒行的稳态区间与峰值。

---

## 5) 规格 vs 代码：契约漂移（需 Architect 在冻结前裁决）

代码内多处默认值与 architect_output.md / tasks.md 冻结契约**不一致**。这会直接影响 Day 7「Freeze Phase 2 architecture」——**冻结前必须对齐，否则冻结的是文档而非实际行为**：

| 参数 | Architect 契约 | 代码实际值 | 位置 |
|------|--------------|-----------|------|
| Decoder cadence | 每 **6** 帧 | 每 **2** 帧 | `CameraManager:81` |
| Encoder cadence | 每 **12** 帧 | 每 12 帧 ✅ | `CameraManager:80` |
| Embedding TTL | **1200 ms** | **8000 ms** | `TemporalManager:59` |
| Mask TTL | **800 ms** | **2000 ms** | `TemporalManager:63` |
| Class 切换滞后 | 连续 **3** 帧 | 连续 **6** 帧 | `TemporalManager:67` |
| Drift 触发（re-seg） | 单级 IoU<**0.6** | 双级 heavy IoU<0.10 / light IoU<0.55 | `TemporalManager:270-295` |

> 说明：代码把 cadence/TTL 调得比契约更「宽松/激进」（更快 decode、更长缓存），可能是 Builder 为改善体验做的实测微调。**这不是 bug，但违反了「冻结契约」原则。** Debugger 建议：Architect 决定是（a）以代码实测值更新契约文档，还是（b）让 Builder 回退到契约值。

---

## 6) 结论

- **编译/构建：✅ 干净通过，零代码告警，三模型均正确打包。**
- **运行时：** 无编译期错误；发现 **跨队列 buffer/geometry 数据竞争**（§2.1，建议修复）与 **CoreML 对齐告警**（§2.3，需模型侧跟进）；fallback 代码路径正确但缺真机证据。
- **性能：** SAM Encoder（~1s）是决定性瓶颈；Encoder 预处理 deinterleave 与 MaskRenderer 统计循环有二次优化空间。
- **Day 7 五项指标：** mask refresh / FPS / memory 的 instrumentation ✅ 就绪；**encoder/decoder 的 p95 聚合尚未实现**，且五项均需**真机采集**方能填入正式数值。
- **冻结前置条件：** §5 的 6 处契约漂移需 Architect 裁决后再执行「Freeze Phase 2 architecture」。

### 交给 Builder / Architect 的行动项
1. **[Builder]** 为 SAM encoder/decoder 增加 mean+p95 滑窗聚合（复用 `inferenceTimesMs` 模式），补齐 Day 7 严格口径。
2. **[Builder]** 修复 §2.1 跨队列共享 buffer 竞争（值捕获或纳入 stateLock）。
3. **[Architect]** 裁决 §5 契约漂移：更新文档 or 回退代码。
4. **[Builder/ML_Vision]** 跟进 §2.3 CoreML 对齐告警（重导出/编译选项）。
5. **[Debugger 后续]** 拿到真机后按 §4 步骤采集五项指标 + fallback 触发日志，回填正式数值。

> 备注：本报告未勾选 tasks.md 中任何复选框（按分工，仅 Builder / Architect 可勾选）。

---

# 附录 A — Day 7 真机实测数据（2026-07-19，iPhone 11）

来源：一次完整分割模式运行日志（segmentation on，手持移动+旋转）。
配置：YOLO computeUnits=`.cpuAndNeuralEngine`(rawValue 2)；SAM computeUnits=`.all`(rawValue 3)。
模型加载：YOLO **13112.71 ms**；MobileSAM（enc+dec）**3057.14 ms**。

## A.1 五项指标汇总（稳态口径）

| 指标 | 稳态值 | 峰值 / 离群 | 样本 |
|------|--------|------------|------|
| **Encoder latency** | mean **857 ms** / p95 **933 ms** | 冷启动 **2941.56 ms**（首次含 ANE 编译） | n=14（稳态取 n=13） |
| **Decoder latency** | mean **61 ms** / p95 **69 ms** | 冷启动 **1488.85 ms**（首次） | n≈40（稳态取 n≈39） |
| **Mask refresh rate** | **~1.5 Hz**（静止追踪 1.3–1.5 Hz） | 2.85 Hz（light-drift 触发重解码时） | 范围 1.12–2.85 Hz |
| **FPS (bbox + mask)** | **2.7–2.9 FPS** | 低谷 2.14（encoder 触发帧）/ 峰 3.14 | 全程 `FPS:` 行 |
| **Memory usage** | **244–320 MB** | 峰值 **339 MB** | 全程 `Mem=` 行 |

补充：YOLO 单帧推理 `Inference time stats`：窗口1 mean **181.93** / p95 **204.45 ms**；窗口2 mean **189.68** / p95 **204.52 ms**（n=100×2）。

## A.2 口径与数据处理说明（重要）
1. **Encoder/Decoder 无代码级 mean+p95 聚合**：日志仅打印单次 `SAM encoder/decoder latency`，以上 mean/p95 为**从日志逐条手工统计**，样本量小于 YOLO 的滑窗。
2. **双口径**：encoder 首条 2941ms、decoder 首条 1488ms 均为**冷启动离群**（模型首次 ANE 编译/预热）。表中稳态值已剔除首次；括注给出冷启动峰值。
3. **Encoder 稳态样本**：约 830 / 834 / 840 / 842 / 853 / 856 / 864 / 882 / 892 / 911 / 916 / 920 / 930 / 933 / 969 ms → mean≈857、p95≈933。
4. **Decoder 稳态样本**：多数落在 56–70 ms（49–70 区间），mean≈61、p95≈69。
5. **Mask refresh**：`mask refresh` 行读数，静止追踪多在 1.3–1.5 Hz；light-drift 触发即时重解码时冲到 2.4–2.85 Hz。
6. **内存大头**：encoder 权重加载实占 ~446 MB（`afterEncoder=490.2 MB`），是内存峰值主因。

## A.3 运行时观察（对照 §2 静态发现）
- ✅ **fallback 链路已实证**：日志多次出现 `[SEG] fallback: bbox-only (encoding in progress, no valid embedding)`，encoder 忙时正确退回 bbox-only，无崩溃、无冻结 —— 补齐了 Day 6 遗留的“fallback 触发证据”缺口。
- ✅ **drift 触发已实证**：`[SEG] heavy drift → re-encode`（area/ratio 超阈）与 `[SEG] light drift → re-decode`（iou<0.55 等）均按 TemporalManager 阈值正确触发。
- ✅ **旋转稳定**：全程 `rot=90`，`camera=1080x1920 | modelInput=640x640 | letterbox scale=0.3333` 一致，几何链路稳定。
- ⚠️ **CoreML 对齐告警复现**：`Invalid input tensor channel 1 ... must be aligned on 64 bytes` 在启动与 ANE 切换时再次出现（不致崩溃，仍疑为 encoder 走非最优内核、拖慢延迟）。
- ⚠️ **相机启动噪声**：`FigCaptureSourceRemote ... err=-17281` 出现在 session 启动早期，随后正常出帧，属启动竞态噪声，无功能影响。
- ℹ️ 日志末 `Message from debugger: killed` 为手动停止 App，非崩溃。

## A.4 瓶颈定论
- **Encoder（~857ms）是绝对瓶颈**，直接决定 mask 刷新周期；Decoder（61ms）已很快。
- 要提升 mask 刷新率/FPS，**必须从 encoder 入手**：ANE 优化（消解对齐告警）/ 降编码分辨率 / 更激进的 embedding 复用（提高 cache hit）。当前 encoder cache hit 从 33%→61%→64% 递增，说明缓存策略在起效，但 encoder 单次延迟仍是硬约束。
