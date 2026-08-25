# Day 4 — Device Performance Baseline (v0)

日期：2026-02-18
目标设备：iPhone 11 (A13)
范围：端到端性能基线（detector-only vs detector+segmenter），以及 decode/NMS 与启动加载的瓶颈归因与可执行建议。

> 重要说明：本机当前 `xcodebuild` 仍指向 CommandLineTools（无完整 Xcode toolchain），因此我无法在此环境直接复现/采集 iPhone 11 的 Instruments 数据与真机日志。本报告以：
> - 代码静态分析
> - 已存在 Day3 bench 文档（`shared/day3_device_bench.md`）
> - 当前 DetectorEngine 日志埋点（infer/decode 分段）
> 为依据，给出 **可立即执行的 perf triage 建议**，并附带一份“你跑一次就能补齐数据”的采集步骤。

---

## 1) 性能现状（来自现有实现/日志点）

### 1.0 iPhone 11 真机实测日志（James 提供，2026-02-18）
**SegmentationEngine（MobileSAM split, strict preprocess）**
- 首次触发（冷启动/首次编译/首次推理开销）：
  - seg=17684.44 ms（enc=9649.69, dec=6555.23）| IoU=0.871 | scale=0.533333 new=1024×576
- 稳态（剔除首次 outlier，后续 6 次）：
  - seg 平均 **861.58 ms**（min 856.24 / max 871.02）
  - enc 平均 **580.03 ms**
  - dec 平均 **25.85 ms**
  - IoU 平均 **0.8358**（min 0.766 / max 0.990）

**DetectorEngine（YOLOv9c）**（19 条样本）
- submit→infer 平均 **322.58 ms**
- infer 平均 **184.65 ms**
- decode+nms 平均 **132.27 ms**
- dets 平均 **2.16**（min 0 / max 5）

### 1.1 分段计时点已具备
`DetectorEngine` 会打印：
- `getModel`（YOLOv9cModelCache.get）
- `infer`（CoreML prediction）
- `decode+nms`（Swift 后处理）

示例 log format：
`[DetectorEngine] submit→infer ... | getModel ... | infer ... | decode+nms ... | dets=...`

### 1.2 当前 segmenter 仍是 placeholder
- `SegmentationEngine` 用 bbox 矩形填充当 mask，且 `usleep(10ms)` 仅为模拟。
- 因此 **detector+segmenter** 的对比目前只能反映“调度开销”与“overlay 开销”，不能代表 MobileSAM 真实开销。

---

## 2) 主要瓶颈与建议（可执行）

### Issue D4-PERF-001 — decode 阶段复杂度偏高（8400×80 全扫描 + NMS），易出现 ~100ms 级 CPU 开销
- **位置**：
  - `swift_app/JudgeEverythingApp/Sources/YOLOv9Decoder.swift`
  - `swift_app/JudgeEverythingApp/Sources/NMS.swift`
- **严重度**：High（直接影响 FPS/掉帧；也是 tasks.md 明确点名的 ~115ms 风险）
- **原因归因**：
  1) 对每个 anchor(8400) 扫描 80 类找 bestCls（672k 次 float 读取/比较），Swift 循环在真机上可能仍较重。
  2) NMS 为 O(M^2) 贪心；虽然 pre-cap 到 300，但按 class 分组 + 多次 sort/dict 仍有额外开销。
- **建议修复（按收益/成本排序）**：
  1) **提高 scoreThreshold 或增加 pre-filter**：例如先用较高阈值（0.35/0.4）验证 perf 上限，再逐步下调。
  2) **调小 preNmsTopK**：当前 300，可尝试 150/200；并记录 dets 数与 mAP/视觉效果权衡。
  3) **限制 class allowlist（场景允许时）**：例如只保留 person/vehicle 等少数类，可把 80 类扫描大幅削减。
  4) **减少内存分配**：
     - `byClass: [Int:[Detection]]` 会频繁分配数组；可改为固定 80 个 bucket（`[[Detection]](repeating:[], count:80)`）避免 dict。
  5) **向量化/Accelerate**（中成本）：将每个 anchor 的 class 最大值搜索改为 vDSP max（需要整理内存布局）。
  6) **把 decode/NMS 下沉到模型图**（高收益/高风险）：导出带 NMS 的 end2end CoreML（mlprogram 更可能），减少 CPU 后处理。

### Issue D4-PERF-002 — 冷启动加载时间（~9s）风险：首帧触发 CoreML compilation + ANE 调度
- **位置**：
  - `swift_app/JudgeEverythingApp/Sources/YOLOv9cModelCache.swift`
  - `swift_app/JudgeEverythingApp/Sources/DetectorEngine.swift`
- **严重度**：High（用户可感知；影响 Day4 里程碑）
- **建议修复**：
  1) **在 app 启动后异步 warm-up**：调用 `YOLOv9cModelCache.shared.warmUpIfNeeded()`，并把 warmUp 与首帧推理解耦（例如 `onAppear` 后延迟 0.2s）。
  2) **bring-up 阶段优先用 `.cpuAndGPU`**：避免 ANE 不支持 op 时的不可控 fallback/初始化成本；稳定后再切 `.all`。
  3) **优先切换到 `.mlpackage`/`mlprogram` 导出**：tasks.md 已建议该路线，通常对 cold start/ANE path 更友好。

### Issue D4-PERF-003 — DetectorEngine 每帧 `get(computeUnits:.all)` 有锁竞争风险
- **位置**：`YOLOv9cModelCache.get` 内部 `NSLock()`
- **严重度**：Medium
- **说明**：虽然缓存命中时开销不大，但每帧仍会 lock/unlock；在高 fps 下可观。
- **建议**：在 `DetectorEngine` 初始化时先 `let model = try get()` 缓存到 engine 私有字段（computeUnits 不变时无需每帧 get）。

### Issue D4-PERF-004 — SegmentationEngine 当前每次 tick 对 detections 进行 sort
- **位置**：`SegmentationEngine.tick`（`detections.sorted`）
- **严重度**：Low（topK=1 时可优化）
- **建议**：用线性 scan 找 max（避免 O(K log K)）。

---

## 3) 需要补齐的“端到端基线数据”（建议你在 iPhone 11 跑一次即可）

### 3.1 detector-only 基线
1) 临时禁用 segmenter（或 `everyNFrames = Int.max`）
2) 运行 30 秒，记录：
   - 平均 infer(ms)
   - 平均 decode(ms)
   - dets 数量分布（例如每帧 dets count）
   - 体感 FPS/掉帧

### 3.2 detector+segmenter（真实 MobileSAM）基线
等 MobileSAM artifacts 到位后：
- encoder cadence（每 N 帧）与 decoder cadence（每 N 帧）分别记录
- 记录 embedding 缓存命中率
- 输出：infer/detect、seg(encoder/decoder)、render 三段耗时

---

## 4) 当前结论（v0）
- decode+NMS 与 cold start 是最确定的性能风险点；建议优先按本文件的 D4-PERF-001/002 逐条落地。
- 真实 MobileSAM 端到端耗时需要等 CoreML artifacts 接入后重新测（当前 placeholder 不具代表性）。

---

## 5) Golden 验收结果（iPhone 11，HUD Run Golden）
- `MobileSAMGoldenTest`：HUD 显示 IoU = **0.963**（截图：`shared/run_golden.PNG`）
- 说明：golden offline 流水线（bus.jpg + mobilesam_bus_case.json + expected mask）已在真机成功跑通。

---

## 6) 补充：本次真机日志新增关键点（2026-02-18）
- YOLOv9c cold load：`~9546 ms`（computeUnits rawValue:2 / 通常对应 cpuAndGPU）
- warmUp prediction：`~226 ms`
- detector 稳态：infer 约 170–225ms；decode+nms 约 115–155ms（个别帧 decode 可飙到 ~588ms，疑似与 Run Golden/segmentation 并发导致 CPU 抢占）
- segmentation：
  - 看到一次 seg=2555ms（enc=879ms dec=188ms）属于 warm-up/缓存未命中或系统调度波动；
  - 多数稳态 seg≈870–905ms（enc≈575–613ms，dec≈28–36ms）
- golden：`IoU=0.9629`（HUD/Console）
