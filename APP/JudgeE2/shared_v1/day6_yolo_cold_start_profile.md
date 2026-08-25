# Day 6 — YOLO Cold Start Instruments Profile (D6-D-LOAD-PROFILE)

日期：2026-02-18
设备：iPhone 11（A13）
iOS：
App build：Debug/Release：
Xcode：

目的：用 Instruments 定位 YOLO cold start（首次 `MLModel(contentsOf:)` 与首次 prediction）期间的主线程阻塞点/符号栈，并判断是否可通过预热、线程调整、模型格式（mlprogram）等方式改善。

> 当前状态：Builder 已完成按钮/marker；但此调试环境的 `xcodebuild` 仍指向 CommandLineTools（无法直接启动 Xcode/Instruments 自动化采集）。需要在 Xcode UI 中手动 Profile 并导出 `.trace` / 截图后回填本报告。

---

## 1) Instruments 配置

建议同时采集：
- Time Profiler（必须）
- Core ML（可选，但推荐）
- (可选) Points of Interest / os_signpost（若后续加 signpost）

采集阶段：
1) App 启动 → 触发 `YOLOv9cModelCache.get()`
2) 紧接着触发 `warmUpIfNeeded()` 或首次 `prediction`

---

## 2) 操作步骤（必须，按此顺序采集）

1) Xcode 打开工程：`JudgeEverything/swift_app/JudgeEverythingApp.xcodeproj`
2) 选择真机 iPhone 11 作为运行目标
3) Product → **Profile**（或 Cmd+I）
4) Instruments 中选择模板：
   - **Time Profiler**（必须）
   - 再添加：**Core ML** instrument（推荐）
5) 点击 Record 开始录制后，在 app 内按顺序点击 Builder 新增的按钮（每次间隔 2–3 秒，便于切分区间）：
   - `[YOLO_PROFILE] LOAD`（只触发 `MLModel(contentsOf:)`）
   - `[YOLO_PROFILE] WARMUP`（只触发首次 prediction warmup）
   - `[YOLO_PROFILE] STEADY`（连续 N=30 次 prediction）

6) 停止录制，保存 trace 文件（.trace）。

> 采集注意：
> - 建议分别对 `.cpuAndGPU` 与 `.all` 各采一条 trace（两条 trace），便于对比 ANE 相关回退/初始化。
> - 尽量关闭其他后台重负载，避免噪声。

---

## 3) 需要输出的证据（从 trace/截图整理到本文件）

### 3.1 主线程热点栈（Top stacks，前 5）

**LOAD 阶段（MLModel(contentsOf:)）Top 5**
> 待补（请确认该 Top5 属于 LOAD 还是 WARMUP；你刚贴的看起来更像 warmup/首次 dispatch）。

**WARMUP 阶段（first prediction）Top 5（来自你提供的 Time Profiler Top Stacks）**
1) `try_dispatch(std::shared_ptr<Espresso::abstract_context>, ...)` — 206ms (23.3%)
2) `Espresso::cpu_context_transfer_algo_t::assign_to_fallback_context_v2(...)` — 133ms (15.1%)
3) `Espresso::cpu_context_transfer_algo_t::compute_graph_shortest_path_v3(...)` — 131ms (14.8%)
4) （未提供，待补）
5) （未提供，待补）

**STEADY 阶段（N=30 prediction）Top 5（可选）**
1) 
2) 
3) 
4) 
5) 

### 3.2 Core ML 事件（Core ML instrument）
- model load/compile：
- warmup prediction：
- steady prediction：
- 是否发生 ANE 回退/初始化（是/否；如是，相关日志/事件）：

---

## 4) 结论与建议（填写）
- cold start 时间主要花在：
  - CoreML compile？
  - ANE 初始化？
  - 文件 IO / 解压？
  - Swift/FeatureProvider 构建？
- 是否存在主线程同步等待（应该避免）：
  - 若是：建议将 load/warmup 完全异步，并在 UI 上显示进度
- 是否建议迁移到 `.mlpackage/mlprogram` 或更改 computeUnits 默认值

---

## 5) 附件
- Instruments trace 文件（.trace）路径：`/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/record.trace`
- 备注：trace 内含 `instrument_data/*/run_data/1.run.zip`（已确认文件存在）。
- 截图：
