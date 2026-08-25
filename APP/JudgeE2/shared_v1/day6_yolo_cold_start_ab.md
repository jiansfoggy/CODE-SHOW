# Day 6 — YOLO Cold Start A/B Benchmark (D6-D-LOAD-AB)

日期：2026-02-24
设备：iPhone 11（A13）
iOS：
App build：Debug/Release：
Xcode：

目的：可复现对比 `MLComputeUnits = .all` vs `.cpuAndGPU` 在 YOLOv9c 的 cold start 与 steady 性能差异。

---

## 0) 测试矩阵（A/B）
- A：computeUnits = `.cpuAndGPU`
- B：computeUnits = `.all`

每组均测：
1) `firstLoadMs`：首次 `MLModel(contentsOf:)`
2) `firstWarmupMs`：首次 prediction（warmup）
3) `steadyPredMs`：稳态 prediction（建议 30 次，丢弃前 3 次 warm cache 波动）

> 建议每组重复运行 5 次（重新启动 app 或强制重启进程），以获得均值与 P95。

---

## 1) 运行前准备（保证可复现）
- 关闭后台高负载任务，保持设备温度稳定（避免热降频）
- 每次 A/B 切换前：
  - 强制杀掉 app（swipe away）
  - 可选：重启设备（用于最严格 cold start）
- 记录：是否连接 Xcode 调试（Debug 连接会影响时延）

---

## 2) 采样方法（建议实现/使用的日志点）

### 2.1 加载时间
- 在 `YOLOv9cModelCache.get(computeUnits:)` 内已有打印：`loadModel ... X ms`
- 把该值作为 `firstLoadMs`

### 2.2 warmup 时间
- 在 `YOLOv9cModelCache.warmUpIfNeeded(...)` 内已有打印：`warmUp prediction X ms`
- 把该值作为 `firstWarmupMs`

### 2.3 steady prediction
- 从 `DetectorEngine` 日志提取 `infer`（prediction）时延：
  - `[DetectorEngine] ... | infer XX ms | ...`
- 建议统计：平均、P50、P95

---

## 3) 结果记录（填写）

### Group A — `.cpuAndGPU`
- firstLoadMs：5678.37 ms
- firstWarmupMs：1715.12 ms
- steadyPredMs（n=30）：
  - mean：1293.99 ms
  - p50：未统计
  - p95：1601.16 ms
  - min/max：1054.81 / 1650.74 ms

### Group B — `.all`
- firstLoadMs：9436.63 ms
- firstWarmupMs：365.66 ms
- steadyPredMs（n=30）：
  - mean：188.53 ms
  - p50：未统计
  - p95：201.43 ms
  - min/max：179.52 / 202.33 ms

---

## 4) 结论与建议
- `.all` 是否显著增加 cold start？
- `.all` 是否显著降低 steady infer？
- 若 cold start 成本过高：建议默认 `.cpuAndGPU` bring-up，后台预热后再切 `.all`（如需要）。

---

## 5) 原始日志/附件
- 原始日志文件：
  - `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/冷启动AB_cpuAndGPU`
  - `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/冷启动AB_ALL`
- 关键日志片段已包含在上述文件（含 load/warmup/steady 统计行）。
