# Day3 Device Bench (iPhone 11) — YOLOv9 Detector

日期：2026-02-18
设备：iPhone 11 (A13)
目标：给 Day4 Gate 使用的**可复现**性能基线，拆分并记录：
- model load（冷启动）
- warm-up（一次 dummy predict）
- first predict（首帧推理）
- steady-state predict（稳定阶段）
- decode+nms（后处理）
- backpressure 策略影响（latest-frame 单槽）

> 注意：此文件是 Debugger 交付物（Gate 要求）。

---

## 0. 测试环境
- App：JudgeEverything
- 分支/commit：未记录（建议后续补上 git hash）
- Xcode：/Applications/1 Workflow/Xcode.app
- computeUnits：已完成 AB 样本（`.all` vs `.cpuAndGPU`）。当前观测：
  - `.all`：**load 慢（~9.3s）但 infer 快（~0.18–0.20s）**
  - `.cpuAndGPU`：**load 快（~2.1s）但 infer 慢（~0.9–1.0s）**
  建议后续用 2–3 次冷启动复测确认稳定性，并根据产品目标选择默认值（调试阶段可偏向更快启动；实时阶段偏向更高 FPS）。
- 预处理：CIImage letterbox → 640×640 BGRA CVPixelBuffer
- 输出：使用 `var_3019` (1,84,8400) float32
- 后处理：Swift decode + NMS（带 preNmsTopK 截断）

---

## 1. 当前已采集的真机日志证据（computeUnits = .all）

### 1.1 冷启动 model load
证据（来自控制台日志）：
- `.all`：`[YOLOv9cModelCache] loadModel computeUnits=... 9322.42 ms`
- `.cpuAndGPU`（用户提供）：`[YOLOv9cModelCache] loadModel computeUnits=... 2100.00 ms`

结论：
- 冷启动 `MLModel(contentsOf:)` 在 `.all` 下约 **9.3s**；在 `.cpuAndGPU` 下约 **2.1s**（差异巨大）。
- 这强烈建议 Day4 默认先用 `.cpuAndGPU`（至少在调试阶段），但仍需做 2–3 次冷启动复测确认稳定性。

### 1.2 warm-up（相机启动后，Strategy B：预览立即显示，推理等待 modelReady）
证据（computeUnits=.all 样本）：
- `[YOLOv9cModelCache] warmUp prediction 385.24 ms`
- `[CameraPreview] modelReady=true`（出现后才开始推理）

证据（computeUnits=.cpuAndGPU 样本，用户提供）：
- `[YOLOv9cModelCache] warmUp prediction 2811.45 ms`

结论：
- warm-up 是一次 dummy predict。
- 在 `.all` 下 warm-up 约 **385ms**；在 `.cpuAndGPU` 下 warm-up 约 **2811ms**（显著更慢，需进一步确认是否偶发/是否与 GPU pipeline 初始化有关）。

### 1.3 steady-state 推理与后处理（warm-up 后）
证据样本：
- `[DetectorEngine] ... getModel 0.00 ms | infer 195.38 ms | decode+nms 116.99 ms | dets=2`
- `[DetectorEngine] ... getModel 0.00 ms | infer 173.17 ms | decode+nms 114.71 ms | dets=2`

结论：
- `getModel` 已接近 0ms（缓存命中 OK）
- 推理（infer）约 **173–195ms**
- decode+nms 约 **115–117ms**（占比偏高，需继续优化）

---

## 2. 仍需补齐的 AB / 多轮统计（Gate 未完成项）

### 2.1 computeUnits AB：`.all` vs `.cpuAndGPU`
**为什么要做**：确认 ANE/GPU/CPU 的选择是否影响：
- 冷启动 load
- warm-up
- steady infer

**怎么做（建议操作步骤）**：
1) 在 `Sources/YOLOv9cModelCache.swift` 临时把 `get(computeUnits:)` 的调用入口改为 `.cpuAndGPU`（或提供 UI 开关）。
2) 冷启动 App（杀进程后重启），记录：
   - `[YOLOv9cModelCache] loadModel ... ms`
   - `[YOLOv9cModelCache] warmUp prediction ... ms`
   - 3–5 条 steady-state `[DetectorEngine] ... infer ... decode+nms ...`
3) 对比 `.all` 的同类数据。

**记录表（待填写）**：

| computeUnits | loadModel(ms) | warmUp(ms) | steady infer(ms) | steady decode+nms(ms) | 备注 |
|---|---:|---:|---:|---:|---|
| .all | 9322 | 385 | 173–195 | 115–117 | 已采集（Strategy B + modelReady gate 后） |
| .cpuAndGPU | 2100 | 2811 | 901–1024 | 132–133 | 用户提供日志；dets=0（可能阈值/场景导致） |

### 2.2 多轮统计（降低偶然性）
建议每个 computeUnits 跑 3 次冷启动（kill app → reopen），记录 loadModel 的 min/median/max。

---

## 3. 性能瓶颈结论与建议（面向 Day4）

### 3.1 冷启动 load 9s
- 优先级：High
- 建议：
  - 先做 `.all` vs `.cpuAndGPU` AB
  - 若都慢：推动 ML_Vision 导出 `.mlpackage`/`mlprogram` 路线；并考虑“延迟加载/进入相机后再加载”的产品策略。

### 3.2 decode+nms 115ms
- 优先级：Medium
- 已落地：preNmsTopK 截断 + latest-frame 单槽 backpressure
- 建议：
  - 进一步下调 `preNmsTopK`（例如 200）
  - 适当提高 `scoreThreshold`（例如 0.35/0.5）
  - 如仍偏高：考虑 vDSP/向量化或更高效 NMS。

---

## 4. 附：关键源码位置（便于复现/审计）
- warm-up + load 打点：`Sources/YOLOv9cModelCache.swift`
- 推理耗时拆分日志：`Sources/DetectorEngine.swift`
- 推理 gate（modelReady）：`Sources/CameraPreview.swift`
- decode/NMS 优化（stride 校验、preNmsTopK）：`Sources/YOLOv9Decoder.swift`
