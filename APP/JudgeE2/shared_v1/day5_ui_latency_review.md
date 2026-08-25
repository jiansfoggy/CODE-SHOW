# Day 5 — UI thread & Latency Review (v1)

日期：2026-02-18
设备：iPhone 11 / iOS 17（日志来自真机运行）
范围：主线程阻塞点、UI 卡顿风险、以及与 SwiftUI/Canvas 渲染相关的潜在性能问题。

> 限制：当前无 Instruments trace 附件，因此本报告以代码路径静态分析 + console timing 日志为依据，给出高风险点与可执行修复建议。建议后续补一份 Instruments（Time Profiler + Main Thread Checker）截图/trace 以确认热点占比。

---

## 1. 结论摘要

- **主线程 publish warning 已消除**（Publishing changes from background threads…）：通过将 camera onFrame→detector/segmenter 调用切回 MainActor，避免 SwiftUI/Combine 警告。
- 当前 UI 卡顿风险主要来自：
  1) **MaskOverlay 每帧绘制 camera-resolution `CGImage`**（Canvas draw）
  2) **YOLO decode+nms (~115–155ms) 属于 CPU 热点**，会间接导致 UI 更新不流畅
  3) `Run Golden` 与 realtime detector/segmenter 并发时，出现一次 decode 峰值 `~588ms`，疑似 CPU 争用/资源竞争引起。

---

## 2. 主要风险点（按优先级）

### Issue D5-UI-001 — MaskOverlay 每帧绘制大尺寸 CGImage，可能造成主线程/渲染线程压力
- **位置**：`swift_app/JudgeEverythingApp/Sources/MaskOverlay.swift`
- **现状**：
  - `Canvas { ctx.draw(Image(cgImage), in: rect) }` 每次 body 更新都会绘制整个 mask 图。
  - mask 图是 camera-resolution（例如 1080×810 或类似），即使 seg 低频，UI 仍会每帧重绘。
- **风险**：
  - SwiftUI Canvas 的绘制调度可能在 UI/渲染线程上增加开销。
  - 若 mask 图像频繁变化（seg cadence 提高），可能引入 UI jitter。
- **建议**：
  1) 将 mask 渲染改为 **Metal**（纹理采样 + alpha blend），避免 CPU/CGImage 路径。
  2) 若短期保持 CGImage：仅在 `cachedMask` 更新时刷新显示（避免每帧生成新 Image 对象）；或把 mask 存为 `Image` 缓存。
  3) 为 MaskOverlay 增加 debug 开关，必要时关闭 mask 以隔离 UI 抖动来源。

### Issue D5-UI-002 — Run Golden 与 realtime pipeline 并发导致 decode 峰值
- **证据**：日志中出现一次：`decode+nms 587.78 ms`，紧邻 `Run Golden tapped`。
- **风险**：
  - golden 路径会跑 encoder/decoder + CI 处理，与 detector decode 争用 CPU/内存带宽。
- **建议**：
  1) Run Golden 时临时暂停 realtime detector/segmenter（或降低频率），避免并发。
  2) 或将 golden 测试放入独立模式（静态图像，不开 camera）。

### Issue D5-UI-003 — SegmentationEngine 的图像转换路径重（但在后台线程）
- **位置**：`SegmentationEngine.makeCHWFloat32Input`（1024×1024 BGRA→NCHW float32）
- **现状**：每次 encoder 运行要做 ~1024×1024 像素循环和 3 个 plane 写入。
- **影响**：虽然在后台，但会增加整体 CPU 占用，导致 UI/decoder 竞争。
- **建议**：
  1) 尝试用 Accelerate/vImage 做 BGRA→PlanarRGB + float 转换。
  2) 若可行：把 encoder 的 CoreML 输入改为 ImageType（减少 Swift 侧拷贝），或在导出阶段支持 image input。

---

## 3. 建议的 Instruments 采集步骤（用于补齐证据）
1) Time Profiler：关注 Main Thread + Render Server
2) Core Animation：检查 FPS & frame time
3) 采集两段：
   - A: detector-only（关 seg）
   - B: detector+seg（当前 cadence）
4) Run Golden 时再采一段，确认 decode 峰值来源。

---

## 4. 可复现步骤
- 打开 app，等待 modelReady=true
- 观察 HUD：infer/decode/seg
- 点击 Run Golden，观察 decode 峰值是否复现
- 开关 seg cadence（everyNFrames）对比 UI 流畅度
