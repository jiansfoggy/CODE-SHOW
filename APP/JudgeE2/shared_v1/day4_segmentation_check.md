# Day 4 — Segmentation Correctness Check (v0)

日期：2026-02-18
设备：iPhone 11（目标）
范围：MobileSAM 分割正确性验证（mask 是否与 bbox/物体位置一致，且不发生翻转/偏移），并核对坐标映射链路。

> 重要说明：当前 repo 中 `SegmentationEngine.swift` 仍是 **placeholder**（用 bbox 矩形填充当作 mask，且 `usleep(10ms)` 仅用于模拟异步耗时）。因此本报告只能验证：
> 1) **调度/缓存/overlay 链路不阻塞 detector**；
> 2) **mask overlay 的坐标映射**是否与 `architect_output.md` 的 aspectFill 映射一致；
> 3) golden case 文件已就绪，但缺少 MobileSAM CoreML artifact，无法在设备上跑真实 prompt→mask。

---

## 0. 可用输入（已在 shared/golden）
- 图像：`shared/golden/bus.jpg`
- 期望 mask：`shared/golden/mobilesam_bus_mask.png`
- 期望摘要：`shared/golden/mobilesam_bus_case.json`

缺失：MobileSAM CoreML artifacts（Option A：encoder+decoder split 的 `.mlpackage`/`.mlmodelc`）。

---

## 1. 代码链路核对（当前 v0 实现）

### 1.1 调度策略（符合 Day4 任务的“先正确后快”要求）
- 仅取 top-1 detection：`SegmentationEngine.tick(...)
  -> detections.sorted(by: score).first`
- 每 N 帧触发一次（默认 N=6）：`frameIndex % everyNFrames == 0`
- 独立队列：`DispatchQueue(label: "mobilesam.seg.queue")`
- 缓存：`cachedMaskRect`（未来替换为 bitmap mask）

**结论**：调度/队列隔离正确；不会阻塞相机与 detector（前提：未来 MobileSAM 推理也保持在该队列，并且限制频率/实例数）。

### 1.2 overlay 坐标映射一致性
- `MaskOverlay.swift` 使用与 `BBoxOverlay` 同构的 aspectFill 公式：
  - `s = max(Wv/Wc, Hv/Hc)`
  - `ox=(Wv-Wc*s)/2, oy=(Hv-Hc*s)/2`
  - `camToView(x,y)=(ox+x*s, oy+y*s)`

**结论**：当前 mask overlay 的 mapping 与 `shared/architect_output.md` §A.3 的契约一致，且与 previewLayer 的 `.resizeAspectFill` 匹配。

---

## 2. 需要在 MobileSAM 真正接入后立刻验证的“高风险点”

### Risk S4-SEG-001 — MobileSAM 1024 输入空间的 resize+pad 变换未落地
- **原因**：SAM encoder 输入固定 1024×1024，必然存在 resize+pad（类似 YOLO 的 letterbox）。若 transform 未记录/未统一，将导致：mask 偏移、缩放错误、翻转。
- **建议**：实现 `MobileSamTransform`（字段建议：`r, padX, padY, Wc, Hc, Wi=1024, Hi=1024`），并在 HUD 打印。

### Risk S4-SEG-002 — box prompt 坐标空间可能混用（camera px vs 1024 px）
- **原因**：Detector 输出是 Canonical camera px；MobileSAM decoder 需要 box 在 1024 输入空间（并用 2 点 label=2/3 表示）。
- **建议**：在 Segmenter 内部明确两段映射：
  1) camera px → 1024 input px（使用 MobileSamTransform）
  2) low-res mask (256×256) → 1024 → camera px（逆变换）

### Risk S4-SEG-003 — orientation/mirror
- **原因**：DetectorEngine 当前默认 `exifOrientation = .right`，若前后摄像头/系统版本变化，可能造成“模型看到的方向”和“preview 显示方向”不一致；分割会比检测更敏感。
- **建议**：把 orientation/mirror 作为 `FrameGeometry` 明确字段，且保证 detector 与 segmenter 使用同一“视觉方向”的输入。

---

## 3. 可复现验证步骤（MobileSAM artifacts 到位后）

1) 在 iPhone 11 上运行 app，锁定一帧（或用 `bus.jpg` 作为离线输入）
2) 使用 `mobilesam_bus_case.json` 中的 bbox prompt（或 detector top-1）
3) 生成 low-res mask logits → threshold `>0`
4) 将 mask 映射回 Canonical camera px 并 overlay
5) 对比 `mobilesam_bus_mask.png`（允许一定误差）：
   - mask bbox 是否落在 prompt bbox 内（可允许扩张）
   - mask area 与 json 的 `mask_area_px` 在可接受范围

---

## 4. 当前结论（v0）
- ✅ 调度/缓存/HUD/overlay 链路结构正确。
- ❌ 由于尚无 MobileSAM CoreML artifacts，无法完成“真实 mask 与物体一致”的正确性验收。
