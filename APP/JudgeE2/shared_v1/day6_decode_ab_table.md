# Day 6 — Decode/NMS Preset A/B Table (D6-B-NMS-KNOBS)

目标：仅靠调参（不改算法）将 **steady decode+nms** 压到 **<60–80ms**，并记录 dets 数量分布。

## 旋钮
- `scoreThreshold`
- `preNmsTopK`
- `topK`
- `classAwareNms` (on/off)

## 如何采集（建议）
1. 真机 iPhone 11，稳定场景下运行 30–60 秒（避免大幅转动）。
2. 在 HUD 里切换 preset。
3. 点击 “Start Decode AB” 触发采样（默认收集 60 个样本）。
4. 控制台会打印一行：`[DECODE_AB] preset=... mean=... p95=... detHist=... knobs: ...`
5. 将该行粘贴到下表对应 preset 的 “Console summary” 一栏。

> 备注：decodeMs 指 **YOLO 输出 decode + NMS** 耗时（不含 infer）。

## Preset 结果表

| Preset | scoreThr | preNmsTopK | topK | classAware | decode mean (ms) | decode p95 (ms) | dets hist (0 / 1-5 / 6-20 / 21-50 / >50) | Console summary |
|---|---:|---:|---:|---|---:|---:|---|---|
| baseline | 0.25 | 300 | 100 | on |  |  |  |  |
| fast | 0.35 | 150 | 50 | on |  |  |  |  |
| (add) |  |  |  |  |  |  |  |  |

## Notes
- 如果出现 decode 峰值（>500ms），请同时记录当时是否在跑：Run Golden / segmentation stress。
- 并发干扰策略见：D6-B-CONCURRENCY-GUARD（Detector postprocess 降频/暂停）。
