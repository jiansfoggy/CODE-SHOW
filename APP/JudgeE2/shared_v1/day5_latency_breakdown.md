# Day 5 — End-to-End Latency Breakdown (v1)

日期：2026-02-18
设备：iPhone 11 / iOS 17
范围：端到端 latency 分解（preprocess / infer / decode / segmentation preprocess / encoder / decoder / mask render）。

> 数据来源：console timing 日志（DetectorEngine / SegmentationEngine）。尚缺 Instruments 分位数统计。

---

## 1) Detector (YOLOv9c) latency

来自 James 真机日志样本（多次稳定）：
- `infer`: ~170–225 ms
- `decode+nms`: ~115–155 ms
- `submit→infer`: ~285–340 ms（含排队/调度/其它开销）

异常/峰值：
- Run Golden 并发附近出现 `decode+nms ~588 ms`（疑似 CPU 争用）。

结论：
- decode+nms 是稳定的 CPU 热点（与 Day4 风险点一致）。

---

## 2) Segmenter (MobileSAM split) latency

日志格式：`seg X ms (enc A, dec B)`

观察到：
- 首次触发/热身阶段有 outlier（例如 seg>2s；Builder progress 中亦记录过更大的首次值）
- 稳态：
  - encoder: ~575–613 ms
  - decoder: ~28–36 ms
  - seg total: ~868–905 ms（包含 preprocess + mask 后处理/渲染）

结论：
- split 结构有效：decoder 很快；瓶颈在 encoder。

---

## 3) 当前缺口（建议 Day5 补齐的打点）

为了能真正指导优化顺序，建议 SegmentationEngine 增加更细打点：
1) `sam_preprocess_ms`：camera→1024 resize+pad
2) `enc_input_pack_ms`：BGRA pixelBuffer → NCHW float32 MLMultiArray
3) `enc_ms`
4) `dec_ms`
5) `mask_post_ms`：threshold + 256→1024 upsample + crop + scale back
6) `mask_render_ms`：CIContext createCGImage / Canvas 绘制（如可分离）

同样，DetectorEngine 建议增加：
- `letterbox_ms`（CI render 640×640）
- `feature_provider_ms`（如可）

---

## 4) 优先优化建议（由 latency breakdown 推导）
1) **先做调度策略**：embedding cache + encoder cadence（减少 encoder 调用频率）
2) **再做 YOLO decode 降本**：preNmsTopK/threshold/allowlist 等低风险 AB
3) **最后再做渲染路径升级**：mask overlay 走 Metal（降低 UI/CPU 争用）
