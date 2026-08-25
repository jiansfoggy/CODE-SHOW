# Day 6 — End-to-End Latency Breakdown (v2, with contention windows)

日期：2026-02-19
设备：iPhone 11 (A13) / iOS 17
范围：端到端 latency 分解 + **并发/争用（contention）时段单独标注**。

> 本文是对 `shared/day5_latency_breakdown.md` 的升级版：
> - 增补 Instruments（Core ML / Time Profiler）AB 证据
> - 把“并发导致的峰值/退化”单独列出来，避免把异常峰值当作稳态

---

## 0) Data sources（证据来源）

### 0.1 Console timing logs（app 内打点）
- `DetectorEngine`：`infer`、`decode+nms`、`submit→infer`
- `SegmentationEngine`：`seg total (enc, dec)`，以及 Day5 已加入的 cadence/cache 统计

（基线总结见 `shared/day5_latency_breakdown.md` 与 `shared/debug_report.md`）

### 0.2 Instruments traces + screenshots（本次 Day6 核心证据）

#### A) computeUnits = `.cpuAndGPU`
- trace：`/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/record2_cpuAndGPU.trace`
- CoreML 截图：`record2_coreML_cpuAndGPU.png`
- Time Profiler 截图：`record2_time_profile_cpuAndGPU.png`

#### B) computeUnits = `.all`
- trace：`/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/record3.trace`
- CoreML 截图：`record3_coreML_all.png`
- Time Profiler 截图：`record3_time_profile_all.png`

> 注：profile 黑屏问题已定位为 Instruments template 默认带 hang 检测项；删除 hang instrument 后恢复（见 `shared/debug_report.md` Day6 Update）。

---

## 1) Baseline（无争用/稳态）Latency breakdown

这里的“稳态”指：
- 模型已经加载完成（不包含 cold load）
- 没有同时跑 Run Golden / profile 触发 / 额外重负载任务
- camera + detector +（可选）segmenter 在常规 cadence 下运行

### 1.1 Detector — YOLOv9c (console baseline)
来自 `shared/day5_latency_breakdown.md`：
- `infer`: ~170–225 ms
- `decode+nms`: ~115–155 ms
- `submit→infer`: ~285–340 ms（含排队/调度/其它开销）

> 结论：decode+nms 是稳定 CPU 热点；infer 本体在 A13 上已经接近可用但仍偏紧。

### 1.2 Segmenter — MobileSAM split (console baseline)
来自 `shared/day5_latency_breakdown.md`：
- encoder: ~575–613 ms
- decoder: ~28–36 ms
- seg total: ~868–905 ms（包含 preprocess + mask 后处理/渲染）

> 结论：encoder 是绝对瓶颈；必须靠 cadence + embedding cache 控制调用频率。

---

## 2) AB（computeUnits）对稳态的影响（Core ML Instrument 证据）

> 这里是 Instruments 的“模型侧”时间（prediction 级别），与 console 的“端到端分段”互补。

### 2.1 AB summary table

| Item | `.cpuAndGPU` (record2) | `.all` (record3) | Interpretation |
|---|---:|---:|---|
| **YOLO Load** | ~0.507 s | **~8.63 s** | `.all` cold load 极慢（关键阻塞） |
| **YOLO Prediction avg** | **~1.05 s** (n=27) | **~229 ms** (n=120) | `.cpuAndGPU` 推理退化到 1s/帧，不可用；`.all` 接近 console 的 170–225ms |
| **SAM encoder avg** | ~1.29 s | ~638 ms | `.all` 对 SAM 也更快 |
| **SAM decoder avg** | ~69.7 ms | ~35.8 ms | `.all` decoder 更快 |

关键结论：
- `.cpuAndGPU`：**启动 load 快**，但 **YOLO 推理 ~1s/帧**（实时不可用）
- `.all`：**推理快**（YOLO ~229ms/帧），但 **cold load 8–9s 级别**（体验/可交互时间最大阻塞点）

### 2.2 推荐的“现实可用”策略（基于 AB 证据）
- bring-up/实时运行：**优先 `.all`**（否则 detector 直接掉到 1s/帧级别）
- UX 侧掩护 cold load：
  - 首屏先起 camera preview（你们已做到）
  - 后台异步 load + warmup
  - HUD/状态提示“模型加载中”（不要让用户以为卡死）

---

## 3) Contention windows（并发/争用）单独标注

本节专门列“当两个重任务同时发生”时的退化现象：它们会造成峰值延迟（p95/p99）爆炸，但不代表稳态。

### 3.1 Contention #1 — Run Golden 并发导致 YOLO decode+nms 峰值
- **证据（console）**：
  - 历史基线（Day5）：Run Golden 并发附近 `decode+nms ~588 ms`；正常稳态 `~115–155 ms`
  - **本次（Day6 logs, mitigation 后）**：
    - Case1：yolo_decode_ms max=223.32ms, p95=168.78ms
    - Case2：yolo_decode_ms max=227.73ms, p95=208.88ms
    - golden_running=1 期间 yolo_decode_ms max=0.00ms（decode 被暂停）
- **判定**：Golden 并发的 decode 峰值已被“暂停 decode”策略抑制；未再出现 >300ms spike。

**建议（可执行）**：
1) Run Golden 时暂停 realtime pipeline（detector/segmenter tick 暂停 1–2 秒），避免污染实时延迟。
2) 或者 Golden 在独立队列执行但加“互斥门”：golden running 时 realtime 只画 bbox，不做 decode/NMS/seg。

### 3.2 Contention #2 — SAM encoder 与 YOLO decode 同时活跃（端到端峰值风险）
- **现象**：当 encoder cadence 触发且同一窗口内 YOLO decode+nms 也执行，会出现明显卡顿（即使平均值看起来还行）。
- **本次证据（Day6 logs）**：
  - Case1：sam_enc_ms=1343.43ms（frame=174）
  - Case2：sam_enc_ms=1179.04ms（frame=360）
  - 该级别 encoder 峰值在同窗期容易与 decode 冲突，需继续做错峰/互斥。
- **根因**：encoder（~1.1–1.3s 级）+ decode（~130–220ms 级）都偏 CPU/内存带宽敏感；同时跑会把 tail latency 拉爆。

**建议（可执行）**：
1) 把 encoder cadence 调到更保守（例如 18–24 帧一次），decoder 事件驱动。
2) 引入“互斥调度”：同一帧窗口内只允许一个重任务：
   - 若本帧刚做过 YOLO decode（或 decode 超过阈值），则延后 encoder 到下一帧。
3) HUD 打印：encoderRuns/decoderRuns、cache hit rate、以及“上次 encoder 距今帧数”，用数据确认策略生效。

### 3.3 Contention #3 — Profile 采样窗口内 camera 回调噪声（Time Profiler 证据）
- **证据**：`record2_time_profile_cpuAndGPU.png` 采样列表里可见 `AVCapture...` / `FigRemote...` backtrace（主线程）噪声。
- **结论**：profile 时最好让 app 进入“纯基准模式”：暂停 camera、暂停 realtime loop，只跑 AB harness marker。

**建议（已在工程里部分实现）**：
- profile 模式继续保持 `cameraEnabled=false`，避免把 AVCapture 噪声混入 CoreML 关键路径。

---

## 4) What to measure next（为了更精确区分“稳态 vs 争用峰值”）

为了把 contention 量化成可复现指标，建议补齐/固化以下统计（如能加到 HUD/日志更好）：

### 4.1 Detector（建议新增）
- `letterbox_ms`（CI render 640×640）
- `candidate_count_preNms` / `candidate_count_postThreshold`
- `nms_candidates_in` / `nms_kept`

### 4.2 Segmenter（建议新增）
- `sam_preprocess_ms`（resize+pad 1024）
- `enc_input_pack_ms`（pixelBuffer → NCHW float32）
- `mask_post_ms`（threshold + upsample + crop + scale back）
- `mask_render_ms`（createCGImage / overlay）

### 4.3 Contention 标注（关键）
- 在日志里给出明确 tag：
  - `[CONTENTION] golden_running=1`
  - `[CONTENTION] encoder_triggered=1`
  - `[CONTENTION] decode_spike=1 (ms=...)`

这样才能在后续统计里把异常区间剥离，得到真实 p50/p95。

---

## 5) Action items（Debugger 视角的结论/建议）

1) **现阶段 computeUnits 选择**：实时运行必须倾向 `.all`（否则 1s/帧）。
2) **把 cold load 当成产品/调度问题**：首屏先预览 + 后台 load + HUD 提示；不要阻塞 UI。
3) **把 contention 当成第一优先级治理对象**：
   - Golden 与 realtime 必须互斥
   - encoder 与 decode 需要调度互斥/错峰
4) 若要真正缩短 `.all` 的 cold load：中期动作是 YOLO 迁移到 `.mlpackage/mlprogram` 并重新做 load profile（见已有 `shared/day6_yolo_mlprogram_card.md` / `day6_yolo_cold_start_profile.md`）。
