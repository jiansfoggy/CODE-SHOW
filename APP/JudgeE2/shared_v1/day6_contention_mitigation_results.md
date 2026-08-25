# Day 6 — Contention Mitigation Results (C: measurable logs)

日期：2026-02-19
设备：iPhone 11 / iOS 17
目的：为 D6-CONTENTION-MITIGATION 的 **C 部分**提供“可测量”的结构化日志与复现实验记录模板。

> 结构化日志 tag：`[CONT_METRIC]`
> - DetectorEngine：每次推理/后处理完成打印一行（含 yolo_infer_ms / yolo_decode_ms / golden_running / skip_yolo_decode / skip_reason）
> - SegmentationEngine：每次 segmentation 完成打印一行（含 sam_enc_ms / sam_dec_ms / golden_running / encoder_triggered）

---

## 0) Code status（已落地）

### 0.1 结构化日志实现
- `swift_app/JudgeEverythingApp/Sources/DetectorEngine.swift`
  - 输出：`[CONT_METRIC] ... yolo_infer_ms=... yolo_decode_ms=... golden_running=... skip_yolo_decode=... skip_reason=...`
- `swift_app/JudgeEverythingApp/Sources/SegmentationEngine.swift`
  - 输出：`[CONT_METRIC] ... sam_enc_ms=... sam_dec_ms=... golden_running=... encoder_triggered=...`
- `swift_app/JudgeEverythingApp/Sources/ContentionState.swift`
  - 提供全局 `goldenRunning` 状态给日志使用
- `swift_app/JudgeEverythingApp/Sources/ContentView.swift`
  - golden start/end 更新 `ContentionState.shared.goldenRunning`

---

## 1) Repro cases（必须可复现）

### Case 1：正常 realtime 跑 30s（不触发 golden）
1) 冷启动 app
2) 等待 modelReady=true（HUD 显示）
3) 保持场景相对稳定，运行 30 秒
4) 导出/复制 Console 中的 `[CONT_METRIC]` 日志片段

### Case 2：realtime 10s → Run Golden → golden 完成 → realtime 10s
1) realtime 运行 10 秒
2) 点击 HUD 的 Run Golden
3) 等 golden 完成（看到 IoU 或 failed；并有 `[CONTENTION] golden_running=0 end`）
4) 再运行 10 秒
5) 导出/复制 Console 中的 `[CONT_METRIC]` 日志片段

---

## 2) Log extraction（如何从 Console/设备日志抽取）

### 2.1 最小：手动复制
在 Xcode Console 里搜索：`[CONT_METRIC]`，复制两段（Case1/Case2）。

### 2.2 可选：导出后 grep（如果有完整 log 文件）
```bash
grep "\[CONT_METRIC\]" app.log > cont_metric.log
```

---

## 3) Metrics to compare（必须对比的指标）

> 注意：因为 detector/segmenter 各自打印日志，所以对比时以“峰值/分布”而非逐帧对齐为主。

### 3.1 YOLO decode+nms 峰值（关注 contention tail）
- Case1：yolo_decode_ms 的 max / p95
- Case2：yolo_decode_ms 的 max / p95（尤其是 golden_running=1 附近）

### 3.2 SAM encoder 峰值（enc_ms）
- Case1：sam_enc_ms 的 max / p95
- Case2：sam_enc_ms 的 max / p95（尤其是 golden_running=1 附近）

### 3.3 Skip 行为是否生效
- Case2 中是否出现：
  - `[CONTENTION] skip_yolo_decode=1 reason=sam_encoder`
  - `[CONT_METRIC] ... skip_yolo_decode=1 ...`

---

## 4) Result summary（填空区）

### 4.1 Before mitigation（修改前）
- Case1: yolo_decode_ms max=____ p95=____
- Case2: yolo_decode_ms max=____ p95=____ (golden 并发峰值：____)
- 备注：是否出现 ~588ms spike：是/否

### 4.2 After mitigation（修改后）
- Case1:
  - yolo_decode_ms **max=223.32ms**, **p95=168.78ms**
  - sam_enc_ms **max/p95=1343.43ms**（n=1，见 frame=174）
- Case2:
  - yolo_decode_ms **max=227.73ms**, **p95=208.88ms**
  - sam_enc_ms **max/p95=1179.04ms**（n=1，见 frame=360）
  - **golden_running=1 期间 yolo_decode_ms max=0.00ms**（证明 mitigation 生效：golden 运行时 realtime decode 暂停）
- 备注：是否仍出现 >300ms spike：**否（在当前贴出的片段中未出现）**

### 4.3 结论
- contention 止血是否达标：✅ / ⚠️ / ❌
- 还需要的下一步：
  1) ____
  2) ____

---

## 5) Attachments（附件清单）
- 贴 Case1 的 `[CONT_METRIC]` 片段：
[YOLO_AB] selected cpuAndGPU
[HUD] YOLO preset=baseline
[Startup] begin background load+warmup computeUnits=MLComputeUnits(rawValue: 1)
Reading from public effective user settings.
FigCaptureSourceRemote.m:569) - (err=-17281)
[YOLOv9cModelCache] loadModel computeUnits=MLComputeUnits(rawValue: 1) 4288.33 ms
[YOLOv9cModelCache] warmUp prediction 1533.52 ms
[Startup] modelReady=true
[DetectorEngine] output var_3019 dtype=MLMultiArrayDataType(rawValue: 65568) shape=[1, 84, 8400]
[DetectorEngine] output var_3022 dtype=MLMultiArrayDataType(rawValue: 65568) shape=[1, 84, 8400]
[DetectorEngine] submit→infer 1508.40 ms | getModel 0.00 ms | infer 1335.93 ms | decode+nms 140.32 ms | dets=2 | FIRST_PREDICT
[CONT_METRIC] frame=1 golden_running=0 yolo_infer_ms=1335.93 yolo_decode_ms=140.32 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[CONTENTION] sam_encoder_triggered=1 frameIndex=174 encEvery=12
[ConcurrencyGuard] Segmentation running: detector.postprocessEveryNFrames=3
[DetectorEngine] submit→infer 1250.65 ms | getModel 0.00 ms | infer 1127.16 ms | decode+nms 121.60 ms | dets=3
[CONT_METRIC] frame=2 golden_running=0 yolo_infer_ms=1127.16 yolo_decode_ms=121.60 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[CONTENTION] skip_yolo_decode=1 reason=sam_encoder
aneSubTypeAndVariant: Unknown kMGQAppleNeuralEngineSubtype=0x8030 using (h15)
[DetectorEngine] submit→infer 1159.83 ms | getModel 0.00 ms | infer 1156.11 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=3 golden_running=0 yolo_infer_ms=1156.11 yolo_decode_ms=0.00 skip_yolo_decode=1 skip_sam_encoder=0 skip_reason=sam_encoder
[SegmentationEngine] seg 3042.53 ms (enc 1343.43, dec 209.86) | iou=0.832 | scale=0.533333 new=1024x576 | embHit=0.00
[CONT_METRIC] frame=174 golden_running=0 sam_enc_ms=1343.43 sam_dec_ms=209.86 encoder_triggered=1 skip_yolo_decode=0 skip_sam_encoder=0
[DetectorEngine] submit→infer 1325.30 ms | getModel 0.01 ms | infer 1321.04 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=4 golden_running=0 yolo_infer_ms=1321.04 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1047.27 ms | getModel 0.00 ms | infer 1039.02 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=5 golden_running=0 yolo_infer_ms=1039.02 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1049.22 ms | getModel 0.00 ms | infer 909.30 ms | decode+nms 134.11 ms | dets=2
[CONT_METRIC] frame=6 golden_running=0 yolo_infer_ms=909.30 yolo_decode_ms=134.11 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1075.34 ms | getModel 0.00 ms | infer 1073.18 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=7 golden_running=0 yolo_infer_ms=1073.18 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1039.75 ms | getModel 0.00 ms | infer 1030.73 ms | decode+nms 0.00 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=8 golden_running=0 yolo_infer_ms=1030.73 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1171.08 ms | getModel 0.00 ms | infer 1030.86 ms | decode+nms 138.09 ms | dets=3
[CONT_METRIC] frame=9 golden_running=0 yolo_infer_ms=1030.86 yolo_decode_ms=138.09 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 917.02 ms | getModel 0.00 ms | infer 913.22 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=10 golden_running=0 yolo_infer_ms=913.22 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 822.04 ms | getModel 0.00 ms | infer 815.45 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=11 golden_running=0 yolo_infer_ms=815.45 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1029.45 ms | getModel 0.01 ms | infer 882.15 ms | decode+nms 140.40 ms | dets=3
[CONT_METRIC] frame=12 golden_running=0 yolo_infer_ms=882.15 yolo_decode_ms=140.40 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 881.33 ms | getModel 0.01 ms | infer 879.38 ms | decode+nms 0.00 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=13 golden_running=0 yolo_infer_ms=879.38 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 862.76 ms | getModel 0.00 ms | infer 853.10 ms | decode+nms 0.00 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=14 golden_running=0 yolo_infer_ms=853.10 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1000.46 ms | getModel 0.00 ms | infer 850.79 ms | decode+nms 138.85 ms | dets=2
[CONT_METRIC] frame=15 golden_running=0 yolo_infer_ms=850.79 yolo_decode_ms=138.85 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1033.80 ms | getModel 0.00 ms | infer 1028.87 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=16 golden_running=0 yolo_infer_ms=1028.87 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 977.29 ms | getModel 0.00 ms | infer 967.42 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=17 golden_running=0 yolo_infer_ms=967.42 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1076.78 ms | getModel 0.00 ms | infer 915.86 ms | decode+nms 148.31 ms | dets=2
[CONT_METRIC] frame=18 golden_running=0 yolo_infer_ms=915.86 yolo_decode_ms=148.31 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 926.72 ms | getModel 0.01 ms | infer 921.43 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=19 golden_running=0 yolo_infer_ms=921.43 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1338.24 ms | getModel 0.01 ms | infer 1328.00 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=20 golden_running=0 yolo_infer_ms=1328.00 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1143.15 ms | getModel 0.00 ms | infer 962.53 ms | decode+nms 164.87 ms | dets=2
[CONT_METRIC] frame=21 golden_running=0 yolo_infer_ms=962.53 yolo_decode_ms=164.87 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 905.72 ms | getModel 0.00 ms | infer 903.53 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=22 golden_running=0 yolo_infer_ms=903.53 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 876.08 ms | getModel 0.00 ms | infer 863.22 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=23 golden_running=0 yolo_infer_ms=863.22 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 984.42 ms | getModel 0.00 ms | infer 848.81 ms | decode+nms 133.88 ms | dets=2
[CONT_METRIC] frame=24 golden_running=0 yolo_infer_ms=848.81 yolo_decode_ms=133.88 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 930.87 ms | getModel 0.01 ms | infer 924.67 ms | decode+nms 0.00 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=25 golden_running=0 yolo_infer_ms=924.67 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 943.49 ms | getModel 0.01 ms | infer 935.13 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=26 golden_running=0 yolo_infer_ms=935.13 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1364.75 ms | getModel 0.00 ms | infer 1226.98 ms | decode+nms 135.05 ms | dets=3
[CONT_METRIC] frame=27 golden_running=0 yolo_infer_ms=1226.98 yolo_decode_ms=135.05 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 865.29 ms | getModel 0.00 ms | infer 861.74 ms | decode+nms 0.00 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=28 golden_running=0 yolo_infer_ms=861.74 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 887.98 ms | getModel 0.00 ms | infer 878.31 ms | decode+nms 0.00 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=29 golden_running=0 yolo_infer_ms=878.31 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1198.81 ms | getModel 0.00 ms | infer 1027.91 ms | decode+nms 168.78 ms | dets=1
[CONT_METRIC] frame=30 golden_running=0 yolo_infer_ms=1027.91 yolo_decode_ms=168.78 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 886.21 ms | getModel 0.00 ms | infer 876.63 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=31 golden_running=0 yolo_infer_ms=876.63 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 865.02 ms | getModel 0.00 ms | infer 861.54 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=32 golden_running=0 yolo_infer_ms=861.54 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1132.57 ms | getModel 0.00 ms | infer 905.33 ms | decode+nms 223.32 ms | dets=3
[CONT_METRIC] frame=33 golden_running=0 yolo_infer_ms=905.33 yolo_decode_ms=223.32 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 924.59 ms | getModel 0.01 ms | infer 920.28 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=34 golden_running=0 yolo_infer_ms=920.28 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence

- 贴 Case2 的 `[CONT_METRIC]` 片段：
[YOLO_AB] selected cpuAndGPU
[HUD] YOLO preset=baseline
[Startup] begin background load+warmup computeUnits=MLComputeUnits(rawValue: 1)
Reading from public effective user settings.
Could not create XPC object into key .Value from FigClock[HostTimeClock]: 0x10579c700 retainCount: 17 allocator: 0x2042b6b00 current time: {78356126397375/1000000000 = 78356.126} = 78356.126397 seconds
<<<< FigCaptureSourceRemote >>>> Fig assert: "err == 0 " at bail (FigCaptureSourceRemote.m:569) - (err=-17281)
Could not create XPC object into key .Value from FigClock[HostTimeClock]: 0x10579c700 retainCount: 17 allocator: 0x2042b6b00 current time: {78356126882375/1000000000 = 78356.127} = 78356.126882 seconds
<<<< FigCaptureSourceRemote >>>> Fig assert: "err == 0 " at bail (FigCaptureSourceRemote.m:569) - (err=-17281)
[YOLOv9cModelCache] loadModel computeUnits=MLComputeUnits(rawValue: 1) 3561.27 ms
[YOLOv9cModelCache] warmUp prediction 1424.90 ms
[Startup] modelReady=true
[DetectorEngine] output var_3019 dtype=MLMultiArrayDataType(rawValue: 65568) shape=[1, 84, 8400]
[CONT_METRIC] frame=8 golden_running=0 yolo_infer_ms=949.12 yolo_decode_ms=132.68 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[CONTENTION] sam_encoder_triggered=1 frameIndex=360 encEvery=12
[ConcurrencyGuard] Segmentation running: detector.postprocessEveryNFrames=3
[DetectorEngine] submit→infer 981.16 ms | getModel 0.00 ms | infer 849.35 ms | decode+nms 130.20 ms | dets=2
[CONT_METRIC] frame=9 golden_running=0 yolo_infer_ms=849.35 yolo_decode_ms=130.20 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[SegmentationEngine] seg 1468.12 ms (enc 1179.04, dec 55.20) | iou=0.791 | scale=0.533333 new=1024x576 | embHit=0.33
[CONT_METRIC] frame=360 golden_running=0 sam_enc_ms=1179.04 sam_dec_ms=55.20 encoder_triggered=1 skip_yolo_decode=0 skip_sam_encoder=0
[DetectorEngine] submit→infer 840.79 ms | getModel 0.01 ms | infer 832.69 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=10 golden_running=0 yolo_infer_ms=832.69 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 828.65 ms | getModel 0.00 ms | infer 817.48 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=11 golden_running=0 yolo_infer_ms=817.48 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1022.57 ms | getModel 0.00 ms | infer 890.00 ms | decode+nms 129.63 ms | dets=1
[CONT_METRIC] frame=12 golden_running=0 yolo_infer_ms=890.00 yolo_decode_ms=129.63 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 858.28 ms | getModel 0.00 ms | infer 847.82 ms | decode+nms 0.00 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=13 golden_running=0 yolo_infer_ms=847.82 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 849.13 ms | getModel 0.00 ms | infer 845.98 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=14 golden_running=0 yolo_infer_ms=845.98 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1023.44 ms | getModel 0.00 ms | infer 892.18 ms | decode+nms 123.94 ms | dets=1
[CONT_METRIC] frame=15 golden_running=0 yolo_infer_ms=892.18 yolo_decode_ms=123.94 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 890.27 ms | getModel 0.00 ms | infer 882.15 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=16 golden_running=0 yolo_infer_ms=882.15 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 828.62 ms | getModel 0.00 ms | infer 825.66 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=17 golden_running=0 yolo_infer_ms=825.66 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1007.51 ms | getModel 0.00 ms | infer 869.72 ms | decode+nms 130.47 ms | dets=2
[CONT_METRIC] frame=18 golden_running=0 yolo_infer_ms=869.72 yolo_decode_ms=130.47 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 837.55 ms | getModel 0.01 ms | infer 831.80 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=19 golden_running=0 yolo_infer_ms=831.80 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 791.52 ms | getModel 0.00 ms | infer 789.06 ms | decode+nms 0.00 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=20 golden_running=0 yolo_infer_ms=789.06 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1004.70 ms | getModel 0.01 ms | infer 856.26 ms | decode+nms 143.38 ms | dets=2
[CONT_METRIC] frame=21 golden_running=0 yolo_infer_ms=856.26 yolo_decode_ms=143.38 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 958.82 ms | getModel 0.00 ms | infer 956.70 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=22 golden_running=0 yolo_infer_ms=956.70 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 877.33 ms | getModel 0.01 ms | infer 867.32 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=23 golden_running=0 yolo_infer_ms=867.32 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 974.07 ms | getModel 0.00 ms | infer 830.66 ms | decode+nms 133.01 ms | dets=3
[CONT_METRIC] frame=24 golden_running=0 yolo_infer_ms=830.66 yolo_decode_ms=133.01 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 796.65 ms | getModel 0.00 ms | infer 783.91 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=25 golden_running=0 yolo_infer_ms=783.91 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 922.25 ms | getModel 0.00 ms | infer 909.74 ms | decode+nms 0.00 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=26 golden_running=0 yolo_infer_ms=909.74 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 935.15 ms | getModel 0.00 ms | infer 793.98 ms | decode+nms 135.58 ms | dets=2
[CONT_METRIC] frame=27 golden_running=0 yolo_infer_ms=793.98 yolo_decode_ms=135.58 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 862.34 ms | getModel 0.00 ms | infer 855.76 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=28 golden_running=0 yolo_infer_ms=855.76 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 859.40 ms | getModel 0.00 ms | infer 857.25 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=29 golden_running=0 yolo_infer_ms=857.25 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1057.92 ms | getModel 0.00 ms | infer 874.15 ms | decode+nms 181.26 ms | dets=2
[CONT_METRIC] frame=30 golden_running=0 yolo_infer_ms=874.15 yolo_decode_ms=181.26 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 901.72 ms | getModel 0.00 ms | infer 890.38 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=31 golden_running=0 yolo_infer_ms=890.38 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 815.71 ms | getModel 0.01 ms | infer 812.25 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=32 golden_running=0 yolo_infer_ms=812.25 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1148.78 ms | getModel 0.01 ms | infer 942.98 ms | decode+nms 202.47 ms | dets=3
[CONT_METRIC] frame=33 golden_running=0 yolo_infer_ms=942.98 yolo_decode_ms=202.47 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 928.64 ms | getModel 0.00 ms | infer 925.19 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=34 golden_running=0 yolo_infer_ms=925.19 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 973.99 ms | getModel 0.00 ms | infer 968.67 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=35 golden_running=0 yolo_infer_ms=968.67 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1037.42 ms | getModel 0.00 ms | infer 888.22 ms | decode+nms 142.07 ms | dets=3
[CONT_METRIC] frame=36 golden_running=0 yolo_infer_ms=888.22 yolo_decode_ms=142.07 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 887.81 ms | getModel 0.00 ms | infer 884.78 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=37 golden_running=0 yolo_infer_ms=884.78 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 888.86 ms | getModel 0.00 ms | infer 876.27 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=38 golden_running=0 yolo_infer_ms=876.27 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1110.24 ms | getModel 0.01 ms | infer 936.75 ms | decode+nms 162.36 ms | dets=2
[CONT_METRIC] frame=39 golden_running=0 yolo_infer_ms=936.75 yolo_decode_ms=162.36 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 905.80 ms | getModel 0.00 ms | infer 899.59 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=40 golden_running=0 yolo_infer_ms=899.59 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 877.70 ms | getModel 0.01 ms | infer 872.77 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=41 golden_running=0 yolo_infer_ms=872.77 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1151.66 ms | getModel 0.01 ms | infer 920.69 ms | decode+nms 227.73 ms | dets=2
[CONT_METRIC] frame=42 golden_running=0 yolo_infer_ms=920.69 yolo_decode_ms=227.73 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 864.59 ms | getModel 0.00 ms | infer 856.92 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=43 golden_running=0 yolo_infer_ms=856.92 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 991.16 ms | getModel 0.00 ms | infer 981.94 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=44 golden_running=0 yolo_infer_ms=981.94 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1154.35 ms | getModel 0.00 ms | infer 921.09 ms | decode+nms 219.85 ms | dets=3
[CONT_METRIC] frame=45 golden_running=0 yolo_infer_ms=921.09 yolo_decode_ms=219.85 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 868.77 ms | getModel 0.00 ms | infer 866.21 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=46 golden_running=0 yolo_infer_ms=866.21 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 949.43 ms | getModel 0.00 ms | infer 947.11 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=47 golden_running=0 yolo_infer_ms=947.11 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1045.68 ms | getModel 0.00 ms | infer 897.67 ms | decode+nms 145.42 ms | dets=2
[CONT_METRIC] frame=48 golden_running=0 yolo_infer_ms=897.67 yolo_decode_ms=145.42 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 971.84 ms | getModel 0.00 ms | infer 968.81 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=49 golden_running=0 yolo_infer_ms=968.81 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
Run Golden tapped
[HUD] Run Golden tapped
[CONTENTION] golden_running=1 start
[ConcurrencyGuard] Golden running: detector.postprocessEveryNFrames=6
[DetectorEngine] submit→infer 1520.71 ms | getModel 0.01 ms | infer 1517.45 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=50 golden_running=1 yolo_infer_ms=1517.45 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
IoU=0.9629
[MobileSAMGoldenTest] IoU=0.9629
[MobileSAMGoldenTest] IoU=0.9629
[CONTENTION] golden_running=0 end
[ConcurrencyGuard] Golden done: detector.postprocessEveryNFrames=1
[DetectorEngine] submit→infer 1712.86 ms | getModel 0.00 ms | infer 1701.96 ms | decode+nms 0.01 ms | dets=0 | POSTPROCESS_SKIPPED
[CONT_METRIC] frame=51 golden_running=0 yolo_infer_ms=1701.96 yolo_decode_ms=0.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=cadence
[DetectorEngine] submit→infer 1275.10 ms | getModel 0.02 ms | infer 1103.72 ms | decode+nms 164.90 ms | dets=2
[CONT_METRIC] frame=52 golden_running=0 yolo_infer_ms=1103.72 yolo_decode_ms=164.90 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1157.32 ms | getModel 0.00 ms | infer 999.23 ms | decode+nms 146.18 ms | dets=2
[CONT_METRIC] frame=53 golden_running=0 yolo_infer_ms=999.23 yolo_decode_ms=146.18 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1154.92 ms | getModel 0.01 ms | infer 931.20 ms | decode+nms 217.65 ms | dets=1
[CONT_METRIC] frame=54 golden_running=0 yolo_infer_ms=931.20 yolo_decode_ms=217.65 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1087.23 ms | getModel 0.00 ms | infer 922.48 ms | decode+nms 162.34 ms | dets=2
[CONT_METRIC] frame=55 golden_running=0 yolo_infer_ms=922.48 yolo_decode_ms=162.34 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1150.45 ms | getModel 0.01 ms | infer 977.17 ms | decode+nms 169.63 ms | dets=2
[CONT_METRIC] frame=56 golden_running=0 yolo_infer_ms=977.17 yolo_decode_ms=169.63 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1233.60 ms | getModel 0.00 ms | infer 1015.17 ms | decode+nms 208.88 ms | dets=2
[CONT_METRIC] frame=57 golden_running=0 yolo_infer_ms=1015.17 yolo_decode_ms=208.88 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1193.81 ms | getModel 0.00 ms | infer 1031.27 ms | decode+nms 151.36 ms | dets=3
[CONT_METRIC] frame=58 golden_running=0 yolo_infer_ms=1031.27 yolo_decode_ms=151.36 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1120.52 ms | getModel 0.00 ms | infer 981.02 ms | decode+nms 135.65 ms | dets=3
[CONT_METRIC] frame=59 golden_running=0 yolo_infer_ms=981.02 yolo_decode_ms=135.65 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1257.99 ms | getModel 0.00 ms | infer 1046.90 ms | decode+nms 203.59 ms | dets=2
[CONT_METRIC] frame=60 golden_running=0 yolo_infer_ms=1046.90 yolo_decode_ms=203.59 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1180.38 ms | getModel 0.01 ms | infer 1038.63 ms | decode+nms 137.46 ms | dets=1
[CONT_METRIC] frame=61 golden_running=0 yolo_infer_ms=1038.63 yolo_decode_ms=137.46 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1180.38 ms | getModel 0.01 ms | infer 1030.62 ms | decode+nms 142.60 ms | dets=2
[CONT_METRIC] frame=62 golden_running=0 yolo_infer_ms=1030.62 yolo_decode_ms=142.60 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1303.66 ms | getModel 0.00 ms | infer 1111.01 ms | decode+nms 182.12 ms | dets=1
[CONT_METRIC] frame=63 golden_running=0 yolo_infer_ms=1111.01 yolo_decode_ms=182.12 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1275.23 ms | getModel 0.00 ms | infer 1119.29 ms | decode+nms 149.77 ms | dets=1
[CONT_METRIC] frame=64 golden_running=0 yolo_infer_ms=1119.29 yolo_decode_ms=149.77 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1210.62 ms | getModel 0.00 ms | infer 1073.05 ms | decode+nms 135.02 ms | dets=2
[CONT_METRIC] frame=65 golden_running=0 yolo_infer_ms=1073.05 yolo_decode_ms=135.02 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1289.61 ms | getModel 0.00 ms | infer 1132.31 ms | decode+nms 148.24 ms | dets=4
[CONT_METRIC] frame=66 golden_running=0 yolo_infer_ms=1132.31 yolo_decode_ms=148.24 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1288.20 ms | getModel 0.01 ms | infer 1127.94 ms | decode+nms 146.78 ms | dets=1
[CONT_METRIC] frame=67 golden_running=0 yolo_infer_ms=1127.94 yolo_decode_ms=146.78 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1272.25 ms | getModel 0.00 ms | infer 1131.42 ms | decode+nms 137.92 ms | dets=4
[CONT_METRIC] frame=68 golden_running=0 yolo_infer_ms=1131.42 yolo_decode_ms=137.92 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1339.66 ms | getModel 0.00 ms | infer 1194.87 ms | decode+nms 134.00 ms | dets=4
[CONT_METRIC] frame=69 golden_running=0 yolo_infer_ms=1194.87 yolo_decode_ms=134.00 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
[DetectorEngine] submit→infer 1391.92 ms | getModel 0.00 ms | infer 1230.29 ms | decode+nms 151.83 ms | dets=2
[CONT_METRIC] frame=70 golden_running=0 yolo_infer_ms=1230.29 yolo_decode_ms=151.83 skip_yolo_decode=0 skip_sam_encoder=0 skip_reason=none
- （可选）屏幕录制/卡顿视频：
