# JudgeE2 — 任务计划与状态

## Metadata
Project: real-time video instance segmentation iOS app
Target: iPhone 11
Detection: YOLO-v9
Segmentation: MobileSAM
Model weights path:
/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models

## Day 1
- [x] Architect: produce high-level architecture spec
- [x] ML_Vision: plan model export & performance strategy
- [x] Builder: set up repo skeleton and env
- [ ] Debugger: environment validation + static scan (`shared/debug_report.md`)

## Day 2 (Priority: get a compiling iOS app + first CoreML artifact + load/infer smoke test)

### Architect — Day 2 tasks
- [x] **坐标系/几何契约 v1**：统一并写入 `shared/` 文档
  - view(tap) ↔ previewLayer ↔ camera frame(px) ↔ model input(letterbox) ↔ output decode
  - 定义 origin、axis、单位、rotation、aspectFill/fit、letterbox padding 的计算方式
- [x] **I/O 契约表**：为 YOLOv9 detection + MobileSAM（后续）定义 Swift 侧期望的 tensor 名称/shape/dtype
- [x] **风险清单更新**：对齐“先跑通再优化”的里程碑（编译、相机、加载、单帧推理、overlay）

### ML_Vision — Day 2 tasks
- [x] **产出 YOLOv9 CoreML draft**（可先不做 ANE 最优）
  - 导出路径：PyTorch → (ONNX as reference) → CoreML（mlmodel）
  - 当前产物：`models/yolov9-c.mlmodel`（coremltools 转换成功）
  - compute unit 建议：先 `.all`；若有兼容问题再试 `.cpuAndGPU`
- [x] **给出明确 I/O spec**（交付给 Builder/Debugger）
  - input：`image`，RGB 640×640，内置 scale=1/255；app 侧建议 letterbox 到 640×640
  - output：`var_3019` / `var_3022`，均为 float32 `(1,84,8400)`；含义见 `shared/model_plan.md`
- [x] **提供 1 张 golden test image + 期望输出格式**（用于 load test / decode sanity check）
  - `shared/golden/bus.jpg` + `shared/golden/expected_bus_yolov9c_raw.json`

### Builder — Day 2 tasks
- [x] **创建最小可编译 Xcode 工程**（`JudgeE2/swift_app/`）
  - iOS deployment target：与 ML_Vision 对齐（建议 iOS 17）
  - App icon 用这个路径下的图片 models/icon.webp
  - 目标：能在 simulator 跑起、能真机编译（先不要求签名自动化）
- [x] **相机预览跑通**：AVFoundation capture + previewLayer（先不接模型）
- [x] **CoreML 接入骨架**
  - 将 YOLOv9 CoreML artifact 加入 app bundle（推荐使用编译后的 `.mlmodelc` 或 Xcode 自动编译流程）
  - 预留 inference pipeline：load → prepare input → predict → print outputs

### Debugger — Day 2 tasks
- [ ] **xcodebuild 编译检查**：iOS Simulator 编译通过（修复 target 平台设置、CoreML codegen 依赖问题；见 `shared/debug_report.md`）
- [ ] **CoreML load+infer smoke test（Simulator）**
  - 稳定抓取日志：output `var_3019`/`var_3022` shape=`(1,84,8400)` + load/predict 耗时（见 `shared/debug_report.md`）
- [ ] **记录与归档问题**：Simulator 相机不可用/不稳定（`AVFoundationErrorDomain -11800` / `OSStatus -12782`），详见 `shared/debug_report.md` Issue `D2-ENV-002`

## Day 3 (Priority: 真机打通 camera → preprocess(letterbox+geometry) → CoreML infer → decode/NMS → bbox overlay)

### Builder — Day 3 tasks
- [ ] **真机相机预览验证（iPhone 11）**：确认 capture session + previewLayer 正常（Simulator 相机不作为依据）
- [ ] **camera frame → 推理输入接线（v0）**
  - 引入/复用 `FrameGeometry` + `LetterboxTransform(r,px,py,Wi,Hi,Wc,Hc)` 元数据（对齐 `shared/architect_output.md`）
  - 完成：`CVPixelBuffer(camera)` → letterbox → 640×640 input（对齐 `shared/model_plan.md`）
- [ ] **CoreML 推理 loop（v0）**：后台队列运行；打印 output keys/shape
- [ ] **bbox-only overlay v0**：decode+NMS 后画框（不做分割），确保坐标映射闭环（canonical camera px ↔ preview）
- [ ] **CoreML 模型缓存**：`MLModel(contentsOf:)` 单例化，避免重复 load（降低启动/调试开销）

### ML_Vision — Day 3 tasks
- [ ] **提供最小 Python reference decoder（必须）**：raw head `(1,84,8400)` → boxes/classes + score threshold + NMS
  - 交付：`shared/yolov9_reference_decoder.py`
  - 明确：bbox=xywh(center)→xyxy；无 objectness；cls 已 sigmoid；84=4+80(COCO)
  - 默认参数：scoreThreshold=0.25 / iouThreshold=0.65 / topK=100
- [ ] **确认 output 使用策略**：`var_3019` vs `var_3022` 哪个为主（或二者差异/用途），写入 `shared/model_plan.md`
  - 结论：`var_3019` 与 PyTorch 输出一致（mean abs diff≈0.0066），作为主输出；`var_3022` 暂不用于 decode

### Debugger — Day 3 tasks
- [ ] **iPhone 11 真实性能复测（bench v0）**
  - 记录：model load time（含“缓存前后对比”）+ per-inference time
  - computeUnits：`.all`；必要时对比 `.cpuAndGPU`
  - 输出：`shared/day3_device_bench.md`
- [ ] **几何映射 sanity check（5 点测试）**：orientation / aspectFill / 映射与契约一致性验证；输出 `shared/day3_geometry_check.md`
- [ ] **性能风险排查**：主线程阻塞点、丢帧/backpressure；给出可执行修改建议

### Architect — Day 3 tasks
- [ ] **review FrameGeometry/letterbox/overlay**：确认与 `shared/architect_output.md` 一致，必要时补充边界条件与默认阈值建议

## Day 4 (Priority: 先收敛 Day3 真机验收与性能基线 → 再接入 MobileSAM v0)

### Gate (Day4 开始前必须满足)
- [ ] **补齐 Day3 未交付文档**（否则 MobileSAM 接入将放大几何/性能不确定性）
  - [ ] Debugger：`shared/day3_device_bench.md`（含 `.all` vs `.cpuAndGPU` AB、load/warmup/steady 分段）
  - [ ] Debugger：`shared/day3_geometry_check.md`（5 点测试 + aspectFill/letterbox 回投一致性）
  - Builder：在 iPhone 11 上确认 camera preview 稳定（并在 Day3 Builder task 打勾）

### Builder — Day 4 tasks (MobileSAM integration v0)
- [ ] **集成 MobileSAM（bbox prompt, v0）**：仅实现“单实例/低频”跑通（先正确后快）
  - 输入：同一帧（或最近帧）图像 + bbox（来自 YOLO Canonical camera px）
  - 输出：mask overlay（可先低分辨率/粗糙）
- [ ] **Segmentation 调度策略 v0**：确保不阻塞相机与 detector
  - 只对 top-1 检测目标做 segmentation
  - segmentation 每 N 帧触发一次（N 可调），其余帧复用缓存 mask
- [ ] **可视化 Debug HUD**：在画面上显示当前 pipeline 状态
  - modelReady / infer(ms) / decode(ms) / seg(ms)
  - 当前 `r,px,py,Wc,Hc`（便于快速定位几何问题）

### ML_Vision — Day 4 tasks (MobileSAM export + perf guidance)
- [ ] **给出 MobileSAM CoreML 产物策略（A/B 方案落地选择）**
  - Option A：encoder/decoder split（推荐，复用 embedding；建议 2 个 `.mlpackage`/mlprogram）
  - Option B：monolithic（fallback）
  - iOS 版本建议：iOS 17（不建议低于 iOS 15）；computeUnits：bring-up 用 `.cpuAndGPU`，稳定后再试 `.all`
- [ ] **提供 MobileSAM I/O spec（Swift 可接入）**
  - 输入：encoder `image`(1024×1024, SAM normalize) → `image_embeddings`(1,256,64,64)
  - decoder 输入：`image_embeddings` + `point_coords/labels`（box 用 2 点 label=2/3）+ optional mask_input
  - 输出：建议固定 `low_res_masks`(1,1,256,256) logits + `iou_predictions`，阈值 `logits>0`
- [ ] **提供 segmentation golden case**
  - `shared/golden/bus.jpg` + `shared/golden/mobilesam_bus_mask.png` + `shared/golden/mobilesam_bus_case.json`

### Debugger — Day 4 tasks (Segmentation validation + perf triage)
- [ ] **segmentation correctness 验证 v0**
  - 检查 mask 是否与 bbox/物体位置一致（不反转、不偏移）
  - 输出：`shared/day4_segmentation_check.md`
- [ ] **端到端性能基线 v0（iPhone 11）**
  - detector-only 与 detector+segmenter 的对比（FPS/延迟/掉帧）
  - 输出：`shared/day4_device_perf.md`
- [ ] **风险排查：启动加载时间（~9s）与 decode+nms（~115ms）优化建议**
  - 给出可执行调参建议（threshold/topK/preNmsTopK 等）
  - 如有必要：建议导出 `.mlpackage/mlprogram` 路线以降低 cold start

## Day 5 (Priority: latency/UI stability + perf knobs A/B + segmentation cadence)

### Builder — Day 5 tasks
- [ ] **Overlay/Render 稳定化 v1**
  - 目标：mask overlay 与 bbox overlay 在 `.resizeAspectFill` 下稳定；无闪烁/拉伸错误
  - 若当前 mask 为 camera-res `CGImage`：确认渲染路径不会在主线程做大规模缩放/像素拷贝
- [ ] **性能调度 v1：embedding cache + encoder cadence**
  - 目标：MobileSAM split 走“encoder 低频、decoder 相对高频”的策略
  - 建议：encoder 每 N 帧（N>=12）更新一次 embedding；decoder 每 N 帧或每次 top-1 更新
  - 输出：在 HUD/日志中打印 encoder/decoder 触发频率与命中率（cache hit）
- [ ] **YOLO decode/NMS 参数 A/B（以降低 ~115ms decode+nms）**
  - 先做可回滚的参数调节：`scoreThreshold`、`preNmsTopK`、`topK`
  - 记录：decode+nms(ms) 与 dets 数量分布变化

### Debugger — Day 5 tasks
- [ ] **UI thread & latency review（Instruments/日志）**
  - 检查主线程是否有 CI/CG/MLMultiArray 大拷贝、同步渲染阻塞
  - 输出：`shared/day5_ui_latency_review.md`（主线程热点、建议修改点、可复现步骤）
- [ ] **端到端 latency breakdown v1**
  - 细分统计：preprocess / YOLO infer / decode+nms / SAM preprocess / encoder / decoder / mask render
  - 输出：`shared/day5_latency_breakdown.md`（含建议的采样窗口与平均/分位数）
- [ ] **风险复测：cold start（~9.5s）与并发干扰**
  - 复测：YOLO load + warmUp 时序（是否阻塞首帧/首屏）
  - 观察：Run Golden / segmentation 并发是否导致 decode 峰值飙升

### ML_Vision — Day 5 tasks
- [ ] **量化/格式路线建议 v1（以 cold start 与 ANE 路径为主）**
  - YOLO：评估/建议 `.mlpackage/mlprogram`（或其它导出策略）以降低 cold start
  - MobileSAM：确认 encoder/decoder 在 iPhone 11 上的 computeUnits 建议（`.cpuAndGPU` vs `.all`）
  - 输出：`shared/day5_quantization_and_format.md`

### Architect — Day 5 tasks
- [ ] **Pipeline 调度与缓存策略 review v1**
  - 复核：top-1 policy、encoder cadence、mask 缓存/过期策略是否满足交互体验
  - 给出默认参数建议（N、阈值、回退策略）并写入 `shared/` 文档（文件名由 Architect 自定）

## Day 6 (Priority: contention & cold-start止血，补齐几何/orientation 契约)
To-do 顺序：**Builder → Debugger → ML_Vision → Architect**

### Builder — Day 6 tasks
- [ ] **D6-B-CONCURRENCY-GUARD**：巩固 `postprocessEveryNFrames`/skip_yolo_decode 在 SAM encoder & Run Golden 期间的生效逻辑；提交代码，并在 `shared/day6_contention_mitigation_results.md` 填写 §4.3 结论 + 剩余 TODO。
- [ ] **D6-B-DECODE-AB**：在 HUD 运行 baseline/fast 两档（各 60 样本）并填满 `shared/day6_decode_ab_table.md`（mean/p95 + detHist），确认 decode+nms 目标 <60–80ms 稳态、无 >300ms 尾部。
- [ ] **D6-B-ORIENTATION-V2**：实现前/后摄切换 + 动态 orientation/`isMirrored` 贯通（FrameGeometry 一处统一）；更新 HUD 显示当前 orientation/exif；完成后配合 Debugger 做回归（引用 `shared/day6_orientation_geometry_regression.md`）。
- [ ] **D6-B-MLPROGRAM-TOGGLE**：在 app 中加入 YOLO artifact 选择（`.mlmodel` vs `.mlpackage`），保证与 ML_Vision 交付的 mlprogram I/O 契约对齐，可切换 computeUnits；记录使用方法于 `shared/builder_progress.md`。

### Debugger — Day 6 tasks
- [ ] **D6-D-COLDSTART-AB**：在 iPhone 11 上跑 `.cpuAndGPU` vs `.all` 的 cold start + steady AB（含 mlmodel vs mlprogram），填入 `shared/day6_yolo_cold_start_ab.md` + `shared/day6_yolo_cold_start_profile.md`（注明 iOS/Xcode/是否连线）。
- [ ] **D6-D-LATENCY-WITH-CONTENTION**：按 `shared/day6_contention_mitigation_results.md` Case1/Case2 抽取 `[CONT_METRIC]`，统计 yolo_decode_ms / sam_enc_ms 的 max/p95，并补全文档 §4.1/4.2；同步更新 `shared/day6_latency_breakdown_with_contention.md`（表格化各阶段）。
- [ ] **D6-D-GEOMETRY-REGRESSION**：在 Builder 提供 orientation/front-camera 支持后，完成回归矩阵 C1–C6（见 `shared/day6_orientation_geometry_regression.md`），上传截图路径并判定 Pass/Fail。

### ML_Vision — Day 6 tasks
- [ ] **D6-M-EXPORT-MLPROGRAM**：交付 YOLOv9c mlprogram（路径/commit hash）+ 最小集成说明；在 `shared/day6_yolo_mlprogram_card.md` 标明 minimum iOS、computeUnits 建议、与 baseline 数值对比（如误差/性能预期）。
- [ ] **D6-M-SAM-COMPUTE-RECO**：提供 MobileSAM encoder/decoder computeUnits 与 cadence 推荐（结合 iPhone11 实测），补充/确认 `shared/day6_sam_computeunits_reco.md` 与 `shared/day6_sam_encoder_profile.md` 结论。
- [ ] **D6-M-INMODEL-POSTPROC-OPTION**：评估 YOLO in-model postproc 方案的可行性与风险，结论写入 `shared/day6_inmodel_postproc_option.md`（是否推进、对 decode+nms 的预期收益）。

### Architect — Day 6 tasks
- [ ] **D6-A-ORIENTATION-CONTRACT-V2**：在 `shared/architect_output.md` 增补“动态 orientation/front-camera 镜像契约 + 渲染/模型输入对齐规则”（对应 Builder 的实现边界条件），完成后勾选。
- [ ] **D6-A-SCHEDULING-GUIDE**：结合 Day6 调度/skip 策略，整理默认参数（postprocessEveryNFrames、encoderEveryNFrames、decoderEveryNFrames、topK/scoreThr）与回退策略，写入 `shared/architect_output.md`。

## Day 7 (Priority: 收尾回归 + 冷启动/解码收敛 + 最终架构确认)
To-do 顺序：**Builder → Debugger → ML_Vision → Architect**

### Builder — Day 7 tasks
- [ ] **D7-B-ORIENTATION+FRONT-CAM 完整贯通**：完成前/后摄切换 + 动态 orientation/`isMirrored` 贯通（FrameGeometry 单一来源）；HUD 显示当前 orientation/exif/mirror；确保 preview/detector/segmenter 使用同一几何语义。
- [ ] **D7-B-DECODE-AB 落地**：在真机跑 baseline/fast 预设（各 60 样本），回填 `shared/day6_decode_ab_table.md`（mean/p95 + detHist）。
- [ ] **D7-B-MLPROGRAM 运行开关验收**：确认 `.mlmodel` vs `.mlpackage` 可切换 + computeUnits 可切换；记录操作步骤于 `shared/builder_progress.md`。

### Debugger — Day 7 tasks
- [ ] **D7-D-GEOMETRY-REGRESSION C1–C6**：在 Builder 完成前摄/动态 orientation 后执行全矩阵回归，补齐截图路径与 Pass/Fail（`shared/day6_orientation_geometry_regression.md`）。
- [ ] **D7-D-COLDSTART-AB+PROFILE**：完成 YOLO cold start A/B（含 mlmodel vs mlprogram）；补齐 `shared/day6_yolo_cold_start_ab.md` 与 `shared/day6_yolo_cold_start_profile.md`（注明 iOS/Xcode/是否连线）。
- [ ] **D7-D-CONTENTION-METRICS 收尾**：整理 `[CONT_METRIC]` 统计，补齐 `shared/day6_contention_mitigation_results.md` §4.1/4.2 + `shared/day6_latency_breakdown_with_contention.md`。

### ML_Vision — Day 7 tasks
- [ ] **D7-M-MLPROGRAM 兼容性/性能建议收敛**：基于 Day7 AB 结果，给出默认 computeUnits/格式建议（是否推荐 mlprogram 默认启用），更新 `shared/day6_yolo_mlprogram_card.md`。
- [ ] **D7-M-SAM FP16 路线决策**：基于现有 encoder profile 与 IoU 观测，明确是否进入 P1（compute_precision=FP16, I/O 不变），写结论到 `shared/day6_sam_computeunits_reco.md`。

### Architect — Day 7 tasks
- [ ] **D7-A-FINAL-ARCH-REVIEW**：最终架构复核（cold start 策略 + contention 调度 + orientation/mirror 契约），在 `shared/architect_output.md` 增补“Final Review”小节与默认参数汇总。
- [ ] **D7-A-PROJECT-WRAP-UP**：整理 Day1–Day7 关键决策与未完成项清单，输出 `shared/day7_wrapup.md`（仅摘要，不写代码）。
