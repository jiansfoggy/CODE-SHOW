# JudgeEverything — 任务计划与状态

## Metadata
Project: real-time video instance segmentation iOS app
Target: iPhone 11
Detection: YOLO-v9
Segmentation: MobileSAM
Model weights path:
/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/models

## Day 1
- [x] Architect: produce high-level architecture spec
- [x] ML_Vision: plan model export & performance strategy
- [x] Builder: set up repo skeleton and env
- [x] Debugger: environment validation + static scan (`shared/debug_report.md`)

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
- [x] **创建最小可编译 Xcode 工程**（`JudgeEverything/swift_app/`）
  - iOS deployment target：与 ML_Vision 对齐（建议 iOS 17）
  - 目标：能在 simulator 跑起、能真机编译（先不要求签名自动化）
- [x] **相机预览跑通**：AVFoundation capture + previewLayer（先不接模型）
- [x] **CoreML 接入骨架**
  - 将 YOLOv9 CoreML artifact 加入 app bundle（推荐使用编译后的 `.mlmodelc` 或 Xcode 自动编译流程）
  - 预留 inference pipeline：load → prepare input → predict → print outputs

### Debugger — Day 2 tasks
- [x] **xcodebuild 编译检查**：iOS Simulator 编译通过（修复 target 平台设置、CoreML codegen 依赖问题；见 `shared/debug_report.md`）
- [x] **CoreML load+infer smoke test（Simulator）**
  - 稳定抓取日志：output `var_3019`/`var_3022` shape=`(1,84,8400)` + load/predict 耗时（见 `shared/debug_report.md`）
- [x] **记录与归档问题**：Simulator 相机不可用/不稳定（`AVFoundationErrorDomain -11800` / `OSStatus -12782`），详见 `shared/debug_report.md` Issue `D2-ENV-002`

## Day 3 (Priority: 真机打通 camera → preprocess(letterbox+geometry) → CoreML infer → decode/NMS → bbox overlay)

### Builder — Day 3 tasks
- [x] **真机相机预览验证（iPhone 11）**：确认 capture session + previewLayer 正常（Simulator 相机不作为依据）
- [x] **camera frame → 推理输入接线（v0）**
  - 引入/复用 `FrameGeometry` + `LetterboxTransform(r,px,py,Wi,Hi,Wc,Hc)` 元数据（对齐 `shared/architect_output.md`）
  - 完成：`CVPixelBuffer(camera)` → letterbox → 640×640 input（对齐 `shared/model_plan.md`）
- [x] **CoreML 推理 loop（v0）**：后台队列运行；打印 output keys/shape
- [x] **bbox-only overlay v0**：decode+NMS 后画框（不做分割），确保坐标映射闭环（canonical camera px ↔ preview）
- [x] **CoreML 模型缓存**：`MLModel(contentsOf:)` 单例化，避免重复 load（降低启动/调试开销）

### ML_Vision — Day 3 tasks
- [x] **提供最小 Python reference decoder（必须）**：raw head `(1,84,8400)` → boxes/classes + score threshold + NMS
  - 交付：`shared/yolov9_reference_decoder.py`
  - 明确：bbox=xywh(center)→xyxy；无 objectness；cls 已 sigmoid；84=4+80(COCO)
  - 默认参数：scoreThreshold=0.25 / iouThreshold=0.65 / topK=100
- [x] **确认 output 使用策略**：`var_3019` vs `var_3022` 哪个为主（或二者差异/用途），写入 `shared/model_plan.md`
  - 结论：`var_3019` 与 PyTorch 输出一致（mean abs diff≈0.0066），作为主输出；`var_3022` 暂不用于 decode

### Debugger — Day 3 tasks
- [x] **iPhone 11 真实性能复测（bench v0）**
  - 记录：model load time（含“缓存前后对比”）+ per-inference time
  - computeUnits：`.all`；必要时对比 `.cpuAndGPU`
  - 输出：`shared/day3_device_bench.md`
- [x] **几何映射 sanity check（5 点测试）**：orientation / aspectFill / 映射与契约一致性验证；输出 `shared/day3_geometry_check.md`
- [x] **性能风险排查**：主线程阻塞点、丢帧/backpressure；给出可执行修改建议

### Architect — Day 3 tasks
- [x] **review FrameGeometry/letterbox/overlay**：确认与 `shared/architect_output.md` 一致，必要时补充边界条件与默认阈值建议

## Day 4 (Priority: 先收敛 Day3 真机验收与性能基线 → 再接入 MobileSAM v0)

### Gate (Day4 开始前必须满足)
- [x] **补齐 Day3 未交付文档**（否则 MobileSAM 接入将放大几何/性能不确定性）
  - [x] Debugger：`shared/day3_device_bench.md`（含 `.all` vs `.cpuAndGPU` AB、load/warmup/steady 分段）
  - [x] Debugger：`shared/day3_geometry_check.md`（5 点测试 + aspectFill/letterbox 回投一致性）
  - Builder：在 iPhone 11 上确认 camera preview 稳定（并在 Day3 Builder task 打勾）

### Builder — Day 4 tasks (MobileSAM integration v0)
- [x] **集成 MobileSAM（bbox prompt, v0）**：仅实现“单实例/低频”跑通（先正确后快）
  - 输入：同一帧（或最近帧）图像 + bbox（来自 YOLO Canonical camera px）
  - 输出：mask overlay（可先低分辨率/粗糙）
- [x] **Segmentation 调度策略 v0**：确保不阻塞相机与 detector
  - 只对 top-1 检测目标做 segmentation
  - segmentation 每 N 帧触发一次（N 可调），其余帧复用缓存 mask
- [x] **可视化 Debug HUD**：在画面上显示当前 pipeline 状态
  - modelReady / infer(ms) / decode(ms) / seg(ms)
  - 当前 `r,px,py,Wc,Hc`（便于快速定位几何问题）

### ML_Vision — Day 4 tasks (MobileSAM export + perf guidance)
- [x] **给出 MobileSAM CoreML 产物策略（A/B 方案落地选择）**
  - Option A：encoder/decoder split（推荐，复用 embedding；建议 2 个 `.mlpackage`/mlprogram）
  - Option B：monolithic（fallback）
  - iOS 版本建议：iOS 17（不建议低于 iOS 15）；computeUnits：bring-up 用 `.cpuAndGPU`，稳定后再试 `.all`
- [x] **提供 MobileSAM I/O spec（Swift 可接入）**
  - 输入：encoder `image`(1024×1024, SAM normalize) → `image_embeddings`(1,256,64,64)
  - decoder 输入：`image_embeddings` + `point_coords/labels`（box 用 2 点 label=2/3）+ optional mask_input
  - 输出：建议固定 `low_res_masks`(1,1,256,256) logits + `iou_predictions`，阈值 `logits>0`
- [x] **提供 segmentation golden case**
  - `shared/golden/bus.jpg` + `shared/golden/mobilesam_bus_mask.png` + `shared/golden/mobilesam_bus_case.json`

### Debugger — Day 4 tasks (Segmentation validation + perf triage)
- [x] **segmentation correctness 验证 v0**
  - 检查 mask 是否与 bbox/物体位置一致（不反转、不偏移）
  - 输出：`shared/day4_segmentation_check.md`
- [x] **端到端性能基线 v0（iPhone 11）**
  - detector-only 与 detector+segmenter 的对比（FPS/延迟/掉帧）
  - 输出：`shared/day4_device_perf.md`
- [x] **风险排查：启动加载时间（~9s）与 decode+nms（~115ms）优化建议**
  - 给出可执行调参建议（threshold/topK/preNmsTopK 等）
  - 如有必要：建议导出 `.mlpackage/mlprogram` 路线以降低 cold start

## Day 5 (Priority: latency/UI stability + perf knobs A/B + segmentation cadence)

### Builder — Day 5 tasks
- [x] **Overlay/Render 稳定化 v1**
  - 目标：mask overlay 与 bbox overlay 在 `.resizeAspectFill` 下稳定；无闪烁/拉伸错误
  - 若当前 mask 为 camera-res `CGImage`：确认渲染路径不会在主线程做大规模缩放/像素拷贝
- [x] **性能调度 v1：embedding cache + encoder cadence**
  - 目标：MobileSAM split 走“encoder 低频、decoder 相对高频”的策略
  - 建议：encoder 每 N 帧（N>=12）更新一次 embedding；decoder 每 N 帧或每次 top-1 更新
  - 输出：在 HUD/日志中打印 encoder/decoder 触发频率与命中率（cache hit）
- [x] **YOLO decode/NMS 参数 A/B（以降低 ~115ms decode+nms）**
  - 先做可回滚的参数调节：`scoreThreshold`、`preNmsTopK`、`topK`
  - 记录：decode+nms(ms) 与 dets 数量分布变化

### Debugger — Day 5 tasks
- [x] **UI thread & latency review（Instruments/日志）**
  - 检查主线程是否有 CI/CG/MLMultiArray 大拷贝、同步渲染阻塞
  - 输出：`shared/day5_ui_latency_review.md`（主线程热点、建议修改点、可复现步骤）
- [x] **端到端 latency breakdown v1**
  - 细分统计：preprocess / YOLO infer / decode+nms / SAM preprocess / encoder / decoder / mask render
  - 输出：`shared/day5_latency_breakdown.md`（含建议的采样窗口与平均/分位数）
- [ ] **风险复测：cold start（~9.5s）与并发干扰**
  - 复测：YOLO load + warmUp 时序（是否阻塞首帧/首屏）
  - 观察：Run Golden / segmentation 并发是否导致 decode 峰值飙升

### ML_Vision — Day 5 tasks
- [x] **量化/格式路线建议 v1（以 cold start 与 ANE 路径为主）**
  - YOLO：评估/建议 `.mlpackage/mlprogram`（或其它导出策略）以降低 cold start
  - MobileSAM：确认 encoder/decoder 在 iPhone 11 上的 computeUnits 建议（`.cpuAndGPU` vs `.all`）
  - 输出：`shared/day5_quantization_and_format.md`

### Architect — Day 5 tasks
- [x] **Pipeline 调度与缓存策略 review v1**
  - 复核：top-1 policy、encoder cadence、mask 缓存/过期策略是否满足交互体验
  - 给出默认参数建议（N、阈值、回退策略）并写入 `shared/` 文档（文件名由 Architect 自定）

## Day 6
- [ ] Builder: optimize and finalize
- [ ] Debugger: performance verification report
- [ ] ML_Vision: model adjustment if needed

## Day 7
- [ ] Debugger: final regression tests
- [ ] Architect: final architecture review
- [ ] Project wrap-up
