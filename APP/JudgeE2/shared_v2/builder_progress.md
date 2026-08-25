# Builder Progress — JudgeE2

## Day 1 — Clean iOS Foundation

**Status:** Complete.

### Completed
- Xcode project located at `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/JudgeE2`.
- iOS deployment target **17.0** in `JudgeE2.xcodeproj`.
- Bundle Identifier: `js.JudgeE2`.
- Signing Team: `W95LVGJ7G3` (Automatic).
- Simulator + iPhone 11 both launch the default app (white screen).

## Day 2 — CoreML Model Load (Builder)

**Status:** Complete.

### Completed
- Added `yolov9-c.mlmodel` to app target (copied to `JudgeE2/JudgeE2/Models/`).
- Confirmed auto-generated model class: `yolov9_c` (via CoreML generate).
- Added minimal load test (`ModelLoader.testLoad()`), called in `JudgeE2App.init()`.
- Prints **"Model loaded successfully"** on app start.

## Day 3 — Single Image Inference (Builder)

**Status:** Complete.

### Completed
- Created dummy 640×640 BGRA pixel buffer input.
- Ran a single inference via `yolov9_c.prediction(image:)`.
- Printed output tensor shapes for `var_3019` and `var_3022`.
- Measured and logged single inference time (ms).

## Day 4 — Camera Pipeline (Builder)

**Status:** Complete.

### Completed
- Added AVFoundation camera preview (`CameraPreview` + `PreviewView`).
- Captured frame buffers via `AVCaptureVideoDataOutput` delegate.
- Implemented 640×640 letterbox conversion with CoreImage.
- Fed frames into `yolov9_c.prediction(image:)`.
- Logged per-frame inference time (ms).

### Verification (2026-02-27)
- Confirmed `CameraManager.swift` handles session config, frame capture, letterbox, and per-frame inference logging.
- Confirmed `ContentView` uses `CameraPreview` with start/stop lifecycle.

### Notes
- Added camera usage description in build settings: `INFOPLIST_KEY_NSCameraUsageDescription`.

---

## Day 5 — Decode + NMS (Builder)

**Status:** Complete.

### Completed
- Implemented decode for YOLOv9-c output tensor `(1, 84, 8400)`.
- Added confidence filtering (sigmoid + threshold 0.25).
- Added class-aware NMS (IoU 0.45).
- Prints detection count per frame in the inference log.

### Code Notes
- Implemented in `CameraManager.swift`:
  - `decodeDetections(from:confidenceThreshold:)`
  - `classAwareNMS(_:iouThreshold:)`
  - `iou(_:_:), sigmoid(_:)`
- Uses `var_3019` output for decode.

---

## Day 6 — Bounding Box Overlay (Builder)

**Status:** Complete.

### Completed
- Mapped detection coordinates to preview layer via metadata-normalized rects.
- Handled aspectFill scaling using `layerRectConverted(fromMetadataOutputRect:)`.
- Handled device orientation and kept captured image horizontal by adjusting `videoRotationAngle` based on device rotation.
- Drew bounding boxes in real time with `CAShapeLayer` overlay.

### Code Notes
- `CameraManager.swift`: orientation observer + `updateRotation()` + rotation-aware `mapToMetadataRect`.
- `CameraPreview.swift`: overlay layer path update from normalized rects.

---

## Phase 2 — Day 2 (Builder)

**Status:** Complete (N/A).

### Notes
- Phase 2 Day 2 计划中未分配 Builder 任务；无需改动工程或代码。
- 已阅读 `shared/model_plan.md` 与 `shared/tasks.md` 以确认模型准备信息。

---

## Phase 2 — Day 3 (Builder)

**Status:** Complete.

### Completed
- 添加 `SAMEncoder`（MobileSAM_ImageEncoder）加载与预处理，支持 1024×1024 ResizeLongestSide + 归一化（CHW）。
- 在 `CameraManager` 中加入独立 `segmentationQueue`，确保 encoder 异步运行（不阻塞检测管线）。
- 实现 encoder 低频触发（默认每 12 帧），并缓存 embedding（TTL 1200ms）。
- 记录并打印 encoder latency 与缓存命中率。

### Files
- `JudgeE2/Segmentation/SAMEncoder..swift`
- `JudgeE2/Detection/CameraManager.swift`

---

## Phase 2 — Day 4 (Builder)

**Status:** Complete.

### Completed
- 实现 `PromptBuilder`：将检测框（原始像素坐标）按与 Encoder 一致的 ResizeLongestSide(1024)+居中 padding 映射为 SAM box prompt。
- 新增 `SAMDecoder`：加载 MobileSAM decoder，执行 `decode(embedding, prompt)` 输出 `low_res_masks`。
- 在 `CameraManager` 中加入 decoder 调度：默认每 6 帧触发；实现 mask TTL（800ms）与过期刷新逻辑。
- 记录解码耗时与 mask 刷新间隔/频率日志。

### Files
- `JudgeE2/Segmentation/PromptBuilder.swift`
- `JudgeE2/Segmentation/SAMDecoder.swift`
- `JudgeE2/Detection/CameraManager.swift`

---

## Phase 2 — Day 5 (Builder)

**Status:** Complete.

### Completed
- 添加 mask overlay layer（`PreviewView` 中新增 `maskLayer`），并在 `CameraPreview` 更新时同步绘制。
- 使用 `videoPreviewLayer.layerRectConverted(fromMetadataOutputRect:)` 复用 Phase 1 的映射链路，将 mask 叠加到 preview。
- 在 `MaskRenderer` 中进行 alpha blending + 颜色映射（青色，alpha 约 0.45），输出可视化 mask。
- 复用 Phase 1 旋转/镜像逻辑：通过 `maskRotationAngle` + `maskMirrored` 应用到 `maskLayer`。

### Files
- `JudgeE2/Segmentation/MaskRenderer.swift`
- `JudgeE2/Detection/CameraPreview.swift`
- `JudgeE2/Detection/CameraManager.swift`
- `JudgeE2/UI/ContentView.swift`

---

## Phase 2 — Day 6 (Builder)

**Status:** Complete.

### Completed
- 增加 **Temporal Manager** 行为：主目标选择（Top-1 + hysteresis），在 primary 丢失 3 帧后切换。
- 实现漂移检测（IoU/中心漂移/面积/比例/score drop）触发重新分割。
- 补齐 TTL + 失效触发：几何变化/primary 切换/漂移等触发 embedding + mask 失效并优先刷新。
- 实现 **primary 优先刷新**（force decode/encoder bypass cadence）。

### Files
- `JudgeE2/Detection/CameraManager.swift`

---

## Phase 2 — Day 7 收尾 (Builder)

**Status:** Complete.

### 已完成三项

#### 1. SAM Encoder/Decoder 滑窗统计（mean + p95 聚合）
- 为 `CameraManager` 新增六个属性：
  - `samEncoderTimesMs: [Double]` / `samEncoderStatsWindow = 100` / `samEncoderCallCount`
  - `samDecoderTimesMs: [Double]` / `samDecoderStatsWindow = 100` / `samDecoderCallCount`
- **剔除冷启动样本**：首次 encoder 和 decoder 调用（ANE 编译延迟）标注为 `cold start — excluded from stats`，不计入滑窗。
- 滑窗满 100 样本时自动打印：`SAM encoder stats (n=100): mean=... ms | p95=... ms`（decoder 同理）。
- 属性只在各自的串行对列（`encoderQueue` / `decoderQueue`）内访问，无锁开销。
- 仿照 YOLO `inferenceTimesMs` 滑窗模式实现，风格一致。

#### 2. 修复 §2.1 跨队列数据竞争
- **问题**：`warmupSegmentationIfPossible()` 在 `sessionQueue` 读 `latestCameraBuffer`，其闭包在 `encoderQueue` 读 `self.lastLetterbox`，两者均无锁且趪越队列边界。
- **修复方案**：将 `warmupSegmentationIfPossible` 函数体封装到 `videoQueue.async` 内，在 `videoQueue`（这两个字段的唯一写入队列）上取 `cameraBuffer` 快照和 `capturedLetterbox` 快照，再传入 `encoderQueue.async` 闭包，部局变量才读。
- 无需新增锁（利用队列串行性保证快照原子性），`captureOutput` 成程路径无额外开销。

#### 3. 代勾 tasks.md Day 7 五项（Debugger 无权勾选）
- 依据 `debug_report.md` 附录 A 真机实测数据将以下五项全部勾选：
  - Encoder latency → mean 857 ms / p95 933 ms
  - Decoder latency → mean 61 ms / p95 69 ms
  - Mask refresh rate → ~1.5 Hz
  - FPS → 2.7–2.9 FPS
  - Memory → 244–320 MB，峰値 339 MB

### Phase 2 冻结确认（2026-07-19）

Architect 裁决：§5 全部 6 项追认“**以代码实测値更新契约文档**”（option a），0 项回退。

Builder 核验确认：代码与决议一致，无需任何修改。

| 参数 | 决议冻结値 | 代码状态 |
|------|--------------|--------|
| Encoder cadence | 12 帧 | `encoderEveryNFrames = 12` ✅ |
| Decoder cadence | 2 帧 | `decoderEveryNFrames = 2` ✅ |
| Embedding TTL | 8000 ms | `embeddingTTLms = 8000` ✅ |
| Mask TTL | 2000 ms | `maskTTLms = 2000` ✅ |
| Class 切换滞后 | 6 帧 | `classHysteresisThreshold = 6` ✅ |
| Heavy drift | IoU < 0.10 | 双级阈値已实现 ✅ |

**Phase 2 正式决结。Phase 3 优化备选项（按 Architect 优先级）：**
1. 🔴 ANE 64-byte tensor 对齐告警消解（预期 Encoder 延迟 857→600 ms）
2. 🟡 Encoder 分辨率 1024→512（需 ML_Vision 评估精度损失）
3. 🟢 Embedding cache hit 率优化（目标 >80%）

- **§2.3 CoreML 对齐告警**：需模型侧重导出，待 Phase 3 跨入 ML_Vision 处理。

### 文件
- `JudgeE2/Detection/CameraManager.swift`
- `shared/tasks.md`
- `shared/builder_progress.md`

### 编译验证
- `xcodebuild clean build`（iphonesimulator，iPhone 11）：**BUILD SUCCEEDED**，零代码告警。
