# Day 6 — Orientation & Geometry Regression (D6-D-GEOMETRY-REGRESSION)

日期：2026-02-19
设备：iPhone 11 (A13) / iOS 17
目标：做 **5 点测试 + 前/后摄像头 + 旋转（portrait/landscape）** 的回归验证，确保：
- preview（`.resizeAspectFill`）与 overlay（bbox/mask/5点）映射一致
- 不出现整体翻转（上下/左右）
- 不出现系统性偏移（尤其是旋转/镜像切换时）

输出要求：包含结论 + 复现步骤 + 截图清单。

---

## 0) 当前实现现状（会影响回归范围）

### 0.1 现状：EXIF orientation 被硬编码为 `.right`
- 代码：`swift_app/JudgeEverythingApp/Sources/CameraPreview.swift`
  - `let exifOrientation: CGImagePropertyOrientation = .right`
  - `connection.videoRotationAngle = 90 // portrait`

**影响**：目前 app 的 detector + segmenter 在“逻辑上”始终按 portrait/`.right` 去解释图像；
- portrait 运行一般 OK
- landscape（横屏）回归在现状下**很可能失败/不具备意义**（因为 orientation 并未动态更新）

### 0.2 现状：只配置了后置摄像头 `.back`
- 代码：`CameraController.configureSessionIfNeeded()`
  - `AVCaptureDevice.default(... position: .back)`

**影响**：目前无法在 app 内直接切前摄，因此“前/后摄像头回归”需要先加一个切换开关（Builder 侧改动）。

### 0.3 已具备：Day3 的 5 点/几何 HUD 工具
- 现有报告：`shared/day3_geometry_check.md`
- 现有证据截图：`shared/screenshot.PNG`

---

## 1) 回归矩阵（要覆盖哪些组合）

> 目标覆盖：camera position × device orientation

| Case | Camera | Device Orientation | Expected |
|---|---|---|---|
| C1 | Back | Portrait | ✅ 5 点不翻转；bbox/mask 与物体一致 |
| C2 | Back | LandscapeLeft | ✅ 同上（overlay 与 preview 一致） |
| C3 | Back | LandscapeRight | ✅ 同上 |
| C4 | Front | Portrait | ✅ 镜像策略一致（不出现“mask/bbox 与 preview 镜像相反”） |
| C5 | Front | LandscapeLeft | ✅ 同上 |
| C6 | Front | LandscapeRight | ✅ 同上 |

**注意**：在当前“exifOrientation 硬编码 + 仅后摄”的实现下，只有 C1 是已可验证/有意义的。

---

## 2) 5 点测试（Five-point sanity）— 复现步骤

### 2.1 测试前准备
1) iPhone 11 打开“旋转锁定”=关闭（允许旋转）
2) 打开 app，确保 overlay 的 debugGeometry（黄点）开启（见 `day3_geometry_check.md`）
3) 预览必须是 `.resizeAspectFill`（当前是）

### 2.2 每个 Case 的通用步骤
对矩阵中的每个 Case（C1~C6），重复：
1) 选择摄像头（Back/Front）
2) 把手机旋转到目标方向（Portrait/LandscapeLeft/LandscapeRight）并停住 1s（等待 pipeline 稳定）
3) 截 1 张“静态证据截图”（见 §4 截图清单）
4) 录 5–10s 短视频（可选，但强烈建议用于检测“突然翻转/跳变”）

### 2.3 Pass/Fail 判据（视觉几何判据）
- 5 点（TL/TR/BL/BR/C）在屏幕几何上应保持：
  - TL 靠近左上，TR 靠近右上，BL 靠近左下，BR 靠近右下，C 在屏幕中心附近
- bbox overlay 与物体位置一致，不出现：
  - 上下颠倒（y 翻转）
  - 左右镜像（前摄最常见）
  - 持续的固定偏移（例如整体向左上偏几十像素）
- mask overlay（如果开启）应与 bbox/物体一致（允许边缘粗糙，但不能整体错位）

---

## 3) 已有证据 & 当前结论（截至 2026-02-24）

### 3.1 已有证据（Back + Portrait）
- 参考：`shared/day3_geometry_check.md` §1.2
- 截图：`shared/screenshot.PNG`

**结论（C1）**：
- HUD 的 letterbox 参数与理论一致（1920×1080 → 640×640 的 `r=0.3333, padY=140`）。
- 5 点与中心十字无明显整体翻转/错位。

### 3.2 新增截图（C1–C6）
- 路径：`/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/几何回归截图（C1–C6）/C1.PNG` ~ `C6.PNG`
- 人工判定结果（James 提供）：
  - C1：Pass（无偏移/无镜像）
  - C2–C6：画面未旋转，前置摄像头未打开（因此无法覆盖横屏/前摄回归）；分割效果良好

### 3.3 未覆盖/当前无法完成的回归项
- C2/C3（Back + 横屏）：当前 `exifOrientation` 固定 `.right`，横屏未生效
- C4~C6（Front 摄像头相关）：前摄未开启/未贯通镜像标记

**当前总体结论**：
- ✅ Back+Portrait（C1）通过（已有 Day3 证据）
- ⚠️ 横屏/前摄：**需要实现层面补齐后才能回归**（见 §5）。

---

## 4) 截图清单（必须产出哪些截图，如何命名）

已收到截图：
- C1: `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/几何回归截图（C1–C6）/C1.PNG`
- C2: `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/几何回归截图（C1–C6）/C2.PNG`
- C3: `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/几何回归截图（C1–C6）/C3.PNG`
- C4: `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/几何回归截图（C1–C6）/C4.PNG`
- C5: `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/几何回归截图（C1–C6）/C5.PNG`
- C6: `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/几何回归截图（C1–C6）/C6.PNG`

建议统一命名：
`shared/geom_reg_C<case>_<camera>_<orientation>_<timestamp>.png`

每个 Case 至少 1 张截图，截图必须包含：
- 预览画面
- 5 点黄点 + 中心十字
- HUD（至少包含 camW/camH、r/pad、以及 SAM transform 若开了分割）

建议额外截图（可选但推荐）：
- 含明显水平/垂直参考线的场景（门框/窗框），用于判断是否翻转。

---

## 5) 需要 Builder 补齐的最小功能（否则回归无法完成）

### 5.1 旋转/EXIF orientation 动态更新（必要）
当前：`exifOrientation = .right`（硬编码）。

建议：
- 以 `AVCaptureConnection.videoRotationAngle` / `UIDevice.current.orientation` / `AVCaptureVideoOrientation` 为单一真相，动态计算 `CGImagePropertyOrientation`。
- 让 detector + segmenter + preview 使用同一份 orientation（FrameGeometry 里记录）。

### 5.2 前摄切换 + 镜像贯通（必要）
- 在 session 配置中支持 `.front` 设备
- 记录并贯通 `isMirrored`（前摄常见）
- overlay 映射需要基于 `isMirrored` 做 x 轴镜像（或在输入阶段完成镜像，保证 canonical 一致）

### 5.3 建议加一个 Debug HUD 开关
- `Camera: back/front`
- `Orientation: portrait/landscapeLeft/landscapeRight`
- 显示当前 computed `exifOrientation` rawValue

---

## 6) 风险点（最容易出错的位置）

1) **previewLayer 与 videoDataOutput 的旋转/镜像不一致**
   - preview 看起来是对的，但送入模型的帧方向不同 → overlay 必然错。

2) **CIImage.oriented(exif) 与 CI 坐标系（y 轴）叠加**
   - 旋转后 extent/坐标方向变换，容易在 letterbox / mask 回投中引入翻转。

3) **前摄镜像**
   - preview 通常是镜像显示，但模型输入未镜像（或反之）→ bbox/mask 左右相反。

---

## 7) 下一步建议（执行顺序）

1) Builder：先实现“前摄切换 + 动态 orientation + isMirrored 贯通”三个最小改动
2) Debugger：按 §1 回归矩阵逐项跑，并把截图/录屏路径补到本文 §4
3) 若发现 Fail：在本文新增 Issue 小节，记录：
   - Case ID（C?）
   - 复现步骤
   - 期望 vs 实际
   - 可能责任点文件（CameraPreview.swift / PixelBufferLetterbox.swift / BBoxOverlay.swift / SegmentationEngine.swift）

---

## Appendix A — 关联文档
- 坐标/几何契约：`shared/architect_output.md`（A.3/A.4）
- Day3 5 点 sanity：`shared/day3_geometry_check.md`
- 当前相机/EXIF：`swift_app/JudgeEverythingApp/Sources/CameraPreview.swift`
