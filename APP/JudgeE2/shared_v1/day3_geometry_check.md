# Day3 Geometry Check — 5-point Sanity + AspectFill/Letterbox 回投

日期：2026-02-18
设备：iPhone 11
目标：满足 Day4 Gate：证明坐标/几何契约在真机上闭环正确（至少不翻转/不偏移/不镜像错误）。

契约参考：
- `shared/architect_output.md`（A.3 preview aspectFill 映射；A.4 letterbox 变换）

---

## 0. 当前实现状态（代码已具备可视化工具）
已在 overlay 中加入 Debug 几何可视化（黄色点 + HUD 文本），用于快速排雷：
- 四角 + 中心（5 点）黄点
- 中心十字
- HUD 显示：`r / padX / padY / camW / camH / inputW / inputH / contentW / contentH`

相关文件：
- `swift_app/JudgeEverythingApp/Sources/BBoxOverlay.swift`（debugGeometry=true）
- `swift_app/JudgeEverythingApp/Sources/DetectorEngine.swift`（发布 lastFrameGeometry / lastLetterbox）
- `swift_app/JudgeEverythingApp/Sources/ContentView.swift`（开启 debugGeometry）

---

## 1. 5 点测试（证据已补齐 1 张截图；仍建议补 1 张对齐场景图）

### 1.1 测试方法
1) 真机运行 app，允许相机权限。
2) 保持手机竖屏（portrait）。
3) 观察 overlay 的 5 个黄点（TL/TR/BL/BR/C）位置：
   - TL 应靠近画面左上
   - TR 应靠近画面右上
   - BL 应靠近画面左下
   - BR 应靠近画面右下
   - C（中心）应在屏幕中心附近
4) 缓慢移动手机：黄点应固定在屏幕几何位置（不随场景漂移）。

### 1.2 证据（已提供截图）
- 截图路径：`shared/screenshot.PNG`
- 截图可见：
  - 中心黄点 + 十字（标注“C”）
  - HUD 参数：
    - `r=0.3333`
    - `padX=0.0`
    - `padY=140.0`
    - `cam=1920x1080`
    - `input=640x640`
    - `content=640.0x360.0`

**初步结论（基于该截图）**：
- `r/padX/padY` 数值与 1920×1080 → 640×640 的 letterbox 预期一致：
  - r = 640/1920 = 0.3333
  - contentH = 1080*r = 360
  - padY = (640-360)/2 = 140
- 中心点在屏幕中心附近，未见明显整体翻转/错位迹象。

### 1.3 仍建议补充的证据（用于更强的翻转/偏移判断）
- 截图 2：含明显水平/垂直参考线的场景（例如门框/窗框），观察 bbox 与物体对齐，辅助判断是否上下翻转。
- 可选：录屏 5–10s，展示移动/轻微旋转时 overlay 不抖动/不翻转。

---

## 2. aspectFill 映射一致性检查（A.3）

### 检查点
- 预览使用 `AVCaptureVideoPreviewLayer.videoGravity = .resizeAspectFill`
- overlay 映射应同样使用 aspectFill 的 scale + offset（ox/oy）公式

### 当前实现
- `BBoxOverlay.swift` 中使用：
  - `s = max(Wv/Wc, Hv/Hc)`
  - `ox = (Wv - Wc*s)/2, oy = (Hv - Hc*s)/2`
  - `camToView(p) = (ox + p.x*s, oy + p.y*s)`

### 仍需确认
- 若后摄/前摄切换或系统版本行为差异导致镜像，需要把 `FrameGeometry.isMirrored` 贯通到 overlay。

---

## 3. letterbox 回投一致性检查（A.4）

### 检查点
- `LetterboxTransform.make()`：r/padX/padY
- `modelToCamera()` 逆变换：`(x-padX)/r, (y-padY)/r`

### 当前实现
- 预处理：`PixelBufferLetterbox.makeInput()` 使用 CIImage scale + translation + composited(over:black)
- decode：输出框在 model-input px（640） → `letterbox.modelToCamera(rectModel)` 回投到 canonical camera px

### 风险点
- CIImage 坐标系/`oriented()` 叠加可能造成 y 翻转或旋转偏差。

---

## 4. 当前结论（截至 2026-02-18）
- 已获取 1 张截图证据：`shared/screenshot.PNG`，HUD 数值与 letterbox 预期一致（见 §1.2）。
- 相机预览稳定（用户确认）。
- 仍建议补 1 张“几何强对齐场景”截图/短录屏，以更强地排除 y 翻转/镜像/裁剪方向错误。

---

## 5. 下一步（最小动作清单）
1) 补 1 张带明显水平/垂直线的场景截图（或 5–10s 录屏）
2) 在本文件补一句结论：bbox 与物体是否整体上下翻转/左右镜像/偏移
3) 如发现翻转：记录对应文件与疑点（`PixelBufferLetterbox.swift`/`CameraPreview.swift` orientation 设置）
