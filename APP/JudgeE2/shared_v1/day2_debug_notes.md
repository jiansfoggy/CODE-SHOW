# Day 2 Debug Notes (Debugger)

日期：2026-02-17

## 1) xcodebuild 编译检查

### 1.1 修复：目标平台被识别为 macOS（导致无法选择 iOS Simulator destination）
- **现象**：
  - `xcodebuild ... -destination 'platform=iOS Simulator,name=iPhone 17'` 报错：
    - `Supported platforms for the buildables in the current scheme is empty.`
    - 只列出 `{ platform:macOS ... }` destinations
- **根因**：Target-level build settings 缺少/未显式声明 iOS 平台相关设置，导致 xcodebuild 解析 scheme 的平台能力异常（表现为默认落到 macOS destination）。
- **修复**：在 `swift_app/JudgeEverythingApp.xcodeproj/project.pbxproj` 的 Target Debug/Release buildSettings 增加：
  - `SDKROOT = iphoneos;`
  - `SUPPORTED_PLATFORMS = "iphoneos iphonesimulator";`
  - `SUPPORTS_MACCATALYST = NO;`
  - `TARGETED_DEVICE_FAMILY = 1;`

### 1.2 修复：编译期错误（YOLOv9c codegen + withCheckedContinuation 推断失败）
- **现象**（初次 iOS Simulator build）：
  - `cannot find 'YOLOv9c' in scope`
  - `generic parameter 'T' could not be inferred`（`withCheckedContinuation`）
- **根因**：
  - CoreML 的 Swift codegen 类可能未生成/不可用（不同 Xcode 设置/版本差异、或资源阶段不匹配）。
  - `withCheckedContinuation` 未显式指定返回类型，Swift 版本/严格度下无法推断。
- **修复**：
  - 将 smoke test 改为**不依赖 codegen**，改用 `MLModel(contentsOf:configuration:)` + `MLDictionaryFeatureProvider` + feature name `image`。
  - 将 continuation 改为 `CheckedContinuation<Void, Never>` 并 `resume(returning: ())`。

### 1.3 编译结果
- 使用 Xcode 路径：`/Applications/1 Workflow/Xcode.app/.../usr/bin/xcodebuild`
- 命令：
  - `xcodebuild -project swift_app/JudgeEverythingApp.xcodeproj -scheme JudgeEverythingApp -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 17' -sdk iphonesimulator clean build`
- **结果**：`BUILD SUCCEEDED`

---

## 2) Runtime / Simulator 运行检查

### 2.1 修复：simctl install 失败（Info.plist 缺少 CFBundleExecutable）
- **现象**：`simctl install` 报错 `missing or invalid CFBundleExecutable in its Info.plist`
- **修复**：在 `swift_app/JudgeEverythingApp/Resources/Info.plist` 增加：
  - `CFBundleExecutable = $(EXECUTABLE_NAME)`
- **结果**：修复后可安装并 launch。

### 2.2 运行时错误：iOS Simulator 相机不可用导致 AVCaptureSession runtime error
- **现象**：Simulator log 显示：
  - `AVFoundationErrorDomain Code=-11800`，`NSOSStatusErrorDomain Code=-12782`
  - `FigCaptureSessionSimulator signalled err=-12782`
- **结论**：这是 **Simulator 侧 camera pipeline 的预期限制/环境问题**，不代表真机必现。
- **建议**：
  - Day2 真机验证相机预览（iPhone 11）必须做；Simulator 只做编译与 CoreML 推理 smoke test。

### 2.3 CoreML load+infer smoke test（Simulator）
- 当前通过 `ContentView.task` 触发 `GoldenYOLOSmokeTest.runIfPossible()`。
- 由于 `simctl launch --stdout/--stderr` 未捕获到输出，且 log 中未检索到 `GoldenYOLO` 字样，本次未能在 Simulator 端确认 smoke test 的打印输出。
- **下一步建议**：
  1) 在 `GoldenYOLOSmokeTest` 内补充 `os_log`（Logger）输出，确保进入 unified logging。
  2) 或在 Xcode Run 控制台确认 `[GoldenYOLO]` 相关输出。
  3) 真机上记录：model init 时间 + prediction 时间 + output keys/shape。

---

## 3) 性能瓶颈（Day2 观测点）
- smoke test 在首屏自动触发：模型加载 + 推理会增加启动抖动（建议 Day3 后改为 Debug-only 或按钮触发）。

---

## 4) Golden I/O 期望（来自 shared/golden/expected_bus_yolov9c_raw.json）
- outputs：`var_3019` 与 `var_3022`
- dtype：float32
- shape：(1, 84, 8400)
