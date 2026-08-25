# Debugger Report — JudgeE2 (Rebuild)

> **恢复说明**：原文件被覆盖，以下为基于会话记录的“重建版”，包含 Day1–Day4 关键问题与修复建议，并追加 2026-02-26 的平台修复验证。
> 若后续找到备份/旧版本，可再做对比合并。

---

## Day 1 — 环境与静态扫描（2026-02-25）

### Issue D1-BUILD-001 — 无 Xcode 工程，无法编译验证
- **描述**：`JudgeE2/swift_app/` 当时无 `.xcodeproj`。
- **路径**：`JudgeE2/swift_app/`
- **严重度**：Medium
- **建议**：Builder 创建最小可编译工程（iOS 17）。

### Issue D1-ENV-001 — xcodebuild 指向 CommandLineTools
- **描述**：`xcode-select -p` 指向 `/Library/Developer/CommandLineTools`。
- **严重度**：Medium
- **建议**：
  - 直接用 Xcode 绝对路径执行 `xcodebuild`；或
  - `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`。

### Issue D1-PERF-001 — Python 导出环境缺少版本锁定
- **路径**：`JudgeE2/python/export/ENV_SETUP.sh`
- **严重度**：Medium
- **建议**：添加 requirements / 版本固定，避免环境漂移。

---

## Day 2 — 编译与 CoreML bring-up（2026-02-26）

### Issue D2-ENV-001 — `Supported platforms ... empty`
- **描述**：Simulator 编译时提示 `Supported platforms for the buildables in the current scheme is empty.`
- **路径**：`JudgeE2/swift_app/JudgeE2/JudgeE2.xcodeproj`
- **严重度**：High
- **建议**：设置
  - `SUPPORTED_PLATFORMS = iphoneos iphonesimulator`
  - `SDKROOT = iphoneos`
  - `SUPPORTS_MACCATALYST = NO`
  - 确保 Scheme 绑定 iOS target。

### Issue D2-RUN-001 — CoreML load+infer smoke test 未完成（编译阻塞）
- **描述**：因编译失败未能跑 Smoke Test。
- **严重度**：High
- **建议**：修复编译后执行并记录输出 shape / 耗时。

### Issue D2-PERF-001 — 无法获取 load/predict 基线
- **描述**：编译阻塞导致无法采集。
- **严重度**：Medium
- **建议**：修复后补采样。

---

## Day 2 — JudgeEverything 参考问题（2026-02-17，历史参考）

### Issue D2-BUILD-001 — Target 缺少 iOS 平台声明
- **严重度**：Critical
- **修复建议**：同上（SDKROOT / SUPPORTED_PLATFORMS / TARGETED_DEVICE_FAMILY）。

### Issue D2-BUILD-002 — CoreML codegen 类不可用
- **描述**：`YOLOv9c` codegen 在 CLI 编译时不可用。
- **建议**：使用 `MLModel(contentsOf:)` + `MLDictionaryFeatureProvider` 访问输出。

### Issue D2-RUN-000 — simctl install 缺少 CFBundleExecutable
- **建议**：`CFBundleExecutable = $(EXECUTABLE_NAME)`。

### Issue D2-ENV-002 — Simulator 相机不稳定（-11800/-12782）
- **结论**：Simulator 仅用于编译/推理；相机要用真机。

---

## Day 3 — 几何/解码/性能（2026-02-17~18）

### Issue D3-BUILD-001 — 新 Swift 文件未加入 Target
- **严重度**：Critical
- **建议**：加入 Compile Sources / Target Membership。

### Issue D3-BUILD-002 — codegen 依赖回归风险
- **严重度**：High
- **建议**：继续使用 MLModel + FeatureProvider 路径。

### Issue D3-RUN-001 — CIImage 坐标系/翻转风险
- **严重度**：High
- **建议**：按几何契约做 5 点测试；确认 y 轴方向与 canonical 一致。

### Issue D3-RUN-003 — YOLO 解码语义不一致
- **严重度**：High
- **建议**：以 `shared/model_plan.md` / `yolov9_reference_decoder.py` 为准：
  - `var_3019` 为主输出
  - class 已 sigmoid，无 objectness
  - bbox 为 xywh(center) → xyxy。

### Issue D3-PERF-001 — 冷启动 load ~9s
- **严重度**：High
- **建议**：AB 测 `.all` vs `.cpuAndGPU`；考虑 `.mlpackage/mlprogram`。

### Issue D3-PERF-002 — decode+nms ~115ms
- **严重度**：Medium
- **建议**：提高阈值、降低 preNmsTopK、优化 NMS。

---

## Day 4 — Segmentation 相关（2026-02-18）

### Issue D4-RUN-001 — SegmentationEngine 仍为 placeholder（当时）
- **严重度**：High
- **建议**：接入 MobileSAM CoreML artifacts 后替换。

### Issue D4-RUN-002 — Orientation/mirror 未统一
- **严重度**：High
- **建议**：以 FrameGeometry 为单一来源贯通。

### Issue D4-PERF-001 — decode+nms CPU 瓶颈
- **严重度**：High
- **建议**：阈值调参 / 向量化 / NMS 优化。

---

# Day 2 Debugger Update — Supported platforms empty fix verified (2026-02-26)

## 0. 结论
- ✅ `Supported platforms ... is empty` 问题已不再复现；Simulator 编译 **BUILD SUCCEEDED**。

## 1. 复现/验证命令
```bash
"/Applications/1 Workflow/Xcode.app/Contents/Developer/usr/bin/xcodebuild" \
  -project /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/swift_app/JudgeE2/JudgeE2.xcodeproj \
  -scheme JudgeE2 \
  -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -sdk iphonesimulator \
  clean build
```

## 2. 编译日志证据（节选）
```
CreateBuildOperation
ComputeTargetDependencyGraph
note: Building targets in dependency order
note: Target dependency graph (1 target)
    Target 'JudgeE2' in project 'JudgeE2' (no dependencies)
...
CodeSign .../Build/Products/Debug-iphonesimulator/JudgeE2.app
Validate .../Build/Products/Debug-iphonesimulator/JudgeE2.app
Touch .../Build/Products/Debug-iphonesimulator/JudgeE2.app
** BUILD SUCCEEDED **
```

## 3. Build Settings 证据（Supported Platforms 非空）
```
SUPPORTED_PLATFORMS = iphoneos iphonesimulator
SUPPORTS_MACCATALYST = NO
```

## 4. 备注
- 之前使用 `iPhone 15` destination 不在当前 Xcode 26.1 simulator 列表中；本次改为存在的 `iPhone 17` 成功编译。
