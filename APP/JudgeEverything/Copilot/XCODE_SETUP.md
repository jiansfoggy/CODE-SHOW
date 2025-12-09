# Xcode 项目设置和打包指南

## 项目创建步骤

### 1. 在 Xcode 中创建新项目

1. 打开 Xcode → File → New → Project
2. 选择 iOS → App
3. 填写项目信息：
   - Product Name: `JudgeEverything`
   - Organization Identifier: `com.yourcompany`（自定义）
   - Interface: SwiftUI 或 Storyboard（推荐 SwiftUI）
   - Language: Swift

### 2. 添加 Core ML 模型文件

1. 转换成功后（在 Mac 上运行转换脚本），你会得到两个文件：
   - `yolov9-c.mlmodel`
   - `mobile_sam.mlmodel`

2. 在 Xcode 中：
   - 右键点击 Project Navigator 中的项目
   - 选择 "Add Files to [ProjectName]..."
   - 选择 `yolov9-c.mlmodel` 和 `mobile_sam.mlmodel`
   - 确保勾选 "Copy items if needed"
   - 确保目标为你的 App target

3. Xcode 会自动生成 Swift 模型包装类（这些会出现在 Derived Data 中）

### 3. 配置相机权限

编辑 `Info.plist`，添加以下键值对：

```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access to perform real-time video segmentation.</string>
```

或在 Xcode UI 中：
1. 选择 Project → Target → Info
2. 找到 "Privacy - Camera Usage Description"
3. 输入权限提示文本

### 4. 添加 Swift 代码文件

将以下文件添加到你的 Xcode 项目（可从本仓库复制）：

- `AppDelegate.swift` — 应用启动配置
- `CameraViewController.swift` — 相机捕捉和分割逻辑
- `SegmentationEngine.swift` — Core ML 模型加载和推理

也可选择性保留（远程服务模式）：
- `NetworkClient.swift` — 远程推理服务（如果你选择不使用 Core ML）

### 5. 模型输入/输出处理

**重要**：Xcode 生成的模型包装类可能与示例代码中的类名不同。

转换后，检查 Xcode 生成的模型类：
1. Build 项目（⌘B）
2. 右键点击 `.mlmodel` 文件 → Open in Finder
3. Xcode 会在 Derived Data 中生成 Swift 模型类

常见的自动生成类名：
- 对于 `yolov9-c.mlmodel` → `YOLOv9C` 或 `yolov9c`
- 对于 `mobile_sam.mlmodel` → `MobileSAM` 或 `mobilesam`

需要在 `SegmentationEngine.swift` 和 `CameraViewController.swift` 中更新类名和初始化代码以匹配实际生成的类。

### 6. 处理模型输出格式

YOLOv9 和 MobileSAM 的输出格式取决于 Core ML 转换时的配置。通常：

**YOLOv9 输出示例**：
- `output` 或 `output0`: shape [1, num_detections, 85]（x, y, w, h, conf, class_probs）

**MobileSAM 输出示例**：
- `masks` 或 `output`: shape [1, H, W]（分割掩码）

在 `SegmentationEngine.swift` 中调整输出解析代码以匹配实际输出。

## App Store 打包步骤

### 1. 配置签名和证书

1. 选择 Project → Signing & Capabilities
2. 选择你的 Team（需要 Apple Developer 账户）
3. 确保 Bundle Identifier 唯一

### 2. 设置 App 图标和启动屏幕

1. Assets.xcassets 中添加 App Icon Set（1024×1024 及其他尺寸）
2. 添加 Launch Screen（可用 Storyboard 或 SwiftUI）

### 3. 配置版本号和构建号

1. Project → Target → General
2. 设置 Version (e.g. 1.0)
3. 设置 Build (e.g. 1)

### 4. 创建 Archive 并上传

```bash
# 使用 Xcode 命令行工具
xcodebuild -workspace YourProject.xcworkspace -scheme YourScheme -configuration Release archive -archivePath path/to/archive.xcarchive

# 或使用 Xcode UI：Product → Archive
```

### 5. 使用 Transporter 上传到 App Store

1. 从 Archive 导出为 App Store IPA
2. 使用 Apple Transporter app 登录 App Store Connect
3. 选择刚导出的 IPA 并上传

### 6. App Store 审核注意事项

- **模型文件大小**：确保 App Bundle 不超过 4GB。YOLOv9 + MobileSAM 的 `.mlmodel` 文件可能较大（合计可能 100-300MB），需确保在 App Store 最大下载大小限制内。
- **App Thinning**：Xcode 会自动优化，仅包含特定设备需要的资源。
- **隐私**：确保声明了相机权限，以及任何数据处理。
- **性能**：在真实设备上测试，确保实时分割性能可接受（iOS 15+ 设备通常性能较好）。

## 测试清单

在提交 App Store 之前：

- [ ] 在真实 iPhone / iPad 上测试相机捕捉
- [ ] 验证点击触发分割（检查检测结果）
- [ ] 验证掩码生成和叠加显示
- [ ] 测试多次点击和快速交互
- [ ] 检查内存使用（避免模型加载导致 OOM）
- [ ] 测试不同亮度和场景下的识别准确度
- [ ] 验证 App 在锁屏、后台切换时的行为

## 常见问题

### Q: 模型输出为何为空或格式错误？
A: 检查 Core ML 转换时的输入/输出配置。可能需要调整 `coremltools.convert()` 参数或手动指定输入/输出名称。

### Q: App 加载时崩溃，提示模型文件缺失？
A: 确保 `.mlmodel` 文件在 Target Membership 中被正确添加到你的 App target。

### Q: 实时分割速度太慢？
A: 
- 考虑降低输入分辨率（但会影响准确度）
- 在后台线程运行推理（已在 `SegmentationEngine` 中处理）
- 使用 Neural Engine（如果设备支持）：在 `MLModelConfiguration` 中设置 `computeUnits = .neuralEngine`

### Q: 如何调试模型输出？
A: 在 `SegmentationEngine.swift` 中添加日志，打印输出张量的形状和值范围，对比预期输出。

