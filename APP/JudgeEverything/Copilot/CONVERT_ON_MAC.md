# macOS Core ML 转换步骤（已精确化）

已克隆的仓库位置：
- YOLOv9: `/workspaces/CODE-SHOW/APP/JudgeEverything/Copilot/python/models/yolov9`
- MobileSAM: `/workspaces/CODE-SHOW/APP/JudgeEverything/Copilot/python/models/MobileSAM`

## 1. 在 macOS 上准备环境

打开 Terminal 并执行：

```bash
# 创建 Python venv
python3 -m venv venv_coreml
source venv_coreml/bin/activate

# 安装关键依赖（特定版本组合）
pip install -U pip setuptools wheel
pip install torch==2.7.0 torchvision==0.18.1 coremltools==6.3
pip install numpy pillow scikit-learn==1.5.1
```

## 2. 上传仓库到 Mac 并准备权重文件

假设你已在 Mac 上克隆了此仓库到 `~/Projects/CODE-SHOW`。确保权重文件已放入：
- `~/Projects/CODE-SHOW/APP/JudgeEverything/Copilot/python/models/yolov9-c.pt`
- `~/Projects/CODE-SHOW/APP/JudgeEverything/Copilot/python/models/mobile_sam.pt`

## 3. 转换 YOLOv9

```bash
cd ~/Projects/CODE-SHOW/APP/JudgeEverything/Copilot/python
source venv_coreml/bin/activate

# 编辑脚本中的 REPO_ROOT 路径（指向克隆的 yolov9）
# 然后运行
python3 convert_yolov9_coreml_mac.py
```

期望输出：`Saved ../coreml/yolov9-c.mlmodel`（文件大小可能 ~100-200MB）

## 4. 转换 MobileSAM

```bash
# 编辑脚本中的 REPO_ROOT 路径（指向 MobileSAM/MobileSAMv2）
python3 convert_mobilesam_coreml_mac.py
```

期望输出：`Saved ../coreml/mobile_sam.mlmodel`

## 5. 上传转换后的文件到仓库

转换成功后，将生成的 `.mlmodel` 文件上传到：
- `APP/JudgeEverything/Copilot/coreml/yolov9-c.mlmodel`
- `APP/JudgeEverything/Copilot/coreml/mobile_sam.mlmodel`

可用 git 或手动复制上传。

## 常见问题

### ImportError: No module named 'mobilesamv2'
确保 REPO_ROOT 指向 `MobileSAM/MobileSAMv2` 目录（注意是 v2 子目录）。

### RuntimeError: weights_only load failed
编辑脚本，确保 `torch.load` 包含 `weights_only=False`。

### Tracing failed: ...
某些模型不支持直接 trace。脚本会自动回退到 `torch.jit.script`，但这需要模型代码兼容。如果还是失败，可能需要手动简化模型或提取特定层。

## 下一步：在 Xcode 中使用 .mlmodel

1. 在 Xcode 中创建新项目或打开现有项目。
2. 将 `.mlmodel` 文件拖入 Xcode：`APP/JudgeEverything/Copilot/coreml/`。
3. Xcode 会自动生成 Swift 模型包装类（如 `YOLOv9C`, `MobileSAM`）。
4. 在 ViewController 中加载并调用模型（见本仓库的 Swift 示例）。

