# MacBook Air M3 (2024) 模型转换快速指南

你的 MacBook Air M3 是完成这个转换的 **理想硬件**。以下是快速步骤。

## 为什么 M3 Mac 最适合？

- ✅ 原生 ARM64 架构，与 iOS 一致
- ✅ Core ML 工具链在 Mac 上原生支持
- ✅ Metal Performance Shaders (MPS) 自动加速 PyTorch 计算
- ✅ 不需要外接 GPU，芯片内置足够计算能力
- ✅ 转换后的 `.mlmodel` 可直接在 iOS 15+ 设备上运行

## 10 分钟快速转换步骤

### 1. 打开 Terminal，创建虚拟环境

```bash
cd ~/Projects/CODE-SHOW/APP/JudgeEverything/Copilot/python
python3 -m venv venv_m3
source venv_m3/bin/activate
```

### 2. 安装依赖（推荐使用 conda 以获得更好的兼容性）

如果你有 conda/miniconda，推荐：

```bash
conda create -n coreml_m3 python=3.11
conda activate coreml_m3
conda install -c pytorch pytorch::pytorch torchvision -c conda-forge
pip install coremltools==6.3 pillow numpy scikit-learn==1.5.1
```

或使用 pip：

```bash
pip install -U pip
pip install torch torchvision coremltools==6.3
pip install pillow numpy scikit-learn==1.5.1
```

### 3. 转换 YOLOv9

脚本已针对 M3 优化（自动启用 MPS）。运行：

```bash
python3 convert_yolov9_coreml_mac.py
```

**预计耗时**：5-10 分钟

输出日志会显示：`Using Metal Performance Shaders (MPS) for acceleration on M-series Mac`

成功标志：
```
Saved ../coreml/yolov9-c.mlmodel
```

### 4. 转换 MobileSAM

```bash
python3 convert_mobilesam_coreml_mac.py
```

**预计耗时**：8-15 分钟

成功标志：
```
Saved ../coreml/mobile_sam.mlmodel
```

## 转换完成后

两个 `.mlmodel` 文件会出现在：
```
APP/JudgeEverything/Copilot/coreml/
├── yolov9-c.mlmodel      (~100-150MB)
└── mobile_sam.mlmodel    (~80-120MB)
```

## 排查问题

### 问题 1: ImportError - MobileSAM 模块未找到

**原因**：脚本中的 `REPO_ROOT` 路径不正确

**解决**：编辑 `convert_mobilesam_coreml_mac.py`，第 11 行：

```python
REPO_ROOT = Path.home() / 'Projects' / 'MobileSAM' / 'MobileSAMv2'
```

改为你实际克隆的路径。查看路径：
```bash
find ~ -name "MobileSAMv2" -type d
```

### 问题 2: MPS 内存不足

M3 的 MPS 内存与 RAM 共享。如果转换中断，可改用 CPU：

编辑脚本，把这行：
```python
if torch.backends.mps.is_available():
    device = torch.device('mps')
```

改为：
```python
device = torch.device('cpu')  # Force CPU
```

然后重新运行（会慢一些，但更稳定）。

### 问题 3: Tracing 失败

如果 `torch.jit.trace` 失败，脚本会自动降级到 `torch.jit.script`。如果还是失败，说明模型存在不兼容的算子。

**解决方案**：
- 确保 PyTorch 版本为 2.7.0（与 coremltools 兼容）
- 检查权重文件是否完整（试试重新下载）

## 下一步

转换完成后，按 `XCODE_SETUP.md` 中的步骤在 Xcode 中创建项目并集成 `.mlmodel` 文件。

你的 M3 Mac 会得到最优化的转换结果，因为 Core ML 和 MPS 都是为 Apple Silicon 量身定制的！

