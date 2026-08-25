# Day 6 — SAM ROI / Downsample Experiment (D6-B-SAM-ROI-EXPERIMENT)

> 可选：仅当 MobileSAM split 仍显著偏慢（尤其 encoder）时，评估工程侧 ROI/降采样策略。
> 约束：不改模型权重/结构。

## 背景
MobileSAM encoder 设计目标是对整张图像提取 embedding（全局上下文）。因此“只喂 ROI”在理论上会改变输入分布，可能影响 mask 质量/稳定性；但在工程上可作为 **trade-off** 实验，用于验证是否存在可接受的速度/质量折中。

## 方案候选（工程侧）
1) **ROI crop → resize to 1024 → encoder**
- 输入改为：以 YOLO top-1 bbox 为中心扩张 margin（例如 1.5× bbox 或固定像素扩边），从 camera frame 裁剪 ROI。
- 将 ROI resize/pad 到 1024×1024，再跑 encoder/decoder。
- 输出 mask 需要回投到 camera 全图坐标。

2) **全图低分辨率 encoder（非 ROI）**
- 先把整图更激进降采样到更小边（例如 512/640），再 upscale 到 1024 给 encoder。
- 目标：减少 preprocess/带宽压力（但 encoder 本身计算量可能不变/收益不大）。

## 评估指标
- 性能：enc(ms) mean / p95，seg(ms) mean / p95
- 质量：golden IoU（bus case）+ 真实视频稳定性（是否抖动/漂移/断裂）
- 工程复杂度：坐标映射复杂度、bug 风险

## 当前状态
- 本文件为实验计划与记录占位。
- 需要在真机 iPhone 11 跑 A/B 并回填数值：
  - Baseline（全图 strict preprocess）
  - ROI crop 方案（如实现）

## 结论（待填写）
- 是否值得做：TBD
- 推荐默认：保持全图 strict preprocess + encoder cadence + cache（优先）
