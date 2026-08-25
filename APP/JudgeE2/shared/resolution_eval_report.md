# Phase 3 Day 3 — 降编码分辨率评估报告

> **作者**：ML_Vision  
> **日期**：Phase 3 Day 3  
> **目的**：供 Architect + Builder 评审，裁决 Phase 3 是否切换编码分辨率  
> **对应任务**：tasks.md Phase 3 Day 3 ML_Vision — "评估降编码分辨率方案（Phase 3 优化入口点 2）"

---

## 1. 实验设置

| 项目 | 详情 |
|------|------|
| 模型 | MobileSAM TinyViT-5M（同一 `mobile_sam.pt` checkpoint） |
| 权重迁移验证 | 全量迁移：0 missing keys / 0 unexpected keys ✅ |
| 测试分辨率 | 1024×1024（基准） / 768×768 / 512×512 |
| 架构适配方式 | 动态 `forward_features`：`img_size // 16` 替换硬编码 64 |
| Decoder 适配 | 非 1024 embedding 双线性上采样至 64×64（[1,256,H,W] → [1,256,64,64]） |
| 提示类型 | 单前景点 prompt，labels=`[1.0, -1.0]`（前景+padding） |
| 坐标归一化 | 点坐标统一映射至 1024 空间（SAM decoder 固定以 1024 归一化） |
| 测试集 | 4 张真实图像（good1/2、rotate1/2）× 5 个 tap 点 = 20 样本 |
| Mac 测量 | PyTorch CPU，warmup=3，iter=8 |
| iPhone 11 预测 | 以 Phase 2 Debugger 实测基准（milfix fp16 mean=857ms）为锚点 |

### 1.1 关键架构发现

TinyViT-5M 的 `forward_features` 原始代码硬编码了 `x.view(B, 64, 64, C)`，导致非 1024 输入直接崩溃。本实验通过运行时猴子补丁修复为动态计算：

```python
feat_size = self.img_size // 16   # 1024→64, 768→48, 512→32
x = x.view(B, feat_size, feat_size, C)
```

**权重兼容性验证**：所有 attention_biases 形状依赖 `window_size`（固定 7/7/14/7），不依赖 `input_resolution`，因此全量权重可无损迁移至 768/512 架构。

---

## 2. 延迟测量结果

### 2.1 Mac CPU 实测（PyTorch）

| 分辨率 | Mac mean (ms) | Mac p95 (ms) | 相对 1024 |
|--------|--------------|-------------|----------|
| 1024×1024 | 867.0 | 965.7 | 1.000 |
| 768×768 | 561.5 | 735.7 | **0.648** |
| 512×512 | 264.5 | 294.6 | **0.305** |

> 注：rotate 图像（竖拍截图，828×1792）为了验证非正方形输入的处理正确性也纳入计算；图像尺寸对 encoder 延迟无影响（输入张量始终为 `[1, 3, res, res]`）。

### 2.2 iPhone 11 延迟预测（CoreML ANE）

**锚点**：Phase 2 Debugger 实测 1024×1024 milfix fp16 encoder = **857ms（均值）/ 933ms（p95）**

| 分辨率 | 预测均值（Mac比例法） | 预测均值（面积法） | 保守估计区间 | 加速倍数 |
|--------|---------------------|-----------------|------------|---------|
| 1024×1024 | 857 ms（基准） | 857 ms | — | ×1.00 |
| 768×768 | **555 ms** | 482 ms | 500–560 ms | **×1.54** |
| 512×512 | **261 ms** | 214 ms | 230–270 ms | **×3.28** |

**预测方法说明**：
- **Mac 比例法**（保守）：`iPhone11 = 857ms × (Mac_res_mean / Mac_1024_mean)`
- **面积法**（理论）：`iPhone11 = 857ms × (res/1024)²`，基于注意力与卷积均线性于输入面积
- ANE 在 attention-heavy 网络上的加速比通常优于 Mac CPU；实际加速预计介于两种预测之间

### 2.3 Decoder 延迟（不变）

降低编码分辨率不影响 Decoder 延迟，Phase 2 实测 Decoder mean=**61ms** 保持不变。

---

## 3. Mask 覆盖精度评估

以 1024×1024 输出为参考掩码，计算 768/512 掩码 IoU：

### 3.1 综合统计（20 样本）

| 分辨率 | IoU mean | IoU median | IoU min | IoU max | iou_pred mean |
|--------|---------|-----------|---------|---------|--------------|
| 1024×1024（参考） | 1.000 | 1.000 | — | — | **0.848** |
| 768×768 | **0.581** | 0.613 | 0.003 | 0.913 | 0.781 |
| 512×512 | **0.592** | 0.632 | 0.002 | 0.883 | 0.774 |

### 3.2 精度解读

**768 vs 1024**：
- 中位 IoU=0.613，大多数情况下 mask 覆盖 60% 以上的参考区域
- iou_pred 降低约 7.9%（0.848 → 0.781），模型自信度下降
- 高质量案例（max=0.913）：接近 1024 水平，在视觉上几乎无差
- 低质量案例（min=0.003）：在图像角落 tap 时，低分辨率特征图精度严重退化

**512 vs 1024**：
- 中位 IoU=0.632，略优于 768（反映了特定测试集的分布偏差）
- 最大值 0.883 高于预期，但最小值极低（0.002）
- iou_pred 下降幅度与 768 相近（0.848 → 0.774）

### 3.3 精度退化根因分析

| 根因 | 影响 | 说明 |
|------|------|------|
| 模型从未在 768/512 上训练 | ⭐⭐⭐ 主因 | 相对位置编码和特征尺度均针对 1024 优化 |
| Embedding 上采样引入插值误差 | ⭐⭐ 次因 | 48×48/32×32 → 64×64 双线性上采样，边界区域信息损失 |
| 点坐标在粗粒度特征图上精度不足 | ⭐ 补充 | 512 时 1 个特征单元对应 32px，tap 精度粗糙 |

> **重要**：若对 768/512 进行哪怕 1-2 epoch 的 fine-tuning，IoU 有望从 0.58 提升至 0.80+。当前测试为纯权重迁移（zero-shot）基线。

---

## 4. 关键实现细节（Builder 必读）

### 4.1 点坐标归一化规则（关键 Bug Fix）

SAM Decoder 固定以 `(1024, 1024)` 归一化点坐标。使用非 1024 编码器时，点坐标**必须**转换到 1024 空间：

```swift
// PointPromptBuilder.buildPointPrompt 中的坐标变换
// 不论编码器实际分辨率是 768 还是 512，
// 都需将 canonical 坐标转换到等效的 1024 空间坐标
let scale_1024 = 1024.0 / Double(max(origW, origH))
let sam_x = canonicalPoint.x * scale_1024   // 永远基于 1024 归一化
let sam_y = canonicalPoint.y * scale_1024
```

**如果使用了 768 编码器，但错误地将 768 空间坐标直接传入 Decoder：**
- Decoder 会把 (177, 384) 解释为 1024 空间中的 (0.173, 0.375)
- 而实际图像内容在 (0.231, 0.500) 处
- 结果：mask 完全错位（IoU≈0），这正是实验初期遇到的 Bug

### 4.2 Embedding 上采样位置

768/512 编码器输出 `[1, 256, 48, 48]` 或 `[1, 256, 32, 32]`。Swift 端需在调用 Decoder 前双线性上采样至 64×64：

```swift
// SAMEncoder.swift 或 SAMDecoder.swift 适配层
if embedding.shape[2] != 64 {
    // vImage or MPS 双线性上采样：[1, 256, H, H] → [1, 256, 64, 64]
}
```

### 4.3 CoreML 导出可行性

若 Architect 批准切换分辨率，导出流程：
1. 修改 `export_encoder_fp16_milfix.py` 中的 `dummy_trace` 尺寸为 `(1, 3, 768, 768)`
2. 同时 monkeypatch `forward_features` 为动态版本
3. 加载原始 checkpoint（全量权重兼容）
4. 转换后输出 `MobileSAM_ImageEncoder_fp16_milfix_768.mlpackage`
5. Decoder mlpackage 保持不变，在 Swift 层添加上采样桥接

---

## 5. 综合评估与推荐

### 5.1 权衡矩阵

| 维度 | 1024（当前） | 768（候选） | 512（激进） |
|------|------------|-----------|-----------|
| 预测 iPhone 11 延迟 | 857 ms | ~555 ms | ~261 ms |
| 加速倍数 | ×1.00 | **×1.54** | **×3.28** |
| Mask IoU vs 1024 | 1.000 | 0.581 | 0.592 |
| iou_pred 均值 | 0.848 | 0.781 | 0.774 |
| 实现复杂度 | 零 | 低（+上采样桥接） | 低（+上采样桥接） |
| 需要 fine-tune？ | — | 推荐（非必须） | 强烈推荐 |

### 5.2 ML_Vision 推荐意见

**768×768**：⚠️ **条件接受**
- 加速 ×1.54，可将 encoder 从 857ms 降至 ~555ms
- 当前 zero-shot IoU = 0.581，精度有损失但结构基本保留
- 建议 Architect 决定是否在 Phase 3 范围内接受此精度 tradeoff
- 若可接受：ML_Vision 可在 Day 4 完成导出（< 10 min）
- **最优路径**：接受 768，同时将 1024-only 路径保留为用户可选的"高精度模式"

**512×512**：❌ **暂不推荐（zero-shot 精度不足）**
- 延迟加速显著（×3.28），但 zero-shot 精度与 768 相近（无额外收益）
- 若需使用 512，必须先进行分辨率专项 fine-tuning（Phase 4 范畴）

### 5.3 Architect 裁决所需信息

请 Architect 在以下问题上给出裁决，以便 Builder Day 4 展开实施：

1. **精度阈值**：Phase 3 Tap-to-Segment 的可接受 mask 质量下限是 IoU=0.58，还是要求 > 0.75？
2. **768 批准？**：若批准 768，是否在 Phase 3 立即切换，还是并行保留 1024 作为可配置项？
3. **fine-tuning 范畴**：fine-tuning 是 Phase 3 还是 Phase 4 任务？（ML_Vision 认为属于 Phase 4）
4. **512 保留？**：是否保留 512 作为"极速模式"选项以供未来 fine-tuning 后使用？

---

## 6. 实验脚本

```
shared/eval_resolution.py           — 完整评估脚本（可重现）
shared/resolution_eval_results.json — 原始数值数据
```

---

*Phase 3 Day 3 ML_Vision 降分辨率评估完成。数据已准备，等待 Architect Day 4 裁决。*
