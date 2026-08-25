# Model Plan — Phase 3, Day 1 (ML_Vision)

> 承接 `shared_v2/model_plan.md`（Phase 1–2 基线）。Phase 3 Day 1 新增内容见下。
> 上游引用：`models/MobileSAM_PromptMaskDecoder.mlpackage`、`models/MobileSAM_ImageEncoder.mlpackage`（Phase 2 已验证）

---

## Phase 3 Day 1 — 任务 A：SAM 点提示格式（Point Prompt I/O 规范）

### A.1 Decoder 完整 I/O（实测确认）

| Tensor | Shape | dtype |
|--------|-------|-------|
| `image_embeddings` | `[1, 256, 64, 64]` | Float16 |
| `point_coords` | **`[1, 2, 2]`** | Float16 |
| `point_labels` | **`[1, 2]`** | Float16 |
| `mask_input` | `[1, 1, 256, 256]` | Float16 |
| `has_mask_input` | `[1]` | Float16 |
| ← `low_res_masks` | `[1, 1, 256, 256]` | Float16 |
| ← `iou_predictions` | `[1, 1]` | Float16 |

> **关键修正**：Architect spec §5.3 预测 `point_coords` 可能为 `[1,1,2]`，实测为 **`[1,2,2]`**（固定 2 点）。Builder 在实现 `PointPromptBuilder` 时必须构造 2 点张量，不得使用 `[1,1,2]`。

---

### A.2 Phase 3 单次前景点 Tap 的 Prompt 构造

模型 `point_coords` 形状固定为 **[1, 2, 2]**，Phase 3 单点 Tap 需要 **2 点输入**（前景点 + padding）：

```
point_coords[0] = [[tap_x_in_SAM_space, tap_y_in_SAM_space],
                   [0.0,                0.0               ]]   ← padding
point_labels[0] = [1.0,   ← foreground
                   -1.0]  ← padding（SAM 约定）
```

**SAM 空间坐标定义**：
- 输入坐标来自 `PointPromptBuilder.buildPointPrompt(canonicalPoint:origSize:inputSize:)`
- 变换：ResizeLongestSide(inputSize=1024) + centered pad → SAM pixel space (0 ~ 1023)
- `point_coords` tensor 填入 **SAM 像素坐标（0~1023）**，**不得预先 ÷ 1024**
- 模型内部 `_embed_points` 执行归一化：`(point_coords + 0.5) / 1024.0`

---

### A.3 点标签语义（模型支持的全部 label 值）

| label 值 | 语义 |
|---------|------|
| `0.0` | 背景点（background） |
| `1.0` | **前景点（foreground）— Phase 3 使用** |
| `2.0` | 框左上角（SAM box convention，Phase 2 使用） |
| `3.0` | 框右下角（SAM box convention，Phase 2 使用） |
| `-1.0` | **Padding / 忽略点 — Phase 3 第 2 点使用** |

**Trace 兼容性确认**：模型使用张量比较操作（`point_labels == i`，`point_labels != -1`）而非 Python if/else 分支。因此 trace 时使用 `[2.0, 3.0]` 的版本，对 Phase 3 的 `[1.0, -1.0]` 输入完全有效，无需重新 trace/re-export Decoder。

---

### A.4 Multi-mask Output 确认

- Decoder 输出：单一最优 mask（`low_res_masks: [1, 1, 256, 256]`），非多候选
- 导出时 `_select_mask()` 固定选择内部 3 个候选 mask 的 index 0
- `iou_predictions: [1, 1]` 为该 mask 的 IoU 质量分
- Phase 3 直接使用此单 mask 输出即可，无需多候选选择逻辑

**Phase 3 输出质量判断**：
- `iou_pred > 0.5`：合理的前景分割
- `iou_pred < 0.1`：质量极差，显示 fallback 状态（不添加无效 mask）
- 阈值可根据实测结果调整

---

### A.5 Phase 2 Box Prompt 与 Phase 3 Point Prompt 对比

| 项目 | Phase 2（Box） | Phase 3（Point） |
|------|---------------|-----------------|
| `point_coords` | `[[x1,y1],[x2,y2]]`，框两对角坐标 | `[[x,y],[0,0]]`，前景点 + padding |
| `point_labels` | `[2.0, 3.0]` | `[1.0, -1.0]` |
| `mask_input` | 全零 `[1,1,256,256]` | 全零 `[1,1,256,256]`（相同） |
| `has_mask_input` | `[0.0]` | `[0.0]`（相同） |
| Decoder 方法 | `decode(embedding:prompt:)` | `decode(embedding:pointPrompt:)`（新增重载） |

---

### A.6 Builder PointPromptBuilder Swift 参考实现

```swift
// Interaction/PointPromptBuilder.swift
// point_coords shape: [1, 2, 2]（固定，不可为 [1,1,2]）

struct PointPromptBuilder {
    struct PointPrompt {
        let pointCoords: MLMultiArray    // [1, 2, 2] Float32
        let pointLabels: MLMultiArray    // [1, 2]   Float32
        let maskInput: MLMultiArray      // [1, 1, 256, 256] Float32（全零）
        let hasMaskInput: MLMultiArray   // [1] Float32（值 0.0）
    }

    static func buildPointPrompt(
        canonicalPoint: CGPoint,
        origSize: CGSize,
        inputSize: Int = 1024
    ) -> PointPrompt? {
        // 1. ResizeLongestSide(inputSize) 变换
        let scale = Float(inputSize) / Float(max(origSize.width, origSize.height))
        var samX = Float(canonicalPoint.x) * scale
        var samY = Float(canonicalPoint.y) * scale
        // 2. clamp 到 [0, inputSize-1]
        samX = max(0, min(samX, Float(inputSize - 1)))
        samY = max(0, min(samY, Float(inputSize - 1)))

        // 3. 构造 point_coords [1, 2, 2]
        guard let coords = try? MLMultiArray(shape: [1, 2, 2], dataType: .float32),
              let labels = try? MLMultiArray(shape: [1, 2], dataType: .float32),
              let maskIn = try? MLMultiArray(shape: [1, 1, 256, 256], dataType: .float32),
              let hasMask = try? MLMultiArray(shape: [1], dataType: .float32)
        else { return nil }

        // 前景点
        coords[[0, 0, 0] as [NSNumber]] = NSNumber(value: samX)
        coords[[0, 0, 1] as [NSNumber]] = NSNumber(value: samY)
        // Padding 点
        coords[[0, 1, 0] as [NSNumber]] = 0.0
        coords[[0, 1, 1] as [NSNumber]] = 0.0

        labels[[0, 0] as [NSNumber]] = 1.0   // foreground
        labels[[0, 1] as [NSNumber]] = -1.0  // padding

        // mask_input / has_mask_input 均为零
        for i in 0..<maskIn.count { maskIn[i] = 0.0 }
        hasMask[0] = 0.0

        return PointPrompt(pointCoords: coords, pointLabels: labels,
                           maskInput: maskIn, hasMaskInput: hasMask)
    }
}
```

> ⚠️ **注意**：`samX/samY` 的单位是 SAM 像素坐标（0~1023），模型内部再 ÷ 1024。上面的实现仅做 ResizeLongestSide，未包含 centered padding。若图像非正方形，还需加上 padding offset。详见 Phase 2 `PromptBuilder.buildBoxPrompt` 中的 padding 计算逻辑，`PointPromptBuilder` 必须使用相同变换。

---

## Phase 3 Day 1 — 任务 B：ANE 对齐告警修复方案

### B.1 告警根因（精确定位）

**告警信息**：`Invalid input tensor channel 1 ... must be aligned on 64 bytes`

**根因**：MobileSAM Encoder neck 中的 `LayerNorm2d` 实现（SAM 原生）对 channel 维度求均值：

```python
# mobile_sam/modeling/image_encoder.py — LayerNorm2d.forward
u = x.mean(1, keepdim=True)          # [1, 256, 64, 64] → [1, 1, 64, 64]
s = (x - u).pow(2).mean(1, keepdim=True)  # [1, 1, 64, 64]
```

产生 4 个 `[1, 1, 64, 64]` Float16 中间张量：
- `u_1_cast_fp16` / `s_1_cast_fp16`（第一个 LayerNorm2d 的均值/方差）
- `u_cast_fp16` / `s_cast_fp16`（第二个 LayerNorm2d 的均值/方差）

**ANE Float16 对齐要求**：iPhone 11 A13 ANE 要求 `channel × sizeof(Float16) = C × 2` 字节必须是 64 字节的倍数，即 C 必须是 32 的倍数。C=1 仅 2 bytes，不满足 → ANE 拒绝调度 → 回退 CPU。

**影响范围**：这 4 个 reduce_mean 操作（以及其下游 sub/div/mul 操作）均回退 CPU，是 encoder 延迟偏高（857ms 均值）的重要原因之一。

**明确排除**：3-channel RGB 输入（`[1, 3, 1024, 1024]`）并非告警根因。第一个 conv 将 3 通道立即扩展为 32 通道（`[1, 32, 512, 512]`），满足对齐要求。

---

### B.2 修复方案（已执行）

**方案选择：Float32 重导出（Option A，立即有效）**

已生成修复模型：
```
models/MobileSAM_ImageEncoder_fp32.mlpackage   (28.0 MB，Float32 全精度)
models/MobileSAM_ImageEncoder.mlpackage        (14.1 MB，原始 Float16，保留不删)
```

导出脚本：`shared/export_encoder_fp32_ane_fix.py`

**修复效果验证（Python 端静态分析）**：
- 原始（Float16）：658 个 `_fp16` 命名 op，含 4 个 `[1,1,64,64]` Float16 reduce_mean 张量
- 修复后（Float32）：0 个 `_fp16` 命名 op，`reduce_mean` 输出仍为 `[1,1,64,64]` 但现在是 **Float32**，不触发 ANE Float16 对齐检查

**I/O 接口不变**：
- Input: `image: [1, 3, 1024, 1024]`（Float32 输入，与 Swift 代码兼容）
- Output: `image_embeddings: [1, 256, 64, 64]`
- 权重来自同一 `mobile_sam.pt` checkpoint，功能等价，无精度损失

---

### B.3 Builder 集成指令（Day 2 执行）

1. 在 Xcode 项目中，删除 `MobileSAM_ImageEncoder.mlpackage` 的引用
2. 拖入 `MobileSAM_ImageEncoder_fp32.mlpackage`（勾选 target membership）
3. 更新 Swift 中模型加载代码（文件名变更）：
   ```swift
   // 旧
   let config = MLModelConfiguration()
   let encoder = try MobileSAM_ImageEncoder(configuration: config)
   // 新
   let encoder = try MobileSAM_ImageEncoder_fp32(configuration: config)
   ```
4. Clean Build（⌘⇧K）→ 重新 Build
5. 真机运行，观察 console：
   - ✅ 预期：`Invalid input tensor channel 1 ... aligned on 64 bytes` 警告**消失**
   - ✅ 预期：encoder 延迟较 Phase 2 的 857ms 有所下降（需 Debugger Day 7 实测）

---

### B.4 Option B（备选优化路径，Phase 3 Day 3 后评估）

若 Float32 encoder 延迟改善不足（如仍 > 600ms），可考虑：

**MIL PassPipeline 融合 LayerNorm**：使用 coremltools MIL 将 4 个 reduce_mean + 下游 sub/div/mul 融合为单一 `layer_norm` MIL op。融合后的 `layer_norm` op 不暴露 C=1 中间张量给 ANE 调度器，可保留 Float16 精度同时消除告警。

脚本占位符：`shared/export_encoder_fp16_milfix.py`（未来编写，需 coremltools MIL pass 经验）

---

## Deliverable 确认（Phase 3 Day 1 ML_Vision）

- [x] **SAM 点提示 I/O 规范**：`point_coords [1,2,2]` / `point_labels [1,2]` 已确认
- [x] **单点前景 Tap Prompt 构造**：`[1.0, -1.0]` 标签 + padding 点 `[0,0]`，SAM 空间坐标（0~1023）
- [x] **Multi-mask 确认**：单一 mask 输出（`iou_predictions [1,1]`），Phase 3 直接使用
- [x] **ANE 告警根因**：LayerNorm2d neck 的 C=1 Float16 reduce_mean 中间张量
- [x] **修复模型已生成**：`MobileSAM_ImageEncoder_fp32.mlpackage`（同 I/O 接口，功能等价）
- [x] **Builder 集成指令**：见 §B.3，Day 2 替换模型后验证告警消除

---

*Phase 3 Day 1 ML_Vision 完成。可进入 Day 2（Builder：TouchHandler + ANE 修复模型集成）。*

---

## Phase 3 Day 3 — 任务：降编码分辨率评估（优化入口点 2）

> 完整报告：`shared/resolution_eval_report.md`  
> 原始数据：`shared/resolution_eval_results.json`  
> 评估脚本：`shared/eval_resolution.py`

### C.1 架构可行性确认

**TinyViT-5M 支持降分辨率运行，方案可行，但需 runtime 补丁。**

原始代码 `tiny_vit_sam.py` line 611 硬编码 `x.view(B, 64, 64, C)`，对非 1024 输入直接崩溃。  
修复方式：将 `64` 改为动态值 `self.img_size // 16`：

```python
feat_size = self.img_size // 16   # 1024→64, 768→48, 512→32
x = x.view(B, feat_size, feat_size, C)
```

**权重兼容性**：所有 attention_biases 依赖 `window_size`（固定），不依赖 `input_resolution`。  
全量权重可无损迁移至 768/512 架构（0 missing keys / 0 unexpected keys）。

### C.2 延迟预测（iPhone 11 CoreML ANE）

| 分辨率 | Mac CPU mean | 相对比例 | iPhone 11 预测均值 | 加速倍数 |
|--------|-------------|---------|------------------|--------|
| 1024×1024 | 867 ms | 1.000 | 857 ms（实测基准） | ×1.00 |
| 768×768 | 562 ms | 0.648 | **~555 ms** | **×1.54** |
| 512×512 | 265 ms | 0.305 | **~261 ms** | **×3.28** |

预测方法：Mac 相对比例法（保守），锚定 Phase 2 Debugger milfix fp16 实测 857ms。  
理论面积法预测 768→482ms、512→214ms，实际值预计在两者之间。

### C.3 Mask 精度评估（zero-shot，4 图×5 点=20 样本）

| 分辨率 | IoU vs 1024 (mean) | IoU vs 1024 (median) | iou_pred mean |
|--------|--------------------|---------------------|---------------|
| 1024×1024 | 1.000 | 1.000 | 0.848 |
| 768×768 | **0.581** | 0.613 | 0.781 |
| 512×512 | **0.592** | 0.632 | 0.774 |

> ⚠️ **重要**：当前精度为 zero-shot（纯权重迁移，无 fine-tuning）。  
> 若对目标分辨率做 1–2 epoch fine-tuning，IoU 预计从 0.58 提升至 0.80+。

### C.4 关键 Bug：点坐标归一化

SAM Decoder 固定以 `(1024, 1024)` 归一化点坐标。使用非 1024 Encoder 时，  
点坐标**必须在传入 Decoder 前换算至 1024 空间**：

```swift
// 不论 encoderInputSize=768 还是 512，坐标始终以 1024 归一化
let scale = 1024.0 / Float(max(origW, origH))
let samX = canonicalPoint.x * scale
let samY = canonicalPoint.y * scale
```

若直接传入 768 空间坐标（未换算），将导致 mask 完全错位（IoU≈0），此 Bug 已在评估阶段发现并修复。

### C.5 CoreML 导出规格（若 Architect 批准）

**768 导出方案**：
1. 修改 `export_encoder_fp16_milfix.py`：`dummy_trace = torch.randn(1, 3, 768, 768)` + monkeypatch `forward_features`
2. 输出：`models/MobileSAM_ImageEncoder_fp16_milfix_768.mlpackage`（~14 MB，权重相同）
3. Swift 适配：Encoder 输出 `[1, 256, 48, 48]` → 双线性上采样 → `[1, 256, 64, 64]` → Decoder
4. `PointPromptBuilder` 保持 1024 归一化（见 C.4），无需修改 Decoder

预计导出时间：< 10 分钟。

### C.6 ML_Vision 推荐

- **768×768**：⚠️ 条件接受 — 加速 ×1.54，zero-shot IoU=0.581（待 Architect 裁决精度阈值）
- **512×512**：❌ 暂不推荐 — zero-shot 精度不足，需 fine-tuning 后再考虑
- **保守选项**：1024 保持不变，Phase 3 专注 Tap UX；降分辨率列入 Phase 4

---

*Phase 3 Day 3 ML_Vision 完成（2026-07-22）。等待 Architect Day 4 裁决降分辨率方案。*

---

# Phase 3 Day 6 — token 0 纳入候选的离线评估（2026-08-11）

> 执行说明：本节由主协调者代 ML_Vision 完成——ML_Vision 两次被会话额度中断，
> 但其评估脚本（`shared/prep_token0_frames.py` / `eval_token0.py` / `eval_token0_pick.py`）
> 已可执行，本节是运行这些脚本后的结果分析。脚本作者的方法学设计保留原样。

## D.0 背景

`shared/export_decoder_multimask.py:172` 为 `return masks[:, 1:4]`，**丢弃了 token 0**。
SAM 的 mask head 输出 4 个 token：token 0 是「单 mask 头」（multimask_output=False 时的
唯一答案），token 1–3 是歧义候选（子部件 / 部件 / 整体）。Day 3 部署的单 mask decoder
经 ML_Vision 核对**就是 token 0**（IoU 1.000）。

而真机实测显示候选选择已退化为常量：10/10 次 `ch0 < ch1 < ch2`，「面积最小且 ≤cap60」
必然命中 ch0 ⇒ **实际等价于恒取 token 1**。

## D.1 方法

4 个场景 × 5 个 tap 点 = **20 次**，从 Session A 录像重建规范帧（1920×1080，
`prep_token0_frames.py` 按 aspectFill 几何精确反推，无自由参数）。含 3 个已知失败
目标（S3 电脑盖 / S2 显示器底座 / S5 白插头）、2 个退化提示（点裸大理石 / 裸泡沫垫）
与多个对照物。`eval_token0_pick.py` 逐行复现设备端选择规则（常量全部转录自
`MaskRenderer.swift`），对同一批 logits 重放四种候选集：

| 记号 | 候选集 |
|---|---|
| **CUR** | token 1,2,3（当前出货） |
| **T0** | 仅 token 0（Day 3 行为） |
| **A4** | token 0,1,2,3（4 候选导出） |
| **FB** | token 0 作 fallback |

## D.2 结果一：token 0 ≈ token 2，不是新粒度档

token 0 与各 token 的 mask IoU（20 次中位）：

| 对比 | 中位 IoU |
|---|---|
| token 0 ↔ **token 2** | **≈ 0.94** |
| token 0 ↔ token 1 | ≈ 0.70 |
| token 0 ↔ token 3 | ≈ 0.44 |

⇒ token 0 落在「部件」档，与 token 2 高度重合。**它不提供新的粒度层级。**

## D.3 结果二：A4 与 FB 对结果零影响（决定性）

- **A4 vs CUR：20 次中 19 次选择完全相同**；唯一差异（S3 mug）两者面积/bbox/fill 完全一致，
  仅 iou_pred 标注不同 ⇒ **实质影响为 0**。
  原因：token 0 面积在 19/20 上 **≥ token 1**，而规则取最小 ⇒ token 0 永远选不上。
- **FB vs CUR：20/20 完全相同**。fallback 从未触发（CUR 总能找到合法候选）。

⇒ **「导出 4 候选」与「token 0 作 fallback」都是无效改动**，不值得付出导出与契约变更成本。

## D.4 结果三：token 0 的 iou_pred 在 20/20 上都不是最高的

逐例检查四个 token 的 iou_pred 最大者：token 2 占 10/20，token 1 占 5，token 3 占 4，
**token 0 占 0**。

T0（替换方案，即回到 Day 3）表现为此消彼长：
- **变好**：S3 chair_back（fill 0.385→0.527、stab 0.869→0.964）、S2 mouse、S1 ipad
- **变差**：S2 tumbler（iou 0.990→0.839、stab 0.910→0.655）、S1 canon_bag、S1 tissue_box
- **明显更差**：退化提示上溢出翻倍——S3 裸大理石 8.4%→**26%** 内容区；S5 裸泡沫 16%→**25%**

## D.5 推荐：**维持现状，不导出 token 0**

四条理由（D.2–D.4）互相独立。**真正的杠杆是选择规则，不是候选集。**

## D.6 数据指出的实际方向（交 Architect / A-1）

**token 2 在 10/20 上 iou_pred 最高，且常常 iou 与 stability 双高，而现行规则几乎从不选它。**
典型：S1 ipad（t2 iou 1.007 / stab 0.984，选中的 t1 是 0.949 / 0.830）、S2 mouse
（t2 0.996 / 0.992 vs t1 0.967 / 0.868）、S3 chair_back（t2 0.976 / 0.970 vs t1 0.856 / 0.869）。

⇒ A-1 复议时的候选方向应是「**在现有 token 1–3 内改进选择**」，而非扩充候选集。

## D.7 保留项（不得省略）

- **R-T1 重建帧未能复现设备端的灾难性失败**：点击 S3 电脑盖 / S5 插头时，四个 token 面积
  均在 0.3%–3% 内容区，无一是「整张桌子」。设备上的大面积溢出**只在点击裸表面时**复现
  （marble_bare / foam_bare）。原因推测为重建帧的 embedding 与设备当时不同、tap 点为近似值。
  ⇒ **本节对三个失败案例的结论强度低于对整体趋势的结论强度。**
- **R-T2 无人工评分配对**：全部指标为 iou_pred / stability / fill，均为模型自评量，
  且 Debugger 已证三者共线（stab~iou Pearson 0.945、iou~fill 0.924）。
  「哪个 token 更接近用户意图」严格来说**未被测量**。
- **R-T3 n=20 且集中于 4 个场景**，不足以支撑阈值级结论。
