# Day6 — MobileSAM computeUnits AB + FP16 Plan (iPhone 11)

任务：
- **D6-M-SAM-COMPUTEUNITS-AB**：给出 iPhone 11 上 encoder/decoder 的 computeUnits 推荐 AB（`.cpuAndGPU` vs `.all`）
- **D6-M-SAM-FP16-PLAN**：确认 encoder 是否可安全 FP16（或其它格式）以争取加速/降低带宽（以不破坏 IoU 为前提）

设备假设：iPhone 11 (A13) / iOS 17。

---

## 1) 当前模型形态（与 computeUnits/FP16 相关）

当前 MobileSAM CoreML artifacts：
- `models/MobileSAM_ImageEncoder.mlpackage`
- `models/MobileSAM_PromptMaskDecoder.mlpackage`

两者均为：
- **type = mlProgram**（mlprogram）
- 输入/输出当前都是 **FLOAT32 MultiArray**（NCHW）
  - encoder input `image`: (1,3,1024,1024) float32
  - encoder output `image_embeddings`: (1,256,64,64) float32
  - decoder inputs/outputs 亦为 float32

> 备注：即便 spec 是 float32，CoreML 仍可能在内部使用 FP16 计算（取决于导出时 compute_precision、runtime、computeUnits）。

---

## 2) computeUnits：推荐策略（先稳再快）

### 2.1 为什么要 AB？
- `.all` 允许 ANE/GPU/CPU 混合调度：**潜在最快**，但也更容易遇到“部分 op 不上 ANE → fallback/调度波动/首次初始化成本”。
- `.cpuAndGPU` 通常更稳：**更可预测**，但 encoder 可能明显更慢（尤其对 A13）。

从已有真机观察（Day4/Day5 文档）：
- encoder 是绝对瓶颈（稳态 ~575–613ms），decoder 很快（~28–36ms）。

因此：
- **encoder 值得重点 AB**（决定端到端体验）
- **decoder 影响较小**（但也可顺带测）

### 2.2 默认推荐（未做 AB 前的工程建议）
- **Encoder：先用 `.cpuAndGPU` bring-up/回归更稳**；当功能/几何/IoU 稳定后，优先尝试 `.all` 争取 ANE 加速。
- **Decoder：默认 `.cpuAndGPU` 即可**（因为本身很快）；若 encoder 选 `.all` 且整体稳定，也可统一 decoder 为 `.all` 以减少系统决策差异。

> 换句话说：**encoder 决定性能上限，decoder 决定稳定性边角**。

---

## 3) computeUnits AB 设计（建议你们按这个跑一次就能得结论）

### 3.1 AB 变量设置
建议做 4 组（最有信息量）：
1) enc `.cpuAndGPU` + dec `.cpuAndGPU`
2) enc `.all` + dec `.cpuAndGPU`
3) enc `.cpuAndGPU` + dec `.all`
4) enc `.all` + dec `.all`

### 3.2 采集指标（每组都要）
每组至少记录 3 类：
- **cold load / first-run**：
  - `MLModel(contentsOf:)` / load 时间（首次 vs 二次）
  - warmup（第一次 predict）时间
- **steady-state**（建议取 50 次调用，统计 avg + p50/p90/p99）：
  - encoder ms
  - decoder ms
  - seg total ms
- **质量（IoU / mask 稳定性）**：
  - 继续用你们现有 Golden（bus）跑 3 次，记录 IoU
  - 同时观察是否出现偶发 mask 错位/全空/崩溃

判定标准（建议）：
- 如果 `.all` 带来 encoder **≥20–30%** 加速，且 p99 没显著变差、IoU 不退化 → 值得用 `.all`。
- 如果 `.all` 的 p99/outlier 明显增加（例如偶发 >2s），或首次加载明显更糟 → 生产默认仍用 `.cpuAndGPU`，把 `.all` 做成开关/auto fallback。

---

## 4) FP16 计划（重点：encoder）

### 4.1 目标与收益来源
对 encoder：
- 计算量大、特征图多，**FP16** 可能带来：
  - 更高吞吐（取决于 ANE/GPU 路径）
  - 更低内存带宽

此外还有一个“隐藏的大头”：**embedding 的读写带宽**
- `image_embeddings` 是 (1,256,64,64) ≈ 1,048,576 floats
  - float32：约 4MB
  - float16：约 2MB
- 如果你们做 embedding cache + 高频 decoder，embedding dtype 变成 float16 可以持续省带宽。

### 4.2 可行的 FP16 路线（从低风险到高收益）

**路线 P1（低风险，推荐先做）— compute_precision=FP16，但保持 I/O 为 float32**
- 导出时设置 `compute_precision=ct.precision.FLOAT16`
- I/O 仍然是 float32 multiarray（Swift 侧无需改类型）
- 预期：在不改变外部接口的情况下，争取内部 fp16 加速。
- 风险：通常较低；主要风险是数值微小变化导致 IoU 轻微波动（一般可接受）。

**路线 P2（中风险，中收益）— embedding 输出改为 float16（I/O 改变）**
- 让 encoder 输出 `image_embeddings` 为 float16；decoder 输入也改为 float16。
- 收益：embedding cache 带宽直接减半；decoder 输入搬运也更省。
- 风险：
  - 需要改 Swift 侧 `MLMultiArray` dtype 处理（float16 读写）
  - 需要确认 coremltools / runtime 对 float16 multiarray I/O 的支持与实际性能

**路线 P3（高风险）— weight-only / palettization / int8**
- 需要校准集与更严格的精度回归；对 IoU 风险最大。
- 当前阶段不建议。

### 4.3 “是否安全”的验证方法（不破坏 IoU）
建议用 3 层回归：
1) **golden bus**：IoU 不低于你们现在基线（建议容忍 -0.01 以内波动）
2) **多场景小集（10–20 张）**：记录 mask area、bbox 内覆盖率、IoU（或至少 Dice）
3) **在线稳定性**：跑 2 分钟实时相机，观察是否出现漂移/全空/抖动。

### 4.4 推荐落地顺序
1) 先做 computeUnits AB（`.cpuAndGPU` vs `.all`），把“调度路径”结论锁死
2) 再做 P1（compute_precision=FP16，I/O 不变），回归 IoU
3) 如果 embedding cache 命中率高且带宽/拷贝是瓶颈，再考虑 P2（embedding I/O float16）

---

## 5) 我给出的最终推荐（当前信息下）

- **默认（稳定优先）**：enc `.cpuAndGPU` + dec `.cpuAndGPU`
- **性能候选（争取 ANE）**：enc `.all` + dec `.all`（需 AB 通过）
- **FP16**：优先做 **P1**（compute_precision=FP16，接口不变），风险最低、最容易回滚。

