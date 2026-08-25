# Day6 — D6-M-INMODEL-POSTPROC-OPTION

任务：评估“将部分 YOLO 后处理下沉到模型侧”的可行性（例如 topk / 简化输出），并给出是否值得做、预计收益/风险（CoreML 支持度、导出复杂度）。

结论（先给）：**短期不建议立刻做“完整 in-model NMS / TopK”**；更推荐先把 Swift 侧 decode+nms 做到“低风险降本”（阈值/TopK/内存分配/向量化），只有当它仍是系统瓶颈且你愿意接受“固定 K 输出 + 更高导出复杂度 + 算子覆盖不确定性”时，再推进“端到端（含 NMS）模型”。

---

## 0) 背景：为什么会想下沉？
现状（iPhone 11 / iOS 17）
- YOLO infer：~170–225 ms
- **decode+nms：~115–155 ms（稳定 CPU 热点）**

因此把后处理下沉到模型侧的潜在收益主要是：
- 降低 CPU decode 时间、减轻主线程/CPU 争用，减少 outlier（例如并发时 decode 飙高）。

---

## 1) 可以下沉哪些后处理？（按“可行性/风险”分层）

### Layer A（低风险、但收益有限）— 简化输出形态
目标：降低 Swift 扫描/argmax 的成本。

**A1. 输出 per-anchor 的 maxScore + classId（替代 80 类扫描）**
- 想法：在模型里做 `max(cls[80])` 和 `argmax(cls[80])`，输出：
  - boxes: (1,4,8400)
  - score: (1,1,8400)
  - classId: (1,1,8400) int
- 预计收益：Swift 从“8400×80 扫描”变为“8400 扫描”，理论上可省一部分 CPU。
- 主要风险：
  - `argmax`/`topk`/`sort` 这类 op 在 CoreML（特别是 mlprogram）上**更容易踩算子覆盖/性能坑**，且 int 输出会带来额外处理。
  - 即便 max/argmax 成功，**NMS 仍在 Swift**，总体 decode+nms 仍可能大头在 NMS。

**A2. Class allowlist（只保留少数类）**
- 想法：在模型侧只输出若干目标类（例如 person/vehicle），从根源上减少 80 类扫描。
- 现实问题：需要改 head/导出图，且会丧失通用性；场景不确定时不推荐。

> 对应建议：如果你们的产品场景允许“只要少数类”，这个比 in-model topk 更稳、更能确定收益。

---

### Layer B（中收益、中风险）— 候选筛选（pre-filter）下沉
目标：减少 NMS 输入候选数量。

**B1. threshold mask + topK candidates（固定 K）**
- 想法：在模型里做阈值过滤，再做 TopK，输出固定 K 个候选（padding）。
- 预计收益：NMS 输入规模显著下降（例如从几百到 K=50/100），Swift NMS 成本下降。
- 主要风险：
  - `topk`/`sort` 相关算子在 CoreML 转换链上不稳定；
  - “阈值/TopK”变成模型的一部分，**调参不再灵活**（每次改阈值都要换模型，或把阈值做成输入但这又增加图复杂度/兼容性风险）。

---

### Layer C（高收益、高风险）— 完整 NMS 下沉（端到端检测）
目标：模型直接输出最终 detections（固定 K 的 boxes/scores/classes）。

**C1. 导出 end2end（含 NMS）的 CoreML**
- 路径：PyTorch/ONNX →（含 NMS op）→ CoreML
- 预计收益：
  - Swift 侧 decode+nms 理论上可接近 0（只做后续坐标映射/渲染）。
  - 对“CPU 争用导致 decode 峰值”更友好。
- 关键风险（也是我不建议现在就做的原因）：
  1) **CoreML 支持度不确定**：
     - NMS 相关算子在不同 CoreML format（neuralnetwork vs mlprogram）、不同 iOS 版本上覆盖差异大。
     - 很多情况下 NMS 会落在 CPU/GPU，未必走 ANE；甚至可能比 Swift 实现更慢。
  2) **输出形态必须固定 K**：CoreML 对“可变数量 detections”支持有限，常见做法是固定 topK 并 padding。
  3) **调参/Debug 变困难**：
     - Swift 侧 NMS/threshold/topK 目前是可热调的；下沉后每次调整需要重新导出模型。
     - Debug 时看不到 raw head，定位错检/漏检更难。
  4) **导出复杂度显著上升**：转换失败/数值差异/算子 fallback 都会提高迭代成本。

---

## 2) 预计收益（给一个“工程上可预期”的量级）

如果做“完整下沉（含 NMS）”且运行时路径靠谱：
- decode+nms 从 ~115–155ms → **可能下降到 <20ms（只剩 mapping + 少量拷贝）**

但现实中常见情况是：
- in-model NMS 不走 ANE、或引入额外开销，最后收益不稳定；
- 或者导出/兼容性成本高，拖慢整体迭代。

因此**收益上限很诱人，但确定性较差**。

---

## 3) 推荐决策

### 建议：先不做（当前阶段）
理由：
- 目前项目 Day6 的主目标是 cold start / mlprogram 稳定性与 ANE 路径验证；
- “in-model 后处理”会显著扩大变量（算子覆盖、iOS 版本、性能路径），不利于收敛。

### 推荐优先级（更稳的降本路径）
1) **Swift decode 优化（低风险，高确定性）**
   - 调 `scoreThreshold` / `preNmsTopK` / `topK`
   - 减少内存分配（按 class 固定 bucket，避免 dict/频繁 append）
   - 尝试 class-agnostic NMS（若可接受）
   - 尝试 Accelerate/vDSP 做 per-anchor max（如内存 layout 允许）
2) 若仍不够，再评估 C1（端到端 NMS）
   - 前提：你能接受 **固定 K 输出** 与阈值固化（或重新导出频繁）。

---

## 4) 如果你决定“值得做”，我建议怎么落地（最小风险方案）

**方案 C1-min（最小可验证版本）**
- 只做：`scoreThreshold` + `topK`（固定 K=100） + class-agnostic NMS
- 输出：
  - `boxes` (1,K,4) xyxy
  - `scores` (1,K)
  - `classes` (1,K) int32
- 目标：先在 iPhone 11 上验证
  - decode CPU 是否显著下降
  - 首次 load / warmup 是否被明显拉长
  - computeUnits `.all` 下是否更稳定走期望的硬件路径

> 注意：该方案会改变 I/O（输出不再是 (1,84,8400) raw head）。因此需要一个新的 model card，并同步 Swift 侧适配。

---

## 5) 总结一句话
- **现在值得做的：Swift 侧 decode+nms 降本（确定性强）**
- **后面可能值得做的：端到端 NMS（收益上限高，但 CoreML 支持/性能路径不确定、导出复杂）**
