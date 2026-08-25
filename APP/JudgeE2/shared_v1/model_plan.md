# ML_Vision — Model Export & Performance Plan (Day 1)

Project: JudgeE2 / JudgeEverything (rebuild)
Target device: iPhone 11 (A13)
Detector: YOLO‑v9
Segmenter: MobileSAM
Weights dir (canonical): `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models`

> 目标：给出可执行的导出/性能路线图 + 交付物契约；不涉及 Swift/UI 实现。

---

## 1) YOLO‑v9 → CoreML 导出计划

### 1.1 依赖与环境
- Python env：建议独立导出环境（torch + coremltools + onnx），与 iOS 工程隔离。
- 产物放置：`/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/`

### 1.2 导出路径（v0 bring‑up）
**路径 A（推荐）**
1) PyTorch → ONNX（可选，仅用于对齐/调试）
2) PyTorch → CoreML（coremltools）
3) 先产出 `.mlmodel`（neuralnetwork 格式）确保“能转 + 能加载”
4) Day2+ 再评估 `.mlpackage` / mlprogram

**输出契约（v0 预期）**
- 输入：`image`，RGB 640×640，内置 scale=1/255（或外部预处理 + scale=1）
- 输出：raw head（优先），便于 Swift 侧 decode/NMS
- output keys/shape 将以 CoreML 导出结果为准，并写入本文件 v1

### 1.3 Day1 交付物（计划）
- `models/yolov9-*.mlmodel`（可加载即可）
- model card（输入/输出/坐标/归一化）
- golden test：`shared/golden/bus.jpg` + raw 输出统计 JSON

---

## 2) MobileSAM 导出/封装计划

### 2.1 推荐结构（Option A — split）
- **Image Encoder**：`image` (1024×1024) → `image_embeddings` (1,256,64,64)
- **Prompt+Mask Decoder**：`image_embeddings` + box prompt → `low_res_masks` (1,1,256,256)

理由：embedding 可缓存，性能可控；适合 iPhone 11。

### 2.2 备选结构（Option B — monolithic）
- `image + box` → `mask`（只用于导出困难时的临时过渡）

### 2.3 预处理与坐标约定
- MobileSAM 输入：将原图最长边缩放至 1024，再 pad 到 1024×1024（右/下 padding）
- bbox prompt：来自 Canonical camera px → 映射到 1024 输入空间
- mask 输出：low‑res 256×256 logits，Swift/Metal 上采样后再映射回 Canonical camera px

---

## 3) 量化与性能路线（Day1 规划）

### 3.1 YOLO‑v9
- v0：FP16 权重（如可行），输出仍 float32
- v1：探索 `.mlpackage` / mlprogram（iOS 17+），优先走 ANE
- INT8：延后，需校准集 + 基线稳定后再做

### 3.2 MobileSAM
- v0：float32 I/O（CoreML 兼容优先）
- v1：尝试 compute_precision=FP16（保持 I/O float32），观察质量/速度
- computeUnits：bring‑up 先 `.cpuAndGPU`，稳定后再尝试 `.all`

---

## 4) YOLO 输出 → MobileSAM prompt 接口

**Canonical 坐标：camera frame px**
1) YOLO raw head decode → xyxy (model input space)
2) 通过 letterbox 逆变换映射回 Canonical camera px
3) 选 top‑K（默认 top‑1 供分割）
4) 将 Canonical bbox 映射到 MobileSAM 1024×1024 输入空间
5) 作为 box prompt：`point_coords = [[x0,y0],[x1,y1]]`, `labels = [2,3]`

---

## 5) Day1 风险与对策

- CoreML 导出失败（op 不支持） → 先落地 `.mlmodel` / neuralnetwork，必要时精简 graph
- 输出 key/shape 不稳定 → 以 model card 固化；iOS 侧只消费固定 key
- 预处理不一致（RGB/BGR、scale） → golden test + 统计对齐

---

## 6) Day1 结论 / Next Steps

1) 先导出 YOLO‑v9 CoreML v0（可加载/可推理）
2) 固化 I/O 契约与 golden test
3) MobileSAM 先走 split 结构；若导出受阻再 fallback
4) Day2 开始补充：具体输出 key/shape + decode 规则

---

# Day 2 — ML_Vision 交付（已完成）

## 2.1 YOLOv9 CoreML draft 产物
- **CoreML 模型**：`/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/yolov9-c.mlmodel`
- **格式**：`.mlmodel`（neuralnetwork）
- **最低 iOS**：iOS 14（导出时 minimum_deployment_target=iOS14）
- **computeUnits（bring-up）**：`.all`（若遇 ANE 兼容问题再试 `.cpuAndGPU`）
- 导出方式（参考）：`models/yolov9/export.py --include coreml --half`
- 目标：先保证 **可加载/可推理**（ANE 最优非优先）

## 2.2 明确 I/O 契约（交付 Builder/Debugger）
**Input**
- name: `image`
- shape: **1×3×640×640**（NCHW）
- type: Image (CoreML ImageType)
- preprocess: **RGB**, scale=**1/255.0**（模型内置）
- app 侧建议：**letterbox** 到 640×640（几何见 `shared/architect_output.md`）

**Outputs（raw head）**
- **Primary**：`var_3019`
  - dtype: **float32**
  - shape: **(1, 84, 8400)**
- Secondary：`var_3022`（同 shape，用于对比/暂不用于 decode）
- bbox format: **xywh(center)**
- 说明：84 = 4 + 80（COCO），**无 objectness 分支**，cls 已 **sigmoid**

## 2.3 Golden Test + Reference Decoder
- golden image：`shared/golden/bus.jpg`
- expected raw output：`shared/golden/expected_bus_yolov9c_raw.json`
- reference decoder：`shared/yolov9_reference_decoder.py`
  - 默认参数：score=0.25 / iou=0.65 / topK=100
  - 作用：用于 Swift 侧 smoke test 对齐 PyTorch 输出范围

---

## Appendix — 预期产物清单（后续落地时更新）
- `models/yolov9-*.mlmodel` / `.mlpackage`
- `models/mobilesam_*` encoder/decoder
- `shared/golden/*`
- `shared/yolov9_reference_decoder.py`
