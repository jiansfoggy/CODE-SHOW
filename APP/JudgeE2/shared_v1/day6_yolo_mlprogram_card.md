# Day6 — YOLOv9 `mlprogram` (.mlpackage) Model Card

Task: **D6-M-EXPORT-MLPROGRAM**

目标：导出 YOLO 的 **mlprogram (`.mlpackage`)** 版本，并与当前 **neuralnetwork (`.mlmodel`)** 对比，验证是否显著降低 cold start（以及是否更稳定走 ANE/缓存）。

---

## 1) Artifacts（产物路径）

### Baseline（已存在）
- **NeuralNetwork**: `models/yolov9-c.mlmodel`
  - type: `neuralNetwork`
  - specVersion: **4**
  - Size: ~97MB

### New（Day6 新增 - 已生成）
- **ML Program**: `models/yolov9-c-raw-mlprogram.mlpackage`
  - type: `mlProgram` (**mlprogram**)
  - specVersion: **8** (CoreML 7 tools)
  - Size: **98MB**

---

## 2) I/O Contract（I/O 是否变化）

结论：**I/O 契约不变**（对 Swift 接线来说应当是 drop-in 替换，decode 逻辑不需要改）。

### Input
- name: `image`
- type: **CoreML ImageType**
- size: **640×640**
- colorspace: RGB
- embedded preprocess: `scale = 1/255`，bias = `[0,0,0]`

### Outputs
两者均输出 2 个 tensor（与旧 `.mlmodel` 一致）：
- `var_3019`: `MultiArray<float>` shape = **(1, 84, 8400)**
- `var_3022`: `MultiArray<float>` shape = **(1, 84, 8400)**

Decoder 侧（沿用既有约定）：
- 84 = 4(box) + 80(class)
- 推荐仍以 `var_3019` 为主（此前已验证其更接近 PyTorch 输出）。

---

## 3) Minimum iOS / Deployment Target（最低 iOS）

- **NeuralNetwork**: 兼容性好，通常 iOS 13/14+。
- **ML Program**: 本次导出明确指定 **iOS 17** (`minimum_deployment_target=iOS17`)。
  - 原因：为了最大化 ANE 算子支持与编译器优化（iOS 17 CoreML compiler 改进）。
  - 注意：App 工程 Deployment Target 必须 >= iOS 17 才能加载此模型。

---

## 4) Known Limitations / Notes（已知限制/注意事项）

1.  **首次加载/首次推理的成本来源不同**
    - mlprogram 可能会在首次运行时触发更明显的“编译/特化/缓存”行为；是否更快需要以真机 AB 测量为准。

2.  **ANE 路径并非仅由模型格式决定**
    - 走 ANE 取决于 op 覆盖、computeUnits、以及运行时调度。
    - 建议 AB 测试时固定：`MLModelConfiguration.computeUnits = .all`（以及可选 `.cpuAndGPU` 作为对照）。

3.  **输出张量名/shape 已固定，但 dtype/精度可能影响数值**
    - 本 `.mlpackage` 使用 `compute_precision=FP16` 导出；输出仍是 float32 multi-array。
    - decode/NMS 阈值如果在边界上，可能需要微调（通常不需要）。

---

## 5) Cold Start AB Test Checklist（建议验证方式）

为了回答“是否显著降低 cold start / 更稳定走 ANE/缓存”，建议在 iPhone 11 上做如下 AB：

- A（baseline）：`yolov9-c.mlmodel`
- B（new）：`yolov9-c-raw-mlprogram.mlpackage`

每个模型分别测：
1.  `MLModel(contentsOf:)` / `load` 耗时（首次 vs 二次）
2.  第 1 次 `prediction` 耗时（首次 vs 二次）
3.  steady-state 推理耗时（例如后 30 帧均值/分位数）

操作建议：
- “真正 cold start”需要：杀进程、必要时重启手机/等待系统回收缓存；至少保证 App 不在后台。
- 同时记录：computeUnits、iOS 版本、是否开启低电量模式。

---

## 6) What’s next（如果要继续推进）

- 如果 mlprogram 版本表现更好：下一步可以考虑进一步做
  - `compute_precision` 固化（fp16/fp32）
  - weight quantization（最后再做，先稳定 cold start + correctness）
