# JudgeE2 — Master Roadmap (Post Phase 1)

Status:
✅ Phase 1 Completed — Stable YOLO Detection Pipeline  
Target Device: iPhone 11  
Architecture Frozen (Camera → YOLO → Decode → NMS → Overlay)

Big Picture:
1. First complete full functional loop (Detection → Segmentation → Interaction → Persistence → UI)
2. Then optimize and polish for production quality

We will divide remaining work into 4 structured 7-Day Phases:

------------------------------------------------------------
PHASE 2 — MobileSAM Real-Time Segmentation Integration
------------------------------------------------------------

Objective:
Integrate MobileSAM instance segmentation on top of YOLO without breaking Phase 1 geometry or scheduling contract.

Principle:
Reuse Phase 1 pipeline. Add modules without rewriting it.

Architecture Additions:
- PromptBuilder
- SAM Encoder (split model preferred)
- SAM Decoder
- TemporalManager (TTL + cache)
- MaskRenderer (overlay)

Must Reuse:
CanonicalFrame
FrameGeometry
LetterboxTransform

------------------------------------------------------------
Phase 2 — Detailed 7-Day Plan
------------------------------------------------------------

Design Rule:
Only assign tasks to agents when required.
Follow agent order:
Architect → ML_Vision → Builder → Debugger

------------------------------------------------------------
Day 1 — Architecture Lock & Integration Contract
------------------------------------------------------------

Architect
- [ ] Define segmentation pipeline insertion point (post-NMS)
- [ ] Freeze geometry reuse contract (NO duplicate transforms)
- [ ] Define prompt format from bbox → SAM input
- [ ] Define encoder/decoder split API
- [ ] Define threading model (background queue only)
- [ ] Define fallback logic (bbox-only mode)

Deliverable:
Approved Phase 2 integration diagram.
No Phase 1 code modified.

------------------------------------------------------------
Day 2 — MobileSAM Model Preparation
------------------------------------------------------------

ML_Vision
- [ ] Convert mobile_sam.pt → ONNX
- [ ] Split encoder / decoder (if supported)
- [ ] Convert to CoreML
- [ ] Confirm input/output tensor names
- [ ] Provide embedding shape
- [ ] Benchmark single-image latency (CPU/GPU)

Debugger
- [ ] Confirm model loads on device
- [ ] Confirm no memory spike

Deliverable:
Working MobileSAM CoreML models load on iPhone 11.

------------------------------------------------------------
Day 3 — Encoder Integration (Low Frequency)
------------------------------------------------------------

Builder
- [ ] Insert Encoder module after detection
- [ ] Run encoder every 12 frames
- [ ] Store embedding cache
- [ ] Log embedding latency

Debugger
- [ ] Confirm embedding TTL works
- [ ] Confirm no main-thread blocking
- [ ] Confirm stable FPS > 15

Deliverable:
Embedding generation working and cached.

------------------------------------------------------------
Day 4 — Decoder Integration (Medium Frequency)
------------------------------------------------------------

Builder
- [ ] Build PromptBuilder (bbox → SAM prompt)
- [ ] Run decoder every 6 frames
- [ ] Implement mask TTL (800ms)
- [ ] Log decode time

Debugger
- [ ] Confirm mask aligns with bbox
- [ ] Confirm no jitter across frames
- [ ] Confirm fallback works

Deliverable:
BBox → Mask working pipeline.

------------------------------------------------------------
Day 5 — Mask Renderer
------------------------------------------------------------

Builder
- [ ] Add mask overlay layer
- [ ] Match geometry to preview layer
- [ ] Add alpha blending
- [ ] Handle rotation (reuse Phase 1 logic)

Debugger
- [ ] Confirm mask not flipped
- [ ] Confirm alignment across orientation
- [ ] Confirm preview smooth

Deliverable:
Real-time mask overlay at 2–5Hz.

------------------------------------------------------------
Day 6 — Temporal Manager
------------------------------------------------------------

Architect
- [ ] Define main-object selection strategy
- [ ] Define bbox drift threshold
- [ ] Define cache invalidation triggers

Builder
- [ ] Implement TTL system
- [ ] Implement drift detection
- [ ] Implement priority mask refresh

Debugger
- [ ] Stress test motion
- [ ] Confirm mask stability during object motion
- [ ] Confirm fallback triggers correctly

Deliverable:
Stable segmentation loop.

------------------------------------------------------------
Day 7 — Stabilization & Phase Freeze
------------------------------------------------------------

Debugger
- [ ] Measure encoder latency
- [ ] Measure decoder latency
- [ ] Measure mask refresh rate
- [ ] Measure FPS (bbox + mask)
- [ ] Record memory usage

Architect
- [ ] Freeze Phase 2 architecture
- [ ] Document integration contracts
- [ ] Define Phase 3 entry points

Deliverable:
Stable Detection + Segmentation pipeline.
Ready for interactive layer.

------------------------------------------------------------
PHASE 3 — User-Triggered Segmentation (Tap-to-Segment)
------------------------------------------------------------

Objective:
Allow segmentation without YOLO (user tap / region select).

Key Additions:
- TouchHandler
- PromptBuilder (point-based)
- Multi-mask support

High-Level Plan:
Day 1–2: Tap coordinate → canonical transform  
Day 3–4: Run SAM using point prompt  
Day 5: Multi-instance selection  
Day 6: Visual feedback + highlight  
Day 7: Stabilization

Result:
User can tap anywhere to segment object.

------------------------------------------------------------
PHASE 4 — Pin, Tag, Annotation System
------------------------------------------------------------

Objective:
Persistent segmentation memory layer.

New Modules:
- PinManager
- AnnotationView
- Local Storage (CoreData or lightweight DB)

Features:
- Add pin on segmented region
- Attach tag
- Attach note
- Save locally
- Reopen and revisit

High-Level Plan:
Day 1–2: Data model + storage
Day 3–4: Pin UI overlay
Day 5: Annotation editor
Day 6: Retrieval logic
Day 7: Stability testing

Result:
Interactive, persistent segmentation memory system.

------------------------------------------------------------
PHASE 4 — Live Mask → Pin → Annotation System
------------------------------------------------------------

Objective:
将瞬态分割结果升级为可持久、可标注的分割记忆层。
分两阶段：先让 mask 成为活跃资源（re-anchor），
再在活跃 mask 上构建 Pin/Annotation 持久化系统。

------------------------------------------------------------
PHASE 4A — Live Mask Foundation（Days 1–3）
------------------------------------------------------------

目标：Phase 3 mask 是"快照"——相机一移动就作废。
Phase 4A 让 mask 持续跟踪物体，为 Pin 提供有意义的锚点。

Day 1 — Phase 3 收尾 + 基础埋点
  - 慢路径 UI 最终裁决（Tier 1 持续脉冲是否需要专属交互语义，
    基于 §29 数据：mean=822.9 ms / p95=915.1 ms 决策）
  - D-7' 六段埋点（CameraManager → decoderQueue → main thread
    各段计时，定位 280–310 ms 未归因残差）
  - tap 锚点编号上屏（补齐 W-1 L3 欠载，两个 secondary
    实例之间加数字标识）

Day 2–3 — Re-anchor 刷新循环（核心基础设施）
  新增：事件驱动漂移检测器
    - 每帧计算当前帧与缓存帧的几何偏移
    - 偏移超过阈值 → 用同一 canonicalPoint 在新 embedding
      上重新 decode → 更新 maskImage（无需用户重新点击）
  结果：tap mask 从"快照"升级为"活跃资源"，
        相机缓慢平移时 mask 保持跟踪物体
  验收：连续走动 3 秒，mask 随目标物体移动，
        不出现位置漂移（允许 ±10px 抖动）

------------------------------------------------------------
PHASE 4B — Pin & Annotation System（Days 4–6）
------------------------------------------------------------

目标：在活跃 mask 上构建持久化标注系统。
（此时 Pin 锚定的是活跃 mask 的 canonicalPoint，
 不是静态截图，重新打开时可重新 decode 恢复。）

New Modules:
  - PinManager        — Pin 生命周期管理，含 canonicalPoint 持久化
  - AnnotationView    — 标注编辑 UI
  - PinStore          — 本地存储（CoreData 或 SQLite，
                        存 canonicalPoint + frameGeometry + tag + note）

Day 4 — 数据模型 + 存储
  - Pin 数据结构：
      struct Pin {
        id: UUID
        canonicalPoint: CGPoint   // SAM 输入坐标（1024×1024）
        frameGeometry: FrameGeometry  // 拍摄时的几何快照
        thumbnail: UIImage        // 128×128 mask 缩略图
        tag: String?
        note: String?
        createdAt: Date
      }
  - PinStore：CoreData 或 JSON-backed store
  - 验收：Pin 写入 / 读取 / 删除 roundtrip 测试通过

Day 5 — Pin 创建 UI
  - 长按 tap anchor marker → 触发"固定此分割"操作
  - 新增 PinCreationSheet（底部弹出）：
      预览缩略图 + 快速标签输入 + 保存按钮
  - 保存后 anchor marker 变为"图钉"图标（区分临时 vs 持久）
  - 验收：Pin 保存后重启 app 仍可见

Day 6 — 标注编辑器 + 检索
  - AnnotationView：全屏编辑器（tag、note、修改、删除）
  - PinList 视图：按时间/标签浏览所有 Pin
  - 重访逻辑：点击 Pin → 加载 frameGeometry → 
    重新 decode canonicalPoint → 恢复 mask（或展示 thumbnail）
  - 验收：10 个 Pin 写入后检索延迟 < 100 ms

------------------------------------------------------------
PHASE 4C — Polish & Expansion（Day 7+）
------------------------------------------------------------

Day 7 — 稳定性测试
  - 50 个 Pin 写入/读取压力测试
  - CoreData 并发写入安全验证（sessionQueue / main thread）
  - Phase 2/3/4 模式切换下 PinManager 内存不泄漏
  - A-1 候选选择改进评估（ML_Vision，需 E-1..E-4 验证门控）

可选扩展（Day 8+，按需）：
  - 双指框选（box prompt 交互模式）
  - 分割结果导出（PNG/JSON，含 mask + canonicalPoint）
  - MobileSAM 模型升级评估（SAM 2 / EfficientSAM）

------------------------------------------------------------
架构约束（Phase 3 持续）
------------------------------------------------------------
- R3 禁令参数不变：minComponentPx=30、cap60、cap85 等
- SAMDecoder.swift / MaskRenderer.swift 继续冻结
- 单 encoder 实例、单几何链、FIFO Pool(max=3) 不变
- Pin 存储不得影响实时推理帧率（CoreData 写入必须异步）

------------------------------------------------------------
Result:
Phase 4A：mask 从快照升级为活跃跟踪资源
Phase 4B：活跃 mask 可被"钉住"、标注、持久化、重访
Phase 4C：系统级稳定性 + 功能扩展
------------------------------------------------------------

------------------------------------------------------------
PHASE 5 — UI Redesign & App Polish
------------------------------------------------------------

Objective:
Transform prototype into product-like app.

Focus:
- Bottom control bar
- Mode switching (Detect / Segment / Annotate)
- Clean animation
- Improved overlay style
- Settings panel

High-Level Plan:
Day 1–3: UI redesign
Day 4–5: UX refinement
Day 6: Performance cleanup
Day 7: Baseline benchmark & Demo build

Result:
Complete functional MVP app.

------------------------------------------------------------
Overall Timeline Summary
------------------------------------------------------------

Phase 2 → Real-time segmentation
Phase 3 → User-triggered segmentation
Phase 4 → Pin + Tag + Persistence
Phase 5 → UI polish

------------------------------------------------------------
Big Picture Strategy
------------------------------------------------------------

Step 1:
Finish all functional loops first.
Correctness > Performance.

Step 2:
Once full workflow exists,
then optimize:
- Quantization
- ANE
- mlprogram
- Frame scheduling
- Memory tuning

This guarantees:
Low rework
Stable iteration
Controlled complexity growth

------------------------------------------------------------
END
------------------------------------------------------------
