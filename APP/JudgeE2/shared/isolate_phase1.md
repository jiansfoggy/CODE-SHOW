# JudgeE2 — Phase 2 Isolation & Safety Protocol
Status: Phase 1 Frozen ✅  
Target: Safe Launch of Phase 2 (Segmentation Integration)  
Device: iPhone 11  

============================================================
🎯 OBJECTIVE
============================================================

Ensure Phase 2 development does NOT break or contaminate the
stable Phase 1 detection pipeline.

Principle:
Isolation first.
Integration second.
Optimization last.

Phase 1 must always remain runnable, measurable, and recoverable.

============================================================
🛡️ LAYER 1 — GIT ISOLATION (MANDATORY)
============================================================

1️⃣ Create dedicated Phase 2 branch:

    git checkout -b phase2-segmentation
    git push -u origin phase2-segmentation

Branch Policy:
- main → Phase 1 stable
- phase2-segmentation → experimental work only

2️⃣ Tag Phase 1 stability point:

    git tag phase1-stable
    git push origin phase1-stable

Rollback Guarantee:

    git checkout main
    OR
    git checkout phase1-stable

This guarantees instant restoration of Phase 1.

============================================================
🛡️ LAYER 2 — FEATURE FLAG ISOLATION (MANDATORY)
============================================================

Introduce runtime mode switching.

Example:

------------------------------------------------------------
enum AppMode {
    case detectionOnly
    case segmentation
}

var currentMode: AppMode = .detectionOnly
------------------------------------------------------------

Pipeline rule:

------------------------------------------------------------
runDetectionPipeline()

if currentMode == .segmentation {
    runSegmentationPipeline(using: detections)
}
------------------------------------------------------------

Rules:
- Default mode = detectionOnly
- Segmentation must NEVER override detection pipeline
- Switching mode must require no code deletion

This ensures Phase 1 always runnable even inside Phase 2 branch.

============================================================
🛡️ LAYER 3 — MODULE DIRECTORY ISOLATION
============================================================

Required project structure:

------------------------------------------------------------
JudgeE2/
│
├── Detection/        (PHASE 1 — FROZEN)
│   ├── Camera
│   ├── YOLO
│   ├── Decode
│   ├── NMS
│   ├── Overlay
│
├── Segmentation/     (PHASE 2 — NEW)
│   ├── PromptBuilder
│   ├── SAMEncoder
│   ├── SAMDecoder
│   ├── TemporalManager
│   ├── MaskRenderer
│
├── Shared/
│   ├── CanonicalFrame
│   ├── FrameGeometry
│   ├── LetterboxTransform
│
└── UI/
------------------------------------------------------------

CRITICAL RULE:
Segmentation may read Detection output.
Segmentation may NOT modify Detection internal logic.

============================================================
🛡️ LAYER 4 — CONTRACT FREEZE (ARCHITECT RULES)
============================================================

The following modules are FROZEN:

- CanonicalFrame
- FrameGeometry
- LetterboxTransform
- Detection pipeline core
- YOLO decode + NMS logic
- Bounding box overlay mapping logic

Phase 2 may:
- Insert after NMS
- Consume detections
- Render mask overlay on separate layer

Phase 2 may NOT:
- Rewrite geometry transforms
- Change tensor shapes of detection
- Modify detection scheduling
- Block main thread
- Change inference threading model

============================================================
🛡️ LAYER 5 — PIPELINE ENTRY PRESERVATION
============================================================

Detection entry must remain intact:

------------------------------------------------------------
func runDetectionPipeline()
------------------------------------------------------------

Segmentation must be independent:

------------------------------------------------------------
func runSegmentationPipeline(using detections: [Detection])
------------------------------------------------------------

Never merge these two into a single monolithic function.

============================================================
🛡️ LAYER 6 — FAILSAFE DESIGN
============================================================

If segmentation fails:
- Timeout
- Encoder busy
- Memory spike
- FPS drop below threshold

System must automatically fallback to:

Detection-only mode

Old mask may remain visible within TTL window.

App must NEVER crash because of segmentation.

============================================================
🛡️ LAYER 7 — PERFORMANCE PROTECTION
============================================================

Minimum guarantees during Phase 2:

- Detection FPS ≥ 15
- Mask refresh rate 2–5 Hz
- No main-thread blocking
- Preview must remain smooth
- Memory must not spike beyond safe range

If these conditions are violated:
Segmentation must be disabled.

============================================================
⚠️ DO NOT DO
============================================================

❌ Do NOT duplicate the entire project.
❌ Do NOT delete Phase 1 files.
❌ Do NOT rewrite geometry.
❌ Do NOT merge detection + segmentation logic.
❌ Do NOT optimize before correctness is verified.
❌ Do NOT enable ANE prematurely.

============================================================
🧠 AGENT RESPONSIBILITY ALIGNMENT
============================================================

Architect:
- Enforce contract freeze
- Approve insertion points
- Prevent geometry duplication

ML_Vision:
- Provide model conversion
- Benchmark encoder/decoder latency
- Ensure tensor contracts correct

Builder:
- Implement segmentation modules
- Maintain separation of concerns
- Respect feature flag

Debugger:
- Monitor FPS
- Monitor memory
- Validate fallback behavior
- Stress test rotation and motion

============================================================
🎯 PHASE 2 ENTRY CHECKLIST
============================================================

Before writing segmentation code:

- [ ] phase2-segmentation branch created
- [ ] phase1-stable tag created
- [ ] Feature flag added
- [ ] Detection pipeline confirmed runnable
- [ ] Segmentation directory created
- [ ] Architect freeze rules acknowledged

Only after all boxes checked:
Phase 2 implementation may begin.

============================================================
BIG PICTURE REMINDER
============================================================

Step 1:
Build complete functional loop.

Step 2:
Stabilize.

Step 3:
Optimize.

Never reverse this order.

============================================================
END OF PROTOCOL
============================================================
