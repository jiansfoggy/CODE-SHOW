# JudgeE2 — Phase 3 Isolation & Safety Protocol
Status: Phase 2 Frozen ✅
Target: Safe Launch of Phase 3 (Tap-to-Segment)
Device: iPhone 11

============================================================
🎯 OBJECTIVE
============================================================

Ensure Phase 3 development does NOT break or contaminate the
stable Phase 2 Detection + Segmentation pipeline.

Principle:
Isolation first.
Integration second.
Optimization last.

Phase 1 and Phase 2 must always remain runnable,
measurable, and recoverable.

============================================================
📊 PHASE 2 FROZEN BASELINES (REFERENCE — DO NOT CHANGE)
============================================================

These are real-device measurements on iPhone 11 (Phase 2 Day 7).
All Phase 3 work must preserve these metrics or improve them.

| Parameter               | Phase 2 Frozen Value                          |
|-------------------------|-----------------------------------------------|
| Encoder cadence         | Every 12 frames (or geometry / heavy drift)   |
| Decoder cadence         | Every 2 frames (or light drift)               |
| Embedding TTL           | 8000 ms                                       |
| Mask TTL                | 2000 ms                                       |
| Class switch hysteresis | 6 consecutive frames                          |
| Heavy drift threshold   | IoU < 0.10 → re-encode                        |
| Light drift threshold   | IoU < 0.55 → re-decode                        |
| Encoder input res       | 1024 × 1024                                   |
| Encoder latency (mean)  | 857 ms (stable-state); 2941 ms cold start     |
| Decoder latency (mean)  | 61 ms (stable-state);  1488 ms cold start     |
| Mask refresh rate       | ~1.5 Hz stable; ~2.8 Hz on drift trigger      |
| Pipeline FPS            | 2.7–2.9 FPS (bbox + mask)                     |
| Memory (normal)         | 244–320 MB                                    |
| Memory (peak)           | 339 MB (dual SAM models loaded)               |

These values are the production contract.
Phase 3 must not cause any metric to regress.

============================================================
🛡️ LAYER 1 — GIT ISOLATION (MANDATORY)
============================================================

1️⃣ Create dedicated Phase 3 branch:

    git checkout -b phase3-tap-segment
    git push -u origin phase3-tap-segment

Branch Policy:
- main              → Phase 1 stable
- phase2-segmentation → Phase 2 frozen (do not commit to this)
- phase3-tap-segment  → experimental work only

2️⃣ Tag Phase 2 stability point:

    git tag phase2-stable
    git push origin phase2-stable

Rollback Guarantee:

    git checkout main
    OR
    git checkout phase2-stable

This guarantees instant restoration of Phase 2 full pipeline.

============================================================
🛡️ LAYER 2 — FEATURE FLAG ISOLATION (MANDATORY)
============================================================

Extend AppMode with a new tap-to-segment case.

------------------------------------------------------------
enum AppMode {
    case detectionOnly      // Phase 1
    case segmentation       // Phase 2 (auto YOLO-driven)
    case tapToSegment       // Phase 3 (user-driven)
}
------------------------------------------------------------

Pipeline rule:

------------------------------------------------------------
runDetectionPipeline()                          // always runs

switch currentMode {
case .segmentation:
    runSegmentationPipeline(using: detections)  // Phase 2 path
case .tapToSegment:
    runTapSegmentationPipeline()                // Phase 3 path
default:
    break
}
------------------------------------------------------------

Rules:
- Default mode = detectionOnly
- Phase 2 (.segmentation) must be fully functional at all times
- Phase 3 (.tapToSegment) must never interfere with Phase 2 path
- Mode switching must require no code deletion
- Both .segmentation and .tapToSegment share the same
  SAMEncoder / SAMDecoder; they do NOT load duplicate models

This ensures Phase 1 and Phase 2 always runnable,
even inside Phase 3 branch.

============================================================
🛡️ LAYER 3 — MODULE DIRECTORY ISOLATION
============================================================

Required project structure:

------------------------------------------------------------
JudgeE2/
│
├── Detection/          (PHASE 1 — FROZEN)
│   ├── CameraManager
│   ├── ModelLoader
│   ├── CameraPreview
│   ├── PerfLogger
│
├── Segmentation/       (PHASE 2 — FROZEN)
│   ├── PromptBuilder
│   ├── SAMEncoder
│   ├── SAMDecoder
│   ├── TemporalManager
│   ├── MaskRenderer
│
├── Interaction/        (PHASE 3 — NEW)
│   ├── TouchHandler
│   ├── PointPromptBuilder
│   ├── TapInstanceManager
│   ├── TapFeedbackAnimator
│
├── Shared/
│   ├── AppMode          (extend with .tapToSegment)
│   ├── InferenceBackend
│
└── UI/
    ├── ContentView
    └── JudgeE2App
------------------------------------------------------------

CRITICAL RULE:
Interaction/ may READ from Segmentation/ outputs.
Interaction/ may CALL SAMEncoder.encode() and SAMDecoder.decode().
Interaction/ may NOT modify any file inside Segmentation/.
Interaction/ may NOT modify any file inside Detection/.

============================================================
🛡️ LAYER 4 — CONTRACT FREEZE (ARCHITECT RULES)
============================================================

The following modules are FROZEN for Phase 3:

PHASE 1 FROZEN (unchanged since Phase 1):
- CanonicalFrame
- FrameGeometry
- LetterboxTransform
- CameraManager (detection scheduling)
- YOLO decode + NMS logic
- Bounding box overlay mapping

PHASE 2 FROZEN (newly frozen for Phase 3):
- SAMEncoder.encode(pixelBuffer:) — signature locked
- SAMDecoder.decode(embedding:bbox:) — existing overload locked
- TemporalManager (primary selection, drift detection, cache logic)
- MaskRenderer (Phase 2 auto-segmentation rendering path)
- Embedding TTL / Mask TTL / cadence parameters (see table above)

Phase 3 MAY:
- Add new overload: SAMDecoder.decode(embedding:point:)
  → This is an ADDITIVE change only; existing decode(embedding:bbox:)
    must remain unchanged and fully functional
- Read SAMEncoder's cached embedding when TTL is valid (shared cache)
- Add new AppMode case .tapToSegment to AppMode.swift
- Create all new modules inside Interaction/ directory
- Add tap overlay / visual feedback on a separate CALayer
- Add TapInstanceManager with its own instance pool (max N=3)

Phase 3 MAY NOT:
- Rewrite or modify SAMEncoder's encode() body
- Change TemporalManager's scheduling logic or drift thresholds
- Modify the existing decode(embedding:bbox:) method signature or body
- Change embedding TTL, mask TTL, encoder cadence, decoder cadence
- Rewrite FrameGeometry or LetterboxTransform
- Add geometry transforms outside the existing geometry chain
- Merge Detection + Segmentation + Interaction into a monolithic function
- Block the main thread from any Interaction/ code
- Change YOLO inference threading model
- Load a second copy of SAMEncoder / SAMDecoder

============================================================
🛡️ LAYER 5 — PIPELINE ENTRY PRESERVATION
============================================================

Phase 1 entry must remain intact:

------------------------------------------------------------
func runDetectionPipeline()
------------------------------------------------------------

Phase 2 auto-segmentation entry must remain intact:

------------------------------------------------------------
func runSegmentationPipeline(using detections: [Detection])
------------------------------------------------------------

Phase 3 adds an independent tap-driven entry:

------------------------------------------------------------
func runTapSegmentationPipeline()
------------------------------------------------------------

Never merge these three into a single monolithic function.

Encoder sharing rule:
Both runSegmentationPipeline() and runTapSegmentationPipeline()
may share the same SAMEncoder instance and its embedding cache.
They must use the same encoderQueue.
Concurrent encoder calls are forbidden.

============================================================
🛡️ LAYER 6 — COORDINATE TRANSFORM CONTRACT
============================================================

Tap coordinate transform chain (Phase 3 specific):

------------------------------------------------------------
UIKit tap (preview layer CGPoint)
    ↓
AVCaptureVideoPreviewLayer reverse-map
(aspectFill crop offset correction)
    ↓
LetterboxTransform.invertLetterbox(point:)   ← reuse Phase 1/2
    ↓
CanonicalFrame pixel space (CGPoint)         ← same space as bbox
    ↓
PointPromptBuilder.buildPrompt(canonicalPoint:imageSize:)
    ↓
SAMDecoder.decode(embedding:point:)
------------------------------------------------------------

Rules:
- Tap reverse-transform MUST reuse FrameGeometry +
  LetterboxTransform. No independent coordinate system.
- Normalization (÷ 1024.0) MUST happen inside PointPromptBuilder,
  NOT in the caller.
- Orientation and mirroring MUST be applied during reverse-transform,
  consistent with Phase 1/2 geometry.
- Out-of-bounds tap points (after reverse-map) MUST be clamped to
  [0, W-1] × [0, H-1]; do NOT reject the tap.
- Log required on every tap:
  [TAP] preview=(x,y) canonical=(cx,cy) orientation=N mirrored=B

============================================================
🛡️ LAYER 7 — FAILSAFE DESIGN
============================================================

If tap segmentation fails for any reason:
- Encoder busy / embedding not ready
- Decoder timeout
- Memory spike
- iou_pred < 0.1 (invalid mask)
- FPS drop below threshold

System must:
- Show tap ripple animation as "loading" indicator
- NOT crash
- NOT freeze UI
- NOT affect Phase 2 auto-segmentation pipeline
- Automatically retry when encoder becomes idle

If Phase 2 (.segmentation mode) is active when a tap occurs:
- Tap must be ignored (no cross-mode interference)
- Mode switch (.tapToSegment) must clear all auto masks first

App must NEVER crash because of Interaction/ code.

============================================================
🛡️ LAYER 8 — PERFORMANCE PROTECTION
============================================================

Phase 3 must not regress Phase 2 baselines:

Minimum guarantees during Phase 3 development:
- Detection FPS ≥ 15 (bbox-only path)
- Phase 2 auto-segmentation FPS ≥ 2.7 (when in .segmentation mode)
- Phase 3 tap-to-mask latency measured and logged
- Memory must not exceed Phase 2 peak (339 MB) + 30 MB overhead
  → Hard limit: 370 MB for 3 simultaneous tap instances
- No main-thread blocking from Interaction/ modules
- Preview must remain smooth (no frame drops in camera preview)
- TapInstanceManager max instances = 3 (FIFO eviction enforced)

If any of the above is violated:
Tap segmentation must be disabled (fallback to detection-only).

Phase 3 optimization targets (do NOT skip Phase 2 baseline first):
- ANE alignment fix (encoder ~857ms → ~600-700ms target)
- Encoder resolution reduction (1024→768 AB test, if precision OK)
- Embedding cache reuse rate > 80%

============================================================
⚠️ DO NOT DO
============================================================

❌ Do NOT duplicate SAMEncoder or SAMDecoder.
❌ Do NOT load a second copy of SAM models (memory would spike).
❌ Do NOT modify TemporalManager for Phase 2 auto-segmentation.
❌ Do NOT change the existing decode(embedding:bbox:) signature.
❌ Do NOT add coordinate normalization outside PointPromptBuilder.
❌ Do NOT merge tap and YOLO segmentation paths.
❌ Do NOT block main thread from TouchHandler or TapInstanceManager.
❌ Do NOT optimize before single-tap functional loop is verified.
❌ Do NOT enable multi-instance before single-instance works.
❌ Do NOT run encoder twice simultaneously (two queues = wrong).

============================================================
🧠 AGENT RESPONSIBILITY ALIGNMENT
============================================================

Architect:
- Enforce Phase 2 contract freeze
- Approve tap coordinate transform chain
- Approve SAMDecoder additive overload design
- Prevent geometry duplication or Phase 2 contamination
- Freeze Phase 3 architecture on Day 7

ML_Vision:
- Provide SAM point-prompt tensor spec (shape + labels)
- Provide ANE alignment fix for Encoder mlpackage
- Evaluate encoder resolution reduction (1024 vs 768 vs 512)
- Confirm multi-mask output semantics from decoder

Builder:
- Implement all Phase 3 modules inside Interaction/
- Add decode(embedding:point:) as additive overload only
- Reuse Phase 1/2 geometry chain without modification
- Maintain TapInstanceManager FIFO eviction (N=3 max)
- Respect encoderQueue sharing contract

Debugger:
- Verify tap canonical coordinate accuracy (< 5% error)
- Measure tap-to-mask end-to-end latency (mean + p95)
- Monitor memory during 3-instance simultaneous activation
- Confirm Phase 2 FPS does not regress after Phase 3 additions
- Stress test: fast taps, boundary taps, rotation + tap, mode switch

============================================================
🎯 PHASE 3 ENTRY CHECKLIST
============================================================

Before writing Interaction/ code:

- [x] phase3-tap-segment branch created
- [x] phase2-stable tag created
- [x] AppMode extended with .tapToSegment (additive only)
- [x] Interaction/ directory created
- [x] Phase 2 pipeline confirmed still runnable on device
- [x] SAMDecoder additive overload design reviewed by Architect
- [x] Coordinate transform chain contract acknowledged
- [x] Agent responsibility alignment understood

Only after all boxes checked:
Phase 3 implementation may begin.

============================================================
BIG PICTURE REMINDER
============================================================

Phase 1: YOLO detection          ✅ FROZEN
Phase 2: SAM auto-segmentation   ✅ FROZEN
Phase 3: Tap-to-segment          🚧 IN PROGRESS
Phase 4: Pin + Tag + Persistence ⬜ PLANNED
Phase 5: UI Polish + App MVP     ⬜ PLANNED

Step 1:
Build complete functional loop.
(Single tap → mask displayed)

Step 2:
Stabilize and verify no Phase 2 regression.

Step 3:
Multi-instance, visual feedback, Phase 3 freeze.

Never reverse this order.

============================================================
END OF PROTOCOL
============================================================
