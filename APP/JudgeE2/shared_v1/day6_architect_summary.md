# Day 6 — Architect Tasks Completed

I have completed the Day 6 architecture tasks as requested:

1.  **D6-A-ORIENTATION-CONTRACT-V2**: Confirmed "I. Orientation & Mirroring 契约 v2" is present in `shared/architect_output.md` and aligns with the regression requirements for front camera and orientation handling.
2.  **D6-A-SCHEDULING-GUIDE**: Added "J. Scheduling & Contention Guide v1 (Day 6)" to `shared/architect_output.md`, synthesizing findings from Day 6 experiments:
    *   **Default Compute Units**: YOLO `.all`, SAM Encoder `.cpuAndGPU` (stable) / `.all` (perf), SAM Decoder `.cpuAndGPU`.
    *   **Cadence**: Encoder ~12 frames, Decoder ~6 frames.
    *   **Contention Mitigation**: 
        *   **Run Golden**: MUST pause realtime pipeline.
        *   **SAM Encoder Active**: Throttle detector postprocess (skip or reduce frequency) to avoid CPU contention.
    *   **Decode/NMS**: Recommended conservative parameters (`scoreThreshold=0.35`, `preNmsTopK=150`, `topK=50`) to keep steady state <80ms.

Both tasks have been checked off in `shared/tasks.md`.