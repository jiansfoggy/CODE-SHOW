# OpenClaw Agent Bootstrap Script

## 1️⃣ Create Architect Agent

Create agent:
- name: architect
- agent_markdown: /Users/jiansun/.openclaw/agents/architect.md
- workspace: /Users/jiansun/.openclaw/workspace-architect

Initialize context:
- Project: Real-time Video Segmentation iOS App
- Model Weights:
  - YOLOv9: /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/yolov9-c.pt
  - MobileSAM: /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/mobile_sam.pt
- Daily Task File:
  /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/shared/task.md


---

## 2️⃣ Create Builder Agent

Create agent:
- name: builder
- agent_markdown: /Users/jiansun/.openclaw/agents/builder.md
- workspace: /Users/jiansun/.openclaw/workspace-builder

Initialize context:
- Use YOLOv9 weight: /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/yolov9-c.pt
- Use MobileSAM weight: /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/mobile_sam.pt
- Follow daily task file:
  /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/shared/task.md


---

## 3️⃣ Create ML_Vision Agent

Create agent:
- name: ml_vision
- agent_markdown: /Users/jiansun/.openclaw/agents/ml_vision.md
- workspace: /Users/jiansun/.openclaw/workspace-ml_vision

Initialize context:
- YOLOv9 model path:
  /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/yolov9-c.pt
- MobileSAM model path:
  /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/models/mobile_sam.pt
- Task file:
  /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/shared/task.md


---

## 4️⃣ Create Debug_Test Agent

Create agent:
- name: debug_test
- agent_markdown: /Users/jiansun/.openclaw/agents/debug_test.md
- workspace: /Users/jiansun/.openclaw/workspace-debug_test

Initialize context:
- Monitor builder + ml_vision outputs
- Task source:
  /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/shared/task.md


---

## 5️⃣ Global Constraints

All agents must:
- Treat model weights as local files (do not attempt download)
- Assume iOS real-time inference target
- Optimize for low latency + on-device performance
- Follow daily tasks strictly from task.md
- Persist outputs inside their own workspace
- Do NOT modify other agents' workspace

---

## 注册4个agents时要用的的信息

agents:
  - name: architect
    file: /Users/jiansun/.openclaw/agents/architect.md
    workspace: /Users/jiansun/.openclaw/workspace-architect

  - name: builder
    file: /Users/jiansun/.openclaw/agents/builder.md
    workspace: /Users/jiansun/.openclaw/workspace-builder

  - name: ml_vision
    file: /Users/jiansun/.openclaw/agents/ml_vision.md
    workspace: /Users/jiansun/.openclaw/workspace-ml_vision

  - name: debug_test
    file: /Users/jiansun/.openclaw/agents/debug_test.md
    workspace: /Users/jiansun/.openclaw/workspace-debug_test
