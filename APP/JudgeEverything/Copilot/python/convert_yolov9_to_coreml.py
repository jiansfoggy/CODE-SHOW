"""
Convert YOLOv9 PyTorch model (yolov9-c.pt) to Core ML (.mlmodel)

This script uses torch.jit.trace and coremltools. You may need to adapt input shape and model signature.
"""
import torch
import coremltools as ct
import numpy as np

MODEL_PATH = "models/yolov9-c.pt"
OUTPUT_PATH = "../coreml/yolov9-c.mlmodel"

# Load YOLOv9 model
model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

# Example input (batch_size=1, 3x640x640)
example_input = torch.rand(1, 3, 640, 640)

# Trace model
traced = torch.jit.trace(model, example_input)

# Convert to Core ML
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="input", shape=example_input.shape, scale=1/255.0, bias=[0,0,0])],
)
mlmodel.save(OUTPUT_PATH)
print(f"YOLOv9 Core ML model saved to {OUTPUT_PATH}")
