"""
Convert MobileSAM PyTorch model (mobile_sam.pt) to Core ML (.mlmodel)

This script uses torch.jit.trace and coremltools. You may need to adapt input shape and model signature.
"""
import torch
import coremltools as ct
import numpy as np

MODEL_PATH = "models/mobile_sam.pt"
OUTPUT_PATH = "../coreml/mobile_sam.mlmodel"

# Load MobileSAM model
model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

# Example input (batch_size=1, 3x256x256)
example_input = torch.rand(1, 3, 256, 256)

# Trace model
traced = torch.jit.trace(model, example_input)

# Convert to Core ML
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="input", shape=example_input.shape, scale=1/255.0, bias=[0,0,0])],
)
mlmodel.save(OUTPUT_PATH)
print(f"MobileSAM Core ML model saved to {OUTPUT_PATH}")
