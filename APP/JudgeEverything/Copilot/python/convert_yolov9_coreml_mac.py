"""
Mac conversion helper for yolov9. Adjust REPO_ROOT to where you cloned yolov9.

Run on macOS with a venv that has torch==2.7.0 and coremltools==6.3 installed.
"""
import sys
from pathlib import Path
import torch
import coremltools as ct

REPO_ROOT = Path.home() / 'Projects' / 'yolov9'  # <- change this to your clone path
sys.path.insert(0, str(REPO_ROOT))

WEIGHTS = Path('models') / 'yolov9-c.pt'
OUT = Path('../coreml') / 'yolov9-c.mlmodel'

def main():
    from models.common import DetectMultiBackend
    model = DetectMultiBackend(str(WEIGHTS), device='cpu')
    model.eval()

    # Representative input: adjust to your preferred input size
    example = torch.rand(1, 3, 640, 640)

    # Try to trace the internal backend if available
    backend = getattr(model, 'model', model)
    traced = torch.jit.trace(backend, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name='input_image', shape=example.shape, scale=1/255.0)],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUT))
    print('Saved', OUT)

if __name__ == '__main__':
    main()
