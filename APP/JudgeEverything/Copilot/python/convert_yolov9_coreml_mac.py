"""
Mac conversion helper for YOLOv9. Uses the actual YOLOv9 DetectModel API.

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
    from models.yolo import DetectModel
    from models.common import DetectMultiBackend
    
    # Use Metal Performance Shaders on M-series Macs for faster inference
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using Metal Performance Shaders (MPS) for acceleration on M-series Mac')
    else:
        device = torch.device('cpu')
        print('MPS not available, using CPU')
    
    # Load the full model with state dict
    checkpoint = torch.load(str(WEIGHTS), map_location=device, weights_only=False)
    
    # If checkpoint is a complete model state, build the model
    if isinstance(checkpoint, dict):
        # This is likely a state_dict; we need to load it into a model instance
        # For YOLOv9, the easiest way is to use the DetectMultiBackend wrapper
        try:
            from models.common import DetectMultiBackend
            model = DetectMultiBackend(str(WEIGHTS), device=device)
            model = model.model if hasattr(model, 'model') else model
        except Exception as e:
            print(f'DetectMultiBackend failed: {e}. Using direct torch load.')
            # Fallback: directly use the checkpoint as a model
            model = checkpoint
    else:
        model = checkpoint
    
    if hasattr(model, 'eval'):
        model.eval()

    # Representative input for YOLOv9: 640x640 is standard
    example = torch.rand(1, 3, 640, 640)

    try:
        # Try to get the actual forward-pass module if model is wrapped
        forward_module = getattr(model, 'model', model)
        traced = torch.jit.trace(forward_module, example)
    except Exception as e:
        print(f'Tracing failed: {e}. Attempting script conversion...')
        try:
            traced = torch.jit.script(model)
        except Exception as e2:
            print(f'Script conversion also failed: {e2}')
            return

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name='input_image', shape=example.shape, scale=1/255.0)],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUT))
    print('Saved', OUT)

if __name__ == '__main__':
    main()
