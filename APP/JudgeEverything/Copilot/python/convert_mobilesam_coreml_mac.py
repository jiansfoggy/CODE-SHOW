"""
Mac conversion helper for MobileSAM. Adjust REPO_ROOT and model constructor to match the
MobileSAM repo structure you cloned.

Run on macOS with a venv that has torch==2.7.0 and coremltools==6.3 installed.
"""
import sys
from pathlib import Path
import torch
import coremltools as ct

REPO_ROOT = Path.home() / 'Projects' / 'MobileSAM'  # <- change this to your clone path
sys.path.insert(0, str(REPO_ROOT))

WEIGHTS = Path('models') / 'mobile_sam.pt'
OUT = Path('../coreml') / 'mobile_sam.mlmodel'

def main():
    # Import the model factory from MobileSAM repo. Replace `build_sam` with the actual function
    # name in the repo. Example: from mobile_sam import build_sam
    try:
        from mobile_sam import build_sam  # adjust this import
    except Exception:
        # If repo exposes differently, update import accordingly
        raise RuntimeError('Update import to match MobileSAM repo: e.g. `from mobile_sam import build_sam`')

    model = build_sam()
    state = torch.load(str(WEIGHTS), map_location='cpu')
    # If state is a dict with 'model' key, extract it
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.eval()

    example = torch.rand(1, 3, 256, 256)
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name='input_image', shape=example.shape, scale=1/255.0)],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUT))
    print('Saved', OUT)

if __name__ == '__main__':
    main()
