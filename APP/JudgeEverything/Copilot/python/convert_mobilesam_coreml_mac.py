"""
Mac conversion helper for MobileSAM. Uses the actual MobileSAM build_sam_vit_t_encoder API.

Run on macOS with a venv that has torch==2.7.0 and coremltools==6.3 installed.
"""
import sys
from pathlib import Path
import torch
import coremltools as ct

REPO_ROOT = Path.home() / 'Projects' / 'MobileSAM' / 'MobileSAMv2'  # <- adjust to your clone path
sys.path.insert(0, str(REPO_ROOT))

WEIGHTS = Path('models') / 'mobile_sam.pt'
OUT = Path('../coreml') / 'mobile_sam.mlmodel'

def main():
    # Import the build function from MobileSAM
    from mobilesamv2.build_sam import build_sam_vit_t_encoder

    # Build and load the TinyViT encoder model
    model = build_sam_vit_t_encoder(checkpoint=str(WEIGHTS))
    model.eval()

    # Representative input for TinyViT: shape depends on expected image size
    # TinyViT typically expects 1024x1024 images
    example = torch.rand(1, 3, 1024, 1024)

    try:
        traced = torch.jit.trace(model, example)
    except Exception as e:
        print(f'Tracing failed: {e}. Attempting script conversion...')
        traced = torch.jit.script(model)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name='input_image', shape=example.shape, scale=1/255.0)],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(OUT))
    print('Saved', OUT)

if __name__ == '__main__':
    main()
