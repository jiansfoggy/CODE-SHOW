macOS: PyTorch -> Core ML Conversion Guide (YOLOv9 + MobileSAM)

Overview
--------
This document walks through converting YOLOv9 (`yolov9-c.pt`) and MobileSAM (`mobile_sam.pt`) PyTorch weights into Core ML `.mlmodel` files on a macOS machine. Core ML conversion requires the model object (not just a state_dict), a compatible PyTorch version, and `coremltools` installed on macOS.

High-level steps
1. Prepare macOS Python environment with compatible versions of PyTorch and coremltools.
2. Clone the upstream repositories (`yolov9` and `MobileSAM`) and install their Python packages (or add to PYTHONPATH).
3. Use the repo model constructors to instantiate model objects and load `state_dict` weights.
4. Export the model to TorchScript (`torch.jit.trace` or `torch.jit.script`) using representative inputs.
5. Use `coremltools` to convert the traced model into `.mlmodel` files.

Important notes
- Use a Mac with Apple silicon or Intel + Xcode. Core ML conversion works best on macOS and `coremltools` may not function fully in Linux containers.
- Recommended package versions (known to be stable together):
  - Python 3.10 or 3.11
  - torch==2.7.0
  - torchvision matching torch (install from the PyTorch wheel page)
  - coremltools==6.3
  - numpy, pillow, scikit-learn<=1.5.1

Setup commands (macOS)
----------------------
Open Terminal and run:

```bash
# Create a venv
python3 -m venv venv_coreml
source venv_coreml/bin/activate

# Upgrade pip
pip install -U pip setuptools wheel

# Install specific versions (adjust torch wheel for your Mac + CUDA/CPU)
pip install coremltools==6.3 scikit-learn==1.5.1 numpy pillow

# Install PyTorch (CPU) - recommended to match tested versions
pip install torch==2.7.0 torchvision==0.18.1

# Optional: install git if missing
# brew install git

```

Clone the upstream repos
------------------------
```bash
cd $HOME/Projects
git clone https://github.com/WongKinYiu/yolov9.git
git clone https://github.com/ChaoningZhang/MobileSAM.git

# Install them as editable packages or add to PYTHONPATH
cd yolov9 && pip install -e .
cd ../MobileSAM && pip install -e .
```

Conversion template: YOLOv9
--------------------------
Save this as `convert_yolov9_coreml_mac.py` and run it from the `Copilot/python` folder after placing `yolov9-c.pt` into `models/`.

```python
import torch
import coremltools as ct
import sys
from pathlib import Path

# adjust paths
REPO_ROOT = Path('/path/to/Projects/yolov9')  # change to where you cloned yolov9
sys.path.insert(0, str(REPO_ROOT))

from models.common import DetectMultiBackend  # yolov9 helper

WEIGHTS = 'models/yolov9-c.pt'
OUTPUT = '../coreml/yolov9-c.mlmodel'

device = 'cpu'
model = DetectMultiBackend(WEIGHTS, device=device)
model.eval()

# Example input - adjust size to model's expected input (e.g., 640)
example = torch.rand(1, 3, 640, 640)

# Some DetectMultiBackend wrappers expect preprocessing; we trace the underlying model
# If DetectMultiBackend exposes a `.model` attribute, trace that; otherwise trace the wrapper call
try:
    backend = model.model if hasattr(model, 'model') else model
    traced = torch.jit.trace(backend, example)
except Exception:
    traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name='input_image', shape=example.shape, scale=1/255.0)],
)
mlmodel.save(OUTPUT)
print('Saved', OUTPUT)
```

Conversion template: MobileSAM
------------------------------
MobileSAM typically provides a model constructor that you must use to build the model and load weights (state_dict). Save this as `convert_mobilesam_coreml_mac.py`.

```python
import torch
import coremltools as ct
import sys
from pathlib import Path

# adjust path to your MobileSAM clone
REPO_ROOT = Path('/path/to/Projects/MobileSAM')
sys.path.insert(0, str(REPO_ROOT))

# Import the build function from MobileSAM repo (adjust name if needed)
from models import create_mobilesam  # example placeholder; adapt to real repo API

WEIGHTS = 'models/mobile_sam.pt'
OUTPUT = '../coreml/mobile_sam.mlmodel'

device = 'cpu'

# Build the model via the repo API
model = create_mobilesam()
state = torch.load(WEIGHTS, map_location=device)
if isinstance(state, dict) and 'model' in state:
    state = state['model']
model.load_state_dict(state)
model.eval()

# Create representative input (size depends on MobileSAM's expected input)
example = torch.rand(1, 3, 256, 256)

traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name='input_image', shape=example.shape, scale=1/255.0)],
)
mlmodel.save(OUTPUT)
print('Saved', OUTPUT)
```

Troubleshooting
---------------
- If you see errors about unsupported ops, consider using `torch.fx` or writing a small wrapper that uses only standard PyTorch ops before tracing.
- If weights are a `state_dict`, ensure you construct the model object from the repository's model class and call `load_state_dict(state_dict)`.
- If the model uses custom CUDA ops or third-party layers, you may need to reimplement or replace them with compatible PyTorch ops.

Swift usage example (after conversion)
------------------------------------
Use the converted `.mlmodel` in Xcode: add the file to your project; Xcode will generate a Swift class you can call.

Detection example (pseudocode):

```swift
let model = try YOLOv9_c(configuration: MLModelConfiguration())
let input = YOLOv9_cInput(input_image: pixelBuffer)
let out = try model.prediction(input: input)
// Parse output depending on how conversion exposed outputs (may need to reshape/tensorBuffer)
```

Mask model (MobileSAM) example:

```swift
let sam = try MobileSAM(configuration: MLModelConfiguration())
let input = MobileSAMInput(input_image: pixelBuffer)
let out = try sam.prediction(input: input)
// out will contain mask tensor / image - overlay on preview
```

Final notes
-----------
I can generate more precise conversion code after you run the first step (cloning & installing repos) and tell me the exact model constructor names exposed by each repo (or paste the model class paths). If you prefer, run the templates above on your Mac and paste any errors here — I'll refine the scripts to match the repo APIs.
