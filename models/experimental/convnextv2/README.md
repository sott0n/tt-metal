# ConvNextV2 TTNN Implementation

ConvNextV2 image classification model running on Tenstorrent hardware.

## Quick Start

### 1. Prerequisites

```bash
# Activate Python environment
source python_env/bin/activate

# Install additional dependencies
pip install transformers pillow requests
```

### 2. Run Demo

```bash
# Random input (basic test)
python models/experimental/convnextv2/demo/demo_convnextv2.py

# With a real image
python models/experimental/convnextv2/demo/demo_convnextv2.py \
    --image "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400"

# Compare with PyTorch reference
python models/experimental/convnextv2/demo/demo_convnextv2.py \
    --image path/to/your/image.jpg \
    --compare
```

### 3. Example Output

```
============================================================
ConvNextV2 TTNN Demo
============================================================
Downloading ConvNextV2-base model from HuggingFace...
Model loaded: facebook/convnextv2-base-1k-224
  - Parameters: 88,717,800
  - Num classes: 1000

Loading image: https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400
Input shape: torch.Size([1, 3, 224, 224])

Opening TTNN device 0...
Converting model parameters to TTNN format...
Running TTNN inference...

============================================================
TTNN Results
============================================================

Top-5 Predictions:
--------------------------------------------------
  1. tabby: 49.02%
  2. Egyptian cat: 29.69%
  3. tiger cat: 9.62%
  4. lynx: 0.38%
  5. Siamese cat: 0.05%

PCC (TTNN vs PyTorch): 0.998485
```

## Python API Usage

```python
import torch
import ttnn
from transformers import ConvNextV2ForImageClassification, AutoImageProcessor
from PIL import Image

from models.experimental.convnextv2.tt.ttnn_convnextv2_model import (
    ttnn_convnextv2_for_image_classification,
    create_convnextv2_parameters,
)

# 1. Load HuggingFace model and processor
model_name = "facebook/convnextv2-base-1k-224"
torch_model = ConvNextV2ForImageClassification.from_pretrained(model_name)
torch_model.eval()
processor = AutoImageProcessor.from_pretrained(model_name)

# 2. Prepare input image
image = Image.open("path/to/image.jpg")
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]  # [1, 3, 224, 224]

# 3. Open TTNN device (l1_small_size required for depthwise conv)
device = ttnn.open_device(device_id=0, l1_small_size=24 * 1024)

# 4. Convert parameters to TTNN format
parameters = create_convnextv2_parameters(torch_model, device)

# 5. Model config
config = {
    "hidden_sizes": [128, 256, 512, 1024],
    "depths": [3, 3, 27, 3],
}

# 6. Convert input to TTNN
ttnn_input = ttnn.from_torch(
    pixel_values,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)

# 7. Run inference
logits = ttnn_convnextv2_for_image_classification(
    ttnn_input,
    parameters=parameters,
    device=device,
    config=config,
    batch_size=1,
)

# 8. Get predictions
output = ttnn.to_torch(logits)
predicted_class = output.argmax(dim=-1).item()
print(f"Predicted class: {predicted_class}")

# 9. Cleanup
ttnn.close_device(device)
```

## Device Configuration

**Important**: This model requires `l1_small_size` configuration for depthwise convolutions.

```python
# Always specify l1_small_size when opening device
device = ttnn.open_device(device_id=0, l1_small_size=24 * 1024)
```

For pytest, use the `device_params` fixture:

```python
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24 * 1024}], indirect=True)
def test_model(device):
    # device is configured with l1_small_size
    pass
```

## Model Architecture

```
ConvNextV2-base (88.7M parameters)
├── Embeddings: Conv2d(3→128, k=4, s=4) + LayerNorm
├── Encoder
│   ├── Stage 0: 3 × ConvNextV2Layer (hidden=128, spatial=56×56)
│   ├── Stage 1: Downsample + 3 × ConvNextV2Layer (hidden=256, spatial=28×28)
│   ├── Stage 2: Downsample + 27 × ConvNextV2Layer (hidden=512, spatial=14×14)
│   └── Stage 3: Downsample + 3 × ConvNextV2Layer (hidden=1024, spatial=7×7)
├── Global Average Pooling
├── Final LayerNorm
└── Classifier: Linear(1024→1000)
```

Each ConvNextV2Layer contains:
- Depthwise Conv2d (7×7, groups=channels)
- LayerNorm (channels-last)
- Pointwise Conv1 (expand 4×)
- GELU activation
- GRN (Global Response Normalization)
- Pointwise Conv2 (contract)
- Residual connection

## Performance

| Metric | Value |
|--------|-------|
| PCC (vs PyTorch) | ~0.998 |
| Top-1 Accuracy Match | ✅ |
| Inference Time* | ~5-8s |

*Including parameter conversion. Cached parameters would be faster.

## Running Tests

```bash
# All tests
pytest models/experimental/convnextv2/tests/ -v

# Operator tests (Step 3)
pytest models/experimental/convnextv2/tests/test_operators.py -v

# Module tests (Step 5)
pytest models/experimental/convnextv2/tests/test_modules.py -v

# E2E tests (Step 7)
pytest models/experimental/convnextv2/tests/test_e2e_model.py -v
```

## Files

```
models/experimental/convnextv2/
├── demo/
│   └── demo_convnextv2.py          # Demo script
├── tt/
│   ├── ttnn_convnextv2_layer.py    # Layer implementation
│   └── ttnn_convnextv2_model.py    # Full model + parameter conversion
├── reference/
│   ├── torch_functional.py         # PyTorch reference
│   └── *.md                        # Documentation
├── tests/
│   ├── test_operators.py           # Operator tests
│   ├── test_modules.py             # Module tests
│   └── test_e2e_model.py           # E2E tests
└── README.md                       # This file
```

## Known Limitations

1. **GRN uses host fallback**: The GRN operation uses `ttnn.to_torch()` for reduction operations, causing host-device data transfers.

2. **Layout conversions**: Each layer performs NCHW↔NHWC conversions. An optimized NHWC version exists but is not used in the main model.

3. **Single batch only**: Currently tested with batch_size=1.

## References

- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
- [HuggingFace Model](https://huggingface.co/facebook/convnextv2-base-1k-224)
