# Step 6: E2E Model Implementation Summary

## Overview

All ConvNextV2 TTNN modules have been composed into a complete end-to-end model.

## Implementation Components

### Main Model Functions

| Function | File | Description |
|----------|------|-------------|
| `ttnn_convnextv2_embeddings` | `tt/ttnn_convnextv2_model.py` | Patch embedding + LayerNorm |
| `ttnn_convnextv2_downsampling` | `tt/ttnn_convnextv2_model.py` | Spatial reduction (2x) |
| `ttnn_convnextv2_stage` | `tt/ttnn_convnextv2_model.py` | One encoder stage |
| `ttnn_convnextv2_encoder` | `tt/ttnn_convnextv2_model.py` | All 4 encoder stages |
| `ttnn_convnextv2_model` | `tt/ttnn_convnextv2_model.py` | Complete backbone |
| `ttnn_convnextv2_for_image_classification` | `tt/ttnn_convnextv2_model.py` | Full model + classifier |
| `create_convnextv2_parameters` | `tt/ttnn_convnextv2_model.py` | HuggingFace weight conversion |

### Layer Building Blocks

| Function | File | Description |
|----------|------|-------------|
| `ttnn_grn` | `tt/ttnn_convnextv2_layer.py` | Global Response Normalization |
| `ttnn_convnextv2_layer` | `tt/ttnn_convnextv2_layer.py` | ConvNextV2Layer block |

## Model Data Flow

```
Input: [N, 3, 224, 224] (NCHW)
    │
    ▼
ttnn_convnextv2_embeddings
├── Conv2d(3→128, k=4, s=4)
└── LayerNorm(128)
    │
    ▼
[N, 128, 56, 56]
    │
    ▼
ttnn_convnextv2_encoder
├── Stage 0: 3 × ConvNextV2Layer
│   └── [N, 128, 56, 56]
├── Stage 1: Downsample + 3 × ConvNextV2Layer
│   └── [N, 256, 28, 28]
├── Stage 2: Downsample + 27 × ConvNextV2Layer
│   └── [N, 512, 14, 14]
└── Stage 3: Downsample + 3 × ConvNextV2Layer
    └── [N, 1024, 7, 7]
    │
    ▼
Global Average Pooling
    │
    ▼
[N, 1024]
    │
    ▼
Final LayerNorm
    │
    ▼
Classifier Linear(1024, 1000)
    │
    ▼
Output: [N, 1000]
```

## Usage Example

```python
import ttnn
import torch
from transformers import ConvNextV2ForImageClassification
from models.experimental.convnextv2.tt import (
    ttnn_convnextv2_for_image_classification,
    create_convnextv2_parameters,
)

# Load HuggingFace model
torch_model = ConvNextV2ForImageClassification.from_pretrained(
    "facebook/convnextv2-base-1k-224"
)
torch_model.eval()

# Open device with l1_small_size for depthwise conv
device = ttnn.open_device(device_id=0, l1_small_size=24 * 1024)

# Convert parameters to TTNN format
parameters = create_convnextv2_parameters(torch_model, device)

# Model config
config = {
    "hidden_sizes": [128, 256, 512, 1024],
    "depths": [3, 3, 27, 3],
}

# Prepare input
torch_input = torch.randn(1, 3, 224, 224)
ttnn_input = ttnn.from_torch(
    torch_input,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)

# Run inference
logits = ttnn_convnextv2_for_image_classification(
    ttnn_input,
    parameters=parameters,
    device=device,
    config=config,
    batch_size=1,
)

# Get predictions
output = ttnn.to_torch(logits)
predicted_class = output.argmax(dim=-1).item()
print(f"Predicted class: {predicted_class}")

ttnn.close_device(device)
```

## Device Configuration

Required device settings for depthwise convolutions:

```python
# Option 1: Using device_params fixture in pytest
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24 * 1024}], indirect=True)
class TestConvNextV2:
    def test_model(self, device):
        # device is configured with l1_small_size
        pass

# Option 2: Direct device creation
device = ttnn.open_device(device_id=0, l1_small_size=24 * 1024)
```

## Memory Considerations

- **L1_SMALL Buffer**: Required for depthwise convolutions with `groups=channels`
- **BFloat16**: All activations and weights use bfloat16 for memory efficiency
- **36 ConvNextV2Layers**: Total of 36 depthwise convolutions in the model

## Checklist

- [x] All modules integrated into main model
- [x] Model wrapper functions created
- [x] Configuration support implemented
- [x] Parameter conversion from HuggingFace
- [x] Memory configuration documented
- [x] Model compiles without errors
- [x] Model runs inference successfully

## Next Step

Proceed to **Step 7: E2E Model Testing** for comprehensive validation.
