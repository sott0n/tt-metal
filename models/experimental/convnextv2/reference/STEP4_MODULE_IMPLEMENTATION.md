# Step 4: Module Implementation Summary

## Overview

Implemented TTNN modules for ConvNextV2 model following the functional programming pattern.

## Project Structure

```
models/experimental/convnextv2/
├── __init__.py
├── reference/
│   ├── __init__.py
│   ├── analyze_model.py              # Step 1: Model analysis
│   ├── MODEL_ANALYSIS.md             # Step 1: Documentation
│   ├── OPERATOR_MAPPING.md           # Step 2: Operator mapping
│   ├── STEP3_PCC_RESULTS.md          # Step 3: Test results
│   ├── STEP4_MODULE_IMPLEMENTATION.md # This file
│   ├── torch_functional.py           # Functional PyTorch reference
│   └── golden_data.pt                # Golden input/output
├── tt/
│   ├── __init__.py
│   ├── ttnn_convnextv2_layer.py      # Main building block
│   └── ttnn_convnextv2_model.py      # Full model implementation
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_operators.py              # Step 3: Operator tests
```

## Implemented Modules

### 1. ttnn_convnextv2_layer.py

#### `ttnn_grn(hidden_states, weight, bias, *, device, eps=1e-6)`
- Global Response Normalization composite operation
- Input: `[N, H, W, C]` (NHWC format)
- Implements L2 norm computation, channel normalization, and learnable parameters

#### `ttnn_convnextv2_layer(hidden_states, *, parameters, device, ...)`
- Main building block with standard NCHW input/output
- Contains: dwconv → layernorm → pwconv1 → gelu → grn → pwconv2 → residual

#### `ttnn_convnextv2_layer_nhwc(hidden_states, *, parameters, device, ...)`
- Optimized version maintaining NHWC format throughout
- Avoids extra permute operations

### 2. ttnn_convnextv2_model.py

#### `ttnn_convnextv2_embeddings(pixel_values, *, parameters, device, batch_size)`
- Patch embedding: Conv2d(3→128, k=4, s=4)
- LayerNorm after embedding

#### `ttnn_convnextv2_downsampling(hidden_states, *, parameters, device, ...)`
- Spatial reduction: LayerNorm + Conv2d(k=2, s=2)
- Used between stages

#### `ttnn_convnextv2_stage(hidden_states, *, parameters, device, ...)`
- One encoder stage: optional downsampling + N layers
- Stage 0: 3 layers, hidden=128
- Stage 1: 3 layers, hidden=256 (with downsampling)
- Stage 2: 27 layers, hidden=512 (with downsampling)
- Stage 3: 3 layers, hidden=1024 (with downsampling)

#### `ttnn_convnextv2_encoder(hidden_states, *, parameters, device, config, batch_size)`
- Full encoder with all 4 stages

#### `ttnn_convnextv2_model(pixel_values, *, parameters, device, config, batch_size)`
- Embeddings + Encoder + Global Avg Pool + Final LayerNorm
- Returns pooled output `[N, 1024]`

#### `ttnn_convnextv2_for_image_classification(pixel_values, *, parameters, device, config, batch_size)`
- Full model with classifier
- Returns logits `[N, 1000]`

#### `create_convnextv2_parameters(torch_model, device)`
- Converts HuggingFace ConvNextV2 weights to TTNN format
- Handles weight transposition for linear layers
- Creates nested SimpleNamespace parameter structure

### 3. torch_functional.py (Reference)

Functional PyTorch implementations for testing:
- `torch_grn()` - GRN reference
- `torch_convnextv2_layer()` - Layer reference
- `torch_convnextv2_embeddings()` - Embeddings reference
- `torch_convnextv2_downsampling()` - Downsampling reference
- Helper functions for parameter extraction

## Data Flow

```
Input: [N, 3, 224, 224] (NCHW)
    │
    ▼
Embeddings: Conv2d(k=4,s=4) + LayerNorm
    │
    ▼
[N, 128, 56, 56]
    │
    ▼
Stage 0: 3 × ConvNextV2Layer
    │
    ▼
[N, 128, 56, 56]
    │
    ▼
Downsampling + Stage 1: 3 × ConvNextV2Layer
    │
    ▼
[N, 256, 28, 28]
    │
    ▼
Downsampling + Stage 2: 27 × ConvNextV2Layer
    │
    ▼
[N, 512, 14, 14]
    │
    ▼
Downsampling + Stage 3: 3 × ConvNextV2Layer
    │
    ▼
[N, 1024, 7, 7]
    │
    ▼
Global Avg Pool + LayerNorm
    │
    ▼
[N, 1024]
    │
    ▼
Classifier: Linear(1024, 1000)
    │
    ▼
Output: [N, 1000]
```

## Usage Example

```python
import ttnn
from transformers import ConvNextV2ForImageClassification
from models.experimental.convnextv2.tt import (
    ttnn_convnextv2_for_image_classification,
    create_convnextv2_parameters,
)

# Load reference model
torch_model = ConvNextV2ForImageClassification.from_pretrained(
    "facebook/convnextv2-base-1k-224"
)

# Open device
device = ttnn.open_device(device_id=0)

# Convert parameters
parameters = create_convnextv2_parameters(torch_model, device)

# Model config
config = {
    "hidden_sizes": [128, 256, 512, 1024],
    "depths": [3, 3, 27, 3],
}

# Prepare input
pixel_values = ttnn.from_torch(
    torch_input,  # [1, 3, 224, 224]
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)

# Run inference
logits = ttnn_convnextv2_for_image_classification(
    pixel_values,
    parameters=parameters,
    device=device,
    config=config,
    batch_size=1,
)

ttnn.close_device(device)
```

## Checklist

- [x] PyTorch modules rewritten as functional code
- [x] TTNN modules implemented using ttnn operators
- [x] Parameter preprocessing function created
- [x] Project structure organized
- [x] All modules compile without errors
- [x] Modules are importable

## Known Limitations

1. **Conv2d Memory Config**: Depthwise and downsampling convolutions may require memory configuration tuning for larger batch sizes.

2. **GRN Implementation**: Uses host-side reduction operations (sum, mean) via `ttnn.to_torch()`. Future optimization could use pure TTNN reduction ops.

3. **Layout Management**: Some reshape/layout operations may need refinement based on actual device testing.

## Next Step

Proceed to **Step 5: Per-Module Testing** to validate each module against PyTorch reference with PCC >= 0.99.
