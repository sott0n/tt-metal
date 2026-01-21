# ConvNextV2 Model Analysis

## Model Overview

- **Model**: facebook/convnextv2-base-1k-224
- **Parameters**: 88.72M
- **Input**: 224x224 RGB images
- **Output**: 1000 ImageNet classes

## Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224 |
| Patch size | 4 |
| Num channels | 3 |
| Hidden sizes | [128, 256, 512, 1024] |
| Depths | [3, 3, 27, 3] |
| Num stages | 4 |
| Hidden act | GELU |

## Module Hierarchy

```
ConvNextV2ForImageClassification
├── convnextv2: ConvNextV2Model
│   ├── embeddings: ConvNextV2Embeddings
│   │   ├── patch_embeddings: Conv2d(3, 128, kernel=4, stride=4)
│   │   └── layernorm: ConvNextV2LayerNorm(128)
│   ├── encoder: ConvNextV2Encoder
│   │   └── stages: ModuleList
│   │       ├── Stage 0: 3 layers, hidden=128
│   │       ├── Stage 1: 3 layers, hidden=256 (downsampling)
│   │       ├── Stage 2: 27 layers, hidden=512 (downsampling)
│   │       └── Stage 3: 3 layers, hidden=1024 (downsampling)
│   └── layernorm: LayerNorm(1024)
└── classifier: Linear(1024, 1000)
```

## ConvNextV2Layer Structure (Main Building Block)

```
Input (NCHW) ─────────────────────────────┐
     │                                    │
     ▼                                    │
┌─────────────────────────────────────┐   │
│ dwconv: Depthwise Conv2d            │   │
│   - kernel: 7x7, groups=channels    │   │
│   - padding: 3                      │   │
└─────────────────────────────────────┘   │
     │                                    │
     ▼ permute to NHWC                    │
┌─────────────────────────────────────┐   │
│ layernorm: LayerNorm (channels_last)│   │
└─────────────────────────────────────┘   │
     │                                    │
     ▼                                    │
┌─────────────────────────────────────┐   │
│ pwconv1: Linear (expand 4x)         │   │
│   - in: hidden_dim                  │   │
│   - out: hidden_dim * 4             │   │
└─────────────────────────────────────┘   │
     │                                    │
     ▼                                    │
┌─────────────────────────────────────┐   │
│ act: GELU                           │   │
└─────────────────────────────────────┘   │
     │                                    │
     ▼                                    │
┌─────────────────────────────────────┐   │
│ grn: Global Response Normalization  │   │  ◄── CUSTOM OP
│   - weight: (1, 1, 1, hidden*4)     │   │
│   - bias: (1, 1, 1, hidden*4)       │   │
└─────────────────────────────────────┘   │
     │                                    │
     ▼                                    │
┌─────────────────────────────────────┐   │
│ pwconv2: Linear (contract)          │   │
│   - in: hidden_dim * 4              │   │
│   - out: hidden_dim                 │   │
└─────────────────────────────────────┘   │
     │                                    │
     ▼ permute to NCHW                    │
     │                                    │
     └──────────► ADD ◄───────────────────┘
                  │
                  ▼
              Output (NCHW)
```

## Operators Summary

| Operator | Count | Notes |
|----------|-------|-------|
| Conv2d | 40 | Includes patch embed (1), dwconv (36), downsampling (3) |
| Linear | 73 | pwconv1 (36), pwconv2 (36), classifier (1) |
| LayerNorm | 41 | ConvNextV2LayerNorm (40) + final LayerNorm (1) |
| GRN | 36 | Custom normalization layer |
| GELU | 36 | Activation function |
| Identity | 37 | Drop path (disabled during inference) |

## Shape Flow

```
Stage       | Operation            | Shape (NCHW)      | Shape (NHWC)
------------|---------------------|-------------------|------------------
Input       | -                   | [1, 3, 224, 224]  | -
Patch Embed | Conv2d(4,4,s=4)     | [1, 128, 56, 56]  | -
Stage 0     | dwconv              | [1, 128, 56, 56]  | -
            | permute             | -                 | [1, 56, 56, 128]
            | layernorm, linear   | -                 | [1, 56, 56, 512]
            | grn, linear         | -                 | [1, 56, 56, 128]
            | permute back        | [1, 128, 56, 56]  | -
Down 0→1    | Conv2d(2,2,s=2)     | [1, 256, 28, 28]  | -
Stage 1     | 3 layers            | [1, 256, 28, 28]  | -
Down 1→2    | Conv2d(2,2,s=2)     | [1, 512, 14, 14]  | -
Stage 2     | 27 layers           | [1, 512, 14, 14]  | -
Down 2→3    | Conv2d(2,2,s=2)     | [1, 1024, 7, 7]   | -
Stage 3     | 3 layers            | [1, 1024, 7, 7]   | -
Pool        | AdaptiveAvgPool2d   | [1, 1024, 1, 1]   | -
Flatten     | -                   | [1, 1024]         | -
Classifier  | Linear              | [1, 1000]         | -
```

## GRN (Global Response Normalization) - Custom Operation

```python
# Reference implementation from HuggingFace
class ConvNextV2GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, hidden_states):
        # hidden_states: (N, H, W, C)
        # Compute L2 norm across spatial dimensions
        gx = torch.norm(hidden_states, p=2, dim=(1, 2), keepdim=True)
        # Normalize by mean
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        # Apply normalization with learnable parameters
        return self.weight * (hidden_states * nx) + self.bias + hidden_states
```

**TTNN Implementation Strategy**:
1. Compute L2 norm: `ttnn.norm(x, dim=(1,2))` or manual computation
2. Compute mean: `ttnn.mean(gx, dim=-1)`
3. Division, multiplication, addition

## Operators to Map to TTNN

| PyTorch | TTNN | Notes |
|---------|------|-------|
| `nn.Conv2d` | `ttnn.conv2d` | Standard and depthwise |
| `nn.Linear` | `ttnn.linear` | Point-wise convolutions |
| `nn.LayerNorm` | `ttnn.layer_norm` | Channels-last format |
| `nn.GELU` | `ttnn.gelu` | Standard activation |
| `GRN` | Custom | See implementation above |
| `mean` | `ttnn.mean` | Global average pooling |
| `add` | `ttnn.add` | Residual connections |
| `permute` | `ttnn.permute` | Format conversion |

## Files Generated

- `analyze_model.py` - Analysis script
- `golden_data.pt` - Golden input/output for validation

## Step 1 Checklist

- [x] Reference model loads and runs successfully
- [x] Model graph has been generated and reviewed
- [x] All operators have been identified and listed
- [x] Input/output shapes are documented for each layer
- [x] Module hierarchy is understood
- [x] Golden inputs/outputs are saved for validation
