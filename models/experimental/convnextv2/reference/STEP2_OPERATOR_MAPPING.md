# ConvNextV2 Operator Mapping

## Overview

This document maps all PyTorch operators used in ConvNextV2 to their TTNN equivalents.

## Operator Mapping Table

| Module/Context | PyTorch Operator | TTNN Operator | Count | Notes |
|----------------|------------------|---------------|-------|-------|
| Patch Embedding | `nn.Conv2d(3, 128, k=4, s=4)` | `ttnn.conv2d` | 1 | Standard conv |
| Embedding LayerNorm | `ConvNextV2LayerNorm` | `ttnn.layer_norm` | 1 | Channels-last |
| Depthwise Conv | `nn.Conv2d(groups=C)` | `ttnn.conv2d(groups=C)` | 36 | 7x7 kernel, groups=channels |
| Layer LayerNorm | `ConvNextV2LayerNorm` | `ttnn.layer_norm` | 36 | Channels-last format |
| pwconv1 (Expand) | `nn.Linear(C, 4*C)` | `ttnn.linear` | 36 | Expand 4x |
| GELU | `nn.GELU` | `ttnn.gelu` | 36 | Standard GELU |
| GRN | `ConvNextV2GRN` | **Custom Composite** | 36 | See GRN section |
| pwconv2 (Contract) | `nn.Linear(4*C, C)` | `ttnn.linear` | 36 | Contract back |
| Residual Add | `+` | `ttnn.add` | 36 | Skip connection |
| Downsampling LayerNorm | `ConvNextV2LayerNorm` | `ttnn.layer_norm` | 3 | Before downsample |
| Downsampling Conv | `nn.Conv2d(k=2, s=2)` | `ttnn.conv2d` | 3 | Spatial reduction |
| Final LayerNorm | `nn.LayerNorm` | `ttnn.layer_norm` | 1 | Standard LayerNorm |
| Global Avg Pool | `AdaptiveAvgPool2d(1)` | `ttnn.global_avg_pool2d` | 1 | Global pooling |
| Classifier | `nn.Linear(1024, 1000)` | `ttnn.linear` | 1 | Classification head |
| Permute NCHW→NHWC | `permute(0,2,3,1)` | `ttnn.permute` | 36 | Inside each layer |
| Permute NHWC→NCHW | `permute(0,3,1,2)` | `ttnn.permute` | 36 | Inside each layer |
| Drop Path | `Identity` | Removed | 37 | Disabled in inference |

## Direct Mappings

### 1. Conv2d Operations

```python
# Standard Conv2d (Patch Embedding)
# PyTorch: nn.Conv2d(3, 128, kernel_size=4, stride=4)
ttnn.conv2d(
    input_tensor=x,  # NHWC format
    weight_tensor=weight,
    bias_tensor=bias,
    in_channels=3,
    out_channels=128,
    kernel_size=(4, 4),
    stride=(4, 4),
    padding=(0, 0),
    device=device,
    batch_size=batch_size,
    input_height=224,
    input_width=224,
    groups=1,
)

# Depthwise Conv2d (7x7)
# PyTorch: nn.Conv2d(C, C, kernel_size=7, padding=3, groups=C)
ttnn.conv2d(
    input_tensor=x,  # NHWC format
    weight_tensor=weight,
    bias_tensor=bias,
    in_channels=hidden_dim,
    out_channels=hidden_dim,
    kernel_size=(7, 7),
    stride=(1, 1),
    padding=(3, 3),
    device=device,
    batch_size=batch_size,
    input_height=H,
    input_width=W,
    groups=hidden_dim,  # Depthwise: groups = in_channels
)

# Downsampling Conv2d
# PyTorch: nn.Conv2d(C_in, C_out, kernel_size=2, stride=2)
ttnn.conv2d(
    input_tensor=x,
    weight_tensor=weight,
    in_channels=C_in,
    out_channels=C_out,
    kernel_size=(2, 2),
    stride=(2, 2),
    padding=(0, 0),
    device=device,
    groups=1,
)
```

### 2. Linear Operations

```python
# PyTorch: nn.Linear(in_features, out_features)
# Note: In ConvNextV2, Linear is applied on NHWC tensors (pointwise conv)
ttnn.linear(
    input_tensor=x,  # Shape: [N*H*W, C] or [N, H, W, C]
    weight=weight,   # Shape: [out_features, in_features]
    bias=bias,
)
```

### 3. LayerNorm

```python
# ConvNextV2LayerNorm (channels-last)
# PyTorch: input shape [N, H, W, C], normalize over C
ttnn.layer_norm(
    input_tensor=x,  # Shape: [N, H, W, C] or equivalent
    normalized_shape=normalized_dim,
    weight=gamma,
    bias=beta,
    epsilon=1e-6,
)
```

### 4. GELU Activation

```python
# PyTorch: nn.GELU()
ttnn.gelu(x)
```

### 5. Residual Addition

```python
# PyTorch: residual + x
ttnn.add(residual, x)
```

### 6. Global Average Pooling

```python
# PyTorch: AdaptiveAvgPool2d((1, 1))
# Input: [N, C, H, W] or [N, H, W, C]
ttnn.global_avg_pool2d(x)
```

### 7. Permute Operations

```python
# NCHW to NHWC
# PyTorch: x.permute(0, 2, 3, 1)
ttnn.permute(x, (0, 2, 3, 1))

# NHWC to NCHW
# PyTorch: x.permute(0, 3, 1, 2)
ttnn.permute(x, (0, 3, 1, 2))
```

## Custom Composite Operation: GRN

### PyTorch Reference Implementation

```python
class ConvNextV2GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, hidden_states):
        # hidden_states: (N, H, W, C)
        # Step 1: Compute L2 norm across spatial dimensions (H, W)
        gx = torch.norm(hidden_states, p=2, dim=(1, 2), keepdim=True)  # (N, 1, 1, C)
        # Step 2: Normalize by channel-wise mean
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)  # (N, 1, 1, C)
        # Step 3: Apply normalization with learnable parameters
        return self.weight * (hidden_states * nx) + self.bias + hidden_states
```

### TTNN Composite Implementation

```python
def ttnn_grn(hidden_states, weight, bias, eps=1e-6):
    """
    Global Response Normalization for ConvNextV2

    Args:
        hidden_states: Tensor of shape [N, H, W, C] in NHWC format
        weight: Learnable weight of shape [1, 1, 1, C]
        bias: Learnable bias of shape [1, 1, 1, C]
        eps: Small epsilon for numerical stability

    Returns:
        Normalized tensor of same shape
    """
    # Step 1: Compute squared values
    x_squared = ttnn.pow(hidden_states, 2)  # [N, H, W, C]

    # Step 2: Sum over spatial dimensions (H, W) -> [N, 1, 1, C]
    # Note: May need to reshape to [N, H*W, C] and sum over dim=1
    gx_squared = ttnn.sum(x_squared, dim=(1, 2), keepdim=True)

    # Step 3: Square root to get L2 norm
    gx = ttnn.sqrt(gx_squared)  # [N, 1, 1, C]

    # Step 4: Compute channel-wise mean of L2 norm
    gx_mean = ttnn.mean(gx, dim=-1, keepdim=True)  # [N, 1, 1, 1]

    # Step 5: Normalize (add eps for numerical stability)
    nx = ttnn.divide(gx, ttnn.add(gx_mean, eps))  # [N, 1, 1, C]

    # Step 6: Apply normalization
    x_normalized = ttnn.multiply(hidden_states, nx)  # [N, H, W, C]

    # Step 7: Apply learnable parameters and add residual
    # output = weight * x_normalized + bias + hidden_states
    weighted = ttnn.multiply(weight, x_normalized)
    with_bias = ttnn.add(weighted, bias)
    output = ttnn.add(with_bias, hidden_states)

    return output
```

### Alternative GRN Implementation (if sum reduction is limited)

```python
def ttnn_grn_alternative(hidden_states, weight, bias, eps=1e-6):
    """
    Alternative GRN implementation using manual reshape
    """
    N, H, W, C = hidden_states.shape

    # Reshape to [N, H*W, C]
    x_flat = ttnn.reshape(hidden_states, (N, H * W, C))

    # Compute squared values
    x_squared = ttnn.pow(x_flat, 2)

    # Sum over spatial dimension (dim=1)
    gx_squared = ttnn.sum(x_squared, dim=1, keepdim=True)  # [N, 1, C]

    # Square root
    gx = ttnn.sqrt(gx_squared)

    # Channel mean
    gx_mean = ttnn.mean(gx, dim=-1, keepdim=True)  # [N, 1, 1]

    # Normalize
    nx = ttnn.divide(gx, ttnn.add(gx_mean, eps))  # [N, 1, C]

    # Apply to flattened input and reshape back
    x_normalized = ttnn.multiply(x_flat, nx)
    x_normalized = ttnn.reshape(x_normalized, (N, H, W, C))

    # Apply parameters and residual
    weighted = ttnn.multiply(weight, x_normalized)
    with_bias = ttnn.add(weighted, bias)
    output = ttnn.add(with_bias, hidden_states)

    return output
```

## Data Format Considerations

### PyTorch vs TTNN Format Flow

```
PyTorch Model (ConvNextV2Layer):
    Input: NCHW [1, C, H, W]
    │
    ├─ dwconv (NCHW)
    │
    ├─ permute to NHWC [1, H, W, C]
    │
    ├─ layernorm, linear, gelu, grn, linear (all NHWC)
    │
    ├─ permute to NCHW [1, C, H, W]
    │
    └─ residual add (NCHW)

TTNN Model (optimized):
    Input: NHWC [1, H, W, C]  ← Can keep NHWC throughout
    │
    ├─ dwconv (NHWC) ← ttnn.conv2d expects NHWC
    │
    ├─ layernorm (NHWC)
    │
    ├─ linear (NHWC)
    │
    ├─ gelu (NHWC)
    │
    ├─ grn (NHWC)
    │
    ├─ linear (NHWC)
    │
    └─ residual add (NHWC)
```

**Optimization Note**: Since TTNN conv2d expects NHWC and LayerNorm/Linear work on the last dimension, we can potentially eliminate the permute operations by keeping data in NHWC throughout.

## Shape Requirements

| Operator | Layout | Alignment | Notes |
|----------|--------|-----------|-------|
| `ttnn.conv2d` | NHWC | Tile (32) | Height/width may need padding |
| `ttnn.linear` | Any | Tile (32) | Inner dim multiple of 32 preferred |
| `ttnn.layer_norm` | TILE_LAYOUT | - | Normalized dim should be last |
| `ttnn.gelu` | TILE_LAYOUT | - | Unary operation |
| `ttnn.add` | TILE_LAYOUT | - | Broadcasting supported |
| `ttnn.global_avg_pool2d` | NHWC | - | Returns [N, 1, 1, C] |

## Unsupported Operations

| Operation | Status | Fallback Strategy |
|-----------|--------|-------------------|
| Drop Path | Not needed | Removed in inference mode |
| GRN | Custom | Composite of basic ops (pow, sum, sqrt, mean, mul, div, add) |

## API Differences

### LayerNorm

- PyTorch: `nn.LayerNorm(normalized_shape)` normalizes over last N dimensions
- TTNN: `ttnn.layer_norm` normalizes over last dimension
- ConvNextV2LayerNorm uses `normalized_shape=(channels,)` which maps directly

### Linear as Pointwise Conv

- In ConvNextV2, Linear layers are used as pointwise (1x1) convolutions
- Input shape: [N, H, W, C]
- Can be implemented as either:
  - `ttnn.linear` after reshaping to [N*H*W, C]
  - Or directly on the 4D tensor if supported

## Checklist

- [x] All PyTorch operators mapped to TTNN equivalents
- [x] Unsupported operators identified (GRN) with fallback plans
- [x] API differences documented (LayerNorm, data format)
- [x] Shape requirements noted (tile alignment, NHWC format)
- [x] Mapping table saved for reference

## Next Step

Proceed to **Step 3: Per-Operator Testing** to validate each operator with PCC >= 0.999.
