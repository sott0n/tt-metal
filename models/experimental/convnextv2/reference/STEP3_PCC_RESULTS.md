# Step 3: Per-Operator Testing Results

## Summary

**Status: ALL PASSED (43/43)**

All operators pass PCC >= 0.999 threshold.

| Status | Count |
|--------|-------|
| PASSED | 43 |
| FAILED | 0 |
| **Total** | **43** |

## PCC Results by Operator

### Activation Functions

| Operator | PyTorch | TTNN | Shapes Tested | PCC Status |
|----------|---------|------|---------------|------------|
| GELU | nn.GELU | ttnn.gelu | 512×56², 1024×28², 2048×14², 4096×7² | ✅ >= 0.999 |

### Linear Operations

| Operator | PyTorch | TTNN | Shapes Tested | PCC Status |
|----------|---------|------|---------------|------------|
| pwconv1 (expand) | nn.Linear | ttnn.linear | 128→512, 256→1024, 512→2048, 1024→4096 | ✅ >= 0.999 |
| pwconv2 (contract) | nn.Linear | ttnn.linear | 512→128, 1024→256, 2048→512, 4096→1024 | ✅ >= 0.999 |
| Classifier | nn.Linear | ttnn.linear | 1024→1000 | ✅ >= 0.999 |

### Normalization Operations

| Operator | PyTorch | TTNN | Shapes Tested | PCC Status |
|----------|---------|------|---------------|------------|
| LayerNorm (channels-last) | nn.LayerNorm | ttnn.layer_norm | 128×56², 256×28², 512×14², 1024×7² | ✅ >= 0.999 |
| Final LayerNorm | nn.LayerNorm | ttnn.layer_norm | 1024 | ✅ >= 0.999 |

### GRN (Custom Composite)

| Operator | PyTorch | TTNN | Shapes Tested | PCC Status |
|----------|---------|------|---------------|------------|
| GRN | ConvNextV2GRN | ttnn_grn (composite) | 512×56², 1024×28², 2048×14², 4096×7² | ✅ >= 0.999 |

### Conv2d Operations

| Operator | PyTorch | TTNN | Shapes Tested | PCC Status |
|----------|---------|------|---------------|------------|
| Patch Embedding | Conv2d(3→128, k=4, s=4) | ttnn.conv2d | 224×224 → 56×56 | ✅ >= 0.999 |
| Depthwise Conv 7×7 | Conv2d(groups=C) | ttnn.conv2d | 128×56², 256×28², 512×14², 1024×7² | ✅ >= 0.999 |
| Downsampling Conv | Conv2d(k=2, s=2) | ttnn.conv2d | 128→256, 256→512, 512→1024 | ✅ >= 0.999 |

### Residual Operations

| Operator | PyTorch | TTNN | Shapes Tested | PCC Status |
|----------|---------|------|---------------|------------|
| Residual Add | + | ttnn.add | 128×56², 256×28², 512×14², 1024×7² | ✅ >= 0.999 |

### Pooling Operations

| Operator | PyTorch | TTNN | Shapes Tested | PCC Status |
|----------|---------|------|---------------|------------|
| Global Avg Pool | AdaptiveAvgPool2d((1,1)) | ttnn.global_avg_pool2d | 1024×7² → 1×1 | ✅ >= 0.999 |

### Layout Operations

| Operator | PyTorch | TTNN | Shapes Tested | PCC Status |
|----------|---------|------|---------------|------------|
| NCHW → NHWC | permute(0,2,3,1) | ttnn.permute | 128×56², 256×28², 512×14², 1024×7² | ✅ >= 0.999 |
| NHWC → NCHW | permute(0,3,1,2) | ttnn.permute | 128×56², 256×28², 512×14², 1024×7² | ✅ >= 0.999 |

## Device Configuration

Tests for Conv2d operators (including depthwise) require `l1_small_size` configuration:

```python
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24 * 1024}], indirect=True)
class TestConv2dOperators:
    ...
```

This configures L1_SMALL buffer size for depthwise convolutions with `groups=channels`.

## Running Tests

```bash
# Run all operator tests
pytest models/experimental/convnextv2/tests/test_operators.py -v

# Run specific test class
pytest models/experimental/convnextv2/tests/test_operators.py::TestConv2dOperators -v
```

## Checklist

- [x] Test infrastructure set up
- [x] Unit tests written for each operator
- [x] GELU tests pass with PCC >= 0.999
- [x] Linear tests pass with PCC >= 0.999
- [x] LayerNorm tests pass with PCC >= 0.999
- [x] GRN composite tests pass with PCC >= 0.999
- [x] Add tests pass with PCC >= 0.999
- [x] Permute tests pass with PCC >= 0.999
- [x] Global pooling tests pass with PCC >= 0.999
- [x] Patch embedding conv tests pass with PCC >= 0.999
- [x] Depthwise conv tests pass with PCC >= 0.999
- [x] Downsampling conv tests pass with PCC >= 0.999

## Conclusion

**All 43 tests passed with PCC >= 0.999**.

All operators are validated and ready for module implementation.
