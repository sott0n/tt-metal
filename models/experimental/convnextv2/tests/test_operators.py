# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ConvNextV2 Per-Operator Testing (Step 3)

Tests each TTNN operator individually against PyTorch reference with PCC validation.
Target: PCC >= 0.999 for all operators.

Usage:
    pytest models/experimental/convnextv2/tests/test_operators.py -v
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# L1 small size for depthwise convolutions
CONVNEXTV2_L1_SMALL_SIZE = 24 * 1024  # 24 KiB


# ConvNextV2-specific shapes from MODEL_ANALYSIS.md
CONVNEXTV2_SHAPES = {
    "stage0": {"hidden": 128, "spatial": 56},
    "stage1": {"hidden": 256, "spatial": 28},
    "stage2": {"hidden": 512, "spatial": 14},
    "stage3": {"hidden": 1024, "spatial": 7},
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": CONVNEXTV2_L1_SMALL_SIZE}], indirect=True)
class TestConv2dOperators:
    """Test Conv2d operations used in ConvNextV2"""

    def test_patch_embedding_conv(self, device):
        """
        Patch Embedding: Conv2d(3, 128, kernel=4, stride=4)
        Input: [1, 3, 224, 224] (NCHW) -> [1, 128, 56, 56]
        """
        torch.manual_seed(0)
        batch_size = 1
        in_channels = 3
        out_channels = 128
        input_height = 224
        input_width = 224
        kernel_size = 4
        stride = 4

        # PyTorch reference (NCHW)
        torch_input_nchw = torch.randn(batch_size, in_channels, input_height, input_width)
        torch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        torch_output = torch_conv(torch_input_nchw)

        # Convert to NHWC for TTNN
        torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).contiguous()

        # Prepare TTNN tensors
        ttnn_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
        ttnn_weight = ttnn.from_torch(torch_conv.weight, dtype=ttnn.bfloat16)
        ttnn_bias = ttnn.from_torch(torch_conv.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)

        # Run TTNN conv2d
        ttnn_output, [out_h, out_w], _ = ttnn.conv2d(
            input_tensor=ttnn_input,
            weight_tensor=ttnn_weight,
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
            bias_tensor=ttnn_bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(0, 0),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        # Convert back to torch
        output = ttnn.to_torch(ttnn_output)
        output = output.reshape(batch_size, out_h, out_w, out_channels)
        output = output.permute(0, 3, 1, 2)  # NHWC -> NCHW

        assert_with_pcc(torch_output, output, 0.999)

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (128, 56),  # Stage 0
            (256, 28),  # Stage 1
            (512, 14),  # Stage 2
            (1024, 7),  # Stage 3
        ],
    )
    def test_depthwise_conv_7x7(self, device, hidden_dim, spatial_size):
        """
        Depthwise Conv: Conv2d(C, C, kernel=7, padding=3, groups=C)
        """
        torch.manual_seed(0)
        batch_size = 1
        kernel_size = 7
        padding = 3

        # PyTorch reference
        torch_input_nchw = torch.randn(batch_size, hidden_dim, spatial_size, spatial_size)
        torch_conv = torch.nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, groups=hidden_dim
        )
        torch_output = torch_conv(torch_input_nchw)

        # Convert to NHWC
        torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).contiguous()

        # TTNN
        ttnn_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
        ttnn_weight = ttnn.from_torch(torch_conv.weight, dtype=ttnn.bfloat16)
        ttnn_bias = ttnn.from_torch(torch_conv.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)

        ttnn_output, [out_h, out_w], _ = ttnn.conv2d(
            input_tensor=ttnn_input,
            weight_tensor=ttnn_weight,
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            device=device,
            bias_tensor=ttnn_bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(padding, padding),
            batch_size=batch_size,
            input_height=spatial_size,
            input_width=spatial_size,
            groups=hidden_dim,  # Depthwise: groups = channels
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        output = ttnn.to_torch(ttnn_output)
        output = output.reshape(batch_size, out_h, out_w, hidden_dim)
        output = output.permute(0, 3, 1, 2)

        assert_with_pcc(torch_output, output, 0.999)

    @pytest.mark.parametrize(
        "in_channels,out_channels,input_spatial",
        [
            (128, 256, 56),  # Stage 0 -> Stage 1
            (256, 512, 28),  # Stage 1 -> Stage 2
            (512, 1024, 14),  # Stage 2 -> Stage 3
        ],
    )
    def test_downsampling_conv(self, device, in_channels, out_channels, input_spatial):
        """
        Downsampling: Conv2d(C_in, C_out, kernel=2, stride=2)
        """
        torch.manual_seed(0)
        batch_size = 1
        kernel_size = 2
        stride = 2

        torch_input_nchw = torch.randn(batch_size, in_channels, input_spatial, input_spatial)
        torch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        torch_output = torch_conv(torch_input_nchw)

        torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).contiguous()

        ttnn_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
        ttnn_weight = ttnn.from_torch(torch_conv.weight, dtype=ttnn.bfloat16)
        ttnn_bias = ttnn.from_torch(torch_conv.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)

        ttnn_output, [out_h, out_w], _ = ttnn.conv2d(
            input_tensor=ttnn_input,
            weight_tensor=ttnn_weight,
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
            bias_tensor=ttnn_bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(0, 0),
            batch_size=batch_size,
            input_height=input_spatial,
            input_width=input_spatial,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        output = ttnn.to_torch(ttnn_output)
        output = output.reshape(batch_size, out_h, out_w, out_channels)
        output = output.permute(0, 3, 1, 2)

        assert_with_pcc(torch_output, output, 0.999)


class TestLinearOperators:
    """Test Linear operations used in ConvNextV2 (pointwise convolutions)"""

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (128, 56),
            (256, 28),
            (512, 14),
            (1024, 7),
        ],
    )
    def test_pwconv1_expand(self, device, hidden_dim, spatial_size):
        """
        pwconv1: Linear(hidden_dim, hidden_dim * 4) - Expand 4x
        Input shape: [N, H*W, C] or reshaped from [N, H, W, C]
        """
        torch.manual_seed(0)
        batch_size = 1
        in_features = hidden_dim
        out_features = hidden_dim * 4

        # Simulate NHWC input flattened to [N*H*W, C]
        seq_len = spatial_size * spatial_size
        torch_input = torch.randn(batch_size * seq_len, in_features)
        torch_linear = torch.nn.Linear(in_features, out_features)
        torch_output = torch_linear(torch_input)

        # TTNN - reshape for matmul compatibility
        ttnn_input = ttnn.from_torch(
            torch_input.unsqueeze(0),  # [1, N*H*W, C]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Weight transpose for TTNN
        weight = torch_linear.weight.T.contiguous()
        ttnn_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_bias = ttnn.from_torch(
            torch_linear.bias.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        ttnn_output = ttnn.linear(ttnn_input, ttnn_weight, bias=ttnn_bias)
        output = ttnn.to_torch(ttnn_output).squeeze(0)

        assert_with_pcc(torch_output, output, 0.999)

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (128, 56),
            (256, 28),
            (512, 14),
            (1024, 7),
        ],
    )
    def test_pwconv2_contract(self, device, hidden_dim, spatial_size):
        """
        pwconv2: Linear(hidden_dim * 4, hidden_dim) - Contract back
        """
        torch.manual_seed(0)
        batch_size = 1
        in_features = hidden_dim * 4
        out_features = hidden_dim

        seq_len = spatial_size * spatial_size
        torch_input = torch.randn(batch_size * seq_len, in_features)
        torch_linear = torch.nn.Linear(in_features, out_features)
        torch_output = torch_linear(torch_input)

        ttnn_input = ttnn.from_torch(
            torch_input.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        weight = torch_linear.weight.T.contiguous()
        ttnn_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_bias = ttnn.from_torch(
            torch_linear.bias.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        ttnn_output = ttnn.linear(ttnn_input, ttnn_weight, bias=ttnn_bias)
        output = ttnn.to_torch(ttnn_output).squeeze(0)

        assert_with_pcc(torch_output, output, 0.999)

    def test_classifier_linear(self, device):
        """
        Classifier: Linear(1024, 1000)
        """
        torch.manual_seed(0)
        batch_size = 1
        in_features = 1024
        out_features = 1000

        torch_input = torch.randn(batch_size, in_features)
        torch_linear = torch.nn.Linear(in_features, out_features)
        torch_output = torch_linear(torch_input)

        ttnn_input = ttnn.from_torch(
            torch_input.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        weight = torch_linear.weight.T.contiguous()
        ttnn_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_bias = ttnn.from_torch(
            torch_linear.bias.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        ttnn_output = ttnn.linear(ttnn_input, ttnn_weight, bias=ttnn_bias)
        output = ttnn.to_torch(ttnn_output).squeeze(0)

        assert_with_pcc(torch_output, output, 0.999)


class TestLayerNormOperators:
    """Test LayerNorm operations used in ConvNextV2 (channels-last format)"""

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (128, 56),
            (256, 28),
            (512, 14),
            (1024, 7),
        ],
    )
    def test_channels_last_layernorm(self, device, hidden_dim, spatial_size):
        """
        ConvNextV2LayerNorm: LayerNorm over channels dimension (last dim)
        Input shape: [N, H, W, C] - channels-last format
        """
        torch.manual_seed(0)
        batch_size = 1

        # NHWC format
        torch_input = torch.randn(batch_size, spatial_size, spatial_size, hidden_dim)
        torch_ln = torch.nn.LayerNorm(hidden_dim, eps=1e-6)
        torch_output = torch_ln(torch_input)

        # Flatten to [N*H*W, C] for TTNN
        flat_input = torch_input.reshape(-1, hidden_dim)
        ttnn_input = ttnn.from_torch(
            flat_input.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        ttnn_weight = ttnn.from_torch(
            torch_ln.weight.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_bias = ttnn.from_torch(
            torch_ln.bias.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        ttnn_output = ttnn.layer_norm(ttnn_input, weight=ttnn_weight, bias=ttnn_bias, epsilon=1e-6)
        output = ttnn.to_torch(ttnn_output).squeeze(0)
        output = output.reshape(batch_size, spatial_size, spatial_size, hidden_dim)

        assert_with_pcc(torch_output, output, 0.999)

    def test_final_layernorm(self, device):
        """
        Final LayerNorm: standard LayerNorm(1024)
        Input shape: [N, C] after global pooling
        """
        torch.manual_seed(0)
        batch_size = 1
        hidden_dim = 1024

        torch_input = torch.randn(batch_size, hidden_dim)
        torch_ln = torch.nn.LayerNorm(hidden_dim)
        torch_output = torch_ln(torch_input)

        ttnn_input = ttnn.from_torch(
            torch_input.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_weight = ttnn.from_torch(
            torch_ln.weight.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_bias = ttnn.from_torch(
            torch_ln.bias.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        ttnn_output = ttnn.layer_norm(ttnn_input, weight=ttnn_weight, bias=ttnn_bias)
        output = ttnn.to_torch(ttnn_output).squeeze(0)

        assert_with_pcc(torch_output, output, 0.999)


class TestActivationOperators:
    """Test activation functions used in ConvNextV2"""

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (512, 56),  # Stage 0: 128*4
            (1024, 28),  # Stage 1: 256*4
            (2048, 14),  # Stage 2: 512*4
            (4096, 7),  # Stage 3: 1024*4
        ],
    )
    def test_gelu(self, device, hidden_dim, spatial_size):
        """
        GELU activation after pwconv1 (expand)
        Input shape: [N*H*W, hidden_dim*4] or [N, H, W, hidden_dim*4]
        """
        torch.manual_seed(0)
        batch_size = 1

        torch_input = torch.randn(batch_size, spatial_size, spatial_size, hidden_dim)
        torch_output = torch.nn.functional.gelu(torch_input)

        # Flatten for TTNN
        flat_input = torch_input.reshape(-1, hidden_dim)
        ttnn_input = ttnn.from_torch(
            flat_input.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        ttnn_output = ttnn.gelu(ttnn_input)
        output = ttnn.to_torch(ttnn_output).squeeze(0)
        output = output.reshape(batch_size, spatial_size, spatial_size, hidden_dim)

        assert_with_pcc(torch_output, output, 0.999)


class TestGRNOperator:
    """
    Test GRN (Global Response Normalization) - Custom composite operation

    This is the most critical test as GRN is not a native TTNN operation
    and must be implemented as a composite of basic operations.

    PyTorch GRN:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # L2 norm over spatial
        nx = gx / (gx.mean(dim=-1, keepdim=True) + eps)
        return weight * (x * nx) + bias + x
    """

    @staticmethod
    def pytorch_grn(x, weight, bias, eps=1e-6):
        """PyTorch reference implementation of GRN"""
        # x: [N, H, W, C]
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # [N, 1, 1, C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + eps)  # [N, 1, 1, C]
        return weight * (x * nx) + bias + x

    @staticmethod
    def ttnn_grn(x, weight, bias, device, eps=1e-6):
        """
        TTNN composite implementation of GRN

        Strategy: Implement using available TTNN primitives
        """
        # Get shape info
        x_torch = ttnn.to_torch(x)
        N, H, W, C = x_torch.shape

        # Reshape to [N, H*W, C] for operations
        x_flat = ttnn.reshape(x, (N, H * W, C))

        # Step 1: Compute x^2
        x_squared = ttnn.pow(x_flat, 2)

        # Step 2: Sum over spatial dimension -> [N, 1, C]
        # Note: ttnn.sum behavior may vary, implement manually if needed
        x_squared_torch = ttnn.to_torch(x_squared)
        gx_squared_torch = x_squared_torch.sum(dim=1, keepdim=True)  # [N, 1, C]

        gx_squared = ttnn.from_torch(gx_squared_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Step 3: Square root -> L2 norm
        gx = ttnn.sqrt(gx_squared)  # [N, 1, C]

        # Step 4: Compute mean over channels
        gx_torch = ttnn.to_torch(gx)
        gx_mean_torch = gx_torch.mean(dim=-1, keepdim=True)  # [N, 1, 1]
        gx_mean = ttnn.from_torch(gx_mean_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Step 5: Normalize
        gx_mean_eps = ttnn.add(gx_mean, eps)
        nx = ttnn.div(gx, gx_mean_eps)  # [N, 1, C]

        # Step 6: Broadcast nx to spatial dimensions and multiply
        nx_torch = ttnn.to_torch(nx)
        nx_expanded_torch = nx_torch.expand(N, H * W, C)
        nx_expanded = ttnn.from_torch(nx_expanded_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        x_normalized = ttnn.mul(x_flat, nx_expanded)  # [N, H*W, C]

        # Step 7: Reshape back to [N, H, W, C]
        x_normalized_4d = ttnn.reshape(x_normalized, (N, H, W, C))

        # Step 8: Apply learnable parameters
        # weight, bias shape: [1, 1, 1, C]
        weighted = ttnn.mul(weight, x_normalized_4d)
        with_bias = ttnn.add(weighted, bias)

        # Step 9: Add residual
        output = ttnn.add(with_bias, x)

        return output

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (512, 56),  # Stage 0: 128*4
            (1024, 28),  # Stage 1: 256*4
            (2048, 14),  # Stage 2: 512*4
            (4096, 7),  # Stage 3: 1024*4
        ],
    )
    def test_grn_composite(self, device, hidden_dim, spatial_size):
        """
        Test GRN composite implementation against PyTorch reference
        """
        torch.manual_seed(0)
        batch_size = 1

        # Input in NHWC format
        torch_input = torch.randn(batch_size, spatial_size, spatial_size, hidden_dim)
        torch_weight = torch.randn(1, 1, 1, hidden_dim) * 0.1  # Small init like in model
        torch_bias = torch.randn(1, 1, 1, hidden_dim) * 0.1

        # PyTorch reference
        torch_output = self.pytorch_grn(torch_input, torch_weight, torch_bias)

        # TTNN
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn_output = self.ttnn_grn(ttnn_input, ttnn_weight, ttnn_bias, device)
        output = ttnn.to_torch(ttnn_output)

        assert_with_pcc(torch_output, output, 0.999)


class TestResidualOperators:
    """Test residual add operations"""

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (128, 56),
            (256, 28),
            (512, 14),
            (1024, 7),
        ],
    )
    def test_residual_add(self, device, hidden_dim, spatial_size):
        """
        Residual addition at end of ConvNextV2Layer
        Both inputs: [N, C, H, W] (NCHW format after permute)
        """
        torch.manual_seed(0)
        batch_size = 1

        torch_input = torch.randn(batch_size, hidden_dim, spatial_size, spatial_size)
        torch_residual = torch.randn(batch_size, hidden_dim, spatial_size, spatial_size)
        torch_output = torch_input + torch_residual

        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn_output = ttnn.add(ttnn_input, ttnn_residual)
        output = ttnn.to_torch(ttnn_output)

        assert_with_pcc(torch_output, output, 0.999)


class TestPoolingOperators:
    """Test global pooling operations"""

    def test_global_avg_pool(self, device):
        """
        Global Average Pool: AdaptiveAvgPool2d((1, 1))
        Input: [N, C, H, W] -> [N, C, 1, 1]
        """
        torch.manual_seed(0)
        batch_size = 1
        hidden_dim = 1024
        spatial_size = 7

        torch_input = torch.randn(batch_size, hidden_dim, spatial_size, spatial_size)
        torch_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        torch_output = torch_pool(torch_input)

        # Convert to NHWC for TTNN
        torch_input_nhwc = torch_input.permute(0, 2, 3, 1).contiguous()

        ttnn_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn_output = ttnn.global_avg_pool2d(ttnn_input)
        output = ttnn.to_torch(ttnn_output)

        # TTNN output is [N, 1, 1, C], convert to [N, C, 1, 1]
        output = output.permute(0, 3, 1, 2)

        assert_with_pcc(torch_output, output, 0.999)


class TestPermuteOperators:
    """Test permute operations for format conversion"""

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (128, 56),
            (256, 28),
            (512, 14),
            (1024, 7),
        ],
    )
    def test_nchw_to_nhwc(self, device, hidden_dim, spatial_size):
        """
        NCHW to NHWC: permute(0, 2, 3, 1)
        """
        torch.manual_seed(0)
        batch_size = 1

        torch_input = torch.randn(batch_size, hidden_dim, spatial_size, spatial_size)
        torch_output = torch_input.permute(0, 2, 3, 1).contiguous()

        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_output = ttnn.permute(ttnn_input, (0, 2, 3, 1))
        output = ttnn.to_torch(ttnn_output)

        assert_with_pcc(torch_output, output, 0.999)

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (128, 56),
            (256, 28),
            (512, 14),
            (1024, 7),
        ],
    )
    def test_nhwc_to_nchw(self, device, hidden_dim, spatial_size):
        """
        NHWC to NCHW: permute(0, 3, 1, 2)
        """
        torch.manual_seed(0)
        batch_size = 1

        torch_input = torch.randn(batch_size, spatial_size, spatial_size, hidden_dim)
        torch_output = torch_input.permute(0, 3, 1, 2).contiguous()

        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_output = ttnn.permute(ttnn_input, (0, 3, 1, 2))
        output = ttnn.to_torch(ttnn_output)

        assert_with_pcc(torch_output, output, 0.999)


# Debugging utilities
def debug_pcc(torch_output, ttnn_output, name=""):
    """Debug helper for PCC issues"""
    import numpy as np
    from scipy.stats import pearsonr

    t1 = torch_output.flatten().float().numpy()
    t2 = ttnn_output.flatten().float().numpy()

    pcc, _ = pearsonr(t1, t2)

    print(f"\n{name} Debug Info:")
    print(f"  PCC: {pcc:.6f}")
    print(f"  Torch - min: {t1.min():.4f}, max: {t1.max():.4f}, mean: {t1.mean():.4f}")
    print(f"  TTNN  - min: {t2.min():.4f}, max: {t2.max():.4f}, mean: {t2.mean():.4f}")
    print(f"  Max diff: {np.abs(t1 - t2).max():.6f}")
    print(f"  Mean diff: {np.abs(t1 - t2).mean():.6f}")

    diff = np.abs(t1 - t2)
    max_idx = np.argmax(diff)
    print(f"  Max diff at idx {max_idx}: torch={t1[max_idx]:.4f}, ttnn={t2[max_idx]:.4f}")

    return pcc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
