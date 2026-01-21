# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ConvNextV2 Per-Module Testing (Step 5)

Tests each TTNN module against PyTorch reference with PCC validation.
Target: PCC >= 0.99 for composed modules.

Usage:
    pytest models/experimental/convnextv2/tests/test_modules.py -v
"""

import pytest
import torch
import ttnn
from types import SimpleNamespace
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.convnextv2.tt.ttnn_convnextv2_layer import (
    ttnn_grn,
    ttnn_convnextv2_layer,
)
from models.experimental.convnextv2.reference.torch_functional import (
    torch_grn,
    torch_convnextv2_layer,
    torch_convnextv2_embeddings,
)

# L1 small size for depthwise convolutions (similar to MobileNetV2)
CONVNEXTV2_L1_SMALL_SIZE = 24 * 1024  # 24 KiB


@pytest.fixture(scope="module")
def torch_model():
    """Load HuggingFace ConvNextV2 model"""
    from transformers import ConvNextV2ForImageClassification

    model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-1k-224")
    model.eval()
    return model


class TestGRNModule:
    """Test GRN (Global Response Normalization) module"""

    @pytest.mark.parametrize(
        "hidden_dim,spatial_size",
        [
            (512, 56),  # Stage 0: 128*4
            (1024, 28),  # Stage 1: 256*4
            (2048, 14),  # Stage 2: 512*4
            (4096, 7),  # Stage 3: 1024*4
        ],
    )
    def test_grn_module(self, device, hidden_dim, spatial_size):
        """Test GRN module against PyTorch reference"""
        torch.manual_seed(0)
        batch_size = 1

        # Input in NHWC format
        torch_input = torch.randn(batch_size, spatial_size, spatial_size, hidden_dim)
        torch_weight = torch.randn(1, 1, 1, hidden_dim) * 0.1
        torch_bias = torch.randn(1, 1, 1, hidden_dim) * 0.1

        # PyTorch reference
        with torch.no_grad():
            torch_output = torch_grn(torch_input, torch_weight, torch_bias)

        # TTNN
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn_output = ttnn_grn(ttnn_input, ttnn_weight, ttnn_bias, device=device)
        output = ttnn.to_torch(ttnn_output)

        assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": CONVNEXTV2_L1_SMALL_SIZE}], indirect=True)
class TestConvNextV2LayerModule:
    """Test ConvNextV2Layer module (main building block)"""

    @staticmethod
    def create_layer_parameters(layer_module, device):
        """Create TTNN parameters from PyTorch layer module"""

        def to_ttnn(tensor, transpose=False, add_dims=0):
            if transpose:
                tensor = tensor.T.contiguous()
            for _ in range(add_dims):
                tensor = tensor.unsqueeze(0)
            return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        return SimpleNamespace(
            dwconv=SimpleNamespace(
                weight=ttnn.from_torch(layer_module.dwconv.weight, dtype=ttnn.bfloat16),
                bias=ttnn.from_torch(layer_module.dwconv.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
            ),
            layernorm=SimpleNamespace(
                weight=to_ttnn(layer_module.layernorm.weight, add_dims=1),
                bias=to_ttnn(layer_module.layernorm.bias, add_dims=1),
            ),
            pwconv1=SimpleNamespace(
                weight=to_ttnn(layer_module.pwconv1.weight, transpose=True),
                bias=to_ttnn(layer_module.pwconv1.bias, add_dims=1),
            ),
            grn=SimpleNamespace(
                weight=to_ttnn(layer_module.grn.weight),
                bias=to_ttnn(layer_module.grn.bias),
            ),
            pwconv2=SimpleNamespace(
                weight=to_ttnn(layer_module.pwconv2.weight, transpose=True),
                bias=to_ttnn(layer_module.pwconv2.bias, add_dims=1),
            ),
        )

    @pytest.mark.parametrize(
        "stage_idx,hidden_dim,spatial_size",
        [
            (0, 128, 56),
            (1, 256, 28),
            (2, 512, 14),
            (3, 1024, 7),
        ],
    )
    def test_convnextv2_layer(self, device, torch_model, stage_idx, hidden_dim, spatial_size):
        """Test ConvNextV2Layer against HuggingFace reference"""
        torch.manual_seed(0)
        batch_size = 1

        # Get the layer from the model
        layer_module = torch_model.convnextv2.encoder.stages[stage_idx].layers[0]

        # Create input in NCHW format
        torch_input = torch.randn(batch_size, hidden_dim, spatial_size, spatial_size)

        # PyTorch reference (using the actual layer module)
        with torch.no_grad():
            torch_output = layer_module(torch_input)

        # Create TTNN parameters
        parameters = self.create_layer_parameters(layer_module, device)

        # Convert input to TTNN
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Run TTNN layer
        ttnn_output = ttnn_convnextv2_layer(
            ttnn_input,
            parameters=parameters,
            device=device,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            height=spatial_size,
            width=spatial_size,
        )

        output = ttnn.to_torch(ttnn_output)

        # Use 0.99 threshold for composed modules (accumulated numerical error)
        assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": CONVNEXTV2_L1_SMALL_SIZE}], indirect=True)
class TestConvNextV2EmbeddingsModule:
    """Test ConvNextV2Embeddings module"""

    @staticmethod
    def create_embeddings_parameters(embeddings_module, device):
        """Create TTNN parameters from PyTorch embeddings module"""

        def to_ttnn(tensor, add_dims=0):
            for _ in range(add_dims):
                tensor = tensor.unsqueeze(0)
            return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        return SimpleNamespace(
            patch_embeddings=SimpleNamespace(
                weight=ttnn.from_torch(embeddings_module.patch_embeddings.weight, dtype=ttnn.bfloat16),
                bias=ttnn.from_torch(embeddings_module.patch_embeddings.bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
            ),
            layernorm=SimpleNamespace(
                weight=to_ttnn(embeddings_module.layernorm.weight, add_dims=1),
                bias=to_ttnn(embeddings_module.layernorm.bias, add_dims=1),
            ),
        )

    def test_embeddings(self, device, torch_model):
        """Test ConvNextV2Embeddings against HuggingFace reference"""
        torch.manual_seed(0)
        batch_size = 1

        # Create input image
        torch_input = torch.randn(batch_size, 3, 224, 224)

        # Get embeddings module
        embeddings_module = torch_model.convnextv2.embeddings

        # PyTorch reference
        with torch.no_grad():
            torch_output = embeddings_module(torch_input)

        # Create TTNN parameters
        parameters = self.create_embeddings_parameters(embeddings_module, device)

        # TTNN
        from models.experimental.convnextv2.tt.ttnn_convnextv2_model import ttnn_convnextv2_embeddings

        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn_output = ttnn_convnextv2_embeddings(
            ttnn_input,
            parameters=parameters,
            device=device,
            batch_size=batch_size,
        )

        output = ttnn.to_torch(ttnn_output)

        assert_with_pcc(torch_output, output, 0.99)


class TestConvNextV2FunctionalReference:
    """Test functional PyTorch reference against HuggingFace model"""

    def test_functional_grn_matches_hf(self, torch_model):
        """Verify functional GRN matches HuggingFace implementation"""
        torch.manual_seed(0)

        # Get a GRN module from the model
        grn_module = torch_model.convnextv2.encoder.stages[0].layers[0].grn

        # Input in NHWC format (same as GRN expects)
        hidden_states = torch.randn(1, 56, 56, 512)

        # HuggingFace GRN
        with torch.no_grad():
            hf_output = grn_module(hidden_states)

        # Functional GRN
        functional_output = torch_grn(hidden_states, grn_module.weight, grn_module.bias)

        # Should match exactly (same algorithm)
        assert_with_pcc(hf_output, functional_output, 0.9999)

    def test_functional_layer_matches_hf(self, torch_model):
        """Verify functional layer matches HuggingFace implementation"""
        torch.manual_seed(0)

        # Get layer from model
        layer_module = torch_model.convnextv2.encoder.stages[0].layers[0]

        # Input in NCHW format
        hidden_states = torch.randn(1, 128, 56, 56)

        # HuggingFace layer
        with torch.no_grad():
            hf_output = layer_module(hidden_states)

        # Functional layer
        functional_output = torch_convnextv2_layer(
            hidden_states,
            dwconv_weight=layer_module.dwconv.weight,
            dwconv_bias=layer_module.dwconv.bias,
            layernorm_weight=layer_module.layernorm.weight,
            layernorm_bias=layer_module.layernorm.bias,
            pwconv1_weight=layer_module.pwconv1.weight,
            pwconv1_bias=layer_module.pwconv1.bias,
            grn_weight=layer_module.grn.weight,
            grn_bias=layer_module.grn.bias,
            pwconv2_weight=layer_module.pwconv2.weight,
            pwconv2_bias=layer_module.pwconv2.bias,
        )

        assert_with_pcc(hf_output, functional_output, 0.9999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": CONVNEXTV2_L1_SMALL_SIZE}], indirect=True)
class TestMultipleLayersAccumulation:
    """Test error accumulation across multiple layers"""

    def test_three_layers_stage0(self, device, torch_model):
        """Test processing 3 layers in stage 0 (like actual model)"""
        torch.manual_seed(0)
        batch_size = 1
        hidden_dim = 128
        spatial_size = 56

        # Create input
        torch_input = torch.randn(batch_size, hidden_dim, spatial_size, spatial_size)

        # Process through all 3 layers in stage 0
        x = torch_input.clone()
        for layer_idx in range(3):
            layer_module = torch_model.convnextv2.encoder.stages[0].layers[layer_idx]
            with torch.no_grad():
                x = layer_module(x)
        torch_output = x

        # TTNN - process through all 3 layers
        x_ttnn = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        for layer_idx in range(3):
            layer_module = torch_model.convnextv2.encoder.stages[0].layers[layer_idx]
            parameters = TestConvNextV2LayerModule.create_layer_parameters(layer_module, device)

            x_ttnn = ttnn_convnextv2_layer(
                x_ttnn,
                parameters=parameters,
                device=device,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                height=spatial_size,
                width=spatial_size,
            )

        output = ttnn.to_torch(x_ttnn)

        # Lower threshold for accumulated error across 3 layers
        assert_with_pcc(torch_output, output, 0.98)


# Debugging utilities
def debug_layer_step_by_step(hidden_states, layer_module, device):
    """Debug layer by comparing intermediate outputs"""
    import numpy as np
    from scipy.stats import pearsonr

    print("\n=== Layer Step-by-Step Debug ===")

    batch_size, hidden_dim, H, W = hidden_states.shape

    # Step 1: Depthwise conv
    torch_dwconv = torch.nn.functional.conv2d(
        hidden_states, layer_module.dwconv.weight, layer_module.dwconv.bias, padding=3, groups=hidden_dim
    )
    print(f"Dwconv output shape: {torch_dwconv.shape}")

    # Step 2: Permute to NHWC
    torch_nhwc = torch_dwconv.permute(0, 2, 3, 1)
    print(f"After permute (NHWC): {torch_nhwc.shape}")

    # Step 3: LayerNorm
    torch_ln = torch.nn.functional.layer_norm(
        torch_nhwc, (hidden_dim,), layer_module.layernorm.weight, layer_module.layernorm.bias, eps=1e-6
    )
    print(f"After LayerNorm: {torch_ln.shape}")

    # Step 4: pwconv1
    torch_pw1 = torch.nn.functional.linear(torch_ln, layer_module.pwconv1.weight, layer_module.pwconv1.bias)
    print(f"After pwconv1: {torch_pw1.shape}")

    # Step 5: GELU
    torch_gelu = torch.nn.functional.gelu(torch_pw1)
    print(f"After GELU: {torch_gelu.shape}")

    # Step 6: GRN
    torch_grn_out = torch_grn(torch_gelu, layer_module.grn.weight, layer_module.grn.bias)
    print(f"After GRN: {torch_grn_out.shape}")

    # Step 7: pwconv2
    torch_pw2 = torch.nn.functional.linear(torch_grn_out, layer_module.pwconv2.weight, layer_module.pwconv2.bias)
    print(f"After pwconv2: {torch_pw2.shape}")

    # Step 8: Permute back to NCHW
    torch_nchw = torch_pw2.permute(0, 3, 1, 2)
    print(f"After permute (NCHW): {torch_nchw.shape}")

    # Step 9: Residual
    torch_final = torch_nchw + hidden_states
    print(f"Final output: {torch_final.shape}")

    return torch_final


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
