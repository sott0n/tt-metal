# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ConvNextV2 E2E Model Testing (Step 7)

Tests the complete ConvNextV2 model end-to-end against HuggingFace reference.
Target: PCC >= 0.99 for full model output.

Usage:
    pytest models/experimental/convnextv2/tests/test_e2e_model.py -v
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# L1 small size for depthwise convolutions
CONVNEXTV2_L1_SMALL_SIZE = 24 * 1024  # 24 KiB


@pytest.fixture(scope="module")
def torch_model():
    """Load HuggingFace ConvNextV2 model"""
    from transformers import ConvNextV2ForImageClassification

    model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-1k-224")
    model.eval()
    return model


@pytest.fixture(scope="module")
def model_config():
    """ConvNextV2-base configuration"""
    return {
        "hidden_sizes": [128, 256, 512, 1024],
        "depths": [3, 3, 27, 3],
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": CONVNEXTV2_L1_SMALL_SIZE}], indirect=True)
class TestConvNextV2E2EModel:
    """Test full ConvNextV2 model end-to-end"""

    def test_embeddings_e2e(self, device, torch_model):
        """Test embeddings module in E2E context"""
        from models.experimental.convnextv2.tt.ttnn_convnextv2_model import (
            ttnn_convnextv2_embeddings,
            create_convnextv2_parameters,
        )

        torch.manual_seed(0)
        batch_size = 1

        # Create input
        torch_input = torch.randn(batch_size, 3, 224, 224)

        # PyTorch reference
        embeddings_module = torch_model.convnextv2.embeddings
        with torch.no_grad():
            torch_output = embeddings_module(torch_input)

        # Create TTNN parameters (full model, but we only use embeddings)
        parameters = create_convnextv2_parameters(torch_model, device)

        # TTNN
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn_output = ttnn_convnextv2_embeddings(
            ttnn_input,
            parameters=parameters.convnextv2.embeddings,
            device=device,
            batch_size=batch_size,
        )

        output = ttnn.to_torch(ttnn_output)

        assert_with_pcc(torch_output, output, 0.99)

    def test_single_stage(self, device, torch_model, model_config):
        """Test a single encoder stage"""
        from models.experimental.convnextv2.tt.ttnn_convnextv2_model import (
            ttnn_convnextv2_stage,
            create_convnextv2_parameters,
        )

        torch.manual_seed(0)
        batch_size = 1
        stage_idx = 0
        hidden_dim = 128
        spatial_size = 56
        num_layers = 3  # Stage 0 has 3 layers

        # Create input (output of embeddings)
        torch_input = torch.randn(batch_size, hidden_dim, spatial_size, spatial_size)

        # PyTorch reference - process through stage 0
        stage_module = torch_model.convnextv2.encoder.stages[stage_idx]
        with torch.no_grad():
            x = torch_input.clone()
            for layer in stage_module.layers:
                x = layer(x)
        torch_output = x

        # Create TTNN parameters
        parameters = create_convnextv2_parameters(torch_model, device)

        # TTNN
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn_output = ttnn_convnextv2_stage(
            ttnn_input,
            parameters=parameters.convnextv2.encoder.stages[stage_idx],
            device=device,
            stage_idx=stage_idx,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            height=spatial_size,
            width=spatial_size,
            downsample=False,  # Stage 0 has no downsampling
        )

        output = ttnn.to_torch(ttnn_output)

        # Lower threshold for accumulated error across 3 layers
        assert_with_pcc(torch_output, output, 0.98)

    def test_full_model_inference(self, device, torch_model, model_config):
        """Test complete model inference"""
        from models.experimental.convnextv2.tt.ttnn_convnextv2_model import (
            ttnn_convnextv2_for_image_classification,
            create_convnextv2_parameters,
        )

        torch.manual_seed(0)
        batch_size = 1

        # Create input image
        torch_input = torch.randn(batch_size, 3, 224, 224)

        # PyTorch reference - full model
        with torch.no_grad():
            torch_output = torch_model(torch_input)
        torch_logits = torch_output.logits

        # Create TTNN parameters
        parameters = create_convnextv2_parameters(torch_model, device)

        # TTNN
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn_logits = ttnn_convnextv2_for_image_classification(
            ttnn_input,
            parameters=parameters,
            device=device,
            config=model_config,
            batch_size=batch_size,
        )

        output = ttnn.to_torch(ttnn_logits)

        # Check shape
        assert output.shape == torch_logits.shape, f"Shape mismatch: {output.shape} vs {torch_logits.shape}"

        # Check PCC - use lower threshold for full model (36+ layers, accumulated error)
        assert_with_pcc(torch_logits, output, 0.90)

    def test_top5_accuracy(self, device, torch_model, model_config):
        """Test top-5 classification accuracy matches"""
        from models.experimental.convnextv2.tt.ttnn_convnextv2_model import (
            ttnn_convnextv2_for_image_classification,
            create_convnextv2_parameters,
        )

        torch.manual_seed(42)
        batch_size = 1

        # Create input image
        torch_input = torch.randn(batch_size, 3, 224, 224)

        # PyTorch reference
        with torch.no_grad():
            torch_output = torch_model(torch_input)
        torch_logits = torch_output.logits
        torch_top5 = torch.topk(torch_logits, k=5, dim=1).indices[0].tolist()

        # Create TTNN parameters
        parameters = create_convnextv2_parameters(torch_model, device)

        # TTNN
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn_logits = ttnn_convnextv2_for_image_classification(
            ttnn_input,
            parameters=parameters,
            device=device,
            config=model_config,
            batch_size=batch_size,
        )

        output = ttnn.to_torch(ttnn_logits)
        ttnn_top5 = torch.topk(output, k=5, dim=1).indices[0].tolist()

        # Check if top-1 or top-5 matches
        top1_match = torch_top5[0] == ttnn_top5[0]
        top5_overlap = len(set(torch_top5) & set(ttnn_top5))

        print(f"PyTorch top-5: {torch_top5}")
        print(f"TTNN top-5: {ttnn_top5}")
        print(f"Top-1 match: {top1_match}, Top-5 overlap: {top5_overlap}/5")

        # At minimum, top-1 should match or 3/5 of top-5 should overlap
        assert top1_match or top5_overlap >= 3, f"Classification mismatch: top1={top1_match}, overlap={top5_overlap}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": CONVNEXTV2_L1_SMALL_SIZE}], indirect=True)
class TestConvNextV2Performance:
    """Performance-related tests"""

    def test_inference_runs(self, device, torch_model, model_config):
        """Basic test that inference completes without errors"""
        from models.experimental.convnextv2.tt.ttnn_convnextv2_model import (
            ttnn_convnextv2_for_image_classification,
            create_convnextv2_parameters,
        )

        torch.manual_seed(0)
        batch_size = 1

        # Create input
        torch_input = torch.randn(batch_size, 3, 224, 224)

        # Create TTNN parameters
        parameters = create_convnextv2_parameters(torch_model, device)

        # TTNN inference
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Should complete without errors
        output = ttnn_convnextv2_for_image_classification(
            ttnn_input,
            parameters=parameters,
            device=device,
            config=model_config,
            batch_size=batch_size,
        )

        result = ttnn.to_torch(output)

        # Basic sanity checks
        assert result.shape == (1, 1000), f"Unexpected output shape: {result.shape}"
        assert not torch.isnan(result).any(), "Output contains NaN values"
        assert not torch.isinf(result).any(), "Output contains Inf values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
