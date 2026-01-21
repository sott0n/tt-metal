# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ConvNextV2 Functional PyTorch Reference Implementation

This module provides functional (non-class-based) PyTorch implementations
of ConvNextV2 modules for testing against TTNN implementations.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def torch_grn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Global Response Normalization (GRN) - Functional implementation

    Args:
        hidden_states: Input tensor [N, H, W, C] (NHWC format)
        weight: Learnable weight [1, 1, 1, C]
        bias: Learnable bias [1, 1, 1, C]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor [N, H, W, C]
    """
    # Compute L2 norm across spatial dimensions
    gx = torch.norm(hidden_states, p=2, dim=(1, 2), keepdim=True)  # [N, 1, 1, C]

    # Normalize by channel-wise mean
    nx = gx / (gx.mean(dim=-1, keepdim=True) + eps)  # [N, 1, 1, C]

    # Apply normalization with learnable parameters and residual
    return weight * (hidden_states * nx) + bias + hidden_states


def torch_convnextv2_layer(
    hidden_states: torch.Tensor,
    *,
    dwconv_weight: torch.Tensor,
    dwconv_bias: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
) -> torch.Tensor:
    """
    ConvNextV2Layer - Functional implementation

    Args:
        hidden_states: Input tensor [N, C, H, W] (NCHW format)
        dwconv_weight: Depthwise conv weight [C, 1, 7, 7]
        dwconv_bias: Depthwise conv bias [C]
        layernorm_weight: LayerNorm weight [C]
        layernorm_bias: LayerNorm bias [C]
        pwconv1_weight: Pointwise conv1 weight [4*C, C]
        pwconv1_bias: Pointwise conv1 bias [4*C]
        grn_weight: GRN weight [1, 1, 1, 4*C]
        grn_bias: GRN bias [1, 1, 1, 4*C]
        pwconv2_weight: Pointwise conv2 weight [C, 4*C]
        pwconv2_bias: Pointwise conv2 bias [C]

    Returns:
        Output tensor [N, C, H, W]
    """
    residual = hidden_states
    hidden_dim = hidden_states.shape[1]

    # Depthwise Conv2d
    x = F.conv2d(hidden_states, dwconv_weight, dwconv_bias, padding=3, groups=hidden_dim)

    # Permute to NHWC for LayerNorm
    x = x.permute(0, 2, 3, 1)  # [N, H, W, C]

    # LayerNorm (channels-last)
    x = F.layer_norm(x, (hidden_dim,), layernorm_weight, layernorm_bias, eps=1e-6)

    # Pointwise Conv1 (expand 4x) - Linear
    x = F.linear(x, pwconv1_weight, pwconv1_bias)

    # GELU
    x = F.gelu(x)

    # GRN
    x = torch_grn(x, grn_weight, grn_bias)

    # Pointwise Conv2 (contract) - Linear
    x = F.linear(x, pwconv2_weight, pwconv2_bias)

    # Permute back to NCHW
    x = x.permute(0, 3, 1, 2)  # [N, C, H, W]

    # Residual connection
    x = x + residual

    return x


def torch_convnextv2_embeddings(
    pixel_values: torch.Tensor,
    *,
    patch_embed_weight: torch.Tensor,
    patch_embed_bias: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
) -> torch.Tensor:
    """
    ConvNextV2Embeddings - Functional implementation

    Args:
        pixel_values: Input images [N, 3, 224, 224]
        patch_embed_weight: Patch embedding conv weight [128, 3, 4, 4]
        patch_embed_bias: Patch embedding conv bias [128]
        layernorm_weight: LayerNorm weight [128]
        layernorm_bias: LayerNorm bias [128]

    Returns:
        Embedded tensor [N, 128, 56, 56]
    """
    # Patch embedding conv
    x = F.conv2d(pixel_values, patch_embed_weight, patch_embed_bias, stride=4)

    # Permute for LayerNorm
    x = x.permute(0, 2, 3, 1)  # [N, 56, 56, 128]

    # LayerNorm
    x = F.layer_norm(x, (128,), layernorm_weight, layernorm_bias, eps=1e-6)

    # Permute back
    x = x.permute(0, 3, 1, 2)  # [N, 128, 56, 56]

    return x


def torch_convnextv2_downsampling(
    hidden_states: torch.Tensor,
    *,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    in_channels: int,
) -> torch.Tensor:
    """
    ConvNextV2 Downsampling - Functional implementation

    Args:
        hidden_states: Input tensor [N, C_in, H, W]
        layernorm_weight: LayerNorm weight [C_in]
        layernorm_bias: LayerNorm bias [C_in]
        conv_weight: Downsampling conv weight [C_out, C_in, 2, 2]
        conv_bias: Downsampling conv bias [C_out]
        in_channels: Number of input channels

    Returns:
        Downsampled tensor [N, C_out, H/2, W/2]
    """
    # Permute to NHWC
    x = hidden_states.permute(0, 2, 3, 1)

    # LayerNorm
    x = F.layer_norm(x, (in_channels,), layernorm_weight, layernorm_bias, eps=1e-6)

    # Permute back
    x = x.permute(0, 3, 1, 2)

    # Downsampling conv
    x = F.conv2d(x, conv_weight, conv_bias, stride=2)

    return x


def extract_layer_parameters(layer_module):
    """
    Extract parameters from a HuggingFace ConvNextV2Layer module

    Args:
        layer_module: ConvNextV2Layer PyTorch module

    Returns:
        Dictionary of tensor parameters
    """
    return {
        "dwconv_weight": layer_module.dwconv.weight,
        "dwconv_bias": layer_module.dwconv.bias,
        "layernorm_weight": layer_module.layernorm.weight,
        "layernorm_bias": layer_module.layernorm.bias,
        "pwconv1_weight": layer_module.pwconv1.weight,
        "pwconv1_bias": layer_module.pwconv1.bias,
        "grn_weight": layer_module.grn.weight,
        "grn_bias": layer_module.grn.bias,
        "pwconv2_weight": layer_module.pwconv2.weight,
        "pwconv2_bias": layer_module.pwconv2.bias,
    }


def extract_embeddings_parameters(embeddings_module):
    """
    Extract parameters from ConvNextV2Embeddings module

    Args:
        embeddings_module: ConvNextV2Embeddings PyTorch module

    Returns:
        Dictionary of tensor parameters
    """
    return {
        "patch_embed_weight": embeddings_module.patch_embeddings.weight,
        "patch_embed_bias": embeddings_module.patch_embeddings.bias,
        "layernorm_weight": embeddings_module.layernorm.weight,
        "layernorm_bias": embeddings_module.layernorm.bias,
    }
