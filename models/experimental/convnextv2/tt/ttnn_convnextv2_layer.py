# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ConvNextV2Layer TTNN Implementation

Main building block of ConvNextV2, containing:
- Depthwise Conv2d (7x7)
- LayerNorm (channels-last)
- Pointwise Conv (expand 4x)
- GELU activation
- GRN (Global Response Normalization)
- Pointwise Conv (contract)
- Residual connection
"""

import ttnn
import torch


def ttnn_grn(
    hidden_states,
    weight,
    bias,
    *,
    device,
    eps: float = 1e-6,
):
    """
    Global Response Normalization (GRN) for ConvNextV2

    GRN normalizes features based on their global spatial L2 norm.

    Args:
        hidden_states: Input tensor of shape [N, H, W, C] in NHWC format
        weight: Learnable weight of shape [1, 1, 1, C]
        bias: Learnable bias of shape [1, 1, 1, C]
        device: TTNN device
        eps: Small epsilon for numerical stability

    Returns:
        Normalized tensor of same shape [N, H, W, C]

    PyTorch Reference:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # L2 norm over spatial
        nx = gx / (gx.mean(dim=-1, keepdim=True) + eps)    # Normalize by channel mean
        return weight * (x * nx) + bias + x                 # Apply params + residual
    """
    # Get shape info from tensor
    x_shape = hidden_states.shape
    N, H, W, C = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

    # Reshape to [N, H*W, C] for easier reduction
    x_flat = ttnn.reshape(hidden_states, (N, H * W, C))

    # Step 1: Compute x^2
    x_squared = ttnn.pow(x_flat, 2)

    # Step 2: Sum over spatial dimension -> [N, 1, C]
    # Use to_torch for reduction operations that may not be fully supported
    x_squared_torch = ttnn.to_torch(x_squared)
    gx_squared_torch = x_squared_torch.sum(dim=1, keepdim=True)  # [N, 1, C]

    # Step 3: Square root -> L2 norm
    gx_torch = torch.sqrt(gx_squared_torch)  # [N, 1, C]

    # Step 4: Compute mean over channels
    gx_mean_torch = gx_torch.mean(dim=-1, keepdim=True)  # [N, 1, 1]

    # Step 5: Normalize
    nx_torch = gx_torch / (gx_mean_torch + eps)  # [N, 1, C]

    # Step 6: Broadcast nx to spatial dimensions
    nx_expanded_torch = nx_torch.expand(N, H * W, C)

    # Convert back to TTNN
    nx_expanded = ttnn.from_torch(
        nx_expanded_torch.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Step 7: Apply normalization: x * nx
    x_normalized = ttnn.mul(x_flat, nx_expanded)

    # Reshape back to [N, H, W, C]
    x_normalized_4d = ttnn.reshape(x_normalized, (N, H, W, C))

    # Step 8: Apply learnable parameters: weight * x_normalized + bias
    weighted = ttnn.mul(weight, x_normalized_4d)
    with_bias = ttnn.add(weighted, bias)

    # Step 9: Add residual connection
    output = ttnn.add(with_bias, hidden_states)

    return output


def ttnn_convnextv2_layer(
    hidden_states,
    *,
    parameters,
    device,
    hidden_dim: int,
    batch_size: int,
    height: int,
    width: int,
):
    """
    ConvNextV2Layer - Main building block

    Data flow:
        Input (NCHW) -> dwconv -> permute(NHWC) -> layernorm -> pwconv1 ->
        gelu -> grn -> pwconv2 -> permute(NCHW) -> residual add -> Output

    Args:
        hidden_states: Input tensor of shape [N, C, H, W] (NCHW format)
        parameters: Module parameters containing:
            - dwconv.weight, dwconv.bias
            - layernorm.weight, layernorm.bias
            - pwconv1.weight, pwconv1.bias
            - grn.weight, grn.bias
            - pwconv2.weight, pwconv2.bias
        device: TTNN device
        hidden_dim: Number of channels
        batch_size: Batch size
        height: Spatial height
        width: Spatial width

    Returns:
        Output tensor of shape [N, C, H, W]
    """
    residual = hidden_states

    # Convert to NHWC for conv2d
    # TTNN conv2d expects NHWC input
    x_nhwc = ttnn.permute(hidden_states, (0, 2, 3, 1))

    # Depthwise Conv2d (7x7, groups=hidden_dim)
    x_conv, _, _ = ttnn.conv2d(
        input_tensor=x_nhwc,
        weight_tensor=parameters.dwconv.weight,
        in_channels=hidden_dim,
        out_channels=hidden_dim,
        device=device,
        bias_tensor=parameters.dwconv.bias,
        kernel_size=(7, 7),
        stride=(1, 1),
        padding=(3, 3),
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        groups=hidden_dim,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    # Reshape conv output: [N*H*W, C] -> [N, H, W, C]
    x = ttnn.reshape(x_conv, (batch_size, height, width, hidden_dim))

    # LayerNorm (channels-last: normalize over C dimension)
    # Flatten to [N*H*W, C] for layer_norm
    x_flat = ttnn.reshape(x, (batch_size * height * width, hidden_dim))
    x_flat = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT)

    x_flat = ttnn.layer_norm(
        x_flat,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
        epsilon=1e-6,
    )

    # Pointwise Conv1 (expand 4x): Linear(hidden_dim, hidden_dim * 4)
    x_flat = ttnn.linear(
        x_flat,
        parameters.pwconv1.weight,
        bias=parameters.pwconv1.bias,
    )

    # GELU activation
    x_flat = ttnn.gelu(x_flat)

    # Reshape to [N, H, W, C*4] for GRN
    x = ttnn.reshape(x_flat, (batch_size, height, width, hidden_dim * 4))

    # GRN (Global Response Normalization)
    x = ttnn_grn(
        x,
        parameters.grn.weight,
        parameters.grn.bias,
        device=device,
    )

    # Flatten back for pwconv2
    x_flat = ttnn.reshape(x, (batch_size * height * width, hidden_dim * 4))

    # Pointwise Conv2 (contract): Linear(hidden_dim * 4, hidden_dim)
    x_flat = ttnn.linear(
        x_flat,
        parameters.pwconv2.weight,
        bias=parameters.pwconv2.bias,
    )

    # Reshape to [N, H, W, C]
    x = ttnn.reshape(x_flat, (batch_size, height, width, hidden_dim))

    # Permute back to NCHW: [N, H, W, C] -> [N, C, H, W]
    x = ttnn.permute(x, (0, 3, 1, 2))

    # Residual connection
    output = ttnn.add(x, residual)

    return output


def ttnn_convnextv2_layer_nhwc(
    hidden_states,
    *,
    parameters,
    device,
    hidden_dim: int,
    batch_size: int,
    height: int,
    width: int,
):
    """
    ConvNextV2Layer optimized version - maintains NHWC format throughout

    This version avoids permute operations by keeping data in NHWC format.
    Input and output are both in NHWC format.

    Args:
        hidden_states: Input tensor of shape [N, H, W, C] (NHWC format)
        parameters: Module parameters
        device: TTNN device
        hidden_dim: Number of channels
        batch_size: Batch size
        height: Spatial height
        width: Spatial width

    Returns:
        Output tensor of shape [N, H, W, C] (NHWC format)
    """
    residual = hidden_states

    # Depthwise Conv2d (7x7, groups=hidden_dim)
    # Input is already NHWC
    x_conv, _, _ = ttnn.conv2d(
        input_tensor=hidden_states,
        weight_tensor=parameters.dwconv.weight,
        in_channels=hidden_dim,
        out_channels=hidden_dim,
        device=device,
        bias_tensor=parameters.dwconv.bias,
        kernel_size=(7, 7),
        stride=(1, 1),
        padding=(3, 3),
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        groups=hidden_dim,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    # Reshape conv output: [N*H*W, C] -> [N, H, W, C]
    x = ttnn.reshape(x_conv, (batch_size, height, width, hidden_dim))

    # LayerNorm (channels-last)
    x_flat = ttnn.reshape(x, (batch_size * height * width, hidden_dim))
    x_flat = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT)

    x_flat = ttnn.layer_norm(
        x_flat,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
        epsilon=1e-6,
    )

    # Pointwise Conv1 (expand 4x)
    x_flat = ttnn.linear(
        x_flat,
        parameters.pwconv1.weight,
        bias=parameters.pwconv1.bias,
    )

    # GELU
    x_flat = ttnn.gelu(x_flat)

    # Reshape for GRN
    x = ttnn.reshape(x_flat, (batch_size, height, width, hidden_dim * 4))

    # GRN
    x = ttnn_grn(
        x,
        parameters.grn.weight,
        parameters.grn.bias,
        device=device,
    )

    # Flatten and apply pwconv2
    x_flat = ttnn.reshape(x, (batch_size * height * width, hidden_dim * 4))
    x_flat = ttnn.linear(
        x_flat,
        parameters.pwconv2.weight,
        bias=parameters.pwconv2.bias,
    )

    # Reshape to [N, H, W, C]
    x = ttnn.reshape(x_flat, (batch_size, height, width, hidden_dim))

    # Residual connection (both in NHWC)
    output = ttnn.add(x, residual)

    return output
