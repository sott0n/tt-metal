# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ConvNextV2 Full Model TTNN Implementation

Architecture:
    ConvNextV2ForImageClassification
    ├── convnextv2: ConvNextV2Model
    │   ├── embeddings: ConvNextV2Embeddings
    │   │   ├── patch_embeddings: Conv2d(3, 128, k=4, s=4)
    │   │   └── layernorm: LayerNorm(128)
    │   ├── encoder: ConvNextV2Encoder
    │   │   └── stages: [Stage0, Stage1, Stage2, Stage3]
    │   │       └── Each stage: downsampling (optional) + layers
    │   └── layernorm: LayerNorm(1024)
    └── classifier: Linear(1024, 1000)
"""

import ttnn
import torch
from typing import Optional, List, Tuple

from models.experimental.convnextv2.tt.ttnn_convnextv2_layer import (
    ttnn_convnextv2_layer,
    ttnn_convnextv2_layer_nhwc,
)


def ttnn_convnextv2_embeddings(
    pixel_values,
    *,
    parameters,
    device,
    batch_size: int,
):
    """
    ConvNextV2Embeddings - Patch embedding and initial normalization

    Args:
        pixel_values: Input image tensor [N, 3, 224, 224] (NCHW)
        parameters: Module parameters containing:
            - patch_embeddings.weight, patch_embeddings.bias
            - layernorm.weight, layernorm.bias
        device: TTNN device
        batch_size: Batch size

    Returns:
        Embedded tensor [N, 128, 56, 56] (NCHW)
    """
    # Convert to NHWC for conv2d
    x_nhwc = ttnn.permute(pixel_values, (0, 2, 3, 1))

    # Patch embedding: Conv2d(3, 128, kernel=4, stride=4)
    x_conv, (out_h, out_w), _ = ttnn.conv2d(
        input_tensor=x_nhwc,
        weight_tensor=parameters.patch_embeddings.weight,
        in_channels=3,
        out_channels=128,
        device=device,
        bias_tensor=parameters.patch_embeddings.bias,
        kernel_size=(4, 4),
        stride=(4, 4),
        padding=(0, 0),
        batch_size=batch_size,
        input_height=224,
        input_width=224,
        groups=1,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    # Reshape: [N*H*W, C] -> [N, H, W, C]
    hidden_dim = 128
    x = ttnn.reshape(x_conv, (batch_size, out_h, out_w, hidden_dim))

    # LayerNorm (channels-last format)
    x_flat = ttnn.reshape(x, (batch_size * out_h * out_w, hidden_dim))
    x_flat = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT)

    x_flat = ttnn.layer_norm(
        x_flat,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
        epsilon=1e-6,
    )

    # Reshape back to [N, H, W, C]
    x = ttnn.reshape(x_flat, (batch_size, out_h, out_w, hidden_dim))

    # Permute to NCHW for compatibility with standard conv flow
    x = ttnn.permute(x, (0, 3, 1, 2))

    return x  # [N, 128, 56, 56]


def ttnn_convnextv2_downsampling(
    hidden_states,
    *,
    parameters,
    device,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    height: int,
    width: int,
):
    """
    ConvNextV2 Downsampling layer

    Reduces spatial dimensions by 2x and increases channels.

    Args:
        hidden_states: Input tensor [N, C_in, H, W] (NCHW)
        parameters: Contains layernorm and conv weights
        device: TTNN device
        in_channels: Input channel count
        out_channels: Output channel count
        batch_size: Batch size
        height: Input spatial height
        width: Input spatial width

    Returns:
        Downsampled tensor [N, C_out, H/2, W/2]
    """
    # Permute to NHWC
    x = ttnn.permute(hidden_states, (0, 2, 3, 1))

    # LayerNorm (channels-last)
    x_flat = ttnn.reshape(x, (batch_size * height * width, in_channels))
    x_flat = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT)

    x_flat = ttnn.layer_norm(
        x_flat,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
        epsilon=1e-6,
    )

    # Reshape back to NHWC
    x = ttnn.reshape(x_flat, (batch_size, height, width, in_channels))

    # Downsampling Conv: kernel=2, stride=2
    x_conv, (out_h, out_w), _ = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.conv.weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=parameters.conv.bias,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        groups=1,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    # Reshape and permute to NCHW
    x = ttnn.reshape(x_conv, (batch_size, out_h, out_w, out_channels))
    x = ttnn.permute(x, (0, 3, 1, 2))

    return x  # [N, C_out, H/2, W/2]


def ttnn_convnextv2_stage(
    hidden_states,
    *,
    parameters,
    device,
    stage_idx: int,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    height: int,
    width: int,
    downsample: bool = True,
):
    """
    ConvNextV2Stage - One stage of the encoder

    Args:
        hidden_states: Input tensor [N, C_in, H, W]
        parameters: Stage parameters
        device: TTNN device
        stage_idx: Stage index (0-3)
        hidden_dim: Channel dimension for this stage
        num_layers: Number of ConvNextV2Layer blocks
        batch_size: Batch size
        height: Spatial height
        width: Spatial width
        downsample: Whether to apply downsampling at the start

    Returns:
        Output tensor [N, C_out, H_out, W_out]
    """
    x = hidden_states

    # Downsampling (for stages 1, 2, 3)
    if downsample and stage_idx > 0:
        # Input channels are half of current hidden_dim
        in_channels = hidden_dim // 2
        x = ttnn_convnextv2_downsampling(
            x,
            parameters=parameters.downsampling_layer[0],
            device=device,
            in_channels=in_channels,
            out_channels=hidden_dim,
            batch_size=batch_size,
            height=height * 2,  # Before downsampling
            width=width * 2,
        )

    # Process layers
    for layer_idx in range(num_layers):
        x = ttnn_convnextv2_layer(
            x,
            parameters=parameters.layers[layer_idx],
            device=device,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            height=height,
            width=width,
        )

    return x


def ttnn_convnextv2_encoder(
    hidden_states,
    *,
    parameters,
    device,
    config,
    batch_size: int,
):
    """
    ConvNextV2Encoder - Full encoder with all stages

    Args:
        hidden_states: Input from embeddings [N, 128, 56, 56]
        parameters: Encoder parameters
        device: TTNN device
        config: Model configuration dict with hidden_sizes and depths
        batch_size: Batch size

    Returns:
        Encoded tensor [N, 1024, 7, 7]
    """
    hidden_sizes = config["hidden_sizes"]  # [128, 256, 512, 1024]
    depths = config["depths"]  # [3, 3, 27, 3]
    spatial_sizes = [56, 28, 14, 7]

    x = hidden_states

    for stage_idx in range(4):
        x = ttnn_convnextv2_stage(
            x,
            parameters=parameters.stages[stage_idx],
            device=device,
            stage_idx=stage_idx,
            hidden_dim=hidden_sizes[stage_idx],
            num_layers=depths[stage_idx],
            batch_size=batch_size,
            height=spatial_sizes[stage_idx],
            width=spatial_sizes[stage_idx],
            downsample=(stage_idx > 0),
        )

    return x


def ttnn_convnextv2_model(
    pixel_values,
    *,
    parameters,
    device,
    config,
    batch_size: int,
):
    """
    ConvNextV2Model - Main model (without classifier)

    Args:
        pixel_values: Input images [N, 3, 224, 224]
        parameters: Model parameters
        device: TTNN device
        config: Model configuration
        batch_size: Batch size

    Returns:
        Pooled output tensor [N, 1024]
    """
    # Embeddings
    x = ttnn_convnextv2_embeddings(
        pixel_values,
        parameters=parameters.embeddings,
        device=device,
        batch_size=batch_size,
    )

    # Encoder
    x = ttnn_convnextv2_encoder(
        x,
        parameters=parameters.encoder,
        device=device,
        config=config,
        batch_size=batch_size,
    )

    # x is [N, 1024, 7, 7] (NCHW)
    # Global Average Pooling
    x_nhwc = ttnn.permute(x, (0, 2, 3, 1))  # [N, 7, 7, 1024]
    x_pooled = ttnn.global_avg_pool2d(x_nhwc)  # [N, 1, 1, 1024]

    # Flatten to [N, 1024]
    x_flat = ttnn.reshape(x_pooled, (batch_size, 1024))

    # Final LayerNorm
    x_flat = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT)
    x_flat = ttnn.layer_norm(
        x_flat,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
    )

    return x_flat


def ttnn_convnextv2_for_image_classification(
    pixel_values,
    *,
    parameters,
    device,
    config,
    batch_size: int,
):
    """
    ConvNextV2ForImageClassification - Full model with classifier

    Args:
        pixel_values: Input images [N, 3, 224, 224]
        parameters: Full model parameters including classifier
        device: TTNN device
        config: Model configuration
        batch_size: Batch size

    Returns:
        Logits tensor [N, 1000]
    """
    # ConvNextV2Model (backbone)
    pooled_output = ttnn_convnextv2_model(
        pixel_values,
        parameters=parameters.convnextv2,
        device=device,
        config=config,
        batch_size=batch_size,
    )

    # Classifier: Linear(1024, 1000)
    logits = ttnn.linear(
        pooled_output,
        parameters.classifier.weight,
        bias=parameters.classifier.bias,
    )

    return logits


# Parameter preprocessing functions

def preprocess_conv_weight(weight, dtype=ttnn.bfloat16):
    """Preprocess Conv2d weight for TTNN"""
    # TTNN conv2d expects weights in specific format
    return weight


def preprocess_linear_weight(weight, dtype=ttnn.bfloat16):
    """Preprocess Linear weight for TTNN (transpose)"""
    return weight.T.contiguous()


def create_convnextv2_parameters(torch_model, device):
    """
    Convert PyTorch ConvNextV2 model parameters to TTNN format

    Args:
        torch_model: HuggingFace ConvNextV2ForImageClassification model
        device: TTNN device

    Returns:
        Nested parameter structure for TTNN model
    """
    from types import SimpleNamespace

    def to_ttnn_tensor(tensor, transpose=False, add_dims=0):
        """Convert torch tensor to TTNN tensor"""
        if transpose:
            tensor = tensor.T.contiguous()
        for _ in range(add_dims):
            tensor = tensor.unsqueeze(0)
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def convert_layernorm(ln_module):
        """Convert LayerNorm parameters"""
        return SimpleNamespace(
            weight=to_ttnn_tensor(ln_module.weight.unsqueeze(0)),
            bias=to_ttnn_tensor(ln_module.bias.unsqueeze(0)),
        )

    def convert_linear(linear_module):
        """Convert Linear parameters"""
        return SimpleNamespace(
            weight=to_ttnn_tensor(linear_module.weight, transpose=True),
            bias=to_ttnn_tensor(linear_module.bias.unsqueeze(0)),
        )

    def convert_conv2d(conv_module):
        """Convert Conv2d parameters"""
        bias = None
        if conv_module.bias is not None:
            bias = ttnn.from_torch(
                conv_module.bias.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
            )
        return SimpleNamespace(
            weight=ttnn.from_torch(conv_module.weight, dtype=ttnn.bfloat16),
            bias=bias,
        )

    def convert_grn(grn_module):
        """Convert GRN parameters"""
        return SimpleNamespace(
            weight=to_ttnn_tensor(grn_module.weight),
            bias=to_ttnn_tensor(grn_module.bias),
        )

    def convert_layer(layer_module):
        """Convert ConvNextV2Layer parameters"""
        return SimpleNamespace(
            dwconv=convert_conv2d(layer_module.dwconv),
            layernorm=convert_layernorm(layer_module.layernorm),
            pwconv1=convert_linear(layer_module.pwconv1),
            grn=convert_grn(layer_module.grn),
            pwconv2=convert_linear(layer_module.pwconv2),
        )

    def convert_downsampling(downsample_modules):
        """Convert downsampling layer parameters"""
        # downsample_modules is a ModuleList with [LayerNorm, Conv2d]
        return [
            SimpleNamespace(
                layernorm=convert_layernorm(downsample_modules[0]),
                conv=convert_conv2d(downsample_modules[1]),
            )
        ]

    def convert_stage(stage_module, stage_idx):
        """Convert ConvNextV2Stage parameters"""
        layers = [convert_layer(layer) for layer in stage_module.layers]

        downsampling = None
        if stage_idx > 0 and hasattr(stage_module, "downsampling_layer"):
            downsampling = convert_downsampling(stage_module.downsampling_layer)

        return SimpleNamespace(
            layers=layers,
            downsampling_layer=downsampling,
        )

    # Build parameter tree
    model = torch_model

    # Embeddings
    embeddings = SimpleNamespace(
        patch_embeddings=convert_conv2d(model.convnextv2.embeddings.patch_embeddings),
        layernorm=convert_layernorm(model.convnextv2.embeddings.layernorm),
    )

    # Encoder stages
    stages = [
        convert_stage(model.convnextv2.encoder.stages[i], i)
        for i in range(4)
    ]
    encoder = SimpleNamespace(stages=stages)

    # Final layernorm
    final_layernorm = convert_layernorm(model.convnextv2.layernorm)

    # ConvNextV2 model
    convnextv2 = SimpleNamespace(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=final_layernorm,
    )

    # Classifier
    classifier = convert_linear(model.classifier)

    # Full model parameters
    parameters = SimpleNamespace(
        convnextv2=convnextv2,
        classifier=classifier,
    )

    return parameters
