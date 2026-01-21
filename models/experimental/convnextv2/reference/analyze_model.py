# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ConvNextV2 Reference Model Analysis

This script analyzes the ConvNextV2 model from HuggingFace to extract:
1. Model structure and hierarchy
2. All operators/modules used
3. Input/output shapes for each layer
4. Parameter counts

Usage:
    python analyze_model.py
"""

import torch
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config, AutoImageProcessor


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def analyze_model():
    # Load model
    model_name = "facebook/convnextv2-base-1k-224"
    print(f"Loading model: {model_name}")

    config = ConvNextV2Config.from_pretrained(model_name)
    model = ConvNextV2ForImageClassification.from_pretrained(model_name)
    model.eval()

    # Print configuration
    print_separator("Model Configuration")
    print(f"Image size: {config.image_size}")
    print(f"Patch size: {config.patch_size}")
    print(f"Num channels: {config.num_channels}")
    print(f"Hidden sizes: {config.hidden_sizes}")
    print(f"Depths: {config.depths}")
    print(f"Num stages: {config.num_stages}")
    print(f"Num labels: {config.num_labels}")
    print(f"Hidden act: {config.hidden_act}")
    print(f"Drop path rate: {config.drop_path_rate}")

    # Generate sample input
    print_separator("Sample Input")
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {input_tensor.shape} (NCHW format)")

    # Get golden output
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Output logits shape: {output.logits.shape}")

    # List all module types
    print_separator("Module Types Used")
    module_types = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in module_types:
            module_types[module_type] = []
        module_types[module_type].append(name)

    for mt in sorted(module_types.keys()):
        count = len(module_types[mt])
        print(f"  {mt}: {count} instances")

    # Print module hierarchy
    print_separator("Module Hierarchy (Top Level)")

    def print_module_tree(module, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        for name, child in module.named_children():
            print(f"{prefix}{name}: {type(child).__name__}")
            print_module_tree(child, prefix + "  ", max_depth, current_depth + 1)

    print_module_tree(model)

    # Detailed analysis of ConvNextV2Layer (the main building block)
    print_separator("ConvNextV2Layer Structure (Main Building Block)")
    for name, module in model.named_modules():
        if "layers.0" in name and name.endswith("layers.0"):
            print(f"\nAnalyzing: {name}")
            for child_name, child in module.named_children():
                print(f"  {child_name}: {type(child).__name__}")
                if hasattr(child, "weight"):
                    print(f"    weight shape: {child.weight.shape}")
                if hasattr(child, "bias") and child.bias is not None:
                    print(f"    bias shape: {child.bias.shape}")
            break

    # Extract all operators for TTNN mapping
    print_separator("Operators to Map to TTNN")

    operators = {
        "Conv2d": [],
        "Linear": [],
        "LayerNorm": [],
        "GELU": [],
        "GRN (Global Response Normalization)": [],
        "Add (Residual)": [],
        "AdaptiveAvgPool": [],
    }

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            info = f"{name}: in={module.in_channels}, out={module.out_channels}, kernel={module.kernel_size}, stride={module.stride}, groups={module.groups}"
            operators["Conv2d"].append(info)
        elif isinstance(module, torch.nn.Linear):
            info = f"{name}: in={module.in_features}, out={module.out_features}"
            operators["Linear"].append(info)
        elif "LayerNorm" in type(module).__name__:
            if hasattr(module, "normalized_shape"):
                info = f"{name}: shape={module.normalized_shape}"
            else:
                info = f"{name}"
            operators["LayerNorm"].append(info)
        elif "GRN" in type(module).__name__:
            operators["GRN (Global Response Normalization)"].append(name)

    print("Conv2d operations:")
    for op in operators["Conv2d"][:5]:  # Show first 5
        print(f"  {op}")
    print(f"  ... and {len(operators['Conv2d']) - 5} more" if len(operators["Conv2d"]) > 5 else "")

    print("\nLinear operations:")
    for op in operators["Linear"][:5]:
        print(f"  {op}")
    print(f"  ... and {len(operators['Linear']) - 5} more" if len(operators["Linear"]) > 5 else "")

    print("\nLayerNorm operations:")
    for op in operators["LayerNorm"][:3]:
        print(f"  {op}")
    print(f"  ... and {len(operators['LayerNorm']) - 3} more" if len(operators["LayerNorm"]) > 3 else "")

    print(f"\nGRN operations: {len(operators['GRN (Global Response Normalization)'])} instances")

    # Parameter count
    print_separator("Parameter Summary")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,}")

    # Shape analysis through forward hooks
    print_separator("Shape Analysis (Forward Pass)")

    shapes = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                in_shape = input[0].shape if hasattr(input[0], "shape") else "N/A"
            else:
                in_shape = input.shape if hasattr(input, "shape") else "N/A"

            if isinstance(output, tuple):
                out_shape = output[0].shape if hasattr(output[0], "shape") else "N/A"
            else:
                out_shape = output.shape if hasattr(output, "shape") else "N/A"

            shapes[name] = {"input": in_shape, "output": out_shape}

        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) or "LayerNorm" in type(module).__name__:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    # Print key shapes
    key_layers = [
        "convnextv2.embeddings.patch_embeddings",
        "convnextv2.embeddings.layernorm",
        "convnextv2.encoder.stages.0.layers.0.dwconv",
        "convnextv2.encoder.stages.0.layers.0.layernorm",
        "convnextv2.encoder.stages.0.layers.0.pwconv1",
        "convnextv2.encoder.stages.0.layers.0.pwconv2",
        "convnextv2.layernorm",
        "classifier",
    ]

    for layer in key_layers:
        if layer in shapes:
            print(f"{layer}:")
            print(f"  Input:  {shapes[layer]['input']}")
            print(f"  Output: {shapes[layer]['output']}")

    # Save golden data
    print_separator("Saving Golden Data")
    golden_data = {
        "input": input_tensor,
        "output_logits": output.logits,
        "config": {
            "image_size": config.image_size,
            "patch_size": config.patch_size,
            "num_channels": config.num_channels,
            "hidden_sizes": config.hidden_sizes,
            "depths": config.depths,
        },
    }

    save_path = "golden_data.pt"
    torch.save(golden_data, save_path)
    print(f"Golden data saved to: {save_path}")

    print_separator("Analysis Complete")
    print(
        """
Summary for TTNN Porting:

Key modules to implement:
1. ConvNextV2Embeddings
   - Conv2d (patch_size kernel, patch_size stride)
   - LayerNorm

2. ConvNextV2Layer (repeated block)
   - Depthwise Conv2d (7x7, groups=channels)
   - LayerNorm (channels_last)
   - Linear (expand to 4x hidden dim)
   - GELU activation
   - GRN (Global Response Normalization) - CUSTOM OP
   - Linear (contract back)
   - Residual Add

3. ConvNextV2Stage
   - Optional downsampling (LayerNorm + Conv2d stride=2)
   - Multiple ConvNextV2Layer blocks

4. Classification Head
   - Global Average Pooling
   - LayerNorm
   - Linear classifier

Special attention needed for:
- GRN (Global Response Normalization): Custom operation
- Data format: Model uses channels-last internally for LayerNorm
- Depthwise convolutions: groups=in_channels
"""
    )


if __name__ == "__main__":
    analyze_model()
