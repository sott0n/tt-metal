#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ConvNextV2 TTNN Demo

This script demonstrates how to run ConvNextV2 image classification on Tenstorrent hardware.

Usage:
    # Basic usage (random input)
    python models/experimental/convnextv2/demo/demo_convnextv2.py

    # With a real image
    python models/experimental/convnextv2/demo/demo_convnextv2.py --image path/to/image.jpg

    # Compare with PyTorch
    python models/experimental/convnextv2/demo/demo_convnextv2.py --compare

Prerequisites:
    pip install transformers pillow requests
"""

import argparse
import torch
import ttnn
from pathlib import Path


def download_model():
    """Download ConvNextV2 model from HuggingFace"""
    from transformers import ConvNextV2ForImageClassification, AutoImageProcessor

    print("Downloading ConvNextV2-base model from HuggingFace...")
    model_name = "facebook/convnextv2-base-1k-224"

    model = ConvNextV2ForImageClassification.from_pretrained(model_name)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(model_name)

    print(f"Model loaded: {model_name}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Num classes: {model.config.num_labels}")

    return model, processor


def load_image(image_path: str, processor):
    """Load and preprocess an image"""
    from PIL import Image
    import requests
    from io import BytesIO

    if image_path.startswith("http"):
        # Download from URL
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # Load from local file
        image = Image.open(image_path).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"], image


def get_imagenet_labels():
    """Get ImageNet class labels"""
    import requests

    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(url, timeout=5)
        labels = response.text.strip().split("\n")
        return labels
    except:
        return None


def run_pytorch_inference(model, pixel_values):
    """Run inference with PyTorch model"""
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs.logits


def run_ttnn_inference(torch_model, pixel_values, device):
    """Run inference with TTNN model"""
    from models.experimental.convnextv2.tt.ttnn_convnextv2_model import (
        ttnn_convnextv2_for_image_classification,
        create_convnextv2_parameters,
    )

    # Model config
    config = {
        "hidden_sizes": [128, 256, 512, 1024],
        "depths": [3, 3, 27, 3],
    }

    # Convert parameters to TTNN format
    print("Converting model parameters to TTNN format...")
    parameters = create_convnextv2_parameters(torch_model, device)

    # Convert input to TTNN
    ttnn_input = ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Run inference
    print("Running TTNN inference...")
    ttnn_logits = ttnn_convnextv2_for_image_classification(
        ttnn_input,
        parameters=parameters,
        device=device,
        config=config,
        batch_size=pixel_values.shape[0],
    )

    # Convert output back to PyTorch
    output = ttnn.to_torch(ttnn_logits)
    return output


def print_predictions(logits, labels=None, top_k=5):
    """Print top-k predictions"""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

    print(f"\nTop-{top_k} Predictions:")
    print("-" * 50)

    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
        idx = idx.item()
        prob = prob.item()
        if labels and idx < len(labels):
            label = labels[idx]
        else:
            label = f"Class {idx}"
        print(f"  {i+1}. {label}: {prob*100:.2f}%")


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient"""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()

    t1_centered = t1 - t1.mean()
    t2_centered = t2 - t2.mean()

    numerator = (t1_centered * t2_centered).sum()
    denominator = torch.sqrt((t1_centered ** 2).sum() * (t2_centered ** 2).sum())

    return (numerator / denominator).item()


def main():
    parser = argparse.ArgumentParser(description="ConvNextV2 TTNN Demo")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or URL to input image. If not provided, uses random input.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare TTNN output with PyTorch reference",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID to use",
    )
    args = parser.parse_args()

    # Sample image URLs for testing
    sample_images = {
        "cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "dog": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
    }

    print("=" * 60)
    print("ConvNextV2 TTNN Demo")
    print("=" * 60)

    # Step 1: Download model
    torch_model, processor = download_model()

    # Step 2: Prepare input
    if args.image:
        print(f"\nLoading image: {args.image}")
        pixel_values, _ = load_image(args.image, processor)
    else:
        print("\nUsing random input (no image provided)")
        print("  Tip: Use --image <path> to test with a real image")
        torch.manual_seed(42)
        pixel_values = torch.randn(1, 3, 224, 224)

    print(f"Input shape: {pixel_values.shape}")

    # Step 3: Open TTNN device
    print(f"\nOpening TTNN device {args.device_id}...")
    # l1_small_size is required for depthwise convolutions
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24 * 1024)

    try:
        # Step 4: Run TTNN inference
        ttnn_logits = run_ttnn_inference(torch_model, pixel_values, device)

        # Step 5: Get predictions
        labels = get_imagenet_labels()

        print("\n" + "=" * 60)
        print("TTNN Results")
        print("=" * 60)
        print_predictions(ttnn_logits, labels)

        # Step 6: Compare with PyTorch (optional)
        if args.compare:
            print("\n" + "=" * 60)
            print("PyTorch Reference Results")
            print("=" * 60)
            pytorch_logits = run_pytorch_inference(torch_model, pixel_values)
            print_predictions(pytorch_logits, labels)

            # Compute PCC
            pcc = compute_pcc(pytorch_logits, ttnn_logits)
            print(f"\nPCC (TTNN vs PyTorch): {pcc:.6f}")

            # Compare top-1 predictions
            ttnn_pred = ttnn_logits.argmax(dim=-1).item()
            pytorch_pred = pytorch_logits.argmax(dim=-1).item()
            print(f"Top-1 match: {ttnn_pred == pytorch_pred} (TTNN: {ttnn_pred}, PyTorch: {pytorch_pred})")

    finally:
        # Step 7: Close device
        print("\nClosing device...")
        ttnn.close_device(device)

    print("\nDone!")


if __name__ == "__main__":
    main()
