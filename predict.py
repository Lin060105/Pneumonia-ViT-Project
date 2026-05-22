"""Command-line prediction for a single chest X-ray image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from model_utils import decide_screening_status, get_device, image_to_tensor, load_model_checkpoint, predict_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pneumonia probability for one image")
    parser.add_argument("image_path", help="Path to a JPG/PNG chest X-ray image")
    parser.add_argument("--model-path", default="saved_models/pneumonia_binary_best.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--uncertainty-margin", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    model_path = Path(args.model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = get_device()
    model, metadata = load_model_checkpoint(model_path, device=device)
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_to_tensor(image, device)

    with torch.no_grad():
        probabilities = predict_probabilities(model, image_tensor)
    screening = decide_screening_status(
        probabilities,
        threshold=args.threshold,
        uncertainty_margin=args.uncertainty_margin,
    )

    print(f"Model: {metadata.get('model_name', 'unknown')}")
    print(f"Image: {image_path}")
    print(f"Decision: {screening['decision']}")
    print(f"P(Normal): {screening['normal_probability']:.4f}")
    print(f"P(Pneumonia): {screening['pneumonia_probability']:.4f}")
    print(f"Threshold: {screening['threshold']:.2f}")


if __name__ == "__main__":
    main()
