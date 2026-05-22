"""Generate a pneumonia-focused Grad-CAM heatmap for the binary ViT model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model_utils import DEFAULT_IMAGE_SIZE, PNEUMONIA_INDEX, get_device, image_to_tensor, load_model_checkpoint


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Grad-CAM explanation for one image")
    parser.add_argument("image_path", nargs="?", default="chest_xray/test/PNEUMONIA/person10_virus_35.jpeg")
    parser.add_argument("--model-path", default="saved_models/pneumonia_binary_best.pth")
    parser.add_argument("--output", default="heatmap_result.jpg")
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
    model, _metadata = load_model_checkpoint(model_path, device=device)
    image = Image.open(image_path).convert("RGB")
    input_tensor = image_to_tensor(image, device)

    rgb_img = np.array(image.resize(DEFAULT_IMAGE_SIZE))
    rgb_img_float = np.float32(rgb_img) / 255.0
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(PNEUMONIA_INDEX)],
    )[0, :]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    cv2.imwrite(args.output, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved pneumonia-focused Grad-CAM to {args.output}")


if __name__ == "__main__":
    main()
