"""Generate pneumonia-focused Grad-CAM heatmaps for the binary ViT model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import (
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    PNEUMONIA_INDEX,
    get_device,
    get_eval_transform,
    image_to_tensor,
    load_model_checkpoint,
)


DEFAULT_IMAGE_PATH = "chest_xray/test/PNEUMONIA/person10_virus_35.jpeg"


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Grad-CAM explanations for a binary ViT model")
    parser.add_argument("image_path", nargs="?", default=None)
    parser.add_argument("--model-path", default="saved_models/pneumonia_binary_best.pth")
    parser.add_argument("--output", default="heatmap_result.jpg")
    parser.add_argument("--auto-cases", action="store_true", help="Create TP/TN/FP/FN representative case grid")
    parser.add_argument("--dataset-dir", default="chest_xray/test", help="ImageFolder dataset for --auto-cases")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--samples-per-group", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--paper-figure", default=None)
    return parser.parse_args()


def target_layers_for_vit(model: torch.nn.Module):
    if not hasattr(model, "blocks"):
        raise ValueError("This Grad-CAM script expects a ViT model with transformer blocks.")
    return [model.blocks[-1].norm1]


def build_gradcam_overlay(cam: GradCAM, image: Image.Image, input_tensor: torch.Tensor) -> np.ndarray:
    rgb_img = np.array(image.resize(DEFAULT_IMAGE_SIZE))
    rgb_img_float = np.float32(rgb_img) / 255.0
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(PNEUMONIA_INDEX)],
    )[0, :]
    return show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)


def save_single_gradcam(
    model: torch.nn.Module,
    image_path: Path,
    output_path: str | Path,
    device: torch.device,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    input_tensor = image_to_tensor(image, device)
    cam = GradCAM(model=model, target_layers=target_layers_for_vit(model), reshape_transform=reshape_transform)
    visualization = build_gradcam_overlay(cam, image, input_tensor)

    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved pneumonia-focused Grad-CAM to {output_path}")


def collect_case_predictions(
    model: torch.nn.Module,
    dataset_dir: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    dataset = ImageFolderWithPaths(dataset_dir, transform=get_eval_transform())
    class_names = tuple(sorted(dataset.class_to_idx, key=dataset.class_to_idx.get))
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {class_names} in {dataset_dir}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows = []
    model.eval()
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)[:, PNEUMONIA_INDEX].cpu().numpy()
            for label, probability, path in zip(labels.numpy(), probs, paths):
                rows.append(
                    {
                        "path": str(path),
                        "true_binary": int(label),
                        "true_label": CLASS_NAMES[int(label)],
                        "p_pneumonia": float(probability),
                    }
                )
    return pd.DataFrame(rows)


def label_case_type(true_binary: int, predicted_binary: int) -> str:
    if true_binary == 1 and predicted_binary == 1:
        return "TP"
    if true_binary == 0 and predicted_binary == 0:
        return "TN"
    if true_binary == 0 and predicted_binary == 1:
        return "FP"
    return "FN"


def select_representative_cases(predictions: pd.DataFrame, threshold: float, samples_per_group: int) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["predicted_binary"] = (predictions["p_pneumonia"] >= threshold).astype(int)
    predictions["predicted_label"] = predictions["predicted_binary"].map(lambda value: CLASS_NAMES[int(value)])
    predictions["case_type"] = [
        label_case_type(true_binary, predicted_binary)
        for true_binary, predicted_binary in zip(predictions["true_binary"], predictions["predicted_binary"])
    ]

    selected_frames = []
    sort_specs = {
        "TP": ("p_pneumonia", False),
        "TN": ("p_pneumonia", True),
        "FP": ("p_pneumonia", False),
        "FN": ("p_pneumonia", True),
    }
    for case_type, (column, ascending) in sort_specs.items():
        selected = (
            predictions[predictions["case_type"] == case_type]
            .sort_values(column, ascending=ascending)
            .head(samples_per_group)
        )
        selected_frames.append(selected)

    if not selected_frames:
        return predictions.iloc[0:0]
    return pd.concat(selected_frames, ignore_index=True)


def save_case_grid(
    model: torch.nn.Module,
    selected_cases: pd.DataFrame,
    output_path: Path,
    samples_per_group: int,
    device: torch.device,
) -> None:
    case_order = ["TP", "TN", "FP", "FN"]
    columns = max(1, samples_per_group)
    fig, axes = plt.subplots(len(case_order), columns, figsize=(3.2 * columns, 3.3 * len(case_order)))
    axes = np.asarray(axes).reshape(len(case_order), columns)
    cam = GradCAM(model=model, target_layers=target_layers_for_vit(model), reshape_transform=reshape_transform)

    for row_index, case_type in enumerate(case_order):
        group = selected_cases[selected_cases["case_type"] == case_type].reset_index(drop=True)
        for column_index in range(columns):
            axis = axes[row_index, column_index]
            axis.axis("off")
            if column_index >= len(group):
                axis.text(0.5, 0.5, f"No {case_type} case", ha="center", va="center", transform=axis.transAxes)
                continue

            case = group.iloc[column_index]
            image_path = Path(str(case["path"]))
            image = Image.open(image_path).convert("RGB")
            input_tensor = image_to_tensor(image, device)
            visualization = build_gradcam_overlay(cam, image, input_tensor)
            axis.imshow(visualization)
            axis.set_title(
                f"{case_type} | P={case['p_pneumonia']:.2f}\n{image_path.name}",
                fontsize=8,
            )

        axes[row_index, 0].set_ylabel(case_type, rotation=0, labelpad=28, fontsize=11, weight="bold")

    fig.suptitle("Pneumonia-focused Grad-CAM Representative Cases", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_representative_case_outputs(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.paper_figure) if args.paper_figure else output_dir / "gradcam_representative_cases.png"
    selected_csv = output_dir / "gradcam_representative_cases.csv"

    predictions = collect_case_predictions(
        model,
        args.dataset_dir,
        args.batch_size,
        args.num_workers,
        device,
    )
    selected_cases = select_representative_cases(predictions, args.threshold, args.samples_per_group)
    selected_cases.to_csv(selected_csv, index=False)
    save_case_grid(model, selected_cases, figure_path, args.samples_per_group, device)

    print(f"Saved representative Grad-CAM case metadata to {selected_csv}")
    print(f"Saved representative Grad-CAM figure to {figure_path}")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = get_device()
    model, _metadata = load_model_checkpoint(model_path, device=device)

    if args.auto_cases:
        generate_representative_case_outputs(model, args, device)
        if args.image_path is None:
            return

    image_path = Path(args.image_path or DEFAULT_IMAGE_PATH)
    save_single_gradcam(model, image_path, args.output, device)


if __name__ == "__main__":
    main()
