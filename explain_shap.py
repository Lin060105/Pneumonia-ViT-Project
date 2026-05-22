"""Generate a small SHAP explanation figure for the binary ViT model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import shap
import torch
from torchvision import datasets, transforms

from model_utils import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, get_device, load_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SHAP image explanations")
    parser.add_argument("--model-path", default="saved_models/pneumonia_binary_best.pth")
    parser.add_argument("--data-dir", default="chest_xray/test")
    parser.add_argument("--output", default="shap_explanation.png")
    parser.add_argument("--max-evals", type=int, default=500)
    return parser.parse_args()


def pick_one_per_class(dataset: datasets.ImageFolder) -> torch.Tensor:
    selected = []
    for class_index in range(len(CLASS_NAMES)):
        for image, label in dataset:
            if label == class_index:
                selected.append(image)
                break
    if len(selected) != len(CLASS_NAMES):
        raise ValueError("Could not find at least one image for every class.")
    return torch.stack(selected)


def main() -> None:
    args = parse_args()
    device = get_device()
    model, _metadata = load_model_checkpoint(args.model_path, device=device)

    base_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(args.data_dir, transform=base_transform)
    class_names = tuple(sorted(dataset.class_to_idx, key=dataset.class_to_idx.get))
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {class_names}")

    images_tensor = pick_one_per_class(dataset)
    images_np = images_tensor.permute(0, 2, 3, 1).numpy()

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    def predict_fn(images_np_batch):
        images = torch.tensor(images_np_batch).permute(0, 3, 1, 2).float().to(device)
        images = (images - mean) / std
        with torch.no_grad():
            return model(images).cpu().numpy()

    masker = shap.maskers.Image("blur(128,128)", images_np[0].shape)
    explainer = shap.Explainer(predict_fn, masker, output_names=list(CLASS_NAMES))
    shap_values = explainer(
        images_np,
        max_evals=args.max_evals,
        outputs=shap.Explanation.argsort.flip[:1],
    )

    shap.image_plot(shap_values, show=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP explanation to {output_path}")


if __name__ == "__main__":
    main()
