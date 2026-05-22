"""Bias and fairness analysis using real sensitive-attribute metadata.

This script intentionally refuses to fabricate demographic groups. Provide a CSV
that maps image filenames to a real sensitive attribute, for example age_group,
sex, scanner_site, or hospital_id.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fairlearn.metrics import MetricFrame, false_negative_rate, selection_rate
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import CLASS_NAMES, PNEUMONIA_INDEX, get_device, get_eval_transform, load_model_checkpoint


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fairness analysis with real metadata")
    parser.add_argument("--model-path", default="saved_models/pneumonia_binary_best.pth")
    parser.add_argument("--test-dir", default="chest_xray/test")
    parser.add_argument("--metadata-csv", required=True, help="CSV containing filename and group columns")
    parser.add_argument("--image-column", default="filename")
    parser.add_argument("--group-column", required=True, help="Sensitive attribute column to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="results/bias_analysis_report.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model, _metadata = load_model_checkpoint(args.model_path, device=device)

    metadata = pd.read_csv(args.metadata_csv)
    required_columns = {args.image_column, args.group_column}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")

    metadata = metadata[[args.image_column, args.group_column]].dropna()
    metadata["_basename"] = metadata[args.image_column].map(lambda value: Path(str(value)).name)
    group_lookup = metadata.set_index("_basename")[args.group_column].to_dict()

    dataset = ImageFolderWithPaths(args.test_dir, transform=get_eval_transform())
    class_names = tuple(sorted(dataset.class_to_idx, key=dataset.class_to_idx.get))
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {class_names}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    sensitive_features: list[str] = []
    skipped = 0

    model.eval()
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)[:, PNEUMONIA_INDEX].cpu().numpy()
            preds = (probs >= args.threshold).astype(int)

            for label, pred, path in zip(labels.numpy(), preds, paths):
                group = group_lookup.get(Path(path).name)
                if group is None:
                    skipped += 1
                    continue
                y_true.append(int(label))
                y_pred.append(int(pred))
                sensitive_features.append(str(group))

    if not y_true:
        raise ValueError("No test images matched the provided metadata CSV.")
    if skipped:
        logging.warning("Skipped %d image(s) without metadata.", skipped)

    metrics = {
        "Accuracy": accuracy_score,
        "False Negative Rate": false_negative_rate,
        "Selection Rate": selection_rate,
    }
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        sensitive_features=np.array(sensitive_features),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = metric_frame.by_group
    report.to_csv(output_path)

    summary = metric_frame.difference(method="between_groups")
    logging.info("Saved fairness report to %s", output_path)
    print(report)
    print("\nBetween-group differences:")
    print(summary)


if __name__ == "__main__":
    main()
