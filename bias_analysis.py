"""Subgroup and fairness analysis using real metadata only.

This script intentionally refuses to fabricate demographic groups. Provide a
CSV that maps image filenames to real attributes such as sex, age group, view,
scanner site, or hospital id.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import CLASS_NAMES, PNEUMONIA_INDEX, get_device, get_eval_transform, load_model_checkpoint


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GROUP_ALIASES = {
    "sex": ["sex", "gender", "patient_sex", "patientsex"],
    "age_group": ["age_group", "age group", "agegroup", "age_band", "age band", "age_category"],
    "view": ["view", "view_position", "view position", "viewposition", "projection", "position"],
}


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subgroup analysis with real metadata")
    parser.add_argument("--model-path", default="saved_models/pneumonia_binary_best.pth")
    parser.add_argument("--test-dir", default="chest_xray/test")
    parser.add_argument("--metadata-csv", required=True, help="CSV containing filename and subgroup columns")
    parser.add_argument("--image-column", default="filename")
    parser.add_argument(
        "--group-column",
        default=None,
        help="Backward-compatible single subgroup column, e.g. age_group",
    )
    parser.add_argument(
        "--group-columns",
        nargs="*",
        default=None,
        help="Subgroup columns or canonical names. Defaults to sex age_group view when present.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="results/bias_analysis_report.csv")
    return parser.parse_args()


def normalize_column_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def find_column(metadata: pd.DataFrame, requested: str, required: bool) -> str | None:
    normalized_lookup = {normalize_column_name(column): column for column in metadata.columns}
    normalized_requested = normalize_column_name(requested)
    candidates = [requested, normalized_requested, *GROUP_ALIASES.get(normalized_requested, [])]

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]

    if required:
        raise ValueError(f"Metadata is missing requested subgroup column: {requested}")
    return None


def add_age_group_if_needed(metadata: pd.DataFrame) -> pd.DataFrame:
    if find_column(metadata, "age_group", required=False):
        return metadata

    age_column = find_column(metadata, "age", required=False)
    if age_column is None:
        return metadata

    metadata = metadata.copy()
    age = pd.to_numeric(metadata[age_column], errors="coerce")
    metadata["_derived_age_group"] = pd.cut(
        age,
        bins=[-math.inf, 18, 40, 65, 80, math.inf],
        labels=["<18", "18-39", "40-64", "65-79", "80+"],
        right=False,
    ).astype("object")
    GROUP_ALIASES["age_group"] = [*GROUP_ALIASES["age_group"], "_derived_age_group"]
    return metadata


def resolve_group_columns(metadata: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, str]]:
    metadata = add_age_group_if_needed(metadata)
    requested: list[tuple[str, bool]] = []

    if args.group_column:
        requested.append((args.group_column, True))
    if args.group_columns:
        requested.extend((column, True) for column in args.group_columns)
    if not requested:
        requested = [("sex", False), ("age_group", False), ("view", False)]

    resolved: list[tuple[str, str]] = []
    seen = set()
    for requested_name, required in requested:
        column = find_column(metadata, requested_name, required=required)
        if column is None:
            logging.warning("Metadata column for %s not found; skipping.", requested_name)
            continue
        canonical = normalize_column_name(requested_name)
        if canonical not in GROUP_ALIASES:
            canonical = normalize_column_name(column)
        key = (canonical, column)
        if key not in seen:
            resolved.append(key)
            seen.add(key)

    if not resolved:
        raise ValueError("No subgroup columns were found in metadata.")
    return resolved


def metadata_lookup(metadata: pd.DataFrame, image_column: str) -> dict[str, pd.Series]:
    if image_column not in metadata.columns:
        raise ValueError(f"Metadata is missing image column: {image_column}")

    lookup: dict[str, pd.Series] = {}
    for _, row in metadata.iterrows():
        raw_value = str(row[image_column])
        basename = Path(raw_value).name.lower()
        stem = Path(raw_value).stem.lower()
        lookup.setdefault(basename, row)
        lookup.setdefault(stem, row)
    return lookup


def collect_predictions(args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    device = get_device()
    model, _metadata = load_model_checkpoint(args.model_path, device=device)

    metadata = pd.read_csv(args.metadata_csv)
    metadata = add_age_group_if_needed(metadata)
    groups = resolve_group_columns(metadata, args)
    lookup = metadata_lookup(metadata, args.image_column)

    dataset = ImageFolderWithPaths(args.test_dir, transform=get_eval_transform())
    class_names = tuple(sorted(dataset.class_to_idx, key=dataset.class_to_idx.get))
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {class_names}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows: list[dict[str, object]] = []
    skipped = 0

    model.eval()
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1)[:, PNEUMONIA_INDEX].cpu().numpy()
            predictions = (probabilities >= args.threshold).astype(int)

            for label, prediction, probability, path in zip(labels.numpy(), predictions, probabilities, paths):
                image_path = Path(str(path))
                metadata_row = lookup.get(image_path.name.lower())
                if metadata_row is None:
                    metadata_row = lookup.get(image_path.stem.lower())
                if metadata_row is None:
                    skipped += 1
                    continue

                row: dict[str, object] = {
                    "path": str(image_path),
                    "filename": image_path.name,
                    "true_binary": int(label),
                    "true_label": CLASS_NAMES[int(label)],
                    "predicted_binary": int(prediction),
                    "predicted_label": CLASS_NAMES[int(prediction)],
                    "p_pneumonia": float(probability),
                    "threshold": args.threshold,
                }
                for canonical, column in groups:
                    row[canonical] = metadata_row[column]
                    row[f"{canonical}__metadata_column"] = column
                rows.append(row)

    if not rows:
        raise ValueError("No test images matched the provided metadata CSV.")
    return pd.DataFrame(rows), skipped


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return (centre - margin) / denominator, (centre + margin) / denominator


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, probabilities))


def safe_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(average_precision_score(labels, probabilities))


def subgroup_metrics(frame: pd.DataFrame, group_column: str, group_value: object) -> dict[str, object]:
    y_true = frame["true_binary"].to_numpy(dtype=int)
    y_pred = frame["predicted_binary"].to_numpy(dtype=int)
    probabilities = frame["p_pneumonia"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity_denominator = tp + fn
    specificity_denominator = tn + fp
    sensitivity = tp / sensitivity_denominator if sensitivity_denominator else np.nan
    specificity = tn / specificity_denominator if specificity_denominator else np.nan
    sensitivity_low, sensitivity_high = wilson_ci(int(tp), int(sensitivity_denominator))
    specificity_low, specificity_high = wilson_ci(int(tn), int(specificity_denominator))

    return {
        "group_column": group_column,
        "group_value": group_value,
        "n": int(len(frame)),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "prevalence": float((y_true == 1).mean()) if len(frame) else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(frame) else np.nan,
        "sensitivity": sensitivity,
        "sensitivity_95_ci_lower": sensitivity_low,
        "sensitivity_95_ci_upper": sensitivity_high,
        "specificity": specificity,
        "specificity_95_ci_lower": specificity_low,
        "specificity_95_ci_upper": specificity_high,
        "precision_ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "selection_rate": float((y_pred == 1).mean()) if len(frame) else np.nan,
        "auc_roc": safe_auc(y_true, probabilities),
        "auc_pr": safe_average_precision(y_true, probabilities),
        "brier_score": float(brier_score_loss(y_true, probabilities)) if len(frame) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_subgroup_report(predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        column
        for column in predictions.columns
        if column not in {
            "path",
            "filename",
            "true_binary",
            "true_label",
            "predicted_binary",
            "predicted_label",
            "p_pneumonia",
            "threshold",
        }
        and not column.endswith("__metadata_column")
    ]

    rows = []
    for group_column in group_columns:
        for group_value, frame in predictions.dropna(subset=[group_column]).groupby(group_column, dropna=True):
            rows.append(subgroup_metrics(frame, group_column, group_value))
    return pd.DataFrame(rows)


def between_group_differences(report: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "sensitivity", "specificity", "false_negative_rate", "selection_rate", "auc_roc"]
    rows = []
    for group_column, frame in report.groupby("group_column"):
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna()
            rows.append(
                {
                    "group_column": group_column,
                    "metric": metric,
                    "min": float(values.min()) if not values.empty else np.nan,
                    "max": float(values.max()) if not values.empty else np.nan,
                    "between_group_difference": float(values.max() - values.min()) if not values.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    predictions, skipped = collect_predictions(args)
    if skipped:
        logging.warning("Skipped %d image(s) without metadata.", skipped)

    report = build_subgroup_report(predictions)
    differences = between_group_differences(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path.with_name(f"{output_path.stem}_predictions.csv")
    differences_path = output_path.with_name(f"{output_path.stem}_between_group_differences.csv")

    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    differences.to_csv(differences_path, index=False, encoding="utf-8-sig")

    logging.info("Saved subgroup report to %s", output_path)
    logging.info("Saved matched predictions to %s", predictions_path)
    logging.info("Saved between-group differences to %s", differences_path)
    print(report.to_string(index=False))
    print("\nBetween-group differences:")
    print(differences.to_string(index=False))


if __name__ == "__main__":
    main()
