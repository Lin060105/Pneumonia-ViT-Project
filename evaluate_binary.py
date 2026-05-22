"""Evaluate the binary ViT pneumonia model with a fixed clinical threshold."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import CLASS_NAMES, PNEUMONIA_INDEX, get_device, get_eval_transform, load_model_checkpoint


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a binary pneumonia model")
    parser.add_argument("--model-path", default="saved_models/pneumonia_binary_best.pth")
    parser.add_argument("--test-dir", default="chest_xray/test")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed pneumonia probability threshold")
    parser.add_argument(
        "--threshold-from-dir",
        default=None,
        help="Optional validation folder used only to choose threshold before testing",
    )
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def collect_predictions(
    model: torch.nn.Module,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = datasets.ImageFolder(data_dir, transform=get_eval_transform())
    class_names = tuple(sorted(dataset.class_to_idx, key=dataset.class_to_idx.get))
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {class_names} in {data_dir}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels_all: list[int] = []
    probs_all: list[float] = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, PNEUMONIA_INDEX]
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    return np.array(labels_all), np.array(probs_all)


def predictions_at_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = predictions_at_threshold(probabilities, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    accuracy = safe_divide(tp + tn, len(labels))
    f1 = safe_divide(2 * precision * sensitivity, precision + sensitivity)

    metrics = {
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision_PPV": precision,
        "NPV": npv,
        "F1": f1,
        "AUC_ROC": roc_auc_score(labels, probabilities),
        "AUC_PR": average_precision_score(labels, probabilities),
        "Brier_Score": brier_score_loss(labels, probabilities),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "TP": float(tp),
    }
    return metrics


def choose_threshold_for_sensitivity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float,
) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(0.0, 1.0, 1001)
    candidates: list[tuple[float, dict[str, float]]] = []

    for threshold in thresholds:
        metrics = compute_metrics(labels, probabilities, float(threshold))
        if metrics["Sensitivity"] >= target_sensitivity:
            candidates.append((float(threshold), metrics))

    if not candidates:
        threshold = 0.0
        return threshold, compute_metrics(labels, probabilities, threshold)

    return max(candidates, key=lambda item: (item[1]["Specificity"], item[0]))


def bootstrap_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    metric_func: Callable[[np.ndarray, np.ndarray, float], float],
    n_bootstraps: int,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    indices = np.arange(len(labels))

    for _ in range(n_bootstraps):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sample])) < 2:
            continue
        scores.append(metric_func(labels[sample], probabilities[sample], threshold))

    if not scores:
        return np.nan, np.nan
    return tuple(np.percentile(scores, [2.5, 97.5]).tolist())


def add_confidence_intervals(
    metrics: dict[str, float],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_bootstraps: int,
) -> pd.DataFrame:
    metric_functions: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = {
        "Accuracy": lambda y, p, t: compute_metrics(y, p, t)["Accuracy"],
        "Sensitivity": lambda y, p, t: compute_metrics(y, p, t)["Sensitivity"],
        "Specificity": lambda y, p, t: compute_metrics(y, p, t)["Specificity"],
        "Precision_PPV": lambda y, p, t: compute_metrics(y, p, t)["Precision_PPV"],
        "NPV": lambda y, p, t: compute_metrics(y, p, t)["NPV"],
        "AUC_ROC": lambda y, p, _t: roc_auc_score(y, p),
        "AUC_PR": lambda y, p, _t: average_precision_score(y, p),
    }

    rows = []
    for metric_name, value in metrics.items():
        lower = np.nan
        upper = np.nan
        if metric_name in metric_functions and n_bootstraps > 0:
            lower, upper = bootstrap_ci(
                labels,
                probabilities,
                threshold,
                metric_functions[metric_name],
                n_bootstraps,
            )
        rows.append(
            {
                "Metric": metric_name,
                "Value": value,
                "95% CI Lower": lower,
                "95% CI Upper": upper,
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    predictions = predictions_at_threshold(probabilities, threshold)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0, 0])
    axes[0, 0].set_title(f"Confusion Matrix @ threshold={threshold:.3f}")
    axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xlabel("Predicted Label")

    prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=10, strategy="quantile")
    axes[0, 1].plot(prob_pred, prob_true, marker="o", linewidth=2, label="ViT")
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axes[0, 1].set_title("Calibration Curve")
    axes[0, 1].set_xlabel("Mean Predicted Pneumonia Probability")
    axes[0, 1].set_ylabel("Observed Pneumonia Fraction")
    axes[0, 1].legend()

    fpr, tpr, _ = roc_curve(labels, probabilities)
    axes[1, 0].plot(fpr, tpr, linewidth=2)
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1, 0].set_title("ROC Curve")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    axes[1, 1].plot(recall, precision, linewidth=2)
    axes[1, 1].set_title("Precision-Recall Curve")
    axes[1, 1].set_xlabel("Recall / Sensitivity")
    axes[1, 1].set_ylabel("Precision / PPV")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model from %s on %s", args.model_path, device)
    model, metadata = load_model_checkpoint(args.model_path, device=device)

    threshold = args.threshold
    if threshold is None:
        threshold = float(metadata.get("threshold", 0.5))

    if args.threshold_from_dir:
        logging.info("Selecting threshold from validation folder: %s", args.threshold_from_dir)
        val_labels, val_probs = collect_predictions(
            model,
            args.threshold_from_dir,
            args.batch_size,
            args.num_workers,
            device,
        )
        threshold, val_metrics = choose_threshold_for_sensitivity(
            val_labels,
            val_probs,
            args.target_sensitivity,
        )
        pd.DataFrame([val_metrics]).to_csv(output_dir / "threshold_selection_report.csv", index=False)
        logging.info(
            "Selected threshold %.3f for target sensitivity %.3f",
            threshold,
            args.target_sensitivity,
        )

    logging.info("Evaluating test folder: %s", args.test_dir)
    labels, probabilities = collect_predictions(model, args.test_dir, args.batch_size, args.num_workers, device)
    metrics = compute_metrics(labels, probabilities, threshold)
    report = add_confidence_intervals(metrics, labels, probabilities, threshold, args.bootstrap_samples)
    report.to_csv(output_dir / "clinical_metrics_report.csv", index=False)
    save_plots(labels, probabilities, threshold, output_dir / "clinical_evaluation_plots.png")

    logging.info("Saved metrics to %s", output_dir / "clinical_metrics_report.csv")
    logging.info("Saved plots to %s", output_dir / "clinical_evaluation_plots.png")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
