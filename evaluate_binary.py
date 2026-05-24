"""Advanced clinical evaluation for binary pneumonia screening models.

The script keeps the original fixed-threshold workflow while adding:
- 95% bootstrap confidence intervals
- paired bootstrap ROC AUC comparison between checkpoints
- Brier score, ECE, calibration bins, and calibration plots
- decision curve analysis
- external validation on a locked threshold, including preprocessed RSNA data
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from model_utils import (
    CLASS_NAMES,
    PNEUMONIA_INDEX,
    create_model,
    get_device,
    get_eval_transform,
    load_model_checkpoint,
    unpack_checkpoint,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class PredictionSet:
    name: str
    labels: np.ndarray
    probabilities: np.ndarray
    paths: list[str]


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


class RSNADicomDataset(Dataset):
    """Read RSNA Pneumonia Detection Challenge DICOM files from label CSV.

    Labels are collapsed to one binary target per patient using max(Target).
    This avoids double-counting patients with multiple bounding boxes.
    """

    class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(self, labels_csv: str | Path, image_dir: str | Path, transform=None) -> None:
        self.labels_csv = Path(labels_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform

        labels = pd.read_csv(self.labels_csv)
        required_columns = {"patientId", "Target"}
        missing = required_columns.difference(labels.columns)
        if missing:
            raise ValueError(f"RSNA label CSV is missing required columns: {sorted(missing)}")

        patient_targets = labels.groupby("patientId")["Target"].max().astype(int)
        self.samples: list[tuple[Path, int]] = []
        for patient_id, target in patient_targets.items():
            dcm_path = self.image_dir / f"{patient_id}.dcm"
            if dcm_path.exists():
                self.samples.append((dcm_path, int(target)))

        if not self.samples:
            raise FileNotFoundError(f"No RSNA DICOM files from {self.labels_csv} were found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("pydicom is required for --rsna-labels-csv/--rsna-image-dir evaluation") from exc

        path, label = self.samples[index]
        dicom = pydicom.dcmread(str(path))
        array = dicom.pixel_array.astype(np.float32)
        array -= float(array.min())
        max_value = float(array.max())
        if max_value > 0:
            array /= max_value
        image = Image.fromarray(np.uint8(array * 255)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a binary pneumonia model")
    parser.add_argument("--model-path", default="saved_models/pneumonia_binary_best.pth")
    parser.add_argument("--model-name", default=None, help="Optional timm model override for legacy checkpoints")
    parser.add_argument("--test-dir", default="chest_xray/test")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed pneumonia probability threshold")
    parser.add_argument(
        "--threshold-from-dir",
        default=None,
        help="Internal validation ImageFolder used only to choose threshold before test/external evaluation",
    )
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--external-dir", default=None, help="External ImageFolder root, e.g. data/rsna_processed")
    parser.add_argument("--external-rsna-dir", default=None, help="Alias for --external-dir")
    parser.add_argument("--rsna-labels-csv", default=None, help="RSNA stage_2_train_labels.csv for DICOM evaluation")
    parser.add_argument("--rsna-image-dir", default=None, help="Folder containing RSNA .dcm files")
    parser.add_argument("--compare-model-paths", nargs="*", default=[], help="Additional checkpoints for ROC AUC comparison")
    parser.add_argument(
        "--compare-model-names",
        nargs="*",
        default=None,
        help="Optional timm model overrides matching --compare-model-paths",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def torch_load_checkpoint(path: str | Path, map_location: torch.device | str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_evaluation_model(
    path: str | Path,
    device: torch.device,
    model_name_override: str | None = None,
) -> tuple[torch.nn.Module, dict[str, object]]:
    if model_name_override is None:
        return load_model_checkpoint(path, device=device)

    payload = torch_load_checkpoint(path, map_location=device)
    state_dict, metadata = unpack_checkpoint(payload)
    metadata["model_name"] = model_name_override
    class_names = tuple(metadata.get("class_names", CLASS_NAMES))
    model = create_model(model_name=model_name_override, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    metadata.setdefault("class_names", class_names)
    return model, metadata


def model_label(path: str | Path, metadata: dict[str, object]) -> str:
    return str(metadata.get("display_name") or metadata.get("model_name") or Path(path).stem)


def build_imagefolder_dataset(data_dir: str | Path) -> ImageFolderWithPaths:
    dataset = ImageFolderWithPaths(data_dir, transform=get_eval_transform())
    class_names = tuple(sorted(dataset.class_to_idx, key=dataset.class_to_idx.get))
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {class_names} in {data_dir}")
    return dataset


def build_external_dataset(args: argparse.Namespace) -> Dataset | None:
    if args.rsna_labels_csv or args.rsna_image_dir:
        if not args.rsna_labels_csv or not args.rsna_image_dir:
            raise ValueError("--rsna-labels-csv and --rsna-image-dir must be provided together")
        return RSNADicomDataset(args.rsna_labels_csv, args.rsna_image_dir, transform=get_eval_transform())

    external_dir = args.external_dir or args.external_rsna_dir
    if external_dir:
        return build_imagefolder_dataset(external_dir)
    return None


def collect_predictions(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    name: str,
) -> PredictionSet:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels_all: list[int] = []
    probs_all: list[float] = []
    paths_all: list[str] = []

    model.eval()
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, PNEUMONIA_INDEX]
            labels_all.extend(labels.cpu().numpy().astype(int).tolist())
            probs_all.extend(probabilities.cpu().numpy().astype(float).tolist())
            paths_all.extend([str(path) for path in paths])

    return PredictionSet(
        name=name,
        labels=np.asarray(labels_all, dtype=int),
        probabilities=np.asarray(probs_all, dtype=float),
        paths=paths_all,
    )


def collect_predictions_from_folder(
    model: torch.nn.Module,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    name: str,
) -> PredictionSet:
    return collect_predictions(model, build_imagefolder_dataset(data_dir), batch_size, num_workers, device, name)


def predictions_at_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def safe_roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def safe_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, bin_edges[1:-1], right=False)
    ece = 0.0
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if not np.any(in_bin):
            continue
        observed = float(labels[in_bin].mean())
        predicted = float(probabilities[in_bin].mean())
        ece += float(in_bin.mean()) * abs(observed - predicted)
    return ece


def calibration_bin_table(labels: np.ndarray, probabilities: np.ndarray, n_bins: int) -> pd.DataFrame:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, bin_edges[1:-1], right=False)
    rows = []
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        rows.append(
            {
                "bin": bin_id + 1,
                "probability_lower": bin_edges[bin_id],
                "probability_upper": bin_edges[bin_id + 1],
                "n": int(in_bin.sum()),
                "mean_predicted_probability": float(probabilities[in_bin].mean()) if np.any(in_bin) else np.nan,
                "observed_pneumonia_fraction": float(labels[in_bin].mean()) if np.any(in_bin) else np.nan,
                "absolute_calibration_error": (
                    abs(float(labels[in_bin].mean()) - float(probabilities[in_bin].mean())) if np.any(in_bin) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    ece_bins: int = 10,
) -> dict[str, float]:
    predictions = predictions_at_threshold(probabilities, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    accuracy = safe_divide(tp + tn, len(labels))
    f1 = safe_divide(2 * precision * sensitivity, precision + sensitivity)

    return {
        "Threshold": threshold,
        "N": float(len(labels)),
        "Prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision_PPV": precision,
        "NPV": npv,
        "F1": f1,
        "AUC_ROC": safe_roc_auc(labels, probabilities),
        "AUC_PR": safe_average_precision(labels, probabilities),
        "Brier_Score": float(brier_score_loss(labels, probabilities)) if len(labels) else float("nan"),
        "ECE": expected_calibration_error(labels, probabilities, n_bins=ece_bins),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "TP": float(tp),
    }


def choose_threshold_for_sensitivity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float,
    ece_bins: int,
) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(0.0, 1.0, 1001)
    candidates: list[tuple[float, dict[str, float]]] = []

    for threshold in thresholds:
        metrics = compute_metrics(labels, probabilities, float(threshold), ece_bins=ece_bins)
        if metrics["Sensitivity"] >= target_sensitivity:
            candidates.append((float(threshold), metrics))

    if not candidates:
        threshold = 0.0
        return threshold, compute_metrics(labels, probabilities, threshold, ece_bins=ece_bins)

    return max(candidates, key=lambda item: (item[1]["Specificity"], item[0]))


MetricFunction = Callable[[np.ndarray, np.ndarray, float], float]


def bootstrap_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    metric_func: MetricFunction,
    n_bootstraps: int,
    seed: int,
    require_two_classes: bool = False,
) -> tuple[float, float]:
    if n_bootstraps <= 0 or len(labels) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    scores: list[float] = []
    indices = np.arange(len(labels))

    for _ in range(n_bootstraps):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if require_two_classes and len(np.unique(labels[sample])) < 2:
            continue
        try:
            score = float(metric_func(labels[sample], probabilities[sample], threshold))
        except ValueError:
            continue
        if np.isfinite(score):
            scores.append(score)

    if not scores:
        return np.nan, np.nan
    return tuple(np.percentile(scores, [2.5, 97.5]).tolist())


def add_confidence_intervals(
    metrics: dict[str, float],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_bootstraps: int,
    seed: int,
    ece_bins: int,
) -> pd.DataFrame:
    metric_functions: dict[str, tuple[MetricFunction, bool]] = {
        "Accuracy": (lambda y, p, t: compute_metrics(y, p, t, ece_bins)["Accuracy"], False),
        "Sensitivity": (lambda y, p, t: compute_metrics(y, p, t, ece_bins)["Sensitivity"], False),
        "Specificity": (lambda y, p, t: compute_metrics(y, p, t, ece_bins)["Specificity"], False),
        "Precision_PPV": (lambda y, p, t: compute_metrics(y, p, t, ece_bins)["Precision_PPV"], False),
        "NPV": (lambda y, p, t: compute_metrics(y, p, t, ece_bins)["NPV"], False),
        "F1": (lambda y, p, t: compute_metrics(y, p, t, ece_bins)["F1"], False),
        "AUC_ROC": (lambda y, p, _t: safe_roc_auc(y, p), True),
        "AUC_PR": (lambda y, p, _t: safe_average_precision(y, p), True),
        "Brier_Score": (lambda y, p, _t: float(brier_score_loss(y, p)), False),
        "ECE": (lambda y, p, _t: expected_calibration_error(y, p, n_bins=ece_bins), False),
    }

    rows = []
    for metric_name, value in metrics.items():
        lower = np.nan
        upper = np.nan
        if metric_name in metric_functions:
            metric_func, require_two_classes = metric_functions[metric_name]
            lower, upper = bootstrap_ci(
                labels,
                probabilities,
                threshold,
                metric_func,
                n_bootstraps,
                seed,
                require_two_classes=require_two_classes,
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


def decision_curve_table(labels: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, 99)
    n = len(labels)
    prevalence = float(labels.mean()) if n else float("nan")
    rows = []
    for threshold in thresholds:
        predictions = predictions_at_threshold(probabilities, float(threshold))
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        odds = threshold / (1.0 - threshold)
        model_net_benefit = (tp / n) - (fp / n) * odds if n else np.nan
        treat_all_net_benefit = prevalence - (1.0 - prevalence) * odds if n else np.nan
        rows.append(
            {
                "threshold_probability": float(threshold),
                "model_net_benefit": float(model_net_benefit),
                "treat_all_net_benefit": float(treat_all_net_benefit),
                "treat_none_net_benefit": 0.0,
                "tp": float(tp),
                "fp": float(fp),
                "tn": float(tn),
                "fn": float(fn),
            }
        )
    return pd.DataFrame(rows)


def prediction_dataframe(prediction_set: PredictionSet, threshold: float) -> pd.DataFrame:
    predictions = predictions_at_threshold(prediction_set.probabilities, threshold)
    return pd.DataFrame(
        {
            "path": prediction_set.paths,
            "true_label": [CLASS_NAMES[label] for label in prediction_set.labels],
            "true_binary": prediction_set.labels,
            "p_pneumonia": prediction_set.probabilities,
            "predicted_binary": predictions,
            "predicted_label": [CLASS_NAMES[prediction] for prediction in predictions],
            "threshold": threshold,
        }
    )


def plot_unavailable(axis: plt.Axes, title: str, message: str) -> None:
    axis.set_title(title)
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)


def save_evaluation_plots(
    prediction_set: PredictionSet,
    threshold: float,
    output_path: Path,
    ece_bins: int,
) -> None:
    labels = prediction_set.labels
    probabilities = prediction_set.probabilities
    predictions = predictions_at_threshold(probabilities, threshold)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics = compute_metrics(labels, probabilities, threshold, ece_bins=ece_bins)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0, 0])
    axes[0, 0].set_title(f"{prediction_set.name} Confusion Matrix @ threshold={threshold:.3f}")
    axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xlabel("Predicted Label")

    try:
        prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=ece_bins, strategy="quantile")
        axes[0, 1].plot(prob_pred, prob_true, marker="o", linewidth=2, label=prediction_set.name)
        axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        axes[0, 1].set_title(f"Calibration Curve | Brier={metrics['Brier_Score']:.3f} ECE={metrics['ECE']:.3f}")
        axes[0, 1].set_xlabel("Mean Predicted Pneumonia Probability")
        axes[0, 1].set_ylabel("Observed Pneumonia Fraction")
        axes[0, 1].legend()
    except ValueError:
        plot_unavailable(axes[0, 1], "Calibration Curve", "Calibration unavailable")

    if len(np.unique(labels)) >= 2:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        axes[1, 0].plot(fpr, tpr, linewidth=2, label=f"AUC={metrics['AUC_ROC']:.3f}")
        axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[1, 0].set_title("ROC Curve")
        axes[1, 0].set_xlabel("False Positive Rate")
        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1, 0].legend()

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        axes[1, 1].plot(recall, precision, linewidth=2, label=f"AP={metrics['AUC_PR']:.3f}")
        axes[1, 1].set_title("Precision-Recall Curve")
        axes[1, 1].set_xlabel("Recall / Sensitivity")
        axes[1, 1].set_ylabel("Precision / PPV")
        axes[1, 1].legend()
    else:
        plot_unavailable(axes[1, 0], "ROC Curve", "ROC unavailable: one class present")
        plot_unavailable(axes[1, 1], "Precision-Recall Curve", "PR unavailable: one class present")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_decision_curve_plot(table: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot(table["threshold_probability"], table["model_net_benefit"], linewidth=2, label="Model")
    axis.plot(table["threshold_probability"], table["treat_all_net_benefit"], linestyle="--", label="Treat all")
    axis.plot(table["threshold_probability"], table["treat_none_net_benefit"], linestyle=":", label="Treat none")
    axis.set_title(title)
    axis.set_xlabel("Threshold Probability")
    axis.set_ylabel("Net Benefit")
    axis.set_xlim(0.01, 0.99)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def output_names(prefix: str) -> dict[str, str]:
    if prefix == "clinical":
        return {
            "metrics": "clinical_metrics_report.csv",
            "plots": "clinical_evaluation_plots.png",
            "predictions": "clinical_predictions.csv",
            "calibration": "clinical_calibration_bins.csv",
            "decision_curve": "clinical_decision_curve.csv",
            "decision_curve_plot": "clinical_decision_curve.png",
        }
    return {
        "metrics": f"{prefix}_metrics_report.csv",
        "plots": f"{prefix}_evaluation_plots.png",
        "predictions": f"{prefix}_predictions.csv",
        "calibration": f"{prefix}_calibration_bins.csv",
        "decision_curve": f"{prefix}_decision_curve.csv",
        "decision_curve_plot": f"{prefix}_decision_curve.png",
    }


def save_prediction_set_outputs(
    prediction_set: PredictionSet,
    threshold: float,
    output_dir: Path,
    prefix: str,
    n_bootstraps: int,
    seed: int,
    ece_bins: int,
) -> pd.DataFrame:
    names = output_names(prefix)
    metrics = compute_metrics(prediction_set.labels, prediction_set.probabilities, threshold, ece_bins=ece_bins)
    report = add_confidence_intervals(
        metrics,
        prediction_set.labels,
        prediction_set.probabilities,
        threshold,
        n_bootstraps,
        seed,
        ece_bins,
    )
    report.to_csv(output_dir / names["metrics"], index=False)

    prediction_dataframe(prediction_set, threshold).to_csv(output_dir / names["predictions"], index=False)
    calibration_bin_table(prediction_set.labels, prediction_set.probabilities, ece_bins).to_csv(
        output_dir / names["calibration"],
        index=False,
    )
    dca = decision_curve_table(prediction_set.labels, prediction_set.probabilities)
    dca.to_csv(output_dir / names["decision_curve"], index=False)

    save_evaluation_plots(prediction_set, threshold, output_dir / names["plots"], ece_bins)
    save_decision_curve_plot(dca, output_dir / names["decision_curve_plot"], f"{prediction_set.name} Decision Curve")
    return report


def paired_auc_difference_ci(
    labels: np.ndarray,
    reference_probabilities: np.ndarray,
    comparison_probabilities: np.ndarray,
    n_bootstraps: int,
    seed: int,
) -> dict[str, float]:
    reference_auc = safe_roc_auc(labels, reference_probabilities)
    comparison_auc = safe_roc_auc(labels, comparison_probabilities)
    observed_difference = comparison_auc - reference_auc

    if n_bootstraps <= 0 or len(np.unique(labels)) < 2:
        return {
            "reference_auc": reference_auc,
            "comparison_auc": comparison_auc,
            "auc_difference_vs_reference": observed_difference,
            "difference_95_ci_lower": np.nan,
            "difference_95_ci_upper": np.nan,
            "paired_bootstrap_p_value": np.nan,
        }

    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    differences: list[float] = []
    for _ in range(n_bootstraps):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sample])) < 2:
            continue
        ref_auc = safe_roc_auc(labels[sample], reference_probabilities[sample])
        cmp_auc = safe_roc_auc(labels[sample], comparison_probabilities[sample])
        if np.isfinite(ref_auc) and np.isfinite(cmp_auc):
            differences.append(cmp_auc - ref_auc)

    if not differences:
        lower = upper = p_value = np.nan
    else:
        diff_array = np.asarray(differences, dtype=float)
        lower, upper = np.percentile(diff_array, [2.5, 97.5]).tolist()
        p_value = 2 * min(float(np.mean(diff_array <= 0)), float(np.mean(diff_array >= 0)))
        p_value = min(1.0, p_value)

    return {
        "reference_auc": reference_auc,
        "comparison_auc": comparison_auc,
        "auc_difference_vs_reference": observed_difference,
        "difference_95_ci_lower": lower,
        "difference_95_ci_upper": upper,
        "paired_bootstrap_p_value": p_value,
    }


def validate_compare_args(args: argparse.Namespace) -> Sequence[str | None]:
    if args.compare_model_names is None:
        return [None] * len(args.compare_model_paths)
    if len(args.compare_model_names) != len(args.compare_model_paths):
        raise ValueError("--compare-model-names must match the length of --compare-model-paths")
    return args.compare_model_names


def save_model_comparison(
    args: argparse.Namespace,
    output_dir: Path,
    dataset: Dataset,
    reference_prediction_set: PredictionSet,
    reference_label: str,
    device: torch.device,
) -> None:
    if not args.compare_model_paths:
        return

    compare_model_names = validate_compare_args(args)
    rows = [
        {
            "model": reference_label,
            "model_path": args.model_path,
            "auc_roc": safe_roc_auc(reference_prediction_set.labels, reference_prediction_set.probabilities),
            "auc_difference_vs_reference": 0.0,
            "difference_95_ci_lower": 0.0,
            "difference_95_ci_upper": 0.0,
            "paired_bootstrap_p_value": np.nan,
            "reference_model": True,
        }
    ]

    for compare_path, compare_model_name in zip(args.compare_model_paths, compare_model_names):
        model, metadata = load_evaluation_model(compare_path, device=device, model_name_override=compare_model_name)
        label = model_label(compare_path, metadata)
        comparison_predictions = collect_predictions(
            model,
            dataset,
            args.batch_size,
            args.num_workers,
            device,
            name=label,
        )
        if not np.array_equal(comparison_predictions.labels, reference_prediction_set.labels):
            raise ValueError(f"Labels for comparison model do not match reference: {compare_path}")

        comparison = paired_auc_difference_ci(
            reference_prediction_set.labels,
            reference_prediction_set.probabilities,
            comparison_predictions.probabilities,
            n_bootstraps=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        rows.append(
            {
                "model": label,
                "model_path": compare_path,
                "auc_roc": comparison["comparison_auc"],
                "auc_difference_vs_reference": comparison["auc_difference_vs_reference"],
                "difference_95_ci_lower": comparison["difference_95_ci_lower"],
                "difference_95_ci_upper": comparison["difference_95_ci_upper"],
                "paired_bootstrap_p_value": comparison["paired_bootstrap_p_value"],
                "reference_model": False,
            }
        )

    comparison_path = output_dir / "model_comparison_auc_report.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    logging.info("Saved paired ROC AUC comparison to %s", comparison_path)


def main() -> None:
    args = parse_args()
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model from %s on %s", args.model_path, device)
    model, metadata = load_evaluation_model(args.model_path, device=device, model_name_override=args.model_name)
    primary_label = model_label(args.model_path, metadata)

    threshold = args.threshold
    threshold_source = "command line"
    if threshold is None:
        threshold = float(metadata.get("threshold", 0.5))
        threshold_source = "checkpoint metadata/default"

    if args.threshold_from_dir:
        logging.info("Selecting threshold from internal validation folder: %s", args.threshold_from_dir)
        validation_predictions = collect_predictions_from_folder(
            model,
            args.threshold_from_dir,
            args.batch_size,
            args.num_workers,
            device,
            name="Internal validation",
        )
        threshold, validation_metrics = choose_threshold_for_sensitivity(
            validation_predictions.labels,
            validation_predictions.probabilities,
            args.target_sensitivity,
            ece_bins=args.ece_bins,
        )
        threshold_source = f"internal validation target sensitivity {args.target_sensitivity:.3f}"
        pd.DataFrame([validation_metrics]).to_csv(output_dir / "threshold_selection_report.csv", index=False)
        prediction_dataframe(validation_predictions, threshold).to_csv(
            output_dir / "threshold_selection_predictions.csv",
            index=False,
        )
        logging.info("Selected threshold %.3f from internal validation only", threshold)

    logging.info("Evaluating internal test folder: %s", args.test_dir)
    test_dataset = build_imagefolder_dataset(args.test_dir)
    test_predictions = collect_predictions(
        model,
        test_dataset,
        args.batch_size,
        args.num_workers,
        device,
        name="Internal test",
    )
    clinical_report = save_prediction_set_outputs(
        test_predictions,
        threshold,
        output_dir,
        prefix="clinical",
        n_bootstraps=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        ece_bins=args.ece_bins,
    )
    save_model_comparison(args, output_dir, test_dataset, test_predictions, primary_label, device)

    external_dataset = build_external_dataset(args)
    external_report = None
    if external_dataset is not None:
        logging.info("Evaluating external dataset with locked threshold %.3f (%s)", threshold, threshold_source)
        external_predictions = collect_predictions(
            model,
            external_dataset,
            args.batch_size,
            args.num_workers,
            device,
            name="External validation",
        )
        external_report = save_prediction_set_outputs(
            external_predictions,
            threshold,
            output_dir,
            prefix="external_validation",
            n_bootstraps=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            ece_bins=args.ece_bins,
        )

    logging.info("Threshold used for all final evaluations: %.3f (%s)", threshold, threshold_source)
    logging.info("Saved metrics to %s", output_dir / "clinical_metrics_report.csv")
    logging.info("Saved plots to %s", output_dir / "clinical_evaluation_plots.png")
    print("\nInternal test report:")
    print(clinical_report.to_string(index=False))
    if external_report is not None:
        print("\nExternal validation report:")
        print(external_report.to_string(index=False))


if __name__ == "__main__":
    main()
