"""Train publication baseline models under one shared policy.

Models:
- ResNet18
- ResNet50
- DenseNet121
- EfficientNet-B0
- ViT-B/16

Each model is trained across multiple random seeds with the same image size,
normalization, optimizer, scheduler, class order, and validation strategy.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from tqdm import tqdm

from model_utils import (
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PNEUMONIA_INDEX,
    create_model,
    get_device,
    save_model_checkpoint,
)
from train_binary import build_loaders, set_seed


MODEL_REGISTRY = {
    "resnet18": ("ResNet18", "resnet18"),
    "resnet50": ("ResNet50", "resnet50"),
    "densenet121": ("DenseNet121", "densenet121"),
    "efficientnet_b0": ("EfficientNet-B0", "efficientnet_b0"),
    "vit_b_16": ("ViT-B/16", "vit_base_patch16_224"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all pneumonia baseline models")
    parser.add_argument("--data-dir", default="chest_xray")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY), choices=list(MODEL_REGISTRY))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2025])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--use-existing-val", action="store_true")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet pretrained weights where timm provides them",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="saved_models/baselines")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip model/seed runs already present in baseline_summary.csv with an existing checkpoint",
    )
    return parser.parse_args()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def safe_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_ppv": precision,
        "npv": npv,
        "f1": f1,
        "auc_roc": safe_auc(labels, probabilities),
        "auc_pr": safe_average_precision(labels, probabilities),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def evaluate_with_probabilities(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    probs_all: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            probabilities = torch.softmax(outputs, dim=1)[:, PNEUMONIA_INDEX]
            labels_all.extend(labels.cpu().numpy().tolist())
            probs_all.extend(probabilities.cpu().numpy().tolist())

    labels_np = np.asarray(labels_all, dtype=int)
    probs_np = np.asarray(probs_all, dtype=float)
    metrics = binary_metrics(labels_np, probs_np, threshold=0.5)
    avg_loss = total_loss / len(labels_np) if len(labels_np) else float("nan")
    return avg_loss, metrics


def train_one_run(
    args: argparse.Namespace,
    model_key: str,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    display_name, timm_model_name = MODEL_REGISTRY[model_key]
    set_seed(seed)

    loader_args = SimpleNamespace(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=seed,
        num_workers=args.num_workers,
        val_split=args.val_split,
        use_existing_val=args.use_existing_val,
    )
    train_loader, val_loader, _train_dataset, class_to_idx, split_description = build_loaders(loader_args)

    model = create_model(timm_model_name, num_classes=len(CLASS_NAMES), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{safe_filename(display_name)}_seed{seed}_best.pth"

    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    history_rows: list[dict[str, object]] = []

    print(f"\nTraining {display_name} ({timm_model_name}) | seed={seed} | device={device}")
    print(f"Validation strategy: {split_description}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"{display_name} seed={seed} epoch={epoch}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
            progress.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_accuracy = correct / max(total, 1)
        val_loss, val_metrics = evaluate_with_probabilities(model, val_loader, criterion, device)

        row = {
            "model": display_name,
            "timm_model_name": timm_model_name,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history_rows.append(row)

        score = val_metrics["balanced_accuracy"]
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_bal_acc={score:.4f} "
            f"val_sens={val_metrics['sensitivity']:.4f} val_spec={val_metrics['specificity']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            save_model_checkpoint(
                checkpoint_path,
                model,
                {
                    "model_name": timm_model_name,
                    "display_name": display_name,
                    "class_names": list(CLASS_NAMES),
                    "class_to_idx": dict(class_to_idx),
                    "image_size": list(DEFAULT_IMAGE_SIZE),
                    "mean": list(IMAGENET_MEAN),
                    "std": list(IMAGENET_STD),
                    "threshold": 0.5,
                    "seed": seed,
                    "epochs": args.epochs,
                    "best_epoch": best_epoch,
                    "validation_strategy": split_description,
                    "optimizer": "AdamW",
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "scheduler": "CosineAnnealingLR",
                    "best_val_metrics": best_metrics,
                },
            )
            print(f"Saved best checkpoint to {checkpoint_path}")

    summary = {
        "model": display_name,
        "timm_model_name": timm_model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_checkpoint": str(checkpoint_path),
        "validation_strategy": split_description,
        "pretrained": args.pretrained,
        "image_size": f"{DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]}",
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        **{f"best_val_{key}": value for key, value in best_metrics.items()},
    }
    return history_rows, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    history_path = output_dir / "baseline_training_history.csv"
    summary_path = output_dir / "baseline_summary.csv"
    if args.resume and history_path.exists():
        all_history = pd.read_csv(history_path).to_dict("records")
    else:
        all_history: list[dict[str, object]] = []

    if args.resume and summary_path.exists():
        summaries = pd.read_csv(summary_path).to_dict("records")
    else:
        summaries: list[dict[str, object]] = []

    completed_runs: set[tuple[str, int]] = set()
    for row in summaries:
        checkpoint = Path(str(row.get("best_checkpoint", "")))
        if checkpoint.exists():
            completed_runs.add((str(row.get("model")), int(row.get("seed"))))

    for model_key in args.models:
        for seed in args.seeds:
            display_name, _timm_model_name = MODEL_REGISTRY[model_key]
            if args.resume and (display_name, seed) in completed_runs:
                print(f"Skipping completed run: {display_name} seed={seed}")
                continue

            history, summary = train_one_run(args, model_key, seed, device)
            all_history.extend(history)
            summaries.append(summary)
            pd.DataFrame(all_history).to_csv(history_path, index=False)
            pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print(f"\nSaved epoch-level history to {history_path}")
    print(f"Saved baseline summary to {summary_path}")


if __name__ == "__main__":
    main()
