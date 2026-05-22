"""Train the binary ViT pneumonia screening model.

The test split is intentionally never used for model selection. By default this
script creates a grouped validation split from the training folder so repeated
images from the same Kermany patient id stay on the same side of the split.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets
from tqdm import tqdm

from model_utils import (
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_NAME,
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_model,
    get_device,
    get_eval_transform,
    get_train_transform,
    save_model_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary pneumonia ViT model")
    parser.add_argument("--data-dir", default="chest_xray", help="Dataset root with train/val/test folders")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.15, help="Grouped validation fraction from train/")
    parser.add_argument(
        "--use-existing-val",
        action="store_true",
        help="Use data-dir/val instead of creating a grouped split from train/",
    )
    parser.add_argument("--output-path", default="saved_models/pneumonia_binary_best.pth")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def patient_group_id(path: str) -> str:
    name = Path(path).name
    match = re.search(r"(person\d+|NORMAL2-IM-\d+|IM-\d+)", name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return Path(name).stem


def grouped_stratified_split(
    samples: list[tuple[str, int]],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0 < val_fraction < 0.5:
        raise ValueError("--val-split must be between 0 and 0.5")

    grouped: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (path, label) in enumerate(samples):
        grouped[label][patient_group_id(path)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label, groups_by_id in grouped.items():
        groups = list(groups_by_id.values())
        if len(groups) < 2:
            raise ValueError(f"Class {label} has fewer than two patient groups; cannot split safely.")

        rng.shuffle(groups)
        target_val_samples = max(1, round(sum(len(group) for group in groups) * val_fraction))
        target_val_groups = max(1, round(len(groups) * val_fraction))
        selected = 0

        for group_index, group in enumerate(groups):
            use_for_val = group_index < target_val_groups or selected < target_val_samples
            if use_for_val and len(groups) - group_index > 1:
                val_indices.extend(group)
                selected += len(group)
            else:
                train_indices.extend(group)

    if not train_indices or not val_indices:
        raise ValueError("Validation split produced an empty train or validation set.")

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def dataset_targets(dataset: datasets.ImageFolder | Subset) -> np.ndarray:
    if isinstance(dataset, Subset):
        return np.array([dataset.dataset.targets[index] for index in dataset.indices])
    return np.array(dataset.targets)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, datasets.ImageFolder, dict[str, int], str]:
    data_dir = Path(args.data_dir)
    train_root = data_dir / "train"
    val_root = data_dir / "val"

    if not train_root.exists():
        raise FileNotFoundError(f"Training folder not found: {train_root}")

    generator = torch.Generator().manual_seed(args.seed)

    if args.use_existing_val:
        if not val_root.exists():
            raise FileNotFoundError(f"Validation folder not found: {val_root}")
        train_dataset = datasets.ImageFolder(train_root, transform=get_train_transform())
        val_dataset = datasets.ImageFolder(val_root, transform=get_eval_transform())
        split_description = f"existing validation folder: {val_root}"
    else:
        train_base = datasets.ImageFolder(train_root, transform=get_train_transform())
        val_base = datasets.ImageFolder(train_root, transform=get_eval_transform())
        train_indices, val_indices = grouped_stratified_split(train_base.samples, args.val_split, args.seed)
        train_dataset = Subset(train_base, train_indices)
        val_dataset = Subset(val_base, val_indices)
        split_description = f"grouped split from train/ with val_split={args.val_split:.2f}"

    class_to_idx = train_dataset.dataset.class_to_idx if isinstance(train_dataset, Subset) else train_dataset.class_to_idx
    class_names = tuple(sorted(class_to_idx, key=class_to_idx.get))
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {class_names}")

    targets = dataset_targets(train_dataset)
    class_counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    if np.any(class_counts == 0):
        raise ValueError(f"All classes must be present in training data. Counts: {class_counts.tolist()}")

    class_weights = 1.0 / class_counts
    sample_weights = torch.as_tensor(class_weights[targets], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_dataset, class_to_idx, split_description


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    y_true = np.array(labels)
    y_pred = np.array(predictions)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    preds_all: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            labels_all.extend(labels.cpu().numpy().tolist())
            preds_all.extend(preds.cpu().numpy().tolist())

    metrics = binary_metrics(labels_all, preds_all)
    avg_loss = total_loss / len(labels_all)
    return avg_loss, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(
        f"Training {DEFAULT_MODEL_NAME} on {device} | "
        f"batch={args.batch_size} lr={args.lr} epochs={args.epochs}"
    )

    train_loader, val_loader, train_dataset, class_to_idx, split_description = build_loaders(args)
    print(f"Validation strategy: {split_description}")
    print(f"Class mapping: {class_to_idx}")

    model = create_model(DEFAULT_MODEL_NAME, num_classes=len(CLASS_NAMES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_score = -1.0
    best_metadata: dict[str, object] = {}

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

        scheduler.step()
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_sens={val_metrics['sensitivity']:.4f} "
            f"val_spec={val_metrics['specificity']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
        )

        score = val_metrics["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_metadata = {
                "model_name": DEFAULT_MODEL_NAME,
                "class_names": list(CLASS_NAMES),
                "class_to_idx": dict(class_to_idx),
                "image_size": list(DEFAULT_IMAGE_SIZE),
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "threshold": 0.5,
                "seed": args.seed,
                "epochs": args.epochs,
                "best_epoch": epoch + 1,
                "validation_strategy": split_description,
                "best_val_metrics": val_metrics,
            }
            save_model_checkpoint(args.output_path, model, best_metadata)
            print(f"Saved best checkpoint to {args.output_path}")

    print(f"Best validation balanced accuracy: {best_score:.4f}")
    if best_metadata:
        print(f"Best epoch: {best_metadata['best_epoch']}")


if __name__ == "__main__":
    main()
