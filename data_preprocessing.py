"""Dataset loader helpers for the binary chest X-ray dataset."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import CLASS_NAMES, get_eval_transform, get_train_transform


def build_imagefolder_loaders(
    data_dir: str = "chest_xray",
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[str, ...]]:
    root = Path(data_dir)
    train_dataset = datasets.ImageFolder(root / "train", transform=get_train_transform())
    val_dataset = datasets.ImageFolder(root / "val", transform=get_eval_transform())
    test_dataset = datasets.ImageFolder(root / "test", transform=get_eval_transform())

    class_names = tuple(sorted(train_dataset.class_to_idx, key=train_dataset.class_to_idx.get))
    if class_names != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {class_names}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = build_imagefolder_loaders()
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")
