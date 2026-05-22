"""Legacy utility for splitting PNEUMONIA into BACTERIA and VIRUS folders.

The main project is binary (NORMAL vs PNEUMONIA). This script is kept only for
experiments that intentionally need the older three-class dataset layout. It is
dry-run by default because it moves files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split PNEUMONIA images into BACTERIA/VIRUS folders")
    parser.add_argument("--base-dir", default="chest_xray")
    parser.add_argument("--apply", action="store_true", help="Actually move files. Without this, only report counts.")
    return parser.parse_args()


def restructure_dataset(base_dir: str, apply_changes: bool = False) -> None:
    base = Path(base_dir)
    for split in ["train", "test", "val"]:
        pneumonia_path = base / split / "PNEUMONIA"
        if not pneumonia_path.exists():
            print(f"{split}: no PNEUMONIA folder found")
            continue

        bacteria_files = []
        virus_files = []
        for image_path in pneumonia_path.glob("*.jpeg"):
            name = image_path.name.lower()
            if "bacteria" in name:
                bacteria_files.append(image_path)
            elif "virus" in name:
                virus_files.append(image_path)

        print(f"{split}: bacteria={len(bacteria_files)} virus={len(virus_files)}")
        if not apply_changes:
            continue

        bacteria_path = base / split / "BACTERIA"
        virus_path = base / split / "VIRUS"
        bacteria_path.mkdir(parents=True, exist_ok=True)
        virus_path.mkdir(parents=True, exist_ok=True)

        for image_path in bacteria_files:
            shutil.move(str(image_path), bacteria_path / image_path.name)
        for image_path in virus_files:
            shutil.move(str(image_path), virus_path / image_path.name)

        if not any(pneumonia_path.iterdir()):
            pneumonia_path.rmdir()


if __name__ == "__main__":
    args = parse_args()
    restructure_dataset(args.base_dir, apply_changes=args.apply)
