"""Download and prepare the Kermany chest X-ray dataset."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


DATASET_URL = "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-406f-bf08-fd15a451af24/file_downloaded"
EXPECTED_SHA256 = None


class DownloadProgressBar(tqdm):
    def update_to(self, block_count=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block_count * block_size - self.n)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(zip_ref: zipfile.ZipFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in zip_ref.infolist():
        resolved = (target_root / member.filename).resolve()
        if target_root != resolved and target_root not in resolved.parents:
            raise ValueError(f"Unsafe zip member path: {member.filename}")
    zip_ref.extractall(target_root)


def download_and_extract() -> None:
    zip_path = Path("chest_xray_dataset.zip")
    extract_dir = Path("chest_xray_extracted")
    target_dir = Path("chest_xray")

    if target_dir.exists():
        print(f"Dataset folder already exists: {target_dir}")
        return

    print("Downloading Kermany chest X-ray dataset. This is about 1.2 GB.")
    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading") as progress:
            urllib.request.urlretrieve(DATASET_URL, filename=zip_path, reporthook=progress.update_to)

        if EXPECTED_SHA256:
            actual_sha = sha256sum(zip_path)
            if actual_sha.lower() != EXPECTED_SHA256.lower():
                raise ValueError(f"Checksum mismatch. Expected {EXPECTED_SHA256}, got {actual_sha}")

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            safe_extract(zip_ref, extract_dir)

        source_dir = extract_dir / "chest_xray"
        if not source_dir.exists():
            raise FileNotFoundError(f"Expected extracted folder not found: {source_dir}")

        shutil.move(str(source_dir), str(target_dir))
        print(f"Dataset ready at {target_dir}")

    finally:
        if zip_path.exists():
            os.remove(zip_path)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)


if __name__ == "__main__":
    download_and_extract()
