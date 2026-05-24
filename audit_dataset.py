"""Audit the Kermany chest X-ray dataset for publication-grade reporting.

The script reports split/class counts, patient-level distributions, duplicate
filenames, exact duplicate images by SHA-256, and near-duplicate candidates by a
simple perceptual average hash.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from model_utils import CLASS_NAMES


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    split: str
    class_name: str
    path: Path
    patient_id: str
    sha256: str
    average_hash: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit chest X-ray dataset integrity and distribution")
    parser.add_argument("--data-dir", default="chest_xray", help="Dataset root containing train/val/test folders")
    parser.add_argument("--metadata-csv", default=None, help="Optional metadata CSV to join by image filename")
    parser.add_argument("--image-column", default="filename", help="Image filename column in metadata CSV")
    parser.add_argument("--patient-column", default=None, help="Optional patient id column in metadata CSV")
    parser.add_argument("--output-dir", default="results", help="Directory for audit CSV outputs")
    parser.add_argument("--hash-size", type=int, default=8, help="Average-hash width/height")
    parser.add_argument(
        "--near-duplicate-hamming",
        type=int,
        default=4,
        help="Maximum average-hash Hamming distance for near-duplicate candidates",
    )
    parser.add_argument(
        "--max-near-duplicate-pairs",
        type=int,
        default=1000,
        help="Stop near-duplicate reporting after this many candidate pairs",
    )
    return parser.parse_args()


def patient_group_id(path: str | Path) -> str:
    name = Path(path).name
    match = re.search(r"(person\d+|NORMAL2-IM-\d+|IM-\d+)", name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return Path(name).stem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, hash_size: int) -> int | None:
    try:
        with Image.open(path) as image:
            image = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        logging.warning("Could not hash image %s: %s", path, exc)
        return None

    threshold = float(pixels.mean())
    bits = (pixels >= threshold).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def iter_image_paths(data_dir: Path) -> list[tuple[str, str, Path]]:
    paths: list[tuple[str, str, Path]] = []
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            folder = data_dir / split / class_name
            if not folder.exists():
                logging.warning("Missing expected folder: %s", folder)
                continue
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append((split, class_name, path))
    return paths


def build_records(data_dir: Path, hash_size: int) -> list[ImageRecord]:
    image_paths = iter_image_paths(data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    records: list[ImageRecord] = []
    for split, class_name, path in image_paths:
        records.append(
            ImageRecord(
                split=split,
                class_name=class_name,
                path=path,
                patient_id=patient_group_id(path),
                sha256=sha256_file(path),
                average_hash=average_hash(path, hash_size),
            )
        )
    return records


def class_count_table(records: list[ImageRecord]) -> pd.DataFrame:
    rows = []
    counter = Counter((record.split, record.class_name) for record in records)
    for split in SPLITS:
        total = sum(counter[(split, class_name)] for class_name in CLASS_NAMES)
        normal = counter[(split, "NORMAL")]
        pneumonia = counter[(split, "PNEUMONIA")]
        rows.append(
            {
                "split": split,
                "NORMAL": normal,
                "PNEUMONIA": pneumonia,
                "total": total,
                "pneumonia_fraction": pneumonia / total if total else np.nan,
                "pneumonia_to_normal_ratio": pneumonia / normal if normal else np.nan,
            }
        )
    return pd.DataFrame(rows)


def patient_distribution_table(records: list[ImageRecord]) -> pd.DataFrame:
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        grouped[(record.split, record.class_name)][record.patient_id] += 1

    rows = []
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            counts = list(grouped[(split, class_name)].values())
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "unique_patients": len(counts),
                    "images": int(sum(counts)),
                    "images_per_patient_mean": float(np.mean(counts)) if counts else np.nan,
                    "images_per_patient_median": float(np.median(counts)) if counts else np.nan,
                    "images_per_patient_min": int(np.min(counts)) if counts else 0,
                    "images_per_patient_max": int(np.max(counts)) if counts else 0,
                }
            )
    return pd.DataFrame(rows)


def patient_split_overlap_table(records: list[ImageRecord]) -> pd.DataFrame:
    split_lookup: dict[str, set[str]] = defaultdict(set)
    class_lookup: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split_lookup[record.patient_id].add(record.split)
        class_lookup[record.patient_id].add(record.class_name)

    rows = []
    for patient_id in sorted(split_lookup):
        if len(split_lookup[patient_id]) > 1 or len(class_lookup[patient_id]) > 1:
            rows.append(
                {
                    "patient_id": patient_id,
                    "splits": ";".join(sorted(split_lookup[patient_id])),
                    "classes": ";".join(sorted(class_lookup[patient_id])),
                }
            )
    return pd.DataFrame(rows)


def duplicate_filename_table(records: list[ImageRecord]) -> pd.DataFrame:
    by_name: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_name[record.path.name.lower()].append(record)

    rows = []
    for filename, group in sorted(by_name.items()):
        if len(group) <= 1:
            continue
        rows.append(
            {
                "filename": filename,
                "count": len(group),
                "locations": ";".join(str(record.path) for record in group),
                "splits": ";".join(sorted({record.split for record in group})),
                "classes": ";".join(sorted({record.class_name for record in group})),
            }
        )
    return pd.DataFrame(rows)


def exact_duplicate_table(records: list[ImageRecord]) -> pd.DataFrame:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)

    rows = []
    for digest, group in sorted(by_hash.items()):
        if len(group) <= 1:
            continue
        rows.append(
            {
                "sha256": digest,
                "count": len(group),
                "locations": ";".join(str(record.path) for record in group),
                "splits": ";".join(sorted({record.split for record in group})),
                "classes": ";".join(sorted({record.class_name for record in group})),
            }
        )
    return pd.DataFrame(rows)


def near_duplicate_table(
    records: list[ImageRecord],
    max_distance: int,
    max_pairs: int,
) -> pd.DataFrame:
    hashable_records = [record for record in records if record.average_hash is not None]
    rows = []
    for index, left in enumerate(hashable_records):
        for right in hashable_records[index + 1 :]:
            distance = int((left.average_hash ^ right.average_hash).bit_count())
            if distance <= max_distance:
                rows.append(
                    {
                        "hamming_distance": distance,
                        "left_path": str(left.path),
                        "left_split": left.split,
                        "left_class": left.class_name,
                        "right_path": str(right.path),
                        "right_split": right.split,
                        "right_class": right.class_name,
                        "same_sha256": left.sha256 == right.sha256,
                    }
                )
                if len(rows) >= max_pairs:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def metadata_match_table(
    records: list[ImageRecord],
    metadata_csv: str | None,
    image_column: str,
    patient_column: str | None,
) -> pd.DataFrame:
    if not metadata_csv:
        return pd.DataFrame()

    metadata = pd.read_csv(metadata_csv)
    if image_column not in metadata.columns:
        raise ValueError(f"Metadata CSV is missing image column: {image_column}")

    metadata = metadata.copy()
    metadata["_basename"] = metadata[image_column].map(lambda value: Path(str(value)).name.lower())
    metadata_lookup = metadata.set_index("_basename", drop=False)
    rows = []
    for record in records:
        basename = record.path.name.lower()
        matched = basename in metadata_lookup.index
        metadata_patient = None
        if matched and patient_column and patient_column in metadata.columns:
            match = metadata_lookup.loc[basename]
            if isinstance(match, pd.DataFrame):
                metadata_patient = str(match.iloc[0][patient_column])
            else:
                metadata_patient = str(match[patient_column])
        rows.append(
            {
                "path": str(record.path),
                "split": record.split,
                "class_name": record.class_name,
                "parsed_patient_id": record.patient_id,
                "metadata_matched": matched,
                "metadata_patient_id": metadata_patient,
            }
        )
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info("Saved %s", path)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(data_dir, args.hash_size)
    counts = class_count_table(records)
    patient_distribution = patient_distribution_table(records)
    patient_overlap = patient_split_overlap_table(records)
    duplicate_filenames = duplicate_filename_table(records)
    exact_duplicates = exact_duplicate_table(records)
    near_duplicates = near_duplicate_table(
        records,
        max_distance=args.near_duplicate_hamming,
        max_pairs=args.max_near_duplicate_pairs,
    )
    metadata_matches = metadata_match_table(records, args.metadata_csv, args.image_column, args.patient_column)

    save_table(counts, output_dir / "audit_class_counts.csv")
    save_table(patient_distribution, output_dir / "audit_patient_distribution.csv")
    save_table(patient_overlap, output_dir / "audit_patient_split_overlap.csv")
    save_table(duplicate_filenames, output_dir / "audit_duplicate_filenames.csv")
    save_table(exact_duplicates, output_dir / "audit_exact_duplicate_hashes.csv")
    save_table(near_duplicates, output_dir / "audit_near_duplicate_hashes.csv")
    if not metadata_matches.empty:
        save_table(metadata_matches, output_dir / "audit_metadata_matches.csv")

    print("\nClass counts and imbalance:")
    print(counts.to_string(index=False))
    print("\nPatient-level distribution:")
    print(patient_distribution.to_string(index=False))
    print(f"\nDuplicate filenames: {len(duplicate_filenames)} group(s)")
    print(f"Exact duplicate image hashes: {len(exact_duplicates)} group(s)")
    print(
        "Near-duplicate image candidates: "
        f"{len(near_duplicates)} pair(s) at Hamming <= {args.near_duplicate_hamming}"
    )
    print(f"Patient ids crossing split/class boundaries: {len(patient_overlap)}")
    if not metadata_matches.empty:
        matched = int(metadata_matches["metadata_matched"].sum())
        print(f"Metadata matches: {matched}/{len(metadata_matches)} image(s)")


if __name__ == "__main__":
    main()
