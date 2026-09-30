"""Download an Open Images V7 subset and export portable raw JSONL manifests.

This script intentionally keeps downloading separate from cropping so the raw
selection can be audited and reprocessed without another network transfer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any


TARGET_ALIASES = {
    "Dog": {"Dog"},
    "Cat": {"Cat"},
    "Car": {"Car"},
    "Person": {"Person", "Man", "Woman", "Boy", "Girl"},
    "Building": {
        "Building",
        "House",
        "Office building",
        "Skyscraper",
        "Tower",
        "Castle",
    },
}


def _attribute(detection: Any, *names: str) -> Any:
    field_names = set(getattr(detection, "field_names", ()))
    for name in names:
        if name in field_names:
            return detection[name]
        if hasattr(detection, name):
            return getattr(detection, name)
    return None


def _find_detection_field(sample: Any, fiftyone_module: Any) -> str:
    for field_name in sample.field_names:
        try:
            value = sample[field_name]
        except Exception:
            continue
        if isinstance(value, fiftyone_module.Detections):
            return field_name
    raise RuntimeError(f"No detection field found in sample {sample.id}")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def seed_shared_metadata(dataset_zoo_dir: Path, split: str) -> None:
    """Reuse version-wide Open Images metadata already cached by another split."""
    dataset_root = dataset_zoo_dir / "open-images-v7"
    target_dir = dataset_root / split / "metadata"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("classes.csv", "hierarchy.json"):
        target = target_dir / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        for candidate in dataset_root.glob(f"*/metadata/{filename}"):
            if candidate != target and candidate.stat().st_size > 0:
                shutil.copy2(candidate, target)
                print(f"Reused shared metadata {candidate} -> {target}")
                break


def load_with_retries(loader: Any, retries: int, **kwargs: Any) -> Any:
    """Retry transient remote resets while preserving FiftyOne's local cache."""
    for attempt in range(1, retries + 1):
        try:
            return loader("open-images-v7", **kwargs)
        except Exception as error:
            if attempt == retries:
                raise
            delay = min(15, 2**attempt)
            print(
                f"Download attempt {attempt}/{retries} failed with "
                f"{type(error).__name__}: {error}. Retrying in {delay}s..."
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def dataset_records(
    dataset: Any,
    split: str,
    fiftyone_module: Any,
    allowed_labels: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    detection_field: str | None = None
    for sample in dataset.iter_samples(progress=True):
        if detection_field is None:
            detection_field = _find_detection_field(sample, fiftyone_module)
        detections = sample[detection_field]
        image_path = Path(sample.filepath).resolve()
        serialized = []
        for detection in detections.detections:
            if detection.label not in allowed_labels:
                continue
            serialized.append(
                {
                    "label": detection.label,
                    "bounding_box": [float(value) for value in detection.bounding_box],
                    "confidence": detection.confidence,
                    "is_occluded": _attribute(detection, "IsOccluded", "is_occluded"),
                    "is_truncated": _attribute(detection, "IsTruncated", "is_truncated"),
                    "is_group_of": _attribute(detection, "IsGroupOf", "is_group_of"),
                    "is_depiction": _attribute(detection, "IsDepiction", "is_depiction"),
                    "is_inside": _attribute(detection, "IsInside", "is_inside"),
                }
            )
        if serialized:
            records.append(
                {
                    "image_id": image_path.stem,
                    "filepath": str(image_path),
                    "source_split": split,
                    "detections": serialized,
                }
            )
    return records


def merge_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge repeated source images produced by per-class downloads."""
    merged: dict[str, dict[str, Any]] = {}
    detection_keys: dict[str, set[tuple[str, tuple[float, ...]]]] = {}
    for record in records:
        image_id = str(record["image_id"])
        if image_id not in merged:
            merged[image_id] = {**record, "detections": []}
            detection_keys[image_id] = set()
        for detection in record["detections"]:
            key = (str(detection["label"]), tuple(detection["bounding_box"]))
            if key not in detection_keys[image_id]:
                merged[image_id]["detections"].append(detection)
                detection_keys[image_id].add(key)
    return list(merged.values())


def write_records(records: list[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument(
        "--classes", nargs="+", default=["Dog", "Cat", "Car", "Person", "Building"]
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1200,
        help=(
            "Maximum source images per class and split in balanced mode; maximum total "
            "source images per split in mixed mode"
        ),
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["balanced", "mixed"],
        default="balanced",
        help="Balanced downloads each class separately; mixed may heavily under-sample Cat",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/manifests"))
    parser.add_argument("--dataset-zoo-dir", type=Path, default=Path("data/fiftyone"))
    parser.add_argument("--database-dir", type=Path, default=Path("data/fiftyone-db"))
    parser.add_argument(
        "--download-retries",
        type=int,
        default=4,
        help="Retries for transient Open Images connection failures",
    )
    parser.add_argument(
        "--allow-large-train-metadata",
        action="store_true",
        help="Allow the official train split, which requires parsing metadata for ~9M images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "train" in args.splits and not args.allow_large_train_metadata:
        raise SystemExit(
            "The Open Images train split requires a very large metadata table and can exceed "
            "RAM on student machines. Use validation+test and partition the resulting pool, or "
            "pass --allow-large-train-metadata on a high-memory machine."
        )
    os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(args.dataset_zoo_dir.resolve())
    os.environ["FIFTYONE_DATABASE_DIR"] = str(args.database_dir.resolve())
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as error:
        raise SystemExit(
            "FiftyOne is required for downloading. Use Python 3.10-3.13 and run "
            "`pip install -r requirements-data.txt`."
        ) from error

    for split in args.splits:
        seed_shared_metadata(args.dataset_zoo_dir, split)
        records: list[dict[str, Any]] = []
        class_groups = [[label] for label in args.classes]
        if args.sampling_mode == "mixed":
            class_groups = [list(args.classes)]

        for class_group in class_groups:
            group_slug = "-".join(_slug(label) for label in class_group)
            dataset_name = (
                f"continual-open-images-v7-{split}-{group_slug}-"
                f"{args.seed}-{args.max_samples}"
            )
            print(
                f"Loading split={split}, classes={class_group}, "
                f"max_samples={args.max_samples}"
            )
            dataset = load_with_retries(
                foz.load_zoo_dataset,
                retries=args.download_retries,
                split=split,
                label_types=["detections"],
                classes=class_group,
                max_samples=args.max_samples,
                shuffle=True,
                seed=args.seed,
                dataset_name=dataset_name,
            )
            allowed_labels: set[str] = set()
            for label in class_group:
                allowed_labels.update(TARGET_ALIASES.get(label, {label}))
            records.extend(dataset_records(dataset, split, fo, allowed_labels))

        output_path = args.output_dir / f"{split}.jsonl"
        count = write_records(merge_records(records), output_path)
        print(f"Exported {count} source images to {output_path}")


if __name__ == "__main__":
    main()
