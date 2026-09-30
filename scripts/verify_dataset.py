"""Fail fast on missing files, imbalance, label drift, and split leakage."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


EXPECTED_LABELS = {"dog": 0, "cat": 1, "car": 2, "person": 3, "building": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--expected-per-class", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    source_to_splits: dict[str, set[str]] = defaultdict(set)
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    total_rows = 0

    for manifest in args.manifests:
        if not manifest.exists():
            failures.append(f"Missing manifest: {manifest}")
            continue
        frame = pd.read_csv(manifest)
        total_rows += len(frame)
        counts = frame["label"].value_counts().to_dict()
        mapping = frame[["label", "label_id"]].drop_duplicates()
        actual_mapping = {str(row.label): int(row.label_id) for row in mapping.itertuples()}
        if actual_mapping != EXPECTED_LABELS:
            failures.append(f"Label mapping mismatch in {manifest}: {actual_mapping}")
        if args.expected_per_class is not None:
            for label in EXPECTED_LABELS:
                if counts.get(label, 0) != args.expected_per_class:
                    failures.append(
                        f"{manifest}: {label} has {counts.get(label, 0)}, "
                        f"expected {args.expected_per_class}"
                    )
        for row in frame.itertuples():
            if not Path(row.filepath).exists():
                failures.append(f"Missing image: {row.filepath}")
            split = str(row.source_split)
            source_to_splits[str(row.original_image_id)].add(split)
            if hasattr(row, "sha256") and isinstance(row.sha256, str):
                hash_to_splits[row.sha256].add(split)

    for image_id, splits in source_to_splits.items():
        if len(splits) > 1:
            failures.append(f"Original image {image_id} leaks across splits: {sorted(splits)}")
    for digest, splits in hash_to_splits.items():
        if len(splits) > 1:
            failures.append(f"Exact crop {digest[:12]} leaks across splits: {sorted(splits)}")

    if failures:
        preview = "\n".join(f"- {failure}" for failure in failures[:50])
        raise SystemExit(f"Dataset verification failed ({len(failures)} issues):\n{preview}")
    print(f"Dataset verification passed for {total_rows} samples")


if __name__ == "__main__":
    main()
