"""Partition a balanced crop pool into leakage-free train/val/test manifests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--val-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.pool_manifest)
    required = args.train_per_class + args.val_per_class + args.test_per_class
    split_rows: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for label, class_frame in frame.groupby("label", sort=True):
        class_frame = class_frame.copy()
        indices = list(class_frame.index)
        random.Random(args.seed + int(class_frame.iloc[0]["label_id"])).shuffle(indices)
        class_frame = class_frame.loc[indices].reset_index(drop=True)
        if len(class_frame) < required:
            raise SystemExit(
                f"Class {label} has {len(class_frame)} pool samples, but {required} are required"
            )
        boundaries = {
            "train": (0, args.train_per_class),
            "val": (args.train_per_class, args.train_per_class + args.val_per_class),
            "test": (
                args.train_per_class + args.val_per_class,
                required,
            ),
        }
        for split, (start, stop) in boundaries.items():
            selected = class_frame.iloc[start:stop].copy()
            selected["source_split"] = split
            split_rows[split].append(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, frames in split_rows.items():
        output = pd.concat(frames, ignore_index=True).sort_values(
            ["label_id", "sample_id"]
        )
        output_path = args.output_dir / f"{split}.csv"
        output.to_csv(output_path, index=False)
        print(f"Wrote {len(output)} rows to {output_path}")


if __name__ == "__main__":
    main()
