"""Detect perceptual duplicates across manifests and optionally remove them."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def difference_hash(path: Path) -> str:
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("L").resize((9, 8))
        # Pillow 12 renamed ``getdata`` to ``get_flattened_data``. Keep the
        # project compatible with the declared Pillow >= 10 requirement.
        get_pixels = getattr(image, "get_flattened_data", image.getdata)
        pixels = list(get_pixels())
    bits = []
    for row in range(8):
        offset = row * 9
        bits.extend(pixels[offset + column] > pixels[offset + column + 1] for column in range(8))
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true", help="Rewrite manifests after removing duplicates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames: dict[Path, pd.DataFrame] = {}
    groups: dict[str, list[tuple[Path, int, str]]] = defaultdict(list)
    for manifest in args.manifests:
        frame = pd.read_csv(manifest)
        frames[manifest] = frame
        for index, row in frame.iterrows():
            fingerprint = difference_hash(Path(row["filepath"]))
            groups[fingerprint].append((manifest, int(index), str(row["sample_id"])))

    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    print(f"Found {len(duplicate_groups)} perceptual duplicate groups")
    for items in duplicate_groups[:20]:
        print("  " + " | ".join(sample_id for _, _, sample_id in items))

    if not args.apply:
        return

    # Keep test before validation before train so evaluation sets are not silently weakened.
    priority = {"test": 0, "val": 1, "validation": 1, "train": 2}
    drops: dict[Path, set[int]] = defaultdict(set)
    for items in duplicate_groups:
        ranked = sorted(
            items,
            key=lambda item: priority.get(str(frames[item[0]].iloc[item[1]]["source_split"]), 99),
        )
        for manifest, index, _ in ranked[1:]:
            drops[manifest].add(index)
    for manifest, frame in frames.items():
        cleaned = frame.drop(index=sorted(drops.get(manifest, set()))).reset_index(drop=True)
        cleaned.to_csv(manifest, index=False)
        print(f"Rewrote {manifest}: {len(frame)} -> {len(cleaned)} rows")


if __name__ == "__main__":
    main()
