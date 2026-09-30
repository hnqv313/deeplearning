"""Create dataset statistics and visual contact sheets for manual QA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def make_contact_sheet(frame: pd.DataFrame, output: Path, seed: int, count: int = 25) -> None:
    rng = random.Random(seed)
    indices = list(frame.index)
    rng.shuffle(indices)
    selected = indices[: min(count, len(indices))]
    tile_size = 160
    grid_size = 5
    canvas = Image.new("RGB", (tile_size * grid_size, tile_size * grid_size), "white")
    draw = ImageDraw.Draw(canvas)
    for position, index in enumerate(selected):
        row = frame.loc[index]
        with Image.open(row["filepath"]) as handle:
            image = ImageOps.fit(ImageOps.exif_transpose(handle).convert("RGB"), (tile_size, tile_size))
        x = (position % grid_size) * tile_size
        y = (position // grid_size) * tile_size
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + tile_size, y + 20), fill=(0, 0, 0))
        draw.text((x + 4, y + 3), str(row["label"]), fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    for manifest in args.manifests:
        frame = pd.read_csv(manifest)
        split_name = manifest.stem
        report[split_name] = {
            "rows": len(frame),
            "class_counts": frame["label"].value_counts().sort_index().to_dict(),
            "crop_width": frame["crop_width"].describe().to_dict()
            if "crop_width" in frame
            else None,
            "crop_height": frame["crop_height"].describe().to_dict()
            if "crop_height" in frame
            else None,
        }
        for label, class_frame in frame.groupby("label"):
            make_contact_sheet(
                class_frame,
                args.output_dir / f"{split_name}_{label}.jpg",
                seed=args.seed,
            )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
