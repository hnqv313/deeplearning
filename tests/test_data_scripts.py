from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.build_crops import build_split


def test_build_crops_creates_balanced_manifest(tmp_path: Path):
    source_labels = ["Dog", "Cat", "Car", "Person", "Building"]
    raw_manifest = tmp_path / "raw.jsonl"
    with raw_manifest.open("w", encoding="utf-8") as handle:
        for index, label in enumerate(source_labels):
            image_path = tmp_path / "source" / f"image_{index}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (160, 160), (index * 30, 50, 100)).save(image_path)
            record = {
                "image_id": f"image_{index}",
                "filepath": str(image_path),
                "source_split": "train",
                "detections": [
                    {
                        "label": label,
                        "bounding_box": [0.1, 0.1, 0.8, 0.8],
                        "is_group_of": False,
                        "is_depiction": False,
                        "is_inside": False,
                    }
                ],
            }
            handle.write(json.dumps(record) + "\n")

    output_manifest = tmp_path / "train.csv"
    build_split(
        raw_manifests=[raw_manifest],
        output_root=tmp_path / "processed",
        output_manifest=output_manifest,
        target_per_class=1,
        seed=42,
        padding=0.08,
        minimum_side=80,
        minimum_area_ratio=0.02,
    )
    frame = pd.read_csv(output_manifest)
    assert len(frame) == 5
    assert frame["label"].value_counts().to_dict() == {
        "dog": 1,
        "cat": 1,
        "car": 1,
        "person": 1,
        "building": 1,
    }
    assert all(Path(path).exists() for path in frame["filepath"])
