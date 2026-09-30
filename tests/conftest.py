from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def tiny_manifests(tmp_path: Path) -> dict[str, Path]:
    labels = {"dog": 0, "cat": 1, "car": 2, "person": 3, "building": 4}
    manifests: dict[str, Path] = {}
    split_counts = {"train": 4, "val": 2, "test": 2}
    for split, count in split_counts.items():
        rows = []
        for label, label_id in labels.items():
            for index in range(count):
                image_path = tmp_path / "images" / split / label / f"{index}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                color = (
                    (label_id * 45 + index * 3) % 255,
                    (label_id * 75 + 30) % 255,
                    (label_id * 25 + 80) % 255,
                )
                Image.new("RGB", (32, 32), color).save(image_path)
                rows.append(
                    {
                        "sample_id": f"{split}_{label}_{index}",
                        "original_image_id": f"{split}_{label}_{index}",
                        "filepath": str(image_path),
                        "label": label,
                        "label_id": label_id,
                        "source_split": split,
                        "bbox": "[0, 0, 1, 1]",
                        "sha256": f"{split}-{label}-{index}",
                    }
                )
        manifest = tmp_path / f"{split}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifests[split] = manifest
    return manifests
