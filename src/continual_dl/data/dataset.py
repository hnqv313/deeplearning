"""Manifest-backed image datasets with stable sample identities."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    original_image_id: str
    filepath: str
    label: str
    label_id: int
    source_split: str
    bbox: tuple[float, float, float, float] | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManifestRecord":
        bbox = payload.get("bbox")
        if isinstance(bbox, str) and bbox:
            bbox = ast.literal_eval(bbox)
        if bbox is not None and not (isinstance(bbox, float) and pd.isna(bbox)):
            bbox = tuple(float(value) for value in bbox)
        else:
            bbox = None
        sha256 = payload.get("sha256")
        if isinstance(sha256, float) and pd.isna(sha256):
            sha256 = None
        return cls(
            sample_id=str(payload["sample_id"]),
            original_image_id=str(payload["original_image_id"]),
            filepath=str(payload["filepath"]),
            label=str(payload["label"]),
            label_id=int(payload["label_id"]),
            source_split=str(payload["source_split"]),
            bbox=bbox,
            sha256=None if sha256 is None else str(sha256),
        )


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run the dataset scripts first."
        )
    frame = pd.read_csv(manifest_path)
    required = {
        "sample_id",
        "original_image_id",
        "filepath",
        "label",
        "label_id",
        "source_split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing columns: {missing}")
    return [ManifestRecord.from_dict(row) for row in frame.to_dict(orient="records")]


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Image dataset that preserves IDs required for leakage checks and replay."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.root = Path(root) if root is not None else None

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, filepath: str) -> Path:
        path = Path(filepath)
        if path.is_absolute() or self.root is None:
            return path
        return self.root / path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        path = self._resolve_path(record.filepath)
        with Image.open(path) as image_handle:
            image = ImageOps.exif_transpose(image_handle).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision.transforms.functional import pil_to_tensor

            image = pil_to_tensor(image).float().div(255.0)
        return image, record.label_id, record.sample_id

    def subset_by_labels(self, labels: Iterable[str]) -> "ManifestImageDataset":
        selected = set(labels)
        return ManifestImageDataset(
            [record for record in self.records if record.label in selected],
            transform=self.transform,
            root=self.root,
        )
