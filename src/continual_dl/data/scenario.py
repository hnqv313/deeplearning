"""Class-incremental experience construction from fixed dataset manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .dataset import ManifestImageDataset, ManifestRecord, read_manifest


@dataclass(frozen=True)
class Experience:
    stage_id: int
    new_class_names: tuple[str, ...]
    new_class_ids: tuple[int, ...]
    seen_class_names: tuple[str, ...]
    seen_class_ids: tuple[int, ...]
    new_train_dataset: ManifestImageDataset
    seen_train_dataset: ManifestImageDataset
    val_dataset: ManifestImageDataset
    test_dataset: ManifestImageDataset


def _filter(records: Sequence[ManifestRecord], labels: set[str]) -> list[ManifestRecord]:
    return [record for record in records if record.label in labels]


def build_experiences(
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path,
    stages: Sequence[Sequence[str]],
    class_to_id: dict[str, int],
    train_transform,
    eval_transform,
    root: str | Path | None = None,
) -> list[Experience]:
    train_records = read_manifest(train_manifest)
    val_records = read_manifest(val_manifest)
    test_records = read_manifest(test_manifest)

    experiences: list[Experience] = []
    seen_names: list[str] = []
    for stage_id, stage_names_value in enumerate(stages):
        stage_names = tuple(stage_names_value)
        unknown = sorted(set(stage_names) - set(class_to_id))
        if unknown:
            raise ValueError(f"Unknown classes in stage {stage_id}: {unknown}")
        seen_names.extend(stage_names)
        if len(seen_names) != len(set(seen_names)):
            raise ValueError(f"A class appears in more than one stage: {seen_names}")

        new_set = set(stage_names)
        seen_set = set(seen_names)
        new_train_records = _filter(train_records, new_set)
        seen_train_records = _filter(train_records, seen_set)
        seen_val_records = _filter(val_records, seen_set)
        seen_test_records = _filter(test_records, seen_set)
        if not new_train_records:
            raise ValueError(f"Stage {stage_id} has no training samples for {stage_names}")
        if not seen_val_records or not seen_test_records:
            raise ValueError(f"Stage {stage_id} has an empty validation or test set")

        experiences.append(
            Experience(
                stage_id=stage_id,
                new_class_names=stage_names,
                new_class_ids=tuple(class_to_id[name] for name in stage_names),
                seen_class_names=tuple(seen_names),
                seen_class_ids=tuple(class_to_id[name] for name in seen_names),
                new_train_dataset=ManifestImageDataset(
                    new_train_records, transform=train_transform, root=root
                ),
                seen_train_dataset=ManifestImageDataset(
                    seen_train_records, transform=train_transform, root=root
                ),
                val_dataset=ManifestImageDataset(
                    seen_val_records, transform=eval_transform, root=root
                ),
                test_dataset=ManifestImageDataset(
                    seen_test_records, transform=eval_transform, root=root
                ),
            )
        )
    return experiences
