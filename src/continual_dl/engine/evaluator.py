"""Strategy-independent evaluation."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import DataLoader

from continual_dl.metrics import classification_metrics


@torch.inference_mode()
def evaluate_strategy(
    strategy,
    dataset,
    seen_class_ids: Sequence[int],
    class_names: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=strategy.device.type == "cuda",
    )
    targets: list[int] = []
    predictions: list[int] = []
    for images, batch_targets, _ in loader:
        batch_predictions = strategy.predict(images, seen_class_ids)
        targets.extend(int(value) for value in batch_targets.tolist())
        predictions.extend(int(value) for value in batch_predictions.cpu().tolist())
    return classification_metrics(targets, predictions, seen_class_ids, class_names)
