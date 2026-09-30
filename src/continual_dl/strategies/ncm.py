"""Frozen feature extractor with streaming nearest-class-mean prototypes."""

from __future__ import annotations

import time
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .base import BaseStrategy


class NCMStrategy(BaseStrategy):
    name = "ncm"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.feature_sums: dict[int, torch.Tensor] = {}
        self.class_counts: dict[int, int] = {}

    def fit(self, experience) -> dict:
        started = time.perf_counter()
        loader = DataLoader(
            experience.new_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.model.eval()
        processed = 0
        with torch.inference_mode():
            for images, targets, _ in loader:
                images = images.to(self.device, non_blocking=True)
                features = functional.normalize(self.model.extract_features(images), dim=1).cpu()
                for class_id in targets.unique().tolist():
                    mask = targets == int(class_id)
                    class_features = features[mask]
                    feature_sum = class_features.sum(dim=0)
                    class_id = int(class_id)
                    if class_id in self.feature_sums:
                        self.feature_sums[class_id] += feature_sum
                    else:
                        self.feature_sums[class_id] = feature_sum
                    self.class_counts[class_id] = self.class_counts.get(class_id, 0) + int(mask.sum())
                    processed += int(mask.sum())
        return {
            "epochs": 0,
            "optimizer_steps": 0,
            "processed_images": processed,
            "training_seconds": time.perf_counter() - started,
        }

    def _prototype(self, class_id: int) -> torch.Tensor:
        mean = self.feature_sums[class_id] / self.class_counts[class_id]
        return functional.normalize(mean, dim=0)

    @torch.inference_mode()
    def predict(self, images: torch.Tensor, seen_class_ids: Sequence[int]) -> torch.Tensor:
        missing = [class_id for class_id in seen_class_ids if class_id not in self.feature_sums]
        if missing:
            raise RuntimeError(f"Missing NCM prototypes for class IDs: {missing}")
        self.model.eval()
        features = functional.normalize(
            self.model.extract_features(images.to(self.device, non_blocking=True)), dim=1
        )
        prototypes = torch.stack(
            [self._prototype(int(class_id)) for class_id in seen_class_ids]
        ).to(self.device)
        local_predictions = (features @ prototypes.T).argmax(dim=1)
        mapping = torch.tensor(list(seen_class_ids), device=self.device)
        return mapping[local_predictions]

    def extra_state_dict(self) -> dict:
        return {
            "feature_sums": self.feature_sums,
            "class_counts": self.class_counts,
        }

    def load_extra_state_dict(self, state: dict) -> None:
        self.feature_sums = {
            int(class_id): value.cpu() for class_id, value in state.get("feature_sums", {}).items()
        }
        self.class_counts = {
            int(class_id): int(value) for class_id, value in state.get("class_counts", {}).items()
        }
