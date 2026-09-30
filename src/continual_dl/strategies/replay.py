"""Class-balanced fixed-capacity rehearsal replay."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from continual_dl.data.dataset import ManifestImageDataset, ManifestRecord

from .base import BaseStrategy


class ReplayStrategy(BaseStrategy):
    name = "replay"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.memory_size = int(self.strategy_config.get("memory_size", 200))
        self.replay_ratio = float(self.strategy_config.get("replay_ratio", 0.5))
        if not 0 < self.replay_ratio < 1:
            raise ValueError("replay_ratio must be between 0 and 1")
        self.buffer: list[ManifestRecord] = []

    def training_batches(self, experience) -> Iterable:
        if not self.buffer:
            yield from super().training_batches(experience)
            return
        memory_batch_size = max(1, int(round(self.batch_size * self.replay_ratio)))
        new_batch_size = max(1, self.batch_size - memory_batch_size)
        new_loader = DataLoader(
            experience.new_train_dataset,
            batch_size=new_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        memory_dataset = ManifestImageDataset(
            self.buffer,
            transform=experience.new_train_dataset.transform,
            root=experience.new_train_dataset.root,
        )
        memory_loader = DataLoader(
            memory_dataset,
            batch_size=memory_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        memory_iterator = iter(memory_loader)
        for new_images, new_targets, new_ids in new_loader:
            try:
                memory_images, memory_targets, memory_ids = next(memory_iterator)
            except StopIteration:
                memory_iterator = iter(memory_loader)
                memory_images, memory_targets, memory_ids = next(memory_iterator)
            yield (
                torch.cat([new_images, memory_images], dim=0),
                torch.cat([new_targets, memory_targets], dim=0),
                list(new_ids) + list(memory_ids),
            )

    def after_experience(self, experience) -> None:
        candidates: dict[int, list[ManifestRecord]] = defaultdict(list)
        for record in self.buffer:
            candidates[record.label_id].append(record)
        for record in experience.new_train_dataset.records:
            candidates[record.label_id].append(record)

        seen_ids = sorted(int(value) for value in experience.seen_class_ids)
        base_quota, remainder = divmod(self.memory_size, len(seen_ids))
        rng = random.Random(int(self.training_config.get("seed", 42)) + experience.stage_id)
        updated: list[ManifestRecord] = []
        for position, class_id in enumerate(seen_ids):
            class_candidates = list(candidates[class_id])
            rng.shuffle(class_candidates)
            quota = base_quota + (1 if position < remainder else 0)
            updated.extend(class_candidates[:quota])
        self.buffer = updated

    def extra_state_dict(self) -> dict:
        return {"buffer": [record.to_dict() for record in self.buffer]}

    def load_extra_state_dict(self, state: dict) -> None:
        self.buffer = [ManifestRecord.from_dict(item) for item in state.get("buffer", [])]
