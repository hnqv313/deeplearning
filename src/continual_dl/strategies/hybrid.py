"""Rehearsal representation learning with a prototype classification head."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from continual_dl.data.dataset import ManifestImageDataset

from .replay import ReplayStrategy


class ReplayNCMHybridStrategy(ReplayStrategy):
    """Replay keeps the representation current; prototypes remove the head's new-class bias.

    Prototypes are rebuilt at every stage from the memory buffer using the
    backbone as it stands after that stage. Accumulating features across stages
    would mix prototype spaces from different backbones and make the cosine
    comparison between classes meaningless.
    """

    name = "replay_ncm_hybrid"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prototypes: dict[int, torch.Tensor] = {}

    def after_experience(self, experience) -> None:
        super().after_experience(experience)
        self._refresh_prototypes(experience)

    def _refresh_prototypes(self, experience) -> None:
        # after_experience has already folded the new classes into the buffer.
        dataset = ManifestImageDataset(
            self.buffer,
            transform=experience.new_train_dataset.transform,
            root=experience.new_train_dataset.root,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}
        self.model.eval()
        with torch.inference_mode():
            for images, targets, _ in loader:
                images = images.to(self.device, non_blocking=True)
                features = functional.normalize(self.model.extract_features(images), dim=1).cpu()
                for class_id in targets.unique().tolist():
                    mask = targets == int(class_id)
                    class_features = features[mask]
                    class_id = int(class_id)
                    if class_id in sums:
                        sums[class_id] = sums[class_id] + class_features.sum(dim=0)
                    else:
                        sums[class_id] = class_features.sum(dim=0)
                    counts[class_id] = counts.get(class_id, 0) + int(mask.sum())
        self.prototypes = {
            class_id: functional.normalize(sums[class_id] / counts[class_id], dim=0)
            for class_id in sums
        }

    @torch.inference_mode()
    def predict(self, images: torch.Tensor, seen_class_ids: Sequence[int]) -> torch.Tensor:
        missing = [class_id for class_id in seen_class_ids if class_id not in self.prototypes]
        if missing:
            raise RuntimeError(f"Missing NCM prototypes for class IDs: {missing}")
        self.model.eval()
        features = functional.normalize(
            self.model.extract_features(images.to(self.device, non_blocking=True)), dim=1
        )
        prototypes = torch.stack(
            [self.prototypes[int(class_id)] for class_id in seen_class_ids]
        ).to(self.device)
        local_predictions = (features @ prototypes.T).argmax(dim=1)
        mapping = torch.tensor(list(seen_class_ids), device=self.device)
        return mapping[local_predictions]

    def extra_state_dict(self) -> dict:
        state = super().extra_state_dict()
        state["prototypes"] = {
            class_id: value.clone() for class_id, value in self.prototypes.items()
        }
        return state

    def load_extra_state_dict(self, state: dict) -> None:
        super().load_extra_state_dict(state)
        self.prototypes = {
            int(class_id): value.cpu() for class_id, value in state.get("prototypes", {}).items()
        }
