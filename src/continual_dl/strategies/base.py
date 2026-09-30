"""Shared optimization loop and strategy contract."""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader


class BaseStrategy:
    name = "base"

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        training_config: dict,
        strategy_config: dict,
        num_workers: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.training_config = training_config
        self.strategy_config = strategy_config
        self.batch_size = int(training_config.get("batch_size", 32))
        self.num_workers = int(num_workers)
        self.use_amp = bool(training_config.get("amp", True)) and device.type == "cuda"
        self.optimizer: torch.optim.Optimizer | None = None
        self.cross_entropy = nn.CrossEntropyLoss()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        name = str(self.training_config.get("optimizer", "adamw")).lower()
        learning_rate = float(self.training_config.get("learning_rate", 1e-4))
        weight_decay = float(self.training_config.get("weight_decay", 0.0))
        if name == "adamw":
            return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        if name == "sgd":
            return torch.optim.SGD(
                parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
            )
        raise ValueError(f"Unsupported optimizer: {name}")

    def training_dataset(self, experience):
        return experience.new_train_dataset

    def training_batches(self, experience) -> Iterable:
        return DataLoader(
            self.training_dataset(experience),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def before_experience(self, experience) -> None:
        pass

    def after_experience(self, experience) -> None:
        pass

    def compute_loss(
        self, images: torch.Tensor, targets: torch.Tensor, experience
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.model(images, experience.seen_class_ids)
        loss = self.cross_entropy(logits, targets)
        return loss, {"classification_loss": float(loss.detach())}

    def fit(self, experience) -> dict:
        self.before_experience(experience)
        reset = bool(self.training_config.get("reset_optimizer_each_stage", True))
        if self.optimizer is None or reset:
            self.optimizer = self._make_optimizer()
        epochs = int(
            self.training_config.get(
                "epochs_stage0" if experience.stage_id == 0 else "epochs_incremental", 1
            )
        )
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        started = time.perf_counter()
        total_loss = 0.0
        total_steps = 0

        for _ in range(epochs):
            self.model.train()
            for images, targets, _ in self.training_batches(experience):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss, _ = self.compute_loss(images, targets, experience)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                total_loss += float(loss.detach())
                total_steps += 1

        self.after_experience(experience)
        return {
            "epochs": epochs,
            "optimizer_steps": total_steps,
            "mean_training_loss": total_loss / max(total_steps, 1),
            "training_seconds": time.perf_counter() - started,
        }

    @torch.inference_mode()
    def predict(self, images: torch.Tensor, seen_class_ids: Sequence[int]) -> torch.Tensor:
        self.model.eval()
        logits = self.model(images.to(self.device, non_blocking=True), seen_class_ids)
        return logits.argmax(dim=1)

    def extra_state_dict(self) -> dict:
        return {}

    def load_extra_state_dict(self, state: dict) -> None:
        del state

    def state_dict(self) -> dict:
        return {
            "strategy": self.name,
            "model": self.model.state_dict(),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "extra": self.extra_state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        if state.get("optimizer") is not None:
            self.optimizer = self._make_optimizer()
            self.optimizer.load_state_dict(state["optimizer"])
        self.load_extra_state_dict(state.get("extra", {}))
