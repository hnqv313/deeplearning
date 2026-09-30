"""Online Elastic Weight Consolidation with a diagonal Fisher approximation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .base import BaseStrategy


class EWCStrategy(BaseStrategy):
    name = "ewc"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ewc_lambda = float(self.strategy_config.get("lambda", 100.0))
        self.gamma = float(self.strategy_config.get("gamma", 1.0))
        self.fisher_batches = int(self.strategy_config.get("fisher_batches", 50))
        self.fisher: dict[str, torch.Tensor] = {}
        self.parameter_means: dict[str, torch.Tensor] = {}

    def _penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), device=self.device)
        for name, parameter in self.model.named_parameters():
            if name not in self.fisher:
                continue
            fisher = self.fisher[name].to(self.device)
            mean = self.parameter_means[name].to(self.device)
            penalty = penalty + (fisher * (parameter - mean).pow(2)).sum()
        return penalty

    def compute_loss(self, images: torch.Tensor, targets: torch.Tensor, experience):
        logits = self.model(images, experience.seen_class_ids)
        classification_loss = self.cross_entropy(logits, targets)
        penalty = self._penalty() if self.fisher else torch.zeros((), device=self.device)
        total = classification_loss + 0.5 * self.ewc_lambda * penalty
        return total, {
            "classification_loss": float(classification_loss.detach()),
            "ewc_penalty": float(penalty.detach()),
        }

    def after_experience(self, experience) -> None:
        loader = DataLoader(
            experience.new_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        current = {
            name: torch.zeros_like(parameter, device="cpu")
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.model.eval()
        used_batches = 0
        for images, targets, _ in loader:
            if used_batches >= self.fisher_batches:
                break
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(images, experience.seen_class_ids)
            loss = self.cross_entropy(logits, targets)
            loss.backward()
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None and name in current:
                    current[name] += parameter.grad.detach().cpu().pow(2)
            used_batches += 1

        denominator = max(used_batches, 1)
        for name in current:
            estimate = current[name] / denominator
            if name in self.fisher:
                estimate = self.gamma * self.fisher[name] + estimate
            current[name] = estimate
        self.fisher = current
        self.parameter_means = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.model.zero_grad(set_to_none=True)

    def extra_state_dict(self) -> dict:
        return {"fisher": self.fisher, "parameter_means": self.parameter_means}

    def load_extra_state_dict(self, state: dict) -> None:
        self.fisher = {name: value.cpu() for name, value in state.get("fisher", {}).items()}
        self.parameter_means = {
            name: value.cpu() for name, value in state.get("parameter_means", {}).items()
        }
