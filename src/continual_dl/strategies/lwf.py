"""Learning without Forgetting using old-class logit distillation."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as functional

from .base import BaseStrategy


class LwFStrategy(BaseStrategy):
    name = "lwf"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(self.strategy_config.get("alpha", 1.0))
        self.temperature = float(self.strategy_config.get("temperature", 2.0))
        self.teacher = None
        self.old_class_ids: tuple[int, ...] = ()

    def before_experience(self, experience) -> None:
        if experience.stage_id == 0:
            self.teacher = None
            self.old_class_ids = ()
            return
        self.teacher = deepcopy(self.model).to(self.device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        new_ids = set(experience.new_class_ids)
        self.old_class_ids = tuple(
            class_id for class_id in experience.seen_class_ids if class_id not in new_ids
        )

    def compute_loss(self, images: torch.Tensor, targets: torch.Tensor, experience):
        student_logits = self.model.raw_logits(images)
        classification_logits = self.model.mask_logits(student_logits, experience.seen_class_ids)
        classification_loss = self.cross_entropy(classification_logits, targets)
        if self.teacher is None or not self.old_class_ids:
            return classification_loss, {"classification_loss": float(classification_loss.detach())}

        with torch.no_grad():
            teacher_logits = self.teacher.raw_logits(images)[:, self.old_class_ids]
        student_old_logits = student_logits[:, self.old_class_ids]
        temperature = self.temperature
        distillation_loss = functional.kl_div(
            functional.log_softmax(student_old_logits / temperature, dim=1),
            functional.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature**2)
        total = classification_loss + self.alpha * distillation_loss
        return total, {
            "classification_loss": float(classification_loss.detach()),
            "distillation_loss": float(distillation_loss.detach()),
        }

    def after_experience(self, experience) -> None:
        del experience
        self.teacher = None
