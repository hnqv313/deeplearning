"""Offline joint-training upper bound."""

from .base import BaseStrategy


class JointStrategy(BaseStrategy):
    name = "joint"

    def training_dataset(self, experience):
        return experience.seen_train_dataset
