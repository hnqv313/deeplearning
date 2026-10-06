from __future__ import annotations

import pytest
import torch
from torchvision.transforms import Compose, Resize, ToTensor

from continual_dl.constants import CLASS_TO_ID
from continual_dl.data.scenario import build_experiences
from continual_dl.engine import evaluate_strategy
from continual_dl.models import build_classifier
from continual_dl.strategies import build_strategy


def make_config(strategy_name: str) -> dict:
    strategy_options = {
        "name": strategy_name,
        "memory_size": 6,
        "replay_ratio": 0.5,
        "alpha": 1.0,
        "temperature": 2.0,
        "lambda": 1.0,
        "gamma": 1.0,
        "fisher_batches": 1,
    }
    return {
        "seed": 42,
        "data": {"num_workers": 0},
        "model": {"backbone": "tiny_cnn", "pretrained": False, "num_classes": 5},
        "training": {
            "batch_size": 4,
            "epochs_stage0": 1,
            "epochs_incremental": 1,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "amp": False,
            "reset_optimizer_each_stage": True,
        },
        "strategy": strategy_options,
    }


@pytest.mark.parametrize(
    "strategy_name", ["naive", "joint", "ncm", "replay", "lwf", "ewc", "replay_ncm_hybrid"]
)
def test_strategy_two_stage_smoke(tiny_manifests, strategy_name):
    transform = Compose([Resize((32, 32)), ToTensor()])
    experiences = build_experiences(
        train_manifest=tiny_manifests["train"],
        val_manifest=tiny_manifests["val"],
        test_manifest=tiny_manifests["test"],
        stages=(("dog", "cat"), ("car",)),
        class_to_id=CLASS_TO_ID,
        train_transform=transform,
        eval_transform=transform,
    )
    config = make_config(strategy_name)
    strategy = build_strategy(build_classifier(config["model"]), torch.device("cpu"), config)
    for experience in experiences:
        training_metrics = strategy.fit(experience)
        assert training_metrics["training_seconds"] >= 0
        metrics = evaluate_strategy(
            strategy,
            experience.test_dataset,
            experience.seen_class_ids,
            list(CLASS_TO_ID),
            batch_size=4,
            num_workers=0,
        )
        assert 0.0 <= metrics["average_accuracy"] <= 1.0
