from __future__ import annotations

import pytest
import torch
from torchvision.transforms import Compose, Resize, ToTensor

from continual_dl.constants import CLASS_TO_ID
from continual_dl.data.scenario import build_experiences
from continual_dl.models import build_classifier
from continual_dl.strategies import build_strategy
from continual_dl.strategies.factory import STRATEGIES

TINY_MODEL = {"backbone": "tiny_cnn", "pretrained": False, "num_classes": 5, "dropout": 0.0}
TRANSFORM = Compose([Resize((32, 32)), ToTensor()])


def make_config(**overrides) -> dict:
    config = {
        "seed": 42,
        "data": {"num_workers": 0},
        "model": dict(TINY_MODEL),
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
        "strategy": {
            "name": "replay_ncm_hybrid",
            "memory_size": 8,
            "replay_ratio": 0.5,
        },
    }
    config.update(overrides)
    return config


def build_hybrid(tiny_manifests, config=None):
    experiences = build_experiences(
        train_manifest=tiny_manifests["train"],
        val_manifest=tiny_manifests["val"],
        test_manifest=tiny_manifests["test"],
        stages=(("dog", "cat"), ("car",), ("person",), ("building",)),
        class_to_id=CLASS_TO_ID,
        train_transform=TRANSFORM,
        eval_transform=TRANSFORM,
    )
    resolved = config or make_config()
    strategy = build_strategy(build_classifier(resolved["model"]), torch.device("cpu"), resolved)
    return strategy, experiences


def test_strategy_is_registered() -> None:
    assert "replay_ncm_hybrid" in STRATEGIES
    assert STRATEGIES["replay_ncm_hybrid"].name == "replay_ncm_hybrid"


def test_prototypes_cover_seen_classes_after_each_stage(tiny_manifests) -> None:
    strategy, experiences = build_hybrid(tiny_manifests)

    for experience in experiences:
        strategy.fit(experience)
        assert set(strategy.prototypes) == set(experience.seen_class_ids)


def test_prototypes_are_unit_norm(tiny_manifests) -> None:
    strategy, experiences = build_hybrid(tiny_manifests)

    strategy.fit(experiences[0])

    for prototype in strategy.prototypes.values():
        assert pytest.approx(1.0, abs=1e-5) == float(prototype.norm())


def test_prototypes_are_recomputed_as_backbone_moves(tiny_manifests) -> None:
    strategy, experiences = build_hybrid(tiny_manifests)

    strategy.fit(experiences[0])
    stage0 = {key: value.clone() for key, value in strategy.prototypes.items()}
    strategy.fit(experiences[1])

    for key, value in stage0.items():
        if key in strategy.prototypes:
            assert not torch.equal(value, strategy.prototypes[key])


def test_predict_returns_only_seen_classes(tiny_manifests) -> None:
    strategy, experiences = build_hybrid(tiny_manifests)

    for experience in experiences:
        strategy.fit(experience)
        predictions = strategy.predict(torch.randn(3, 3, 32, 32), experience.seen_class_ids)
        assert set(predictions.tolist()) <= set(experience.seen_class_ids)


def test_predict_requires_prototype_for_every_seen_class(tiny_manifests) -> None:
    strategy, experiences = build_hybrid(tiny_manifests)

    strategy.fit(experiences[0])

    with pytest.raises(RuntimeError, match="Missing NCM prototypes"):
        strategy.predict(torch.randn(2, 3, 32, 32), [0, 1, 2])


def test_buffer_stays_within_memory_budget(tiny_manifests) -> None:
    strategy, experiences = build_hybrid(tiny_manifests)

    for experience in experiences:
        strategy.fit(experience)
        assert len(strategy.buffer) <= 8


def test_extra_state_dict_round_trip(tiny_manifests) -> None:
    strategy, experiences = build_hybrid(tiny_manifests)
    strategy.fit(experiences[0])
    state = strategy.extra_state_dict()

    restored = build_strategy(build_classifier(TINY_MODEL), torch.device("cpu"), make_config())
    restored.load_extra_state_dict(state)

    assert set(restored.prototypes) == set(strategy.prototypes)
    assert len(restored.buffer) == len(strategy.buffer)
