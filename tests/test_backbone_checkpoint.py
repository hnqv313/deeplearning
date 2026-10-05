from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision.transforms import Compose, Resize, ToTensor

from continual_dl.constants import CLASS_TO_ID
from continual_dl.data.scenario import build_experiences
from continual_dl.engine import evaluate_strategy
from continual_dl.models import build_classifier
from continual_dl.models.classifier import ContinualClassifier, build_backbone
from continual_dl.run import resolve_model_config
from continual_dl.strategies import build_strategy


def _tiny_config() -> dict:
    return {"backbone": "tiny_cnn", "pretrained": False, "num_classes": 5, "dropout": 0.0}


def _set_backbone(model: ContinualClassifier, value: float) -> None:
    with torch.no_grad():
        for parameter in model.backbone.parameters():
            parameter.fill_(value)


def _backbone_state(model: ContinualClassifier) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("backbone."): tensor
        for key, tensor in model.state_dict().items()
        if key.startswith("backbone.")
    }


def _write_checkpoint(path: Path, model: ContinualClassifier) -> None:
    payload = {
        "stage_id": 0,
        "config": {"seed": 42},
        "strategy_state": {
            "strategy": "naive",
            "model": model.state_dict(),
            "optimizer": None,
            "extra": {},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_backbone_weights_load_from_run_checkpoint(tmp_path: Path) -> None:
    trained = build_classifier(_tiny_config())
    _set_backbone(trained, 0.25)
    checkpoint = tmp_path / "stage_0.pt"
    _write_checkpoint(checkpoint, trained)

    loaded = build_classifier({**_tiny_config(), "init_checkpoint": str(checkpoint)})

    expected = _backbone_state(trained)
    loaded_state = _backbone_state(loaded)
    assert set(loaded_state) == set(expected)
    assert all(torch.equal(loaded_state[key], expected[key]) for key in expected)
    # The 5-way head is dropped on purpose: NCM builds class prototypes instead.
    assert not torch.equal(loaded.classifier.weight, trained.classifier.weight)


def test_backbone_checkpoint_without_backbone_keys_is_rejected(tmp_path: Path) -> None:
    model = build_classifier(_tiny_config())
    checkpoint = tmp_path / "head_only.pt"
    _write_checkpoint(checkpoint, model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["strategy_state"]["model"] = {
        key: value
        for key, value in payload["strategy_state"]["model"].items()
        if key.startswith("classifier.")
    }
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="backbone"):
        build_backbone("tiny_cnn", pretrained=False, init_checkpoint=checkpoint)


def test_missing_backbone_checkpoint_reports_path(tmp_path: Path) -> None:
    missing = tmp_path / "absent" / "stage_0.pt"

    with pytest.raises(FileNotFoundError, match="stage_0.pt"):
        build_backbone("tiny_cnn", pretrained=False, init_checkpoint=missing)


def test_resolve_model_config_is_per_seed_and_leaves_input_untouched(tmp_path: Path) -> None:
    model_config = {
        "backbone": "tiny_cnn",
        "pretrained": False,
        "init_checkpoint": "outputs/scratch/pretrain/naive/seed_{seed}/checkpoints/stage_0.pt",
    }
    snapshot = dict(model_config)

    first = resolve_model_config(model_config, tmp_path, 42)
    second = resolve_model_config(model_config, tmp_path, 123)

    assert Path(first["init_checkpoint"]) == (
        tmp_path / "outputs/scratch/pretrain/naive/seed_42/checkpoints/stage_0.pt"
    )
    assert Path(second["init_checkpoint"]) == (
        tmp_path / "outputs/scratch/pretrain/naive/seed_123/checkpoints/stage_0.pt"
    )
    assert model_config == snapshot


def test_ncm_runs_on_a_pretrained_backbone_checkpoint(tiny_manifests, tmp_path: Path) -> None:
    transform = Compose([Resize((32, 32)), ToTensor()])
    pretrain_model = build_classifier(_tiny_config())
    _set_backbone(pretrain_model, 0.25)
    checkpoint = tmp_path / "pretrain" / "stage_0.pt"
    _write_checkpoint(checkpoint, pretrain_model)

    config = {
        "seed": 42,
        "data": {"num_workers": 0},
        "model": {**_tiny_config(), "init_checkpoint": str(checkpoint)},
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
        "strategy": {"name": "ncm"},
    }
    strategy = build_strategy(build_classifier(config["model"]), torch.device("cpu"), config)
    experiences = build_experiences(
        train_manifest=tiny_manifests["train"],
        val_manifest=tiny_manifests["val"],
        test_manifest=tiny_manifests["test"],
        stages=(("dog", "cat"), ("car",)),
        class_to_id=CLASS_TO_ID,
        train_transform=transform,
        eval_transform=transform,
    )
    for experience in experiences:
        strategy.fit(experience)
        metrics = evaluate_strategy(
            strategy,
            experience.test_dataset,
            experience.seen_class_ids,
            list(CLASS_TO_ID),
            batch_size=4,
            num_workers=0,
        )
        assert 0.0 <= metrics["average_accuracy"] <= 1.0

    expected = _backbone_state(pretrain_model)
    final_state = _backbone_state(strategy.model)
    assert all(torch.equal(final_state[key], expected[key]) for key in expected)
