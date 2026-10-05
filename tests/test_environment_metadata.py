from __future__ import annotations

import torch
from torch import nn

from continual_dl.utils import backbone_init_metadata, environment_metadata


class _Backbone(nn.Module):
    pretrained_cfg = {
        "architecture": "vit_tiny_patch16_224",
        "tag": "test_tag",
        "hf_hub_id": "example/model.test_tag",
    }


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()


def test_environment_metadata_records_timm_and_pretrained_model() -> None:
    metadata = environment_metadata(torch.device("cpu"), _Model(), {"pretrained": True})

    assert metadata["timm_version"]
    assert metadata["backbone_init"]["pretrained"] is True
    assert metadata["backbone_init"]["pretrained_model"] == _Backbone.pretrained_cfg


def test_random_init_run_does_not_claim_pretrained_weights() -> None:
    metadata = environment_metadata(torch.device("cpu"), _Model(), {"pretrained": False})

    init = metadata["backbone_init"]
    assert init["pretrained"] is False
    assert init["init_checkpoint"] is None
    assert "pretrained_model" not in init


def test_backbone_init_records_loading_checkpoint() -> None:
    init = backbone_init_metadata(
        _Backbone(),
        {"pretrained": False, "init_checkpoint": "outputs/scratch/naive/seed_42/stage_0.pt"},
    )

    assert init["init_checkpoint"] == "outputs/scratch/naive/seed_42/stage_0.pt"
    assert "pretrained_model" not in init
