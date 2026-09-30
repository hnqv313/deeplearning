from __future__ import annotations

import torch
from torch import nn

from continual_dl.utils import environment_metadata


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
    metadata = environment_metadata(torch.device("cpu"), _Model())

    assert metadata["timm_version"]
    assert metadata["pretrained_model"] == _Backbone.pretrained_cfg
