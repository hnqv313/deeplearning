"""A fixed-width classifier with unseen-class masking for fair comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

BACKBONE_PREFIX = "backbone."


class TinyConvBackbone(nn.Module):
    """Small offline smoke-test backbone; not intended for final experiments."""

    num_features = 32

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def load_backbone_state(backbone: nn.Module, checkpoint_path: str | Path) -> None:
    """Copy backbone tensors out of a `continual_dl.run` stage checkpoint.

    Only ``backbone.*`` entries are read, so a checkpoint produced by any
    gradient-based strategy can seed a frozen-feature strategy that discards
    the classifier head.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    try:
        model_state = payload["strategy_state"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Checkpoint has no strategy_state.model entry: {path}") from error
    backbone_state = {
        key.removeprefix(BACKBONE_PREFIX): tensor
        for key, tensor in model_state.items()
        if key.startswith(BACKBONE_PREFIX)
    }
    if not backbone_state:
        raise ValueError(f"Checkpoint contains no backbone.* weights: {path}")
    try:
        backbone.load_state_dict(backbone_state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"Backbone weights do not match the configured architecture: {path}"
        ) from error


def build_backbone(
    name: str, pretrained: bool, init_checkpoint: str | Path | None = None
) -> tuple[nn.Module, int]:
    if name == "tiny_cnn":
        backbone: nn.Module = TinyConvBackbone()
        feature_dim = int(backbone.num_features)
    elif name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
    else:
        try:
            import timm
        except ImportError as error:
            raise RuntimeError(
                f"Backbone '{name}' requires timm. Install the project with `pip install -e .`."
            ) from error
        backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        feature_dim = int(getattr(backbone, "num_features"))
    if init_checkpoint is not None:
        load_backbone_state(backbone, init_checkpoint)
    return backbone, feature_dim


class ContinualClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            features = features[:, 0]
        elif features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        return features

    def raw_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.extract_features(images)))

    @staticmethod
    def mask_logits(logits: torch.Tensor, seen_class_ids: Sequence[int]) -> torch.Tensor:
        seen = set(int(class_id) for class_id in seen_class_ids)
        unseen = [index for index in range(logits.shape[1]) if index not in seen]
        if not unseen:
            return logits
        masked = logits.clone()
        masked[:, unseen] = -torch.finfo(masked.dtype).max
        return masked

    def forward(
        self, images: torch.Tensor, seen_class_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        logits = self.raw_logits(images)
        return logits if seen_class_ids is None else self.mask_logits(logits, seen_class_ids)


def build_classifier(model_config: dict) -> ContinualClassifier:
    init_checkpoint = model_config.get("init_checkpoint")
    backbone, feature_dim = build_backbone(
        str(model_config["backbone"]),
        bool(model_config.get("pretrained", True)),
        None if init_checkpoint is None else str(init_checkpoint),
    )
    return ContinualClassifier(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=int(model_config["num_classes"]),
        dropout=float(model_config.get("dropout", 0.0)),
    )
