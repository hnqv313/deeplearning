"""Reproducibility, device, and serialization helpers."""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(device: torch.device, model: torch.nn.Module | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_version": torch.__version__,
        "device": str(device),
        "git_commit": git_commit(),
    }
    try:
        import timm

        metadata["timm_version"] = timm.__version__
    except ImportError:
        metadata["timm_version"] = None
    if model is not None:
        backbone = getattr(model, "backbone", model)
        pretrained_config = getattr(backbone, "pretrained_cfg", None)
        if isinstance(pretrained_config, dict):
            metadata["pretrained_model"] = {
                key: pretrained_config.get(key)
                for key in ("architecture", "tag", "hf_hub_id")
            }
    if device.type == "cuda":
        metadata["cuda_device"] = torch.cuda.get_device_name(device)
        metadata["cuda_version"] = torch.version.cuda
    return metadata


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)
