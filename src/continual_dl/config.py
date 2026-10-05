"""YAML configuration loading with deterministic deep merging."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    common_path: str | Path,
    strategy_path: str | Path,
    init_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the common configuration, then the strategy override, then the init override.

    The init file is merged last so it can describe a starting point for the
    backbone without duplicating the strategy's own hyperparameters.
    """

    with Path(common_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for path in (strategy_path, init_path):
        if path is None:
            continue
        with Path(path).open("r", encoding="utf-8") as handle:
            config = _deep_merge(config, yaml.safe_load(handle) or {})
    return config
