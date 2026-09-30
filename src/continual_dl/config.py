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


def load_config(common_path: str | Path, strategy_path: str | Path) -> dict[str, Any]:
    """Load the common configuration and merge a strategy-specific override."""

    with Path(common_path).open("r", encoding="utf-8") as handle:
        common = yaml.safe_load(handle) or {}
    with Path(strategy_path).open("r", encoding="utf-8") as handle:
        strategy = yaml.safe_load(handle) or {}
    return _deep_merge(common, strategy)
