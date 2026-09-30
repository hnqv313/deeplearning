from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_results import summarize


def _write_summary(root: Path, seed: int, memory_bytes: int) -> None:
    run_dir = root / "ncm" / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    stages = [
        {
            "stage_id": stage,
            "average_accuracy": 0.9 + stage * 0.01,
            "seen_class_ids": list(range(stage + 2)),
        }
        for stage in range(4)
    ]
    summary = {
        "strategy": "ncm",
        "seed": seed,
        "total_seconds": 10.0 + seed,
        "continual_metrics": {
            "final_average_accuracy": 0.93,
            "average_incremental_accuracy": 0.915,
            "mean_forgetting": 0.02,
            "mean_backward_transfer": -0.02,
            "stages": stages,
        },
        "stage_results": [
            {"stage_id": stage, "peak_gpu_memory_bytes": memory_bytes + stage}
            for stage in range(4)
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_summarize_writes_stage_and_gpu_memory_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_summary(raw, seed=1, memory_bytes=104857600)
    _write_summary(raw, seed=2, memory_bytes=106954752)
    comparison_path = tmp_path / "comparison.csv"

    summarize(raw, comparison_path)

    comparison = pd.read_csv(comparison_path)
    per_seed = pd.read_csv(tmp_path / "per_seed_results.csv")
    stages = pd.read_csv(tmp_path / "stage_accuracy.csv")
    assert len(per_seed) == 2
    assert len(stages) == 8
    assert per_seed["peak_gpu_memory_mib"].tolist() == pytest.approx(
        [(104857600 + 3) / 1048576, (106954752 + 3) / 1048576]
    )
    assert comparison.loc[0, "peak_gpu_memory_mib_mean"] == pytest.approx(101.0)
