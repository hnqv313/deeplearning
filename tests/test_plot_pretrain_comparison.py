from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_pretrain_comparison import _load, _pretraining_seconds, save_final_accuracy

SEEDS = (42, 123, 2026)
STAGES = (0, 1, 2, 3)


def _write_arm(directory: Path, strategy: str = "ncm", seeds=SEEDS, stages=STAGES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "strategy": strategy,
            "seed": seed,
            "final_average_accuracy": 0.5 + 0.1 * index,
            "average_incremental_accuracy": 0.6,
            "total_seconds": 30.0,
        }
        for index, seed in enumerate(seeds)
    ]
    pd.DataFrame(rows).to_csv(directory / "per_seed_results.csv", index=False)
    stage_rows = [
        {
            "strategy": strategy,
            "seed": seed,
            "stage": stage,
            "seen_classes": 5,
            "average_accuracy": 0.4 + 0.1 * stage,
        }
        for seed in seeds
        for stage in stages
    ]
    pd.DataFrame(stage_rows).to_csv(directory / "stage_accuracy.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy": strategy,
                "final_average_accuracy_mean": 0.6,
                "final_average_accuracy_std": 0.05,
                "mean_forgetting_mean": 0.1,
                "mean_forgetting_std": 0.01,
                "total_seconds_mean": 30.0,
            }
        ]
    ).to_csv(directory / "comparison.csv", index=False)
    return directory


def test_load_accepts_complete_three_seed_arm(tmp_path: Path) -> None:
    comparison, per_seed = _load(_write_arm(tmp_path / "arm"), "stage 0")

    assert "ncm" in set(comparison["strategy"])
    assert len(per_seed) == len(SEEDS) * len(STAGES)


def test_load_rejects_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="summarize_results"):
        _load(tmp_path / "absent", "stage 0")


def test_load_rejects_missing_ncm_row(tmp_path: Path) -> None:
    directory = _write_arm(tmp_path / "arm", strategy="replay")

    with pytest.raises(SystemExit, match="no 'ncm' row"):
        _load(directory, "stage 0")


def test_load_rejects_incomplete_seeds(tmp_path: Path) -> None:
    directory = _write_arm(tmp_path / "arm", seeds=(42, 123))

    with pytest.raises(SystemExit, match="2026"):
        _load(directory, "stage 0")


def test_load_rejects_missing_final_stage(tmp_path: Path) -> None:
    directory = _write_arm(tmp_path / "arm", stages=(0, 1, 2))

    with pytest.raises(SystemExit, match="stages 0-3"):
        _load(directory, "stage 0")


def test_load_rejects_accuracy_outside_unit_range(tmp_path: Path) -> None:
    directory = _write_arm(tmp_path / "arm")
    stage_path = directory / "stage_accuracy.csv"
    stage = pd.read_csv(stage_path)
    stage["average_accuracy"] = 1.4
    stage.to_csv(stage_path, index=False)

    with pytest.raises(SystemExit, match=r"\[0, 1\]"):
        _load(directory, "stage 0")


def test_pretraining_seconds_sums_every_seed_summary(tmp_path: Path) -> None:
    for seed, seconds in ((42, 231.4), (123, 240.0)):
        directory = tmp_path / "naive" / f"seed_{seed}"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({"total_seconds": seconds}))

    assert _pretraining_seconds(tmp_path) == pytest.approx(471.4)
    assert _pretraining_seconds(tmp_path / "absent") == 0.0


def test_save_final_accuracy_writes_figure(tmp_path: Path) -> None:
    comparison, _ = _load(_write_arm(tmp_path / "arm"), "stage 0")
    row = comparison.set_index("strategy").loc["ncm"]

    save_final_accuracy([row, row, row], tmp_path)

    assert (tmp_path / "backbone_source_final_accuracy.png").is_file()
