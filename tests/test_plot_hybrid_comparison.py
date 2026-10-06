from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_hybrid_comparison import METHODS, _collect, save_figure
from continual_dl.reporting import EXPECTED_SEEDS

STRATEGIES = METHODS
SEEDS = (42, 123, 2026)
STAGES = (0, 1, 2, 3)


def _write_arm(directory: Path, accuracy: float, forgetting: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "strategy": name,
            "final_average_accuracy_mean": accuracy,
            "final_average_accuracy_std": 0.02,
            "mean_forgetting_mean": forgetting,
            "mean_forgetting_std": 0.01,
        }
        for name in STRATEGIES
    ]
    pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False)
    stage_rows = [
        {
            "strategy": name,
            "seed": seed,
            "stage": index,
            "seen_classes": 5,
            "average_accuracy": accuracy,
        }
        for name in STRATEGIES
        for seed in SEEDS
        for index in STAGES
    ]
    pd.DataFrame(stage_rows).to_csv(directory / "stage_accuracy.csv", index=False)
    return directory


def test_collect_reads_all_three_methods_for_both_backbones(tmp_path: Path) -> None:
    _write_arm(tmp_path / "published", 0.9, 0.05)
    _write_arm(tmp_path / "hybrid", 0.9, 0.05)
    _write_arm(tmp_path / "scratch", 0.4, 0.3)

    groups = _collect(tmp_path / "published", tmp_path / "hybrid", tmp_path / "scratch")

    assert len(groups) == 2
    for strategies in groups.values():
        assert set(strategies) == set(METHODS)
    scratch = groups["Stage-0 pre-trained backbone"]
    assert scratch["replay_ncm_hybrid"]["final_average_accuracy_mean"] == pytest.approx(0.4)


def test_collect_rejects_arm_missing_the_hybrid(tmp_path: Path) -> None:
    _write_arm(tmp_path / "published", 0.9, 0.05)
    _write_arm(tmp_path / "scratch", 0.4, 0.3)
    arm = _write_arm(tmp_path / "hybrid", 0.9, 0.05)
    comparison = pd.read_csv(arm / "comparison.csv")
    comparison[comparison["strategy"] != "replay_ncm_hybrid"].to_csv(
        arm / "comparison.csv", index=False
    )

    with pytest.raises(SystemExit, match="no 'replay_ncm_hybrid' row"):
        _collect(tmp_path / "published", arm, tmp_path / "scratch")


def test_save_figure_writes_png(tmp_path: Path) -> None:
    _write_arm(tmp_path / "published", 0.9, 0.05)
    _write_arm(tmp_path / "hybrid", 0.9, 0.05)
    _write_arm(tmp_path / "scratch", 0.4, 0.3)
    groups = _collect(tmp_path / "published", tmp_path / "hybrid", tmp_path / "scratch")

    save_figure(groups, tmp_path)

    figure = tmp_path / "hybrid_backbone_comparison.png"
    assert figure.is_file()
    assert figure.stat().st_size > 0


def test_expected_seeds_are_the_three_configured_seeds() -> None:
    assert EXPECTED_SEEDS == {42, 123, 2026}
