"""Compare NCM, Replay and the Replay+NCM hybrid across two backbones.

The hybrid arm lives in its own output tree so the published six-method CSVs are
never rewritten. Every input is validated before plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from continual_dl.reporting import EXPECTED_SEEDS, load_strategy_summary

METHODS = ("ncm", "replay", "replay_ncm_hybrid")
DISPLAY_NAMES = {
    "ncm": "NCM",
    "replay": "Replay",
    "replay_ncm_hybrid": "Replay+NCM hybrid",
}
COLORS = {"ncm": "#16a085", "replay": "#3498db", "replay_ncm_hybrid": "#8e44ad"}
BACKBONE_COLUMNS = ("final_average_accuracy_mean", "mean_forgetting_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--hybrid-dir", type=Path, default=Path("outputs/hybrid"))
    parser.add_argument("--scratch-dir", type=Path, default=Path("outputs/scratch"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hybrid_figures"))
    return parser.parse_args()


def _stat(row: pd.Series, column: str) -> float:
    return float(row[column])


def _collect(pretrained_dir: Path, hybrid_dir: Path, scratch_dir: Path) -> dict[str, dict]:
    """Return {backbone group: {strategy: comparison row}}."""
    return {
        "ImageNet-21k backbone": {
            "ncm": load_strategy_summary(pretrained_dir, "ImageNet-21k pretrained", "ncm")[0],
            "replay": load_strategy_summary(pretrained_dir, "ImageNet-21k pretrained", "replay")[0],
            "replay_ncm_hybrid": load_strategy_summary(hybrid_dir, "hybrid", "replay_ncm_hybrid")[0],
        },
        "Stage-0 pre-trained backbone": {
            strategy: load_strategy_summary(scratch_dir, "stage 0", strategy)[0] for strategy in METHODS
        },
    }


def save_figure(groups: dict[str, dict], output_dir: Path) -> None:
    rows = [
        ("Final average accuracy (%)", "final_average_accuracy_mean", "final_average_accuracy_std"),
        ("Mean forgetting (percentage points)", "mean_forgetting_mean", "mean_forgetting_std"),
    ]
    figure, axes = plt.subplots(
        len(rows), len(groups), figsize=(6.0 * len(groups), 4.6 * len(rows))
    )
    axes = axes.reshape(len(rows), len(groups))
    for row_index, (ylabel, mean_column, std_column) in enumerate(rows):
        for column_index, (group, strategies) in enumerate(groups.items()):
            axis = axes[row_index][column_index]
            means = [_stat(strategies[name], mean_column) * 100 for name in METHODS]
            stds = [_stat(strategies[name], std_column) * 100 for name in METHODS]
            bars = axis.bar(
                [DISPLAY_NAMES[name] for name in METHODS],
                means,
                yerr=stds,
                capsize=4,
                color=[COLORS[name] for name in METHODS],
                width=0.6,
            )
            axis.set_title(group, fontsize=11)
            axis.set_ylabel(ylabel)
            axis.set_ylim(0, 103)
            axis.grid(axis="y", alpha=0.25)
            axis.bar_label(bars, labels=[f"{value:.1f}" for value in bars.datavalues], padding=3)
            axis.tick_params(axis="x", labelsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "hybrid_backbone_comparison.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    groups = _collect(args.pretrained_dir, args.hybrid_dir, args.scratch_dir)
    missing = [
        (group, name)
        for group, strategies in groups.items()
        for name in METHODS
        if name not in strategies
    ]
    if missing:
        raise SystemExit(f"Missing rows: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(groups, args.output_dir)
    seeds = sorted(EXPECTED_SEEDS)
    print(
        f"Validated three methods over seeds {seeds} for two backbones and wrote one figure "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
