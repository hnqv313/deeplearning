"""Compare NCM across three backbone initializations.

Reads the report CSVs of the published benchmark plus the two non-ImageNet
arms and writes one PNG per view. Every input is validated before plotting, so
a missing or incomplete arm fails with the command needed to produce it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXPECTED_SEEDS = {42, 123, 2026}
EXPECTED_STAGES = {0, 1, 2, 3}

ARMS = (
    ("ImageNet-21k pretrained", "#2c3e50"),
    ("Pre-trained on stage 0 only", "#16a085"),
    ("Untrained (random init)", "#e67e22"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--stage0-pretrain-dir", type=Path, default=Path("outputs/scratch_pretrain")
    )
    parser.add_argument("--scratch-dir", type=Path, default=Path("outputs/scratch"))
    parser.add_argument("--random-dir", type=Path, default=Path("outputs/scratch_random"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pretrain_figures"))
    return parser.parse_args()


def _stat(row: pd.Series, column: str) -> float:
    return float(row[column])


def _load(directory: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_path = directory / "comparison.csv"
    stage_path = directory / "stage_accuracy.csv"
    for path in (comparison_path, stage_path):
        if not path.is_file():
            raise SystemExit(
                f"Missing {path}.\n"
                f"Run the experiments for the '{label}' arm, then "
                "python scripts/summarize_results.py"
            )
    comparison = pd.read_csv(comparison_path)
    stage = pd.read_csv(stage_path)
    if "ncm" not in set(comparison["strategy"]):
        raise SystemExit(f"{comparison_path} has no 'ncm' row")
    per_seed = stage[stage["strategy"] == "ncm"].copy()
    seeds = set(per_seed["seed"])
    if seeds != EXPECTED_SEEDS:
        raise SystemExit(
            f"{stage_path} 'ncm' seeds are {sorted(seeds)}, expected {sorted(EXPECTED_SEEDS)}"
        )
    observed = per_seed.groupby("seed")["stage"].apply(lambda values: set(values))
    if any(values != EXPECTED_STAGES for values in observed):
        raise SystemExit(f"{stage_path} 'ncm' runs must each contain stages 0-3")
    values = per_seed["average_accuracy"]
    if ((values < 0) | (values > 1)).any():
        raise SystemExit(f"{stage_path} contains accuracy outside [0, 1]")
    return comparison, per_seed


def _mean_row(comparison: pd.DataFrame) -> pd.Series:
    return comparison.set_index("strategy").loc["ncm"]


def _pretraining_seconds(directory: Path) -> float:
    if not directory.is_dir():
        return 0.0
    total = 0.0
    for summary_path in sorted(directory.glob("*/seed_*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            total += float(json.load(handle)["total_seconds"])
    return total


def save_final_accuracy(rows: list[pd.Series], output_dir: Path) -> None:
    labels = [name for name, _ in ARMS]
    colors = [color for _, color in ARMS]
    means = [_stat(row, "final_average_accuracy_mean") * 100 for row in rows]
    stds = [_stat(row, "final_average_accuracy_std") * 100 for row in rows]
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    bars = axis.bar(labels, means, yerr=stds, capsize=4, color=colors, width=0.6)
    axis.set_ylabel("Final average accuracy on the five classes (%)")
    axis.set_ylim(0, 103)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.1f}" for value in bars.datavalues], padding=3)
    axis.tick_params(axis="x", labelsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "backbone_source_final_accuracy.png", dpi=220)
    plt.close(figure)


def save_forgetting(rows: list[pd.Series], output_dir: Path) -> None:
    labels = [name for name, _ in ARMS]
    colors = [color for _, color in ARMS]
    means = [_stat(row, "mean_forgetting_mean") * 100 for row in rows]
    stds = [_stat(row, "mean_forgetting_std") * 100 for row in rows]
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    bars = axis.bar(labels, means, yerr=stds, capsize=4, color=colors, width=0.6)
    axis.set_ylabel("Mean forgetting (percentage points)")
    axis.set_ylim(0, 103)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.1f}" for value in bars.datavalues], padding=3)
    axis.tick_params(axis="x", labelsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "backbone_source_forgetting.png", dpi=220)
    plt.close(figure)


def save_stage_accuracy(stages: list[pd.DataFrame], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for (name, color), stage in zip(ARMS, stages, strict=True):
        aggregate = stage.groupby("stage")["average_accuracy"].agg(["mean", "std"])
        axis.errorbar(
            aggregate.index,
            aggregate["mean"] * 100,
            yerr=aggregate["std"] * 100,
            marker="o",
            linewidth=2,
            capsize=3,
            color=color,
            label=name,
        )
    axis.set_xticks([0, 1, 2, 3], ["Dog + Cat", "+ Car", "+ Person", "+ Building"])
    axis.set_xlabel("Continual-learning stage")
    axis.set_ylabel("Average accuracy on seen classes (%)")
    axis.set_ylim(15, 102)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "backbone_source_stage_accuracy.png", dpi=220)
    plt.close(figure)


def save_runtime(rows: list[pd.Series], pretraining_seconds: float, output_dir: Path) -> None:
    labels = [name for name, _ in ARMS]
    colors = [color for _, color in ARMS]
    ncm_seconds = [_stat(row, "total_seconds_mean") for row in rows]
    # The stage-0 arm stores its Naive pre-training run in a separate tree, so
    # its total is the sum of that run plus the NCM evaluation it feeds.
    totals = ncm_seconds.copy()
    totals[1] = ncm_seconds[1] + pretraining_seconds
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.bar(labels, totals, color=colors, width=0.6, label="Total wall time")
    if pretraining_seconds > 0:
        axis.bar(
            [labels[1]],
            [pretraining_seconds],
            color="#0b6655",
            width=0.6,
            label="Backbone pre-training (naive, stage 0)",
        )
    for index, value in enumerate(totals):
        axis.text(index, value, f"{value:.0f}s", ha="center", va="bottom")
    axis.set_ylabel("Mean wall time per run (seconds)")
    axis.set_ylim(0, max(totals) * 1.18)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelsize=9)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "backbone_source_runtime.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    pretrained_comparison, pretrained_stage = _load(args.pretrained_dir, "ImageNet-21k pretrained")
    stage0_comparison, stage0_stage = _load(args.scratch_dir, "stage 0")
    random_comparison, random_stage = _load(args.random_dir, "untrained (random init)")
    rows = [
        _mean_row(pretrained_comparison),
        _mean_row(stage0_comparison),
        _mean_row(random_comparison),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_final_accuracy(rows, args.output_dir)
    save_forgetting(rows, args.output_dir)
    save_stage_accuracy([pretrained_stage, stage0_stage, random_stage], args.output_dir)
    save_runtime(rows, _pretraining_seconds(args.stage0_pretrain_dir), args.output_dir)
    print(f"Validated three NCM backbone arms and wrote four figures to {args.output_dir}")


if __name__ == "__main__":
    main()
