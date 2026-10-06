"""Compare NCM and Replay across backbone initializations.

Reads the report CSVs of the published benchmark plus the non-ImageNet arms and
writes one PNG per view. Every input is validated before plotting, so a missing
or incomplete arm fails with the command needed to produce it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from continual_dl.reporting import load_strategy_summary

ARMS = (
    ("ImageNet-21k pretrained", "#2c3e50"),
    ("Pre-trained on stage 0 only", "#16a085"),
    ("Untrained (random init)", "#e67e22"),
)

METHOD_ARMS_LABELS = (
    "NCM\nImageNet-21k",
    "NCM\nstage 0",
    "Replay\nImageNet-21k",
    "Replay\nstage 0",
)
METHOD_ARMS_COLORS = ("#2c3e50", "#16a085", "#5dade2", "#48c9b0")


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


def _load(directory: Path, label: str) -> tuple[pd.Series, pd.DataFrame]:
    return load_strategy_summary(directory, label, "ncm")


def _mean_row(comparison: pd.Series) -> pd.Series:
    return comparison


def save_method_comparison(rows: list[pd.Series], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for axis, column, ylabel, filename in (
        (axes[0], "final_average_accuracy_mean", "Final average accuracy (%)", "final_accuracy"),
        (axes[1], "mean_forgetting_mean", "Mean forgetting (percentage points)", "forgetting"),
    ):
        means = [_stat(row, column) * 100 for row in rows]
        stds = [_stat(row, column.replace("_mean", "_std")) * 100 for row in rows]
        bars = axis.bar(
            METHOD_ARMS_LABELS, means, yerr=stds, capsize=4, color=METHOD_ARMS_COLORS, width=0.6
        )
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 103)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in bars.datavalues], padding=3)
        axis.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "method_backbone_comparison.png", dpi=220)
    plt.close(figure)


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
    pretrained_ncm, pretrained_stage = _load(args.pretrained_dir, "ImageNet-21k pretrained")
    stage0_ncm, stage0_stage = _load(args.scratch_dir, "stage 0")
    random_ncm, random_stage = _load(args.random_dir, "untrained (random init)")
    rows = [_mean_row(pretrained_ncm), _mean_row(stage0_ncm), _mean_row(random_ncm)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_final_accuracy(rows, args.output_dir)
    save_forgetting(rows, args.output_dir)
    save_stage_accuracy([pretrained_stage, stage0_stage, random_stage], args.output_dir)
    save_runtime(rows, _pretraining_seconds(args.stage0_pretrain_dir), args.output_dir)

    pretrained_replay, _ = load_strategy_summary(args.pretrained_dir, "ImageNet-21k pretrained", "replay")
    stage0_replay, _ = load_strategy_summary(args.scratch_dir, "stage 0", "replay")
    save_method_comparison(
        [pretrained_ncm, stage0_ncm, pretrained_replay, stage0_replay], args.output_dir
    )
    print(f"Validated three NCM backbone arms and wrote five figures to {args.output_dir}")


if __name__ == "__main__":
    main()
