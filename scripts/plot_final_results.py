"""Validate final benchmark CSVs and generate report-ready comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGY_ORDER = ("naive", "ewc", "lwf", "replay", "ncm", "joint")
DISPLAY_NAMES = {
    "naive": "Naive",
    "ewc": "EWC",
    "lwf": "LwF",
    "replay": "Replay",
    "ncm": "ViT-Tiny + NCM",
    "joint": "Joint (upper bound)",
}
COLORS = {
    "naive": "#7f8c8d",
    "ewc": "#e67e22",
    "lwf": "#9b59b6",
    "replay": "#3498db",
    "ncm": "#16a085",
    "joint": "#2c3e50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=Path("outputs/comparison.csv"))
    parser.add_argument("--per-seed", type=Path, default=Path("outputs/per_seed_results.csv"))
    parser.add_argument("--stage", type=Path, default=Path("outputs/stage_accuracy.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_figures"))
    return parser.parse_args()


def validate_inputs(
    comparison: pd.DataFrame,
    per_seed: pd.DataFrame,
    stage: pd.DataFrame,
) -> None:
    expected = set(STRATEGY_ORDER)
    if set(comparison["strategy"]) != expected:
        raise ValueError("comparison.csv must contain exactly the six expected strategies")
    if len(per_seed) != 18 or set(per_seed["strategy"]) != expected:
        raise ValueError("per_seed_results.csv must contain six strategies x three seeds")
    if per_seed.duplicated(["strategy", "seed"]).any():
        raise ValueError("Duplicate strategy/seed rows in per_seed_results.csv")
    seed_sets = per_seed.groupby("strategy")["seed"].apply(lambda values: set(values))
    if any(seeds != {42, 123, 2026} for seeds in seed_sets):
        raise ValueError("Every strategy must contain seeds 42, 123 and 2026")
    if len(stage) != 72 or stage.duplicated(["strategy", "seed", "stage"]).any():
        raise ValueError("stage_accuracy.csv must contain 6 x 3 x 4 unique rows")
    stages = stage.groupby(["strategy", "seed"])["stage"].apply(lambda values: set(values))
    if any(values != {0, 1, 2, 3} for values in stages):
        raise ValueError("Every strategy/seed pair must contain stages 0 through 3")
    accuracy_columns = [
        comparison["final_average_accuracy_mean"],
        per_seed["final_average_accuracy"],
        per_seed["average_incremental_accuracy"],
        stage["average_accuracy"],
    ]
    if any(((values < 0) | (values > 1)).any() for values in accuracy_columns):
        raise ValueError("Accuracy values must lie in [0, 1]")

    metric_names = (
        "final_average_accuracy",
        "average_incremental_accuracy",
        "mean_forgetting",
        "mean_backward_transfer",
        "total_seconds",
        "peak_gpu_memory_mib",
    )
    recomputed = per_seed.groupby("strategy")[list(metric_names)].agg(["mean", "std"])
    for strategy in STRATEGY_ORDER:
        reported = comparison.set_index("strategy").loc[strategy]
        for metric in metric_names:
            for statistic in ("mean", "std"):
                expected_value = recomputed.loc[strategy, (metric, statistic)]
                reported_value = reported[f"{metric}_{statistic}"]
                if not np.isclose(expected_value, reported_value, atol=1e-5):
                    raise ValueError(
                        f"comparison.csv disagrees with per-seed data for "
                        f"{strategy} {metric}_{statistic}"
                    )

    final_stage = stage[stage["stage"] == 3][
        ["strategy", "seed", "average_accuracy"]
    ].rename(columns={"average_accuracy": "stage_final_accuracy"})
    joined = per_seed.merge(final_stage, on=["strategy", "seed"], validate="one_to_one")
    if not np.allclose(
        joined["final_average_accuracy"], joined["stage_final_accuracy"], atol=5e-5
    ):
        raise ValueError("Stage-3 accuracy disagrees with per-seed final accuracy")


def save_stage_accuracy(stage: pd.DataFrame, output_dir: Path) -> None:
    aggregate = (
        stage.groupby(["strategy", "stage"])["average_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for strategy in STRATEGY_ORDER:
        rows = aggregate[aggregate["strategy"] == strategy].sort_values("stage")
        axis.errorbar(
            rows["stage"],
            rows["mean"] * 100,
            yerr=rows["std"] * 100,
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS[strategy],
            label=DISPLAY_NAMES[strategy],
        )
    axis.set_xticks([0, 1, 2, 3], ["Dog + Cat", "+ Car", "+ Person", "+ Building"])
    axis.set_ylabel("Average accuracy on seen classes (%)")
    axis.set_xlabel("Continual-learning stage")
    axis.set_ylim(15, 102)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "stage_accuracy.png", dpi=220)
    plt.close(figure)


def save_final_accuracy(comparison: pd.DataFrame, output_dir: Path) -> None:
    rows = comparison.set_index("strategy").loc[list(STRATEGY_ORDER)]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(
        x,
        rows["final_average_accuracy_mean"] * 100,
        yerr=rows["final_average_accuracy_std"] * 100,
        capsize=4,
        color=[COLORS[name] for name in rows.index],
    )
    axis.set_xticks(x, [DISPLAY_NAMES[name] for name in rows.index], rotation=18, ha="right")
    axis.set_ylabel("Final average accuracy (%)")
    axis.set_ylim(0, 103)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.1f}" for value in bars.datavalues], padding=3)
    figure.tight_layout()
    figure.savefig(output_dir / "final_accuracy.png", dpi=220)
    plt.close(figure)


def save_transfer_metrics(comparison: pd.DataFrame, output_dir: Path) -> None:
    rows = comparison.set_index("strategy").loc[list(STRATEGY_ORDER)]
    x = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        x - width / 2,
        rows["mean_forgetting_mean"] * 100,
        width,
        yerr=rows["mean_forgetting_std"] * 100,
        capsize=3,
        label="Mean forgetting",
        color="#e74c3c",
    )
    axis.bar(
        x + width / 2,
        rows["mean_backward_transfer_mean"] * 100,
        width,
        yerr=rows["mean_backward_transfer_std"] * 100,
        capsize=3,
        label="Backward transfer",
        color="#2980b9",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, [DISPLAY_NAMES[name] for name in rows.index], rotation=18, ha="right")
    axis.set_ylabel("Metric value (percentage points)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "forgetting_and_bwt.png", dpi=220)
    plt.close(figure)


def save_runtime(comparison: pd.DataFrame, output_dir: Path) -> None:
    rows = comparison.set_index("strategy").loc[list(STRATEGY_ORDER)]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(
        x,
        rows["total_seconds_mean"],
        yerr=rows["total_seconds_std"],
        capsize=4,
        color=[COLORS[name] for name in rows.index],
    )
    axis.set_xticks(x, [DISPLAY_NAMES[name] for name in rows.index], rotation=18, ha="right")
    axis.set_ylabel("Mean wall time per run (seconds)")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.0f}" for value in bars.datavalues], padding=3)
    figure.tight_layout()
    figure.savefig(output_dir / "runtime.png", dpi=220)
    plt.close(figure)


def save_gpu_memory(comparison: pd.DataFrame, output_dir: Path) -> None:
    rows = comparison.set_index("strategy").loc[list(STRATEGY_ORDER)]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(
        x,
        rows["peak_gpu_memory_mib_mean"],
        yerr=rows["peak_gpu_memory_mib_std"],
        capsize=4,
        color=[COLORS[name] for name in rows.index],
    )
    axis.set_xticks(x, [DISPLAY_NAMES[name] for name in rows.index], rotation=18, ha="right")
    axis.set_ylabel("Peak allocated GPU memory (MiB)")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.0f}" for value in bars.datavalues], padding=3)
    figure.tight_layout()
    figure.savefig(output_dir / "gpu_memory.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    comparison = pd.read_csv(args.comparison)
    per_seed = pd.read_csv(args.per_seed)
    stage = pd.read_csv(args.stage)
    validate_inputs(comparison, per_seed, stage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_stage_accuracy(stage, args.output_dir)
    save_final_accuracy(comparison, args.output_dir)
    save_transfer_metrics(comparison, args.output_dir)
    save_runtime(comparison, args.output_dir)
    save_gpu_memory(comparison, args.output_dir)
    print(f"Validated 18 runs and wrote five figures to {args.output_dir}")


if __name__ == "__main__":
    main()
