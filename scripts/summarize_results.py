"""Aggregate benchmark JSON summaries into report-ready CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SUMMARY_KEYS = (
    "final_average_accuracy",
    "average_incremental_accuracy",
    "mean_forgetting",
    "mean_backward_transfer",
)
BYTES_PER_MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/comparison.csv"))
    return parser.parse_args()


def summarize(outputs: Path, output_csv: Path) -> pd.DataFrame:
    raw_rows = []
    stage_rows = []
    for summary_path in sorted(outputs.glob("*/seed_*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metrics = summary["continual_metrics"]
        row = {"strategy": summary["strategy"], "seed": summary["seed"]}
        row.update({key: float(metrics[key]) for key in SUMMARY_KEYS})
        row["total_seconds"] = float(summary["total_seconds"])
        stage_results = summary.get("stage_results", [])
        if not stage_results:
            raise ValueError(f"Missing stage_results in {summary_path}")
        row["peak_gpu_memory_mib"] = max(
            float(stage["peak_gpu_memory_bytes"]) for stage in stage_results
        ) / BYTES_PER_MIB
        raw_rows.append(row)

        metric_stages = metrics.get("stages", [])
        if len(metric_stages) != len(stage_results):
            raise ValueError(f"Stage count mismatch in {summary_path}")
        for stage in metric_stages:
            stage_rows.append(
                {
                    "strategy": summary["strategy"],
                    "seed": summary["seed"],
                    "stage": int(stage["stage_id"]),
                    "seen_classes": len(stage["seen_class_ids"]),
                    "average_accuracy": float(stage["average_accuracy"]),
                }
            )
    if not raw_rows:
        raise ValueError(f"No seed summaries found below {outputs}")
    frame = pd.DataFrame(raw_rows)
    if frame.duplicated(["strategy", "seed"]).any():
        raise ValueError("Duplicate strategy/seed summaries found")
    # Seeds identify repetitions; averaging their numeric IDs is meaningless.
    # Aggregate only reportable measurements and keep seeds in the per-run CSV.
    metric_columns = [*SUMMARY_KEYS, "total_seconds", "peak_gpu_memory_mib"]
    aggregate = frame.groupby("strategy")[metric_columns].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_csv, index=False, float_format="%.9f")
    frame.to_csv(
        output_csv.with_name("per_seed_results.csv"),
        index=False,
        float_format="%.9f",
    )
    pd.DataFrame(stage_rows).to_csv(
        output_csv.with_name("stage_accuracy.csv"),
        index=False,
        float_format="%.9f",
    )
    return aggregate


def main() -> None:
    args = parse_args()
    aggregate = summarize(args.outputs, args.output_csv)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
