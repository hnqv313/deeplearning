"""Validated readers for the aggregated benchmark CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_SEEDS = {42, 123, 2026}
EXPECTED_STAGES = {0, 1, 2, 3}


def load_strategy_summary(
    directory: Path, label: str, strategy: str
) -> tuple[pd.Series, pd.DataFrame]:
    """Return the aggregate row and per-seed stage rows for one strategy.

    Both report scripts read the same CSV pair, so the checks live here rather
    than in either script. A missing or incomplete arm fails with the command
    needed to produce it.
    """
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
    if strategy not in set(comparison["strategy"]):
        raise SystemExit(f"{comparison_path} has no '{strategy}' row")
    per_seed = stage[stage["strategy"] == strategy].copy()
    seeds = set(per_seed["seed"])
    if seeds != EXPECTED_SEEDS:
        raise SystemExit(
            f"{stage_path} '{strategy}' seeds are {sorted(seeds)}, "
            f"expected {sorted(EXPECTED_SEEDS)}"
        )
    observed = per_seed.groupby("seed")["stage"].apply(lambda values: set(values))
    if any(values != EXPECTED_STAGES for values in observed):
        raise SystemExit(f"{stage_path} '{strategy}' runs must each contain stages 0-3")
    values = per_seed["average_accuracy"]
    if ((values < 0) | (values > 1)).any():
        raise SystemExit(f"{stage_path} contains accuracy outside [0, 1]")
    return comparison[comparison["strategy"] == strategy].iloc[0], per_seed