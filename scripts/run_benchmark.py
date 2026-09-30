"""Run the remaining benchmark matrix with resume-by-summary support."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ALL_STRATEGIES = ("naive", "replay", "lwf", "ewc", "joint", "ncm")
DEFAULT_STRATEGIES = ("naive", "replay", "lwf", "ewc", "joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=ALL_STRATEGIES,
        default=list(DEFAULT_STRATEGIES),
        help="Methods to run; defaults to the five GPU runs still pending",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--common-config", type=Path, default=Path("configs/common.yaml"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--force", action="store_true", help="Rerun completed summaries")
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix without training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.common_config.open("r", encoding="utf-8") as handle:
        common = yaml.safe_load(handle) or {}
    seeds = args.seeds or [int(value) for value in common["evaluation"]["seeds"]]
    output_root = Path(common["evaluation"]["output_dir"])

    if args.device == "cuda" and not args.dry_run and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or use --device cpu explicitly."
        )

    scheduled = 0
    skipped = 0
    for strategy in args.strategies:
        for seed in seeds:
            summary_path = output_root / strategy / f"seed_{seed}" / "summary.json"
            if summary_path.exists() and not args.force:
                print(f"SKIP {strategy} seed={seed}: {summary_path} exists")
                skipped += 1
                continue
            command = [
                sys.executable,
                "-m",
                "continual_dl.run",
                "--strategy",
                strategy,
                "--common-config",
                str(args.common_config),
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
            print(f"RUN  {strategy} seed={seed} device={args.device}", flush=True)
            scheduled += 1
            if not args.dry_run:
                subprocess.run(command, check=True)

    print(f"Benchmark matrix complete: scheduled={scheduled}, skipped={skipped}")


if __name__ == "__main__":
    main()
