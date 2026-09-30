"""Command-line entry point for reproducible continual-learning experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from continual_dl.config import load_config
from continual_dl.data.scenario import build_experiences
from continual_dl.data.transforms import build_eval_transform, build_train_transform
from continual_dl.engine import evaluate_strategy
from continual_dl.engine.plots import save_confusion_matrix
from continual_dl.metrics import ContinualMetricTracker
from continual_dl.models import build_classifier
from continual_dl.strategies import build_strategy
from continual_dl.utils import (
    ensure_directory,
    environment_metadata,
    resolve_device,
    save_json,
    set_seed,
)


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def run_experiment(config: dict, project_root: Path, seed: int) -> dict:
    set_seed(seed)
    config = {**config, "seed": seed}
    device = resolve_device(str(config.get("device", "auto")))
    class_order = [str(value) for value in config["data"]["class_order"]]
    class_to_id = {name: index for index, name in enumerate(class_order)}
    image_size = int(config["data"].get("image_size", 224))
    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)
    experiences = build_experiences(
        train_manifest=_project_path(project_root, config["data"]["train_manifest"]),
        val_manifest=_project_path(project_root, config["data"]["val_manifest"]),
        test_manifest=_project_path(project_root, config["data"]["test_manifest"]),
        stages=config["data"]["stages"],
        class_to_id=class_to_id,
        train_transform=train_transform,
        eval_transform=eval_transform,
        root=project_root,
    )
    model = build_classifier(config["model"])
    strategy = build_strategy(model, device, config)
    output_root = _project_path(project_root, config["evaluation"]["output_dir"])
    run_directory = ensure_directory(output_root / strategy.name / f"seed_{seed}")
    ensure_directory(run_directory / "checkpoints")
    ensure_directory(run_directory / "figures")
    save_json(config, run_directory / "config.json")
    save_json(environment_metadata(device, model), run_directory / "environment.json")

    tracker = ContinualMetricTracker()
    stage_results: list[dict] = []
    started = time.perf_counter()
    for experience in experiences:
        print(
            f"[{strategy.name}] stage {experience.stage_id}: "
            f"new={experience.new_class_names}, seen={experience.seen_class_names}"
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training_metrics = strategy.fit(experience)
        validation_metrics = evaluate_strategy(
            strategy,
            experience.val_dataset,
            experience.seen_class_ids,
            class_order,
            batch_size=int(config["training"]["batch_size"]),
            num_workers=int(config["data"].get("num_workers", 0)),
        )
        test_metrics = evaluate_strategy(
            strategy,
            experience.test_dataset,
            experience.seen_class_ids,
            class_order,
            batch_size=int(config["training"]["batch_size"]),
            num_workers=int(config["data"].get("num_workers", 0)),
        )
        tracker.add(experience.stage_id, test_metrics)
        result = {
            "stage_id": experience.stage_id,
            "new_classes": experience.new_class_names,
            "seen_classes": experience.seen_class_names,
            "training": training_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        }
        stage_results.append(result)
        save_json(result, run_directory / f"stage_{experience.stage_id}.json")
        save_confusion_matrix(
            test_metrics["confusion_matrix"],
            experience.seen_class_names,
            run_directory / "figures" / f"stage_{experience.stage_id}_confusion_matrix.png",
        )
        if bool(config["evaluation"].get("save_checkpoints", True)):
            torch.save(
                {
                    "stage_id": experience.stage_id,
                    "config": config,
                    "strategy_state": strategy.state_dict(),
                },
                run_directory / "checkpoints" / f"stage_{experience.stage_id}.pt",
            )
        print(
            f"  test average accuracy={test_metrics['average_accuracy']:.4f}, "
            f"overall={test_metrics['overall_accuracy']:.4f}"
        )

    summary = {
        "strategy": strategy.name,
        "seed": seed,
        "total_seconds": time.perf_counter() - started,
        "continual_metrics": tracker.summary(),
        "stage_results": stage_results,
    }
    save_json(summary, run_directory / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=["naive", "joint", "ncm", "replay", "lwf", "ewc"],
        required=True,
    )
    parser.add_argument("--common-config", type=Path, default=Path("configs/common.yaml"))
    parser.add_argument("--strategy-config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--device", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    strategy_config = args.strategy_config or Path(f"configs/{args.strategy}.yaml")
    config = load_config(args.common_config, strategy_config)
    if args.device is not None:
        config["device"] = args.device
    if args.all_seeds:
        seeds = [int(value) for value in config["evaluation"]["seeds"]]
    else:
        seeds = [int(args.seed if args.seed is not None else config.get("seed", 42))]
    summaries = [run_experiment(config, project_root, seed) for seed in seeds]
    if len(summaries) > 1:
        output_root = _project_path(project_root, config["evaluation"]["output_dir"])
        save_json(
            {"strategy": args.strategy, "runs": summaries},
            output_root / args.strategy / "all_seeds_summary.json",
        )


if __name__ == "__main__":
    main()
