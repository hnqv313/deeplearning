"""Per-stage classification metrics and forgetting summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    seen_class_ids: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    labels = [int(value) for value in seen_class_ids]
    matrix = confusion_matrix(targets, predictions, labels=labels)
    per_class: dict[str, float] = {}
    for row_index, class_id in enumerate(labels):
        denominator = int(matrix[row_index].sum())
        accuracy = float(matrix[row_index, row_index] / denominator) if denominator else 0.0
        per_class[class_names[class_id]] = accuracy
    average_accuracy = float(np.mean(list(per_class.values()))) if per_class else 0.0
    overall_accuracy = float(np.mean(np.asarray(targets) == np.asarray(predictions)))
    return {
        "overall_accuracy": overall_accuracy,
        "average_accuracy": average_accuracy,
        "per_class_accuracy": per_class,
        "confusion_matrix": matrix.tolist(),
        "seen_class_ids": labels,
    }


@dataclass
class ContinualMetricTracker:
    stages: list[dict] = field(default_factory=list)

    def add(self, stage_id: int, metrics: dict) -> None:
        self.stages.append({"stage_id": int(stage_id), **metrics})

    def summary(self) -> dict:
        if not self.stages:
            return {}
        final = self.stages[-1]["per_class_accuracy"]
        class_histories: dict[str, list[float]] = {}
        for stage in self.stages:
            for class_name, accuracy in stage["per_class_accuracy"].items():
                class_histories.setdefault(class_name, []).append(float(accuracy))

        forgetting = {
            class_name: max(history) - history[-1]
            for class_name, history in class_histories.items()
            if len(history) > 1
        }
        backward_transfer = {
            class_name: history[-1] - history[0]
            for class_name, history in class_histories.items()
            if len(history) > 1
        }
        return {
            "final_average_accuracy": float(self.stages[-1]["average_accuracy"]),
            "average_incremental_accuracy": mean(
                float(stage["average_accuracy"]) for stage in self.stages
            ),
            "mean_forgetting": mean(forgetting.values()) if forgetting else 0.0,
            "mean_backward_transfer": mean(backward_transfer.values())
            if backward_transfer
            else 0.0,
            "per_class_forgetting": forgetting,
            "per_class_backward_transfer": backward_transfer,
            "final_per_class_accuracy": final,
            "stages": self.stages,
        }
