import pytest

from continual_dl.metrics import ContinualMetricTracker, classification_metrics


def test_classification_and_forgetting_metrics():
    metrics = classification_metrics(
        targets=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        seen_class_ids=[0, 1],
        class_names=["dog", "cat"],
    )
    assert metrics["per_class_accuracy"] == {"dog": 0.5, "cat": 1.0}
    assert metrics["average_accuracy"] == pytest.approx(0.75)

    tracker = ContinualMetricTracker()
    tracker.add(0, {"average_accuracy": 0.9, "per_class_accuracy": {"dog": 0.9}})
    tracker.add(1, {"average_accuracy": 0.7, "per_class_accuracy": {"dog": 0.6, "cat": 0.8}})
    summary = tracker.summary()
    assert summary["per_class_forgetting"]["dog"] == pytest.approx(0.3)
    assert summary["mean_backward_transfer"] == pytest.approx(-0.3)
