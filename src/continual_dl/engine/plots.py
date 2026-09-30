"""Non-interactive experiment plots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_confusion_matrix(
    matrix: Sequence[Sequence[int]], class_names: Sequence[str], output_path: str | Path
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        np.asarray(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
