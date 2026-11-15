"""Classification and calibration metrics."""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from .types import Scores


def ece(probs: npt.ArrayLike, y: npt.ArrayLike, bins: int = 10) -> float:
    probabilities = np.asarray(probs)
    targets = np.asarray(y)
    conf = probabilities.max(axis=-1)
    pred = probabilities.argmax(-1)
    out = 0.0

    for i in range(bins):
        mask = (conf >= i / bins) & (
            conf < (i + 1) / bins if i < bins - 1 else conf <= 1
        )

        if mask.any():
            out += mask.mean() * abs(
                (pred[mask] == targets[mask]).mean() - conf[mask].mean()
            )

    return float(out)


def score(
    probs: npt.ArrayLike, y: npt.ArrayLike, labels: list[str] | None = None
) -> Scores:
    probabilities = np.asarray(probs, dtype=float)
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    targets = np.asarray(y, dtype=int)
    pred = probabilities.argmax(1)

    return {
        "n": len(targets),
        "accuracy": float(accuracy_score(targets, pred)),
        "macro_f1": float(
            f1_score(
                targets,
                pred,
                labels=list(range(probabilities.shape[1])),
                average="macro",
                zero_division=0,
            )
        ),
        "nll": float(
            log_loss(targets, probabilities, labels=list(range(probabilities.shape[1])))
        ),
        "brier": float(
            np.mean(
                np.sum(
                    (probabilities - np.eye(probabilities.shape[1])[targets]) ** 2,
                    axis=1,
                )
            )
        ),
        "ece_10": ece(probabilities, targets),
        "confusion_matrix": confusion_matrix(
            targets, pred, labels=list(range(probabilities.shape[1]))
        ).tolist(),
    }
