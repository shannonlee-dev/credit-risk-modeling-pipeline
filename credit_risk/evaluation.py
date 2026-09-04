"""Pure prediction evaluation helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    recall_score,
    precision_score,
    roc_auc_score,
    root_mean_squared_error,
)


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert risk scores to binary predictions at a fixed threshold."""
    return (np.asarray(scores) >= threshold).astype(int)


def evaluate_thresholds(
    y_true: pd.Series,
    scores: np.ndarray,
    thresholds: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Calculate threshold operating points without policy annotations."""
    actual = np.asarray(y_true)
    rows = []
    for threshold in thresholds:
        value = float(np.round(threshold, 2))
        predictions = apply_threshold(scores, value)
        rows.append(
            {
                "threshold": value,
                "predicted_overdue": int(predictions.sum()),
                "tp": int(np.sum((actual == 1) & (predictions == 1))),
                "fp": int(np.sum((actual == 0) & (predictions == 1))),
                "fn": int(np.sum((actual == 1) & (predictions == 0))),
                "precision": float(
                    precision_score(actual, predictions, zero_division=0)
                ),
                "recall": float(recall_score(actual, predictions, zero_division=0)),
                "f1": float(f1_score(actual, predictions, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_classification(
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
    batch_prediction_latency_ms: float | None = None,
) -> dict[str, float | int]:
    """Calculate classification metrics for already-generated predictions."""
    actual = np.asarray(y_true)
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "fn": int(np.sum((actual == 1) & (predictions == 0))),
        "fp": int(np.sum((actual == 0) & (predictions == 1))),
    }
    if batch_prediction_latency_ms is not None:
        metrics["batch_prediction_latency_ms"] = float(batch_prediction_latency_ms)
    return metrics


def evaluate_regression(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics for raw predictions."""
    return {
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }
