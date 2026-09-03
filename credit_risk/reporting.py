"""Plots and serialized artifacts for completed model experiments."""

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "credit-risk-matplotlib-cache"),
)

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from credit_risk.regression import ALPHAS


def _save_confusion_matrices(
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (name, values) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            values,
            display_labels=["Normal", "Overdue"],
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(name)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_roc_curves(
    y_test: pd.Series,
    scores: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, values in scores.items():
        RocCurveDisplay.from_predictions(y_test, values, name=name, ax=axis)
    axis.plot([0, 1], [0, 1], "k--", label="Random")
    axis.set_title("Overdue Risk ROC Curves")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_feature_importance(
    feature_importance: dict[str, float],
    output_path: Path,
) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_importance.keys(),
            "importance": feature_importance.values(),
        }
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance, x="importance", y="feature", ax=axis)
    axis.set_title("Tuned Random Forest Feature Importance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_coefficient_paths(
    coefficients: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, model_name in zip(axes, ["Ridge", "Lasso"]):
        first_alpha = str(ALPHAS[0])
        feature_names = coefficients[model_name][first_alpha].keys()
        for feature_name in feature_names:
            values = [
                coefficients[model_name][str(alpha)][feature_name]
                for alpha in ALPHAS
            ]
            axis.plot(ALPHAS, values, marker="o", label=feature_name)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_title(f"{model_name} Coefficient Paths")
        axis.set_xlabel("Alpha")
        axis.set_ylabel("Coefficient in standardized feature space")
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_classification_artifacts(
    result: dict,
    y_test: pd.Series,
    output_dir: str | Path,
) -> None:
    """Save classification tables and plots from an experiment result."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table = result["predictions"]
    logistic_predictions = (table["logistic_probability"] >= 0.5).astype(int)
    tuned_predictions = (table["overdue_probability"] >= 0.5).astype(int)
    _save_confusion_matrices(
        y_test,
        {
            "Logistic Regression": logistic_predictions.to_numpy(),
            "Random Forest (Tuned)": tuned_predictions.to_numpy(),
        },
        destination / "confusion_matrix.png",
    )
    _save_roc_curves(
        y_test,
        {
            "Rule Baseline": table["rule_prediction"].to_numpy(dtype=float),
            "Logistic Regression": table["logistic_probability"].to_numpy(),
            "Random Forest": table["random_forest_probability"].to_numpy(),
            "Random Forest (Tuned)": table["overdue_probability"].to_numpy(),
        },
        destination / "roc_curve.png",
    )
    _save_feature_importance(
        result["feature_importance"],
        destination / "feature_importance.png",
    )
    table.to_csv(destination / "classification_predictions.csv", index=False)


def save_regression_artifacts(
    result: dict,
    y_test: pd.Series,
    output_dir: str | Path,
) -> None:
    """Save regression predictions and coefficient plots."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _save_coefficient_paths(
        result["coefficients"],
        destination / "regularization_coefficients.png",
    )
    prediction_table = pd.DataFrame(
        {
            "actual_credit_score": y_test.to_numpy(),
            "ridge_prediction": result["predictions"]["Ridge"].to_numpy(),
            "lasso_prediction": result["predictions"]["Lasso"].to_numpy(),
        }
    )
    prediction_table.to_csv(
        destination / "credit_score_predictions.csv",
        index=False,
    )


def metrics_report(result: dict) -> dict:
    """Remove non-serializable prediction objects from the report."""
    return {
        "data_distribution": result["data_distribution"],
        "classification": {
            key: value
            for key, value in result["classification"].items()
            if key != "predictions"
        },
        "regression": {
            key: value
            for key, value in result["regression"].items()
            if key != "predictions"
        },
    }


def save_metrics_report(result: dict, output_path: str | Path) -> None:
    """Serialize experiment metrics as UTF-8 JSON."""
    Path(output_path).write_text(
        json.dumps(metrics_report(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
