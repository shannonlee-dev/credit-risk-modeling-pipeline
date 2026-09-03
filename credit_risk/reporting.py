"""Plots and serialized artifacts for completed model experiments."""

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    tempfile.mkdtemp(prefix="credit-risk-matplotlib-"),
)

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from credit_risk.regression import ALPHAS


def _save_confusion_matrices(
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    columns = min(2, len(predictions))
    rows = int(np.ceil(len(predictions) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    for axis, (name, values) in zip(np.asarray(axes).reshape(-1), predictions.items()):
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


def _save_threshold_tradeoff(
    y_train: np.ndarray,
    scores: np.ndarray,
    default_threshold: float,
    f1_threshold: float,
    recall_scenarios: dict[str, dict],
    output_path: Path,
) -> None:
    thresholds = np.linspace(0, 1, 101)
    metrics = {
        "Precision": [
            precision_score(y_train, scores >= threshold, zero_division=0)
            for threshold in thresholds
        ],
        "Recall": [
            recall_score(y_train, scores >= threshold, zero_division=0)
            for threshold in thresholds
        ],
        "F1": [
            f1_score(y_train, scores >= threshold, zero_division=0)
            for threshold in thresholds
        ],
    }
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, values in metrics.items():
        axis.plot(thresholds, values, label=name)
    axis.axvline(default_threshold, color="black", linestyle="--", label="Default 0.5")
    axis.axvline(
        f1_threshold,
        color="tab:purple",
        linestyle="--",
        label="F1 reference",
    )
    for floor, scenario in recall_scenarios.items():
        axis.axvline(
            scenario["threshold"],
            linestyle=":",
            label=f"Recall ≥ {floor} CV",
        )
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Metric")
    axis.set_title("Tuned Random Forest Threshold Trade-off (Train OOF)")
    axis.legend()
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


def _save_random_forest_saturation_curve(
    saturation: dict[str, dict[str, float]],
    selected_n_estimators: int,
    output_path: Path,
) -> None:
    values = sorted((int(key), metrics) for key, metrics in saturation.items())
    n_estimators = [value for value, _ in values]
    means = [metrics["cv_roc_auc_mean"] for _, metrics in values]
    stds = [metrics["cv_roc_auc_std"] for _, metrics in values]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(n_estimators, means, marker="o", label="CV ROC-AUC mean")
    axis.fill_between(
        n_estimators,
        np.asarray(means) - np.asarray(stds),
        np.asarray(means) + np.asarray(stds),
        alpha=0.2,
        label="±1 CV standard deviation",
    )
    axis.axvline(
        selected_n_estimators,
        color="tab:orange",
        linestyle="--",
        label=f"GridSearchCV selected: {selected_n_estimators}",
    )
    axis.set_xlabel("Number of trees (n_estimators)")
    axis.set_ylabel("CV ROC-AUC")
    axis.set_title("Random Forest Tree Count Sensitivity (Train CV)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_logistic_c_curve(
    analysis: dict[str, dict[str, float]],
    selected_c: float,
    output_path: Path,
) -> None:
    """Plot Train-CV ROC-AUC across logistic regularization candidates."""
    values = sorted((float(c_value), metrics) for c_value, metrics in analysis.items())
    c_values = [value for value, _ in values]
    means = [metrics["cv_roc_auc_mean"] for _, metrics in values]
    stds = [metrics["cv_roc_auc_std"] for _, metrics in values]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(c_values, means, marker="o", label="CV ROC-AUC mean")
    axis.fill_between(
        c_values,
        np.asarray(means) - np.asarray(stds),
        np.asarray(means) + np.asarray(stds),
        alpha=0.2,
        label="±1 CV standard deviation",
    )
    axis.axvline(
        selected_c,
        color="tab:orange",
        linestyle="--",
        label=f"Selected C: {selected_c:g}",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Regularization inverse strength (C)")
    axis.set_ylabel("CV ROC-AUC")
    axis.set_title("Logistic Regression C Selection (Train CV)")
    axis.legend()
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
    _save_confusion_matrices(
        y_test,
        {
            "Logistic Regression (0.5)": table["logistic_prediction_default"].to_numpy(),
            "Random Forest (Tuned, 0.5)": table["tuned_rf_prediction_default"].to_numpy(),
        },
        destination / "confusion_matrix.png",
    )
    _save_confusion_matrices(
        y_test,
        {
            "Logistic (0.5)": table["logistic_prediction_default"].to_numpy(),
            "Logistic (Recall ≥ 0.90 CV, illustrative)": table[
                "logistic_prediction_illustrative_recall_090"
            ].to_numpy(),
            "Tuned RF (0.5)": table["tuned_rf_prediction_default"].to_numpy(),
            "Tuned RF (Recall ≥ 0.90 CV, illustrative)": table[
                "tuned_rf_prediction_illustrative_recall_090"
            ].to_numpy(),
        },
        destination / "confusion_matrix_threshold_comparison.png",
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
    _save_random_forest_saturation_curve(
        result["random_forest_saturation"],
        result["best_params"]["model__n_estimators"],
        destination / "random_forest_n_estimators_curve.png",
    )
    _save_logistic_c_curve(
        result["logistic_c_analysis"],
        result["selected_logistic_c"],
        destination / "logistic_c_curve.png",
    )
    rf_thresholds = result["threshold_analysis"]["Random Forest (Tuned)"]
    tradeoff = result["threshold_tradeoff"]
    _save_threshold_tradeoff(
        tradeoff["target"],
        tradeoff["scores"],
        rf_thresholds["default_threshold"],
        rf_thresholds["f1_reference"]["threshold"],
        rf_thresholds["recall_scenarios"],
        destination / "threshold_tradeoff.png",
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
            if key not in {"predictions", "threshold_tradeoff"}
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
