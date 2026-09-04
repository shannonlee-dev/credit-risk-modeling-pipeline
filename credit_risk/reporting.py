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
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from credit_risk.constants import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    DECISION_TREE_MODEL,
    LOGISTIC_REGRESSION_MODEL,
    RANDOM_FOREST_MODEL,
    REGRESSION_ALPHAS,
    REGRESSION_MODELS,
    RULE_BASELINE_MODEL,
    TUNED_RANDOM_FOREST_MODEL,
)


_PLOT_DPI = 150


def _classification_confusion_predictions(
    result: dict,
) -> dict[str, np.ndarray]:
    """Build confusion-matrix labels and predictions from selected thresholds."""
    table = result["predictions"]
    return {
        (
            f"{LOGISTIC_REGRESSION_MODEL} "
            f"(selected {result['selected_logistic_threshold']:.2f})"
        ): table["logistic_prediction_selected"].to_numpy(),
        (
            f"{RANDOM_FOREST_MODEL} (Tuned, selected "
            f"{result['selected_random_forest_threshold']:.2f})"
        ): table["tuned_rf_prediction_selected"].to_numpy(),
    }


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
    figure.savefig(output_path, dpi=_PLOT_DPI)
    plt.close(figure)


def _save_threshold_sweep(
    sweep: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for metric in ["precision", "recall", "f1"]:
        axis.plot(
            sweep["threshold"],
            sweep[metric],
            marker="o",
            markersize=3,
            label=metric.capitalize(),
        )
    axis.axvline(
        DEFAULT_CLASSIFICATION_THRESHOLD,
        color="black",
        linestyle="--",
        label=f"Baseline {DEFAULT_CLASSIFICATION_THRESHOLD:.2f}",
    )
    if sweep["is_selected"].any():
        selected_threshold = sweep.loc[
            sweep["is_selected"], "threshold"
        ].iloc[0]
        axis.axvline(
            selected_threshold,
            color="tab:orange",
            linestyle="--",
            label=f"Selected {selected_threshold:.2f}",
        )
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("OOF metric")
    axis.set_title(f"{model_name} Threshold Sweep (Train OOF)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=_PLOT_DPI)
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
    figure.savefig(output_path, dpi=_PLOT_DPI)
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
    figure.savefig(output_path, dpi=_PLOT_DPI)
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
        label=f"Cost/performance selected: {selected_n_estimators}",
    )
    axis.set_xlabel("Number of trees (n_estimators)")
    axis.set_ylabel("CV ROC-AUC")
    axis.set_title("Random Forest Tree Count Sensitivity (Train CV)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=_PLOT_DPI)
    plt.close(figure)


def _save_coefficient_paths(
    coefficients: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, model_name in zip(axes, REGRESSION_MODELS):
        first_alpha = str(REGRESSION_ALPHAS[0])
        feature_names = coefficients[model_name][first_alpha].keys()
        for feature_name in feature_names:
            values = [
                coefficients[model_name][str(alpha)][feature_name]
                for alpha in REGRESSION_ALPHAS
            ]
            axis.plot(REGRESSION_ALPHAS, values, marker="o", label=feature_name)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_title(f"{model_name} Coefficient Paths")
        axis.set_xlabel("Alpha")
        axis.set_ylabel("Coefficient in standardized feature space")
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=_PLOT_DPI)
    plt.close(figure)


def save_classification_artifacts(
    result: dict,
    y_test: pd.Series,
    output_dir: str | Path,
) -> None:
    """Save legacy classification output into separated stage directories."""
    experiment_destination = Path(output_dir) / "experiment"
    final_destination = Path(output_dir) / "final"
    experiment_destination.mkdir(parents=True, exist_ok=True)
    final_destination.mkdir(parents=True, exist_ok=True)
    table = result["predictions"]
    _save_confusion_matrices(
        y_test,
        _classification_confusion_predictions(result),
        final_destination / "confusion_matrix.png",
    )
    _save_roc_curves(
        y_test,
        {
            RULE_BASELINE_MODEL: table["rule_prediction"].to_numpy(dtype=float),
            LOGISTIC_REGRESSION_MODEL: table["logistic_probability"].to_numpy(),
            DECISION_TREE_MODEL: table["decision_tree_probability"].to_numpy(),
            RANDOM_FOREST_MODEL: table["random_forest_probability"].to_numpy(),
            TUNED_RANDOM_FOREST_MODEL: table["overdue_probability"].to_numpy(),
        },
        final_destination / "roc_curve.png",
    )
    _save_feature_importance(
        result["feature_importance"],
        experiment_destination / "feature_importance.png",
    )
    _save_random_forest_saturation_curve(
        result["random_forest_saturation"],
        result["best_params"]["model__n_estimators"],
        experiment_destination / "random_forest_n_estimators_curve.png",
    )
    result["logistic_threshold_sweep"].to_csv(
        experiment_destination / "logistic_threshold_sweep.csv",
        index=False,
        float_format="%.6f",
    )
    _save_threshold_sweep(
        result["logistic_threshold_sweep"],
        experiment_destination / "logistic_threshold_sweep.png",
        LOGISTIC_REGRESSION_MODEL,
    )
    result["random_forest_threshold_sweep"].to_csv(
        experiment_destination / "random_forest_threshold_sweep.csv",
        index=False,
        float_format="%.6f",
    )
    _save_threshold_sweep(
        result["random_forest_threshold_sweep"],
        experiment_destination / "random_forest_threshold_sweep.png",
        "Tuned Random Forest",
    )
    pd.DataFrame(
        [
            {
                "Model": "Rule Baseline",
                "Accuracy": result["metrics"][RULE_BASELINE_MODEL]["accuracy"],
                "F1-Score": result["metrics"][RULE_BASELINE_MODEL]["f1"],
                "AUC": result["metrics"][RULE_BASELINE_MODEL]["roc_auc"],
            },
            {
                "Model": "Logistic Regression",
                "Accuracy": result["metrics"][LOGISTIC_REGRESSION_MODEL]["accuracy"],
                "F1-Score": result["metrics"][LOGISTIC_REGRESSION_MODEL]["f1"],
                "AUC": result["metrics"][LOGISTIC_REGRESSION_MODEL]["roc_auc"],
            },
            {
                "Model": "Decision Tree",
                "Accuracy": result["metrics"][DECISION_TREE_MODEL]["accuracy"],
                "F1-Score": result["metrics"][DECISION_TREE_MODEL]["f1"],
                "AUC": result["metrics"][DECISION_TREE_MODEL]["roc_auc"],
            },
            {
                "Model": "Random Forest",
                "Accuracy": result["metrics"][RANDOM_FOREST_MODEL]["accuracy"],
                "F1-Score": result["metrics"][RANDOM_FOREST_MODEL]["f1"],
                "AUC": result["metrics"][RANDOM_FOREST_MODEL]["roc_auc"],
            },
            {
                "Model": "Tuned Random Forest",
                "Accuracy": result["metrics"][TUNED_RANDOM_FOREST_MODEL]["accuracy"],
                "F1-Score": result["metrics"][TUNED_RANDOM_FOREST_MODEL]["f1"],
                "AUC": result["metrics"][TUNED_RANDOM_FOREST_MODEL]["roc_auc"],
            },
        ]
    ).to_csv(
        final_destination / "classification_metrics_comparison.csv",
        index=False,
    )
    table.to_csv(
        final_destination / "classification_predictions.csv",
        index=False,
    )


def save_regression_artifacts(
    result: dict,
    y_test: pd.Series,
    output_dir: str | Path,
) -> None:
    """Save legacy regression output into separated stage directories."""
    experiment_destination = Path(output_dir) / "experiment"
    final_destination = Path(output_dir) / "final"
    experiment_destination.mkdir(parents=True, exist_ok=True)
    final_destination.mkdir(parents=True, exist_ok=True)
    _save_coefficient_paths(
        result["coefficients"],
        experiment_destination / "regularization_coefficients.png",
    )
    prediction_table = pd.DataFrame(
        {
            "actual_credit_score": y_test.to_numpy(),
            "ridge_prediction": result["predictions"]["Ridge"].to_numpy(),
            "lasso_prediction": result["predictions"]["Lasso"].to_numpy(),
        }
    )
    prediction_table.to_csv(
        final_destination / "credit_score_predictions.csv",
        index=False,
    )


def _without_timing(value):
    if isinstance(value, dict):
        return {
            key: _without_timing(child)
            for key, child in value.items()
            if key not in {"batch_prediction_latency_ms", "fit_time_seconds"}
        }
    if isinstance(value, list):
        return [_without_timing(child) for child in value]
    return value


def metrics_report(result: dict) -> dict:
    """Serialize stable evaluation data and one benchmark summary."""
    classification = result["classification"]
    saturation = classification["random_forest_saturation"]
    return {
        "data_distribution": result["data_distribution"],
        "classification": _without_timing({
            key: value
            for key, value in classification.items()
            if key
            not in {
                "predictions",
                "logistic_threshold_sweep",
                "random_forest_threshold_sweep",
                "latency_benchmark",
            }
        }),
        "regression": {
            key: value
            for key, value in result["regression"].items()
            if key != "predictions"
        },
        "benchmark": {
            **classification["latency_benchmark"],
            "model_prediction_latency_ms": {
                name: metrics["batch_prediction_latency_ms"]
                for name, metrics in classification["metrics"].items()
            },
            "random_forest_tree_count": {
                n_estimators: {
                    "fit_time_seconds": values["fit_time_seconds"],
                    "batch_prediction_latency_ms": values[
                        "batch_prediction_latency_ms"
                    ],
                }
                for n_estimators, values in saturation.items()
            },
        },
    }


def save_metrics_report(result: dict, output_path: str | Path) -> None:
    """Serialize experiment metrics as UTF-8 JSON."""
    Path(output_path).write_text(
        json.dumps(metrics_report(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_experiment_artifacts(result: dict, output_dir: str | Path) -> None:
    """Write Train-only experiment figures without final-evaluation outputs."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    classification = result["classification"]
    for key, filename, model_name in [
        ("logistic_threshold_sweep", "logistic_threshold_sweep.png", LOGISTIC_REGRESSION_MODEL),
        ("random_forest_threshold_sweep", "random_forest_threshold_sweep.png", "Tuned Random Forest"),
    ]:
        sweep = classification[key].copy()
        sweep["is_baseline"] = sweep["threshold"] == DEFAULT_CLASSIFICATION_THRESHOLD
        sweep["is_selected"] = False
        _save_threshold_sweep(sweep, destination / filename, model_name)
    _save_feature_importance(classification["feature_importance"], destination / "feature_importance.png")
    _save_random_forest_saturation_curve(
        classification["random_forest_saturation"],
        classification["best_params"]["model__n_estimators"],
        destination / "random_forest_n_estimators_curve.png",
    )
    _save_coefficient_paths(
        result["regression"]["coefficients"],
        destination / "regularization_coefficients.png",
    )


def save_final_artifacts(result: dict, output_dir: str | Path) -> None:
    """Write Holdout-only plots and row-level predictions."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table = result["classification"]["predictions"]
    _save_confusion_matrices(
        table["actual_is_overdue"],
        {
            "Logistic Regression": table["logistic_prediction"].to_numpy(),
            "Random Forest (Tuned)": table["random_forest_prediction"].to_numpy(),
        },
        destination / "confusion_matrix.png",
    )
    _save_roc_curves(
        table["actual_is_overdue"],
        {
            RULE_BASELINE_MODEL: table["rule_prediction"].to_numpy(dtype=float),
            LOGISTIC_REGRESSION_MODEL: table["logistic_probability"].to_numpy(),
            DECISION_TREE_MODEL: table["decision_tree_probability"].to_numpy(),
            RANDOM_FOREST_MODEL: table[
                "baseline_random_forest_probability"
            ].to_numpy(),
            TUNED_RANDOM_FOREST_MODEL: table["random_forest_probability"].to_numpy(),
        },
        destination / "roc_curve.png",
    )
    labels = {
        RULE_BASELINE_MODEL: "Rule Baseline",
        LOGISTIC_REGRESSION_MODEL: "Logistic Regression",
        DECISION_TREE_MODEL: "Decision Tree",
        RANDOM_FOREST_MODEL: "Random Forest",
        TUNED_RANDOM_FOREST_MODEL: "Tuned Random Forest",
    }
    pd.DataFrame(
        [
            {
                "Model": labels[name],
                "Accuracy": metrics["accuracy"],
                "F1-Score": metrics["f1"],
                "AUC": metrics["roc_auc"],
            }
            for name, metrics in result["classification"]["metrics"].items()
        ]
    ).to_csv(destination / "classification_metrics_comparison.csv", index=False)
