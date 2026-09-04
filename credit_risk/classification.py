"""Classification builders, final evaluation, and legacy result adaptation."""

from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from credit_risk.constants import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    DECISION_TREE_MODEL,
    LOGISTIC_REGRESSION_MODEL,
    RANDOM_FOREST_MODEL,
    RANDOM_STATE,
    RULE_BASELINE_MODEL,
    TUNED_RANDOM_FOREST_MODEL,
)
from credit_risk.evaluation import apply_threshold, evaluate_classification
from credit_risk.preprocessing import build_preprocessor
from credit_risk.results import FinalSelection


def build_logistic_classifier(c: float | None = None) -> Pipeline:
    """Build the project's leakage-safe logistic classifier."""
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline if c is None else pipeline.set_params(model__C=c)


def build_random_forest_classifier(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
) -> Pipeline:
    """Build the project's leakage-safe random forest classifier."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_decision_tree_classifier() -> Pipeline:
    """Build the single-tree comparison baseline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def rule_based_predict(row: pd.Series) -> int:
    """Classify overdue risk with six explicit business rules."""
    if row["overdue_count_6m"] >= 2:
        return 1
    if row["debt_ratio"] > 0.80 and row["annual_income"] < 4_500:
        return 1
    if row["annual_income"] < 2_500:
        return 1
    if row["spending_score"] > 90 and row["debt_ratio"] > 0.70:
        return 1
    if row["credit_card_count"] >= 8 and row["debt_ratio"] > 0.65:
        return 1
    if row["age"] < 25 and row["debt_ratio"] > 0.75:
        return 1
    return 0


def _average_latency_ms(predictor, repeats: int = 5) -> float:
    predictor()
    started = perf_counter()
    for _ in range(repeats):
        predictor()
    return (perf_counter() - started) * 1_000 / repeats


def _feature_importance(model: Pipeline) -> dict[str, float]:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in preprocessor.get_feature_names_out()
    ]
    importances = model.named_steps["model"].feature_importances_
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        row.feature: float(row.importance)
        for row in importance.itertuples(index=False)
    }


def evaluate_final_classification(
    x_train: pd.DataFrame,
    x_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    selection: FinalSelection,
) -> dict:
    """Fit fixed configurations on full Train and evaluate untouched Holdout."""
    logistic = build_logistic_classifier(selection.logistic_c)
    decision_tree = build_decision_tree_classifier()
    baseline_forest = build_random_forest_classifier()
    tuned_forest = build_random_forest_classifier(
        n_estimators=selection.random_forest_n_estimators or 100,
        max_depth=selection.random_forest_max_depth,
        min_samples_split=selection.random_forest_min_samples_split,
    )
    for model in (logistic, decision_tree, baseline_forest, tuned_forest):
        model.fit(x_train, y_train)

    batch = x_train.iloc[: min(2_000, len(x_train))]
    rule_predictions = x_holdout.apply(rule_based_predict, axis=1).to_numpy()
    logistic_scores = logistic.predict_proba(x_holdout)[:, 1]
    decision_tree_scores = decision_tree.predict_proba(x_holdout)[:, 1]
    baseline_forest_scores = baseline_forest.predict_proba(x_holdout)[:, 1]
    tuned_forest_scores = tuned_forest.predict_proba(x_holdout)[:, 1]
    logistic_predictions = apply_threshold(
        logistic_scores,
        selection.logistic_threshold or DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    decision_tree_predictions = apply_threshold(
        decision_tree_scores,
        DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    baseline_forest_predictions = apply_threshold(
        baseline_forest_scores,
        DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    tuned_forest_predictions = apply_threshold(
        tuned_forest_scores,
        selection.random_forest_threshold or DEFAULT_CLASSIFICATION_THRESHOLD,
    )

    latencies = {
        RULE_BASELINE_MODEL: _average_latency_ms(
            lambda: batch.apply(rule_based_predict, axis=1).to_numpy()
        ),
        LOGISTIC_REGRESSION_MODEL: _average_latency_ms(
            lambda: logistic.predict_proba(batch)
        ),
        DECISION_TREE_MODEL: _average_latency_ms(
            lambda: decision_tree.predict_proba(batch)
        ),
        RANDOM_FOREST_MODEL: _average_latency_ms(
            lambda: baseline_forest.predict_proba(batch)
        ),
        TUNED_RANDOM_FOREST_MODEL: _average_latency_ms(
            lambda: tuned_forest.predict_proba(batch)
        ),
    }
    predictions = {
        RULE_BASELINE_MODEL: rule_predictions,
        LOGISTIC_REGRESSION_MODEL: logistic_predictions,
        DECISION_TREE_MODEL: decision_tree_predictions,
        RANDOM_FOREST_MODEL: baseline_forest_predictions,
        TUNED_RANDOM_FOREST_MODEL: tuned_forest_predictions,
    }
    scores = {
        RULE_BASELINE_MODEL: rule_predictions.astype(float),
        LOGISTIC_REGRESSION_MODEL: logistic_scores,
        DECISION_TREE_MODEL: decision_tree_scores,
        RANDOM_FOREST_MODEL: baseline_forest_scores,
        TUNED_RANDOM_FOREST_MODEL: tuned_forest_scores,
    }
    return {
        "selected_model": selection.selected_model,
        "selected_threshold": (
            selection.logistic_threshold
            if selection.selected_model == LOGISTIC_REGRESSION_MODEL
            else selection.random_forest_threshold
        ),
        "metrics": {
            name: evaluate_classification(
                y_holdout, predictions[name], scores[name], latencies[name]
            )
            for name in predictions
        },
        "predictions": pd.DataFrame(
            {
                "actual_is_overdue": y_holdout.to_numpy(),
                "rule_prediction": rule_predictions,
                "logistic_probability": logistic_scores,
                "logistic_prediction": logistic_predictions,
                "decision_tree_probability": decision_tree_scores,
                "decision_tree_prediction": decision_tree_predictions,
                "baseline_random_forest_probability": baseline_forest_scores,
                "baseline_random_forest_prediction": baseline_forest_predictions,
                "random_forest_probability": tuned_forest_scores,
                "random_forest_prediction": tuned_forest_predictions,
            }
        ),
        "latency_benchmark": {
            "source": "training_feature_batch",
            "batch_rows": len(batch),
            "repeats": 5,
        },
    }


def _annotate_threshold_sweep(
    sweep: pd.DataFrame,
    selected_threshold: float,
) -> pd.DataFrame:
    annotated = sweep.copy()
    annotated.insert(
        1,
        "is_baseline",
        annotated["threshold"] == DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    annotated.insert(
        2,
        "is_selected",
        annotated["threshold"] == selected_threshold,
    )
    return annotated


def _selected_threshold_metrics(
    sweep: pd.DataFrame,
    selected_threshold: float,
) -> dict[str, float]:
    selected = sweep.loc[sweep["threshold"] == selected_threshold].iloc[0]
    return {
        "threshold": float(selected["threshold"]),
        "oof_precision": float(selected["precision"]),
        "oof_recall": float(selected["recall"]),
        "oof_f1": float(selected["f1"]),
    }


def adapt_legacy_classification_result(
    experiment: dict,
    final: dict,
    selection: FinalSelection,
) -> dict:
    """Adapt two-stage outputs to the historical classification dictionary."""
    logistic_sweep = _annotate_threshold_sweep(
        experiment["logistic_threshold_sweep"],
        selection.logistic_threshold,
    )
    forest_sweep = _annotate_threshold_sweep(
        experiment["random_forest_threshold_sweep"],
        selection.random_forest_threshold,
    )
    table = final["predictions"]
    selected_logistic = experiment["logistic_c_analysis"][
        str(experiment["selected_logistic_c"])
    ]
    return {
        "metrics": final["metrics"],
        "logistic_cv": {
            "cv_f1_default_threshold_mean": selected_logistic["cv_f1_mean"],
            "roc_auc_mean": selected_logistic["cv_roc_auc_mean"],
        },
        "logistic_c_analysis": experiment["logistic_c_analysis"],
        "selected_logistic_c": experiment["selected_logistic_c"],
        "selected_classification_model": selection.selected_model,
        "selected_logistic_threshold": selection.logistic_threshold,
        "selected_random_forest_threshold": selection.random_forest_threshold,
        "threshold_selection": {
            LOGISTIC_REGRESSION_MODEL: _selected_threshold_metrics(
                experiment["logistic_threshold_sweep"],
                selection.logistic_threshold,
            ),
            TUNED_RANDOM_FOREST_MODEL: _selected_threshold_metrics(
                experiment["random_forest_threshold_sweep"],
                selection.random_forest_threshold,
            ),
        },
        "best_params": experiment["best_params"],
        "best_cv_roc_auc": experiment["best_cv_roc_auc"],
        "best_cv_f1": experiment["best_cv_f1"],
        "random_forest_grid_analysis": experiment[
            "random_forest_grid_analysis"
        ],
        "random_forest_saturation": experiment["random_forest_saturation"],
        "feature_importance": experiment["feature_importance"],
        "logistic_threshold_sweep": logistic_sweep,
        "random_forest_threshold_sweep": forest_sweep,
        "latency_benchmark": final["latency_benchmark"],
        "predictions": pd.DataFrame(
            {
                "actual_is_overdue": table["actual_is_overdue"],
                "rule_prediction": table["rule_prediction"],
                "logistic_probability": table["logistic_probability"],
                "logistic_prediction_selected": table["logistic_prediction"],
                "decision_tree_probability": table["decision_tree_probability"],
                "decision_tree_prediction": table["decision_tree_prediction"],
                "random_forest_probability": table[
                    "baseline_random_forest_probability"
                ],
                "overdue_probability": table["random_forest_probability"],
                "tuned_rf_prediction_selected": table[
                    "random_forest_prediction"
                ],
            }
        ),
    }


def train_classification(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    grid: dict | None = None,
    fast: bool = False,
) -> dict:
    """Compatibility facade over Train-only experiment and final evaluation."""
    from credit_risk.experiments.classification import (
        run_classification_experiment,
    )
    from credit_risk.experiments.config import FULL_EXPERIMENT, SMOKE_EXPERIMENT

    profile = SMOKE_EXPERIMENT if fast else FULL_EXPERIMENT
    config = profile.classification
    if grid is not None:
        config = replace(
            config,
            rf_grid_n_estimators_values=tuple(grid["model__n_estimators"]),
            rf_max_depth_values=tuple(grid["model__max_depth"]),
            rf_min_samples_split_values=tuple(
                grid["model__min_samples_split"]
            ),
        )
    experiment = run_classification_experiment(x_train, y_train, config)
    best = experiment["best_params"]
    selection = FinalSelection(
        selected_model=LOGISTIC_REGRESSION_MODEL,
        logistic_c=experiment["selected_logistic_c"],
        logistic_threshold=0.45,
        random_forest_n_estimators=best["model__n_estimators"],
        random_forest_max_depth=best["model__max_depth"],
        random_forest_min_samples_split=best["model__min_samples_split"],
        random_forest_threshold=0.33,
        ridge_alpha=1.0,
        lasso_alpha=0.1,
    )
    final = evaluate_final_classification(
        x_train,
        x_test,
        y_train,
        y_test,
        selection,
    )
    return adapt_legacy_classification_result(experiment, final, selection)
