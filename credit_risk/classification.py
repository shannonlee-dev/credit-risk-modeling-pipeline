"""Classification model training and evaluation without file output."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from credit_risk.constants import (
    CV_FOLDS,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    DECISION_TREE_MODEL,
    LOGISTIC_REGRESSION_MODEL,
    RANDOM_FOREST_MODEL,
    RANDOM_STATE,
    RULE_BASELINE_MODEL,
    TUNED_RANDOM_FOREST_MODEL,
)
from credit_risk.preprocessing import build_preprocessor
from credit_risk.evaluation import apply_threshold, evaluate_classification
from credit_risk.results import FinalSelection


SELECTED_LOGISTIC_THRESHOLD = 0.45
SELECTED_RANDOM_FOREST_THRESHOLD = 0.33


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


def evaluate_classifier(
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
    batch_prediction_latency_ms: float,
) -> dict[str, float]:
    """Calculate classification metrics on an untouched test target."""
    y_values = np.asarray(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "batch_prediction_latency_ms": float(batch_prediction_latency_ms),
        "fn": int(np.sum((y_values == 1) & (predictions == 0))),
        "fp": int(np.sum((y_values == 0) & (predictions == 1))),
    }


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


def _threshold_sweep(
    y_train: pd.Series,
    oof_scores: np.ndarray,
    selected_threshold: float | None = None,
) -> pd.DataFrame:
    """Summarize Train-OOF outcomes without selecting a threshold."""
    y_values = y_train.to_numpy()
    rows = []
    for raw_threshold in np.linspace(0.30, 0.60, 31):
        threshold = float(np.round(raw_threshold, 2))
        predictions = oof_scores >= threshold
        rows.append(
            {
                "threshold": threshold,
                "is_baseline": threshold == DEFAULT_CLASSIFICATION_THRESHOLD,
                "is_selected": threshold == selected_threshold,
                "predicted_overdue": int(predictions.sum()),
                "tp": int(np.sum((y_values == 1) & predictions)),
                "fp": int(np.sum((y_values == 0) & predictions)),
                "fn": int(np.sum((y_values == 1) & ~predictions)),
                "precision": float(
                    precision_score(y_train, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_train, predictions, zero_division=0)),
                "f1": float(f1_score(y_train, predictions, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def _random_forest_saturation_analysis(
    forest: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    best_params: dict[str, object],
    latency_batch: pd.DataFrame,
    n_estimator_values: list[int],
) -> dict[str, dict[str, float]]:
    """Measure tree-count sensitivity with the selected non-tree settings."""
    fixed_params = {
        "model__max_depth": best_params["model__max_depth"],
        "model__min_samples_split": best_params["model__min_samples_split"],
    }
    saturation = {}
    for n_estimators in n_estimator_values:
        candidate = clone(forest).set_params(
            **fixed_params,
            model__n_estimators=n_estimators,
        )
        scores = cross_validate(
            candidate,
            x_train,
            y_train,
            scoring={"roc_auc": "roc_auc", "f1": "f1"},
            cv=cv,
            n_jobs=-1,
        )
        candidate.fit(x_train, y_train)
        saturation[str(n_estimators)] = {
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "fit_time_seconds": float(scores["fit_time"].mean()),
            "batch_prediction_latency_ms": float(
                _average_latency_ms(lambda: candidate.predict_proba(latency_batch))
            ),
        }
    return saturation


def _analyze_logistic_c_values(
    logistic: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    c_values: list[float],
) -> tuple[dict[str, dict[str, float]], float]:
    """Select logistic regularization using Train CV ROC-AUC only."""
    analysis = {}
    for c_value in c_values:
        scores = cross_validate(
            clone(logistic).set_params(model__C=c_value),
            x_train,
            y_train,
            scoring={"roc_auc": "roc_auc", "f1": "f1"},
            cv=cv,
            n_jobs=-1,
        )
        analysis[str(c_value)] = {
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
        }
    selected_c = min(
        c_values,
        key=lambda c_value: (-analysis[str(c_value)]["cv_roc_auc_mean"], c_value),
    )
    return analysis, selected_c


def _selected_threshold_metrics(sweep: pd.DataFrame) -> dict[str, float]:
    """Return the selected Train-OOF operating point."""
    selected = sweep.loc[sweep["is_selected"]].iloc[0]
    return {
        "threshold": float(selected["threshold"]),
        "oof_precision": float(selected["precision"]),
        "oof_recall": float(selected["recall"]),
        "oof_f1": float(selected["f1"]),
    }


def evaluate_final_classification(
    x_train: pd.DataFrame,
    x_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    selection: FinalSelection,
) -> dict:
    """Fit selected configurations on full Train and evaluate untouched Holdout."""
    logistic = build_logistic_classifier(selection.logistic_c)
    decision_tree = build_decision_tree_classifier()
    forest = build_random_forest_classifier(
        n_estimators=selection.random_forest_n_estimators or 100,
        max_depth=selection.random_forest_max_depth,
        min_samples_split=selection.random_forest_min_samples_split,
    )
    logistic.fit(x_train, y_train)
    decision_tree.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    batch = x_train.iloc[: min(2_000, len(x_train))]
    rule_predictions = x_holdout.apply(rule_based_predict, axis=1).to_numpy()
    logistic_scores = logistic.predict_proba(x_holdout)[:, 1]
    decision_tree_scores = decision_tree.predict_proba(x_holdout)[:, 1]
    forest_scores = forest.predict_proba(x_holdout)[:, 1]
    logistic_predictions = apply_threshold(
        logistic_scores,
        selection.logistic_threshold or DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    forest_predictions = apply_threshold(
        forest_scores,
        selection.random_forest_threshold or DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    decision_tree_predictions = apply_threshold(
        decision_tree_scores,
        DEFAULT_CLASSIFICATION_THRESHOLD,
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
        TUNED_RANDOM_FOREST_MODEL: _average_latency_ms(
            lambda: forest.predict_proba(batch)
        ),
    }
    predictions = {
        RULE_BASELINE_MODEL: rule_predictions,
        LOGISTIC_REGRESSION_MODEL: logistic_predictions,
        DECISION_TREE_MODEL: decision_tree_predictions,
        TUNED_RANDOM_FOREST_MODEL: forest_predictions,
    }
    scores = {
        RULE_BASELINE_MODEL: rule_predictions.astype(float),
        LOGISTIC_REGRESSION_MODEL: logistic_scores,
        DECISION_TREE_MODEL: decision_tree_scores,
        TUNED_RANDOM_FOREST_MODEL: forest_scores,
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
                "random_forest_probability": forest_scores,
                "random_forest_prediction": forest_predictions,
            }
        ),
        "latency_benchmark": {
            "source": "training_feature_batch",
            "batch_rows": len(batch),
            "repeats": 5,
        },
    }


def train_classification(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    grid: dict | None = None,
    fast: bool = False,
) -> dict:
    """Train classifiers; fast mode trims candidate ranges for smoke/CI runs."""
    logistic = Pipeline(
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
    decision_tree = Pipeline(
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
    forest = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    if fast:
        logistic_c_values = [0.01, 0.1]
        n_estimator_values = [50, 100]
        min_samples_split_values = [20, 40]
    else:
        logistic_c_values = [0.001, 0.003, 0.01, 0.03, 0.1]
        n_estimator_values = [25, 50, 100, 200, 300, 500]
        min_samples_split_values = [5, 10, 20, 40, 80]
    logistic_c_analysis, selected_logistic_c = _analyze_logistic_c_values(
        logistic,
        x_train,
        y_train,
        cv=cv,
        c_values=logistic_c_values,
    )
    logistic = logistic.set_params(model__C=selected_logistic_c)
    logistic.fit(x_train, y_train)
    decision_tree.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    if grid is None:
        parameter_grid = {
            "model__n_estimators": [100],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": min_samples_split_values,
        }
    else:
        parameter_grid = grid
    search = GridSearchCV(
        clone(forest),
        parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    tuned_forest = search.best_estimator_
    random_forest_grid_analysis = {
        (
            "max_depth="
            f"{params['model__max_depth']}, "
            "min_samples_split="
            f"{params['model__min_samples_split']}"
        ): {
            "cv_roc_auc_mean": float(mean_score),
            "cv_roc_auc_std": float(std_score),
        }
        for params, mean_score, std_score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        )
    }
    latency_batch = x_train.iloc[: min(2_000, len(x_train))]
    random_forest_saturation = _random_forest_saturation_analysis(
        forest,
        x_train,
        y_train,
        cv,
        search.best_params_,
        latency_batch,
        n_estimator_values,
    )
    tuned_forest_cv_f1 = cross_validate(
        tuned_forest,
        x_train,
        y_train,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    logistic_oof_scores = cross_val_predict(
        logistic,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold_sweep = _threshold_sweep(
        y_train,
        logistic_oof_scores,
        selected_threshold=SELECTED_LOGISTIC_THRESHOLD,
    )
    random_forest_oof_scores = cross_val_predict(
        tuned_forest,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    random_forest_threshold_sweep = _threshold_sweep(
        y_train,
        random_forest_oof_scores,
        selected_threshold=SELECTED_RANDOM_FOREST_THRESHOLD,
    )

    rule_predictions = x_test.apply(rule_based_predict, axis=1).to_numpy()
    logistic_scores = logistic.predict_proba(x_test)[:, 1]
    decision_tree_scores = decision_tree.predict_proba(x_test)[:, 1]
    forest_scores = forest.predict_proba(x_test)[:, 1]
    tuned_scores = tuned_forest.predict_proba(x_test)[:, 1]
    logistic_predictions = (
        logistic_scores >= SELECTED_LOGISTIC_THRESHOLD
    ).astype(int)
    decision_tree_predictions = (
        decision_tree_scores >= DEFAULT_CLASSIFICATION_THRESHOLD
    ).astype(int)
    forest_predictions = (
        forest_scores >= DEFAULT_CLASSIFICATION_THRESHOLD
    ).astype(int)
    tuned_predictions = (
        tuned_scores >= SELECTED_RANDOM_FOREST_THRESHOLD
    ).astype(int)

    latencies = {
        RULE_BASELINE_MODEL: _average_latency_ms(
            lambda: latency_batch.apply(rule_based_predict, axis=1).to_numpy()
        ),
        LOGISTIC_REGRESSION_MODEL: _average_latency_ms(
            lambda: logistic.predict_proba(latency_batch)
        ),
        DECISION_TREE_MODEL: _average_latency_ms(
            lambda: decision_tree.predict_proba(latency_batch)
        ),
        RANDOM_FOREST_MODEL: _average_latency_ms(
            lambda: forest.predict_proba(latency_batch)
        ),
        TUNED_RANDOM_FOREST_MODEL: _average_latency_ms(
            lambda: tuned_forest.predict_proba(latency_batch)
        ),
    }
    predictions = {
        RULE_BASELINE_MODEL: rule_predictions,
        LOGISTIC_REGRESSION_MODEL: logistic_predictions,
        DECISION_TREE_MODEL: decision_tree_predictions,
        RANDOM_FOREST_MODEL: forest_predictions,
        TUNED_RANDOM_FOREST_MODEL: tuned_predictions,
    }
    scores = {
        RULE_BASELINE_MODEL: rule_predictions.astype(float),
        LOGISTIC_REGRESSION_MODEL: logistic_scores,
        DECISION_TREE_MODEL: decision_tree_scores,
        RANDOM_FOREST_MODEL: forest_scores,
        TUNED_RANDOM_FOREST_MODEL: tuned_scores,
    }
    metrics = {
        name: evaluate_classifier(
            y_test,
            predictions[name],
            scores[name],
            latencies[name],
        )
        for name in predictions
    }
    prediction_table = pd.DataFrame(
        {
            "actual_is_overdue": y_test.to_numpy(),
            "rule_prediction": rule_predictions,
            "logistic_probability": logistic_scores,
            "logistic_prediction_selected": logistic_predictions,
            "decision_tree_probability": decision_tree_scores,
            "decision_tree_prediction": decision_tree_predictions,
            "random_forest_probability": forest_scores,
            "overdue_probability": tuned_scores,
            "tuned_rf_prediction_selected": tuned_predictions,
        }
    )
    return {
        "metrics": metrics,
        "logistic_cv": {
            "cv_f1_default_threshold_mean": logistic_c_analysis[
                str(selected_logistic_c)
            ]["cv_f1_mean"],
            "roc_auc_mean": logistic_c_analysis[str(selected_logistic_c)][
                "cv_roc_auc_mean"
            ],
        },
        "logistic_c_analysis": logistic_c_analysis,
        "selected_logistic_c": selected_logistic_c,
        "selected_classification_model": LOGISTIC_REGRESSION_MODEL,
        "selected_logistic_threshold": SELECTED_LOGISTIC_THRESHOLD,
        "selected_random_forest_threshold": SELECTED_RANDOM_FOREST_THRESHOLD,
        "threshold_selection": {
            LOGISTIC_REGRESSION_MODEL: _selected_threshold_metrics(
                threshold_sweep
            ),
            TUNED_RANDOM_FOREST_MODEL: _selected_threshold_metrics(
                random_forest_threshold_sweep
            ),
        },
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "best_cv_f1": float(tuned_forest_cv_f1["test_score"].mean()),
        "random_forest_grid_analysis": random_forest_grid_analysis,
        "random_forest_saturation": random_forest_saturation,
        "feature_importance": _feature_importance(tuned_forest),
        "logistic_threshold_sweep": threshold_sweep,
        "random_forest_threshold_sweep": random_forest_threshold_sweep,
        "latency_benchmark": {
            "source": "training_feature_batch",
            "batch_rows": len(latency_batch),
            "repeats": 5,
        },
        "predictions": prediction_table,
    }
