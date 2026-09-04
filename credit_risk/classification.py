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
    AGE_COLUMN,
    ANNUAL_INCOME_COLUMN,
    CREDIT_CARD_COUNT_COLUMN,
    CV_FOLDS,
    DEBT_RATIO_COLUMN,
    DEFAULT_CLASSIFICATION_THRESHOLD as DEFAULT_THRESHOLD,
    DECISION_TREE_MODEL,
    LOGISTIC_REGRESSION_MODEL,
    OVERDUE_COUNT_COLUMN,
    RANDOM_FOREST_MODEL,
    RANDOM_STATE,
    RULE_BASELINE_MODEL,
    SPENDING_SCORE_COLUMN,
    TUNED_RANDOM_FOREST_MODEL,
)
from credit_risk.preprocessing import build_preprocessor
from credit_risk.evaluation import apply_threshold, evaluate_classification
from credit_risk.results import FinalSelection


SELECTED_LOGISTIC_THRESHOLD = 0.45
SELECTED_RANDOM_FOREST_THRESHOLD = 0.33
SELECTED_CLASSIFICATION_MODEL = LOGISTIC_REGRESSION_MODEL
SWEEP_THRESHOLDS = np.linspace(0.30, 0.60, 31)
LOGISTIC_C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
N_ESTIMATOR_VALUES = [25, 50, 100, 200, 300, 500]
RF_MAX_DEPTH_VALUES = [None, 8, 16]
RF_MIN_SAMPLES_SPLIT_VALUES = [5, 10, 20, 40, 80]
FAST_LOGISTIC_C_VALUES = [0.01, 0.1]
FAST_N_ESTIMATOR_VALUES = [50, 100]
FAST_RF_MIN_SAMPLES_SPLIT_VALUES = [20, 40]
_DEFAULT_RF_N_ESTIMATORS = 100
RF_LOCAL_REFINEMENT_GRID = {
    "model__n_estimators": [_DEFAULT_RF_N_ESTIMATORS],
    "model__max_depth": RF_MAX_DEPTH_VALUES,
    "model__min_samples_split": RF_MIN_SAMPLES_SPLIT_VALUES,
}

_OVERDUE_COUNT_THRESHOLD = 2
_HIGH_DEBT_RATIO = 0.80
_MODERATE_DEBT_RATIO = 0.70
_CARD_COUNT_DEBT_RATIO = 0.65
_YOUNG_BORROWER_DEBT_RATIO = 0.75
_LOW_INCOME = 4_500
_VERY_LOW_INCOME = 2_500
_HIGH_SPENDING_SCORE = 90
_HIGH_CREDIT_CARD_COUNT = 8
_YOUNG_AGE = 25
_LOGISTIC_MAX_ITERATIONS = 2_000
_LATENCY_REPEATS = 5
_LATENCY_BATCH_SIZE = 2_000
_MILLISECONDS_PER_SECOND = 1_000


def build_logistic_classifier(c: float | None = None) -> Pipeline:
    """Build the project's leakage-safe logistic classifier."""
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=_LOGISTIC_MAX_ITERATIONS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline if c is None else pipeline.set_params(model__C=c)


def build_random_forest_classifier(
    n_estimators: int = _DEFAULT_RF_N_ESTIMATORS,
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


def rule_based_predict(row: pd.Series) -> int:
    """Classify overdue risk with six explicit business rules."""
    if row[OVERDUE_COUNT_COLUMN] >= _OVERDUE_COUNT_THRESHOLD:
        return 1
    if (
        row[DEBT_RATIO_COLUMN] > _HIGH_DEBT_RATIO
        and row[ANNUAL_INCOME_COLUMN] < _LOW_INCOME
    ):
        return 1
    if row[ANNUAL_INCOME_COLUMN] < _VERY_LOW_INCOME:
        return 1
    if (
        row[SPENDING_SCORE_COLUMN] > _HIGH_SPENDING_SCORE
        and row[DEBT_RATIO_COLUMN] > _MODERATE_DEBT_RATIO
    ):
        return 1
    if (
        row[CREDIT_CARD_COUNT_COLUMN] >= _HIGH_CREDIT_CARD_COUNT
        and row[DEBT_RATIO_COLUMN] > _CARD_COUNT_DEBT_RATIO
    ):
        return 1
    if (
        row[AGE_COLUMN] < _YOUNG_AGE
        and row[DEBT_RATIO_COLUMN] > _YOUNG_BORROWER_DEBT_RATIO
    ):
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


def _average_latency_ms(predictor, repeats: int = _LATENCY_REPEATS) -> float:
    predictor()
    started = perf_counter()
    for _ in range(repeats):
        predictor()
    return (perf_counter() - started) * _MILLISECONDS_PER_SECOND / repeats


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
    for raw_threshold in SWEEP_THRESHOLDS:
        threshold = float(np.round(raw_threshold, 2))
        predictions = oof_scores >= threshold
        rows.append(
            {
                "threshold": threshold,
                "is_baseline": threshold == DEFAULT_THRESHOLD,
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
    forest = build_random_forest_classifier(
        n_estimators=selection.random_forest_n_estimators or _DEFAULT_RF_N_ESTIMATORS,
        max_depth=selection.random_forest_max_depth,
        min_samples_split=selection.random_forest_min_samples_split,
    )
    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    batch = x_train.iloc[: min(_LATENCY_BATCH_SIZE, len(x_train))]
    rule_predictions = x_holdout.apply(rule_based_predict, axis=1).to_numpy()
    logistic_scores = logistic.predict_proba(x_holdout)[:, 1]
    forest_scores = forest.predict_proba(x_holdout)[:, 1]
    logistic_predictions = apply_threshold(
        logistic_scores, selection.logistic_threshold or DEFAULT_THRESHOLD
    )
    forest_predictions = apply_threshold(
        forest_scores, selection.random_forest_threshold or DEFAULT_THRESHOLD
    )
    latencies = {
        RULE_BASELINE_MODEL: _average_latency_ms(
            lambda: batch.apply(rule_based_predict, axis=1).to_numpy()
        ),
        LOGISTIC_REGRESSION_MODEL: _average_latency_ms(
            lambda: logistic.predict_proba(batch)
        ),
        TUNED_RANDOM_FOREST_MODEL: _average_latency_ms(
            lambda: forest.predict_proba(batch)
        ),
    }
    predictions = {
        RULE_BASELINE_MODEL: rule_predictions,
        LOGISTIC_REGRESSION_MODEL: logistic_predictions,
        TUNED_RANDOM_FOREST_MODEL: forest_predictions,
    }
    scores = {
        RULE_BASELINE_MODEL: rule_predictions.astype(float),
        LOGISTIC_REGRESSION_MODEL: logistic_scores,
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
                "random_forest_probability": forest_scores,
                "random_forest_prediction": forest_predictions,
            }
        ),
        "latency_benchmark": {
            "source": "training_feature_batch",
            "batch_rows": len(batch),
            "repeats": _LATENCY_REPEATS,
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
                    max_iter=_LOGISTIC_MAX_ITERATIONS,
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
                    n_estimators=_DEFAULT_RF_N_ESTIMATORS,
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
    logistic_c_values = FAST_LOGISTIC_C_VALUES if fast else LOGISTIC_C_VALUES
    n_estimator_values = FAST_N_ESTIMATOR_VALUES if fast else N_ESTIMATOR_VALUES
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

    if grid is not None:
        parameter_grid = grid
    elif fast:
        parameter_grid = {
            **RF_LOCAL_REFINEMENT_GRID,
            "model__min_samples_split": FAST_RF_MIN_SAMPLES_SPLIT_VALUES,
        }
    else:
        parameter_grid = RF_LOCAL_REFINEMENT_GRID
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
    latency_batch = x_train.iloc[: min(_LATENCY_BATCH_SIZE, len(x_train))]
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
    decision_tree_predictions = (decision_tree_scores >= DEFAULT_THRESHOLD).astype(int)
    forest_predictions = (forest_scores >= DEFAULT_THRESHOLD).astype(int)
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
        "selected_classification_model": SELECTED_CLASSIFICATION_MODEL,
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
            "repeats": _LATENCY_REPEATS,
        },
        "predictions": prediction_table,
    }
