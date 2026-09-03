"""Classification model training and evaluation without file output."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from credit_risk.data import RANDOM_STATE
from credit_risk.preprocessing import build_preprocessor


DEFAULT_THRESHOLD = 0.5
RECALL_FLOORS = [0.80, 0.85, 0.90, 0.95]


def rule_based_predict(row: pd.Series) -> int:
    """Classify overdue risk with six explicit business rules."""
    if row["overdue_count_6m"] >= 2:
        return 1
    if row["debt_ratio"] > 0.80 and row["annual_income"] < 4500:
        return 1
    if row["annual_income"] < 2500:
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
    prediction_latency_ms: float,
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
        "prediction_latency_ms": float(prediction_latency_ms),
        "fn": int(np.sum((y_values == 1) & (predictions == 0))),
        "fp": int(np.sum((y_values == 0) & (predictions == 1))),
    }


def _average_latency_ms(predictor, repeats: int = 5) -> float:
    predictor()
    started = perf_counter()
    for _ in range(repeats):
        predictor()
    return (perf_counter() - started) * 1000 / repeats


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


def _tune_threshold(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    scoring: str | object,
) -> float:
    """Select an F1-maximizing operating threshold using training CV only."""
    tuner = TunedThresholdClassifierCV(
        estimator=clone(model),
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
    )
    tuner.fit(x_train, y_train)
    return float(tuner.best_threshold_)


def _select_recall_scenarios(
    y_train: pd.Series,
    oof_scores: np.ndarray,
) -> dict[str, dict[str, float | str]]:
    """Select maximum-precision thresholds at recall floors from Train OOF scores."""
    precision, recall, thresholds = precision_recall_curve(y_train, oof_scores)
    scenarios = {}
    for floor in RECALL_FLOORS:
        candidates = np.flatnonzero(recall[:-1] >= floor)
        best_precision = precision[candidates].max()
        best_threshold = thresholds[
            candidates[precision[candidates] == best_precision]
        ].max()
        scenarios[f"{floor:.2f}"] = {
            "threshold": float(best_threshold),
            "cv_recall_floor": float(floor),
            "selection_policy": "maximize_precision_at_recall_floor",
        }
    return scenarios


def train_classification(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    grid: dict | None = None,
    fast: bool = False,
) -> dict:
    """Train and evaluate overdue-risk classifiers without writing files."""
    logistic = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        logistic,
        x_train,
        y_train,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        cv=cv,
        n_jobs=-1,
    )
    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    if grid is not None:
        parameter_grid = grid
    elif fast:
        parameter_grid = {
            "model__n_estimators": [20],
            "model__max_depth": [8],
            "model__min_samples_split": [2],
        }
    else:
        parameter_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5],
        }
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
    tuned_forest_cv_f1 = cross_validate(
        tuned_forest,
        x_train,
        y_train,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    logistic_f1_threshold = _tune_threshold(
        logistic, x_train, y_train, cv, "f1"
    )
    tuned_forest_f1_threshold = _tune_threshold(
        tuned_forest, x_train, y_train, cv, "f1"
    )
    logistic_oof_scores = cross_val_predict(
        logistic,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    logistic_recall_scenarios = _select_recall_scenarios(
        y_train, logistic_oof_scores
    )
    tuned_forest_oof_scores = cross_val_predict(
        tuned_forest,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    tuned_forest_recall_scenarios = _select_recall_scenarios(
        y_train, tuned_forest_oof_scores
    )

    rule_predictions = x_test.apply(rule_based_predict, axis=1).to_numpy()
    logistic_scores = logistic.predict_proba(x_test)[:, 1]
    forest_scores = forest.predict_proba(x_test)[:, 1]
    tuned_scores = tuned_forest.predict_proba(x_test)[:, 1]
    logistic_predictions = (logistic_scores >= DEFAULT_THRESHOLD).astype(int)
    forest_predictions = (forest_scores >= DEFAULT_THRESHOLD).astype(int)
    tuned_predictions = (tuned_scores >= DEFAULT_THRESHOLD).astype(int)
    logistic_tuned_predictions = (
        logistic_scores >= logistic_recall_scenarios["0.90"]["threshold"]
    ).astype(int)
    logistic_f1_predictions = (logistic_scores >= logistic_f1_threshold).astype(int)
    tuned_forest_tuned_predictions = (
        tuned_scores >= tuned_forest_recall_scenarios["0.90"]["threshold"]
    ).astype(int)
    tuned_forest_f1_predictions = (
        tuned_scores >= tuned_forest_f1_threshold
    ).astype(int)

    latencies = {
        "Rule Baseline": _average_latency_ms(
            lambda: x_test.apply(rule_based_predict, axis=1).to_numpy()
        ),
        "Logistic Regression": _average_latency_ms(
            lambda: logistic.predict_proba(x_test)
        ),
        "Random Forest": _average_latency_ms(lambda: forest.predict_proba(x_test)),
        "Random Forest (Tuned)": _average_latency_ms(
            lambda: tuned_forest.predict_proba(x_test)
        ),
    }
    predictions = {
        "Rule Baseline": rule_predictions,
        "Logistic Regression": logistic_predictions,
        "Random Forest": forest_predictions,
        "Random Forest (Tuned)": tuned_predictions,
    }
    scores = {
        "Rule Baseline": rule_predictions.astype(float),
        "Logistic Regression": logistic_scores,
        "Random Forest": forest_scores,
        "Random Forest (Tuned)": tuned_scores,
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
    def build_threshold_analysis(
        default_predictions: np.ndarray,
        model_scores: np.ndarray,
        f1_threshold: float,
        f1_predictions: np.ndarray,
        recall_scenarios: dict[str, dict[str, float | str]],
        latency: float,
    ) -> dict:
        scenarios = {}
        for floor, scenario in recall_scenarios.items():
            threshold = float(scenario["threshold"])
            scenario_predictions = (model_scores >= threshold).astype(int)
            scenarios[floor] = {
                **scenario,
                "test_metrics": evaluate_classifier(
                    y_test, scenario_predictions, model_scores, latency
                ),
            }
        return {
            "default_threshold": DEFAULT_THRESHOLD,
            "default_metrics": evaluate_classifier(
                y_test, default_predictions, model_scores, latency
            ),
            "f1_reference": {
                "threshold": f1_threshold,
                "metrics": evaluate_classifier(
                    y_test, f1_predictions, model_scores, latency
                ),
            },
            "recall_scenarios": scenarios,
        }

    threshold_analysis = {
        "Logistic Regression": build_threshold_analysis(
            logistic_predictions,
            logistic_scores,
            logistic_f1_threshold,
            logistic_f1_predictions,
            logistic_recall_scenarios,
            latencies["Logistic Regression"],
        ),
        "Random Forest (Tuned)": build_threshold_analysis(
            tuned_predictions,
            tuned_scores,
            tuned_forest_f1_threshold,
            tuned_forest_f1_predictions,
            tuned_forest_recall_scenarios,
            latencies["Random Forest (Tuned)"],
        ),
    }
    prediction_table = pd.DataFrame(
        {
            "actual_is_overdue": y_test.to_numpy(),
            "rule_prediction": rule_predictions,
            "logistic_probability": logistic_scores,
            "logistic_prediction_default": logistic_predictions,
            "logistic_prediction_tuned": logistic_tuned_predictions,
            "random_forest_probability": forest_scores,
            "overdue_probability": tuned_scores,
            "tuned_rf_prediction_default": tuned_predictions,
            "tuned_rf_prediction_tuned": tuned_forest_tuned_predictions,
        }
    )
    return {
        "metrics": metrics,
        "logistic_cv": {
            "f1_mean": float(cv_scores["test_f1"].mean()),
            "roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
        },
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "best_cv_f1": float(tuned_forest_cv_f1["test_score"].mean()),
        "feature_importance": _feature_importance(tuned_forest),
        "threshold_analysis": threshold_analysis,
        "threshold_tradeoff": {
            "target": y_train.to_numpy(),
            "scores": tuned_forest_oof_scores,
        },
        "predictions": prediction_table,
    }
