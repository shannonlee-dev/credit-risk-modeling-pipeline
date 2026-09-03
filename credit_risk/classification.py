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
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from credit_risk.data import RANDOM_STATE
from credit_risk.preprocessing import build_preprocessor


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
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "prediction_latency_ms": float(prediction_latency_ms),
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
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    tuned_forest = search.best_estimator_

    rule_predictions = x_test.apply(rule_based_predict, axis=1).to_numpy()
    logistic_scores = logistic.predict_proba(x_test)[:, 1]
    forest_scores = forest.predict_proba(x_test)[:, 1]
    tuned_scores = tuned_forest.predict_proba(x_test)[:, 1]
    logistic_predictions = (logistic_scores >= 0.5).astype(int)
    forest_predictions = (forest_scores >= 0.5).astype(int)
    tuned_predictions = (tuned_scores >= 0.5).astype(int)

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
    prediction_table = pd.DataFrame(
        {
            "actual_is_overdue": y_test.to_numpy(),
            "rule_prediction": rule_predictions,
            "logistic_probability": logistic_scores,
            "random_forest_probability": forest_scores,
            "overdue_probability": tuned_scores,
        }
    )
    return {
        "metrics": metrics,
        "logistic_cv": {
            "f1_mean": float(cv_scores["test_f1"].mean()),
            "roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
        },
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "feature_importance": _feature_importance(tuned_forest),
        "predictions": prediction_table,
    }
