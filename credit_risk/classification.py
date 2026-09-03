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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from credit_risk.data import RANDOM_STATE
from credit_risk.preprocessing import build_preprocessor


DEFAULT_THRESHOLD = 0.5
SELECTED_LOGISTIC_THRESHOLD = 0.45
SELECTED_RANDOM_FOREST_THRESHOLD = 0.33
SELECTED_CLASSIFICATION_MODEL = "Logistic Regression"
SWEEP_THRESHOLDS = np.linspace(0.30, 0.60, 31)
LOGISTIC_C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
N_ESTIMATOR_VALUES = [25, 50, 100, 200, 300, 500]
RF_MAX_DEPTH_VALUES = [8]
RF_MIN_SAMPLES_SPLIT_VALUES = [5, 10, 20, 40, 80]
RF_LOCAL_REFINEMENT_GRID = {
    "model__n_estimators": [100],
    "model__max_depth": RF_MAX_DEPTH_VALUES,
    "model__min_samples_split": RF_MIN_SAMPLES_SPLIT_VALUES,
}


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
) -> dict[str, dict[str, float]]:
    """Measure tree-count sensitivity with the selected non-tree settings."""
    fixed_params = {
        "model__max_depth": best_params["model__max_depth"],
        "model__min_samples_split": best_params["model__min_samples_split"],
    }
    saturation = {}
    for n_estimators in N_ESTIMATOR_VALUES:
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
) -> tuple[dict[str, dict[str, float]], float]:
    """Select logistic regularization using Train CV ROC-AUC only."""
    analysis = {}
    for c_value in LOGISTIC_C_VALUES:
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
        LOGISTIC_C_VALUES,
        key=lambda c_value: (-analysis[str(c_value)]["cv_roc_auc_mean"], c_value),
    )
    return analysis, selected_c


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
    logistic_c_analysis, selected_logistic_c = _analyze_logistic_c_values(
        logistic,
        x_train,
        y_train,
        cv=cv,
    )
    logistic = logistic.set_params(model__C=selected_logistic_c)
    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    if grid is not None:
        parameter_grid = grid
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
    random_forest_min_samples_split_analysis = {
        str(params["model__min_samples_split"]): {
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
    forest_scores = forest.predict_proba(x_test)[:, 1]
    tuned_scores = tuned_forest.predict_proba(x_test)[:, 1]
    logistic_predictions = (
        logistic_scores >= SELECTED_LOGISTIC_THRESHOLD
    ).astype(int)
    forest_predictions = (forest_scores >= DEFAULT_THRESHOLD).astype(int)
    tuned_predictions = (
        tuned_scores >= SELECTED_RANDOM_FOREST_THRESHOLD
    ).astype(int)

    latencies = {
        "Rule Baseline": _average_latency_ms(
            lambda: latency_batch.apply(rule_based_predict, axis=1).to_numpy()
        ),
        "Logistic Regression": _average_latency_ms(
            lambda: logistic.predict_proba(latency_batch)
        ),
        "Random Forest": _average_latency_ms(
            lambda: forest.predict_proba(latency_batch)
        ),
        "Random Forest (Tuned)": _average_latency_ms(
            lambda: tuned_forest.predict_proba(latency_batch)
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
            "logistic_prediction_selected": logistic_predictions,
            "random_forest_probability": forest_scores,
            "overdue_probability": tuned_scores,
            "tuned_rf_prediction_selected": tuned_predictions,
        }
    )
    return {
        "metrics": metrics,
        "logistic_cv": {
            "f1_mean": logistic_c_analysis[str(selected_logistic_c)]["cv_f1_mean"],
            "roc_auc_mean": logistic_c_analysis[str(selected_logistic_c)][
                "cv_roc_auc_mean"
            ],
        },
        "logistic_c_analysis": logistic_c_analysis,
        "selected_logistic_c": selected_logistic_c,
        "selected_classification_model": SELECTED_CLASSIFICATION_MODEL,
        "selected_logistic_threshold": SELECTED_LOGISTIC_THRESHOLD,
        "selected_random_forest_threshold": SELECTED_RANDOM_FOREST_THRESHOLD,
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "best_cv_f1": float(tuned_forest_cv_f1["test_score"].mean()),
        "random_forest_min_samples_split_analysis": (
            random_forest_min_samples_split_analysis
        ),
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
