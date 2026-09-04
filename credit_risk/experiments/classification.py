"""Train-only classification selection and interpretation experiments."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate

from credit_risk.classification import (
    _average_latency_ms,
    _feature_importance,
    build_logistic_classifier,
    build_random_forest_classifier,
)
from credit_risk.constants import CV_FOLDS, RANDOM_STATE
from credit_risk.evaluation import evaluate_thresholds
from credit_risk.experiments.config import ClassificationExperimentConfig


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def run_classification_experiment(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: ClassificationExperimentConfig,
) -> dict:
    """Run every classification selection analysis using Train data only."""
    cv = _cv()
    logistic_analysis = {}
    for c in config.logistic_c_values:
        scores = cross_validate(
            build_logistic_classifier(c), x_train, y_train,
            scoring={"roc_auc": "roc_auc", "f1": "f1"}, cv=cv, n_jobs=-1,
        )
        logistic_analysis[str(c)] = {
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
        }
    selected_logistic_c = min(
        config.logistic_c_values,
        key=lambda value: (-logistic_analysis[str(value)]["cv_roc_auc_mean"], value),
    )
    forest = build_random_forest_classifier(n_estimators=100)
    grid = {
        "model__n_estimators": [100],
        "model__max_depth": list(config.rf_max_depth_values),
        "model__min_samples_split": list(config.rf_min_samples_split_values),
    }
    search = GridSearchCV(clone(forest), grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    search.fit(x_train, y_train)
    tuned_forest = search.best_estimator_
    logistic_oof = cross_val_predict(
        build_logistic_classifier(selected_logistic_c), x_train, y_train,
        cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    forest_oof = cross_val_predict(tuned_forest, x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    batch = x_train.iloc[: min(2_000, len(x_train))]
    sensitivity = {}
    for trees in config.rf_n_estimators_values:
        candidate = build_random_forest_classifier(
            n_estimators=trees,
            max_depth=search.best_params_["model__max_depth"],
            min_samples_split=search.best_params_["model__min_samples_split"],
        )
        scores = cross_validate(candidate, x_train, y_train, scoring={"roc_auc": "roc_auc", "f1": "f1"}, cv=cv, n_jobs=-1)
        candidate.fit(x_train, y_train)
        sensitivity[str(trees)] = {
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "fit_time_seconds": float(scores["fit_time"].mean()),
            "batch_prediction_latency_ms": float(_average_latency_ms(lambda: candidate.predict_proba(batch))),
        }
    tuned_forest.fit(x_train, y_train)
    return {
        "logistic_c_analysis": logistic_analysis,
        "selected_logistic_c": selected_logistic_c,
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "logistic_threshold_sweep": evaluate_thresholds(y_train, logistic_oof, config.threshold_values),
        "random_forest_threshold_sweep": evaluate_thresholds(y_train, forest_oof, config.threshold_values),
        "random_forest_saturation": sensitivity,
        "feature_importance": _feature_importance(tuned_forest),
        "latency_benchmark": {"source": "training_feature_batch", "batch_rows": len(batch), "repeats": 5},
    }
