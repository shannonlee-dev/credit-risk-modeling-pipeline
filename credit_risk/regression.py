"""Regression model training and evaluation without file output."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from credit_risk.data import RANDOM_STATE
from credit_risk.preprocessing import build_preprocessor

ALPHAS = [0.01, 0.1, 1, 10, 100]


def _regularized_pipeline(model_name: str, alpha: float) -> Pipeline:
    if model_name == "Ridge":
        estimator = Ridge(alpha=alpha)
    else:
        estimator = Lasso(
            alpha=alpha,
            max_iter=20_000,
            random_state=RANDOM_STATE,
        )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def train_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Select and evaluate Ridge/Lasso models without writing files."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rmse: dict[str, dict[str, float]] = {"Ridge": {}, "Lasso": {}}
    coefficients: dict[str, dict[str, dict[str, float]]] = {
        "Ridge": {},
        "Lasso": {},
    }

    for model_name in ["Ridge", "Lasso"]:
        for alpha in ALPHAS:
            pipeline = _regularized_pipeline(model_name, alpha)
            scores = cross_val_score(
                pipeline,
                x_train,
                y_train,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
            )
            alpha_key = str(alpha)
            cv_rmse[model_name][alpha_key] = float(-scores.mean())

            pipeline.fit(x_train, y_train)
            feature_names = pipeline.named_steps[
                "preprocessor"
            ].get_feature_names_out()
            model_coefficients = pipeline.named_steps["model"].coef_
            coefficients[model_name][alpha_key] = {
                feature.replace("numeric__", "").replace(
                    "categorical__",
                    "",
                ): float(coefficient)
                for feature, coefficient in zip(
                    feature_names,
                    model_coefficients,
                )
            }

    selected_alpha = {
        model_name: float(min(cv_rmse[model_name], key=cv_rmse[model_name].get))
        for model_name in ["Ridge", "Lasso"]
    }
    test_metrics: dict[str, dict[str, float]] = {}
    clipped_predictions: dict[str, pd.Series] = {}

    for model_name in ["Ridge", "Lasso"]:
        final_model = _regularized_pipeline(
            model_name,
            selected_alpha[model_name],
        )
        final_model.fit(x_train, y_train)
        raw_predictions = final_model.predict(x_test)
        test_metrics[model_name] = {
            "rmse": float(root_mean_squared_error(y_test, raw_predictions)),
            "mae": float(mean_absolute_error(y_test, raw_predictions)),
            "r2": float(r2_score(y_test, raw_predictions)),
        }
        clipped_predictions[model_name] = pd.Series(
            np.clip(raw_predictions, 0, 1000),
            name=model_name,
        )

    return {
        "cv_rmse": cv_rmse,
        "selected_alpha": selected_alpha,
        "test_metrics": test_metrics,
        "coefficients": coefficients,
        "predictions": clipped_predictions,
    }
