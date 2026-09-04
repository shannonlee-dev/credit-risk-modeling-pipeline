"""Regression builders, final evaluation, and legacy result adaptation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline

from credit_risk.constants import (
    CREDIT_SCORE_MAX,
    CREDIT_SCORE_MIN,
    RANDOM_STATE,
)
from credit_risk.preprocessing import build_preprocessor
from credit_risk.evaluation import evaluate_regression
from credit_risk.results import FinalSelection


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
    """Compatibility facade over Train-only experiment and final evaluation."""
    from credit_risk.experiments.config import FULL_EXPERIMENT
    from credit_risk.experiments.regression import run_regression_experiment

    experiment = run_regression_experiment(
        x_train,
        y_train,
        FULL_EXPERIMENT.regression,
    )
    selection = FinalSelection(
        selected_model="Logistic Regression",
        logistic_c=0.01,
        logistic_threshold=0.45,
        random_forest_n_estimators=100,
        random_forest_max_depth=None,
        random_forest_min_samples_split=2,
        random_forest_threshold=0.33,
        ridge_alpha=experiment["selected_alpha"]["Ridge"],
        lasso_alpha=experiment["selected_alpha"]["Lasso"],
    )
    final = evaluate_final_regression(
        x_train,
        x_test,
        y_train,
        y_test,
        selection,
    )
    return adapt_legacy_regression_result(experiment, final)


def adapt_legacy_regression_result(experiment: dict, final: dict) -> dict:
    """Adapt two-stage outputs to the historical regression dictionary."""
    return {
        "cv_rmse": experiment["cv_rmse"],
        "selected_alpha": final["selected_alpha"],
        "test_metrics": final["test_metrics"],
        "coefficients": experiment["coefficients"],
        "predictions": final["predictions"],
    }


def evaluate_final_regression(
    x_train: pd.DataFrame,
    x_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    selection: FinalSelection,
) -> dict:
    """Fit selected regularized regressors and evaluate untouched Holdout once."""
    selected_alpha = {
        "Ridge": selection.ridge_alpha,
        "Lasso": selection.lasso_alpha,
    }
    metrics = {}
    predictions = {}
    for model_name, alpha in selected_alpha.items():
        model = _regularized_pipeline(model_name, alpha)
        model.fit(x_train, y_train)
        raw_predictions = model.predict(x_holdout)
        metrics[model_name] = evaluate_regression(y_holdout, raw_predictions)
        predictions[model_name] = pd.Series(
            np.clip(raw_predictions, CREDIT_SCORE_MIN, CREDIT_SCORE_MAX),
            name=model_name,
        )
    return {
        "selected_alpha": selected_alpha,
        "test_metrics": metrics,
        "predictions": predictions,
    }
