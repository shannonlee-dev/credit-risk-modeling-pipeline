"""Train-only regularized-regression selection experiments."""

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from credit_risk.constants import CV_FOLDS, RANDOM_STATE, REGRESSION_MODELS
from credit_risk.experiments.config import RegressionExperimentConfig
from credit_risk.regression import _regularized_pipeline


def run_regression_experiment(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: RegressionExperimentConfig,
) -> dict:
    """Select regularization strengths and record coefficient paths on Train only."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_rmse = {name: {} for name in REGRESSION_MODELS}
    coefficients = {name: {} for name in REGRESSION_MODELS}
    for name in REGRESSION_MODELS:
        for alpha in config.alphas:
            pipeline = _regularized_pipeline(name, alpha)
            scores = cross_val_score(pipeline, x_train, y_train, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
            cv_rmse[name][str(alpha)] = float(-scores.mean())
            pipeline.fit(x_train, y_train)
            features = pipeline.named_steps["preprocessor"].get_feature_names_out()
            coefficients[name][str(alpha)] = {
                feature.replace("numeric__", "").replace("categorical__", ""): float(coefficient)
                for feature, coefficient in zip(features, pipeline.named_steps["model"].coef_)
            }
    return {
        "cv_rmse": cv_rmse,
        "selected_alpha": {name: float(min(cv_rmse[name], key=cv_rmse[name].get)) for name in REGRESSION_MODELS},
        "coefficients": coefficients,
    }
