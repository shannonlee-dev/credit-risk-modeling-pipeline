"""Small, explicit candidate profiles for Train-only experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationExperimentConfig:
    logistic_c_values: tuple[float, ...]
    rf_max_depth_values: tuple[int | None, ...]
    rf_min_samples_split_values: tuple[int, ...]
    rf_n_estimators_values: tuple[int, ...]
    threshold_values: tuple[float, ...]


@dataclass(frozen=True)
class RegressionExperimentConfig:
    alphas: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    classification: ClassificationExperimentConfig
    regression: RegressionExperimentConfig


_THRESHOLDS = tuple(float(value) for value in np.linspace(0.30, 0.60, 31))
FULL_EXPERIMENT = ExperimentProfile(
    name="full",
    classification=ClassificationExperimentConfig(
        logistic_c_values=(0.001, 0.003, 0.01, 0.03, 0.1),
        rf_max_depth_values=(None, 8, 16),
        rf_min_samples_split_values=(5, 10, 20, 40, 80),
        rf_n_estimators_values=(25, 50, 100, 200, 300, 500),
        threshold_values=_THRESHOLDS,
    ),
    regression=RegressionExperimentConfig(alphas=(0.01, 0.1, 1, 10, 100)),
)
SMOKE_EXPERIMENT = ExperimentProfile(
    name="smoke",
    classification=ClassificationExperimentConfig(
        logistic_c_values=(0.01, 0.1),
        rf_max_depth_values=(None, 8, 16),
        rf_min_samples_split_values=(20, 40),
        rf_n_estimators_values=(50, 100),
        threshold_values=_THRESHOLDS,
    ),
    regression=FULL_EXPERIMENT.regression,
)
