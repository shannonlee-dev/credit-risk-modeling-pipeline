"""Application workflow that connects computation and reporting."""

from pathlib import Path

import pandas as pd

from credit_risk.classification import train_classification
from credit_risk.constants import (
    CLASSIFICATION_TARGET,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_PATH,
)
from credit_risk.data import (
    class_distribution,
    load_and_validate_data,
    split_classification_data,
    split_regression_data,
)
from credit_risk.regression import train_regression
from credit_risk.reporting import (
    save_classification_artifacts,
    save_metrics_report,
    save_regression_artifacts,
)

LEGACY_DATA_PATH = Path("finance_data.csv")
_METRICS_FILENAME = "metrics.json"


def run_classification(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | Path,
    grid: dict | None = None,
    fast: bool = False,
) -> dict:
    """Train classifiers and save their comparison artifacts."""
    result = train_classification(
        x_train,
        x_test,
        y_train,
        y_test,
        grid=grid,
        fast=fast,
    )
    save_classification_artifacts(result, y_test, output_dir)
    return result


def run_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | Path,
) -> dict:
    """Train regressors and save their comparison artifacts."""
    result = train_regression(x_train, x_test, y_train, y_test)
    save_regression_artifacts(result, y_test, output_dir)
    return result


def run_analysis(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    fast: bool = False,
) -> dict:
    """Run the analysis, optionally using reduced classification candidates."""
    source = Path(data_path)
    if source == DEFAULT_DATA_PATH and not source.exists() and LEGACY_DATA_PATH.exists():
        source = LEGACY_DATA_PATH
    df = load_and_validate_data(source)
    classification_split = split_classification_data(df)
    regression_split = split_regression_data(df)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    result = {
        "data_distribution": {
            "all": class_distribution(df[CLASSIFICATION_TARGET]),
            "train": class_distribution(classification_split[2]),
            "test": class_distribution(classification_split[3]),
        },
        "classification": run_classification(
            *classification_split,
            destination,
            fast=fast,
        ),
        "regression": run_regression(*regression_split, destination),
    }
    save_metrics_report(result, destination / _METRICS_FILENAME)
    return result
