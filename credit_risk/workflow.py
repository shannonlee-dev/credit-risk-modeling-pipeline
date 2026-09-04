"""Application workflow that connects computation and reporting."""

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from credit_risk.classification import (
    adapt_legacy_classification_result,
    train_classification,
)
from credit_risk.constants import (
    CLASSIFICATION_TARGET,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_PATH,
)
from credit_risk.data import (
    class_distribution,
    dataset_fingerprint,
    load_and_validate_data,
    split_classification_data,
    split_regression_data,
)
from credit_risk.experiments.classification import run_classification_experiment
from credit_risk.experiments.config import FULL_EXPERIMENT, SMOKE_EXPERIMENT, ExperimentProfile
from credit_risk.experiments.regression import run_regression_experiment
from credit_risk.classification import evaluate_final_classification
from credit_risk.regression import (
    adapt_legacy_regression_result,
    evaluate_final_regression,
)
from credit_risk.results import FinalSelection
from credit_risk.regression import train_regression
from credit_risk.reporting import (
    save_experiment_artifacts,
    save_final_artifacts,
    save_classification_artifacts,
    save_regression_artifacts,
)

def _protocol_fingerprint(profile: ExperimentProfile) -> str:
    payload = json.dumps(
        {
            "profile": profile.name,
            "classification": profile.classification.__dict__,
            "regression": profile.regression.__dict__,
            "cv_folds": 5,
            "random_state": 42,
            "classification_split": "80:20 stratified",
            "regression_split": "80:20",
        },
        sort_keys=True,
        default=list,
    ).encode()
    return f"sha256:{sha256(payload).hexdigest()}"


def _selection_template(experiment: dict) -> dict:
    best = experiment["classification"]["best_params"]
    return {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "dataset_fingerprint": experiment["dataset_fingerprint"],
        "protocol_fingerprint": experiment["protocol_fingerprint"],
        "classification": {
            "selected_model": None,
            "logistic_regression": {"C": experiment["classification"]["selected_logistic_c"], "threshold": None},
            "random_forest": {
                "n_estimators": None,
                "max_depth": best["model__max_depth"],
                "min_samples_split": best["model__min_samples_split"],
                "threshold": None,
            },
        },
        "regression": {
            "ridge_alpha": experiment["regression"]["selected_alpha"]["Ridge"],
            "lasso_alpha": experiment["regression"]["selected_alpha"]["Lasso"],
        },
    }


def resolve_default_selection(experiment: dict) -> dict:
    """Resolve the repository-approved reproducibility selection."""
    selection = _selection_template(experiment)
    selection["classification"]["selected_model"] = "Logistic Regression"
    selection["classification"]["logistic_regression"]["threshold"] = 0.45
    selection["classification"]["random_forest"]["n_estimators"] = 100
    selection["classification"]["random_forest"]["threshold"] = 0.33
    return selection


def _json_ready(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {key: _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def run_experiment(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    profile: ExperimentProfile = FULL_EXPERIMENT,
) -> dict:
    """Run Train-only experiments and write no final-evaluation artifact."""
    df = load_and_validate_data(data_path)
    classification_split = split_classification_data(df)
    regression_split = split_regression_data(df)
    fingerprint = dataset_fingerprint(data_path)
    protocol = _protocol_fingerprint(profile)
    result = {
        "dataset_fingerprint": fingerprint,
        "protocol_fingerprint": protocol,
        "experiment_id": f"sha256:{sha256((fingerprint + protocol).encode()).hexdigest()}",
        "profile": profile.name,
        "data_distribution": {
            "all": class_distribution(df[CLASSIFICATION_TARGET]),
            "train": class_distribution(classification_split[2]),
        },
        "classification": run_classification_experiment(
            classification_split[0], classification_split[2], profile.classification
        ),
        "regression": run_regression_experiment(
            regression_split[0], regression_split[2], profile.regression
        ),
    }
    destination = Path(output_dir) / "experiment"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "experiment.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "selection.template.json").write_text(json.dumps(_selection_template(result), ensure_ascii=False, indent=2), encoding="utf-8")
    result["classification"]["logistic_threshold_sweep"].to_csv(destination / "logistic_threshold_sweep.csv", index=False)
    result["classification"]["random_forest_threshold_sweep"].to_csv(destination / "random_forest_threshold_sweep.csv", index=False)
    save_experiment_artifacts(result, destination)
    return result


def _run_final_evaluation(
    data_path: str | Path,
    raw_selection: dict,
    output_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    profile: ExperimentProfile = FULL_EXPERIMENT,
) -> dict:
    selection = FinalSelection.from_dict(raw_selection)
    if selection.dataset_fingerprint != dataset_fingerprint(data_path):
        raise ValueError("selection dataset fingerprint does not match current data")
    if selection.protocol_fingerprint != _protocol_fingerprint(profile):
        raise ValueError("selection protocol fingerprint does not match current profile")
    df = load_and_validate_data(data_path)
    classification_split = split_classification_data(df)
    regression_split = split_regression_data(df)
    result = {
        "dataset": {
            "rows": len(df),
            "holdout_rows": len(classification_split[1]),
            "positive_rate": float(df[CLASSIFICATION_TARGET].mean()),
        },
        "selection": raw_selection,
        "classification": evaluate_final_classification(*classification_split, selection),
        "regression": evaluate_final_regression(*regression_split, selection),
    }
    destination = Path(output_dir) / "final"
    destination.mkdir(parents=True, exist_ok=True)
    classification_metrics = {
        key: value
        for key, value in result["classification"]["metrics"][
            selection.selected_model
        ].items()
        if key != "batch_prediction_latency_ms"
    }
    metrics = {
        "dataset": result["dataset"],
        "selection": {"classification_model": selection.selected_model, "threshold": result["classification"]["selected_threshold"]},
        "classification": classification_metrics,
        "regression": {
            name: {"alpha": result["regression"]["selected_alpha"][name], **values}
            for name, values in result["regression"]["test_metrics"].items()
        },
    }
    (destination / "metrics.json").write_text(json.dumps(_json_ready(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    result["classification"]["predictions"].to_csv(destination / "classification_predictions.csv", index=False)
    pd.DataFrame({"actual_credit_score": regression_split[3].to_numpy(), **{name: values.to_numpy() for name, values in result["regression"]["predictions"].items()}}).to_csv(destination / "credit_score_predictions.csv", index=False)
    save_final_artifacts(result, destination)
    return result


def run_final_evaluation(
    data_path: str | Path,
    selection_path: str | Path,
    output_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    profile: ExperimentProfile = FULL_EXPERIMENT,
) -> dict:
    """Validate a selection then evaluate its fixed models on untouched Holdout."""
    raw_selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    return _run_final_evaluation(data_path, raw_selection, output_dir, profile)


def _run_all_stages(
    data_path: str | Path,
    output_dir: str | Path,
    profile: ExperimentProfile,
) -> tuple[dict, dict]:
    experiment = run_experiment(data_path, output_dir, profile)
    raw_selection = resolve_default_selection(experiment)
    selection_path = Path(output_dir) / "experiment" / "selection.json"
    selection_path.write_text(
        json.dumps(raw_selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final = _run_final_evaluation(
        data_path,
        raw_selection,
        output_dir,
        profile,
    )
    return experiment, final


def run_all(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    profile: ExperimentProfile = FULL_EXPERIMENT,
) -> dict:
    """Reproduce the approved default selection through the two-stage path."""
    _, final = _run_all_stages(data_path, output_dir, profile)
    return final

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
    """Compatibility facade over the same two stages used by the CLI."""
    profile = SMOKE_EXPERIMENT if fast else FULL_EXPERIMENT
    experiment, final = _run_all_stages(data_path, output_dir, profile)
    selection = FinalSelection.from_dict(final["selection"])
    return {
        "data_distribution": {
            **experiment["data_distribution"],
            "test": class_distribution(
                final["classification"]["predictions"]["actual_is_overdue"]
            ),
        },
        "classification": adapt_legacy_classification_result(
            experiment["classification"],
            final["classification"],
            selection,
        ),
        "regression": adapt_legacy_regression_result(
            experiment["regression"],
            final["regression"],
        ),
    }
