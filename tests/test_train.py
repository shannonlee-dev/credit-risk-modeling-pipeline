import json
import os
import subprocess
import sys

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from data_gen import generate_finance_data
from train import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
    class_distribution,
    load_and_validate_data,
    rule_based_predict,
    run_analysis,
    run_classification,
    run_regression,
    split_classification_data,
    split_regression_data,
)


@pytest.fixture(scope="module")
def finance_df(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "finance.csv"
    return generate_finance_data(path)


def test_classification_split_prevents_target_leakage(finance_df):
    x_train, x_test, y_train, y_test = split_classification_data(finance_df)

    assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "credit_score" not in x_train.columns
    assert "is_overdue" not in x_train.columns
    assert len(x_train) == 8_000
    assert len(x_test) == 2_000
    assert abs(y_train.mean() - y_test.mean()) < 0.001


def test_regression_split_uses_credit_score_target(finance_df):
    x_train, x_test, y_train, y_test = split_regression_data(finance_df)

    assert list(x_train.columns) == FEATURE_COLUMNS
    assert len(x_train) == 8_000
    assert len(x_test) == 2_000
    assert y_train.name == "credit_score"
    assert y_test.name == "credit_score"


def test_preprocessor_treats_card_count_as_numeric():
    features = pd.DataFrame(
        {
            "age": [20, 40],
            "annual_income": [3000, 5000],
            "spending_score": [30, 70],
            "debt_ratio": [0.2, 0.8],
            "overdue_count_6m": [0, 2],
            "credit_card_count": [1, 9],
        }
    )
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    assert isinstance(preprocessor, ColumnTransformer)
    assert transformed.shape == (2, 6)
    assert preprocessor.get_feature_names_out().tolist() == [
        "numeric__age",
        "numeric__annual_income",
        "numeric__spending_score",
        "numeric__debt_ratio",
        "numeric__overdue_count_6m",
        "numeric__credit_card_count",
    ]
    assert transformed[:, -1].tolist() == pytest.approx([-1.0, 1.0])


def test_class_distribution_reports_literal_counts():
    distribution = class_distribution(pd.Series([0, 0, 1, 0, 1]))

    assert distribution == {
        "count_0": 3,
        "count_1": 2,
        "positive_rate": 0.4,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"overdue_count_6m": 3},
        {"debt_ratio": 0.9, "annual_income": 4000},
        {"annual_income": 2000},
        {"spending_score": 95, "debt_ratio": 0.75},
        {"credit_card_count": 8, "debt_ratio": 0.7},
        {"age": 22, "debt_ratio": 0.8},
    ],
)
def test_rule_model_flags_each_risk_condition(updates):
    safe = pd.Series(
        {
            "age": 40,
            "annual_income": 7000,
            "spending_score": 50,
            "debt_ratio": 0.2,
            "credit_card_count": 2,
            "overdue_count_6m": 0,
        }
    )
    risky = safe.copy()
    for key, value in updates.items():
        risky[key] = value

    assert rule_based_predict(safe) == 0
    assert rule_based_predict(risky) == 1


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"age": [30]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="누락된 필수 열"):
        load_and_validate_data(path)


def test_classification_compares_models_and_saves_artifacts(finance_df, tmp_path):
    sample = finance_df.head(1_200)
    x_train, x_test, y_train, y_test = split_classification_data(sample)
    output_dir = tmp_path / "artifacts"

    result = run_classification(
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        grid={
            "model__n_estimators": [20],
            "model__max_depth": [8],
            "model__min_samples_split": [2],
        },
    )

    assert set(result["metrics"]) == {
        "Rule Baseline",
        "Logistic Regression",
        "Random Forest",
        "Random Forest (Tuned)",
    }
    for metrics in result["metrics"].values():
        assert {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "batch_prediction_latency_ms",
        } <= metrics.keys()
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
    assert result["best_params"] == {
        "model__max_depth": 8,
        "model__min_samples_split": 2,
        "model__n_estimators": 20,
    }
    saturation = result["random_forest_saturation"]
    assert len(saturation) >= 5
    assert set(saturation) == {"25", "50", "100", "200", "300", "500"}
    for values in saturation.values():
        assert {
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
            "cv_f1_mean",
            "fit_time_seconds",
            "batch_prediction_latency_ms",
        } <= values.keys()
        assert 0 <= values["cv_roc_auc_mean"] <= 1
        assert values["cv_roc_auc_std"] >= 0
        assert 0 <= values["cv_f1_mean"] <= 1
        assert values["fit_time_seconds"] > 0
        assert values["batch_prediction_latency_ms"] > 0
    assert result["predictions"]["overdue_probability"].between(0, 1).all()
    assert set(result["threshold_analysis"]) == {
        "Logistic Regression",
        "Random Forest (Tuned)",
    }
    for analysis in result["threshold_analysis"].values():
        assert analysis["default_threshold"] == 0.5
        assert 0 <= analysis["f1_reference"]["threshold"] <= 1
        assert {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "fn",
            "fp",
        } <= analysis["default_metrics"].keys()
        assert set(analysis["recall_scenarios"]) == {
            "0.80",
            "0.85",
            "0.90",
            "0.95",
        }
        assert analysis["f1_reference"]["metrics"]["f1"] >= 0
        for floor, scenario in analysis["recall_scenarios"].items():
            assert 0 <= scenario["threshold"] <= 1
            assert scenario["cv_recall_floor"] == float(floor)
            assert scenario["selection_policy"] == "maximize_precision_at_recall_floor"
            assert scenario["test_metrics"]["fn"] >= 0
            assert scenario["test_metrics"]["fp"] >= 0
            assert scenario["test_metrics"]["roc_auc"] == pytest.approx(
                analysis["default_metrics"]["roc_auc"]
            )
    assert {
        "logistic_prediction_default",
        "logistic_prediction_illustrative_recall_090",
        "tuned_rf_prediction_default",
        "tuned_rf_prediction_illustrative_recall_090",
    } <= set(result["predictions"])
    latency_benchmark = result["latency_benchmark"]
    assert latency_benchmark == {
        "batch_rows": len(x_train),
        "repeats": 5,
        "source": "training_feature_batch",
    }
    tradeoff = result["threshold_tradeoff"]
    assert tradeoff["target"].tolist() == y_train.tolist()
    assert len(tradeoff["scores"]) == len(y_train)
    assert (output_dir / "classification_predictions.csv").is_file()
    for name in [
        "confusion_matrix.png",
        "confusion_matrix_threshold_comparison.png",
        "roc_curve.png",
        "feature_importance.png",
        "random_forest_n_estimators_curve.png",
        "threshold_tradeoff.png",
    ]:
        assert (output_dir / name).stat().st_size > 0


def test_regression_selects_alpha_and_saves_bounded_predictions(
    finance_df,
    tmp_path,
):
    sample = finance_df.head(1_200)
    x_train, x_test, y_train, y_test = split_regression_data(sample)
    output_dir = tmp_path / "artifacts"

    result = run_regression(x_train, x_test, y_train, y_test, output_dir)

    assert set(result["test_metrics"]) == {"Ridge", "Lasso"}
    assert set(result["selected_alpha"]) == {"Ridge", "Lasso"}
    assert set(result["cv_rmse"]["Ridge"]) == {
        "0.01",
        "0.1",
        "1",
        "10",
        "100",
    }
    for metrics in result["test_metrics"].values():
        assert {"rmse", "mae", "r2"} <= metrics.keys()
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
    for predictions in result["predictions"].values():
        assert predictions.between(0, 1000).all()
    assert (output_dir / "credit_score_predictions.csv").is_file()
    assert (output_dir / "regularization_coefficients.png").stat().st_size > 0


def test_run_analysis_saves_serializable_metrics_and_predictions(
    finance_df,
    tmp_path,
):
    data_path = tmp_path / "finance.csv"
    finance_df.head(1_200).to_csv(data_path, index=False)
    output_dir = tmp_path / "artifacts"

    result = run_analysis(data_path, output_dir, fast=True)

    assert set(result) == {"data_distribution", "classification", "regression"}
    assert set(result["data_distribution"]) == {"all", "train", "test"}
    report = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "predictions" not in report["classification"]
    assert "threshold_tradeoff" not in report["classification"]
    assert "predictions" not in report["regression"]
    assert "threshold_analysis" in report["classification"]
    assert len(report["classification"]["random_forest_saturation"]) >= 5
    assert (output_dir / "random_forest_n_estimators_curve.png").is_file()
    assert (output_dir / "classification_predictions.csv").is_file()
    assert (output_dir / "credit_score_predictions.csv").is_file()


def test_training_computation_does_not_write_artifacts(
    finance_df,
    tmp_path,
    monkeypatch,
):
    from credit_risk.classification import train_classification
    from credit_risk.regression import train_regression

    sample = finance_df.head(400)
    classification_split = split_classification_data(sample)
    regression_split = split_regression_data(sample)
    monkeypatch.chdir(tmp_path)

    classification = train_classification(
        *classification_split,
        grid={
            "model__n_estimators": [10],
            "model__max_depth": [4],
            "model__min_samples_split": [2],
        },
    )
    regression = train_regression(*regression_split)

    assert "metrics" in classification
    assert "test_metrics" in regression
    assert list(tmp_path.iterdir()) == []


def test_import_uses_writable_matplotlib_cache():
    environment = os.environ.copy()
    environment["HOME"] = "/proc/credit-risk-unwritable-home"
    environment.pop("MPLCONFIGDIR", None)

    completed = subprocess.run(
        [sys.executable, "-c", "import train"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0
    assert "temporary cache directory" not in completed.stderr
