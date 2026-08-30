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


def test_preprocessor_has_numeric_and_categorical_paths():
    preprocessor = build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)
    assert {name for name, _, _ in preprocessor.transformers} == {
        "numeric",
        "categorical",
    }


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
            "prediction_latency_ms",
        } <= metrics.keys()
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
    assert result["best_params"] == {
        "model__max_depth": 8,
        "model__min_samples_split": 2,
        "model__n_estimators": 20,
    }
    assert result["predictions"]["overdue_probability"].between(0, 1).all()
    assert (output_dir / "classification_predictions.csv").is_file()
    for name in [
        "confusion_matrix.png",
        "roc_curve.png",
        "feature_importance.png",
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
