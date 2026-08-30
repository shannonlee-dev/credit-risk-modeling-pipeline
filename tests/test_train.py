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
