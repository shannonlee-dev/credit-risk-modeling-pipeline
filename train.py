"""Train and evaluate credit-risk classification and regression models."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "age",
    "annual_income",
    "spending_score",
    "debt_ratio",
    "overdue_count_6m",
    "credit_card_count",
]
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "spending_score",
    "debt_ratio",
    "overdue_count_6m",
]
CATEGORICAL_FEATURES = ["credit_card_count"]
TARGET_COLUMNS = ["credit_score", "is_overdue"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS


def load_and_validate_data(path: str | Path) -> pd.DataFrame:
    """Load a generated CSV and reject incomplete schemas."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {data_path}. "
            "먼저 `python3 data_gen.py`를 실행하세요."
        )

    df = pd.read_csv(data_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"누락된 필수 열: {', '.join(missing)}")
    return df


def split_classification_data(df: pd.DataFrame):
    """Create a reproducible stratified classification split."""
    return train_test_split(
        df[FEATURE_COLUMNS],
        df["is_overdue"],
        test_size=0.2,
        stratify=df["is_overdue"],
        random_state=RANDOM_STATE,
    )


def split_regression_data(df: pd.DataFrame):
    """Create a reproducible regression split."""
    return train_test_split(
        df[FEATURE_COLUMNS],
        df["credit_score"],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )


def class_distribution(target: pd.Series) -> dict[str, int | float]:
    """Summarize binary class counts and positive rate."""
    counts = target.value_counts().reindex([0, 1], fill_value=0)
    return {
        "count_0": int(counts.loc[0]),
        "count_1": int(counts.loc[1]),
        "positive_rate": float(target.mean()),
    }


def build_preprocessor() -> ColumnTransformer:
    """Build leakage-safe numeric and categorical preprocessing."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def rule_based_predict(row: pd.Series) -> int:
    """Classify overdue risk with six explicit business rules."""
    if row["overdue_count_6m"] >= 2:
        return 1
    if row["debt_ratio"] > 0.80 and row["annual_income"] < 4500:
        return 1
    if row["annual_income"] < 2500:
        return 1
    if row["spending_score"] > 90 and row["debt_ratio"] > 0.70:
        return 1
    if row["credit_card_count"] >= 8 and row["debt_ratio"] > 0.65:
        return 1
    if row["age"] < 25 and row["debt_ratio"] > 0.75:
        return 1
    return 0
