"""Dataset schema, loading, validation, and train/test splits."""

from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.constants import (
    CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REGRESSION_TARGET,
    REQUIRED_COLUMNS,
    TARGET_COLUMNS,
)


_TEST_SIZE = 0.2


def load_and_validate_data(path: str | Path) -> pd.DataFrame:
    """Load a generated CSV and reject incomplete schemas."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {data_path}. "
            "먼저 `python3 scripts/generate_data.py`를 실행하세요."
        )

    df = pd.read_csv(data_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"누락된 필수 열: {', '.join(missing)}")
    return df


def dataset_fingerprint(path: str | Path) -> str:
    """Return a stable fingerprint for the exact input CSV bytes."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    return f"sha256:{sha256(data_path.read_bytes()).hexdigest()}"


def split_classification_data(df: pd.DataFrame):
    """Create a reproducible stratified classification split."""
    return train_test_split(
        df[FEATURE_COLUMNS],
        df[CLASSIFICATION_TARGET],
        test_size=_TEST_SIZE,
        stratify=df[CLASSIFICATION_TARGET],
        random_state=RANDOM_STATE,
    )


def split_regression_data(df: pd.DataFrame):
    """Create a reproducible regression split."""
    return train_test_split(
        df[FEATURE_COLUMNS],
        df[REGRESSION_TARGET],
        test_size=_TEST_SIZE,
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
