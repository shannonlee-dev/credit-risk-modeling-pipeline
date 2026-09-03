"""Generate the synthetic finance dataset used by the project."""

import random  # Included to match the supplied generation environment.
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.constants import (
    AGE_COLUMN,
    ANNUAL_INCOME_COLUMN,
    CLASSIFICATION_TARGET,
    CREDIT_CARD_COUNT_COLUMN,
    CREDIT_SCORE_MAX,
    CREDIT_SCORE_MIN,
    DATASET_COLUMNS as EXPECTED_COLUMNS,
    DEFAULT_DATA_PATH as DEFAULT_OUTPUT_PATH,
    DEBT_RATIO_COLUMN,
    OVERDUE_COUNT_COLUMN,
    RANDOM_STATE,
    REGRESSION_TARGET,
    SPENDING_SCORE_COLUMN,
)


N_SAMPLES = 10_000
_AGE_RANGE = (20, 70)
_ANNUAL_INCOME_MEAN = 5_000
_ANNUAL_INCOME_STANDARD_DEVIATION = 2_000
_MIN_ANNUAL_INCOME = 1_500
_SPENDING_SCORE_RANGE = (1, 100)
_DEBT_RATIO_RANGE = (0, 1)
_CREDIT_CARD_COUNT_RANGE = (1, 10)
_OVERDUE_COUNT_MEAN = 0.5
_BASE_CREDIT_SCORE = 300
_INCOME_SCALE = 100
_INCOME_WEIGHT = 3
_OVERDUE_PENALTY = 50
_DEBT_RATIO_PENALTY = 100
_CREDIT_SCORE_NOISE_STANDARD_DEVIATION = 30
_OVERDUE_SCORE_QUANTILE = 0.15
_OVERDUE_RANDOM_CUTOFF = 0.2


def generate_finance_data(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Create the supplied reproducible dataset and save it as CSV."""
    np.random.seed(RANDOM_STATE)

    data = {
        AGE_COLUMN: np.random.randint(*_AGE_RANGE, N_SAMPLES),
        ANNUAL_INCOME_COLUMN: np.random.normal(
            _ANNUAL_INCOME_MEAN,
            _ANNUAL_INCOME_STANDARD_DEVIATION,
            N_SAMPLES,
        ).round(0),
        SPENDING_SCORE_COLUMN: np.random.randint(
            *_SPENDING_SCORE_RANGE,
            N_SAMPLES,
        ),
        DEBT_RATIO_COLUMN: np.random.uniform(
            *_DEBT_RATIO_RANGE,
            N_SAMPLES,
        ).round(2),
        CREDIT_CARD_COUNT_COLUMN: np.random.randint(
            *_CREDIT_CARD_COUNT_RANGE,
            N_SAMPLES,
        ),
        OVERDUE_COUNT_COLUMN: np.random.poisson(
            _OVERDUE_COUNT_MEAN,
            N_SAMPLES,
        ),
    }
    df = pd.DataFrame(data)
    df[ANNUAL_INCOME_COLUMN] = df[ANNUAL_INCOME_COLUMN].apply(
        lambda value: max(value, _MIN_ANNUAL_INCOME)
    )

    df[REGRESSION_TARGET] = (
        _BASE_CREDIT_SCORE
        + (df[ANNUAL_INCOME_COLUMN] / _INCOME_SCALE) * _INCOME_WEIGHT
        - (df[OVERDUE_COUNT_COLUMN] * _OVERDUE_PENALTY)
        - (df[DEBT_RATIO_COLUMN] * _DEBT_RATIO_PENALTY)
        + np.random.normal(0, _CREDIT_SCORE_NOISE_STANDARD_DEVIATION, N_SAMPLES)
    ).clip(CREDIT_SCORE_MIN, CREDIT_SCORE_MAX).round(0)

    threshold = df[REGRESSION_TARGET].quantile(_OVERDUE_SCORE_QUANTILE)
    df[CLASSIFICATION_TARGET] = np.where(
        (df[REGRESSION_TARGET] < threshold)
        & (np.random.rand(N_SAMPLES) > _OVERDUE_RANDOM_CUTOFF),
        1,
        0,
    )

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return df


def main() -> None:
    df = generate_finance_data()
    print(f"데이터 생성 완료: {DEFAULT_OUTPUT_PATH}")
    print(f"전체 샘플 수: {len(df)}")
    overdue_rate = df[CLASSIFICATION_TARGET].mean() * 100
    print(f"연체(1) 비율: {overdue_rate:.2f}% (불균형 데이터 확인)")


if __name__ == "__main__":
    main()
