"""Generate the synthetic finance dataset used by the project."""

import random  # Included to match the supplied generation environment.
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.constants import (
    CLASSIFICATION_TARGET,
    CREDIT_SCORE_MAX,
    CREDIT_SCORE_MIN,
    DEFAULT_DATA_PATH,
    RANDOM_STATE,
    REGRESSION_TARGET,
)


def generate_finance_data(
    output_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """Create the supplied reproducible dataset and save it as CSV."""
    np.random.seed(RANDOM_STATE)
    sample_count = 10_000

    data = {
        "age": np.random.randint(20, 70, sample_count),
        "annual_income": np.random.normal(5_000, 2_000, sample_count).round(0),
        "spending_score": np.random.randint(1, 100, sample_count),
        "debt_ratio": np.random.uniform(0, 1, sample_count).round(2),
        "credit_card_count": np.random.randint(1, 10, sample_count),
        "overdue_count_6m": np.random.poisson(0.5, sample_count),
    }
    df = pd.DataFrame(data)
    df["annual_income"] = df["annual_income"].apply(lambda value: max(value, 1_500))

    df[REGRESSION_TARGET] = (
        300
        + (df["annual_income"] / 100) * 3
        - (df["overdue_count_6m"] * 50)
        - (df["debt_ratio"] * 100)
        + np.random.normal(0, 30, sample_count)
    ).clip(CREDIT_SCORE_MIN, CREDIT_SCORE_MAX).round(0)

    threshold = df[REGRESSION_TARGET].quantile(0.15)
    df[CLASSIFICATION_TARGET] = np.where(
        (df[REGRESSION_TARGET] < threshold)
        & (np.random.rand(sample_count) > 0.2),
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
    print(f"데이터 생성 완료: {DEFAULT_DATA_PATH}")
    print(f"전체 샘플 수: {len(df)}")
    overdue_rate = df[CLASSIFICATION_TARGET].mean() * 100
    print(f"연체(1) 비율: {overdue_rate:.2f}% (불균형 데이터 확인)")


if __name__ == "__main__":
    main()
