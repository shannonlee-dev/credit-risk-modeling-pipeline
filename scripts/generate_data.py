"""Generate the synthetic finance dataset used by the project."""

from pathlib import Path
import random  # Included to match the supplied generation environment.

import numpy as np
import pandas as pd

N_SAMPLES = 10_000
RANDOM_STATE = 42
DEFAULT_OUTPUT_PATH = Path("data/generated/finance_data.csv")
EXPECTED_COLUMNS = [
    "age",
    "annual_income",
    "spending_score",
    "debt_ratio",
    "credit_card_count",
    "overdue_count_6m",
    "credit_score",
    "is_overdue",
]


def generate_finance_data(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Create the supplied reproducible dataset and save it as CSV."""
    np.random.seed(RANDOM_STATE)

    data = {
        "age": np.random.randint(20, 70, N_SAMPLES),
        "annual_income": np.random.normal(5000, 2000, N_SAMPLES).round(0),
        "spending_score": np.random.randint(1, 100, N_SAMPLES),
        "debt_ratio": np.random.uniform(0, 1, N_SAMPLES).round(2),
        "credit_card_count": np.random.randint(1, 10, N_SAMPLES),
        "overdue_count_6m": np.random.poisson(0.5, N_SAMPLES),
    }
    df = pd.DataFrame(data)
    df["annual_income"] = df["annual_income"].apply(
        lambda value: max(value, 1500)
    )

    df["credit_score"] = (
        300
        + (df["annual_income"] / 100) * 3
        - (df["overdue_count_6m"] * 50)
        - (df["debt_ratio"] * 100)
        + np.random.normal(0, 30, N_SAMPLES)
    ).clip(0, 1000).round(0)

    threshold = df["credit_score"].quantile(0.15)
    df["is_overdue"] = np.where(
        (df["credit_score"] < threshold)
        & (np.random.rand(N_SAMPLES) > 0.2),
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
    print(f"연체(1) 비율: {df['is_overdue'].mean() * 100:.2f}% (불균형 데이터 확인)")


if __name__ == "__main__":
    main()
