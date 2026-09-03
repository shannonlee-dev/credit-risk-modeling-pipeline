"""Backward-compatible CLI for the modular credit-risk pipeline."""

import argparse
from pathlib import Path

from credit_risk.classification import (
    evaluate_classifier,
    rule_based_predict,
    train_classification,
)
from credit_risk.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    TARGET_COLUMNS,
    class_distribution,
    load_and_validate_data,
    split_classification_data,
    split_regression_data,
)
from credit_risk.preprocessing import build_preprocessor
from credit_risk.regression import ALPHAS, train_regression
from credit_risk.reporting import metrics_report as _metrics_report
from credit_risk.workflow import run_analysis, run_classification, run_regression

__all__ = [
    "ALPHAS",
    "CATEGORICAL_FEATURES",
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "RANDOM_STATE",
    "REQUIRED_COLUMNS",
    "TARGET_COLUMNS",
    "build_preprocessor",
    "class_distribution",
    "evaluate_classifier",
    "load_and_validate_data",
    "rule_based_predict",
    "run_analysis",
    "run_classification",
    "run_regression",
    "split_classification_data",
    "split_regression_data",
    "train_classification",
    "train_regression",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="신용 위험 분류·회귀 모델을 학습하고 평가합니다."
    )
    parser.add_argument("--data", default="data/generated/finance_data.csv")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    try:
        result = run_analysis(args.data, args.output_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print("전체 분석 완료")
    for split_name, distribution in result["data_distribution"].items():
        print(
            f"{split_name}: 정상={distribution['count_0']}, "
            f"연체={distribution['count_1']}, "
            f"양성비율={distribution['positive_rate']:.2%}"
        )
    print(f"결과 저장 위치: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
