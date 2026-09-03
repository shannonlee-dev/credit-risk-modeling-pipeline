"""Command-line entry point for the credit-risk pipeline."""

import argparse
from pathlib import Path

from credit_risk.constants import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_PATH
from credit_risk.workflow import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="신용 위험 분류·회귀 모델을 학습하고 평가합니다."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS_DIR))
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
