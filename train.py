"""Command-line entry point for the credit-risk pipeline."""

import argparse
from pathlib import Path

from credit_risk.constants import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_PATH
from credit_risk.experiments.config import FULL_EXPERIMENT, SMOKE_EXPERIMENT
from credit_risk.workflow import run_all, run_experiment, run_final_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="신용 위험 분류·회귀 모델을 학습하고 평가합니다."
    )
    parser.add_argument("command", nargs="?", choices=["all", "experiment", "final"], default="all")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--selection")
    args = parser.parse_args()

    try:
        profile = FULL_EXPERIMENT if args.profile == "full" else SMOKE_EXPERIMENT
        if args.command == "experiment":
            result = run_experiment(args.data, args.output_dir, profile)
            print(f"실험 완료: {Path(args.output_dir, 'experiment').resolve()}")
            return
        if args.command == "final":
            if args.selection is None:
                parser.error("final requires --selection")
            result = run_final_evaluation(args.data, args.selection, args.output_dir, profile)
        else:
            result = run_all(args.data, args.output_dir, profile)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print("최종 평가 완료")
    print(f"결과 저장 위치: {Path(args.output_dir, 'final').resolve()}")


if __name__ == "__main__":
    main()
