"""Command-line entry point for the credit-risk pipeline."""

import argparse
from pathlib import Path

from credit_risk.constants import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_PATH
from credit_risk.experiments.config import FULL_EXPERIMENT, SMOKE_EXPERIMENT
from credit_risk.workflow import run_all, run_experiment, run_final_evaluation


def _add_execution_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--data",
        default=default if suppress_defaults else str(DEFAULT_DATA_PATH),
    )
    parser.add_argument(
        "--output-dir",
        default=default if suppress_defaults else str(DEFAULT_ARTIFACTS_DIR),
    )
    parser.add_argument(
        "--profile",
        choices=["full", "smoke"],
        default=default if suppress_defaults else "full",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="신용 위험 분류·회귀 모델을 학습하고 평가합니다."
    )
    _add_execution_arguments(parser)
    parser.set_defaults(command="all")
    commands = parser.add_subparsers(dest="command")

    experiment_parser = commands.add_parser("experiment")
    _add_execution_arguments(experiment_parser, suppress_defaults=True)

    final_parser = commands.add_parser("final")
    _add_execution_arguments(final_parser, suppress_defaults=True)
    final_parser.add_argument("--selection", required=True)

    all_parser = commands.add_parser("all")
    _add_execution_arguments(all_parser, suppress_defaults=True)
    parser.set_defaults(command="all")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        profile = FULL_EXPERIMENT if args.profile == "full" else SMOKE_EXPERIMENT
        if args.command == "experiment":
            result = run_experiment(args.data, args.output_dir, profile)
            print(f"실험 완료: {Path(args.output_dir, 'experiment').resolve()}")
            return
        if args.command == "final":
            result = run_final_evaluation(args.data, args.selection, args.output_dir, profile)
        else:
            result = run_all(args.data, args.output_dir, profile)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print("최종 평가 완료")
    print(f"결과 저장 위치: {Path(args.output_dir, 'final').resolve()}")


if __name__ == "__main__":
    main()
