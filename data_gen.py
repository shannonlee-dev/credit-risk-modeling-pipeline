"""Backward-compatible entry point for synthetic finance data generation."""

from scripts.generate_data import (
    DEFAULT_OUTPUT_PATH,
    EXPECTED_COLUMNS,
    generate_finance_data,
    main,
)

__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "EXPECTED_COLUMNS",
    "generate_finance_data",
    "main",
]


if __name__ == "__main__":
    main()
