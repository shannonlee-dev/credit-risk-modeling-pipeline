"""Compatibility entry point for generating the finance dataset."""

from scripts.generate_data import generate_finance_data, main

__all__ = ["generate_finance_data", "main"]


if __name__ == "__main__":
    main()
