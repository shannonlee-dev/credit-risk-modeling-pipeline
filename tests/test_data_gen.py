from pathlib import Path
import subprocess
import sys

import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.generate_data import generate_finance_data


def test_root_data_gen_script_generates_default_dataset(tmp_path):
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(project_root / "data_gen.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    output_path = tmp_path / "data" / "generated" / "finance_data.csv"
    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    assert len(pd.read_csv(output_path)) == 10_000


def test_generate_finance_data_is_reproducible(tmp_path):
    first = generate_finance_data(tmp_path / "first.csv")
    second = generate_finance_data(tmp_path / "second.csv")

    assert first.shape == (10_000, 8)
    assert list(first.columns) == [
        "age",
        "annual_income",
        "spending_score",
        "debt_ratio",
        "credit_card_count",
        "overdue_count_6m",
        "credit_score",
        "is_overdue",
    ]
    assert 0.10 <= first["is_overdue"].mean() <= 0.15
    assert first["credit_score"].between(0, 1000).all()
    assert (tmp_path / "first.csv").is_file()
    assert_frame_equal(first, second)


def test_generate_finance_data_uses_the_new_default_output_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    generate_finance_data()

    assert (tmp_path / "data" / "generated" / "finance_data.csv").is_file()
