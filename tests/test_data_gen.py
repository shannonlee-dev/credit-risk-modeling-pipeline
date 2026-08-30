from pandas.testing import assert_frame_equal

from data_gen import EXPECTED_COLUMNS, generate_finance_data


def test_generate_finance_data_is_reproducible(tmp_path):
    first = generate_finance_data(tmp_path / "first.csv")
    second = generate_finance_data(tmp_path / "second.csv")

    assert first.shape == (10_000, 8)
    assert list(first.columns) == EXPECTED_COLUMNS
    assert 0.10 <= first["is_overdue"].mean() <= 0.15
    assert first["credit_score"].between(0, 1000).all()
    assert (tmp_path / "first.csv").is_file()
    assert_frame_equal(first, second)
