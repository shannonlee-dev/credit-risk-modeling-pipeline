from pandas.testing import assert_frame_equal

from scripts.generate_data import EXPECTED_COLUMNS, generate_finance_data


def test_generate_finance_data_is_reproducible(tmp_path):
    first = generate_finance_data(tmp_path / "first.csv")
    second = generate_finance_data(tmp_path / "second.csv")

    assert first.shape == (10_000, 8)
    assert list(first.columns) == EXPECTED_COLUMNS
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
