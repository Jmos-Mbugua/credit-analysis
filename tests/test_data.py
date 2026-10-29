"""Tests for scorecard.data. They use tiny made-up tables, so no internet is needed."""

import pandas as pd
import pytest

from scorecard.data import (
    FEATURE_NAMES,
    clean,
    default_rate,
    split_train_test,
    standardise_columns,
    validate,
)


def _ucimlrepo_style(n: int = 4) -> pd.DataFrame:
    """A small table named the way ucimlrepo names it (ID, X1..X23, Y)."""
    data = {"ID": range(1, n + 1)}
    data.update({f"X{i}": [i] * n for i in range(1, 24)})
    data["Y"] = [0, 1, 0, 0][:n]
    return pd.DataFrame(data)


def _excel_style(n: int = 4) -> pd.DataFrame:
    """A small table named the way the UCI Excel file names it."""
    data = {"ID": range(1, n + 1)}
    data.update({name: [1] * n for name in FEATURE_NAMES})
    data["default payment next month"] = [1, 1, 0, 0][:n]
    return pd.DataFrame(data)


def test_standardise_columns_renames_ucimlrepo_names() -> None:
    out = standardise_columns(_ucimlrepo_style())
    assert list(out.columns) == ["ID", *FEATURE_NAMES, "default"]
    assert out["LIMIT_BAL"].iloc[0] == 1  # X1
    assert out["PAY_AMT6"].iloc[0] == 23  # X23


def test_standardise_columns_renames_excel_names() -> None:
    out = standardise_columns(_excel_style())
    assert list(out.columns) == ["ID", *FEATURE_NAMES, "default"]
    assert out["default"].tolist() == [1, 1, 0, 0]


def test_standardise_columns_adds_id_when_missing() -> None:
    out = standardise_columns(_ucimlrepo_style().drop(columns="ID"))
    assert out["ID"].tolist() == [1, 2, 3, 4]


def test_standardise_columns_raises_when_feature_missing() -> None:
    with pytest.raises(ValueError, match="missing"):
        standardise_columns(_ucimlrepo_style().drop(columns="X5"))


def test_validate_accepts_good_table() -> None:
    validate(standardise_columns(_ucimlrepo_style()), expected_rows=4)


def test_validate_rejects_wrong_row_count() -> None:
    with pytest.raises(ValueError, match="rows"):
        validate(standardise_columns(_ucimlrepo_style()), expected_rows=5)


def test_validate_rejects_non_binary_target() -> None:
    df = standardise_columns(_ucimlrepo_style())
    df.loc[0, "default"] = 2
    with pytest.raises(ValueError, match="0 and 1"):
        validate(df, expected_rows=4)


def test_validate_rejects_missing_values() -> None:
    df = standardise_columns(_ucimlrepo_style()).astype("float64")
    df.loc[0, "AGE"] = None
    with pytest.raises(ValueError, match="missing"):
        validate(df, expected_rows=4)


def test_validate_rejects_duplicate_ids() -> None:
    df = standardise_columns(_ucimlrepo_style())
    df.loc[1, "ID"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        validate(df, expected_rows=4)


def test_default_rate() -> None:
    df = pd.DataFrame({"default": [1, 0, 0, 0]})
    assert default_rate(df) == pytest.approx(0.25)


# --- clean() -------------------------------------------------------------------

def _raw_style() -> pd.DataFrame:
    """A 4-customer table shaped like data/raw/credit_card_default.csv."""
    data = {"ID": [1, 2, 3, 4]}
    data.update({name: [1, 1, 1, 1] for name in FEATURE_NAMES})
    data["EDUCATION"] = [0, 5, 6, 2]
    data["MARRIAGE"] = [0, 1, 2, 3]
    data["PAY_0"] = [-2, -1, 0, 2]
    data["default"] = [0, 0, 1, 1]
    return pd.DataFrame(data)


def test_clean_renames_pay_0_to_pay_1() -> None:
    out = clean(_raw_style())
    status_columns = out.columns[out.columns.str.fullmatch(r"pay_\d")].tolist()
    assert status_columns == [f"pay_{i}" for i in range(1, 7)]
    assert out["pay_1"].tolist() == [-2, -1, 0, 2]
    assert "pay_0" not in out.columns


def test_clean_groups_undocumented_education_into_other() -> None:
    assert clean(_raw_style())["education"].tolist() == [4, 4, 4, 2]


def test_clean_groups_undocumented_marriage_into_other() -> None:
    assert clean(_raw_style())["marriage"].tolist() == [3, 1, 2, 3]


def test_clean_lowercases_all_columns() -> None:
    out = clean(_raw_style())
    assert all(c == c.lower() for c in out.columns)
    assert {"id", "limit_bal", "default"} <= set(out.columns)


def test_clean_does_not_modify_input() -> None:
    raw = _raw_style()
    clean(raw)
    assert "PAY_0" in raw.columns
    assert raw["EDUCATION"].tolist() == [0, 5, 6, 2]


def test_clean_rejects_duplicate_ids() -> None:
    raw = _raw_style()
    raw.loc[3, "ID"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        clean(raw)


def test_clean_rejects_missing_values() -> None:
    raw = _raw_style().astype("float64")
    raw.loc[0, "AGE"] = None
    with pytest.raises(ValueError, match="missing"):
        clean(raw)


# --- split_train_test() -----------------------------------------------------------

def test_split_train_test_keeps_default_rate_and_sizes() -> None:
    df = pd.DataFrame({"id": range(100), "default": [1] * 20 + [0] * 80})
    train, test = split_train_test(df, test_size=0.3, seed=0)
    assert (len(train), len(test)) == (70, 30)
    assert default_rate(train) == pytest.approx(0.2)
    assert default_rate(test) == pytest.approx(0.2)
    assert set(train["id"]).isdisjoint(test["id"])
