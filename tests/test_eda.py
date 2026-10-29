"""Tests for scorecard.eda, using hand-made tables whose answers can be checked by hand."""

import numpy as np
import pandas as pd
import pytest

from scorecard.eda import (
    customers_with_negative_bill,
    default_rate_by,
    fixed_bands,
    negative_bill_summary,
    quantile_bands,
)


def test_default_rate_by_counts_and_rates() -> None:
    df = pd.DataFrame({"pay_1": [0, 0, 0, 0, 2, 2], "default": [0, 0, 0, 1, 1, 1]})
    table = default_rate_by(df, "pay_1")
    assert table.loc[0, "customers"] == 4
    assert table.loc[0, "defaults"] == 1
    assert table.loc[0, "default_rate"] == pytest.approx(0.25)
    assert table.loc[2, "default_rate"] == pytest.approx(1.0)
    assert table["share"].tolist() == pytest.approx([4 / 6, 2 / 6])


def test_quantile_bands_makes_equal_sized_groups() -> None:
    bands = quantile_bands(pd.Series(range(1, 11)), q=5)
    assert bands.value_counts().tolist() == [2] * 5
    assert bands.iloc[0] == "1-3"
    assert bands.cat.ordered


def test_fixed_bands_labels_and_open_top() -> None:
    bands = fixed_bands(pd.Series([21, 25, 26, 70]), edges=[20, 25, 30, np.inf])
    assert bands.astype(str).tolist() == ["21-25", "21-25", "26-30", "31+"]


def _bills() -> pd.DataFrame:
    data = {f"bill_amt{i}": [100, 100, 100] for i in range(1, 7)}
    data["bill_amt1"] = [-5, 100, -1]
    data["bill_amt3"] = [-5, 100, 100]
    return pd.DataFrame(data)


def test_negative_bill_summary_counts_per_month() -> None:
    summary = negative_bill_summary(_bills())
    assert summary.loc["bill_amt1", "negative_bills"] == 2
    assert summary.loc["bill_amt3", "negative_bills"] == 1
    assert summary.loc["bill_amt2", "negative_bills"] == 0
    assert summary.loc["bill_amt1", "share"] == pytest.approx(2 / 3)


def test_customers_with_negative_bill_counts_each_customer_once() -> None:
    assert customers_with_negative_bill(_bills()) == 2
