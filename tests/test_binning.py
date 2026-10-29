"""Tests for scorecard.binning, using small tables whose answers can be checked by hand."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scorecard.binning import (
    assign_bins,
    load_binnings,
    save_binnings,
    BinSpec,
    bin_labels,
    fine_edges,
    fit_binning,
    iv_strength,
    merge_bins,
    transform_woe,
    woe_iv,
)


def test_woe_iv_hand_calculation() -> None:
    # Bin A: 60 of 80 goods (75%), 5 of 20 bads (25%); bin B: 20 goods (25%), 15 bads (75%).
    woe, iv = woe_iv(np.array([60, 20]), np.array([5, 15]))
    assert woe[0] == pytest.approx(math.log(0.75 / 0.25))
    assert woe[1] == pytest.approx(math.log(0.25 / 0.75))
    assert iv.sum() == pytest.approx(2 * 0.5 * math.log(3))


def test_woe_iv_uses_adjustment_for_empty_bins() -> None:
    woe, _ = woe_iv(np.array([10, 10]), np.array([0, 10]))
    assert np.isfinite(woe).all()
    assert woe[0] == pytest.approx(math.log(0.5 / (0.5 / 10)))


@pytest.mark.parametrize(
    ("iv", "label"),
    [(0.01, "useless"), (0.02, "weak"), (0.15, "medium"), (0.4, "strong"), (0.8, "suspicious")],
)
def test_iv_strength(iv: float, label: str) -> None:
    assert iv_strength(iv) == label


def test_fine_edges_adds_zero_when_values_span_it() -> None:
    edges = fine_edges(pd.Series([-5.0, -1.0, 3.0, 7.0, 9.0]), BinSpec(n_fine=2))
    assert 0.0 in edges
    assert edges[0] == -np.inf and edges[-1] == np.inf


def test_fine_edges_keeps_fixed_edges_inside_range() -> None:
    edges = fine_edges(pd.Series([0, 1, 2, 5]), BinSpec(fixed_edges=(-2, -1, 0, 1, 2)))
    assert edges == [-np.inf, 0, 1, 2, np.inf]


def test_merge_bins_merges_small_bin_with_closest_neighbour() -> None:
    # Middle bin has 2% of customers; its default rate (0.5) is closer to the right bin (0.6).
    goods, bads = np.array([45, 1, 20]), np.array([5, 1, 30])
    edges = merge_bins([-np.inf, 1, 2, np.inf], goods, bads, n_total=102,
                       spec=BinSpec(min_share=0.05, monotonic=False))
    assert edges == [-np.inf, 1, np.inf]


def test_merge_bins_enforces_monotonic_rate() -> None:
    # Default rates 0.10, 0.30, 0.25, 0.50: the 0.30 -> 0.25 dip must be merged away.
    goods, bads = np.array([90, 70, 75, 50]), np.array([10, 30, 25, 50])
    edges = merge_bins([-np.inf, 1, 2, 3, np.inf], goods, bads, n_total=400,
                       spec=BinSpec(min_share=0.05), trend=1)
    assert edges == [-np.inf, 1, 3, np.inf]


def test_merge_bins_respects_falling_trend() -> None:
    goods, bads = np.array([50, 75, 70, 90]), np.array([50, 25, 30, 10])
    edges = merge_bins([-np.inf, 1, 2, 3, np.inf], goods, bads, n_total=400,
                       spec=BinSpec(min_share=0.05), trend=-1)
    assert edges == [-np.inf, 1, 3, np.inf]


def test_bin_labels_integer_and_continuous() -> None:
    assert bin_labels([-np.inf, -1, 0, 2, np.inf], integer_valued=True) == ["<= -1", "0", "1 to 2", ">= 3"]
    assert bin_labels([-np.inf, 0.1, 0.5, np.inf], integer_valued=False) == ["<= 0.1", "(0.1, 0.5]", "> 0.5"]


def _risk_rises_with_x() -> tuple[pd.Series, pd.Series]:
    """200 customers: x = 0..199, the top half default far more often."""
    x = pd.Series(np.arange(200, dtype=float), name="x")
    y = pd.Series(((np.arange(200) % 10) < np.where(np.arange(200) < 100, 1, 6)).astype(int))
    return x, y


def test_fit_binning_produces_monotonic_bins_meeting_minimum_size() -> None:
    x, y = _risk_rises_with_x()
    binning = fit_binning(x, y, BinSpec(n_fine=10, min_share=0.05))
    assert (binning.table["share"] >= 0.05).all()
    assert binning.table["default_rate"].is_monotonic_increasing
    assert binning.iv > 0


def test_fit_binning_gives_missing_values_their_own_bin() -> None:
    x, y = _risk_rises_with_x()
    x.iloc[:10] = np.nan
    binning = fit_binning(x, y, BinSpec(n_fine=10))
    assert "Missing" in binning.table.index
    assert binning.table.loc["Missing", "customers"] == 10


def test_transform_woe_maps_values_to_their_bin_woe() -> None:
    x, y = _risk_rises_with_x()
    binning = fit_binning(x, y, BinSpec(n_fine=10))
    woe = transform_woe(pd.Series([0.0, 199.0, np.nan]), binning)
    assert woe.iloc[0] == pytest.approx(binning.table["woe"].iloc[0])
    assert woe.iloc[1] == pytest.approx(binning.table["woe"].iloc[-1])
    assert woe.iloc[2] == 0.0  # no missing values in training: neutral evidence


def test_assign_bins_returns_labels_including_missing() -> None:
    x, y = _risk_rises_with_x()
    binning = fit_binning(x, y, BinSpec(n_fine=10))
    labels = assign_bins(pd.Series([0.0, 199.0, np.nan]), binning)
    assert labels.iloc[0] == binning.table.index[0]
    assert labels.iloc[1] == binning.table.index[-1]
    assert labels.iloc[2] == "Missing"


def test_save_and_load_binnings_round_trip(tmp_path: Path) -> None:
    x, y = _risk_rises_with_x()
    binnings = {"x": fit_binning(x, y, BinSpec(n_fine=10))}
    save_binnings(binnings, tmp_path / "binnings.pkl")
    loaded = load_binnings(tmp_path / "binnings.pkl")
    assert loaded["x"].edges == binnings["x"].edges
    assert transform_woe(x, loaded["x"]).equals(transform_woe(x, binnings["x"]))
