"""Tests for scorecard.scaling, using a small hand-made model and binning."""

import math

import numpy as np
import pandas as pd
import pytest

from scorecard.binning import Binning
from scorecard.scaling import (
    bin_points,
    build_scorecard,
    check_score_pd_agreement,
    customer_breakdown,
    pd_to_score,
    points_spread,
    score_band_table,
    score_customers,
    score_range,
    score_to_pd,
    scaling_constants,
)

FACTOR, OFFSET = scaling_constants()


def test_factor_and_offset_for_default_settings() -> None:
    # Factor = 20 / ln 2; Offset = 600 - Factor * ln 50.
    assert FACTOR == pytest.approx(28.8539, abs=1e-4)
    assert OFFSET == pytest.approx(487.1229, abs=1e-4)


def test_base_score_has_base_odds_and_pdo_doubles_odds() -> None:
    assert score_to_pd(600, FACTOR, OFFSET) == pytest.approx(1 / 51)  # odds 50:1
    odds = lambda s: (1 - score_to_pd(s, FACTOR, OFFSET)) / score_to_pd(s, FACTOR, OFFSET)  # noqa: E731
    assert odds(620) == pytest.approx(2 * odds(600))
    assert odds(580) == pytest.approx(odds(600) / 2)


def test_higher_score_always_means_lower_pd() -> None:
    scores = np.arange(200, 1001)
    pds = score_to_pd(scores, FACTOR, OFFSET)
    assert (np.diff(pds) < 0).all()


def test_pd_to_score_inverts_score_to_pd() -> None:
    for score in [350.0, 512.3, 600.0, 777.0]:
        assert pd_to_score(score_to_pd(score, FACTOR, OFFSET), FACTOR, OFFSET) == pytest.approx(score)


# --- a two-variable model with hand-made bins ----------------------------------------

COEFFICIENTS = pd.DataFrame({"variable": ["const", "late", "limit"],
                             "coefficient": [-1.2, -0.7, -0.4]})
BINNING_TABLES = pd.DataFrame({
    "variable": ["late", "late", "late", "limit", "limit"],
    "bin": ["<= 0", "1", ">= 2", "<= 50000", ">= 50001"],
    "woe": [0.6, -0.5, -2.0, -0.4, 0.3],
})
BINNINGS = {
    "late": Binning("late", [-np.inf, 0, 1, np.inf], pd.DataFrame(), integer_valued=True),
    "limit": Binning("limit", [-np.inf, 50000, np.inf], pd.DataFrame(), integer_valued=True),
}


def _scorecard() -> pd.DataFrame:
    return build_scorecard(COEFFICIENTS, BINNING_TABLES, FACTOR, OFFSET)


def test_bin_points_formula() -> None:
    expected = -FACTOR * -0.7 * 0.6 + (OFFSET - FACTOR * -1.2) / 2
    assert bin_points(-0.7, 0.6, -1.2, 2, FACTOR, OFFSET) == pytest.approx(expected)


def test_build_scorecard_rounds_points_and_keeps_bin_order() -> None:
    card = _scorecard()
    assert card["bin"].tolist() == BINNING_TABLES["bin"].tolist()
    assert card["points"].dtype == int
    assert card.loc[0, "points"] == round(bin_points(-0.7, 0.6, -1.2, 2, FACTOR, OFFSET))


def test_safer_bin_gets_more_points() -> None:
    late = _scorecard().query("variable == 'late'")["points"].tolist()
    assert late[0] > late[1] > late[2]  # higher WoE = safer = more points


def test_total_points_equal_model_log_odds_as_score() -> None:
    # Example customer: 0 months late (WoE 0.6) and limit 80,000 (WoE 0.3).
    customer = pd.DataFrame({"late": [0], "limit": [80000]})
    total = score_customers(customer, _scorecard(), BINNINGS).iloc[0]
    log_odds_default = -1.2 + -0.7 * 0.6 + -0.4 * 0.3
    exact = OFFSET + FACTOR * -log_odds_default
    assert abs(total - exact) <= 0.5 * 2  # each of 2 variables rounded by at most 0.5


def test_score_customers_and_breakdown_agree() -> None:
    customers = pd.DataFrame({"late": [0, 1, 5], "limit": [80000, 20000, 50000]})
    card = _scorecard()
    scores = score_customers(customers, card, BINNINGS)
    breakdown = customer_breakdown(customers.iloc[2], card, BINNINGS)
    assert breakdown["bin"].tolist() == [">= 2", "<= 50000"]
    assert breakdown["points"].sum() == scores.iloc[2]
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]


def test_score_range_sums_min_and_max_points_per_variable() -> None:
    card = _scorecard()
    lo, hi = score_range(card)
    by_var = card.groupby("variable")["points"]
    assert (lo, hi) == (by_var.min().sum(), by_var.max().sum())


def test_check_score_pd_agreement_passes_within_rounding_and_fails_beyond() -> None:
    model_pd = pd.Series([0.05, 0.2, 0.6])
    exact = pd_to_score(model_pd, FACTOR, OFFSET)
    assert check_score_pd_agreement(exact.round(), model_pd, 2, FACTOR, OFFSET) <= 0.5
    with pytest.raises(ValueError, match="disagree"):
        check_score_pd_agreement(exact + 5, model_pd, 2, FACTOR, OFFSET)


def test_score_band_table_counts_and_bad_rates() -> None:
    scores = pd.Series([505, 515, 519, 520, 539, 545])
    default = pd.Series([1, 1, 0, 0, 1, 0])
    table = score_band_table(scores, default, width=20).set_index("band")
    assert table.index.tolist() == ["500-519", "520-539", "540-559"]
    assert table.loc["500-519", "customers"] == 3
    assert table.loc["500-519", "bad_rate"] == pytest.approx(2 / 3)
    assert table.loc["520-539", "bad_rate"] == pytest.approx(0.5)
    assert table["cumulative_share"].iloc[-1] == pytest.approx(1.0)
    assert math.isclose(table["share"].sum(), 1.0)


def test_points_spread() -> None:
    card = pd.DataFrame({"variable": ["a", "a", "b", "b", "b"], "points": [10, 30, 5, 6, 9]})
    spread = points_spread(card).set_index("variable")
    assert spread.index.tolist() == ["a", "b"]
    assert spread.loc["a", "spread"] == 20
    assert (spread.loc["b", "min_points"], spread.loc["b", "max_points"]) == (5, 9)
