"""Tests for scorecard.strategy on small hand-made portfolios."""

import numpy as np
import pandas as pd
import pytest

from scorecard.strategy import (
    ProfitAssumptions,
    average_balance,
    best_cutoff,
    break_even_pd,
    expected_profit,
    profit_by_zone,
    strategy_table,
    three_zone_decision,
    zone_summary,
)

# Ten customers: scores 500..590 in steps of 10; the three lowest and the 6th are bads.
SCORES = pd.Series([500, 510, 520, 530, 540, 550, 560, 570, 580, 590])
DEFAULTS = pd.Series([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])


def test_strategy_table_hand_calculation() -> None:
    row = strategy_table(SCORES, DEFAULTS, [530]).iloc[0]
    # Approve 530 and above: 7 customers, 1 bad among them (score 550).
    assert row["approved"] == 7
    assert row["approval_rate"] == pytest.approx(0.7)
    assert row["approved_bad_rate"] == pytest.approx(1 / 7)
    assert row["declined_bad_rate"] == pytest.approx(1.0)  # 500, 510, 520 are all bads
    assert row["bads_declined_share"] == pytest.approx(3 / 4)
    assert row["goods_declined_share"] == pytest.approx(0.0)


def test_strategy_table_cutoff_is_inclusive() -> None:
    assert strategy_table(SCORES, DEFAULTS, [500]).loc[0, "approval_rate"] == 1.0
    assert strategy_table(SCORES, DEFAULTS, [501]).loc[0, "approval_rate"] == 0.9


def test_strategy_table_stricter_cutoff_approves_fewer_and_declines_more_bads() -> None:
    table = strategy_table(SCORES, DEFAULTS, list(range(490, 610, 10)))
    assert table["approval_rate"].is_monotonic_decreasing
    assert table["bads_declined_share"].is_monotonic_increasing


def test_strategy_table_empty_groups_give_nan_bad_rates() -> None:
    table = strategy_table(SCORES, DEFAULTS, [400, 700]).set_index("cutoff")
    assert np.isnan(table.loc[400, "declined_bad_rate"])  # nobody declined
    assert np.isnan(table.loc[700, "approved_bad_rate"])  # nobody approved
    assert table.loc[700, "bads_declined_share"] == 1.0


@pytest.mark.parametrize(("score", "decision"), [
    (499, "Decline"), (500, "Refer"), (539, "Refer"), (540, "Approve"), (600, "Approve"),
])
def test_three_zone_decision_boundaries(score: float, decision: str) -> None:
    assert three_zone_decision(score, decline_below=500, approve_from=540) == decision


def test_three_zone_decision_without_refer_zone() -> None:
    assert three_zone_decision(519, 520, 520) == "Decline"
    assert three_zone_decision(520, 520, 520) == "Approve"


def test_three_zone_decision_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        three_zone_decision(500, decline_below=550, approve_from=520)


def test_zone_summary_counts_and_bad_rates() -> None:
    table = zone_summary(SCORES, DEFAULTS, decline_below=520, approve_from=560).set_index("zone")
    assert table["customers"].tolist() == [2, 4, 4]  # 500-510 | 520-550 | 560-590
    assert table.loc["Decline", "bad_rate"] == 1.0
    assert table.loc["Refer", "bad_rate"] == pytest.approx(2 / 4)
    assert table.loc["Approve", "bad_rate"] == 0.0
    assert table["share_of_all_bads"].sum() == pytest.approx(1.0)


def test_break_even_pd() -> None:
    assert break_even_pd(ProfitAssumptions(margin_rate=0.08, lgd=0.60)) == pytest.approx(0.08 / 0.68)
    assert break_even_pd(ProfitAssumptions(margin_rate=0.1, lgd=0.1)) == pytest.approx(0.5)


def test_average_balance_floors_credit_balances_at_zero() -> None:
    df = pd.DataFrame({f"bill_amt{i}": [600, -600] for i in range(1, 7)})
    assert average_balance(df).tolist() == [600, 0]


def test_expected_profit_hand_calculation_and_best_cutoff() -> None:
    scores = pd.Series([500, 520, 540])
    defaults = pd.Series([1, 0, 0])
    balances = pd.Series([1000.0, 1000.0, 1000.0])
    table = expected_profit(scores, defaults, balances, [500, 510, 530, 600],
                            ProfitAssumptions(margin_rate=0.1, lgd=0.5)).set_index("cutoff")
    # Approve all: 2 goods x 100 - 1 bad x 500 = -300. Drop the bad: +200. Only 540: +100.
    assert table.loc[500, "profit"] == pytest.approx(-300)
    assert table.loc[510, "profit"] == pytest.approx(200)
    assert table.loc[530, "profit"] == pytest.approx(100)
    assert table.loc[600, "profit"] == 0
    assert best_cutoff(table.reset_index())["cutoff"] == 510


def test_best_cutoff_breaks_ties_towards_higher_approval() -> None:
    table = pd.DataFrame({"cutoff": [500, 510, 520], "profit": [5.0, 7.0, 7.0]})
    assert best_cutoff(table)["cutoff"] == 510


def test_profit_by_zone_hand_calculation() -> None:
    scores = pd.Series([500, 520, 540, 560])
    defaults = pd.Series([1, 0, 1, 0])
    balances = pd.Series([1000.0, 1000.0, 2000.0, 500.0])
    profit = profit_by_zone(scores, defaults, balances, decline_below=510, approve_from=550,
                            assumptions=ProfitAssumptions(margin_rate=0.1, lgd=0.5))
    assert profit["Decline"] == pytest.approx(-500)         # bad, 1000 x 0.5
    assert profit["Refer"] == pytest.approx(100 - 1000)     # good 1000 x 0.1, bad 2000 x 0.5
    assert profit["Approve"] == pytest.approx(50)           # good 500 x 0.1
