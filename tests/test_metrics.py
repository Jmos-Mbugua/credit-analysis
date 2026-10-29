"""Tests for scorecard.metrics on tiny examples whose answers are known."""

import pandas as pd
import pytest

from scorecard.metrics import (
    auc,
    bootstrap_gini,
    calibration_table,
    gini,
    ks,
    ks_table,
    overfitting_verdict,
    performance_summary,
    roc_points,
)

# Scores are "higher = safer". Bads are 1.
PERFECT = (pd.Series([1, 1, 0, 0]), pd.Series([400, 450, 500, 550]))
REVERSED = (pd.Series([1, 1, 0, 0]), pd.Series([550, 500, 450, 400]))
NO_INFORMATION = (pd.Series([1, 0, 1, 0]), pd.Series([500, 500, 500, 500]))
# Hand example: bads score 1, 2, 4; goods score 3, 4, 5.
HAND = (pd.Series([1, 1, 1, 0, 0, 0]), pd.Series([1, 2, 4, 3, 4, 5]))


def test_perfect_separation_gives_gini_one_and_ks_one() -> None:
    assert auc(*PERFECT) == pytest.approx(1.0)
    assert gini(*PERFECT) == pytest.approx(1.0)
    assert ks(*PERFECT)[0] == pytest.approx(1.0)


def test_perfect_separation_ks_occurs_between_the_groups() -> None:
    # All bads are at or below 450 and no goods are, so the gap is 100% at 450.
    assert ks(*PERFECT)[1] == 450


def test_reversed_ranking_gives_gini_minus_one() -> None:
    assert gini(*REVERSED) == pytest.approx(-1.0)
    assert ks(*REVERSED)[0] == pytest.approx(1.0)  # KS measures separation, not direction


def test_identical_scores_give_gini_zero_and_ks_zero() -> None:
    assert gini(*NO_INFORMATION) == pytest.approx(0.0)
    assert ks(*NO_INFORMATION)[0] == pytest.approx(0.0)


def test_gini_hand_calculation_with_tie() -> None:
    # Of the 9 bad/good pairs, the bad scores lower in 7 and ties in 1 (both 4):
    # AUC = (7 + 0.5) / 9, Gini = 2 * AUC - 1 = 2/3.
    assert auc(*HAND) == pytest.approx(7.5 / 9)
    assert gini(*HAND) == pytest.approx(2 / 3)


def test_ks_hand_calculation() -> None:
    # At or below score 2: 2 of 3 bads, 0 of 3 goods -> gap 2/3, the largest.
    value, at_score = ks(*HAND)
    assert value == pytest.approx(2 / 3)
    assert at_score == 2


def test_ks_table_keeps_tied_scores_together() -> None:
    table = ks_table(*HAND).set_index("score")
    assert table.index.tolist() == [1, 2, 3, 4, 5]
    assert table.loc[4, "cum_bads"] == pytest.approx(1.0)
    assert table.loc[4, "cum_goods"] == pytest.approx(2 / 3)
    assert table["cum_bads"].iloc[-1] == table["cum_goods"].iloc[-1] == pytest.approx(1.0)


def test_roc_points_start_at_zero_and_end_at_one() -> None:
    points = roc_points(*HAND)
    assert points.iloc[0].tolist() == [0.0, 0.0]
    assert points.iloc[-1].tolist() == [1.0, 1.0]


def test_performance_summary() -> None:
    summary = performance_summary(*HAND)
    assert summary["customers"] == 6
    assert summary["default_rate"] == pytest.approx(0.5)
    assert summary["gini"] == pytest.approx(2 / 3)
    assert summary["ks_score"] == 2


@pytest.mark.parametrize(("train", "test", "word"), [
    (0.55, 0.54, "No overfitting"),
    (0.55, 0.51, "Watch"),
    (0.55, 0.45, "Overfitting"),
])
def test_overfitting_verdict(train: float, test: float, word: str) -> None:
    assert overfitting_verdict(train, test).startswith(word)


def test_calibration_table_hand_example() -> None:
    # Four customers at PD 0.1 (one default) and four at PD 0.5 (two defaults).
    pd_hat = pd.Series([0.1] * 4 + [0.5] * 4)
    default = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
    table = calibration_table(default, pd_hat, n_groups=2).set_index("decile")
    assert table["customers"].tolist() == [4, 4]
    assert table.loc[1, "predicted_default_rate"] == pytest.approx(0.1)
    assert table.loc[1, "actual_default_rate"] == pytest.approx(0.25)
    assert table.loc[2, "actual_default_rate"] == pytest.approx(0.5)
    assert table.loc[1, "difference"] == pytest.approx(-0.15)


def test_calibration_table_groups_have_equal_size_and_rising_pd() -> None:
    pd_hat = pd.Series([i / 100 for i in range(100)])
    default = pd.Series([i % 2 for i in range(100)])
    table = calibration_table(default, pd_hat)
    assert (table["customers"] == 10).all()
    assert table["predicted_default_rate"].is_monotonic_increasing


def test_bootstrap_gini_interval_contains_estimate_and_is_degenerate_when_perfect() -> None:
    default = pd.Series([1, 0] * 200)
    score = pd.Series([i if i % 2 else i + 150 for i in range(400)])
    low, high = bootstrap_gini(default, score, n_resamples=200)
    assert low <= gini(default, score) <= high
    assert bootstrap_gini(*PERFECT, n_resamples=50) == pytest.approx((1.0, 1.0))
