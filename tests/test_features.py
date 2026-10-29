"""Tests for scorecard.features on a hand-made two-customer table."""

import numpy as np
import pandas as pd
import pytest

from scorecard.features import CANDIDATES, add_behavioural_features, payment_ratios, utilisation


def _two_customers() -> pd.DataFrame:
    data = {"limit_bal": [1000, 2000]}
    data.update({f"pay_{i}": [0, 0] for i in range(1, 7)})
    data["pay_1"], data["pay_3"] = [2, -1], [1, -2]
    data.update({f"bill_amt{i}": [500, 0] for i in range(1, 7)})
    data["bill_amt1"], data["bill_amt6"] = [800, -100], [200, 0]
    data.update({f"pay_amt{i}": [100, 0] for i in range(1, 7)})
    data["pay_amt1"] = [1000, 0]
    return pd.DataFrame(data)


def test_utilisation_is_bill_over_limit() -> None:
    util = utilisation(_two_customers())
    assert util.loc[0, "util1"] == pytest.approx(0.8)
    assert util.loc[1, "util1"] == pytest.approx(-0.05)


def test_payment_ratio_matches_payment_to_previous_bill_and_caps_at_one() -> None:
    ratios = payment_ratios(_two_customers())
    assert ratios.loc[0, "pay_ratio1"] == 1.0  # paid 1000 against a 500 bill
    assert ratios.loc[0, "pay_ratio2"] == pytest.approx(100 / 500)
    assert np.isnan(ratios.loc[1, "pay_ratio1"])  # nothing was due


def test_add_behavioural_features() -> None:
    out = add_behavioural_features(_two_customers())
    assert out["max_delinquency_6m"].tolist() == [2, 0]
    assert out["months_delinquent_6m"].tolist() == [2, 0]
    assert out.loc[0, "max_util_6m"] == pytest.approx(0.8)
    assert out.loc[0, "util_trend"] == pytest.approx(0.8 - 0.2)
    assert out.loc[0, "avg_util_6m"] == pytest.approx((800 + 500 * 4 + 200) / 6 / 1000)
    assert out.loc[0, "avg_pay_ratio_6m"] == pytest.approx((1 + 0.2 + 0.2 + 0.2 + 100 / 200) / 5)
    assert np.isnan(out.loc[1, "avg_pay_ratio_6m"])
    assert out["months_zero_payment_6m"].tolist() == [0, 6]


def test_protected_characteristics_are_never_candidates() -> None:
    assert "sex" not in CANDIDATES
    assert "marriage" not in CANDIDATES
    assert "id" not in CANDIDATES
