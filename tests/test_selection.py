"""Tests for scorecard.selection using small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from scorecard.selection import (
    cap_by_iv,
    compute_vif,
    cross_validated_gini,
    drop_correlated,
    drop_high_vif,
    filter_by_iv,
    selection_table,
    wrong_sign_variables,
)


def test_filter_by_iv_drops_useless_and_flags_weak() -> None:
    table = filter_by_iv(pd.Series({"strong": 0.5, "weak": 0.05, "useless": 0.01}))
    assert table.loc["strong", "kept"] and not table.loc["strong", "weak"]
    assert table.loc["weak", "kept"] and table.loc["weak", "weak"]
    assert not table.loc["useless", "kept"]


def test_filter_by_iv_threshold_is_inclusive() -> None:
    assert filter_by_iv(pd.Series({"edge": 0.02})).loc["edge", "kept"]


def _correlated_data(n: int = 500) -> pd.DataFrame:
    """a and b are near copies (r ~ 0.99); c is independent."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    return pd.DataFrame({
        "a": a,
        "b": a + rng.normal(scale=0.1, size=n),
        "c": rng.normal(size=n),
    })


def test_drop_correlated_drops_lower_iv_of_pair() -> None:
    kept, decisions = drop_correlated(_correlated_data(), pd.Series({"a": 0.3, "b": 0.5, "c": 0.1}))
    assert kept == ["b", "c"]
    assert decisions["dropped"].tolist() == ["a"]
    assert decisions["kept_instead"].tolist() == ["b"]
    assert decisions["correlation"].iloc[0] > 0.9


def test_drop_correlated_uses_absolute_correlation() -> None:
    data = _correlated_data()
    data["b"] = -data["b"]
    kept, _ = drop_correlated(data, pd.Series({"a": 0.5, "b": 0.3, "c": 0.1}))
    assert kept == ["a", "c"]


def test_drop_correlated_keeps_only_best_of_a_group_of_copies() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=500)
    data = pd.DataFrame({k: base + rng.normal(scale=0.05, size=500) for k in "xyz"})
    kept, decisions = drop_correlated(data, pd.Series({"x": 0.1, "y": 0.3, "z": 0.2}))
    assert kept == ["y"]
    assert set(decisions["dropped"]) == {"x", "z"}
    assert set(decisions["kept_instead"]) == {"y"}


def test_drop_correlated_never_drops_because_of_a_dropped_variable() -> None:
    # Chain: s ~ m (r ~ 0.9), m ~ t (r ~ 0.8), but s and t only r ~ 0.6.
    # IV order t > m > s: t is kept, m drops (correlated with t), and s must
    # then survive, because its only conflict (m) is gone.
    rng = np.random.default_rng(3)
    u, v = rng.normal(size=(2, 2000))
    data = pd.DataFrame({"t": u, "m": u + 0.6 * v, "s": u + 1.0 * v})
    corr = data.corr()
    assert corr.loc["s", "m"] > 0.7 and corr.loc["m", "t"] > 0.7 and corr.loc["s", "t"] < 0.7
    kept, decisions = drop_correlated(data, pd.Series({"t": 0.5, "m": 0.4, "s": 0.3}))
    assert kept == ["t", "s"]
    assert decisions["dropped"].tolist() == ["m"]


def test_drop_correlated_keeps_everything_when_uncorrelated() -> None:
    data = _correlated_data().drop(columns="b")
    kept, decisions = drop_correlated(data, pd.Series({"a": 0.3, "c": 0.1}))
    assert kept == ["a", "c"]
    assert decisions.empty


def test_compute_vif_matches_one_over_one_minus_r_squared() -> None:
    # With two variables, R² of one on the other is r², so VIF = 1 / (1 - r²) for both.
    data = _correlated_data()[["a", "b"]]
    r = data["a"].corr(data["b"])
    vif = compute_vif(data)
    assert vif["a"] == pytest.approx(1 / (1 - r**2))
    assert vif["b"] == pytest.approx(1 / (1 - r**2))


def test_compute_vif_near_one_for_independent_variables() -> None:
    vif = compute_vif(_correlated_data()[["a", "c"]])
    assert (vif < 1.1).all()


def test_drop_high_vif_removes_combination_redundancy() -> None:
    # d = a + c exactly-ish: no single pair is extremely correlated (r ~ 0.7),
    # but together a and c explain d, so VIF catches what pairwise correlation misses.
    rng = np.random.default_rng(2)
    a, c, e = rng.normal(size=(3, 500))
    data = pd.DataFrame({"a": a, "c": c, "d": a + c + rng.normal(scale=0.05, size=500), "e": e})
    kept, steps = drop_high_vif(data, max_vif=5)
    assert len(kept) == 3 and "e" in kept
    assert steps["dropped"].sum() == 1
    assert not steps["dropped"].iloc[-1]
    assert (compute_vif(data[kept]) <= 5).all()


def test_cap_by_iv_keeps_highest() -> None:
    iv = pd.Series({"a": 0.1, "b": 0.5, "c": 0.3})
    assert cap_by_iv(["a", "b", "c"], iv, max_variables=2) == ["b", "c"]
    assert cap_by_iv(["a", "b"], iv, max_variables=5) == ["b", "a"]


def _woe_with_target(n: int = 2000) -> pd.DataFrame:
    """'good' WoE strongly lowers default risk; 'noise' is unrelated."""
    rng = np.random.default_rng(4)
    good, noise = rng.normal(size=(2, n))
    p = 1 / (1 + np.exp(2 * good))
    return pd.DataFrame({"good": good, "noise": noise, "default": (rng.random(n) < p).astype(int)})


def test_cross_validated_gini_high_for_signal_near_zero_for_noise() -> None:
    data = _woe_with_target()
    assert cross_validated_gini(data, ["good"]) > 0.6
    assert abs(cross_validated_gini(data, ["noise"])) < 0.1


def test_wrong_sign_variables_flags_positive_coefficient() -> None:
    data = _woe_with_target()
    assert wrong_sign_variables(data, ["good"]) == []
    data["flipped"] = -data["good"]
    assert wrong_sign_variables(data, ["flipped"]) == ["flipped"]


def test_selection_table_requires_a_reason_for_every_variable() -> None:
    iv = pd.DataFrame({"variable": ["a", "b"], "iv": [0.2, 0.4],
                       "strength": ["medium", "strong"], "type": ["limit", "payment"]})
    table = selection_table(["a", "b"], iv, {"a": "reason a", "b": "reason b"})
    assert table["variable"].tolist() == ["b", "a"]
    assert table.loc[0, "business_reason"] == "reason b"
    with pytest.raises(KeyError, match="a"):
        selection_table(["a"], iv, {})
