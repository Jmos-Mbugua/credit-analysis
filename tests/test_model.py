"""Tests for scorecard.model on synthetic WoE data with a known answer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scorecard.model import (
    average_pd_vs_actual,
    coefficients_table,
    find_offender,
    fit_logit,
    fit_with_checks,
    load_model,
    odds_effect_by_variable,
    predict_pd,
    save_model,
)


def _synthetic_woe(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """'strong' and 'medium' lower risk (true coefficients -1.0 and -0.5),
    'noise' is unrelated, and 'flipped' raises risk (wrong sign for a WoE variable)."""
    rng = np.random.default_rng(seed)
    strong, medium, noise, flipped = rng.normal(size=(4, n))
    log_odds = -1.2 - 1.0 * strong - 0.5 * medium + 0.4 * flipped
    default = (rng.random(n) < 1 / (1 + np.exp(-log_odds))).astype(int)
    return pd.DataFrame({"strong": strong, "medium": medium, "noise": noise,
                         "flipped": flipped, "default": default})


def test_find_offender_returns_none_when_rules_pass() -> None:
    result = fit_logit(_synthetic_woe(), ["strong", "medium"])
    assert find_offender(result) is None


def test_find_offender_flags_positive_coefficient() -> None:
    result = fit_logit(_synthetic_woe(), ["strong", "medium", "flipped"])
    variable, reason = find_offender(result)
    assert variable == "flipped"
    assert "positive coefficient" in reason


def test_find_offender_flags_insignificant_variable() -> None:
    result = fit_logit(_synthetic_woe(), ["strong", "medium", "noise"])
    variable, reason = find_offender(result)
    assert variable == "noise"
    assert "p-value" in reason


def test_find_offender_picks_highest_p_value_first() -> None:
    data = _synthetic_woe()
    rng = np.random.default_rng(9)
    data["noise2"] = rng.normal(size=len(data))
    result = fit_logit(data, ["strong", "noise", "noise2"])
    variable, _ = find_offender(result)
    p = result.pvalues
    assert variable == ("noise" if p["noise"] > p["noise2"] else "noise2")


def test_fit_with_checks_ends_with_negative_significant_coefficients() -> None:
    result, kept, steps = fit_with_checks(_synthetic_woe(), ["strong", "medium", "noise", "flipped"])
    assert sorted(kept) == ["medium", "strong"]
    assert set(steps["variable"]) == {"noise", "flipped"}
    coefs = result.params.drop("const")
    assert (coefs < 0).all()
    assert (result.pvalues.drop("const") < 0.05).all()


def test_fit_with_checks_recovers_true_coefficients() -> None:
    result, _, _ = fit_with_checks(_synthetic_woe(n=20000), ["strong", "medium"])
    assert result.params["strong"] == pytest.approx(-1.0, abs=0.1)
    assert result.params["medium"] == pytest.approx(-0.5, abs=0.1)


def test_predict_pd_mean_matches_default_rate_on_training_data() -> None:
    # A logistic regression with an intercept reproduces the training default rate exactly.
    data = _synthetic_woe()
    result = fit_logit(data, ["strong", "medium"])
    pd_hat = predict_pd(result, data, ["strong", "medium"])
    assert pd_hat.between(0, 1).all()
    assert pd_hat.mean() == pytest.approx(data["default"].mean(), abs=1e-6)


def test_coefficients_table_columns_and_odds_ratio() -> None:
    table = coefficients_table(fit_logit(_synthetic_woe(), ["strong"]))
    assert table["variable"].tolist() == ["const", "strong"]
    assert {"coefficient", "std_error", "p_value"} <= set(table.columns)
    assert table["odds_ratio"].tolist() == pytest.approx(np.exp(table["coefficient"]).tolist())


def test_odds_effect_by_variable_hand_calculation() -> None:
    coefs = pd.DataFrame({"variable": ["const", "x"], "coefficient": [-1.0, -0.5]})
    bins = pd.DataFrame({"variable": ["x", "x", "x"], "bin": ["low", "mid", "high"],
                         "woe": [-1.0, 0.2, 1.0]})
    effect = odds_effect_by_variable(coefs, bins).iloc[0]
    assert (effect["riskiest_bin"], effect["safest_bin"]) == ("low", "high")
    assert effect["woe_range"] == pytest.approx(2.0)
    assert effect["odds_divided_by"] == pytest.approx(np.exp(1.0))


def test_save_and_load_model_round_trip(tmp_path: Path) -> None:
    data = _synthetic_woe()
    result = fit_logit(data, ["strong"])
    save_model(result, tmp_path / "model.pkl")
    loaded = load_model(tmp_path / "model.pkl")
    assert loaded.params.tolist() == pytest.approx(result.params.tolist())
    assert predict_pd(loaded, data, ["strong"]).tolist() == pytest.approx(
        predict_pd(result, data, ["strong"]).tolist())


def test_average_pd_vs_actual() -> None:
    summary = average_pd_vs_actual(pd.Series([0.1, 0.3, 0.2, 0.2]), pd.Series([0, 1, 0, 0]))
    assert summary["customers"] == 4
    assert summary["average_pd"] == pytest.approx(0.2)
    assert summary["actual_default_rate"] == pytest.approx(0.25)
    assert summary["difference"] == pytest.approx(-0.05)
