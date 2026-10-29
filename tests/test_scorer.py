"""Tests for scorecard.scorer: the app's scorer must reproduce the notebook exactly."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scorecard.binning import Binning
from scorecard.features import add_behavioural_features
from scorecard.scaling import pd_to_score
from scorecard.scorer import (
    REQUIRED_FIELDS,
    build_features,
    describe_bin,
    export_artifacts,
    load_artifacts,
    score_customer,
    score_frame,
    validate_customers,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "sample_customers.csv"
FIXTURE = ROOT / "tests" / "fixtures" / "notebook_scores.csv"

# Customer 23469 from the test set: the worked example in notebook 05 (score 479).
CUSTOMER_23469 = {"limit_bal": 50000, "pay_1": 1, "pay_2": 2, "pay_3": 2, "pay_4": 2, "pay_5": 2, "pay_6": 2,
                  "pay_amt1": 0, "pay_amt2": 5000, "pay_amt3": 0, "pay_amt4": 10000, "pay_amt5": 0,
                  "pay_amt6": 10000}


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


# --- same answers as the notebook ----------------------------------------------------

def test_worked_example_matches_notebook(artifacts) -> None:
    result = score_customer(CUSTOMER_23469, artifacts)
    assert result.score == 479
    assert result.points["points"].tolist() == [47, 49, 51, 52, 53, 57, 55, 55, 60]
    assert result.pd == pytest.approx(0.5699, abs=1e-4)
    assert result.decision == "Decline"


def test_sample_scores_match_notebook_pipeline(artifacts) -> None:
    sample = pd.read_csv(SAMPLE)
    expected = pd.read_csv(FIXTURE)
    assert sample["id"].tolist() == expected["id"].tolist()
    result = score_frame(sample, artifacts)
    assert result["score"].tolist() == expected["score"].tolist()


def test_sample_pd_agrees_with_model_pd_within_rounding(artifacts) -> None:
    expected = pd.read_csv(FIXTURE)
    scaling = artifacts.scaling
    exact = pd_to_score(expected["model_pd"], scaling["factor"], scaling["offset"])
    assert (expected["score"] - exact).abs().max() <= 0.5 * len(artifacts.variables)


def test_features_match_training_functions(artifacts) -> None:
    sample = pd.read_csv(SAMPLE)
    training = add_behavioural_features(sample.assign(**{f"bill_amt{i}": 1000 for i in range(1, 7)}))
    app = build_features(sample)
    for column in ["max_delinquency_6m", "months_zero_payment_6m", "pay_1", "limit_bal", "pay_amt2"]:
        assert app[column].tolist() == training[column].tolist()


def test_sample_has_no_protected_or_unused_fields() -> None:
    columns = set(pd.read_csv(SAMPLE, nrows=1).columns)
    assert not {"sex", "marriage", "age", "education"} & columns
    assert set(REQUIRED_FIELDS) <= columns


# --- decisions, PD and reasons --------------------------------------------------------

def _clean_customer() -> dict:
    """Never late, pays every month, high limit: best band almost everywhere."""
    return {"limit_bal": 500000, **{f"pay_{i}": 0 for i in range(1, 7)},
            **{f"pay_amt{i}": 25000 for i in range(1, 7)}}


def test_higher_score_means_lower_pd_and_better_decision(artifacts) -> None:
    good = score_customer(_clean_customer(), artifacts)
    bad = score_customer(CUSTOMER_23469, artifacts)
    assert good.score > bad.score
    assert good.pd < bad.pd
    assert good.decision == "Approve"


def test_decision_follows_policy_thresholds(artifacts) -> None:
    result = score_frame(pd.read_csv(SAMPLE), artifacts)
    policy = artifacts.policy
    assert (result.loc[result["score"] < policy["decline_below"], "decision"] == "Decline").all()
    assert (result.loc[result["score"] >= policy["approve_from"], "decision"] == "Approve").all()
    middle = result["score"].between(policy["decline_below"], policy["approve_from"] - 1)
    assert (result.loc[middle, "decision"] == "Refer").all()


def test_reason_codes_are_biggest_losses_first(artifacts) -> None:
    result = score_customer(CUSTOMER_23469, artifacts)
    top = result.points.sort_values("points_lost", ascending=False, kind="stable").head(3)
    assert len(result.reasons) == 3
    for reason, description in zip(result.reasons, top["description"]):
        assert reason.startswith(description)
    assert "24 points below the best band" in result.reasons[0]


def test_no_reasons_for_variables_already_in_best_band(artifacts) -> None:
    result = score_customer(_clean_customer(), artifacts)
    lost = result.points.set_index("description")["points_lost"]
    assert len(result.reasons) == min(3, int((lost > 0).sum()))
    for reason in result.reasons:
        assert lost[reason.split(":")[0]] > 0


def test_tied_best_bands_are_all_named(artifacts) -> None:
    reason = score_customer(CUSTOMER_23469, artifacts).reasons[0]
    assert reason.endswith("best band: card not used or revolving (minimum paid))")


@pytest.mark.parametrize(("variable", "label", "text"), [
    ("pay_1", ">= 2", "2+ months late"),
    ("pay_1", "<= -2", "card not used"),
    ("pay_5", "0 to 1", "revolving (minimum paid) or 1 month late"),
    ("limit_bal", "30001 to 70000", "NT$30,001 to NT$70,000"),
    ("pay_amt1", "<= 0", "nothing"),
    ("months_zero_payment_6m", "1", "1 month"),
])
def test_describe_bin(variable: str, label: str, text: str) -> None:
    assert describe_bin(variable, label) == text


# --- validation ---------------------------------------------------------------------

def test_validation_lists_missing_columns() -> None:
    errors = validate_customers(pd.DataFrame({"limit_bal": [1000]}))
    assert len(errors) == 1
    assert "pay_1" in errors[0] and "pay_amt6" in errors[0]


def test_validation_catches_bad_values_with_row_numbers() -> None:
    df = pd.DataFrame([CUSTOMER_23469] * 3)
    df.loc[1, "pay_1"] = 9          # status out of range
    df.loc[2, "pay_amt2"] = -5      # negative payment
    df["limit_bal"] = df["limit_bal"].astype(object)
    df.loc[0, "limit_bal"] = "abc"  # not a number
    errors = " ".join(validate_customers(df))
    assert "'limit_bal'" in errors and "row(s) 1" in errors
    assert "'pay_1'" in errors and "row(s) 2" in errors
    assert "'pay_amt2'" in errors and "row(s) 3" in errors


def test_validation_rejects_empty_file() -> None:
    assert validate_customers(pd.DataFrame(columns=REQUIRED_FIELDS))


def test_score_frame_raises_clear_error(artifacts) -> None:
    with pytest.raises(ValueError, match="Missing required column"):
        score_frame(pd.DataFrame({"limit_bal": [1000]}), artifacts)


def test_extra_columns_are_ignored_and_index_kept(artifacts) -> None:
    df = pd.DataFrame([CUSTOMER_23469, _clean_customer()], index=[7, 7]).assign(note=["a", "b"])
    result = score_frame(df, artifacts)
    assert result.index.tolist() == [7, 7]
    assert result["score"].iloc[0] == 479


# --- artifacts ---------------------------------------------------------------------

def test_export_and_load_round_trip(tmp_path: Path) -> None:
    binning = Binning("x", [-np.inf, 0, 10, np.inf], pd.DataFrame(), integer_valued=True)
    points = pd.DataFrame({"variable": "x", "bin": ["<= 0", "1 to 10", ">= 11"],
                           "woe": [-1.0, 0.0, 1.0], "points": [40, 50, 60]})
    export_artifacts({"x": binning}, points, {"factor": 1.0, "offset": 0.0},
                     {"decline_below": 1, "approve_from": 2}, tmp_path)
    loaded = load_artifacts(tmp_path)
    assert loaded.bins["x"].edges == binning.edges
    assert loaded.points["points"].tolist() == [40, 50, 60]


def test_load_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    binning = Binning("x", [-np.inf, 0, np.inf], pd.DataFrame(), integer_valued=True)
    points = pd.DataFrame({"variable": "x", "bin": ["<= 0", ">= 1"], "woe": [0.0, 0.0], "points": [1, 2]})
    export_artifacts({"x": binning}, points, {}, {}, tmp_path)
    (tmp_path / "scorecard_points.csv").write_text("variable,bin,woe,points\nx,<= 5,0,1\nx,>= 6,0,2\n")
    with pytest.raises(ValueError, match="bin edges give"):
        load_artifacts(tmp_path)


def test_scorer_does_not_import_heavy_modelling_libraries() -> None:
    code = ("import sys; sys.path.insert(0, 'src'); import scorecard.scorer; "
            "heavy = [m for m in ('statsmodels', 'sklearn', 'scipy', 'matplotlib') if m in sys.modules]; "
            "print(heavy); sys.exit(1 if heavy else 0)")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
