"""Score customers with the frozen scorecard. Nothing is refitted here.

The scorecard is stored in ``models/`` as plain, human-readable files:

- ``bins.json``: bin edges for each scorecard variable;
- ``scorecard_points.csv``: points (and WoE) for every bin;
- ``scaling.json``: base score, base odds, PDO, Factor and Offset;
- ``policy.json``: the three-zone decision thresholds.

JSON and CSV instead of pickle: they can be read and reviewed by people, diffed in
git, and loading them can't execute code. They also need only numpy and pandas, so
the app doesn't depend on statsmodels or scikit-learn.

Scoring a customer: raw fields -> engineered features (same functions as training)
-> bins -> points -> total score -> PD (from the score) -> decision and reason codes.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scorecard.binning import Binning, assign_bins, bin_labels
from scorecard.features import PAYMENT_COLUMNS, STATUS_COLUMNS, max_delinquency, months_zero_payment
from scorecard.scaling import score_to_pd
from scorecard.strategy import three_zone_decision

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
BINS_FILE, POINTS_FILE = "bins.json", "scorecard_points.csv"
SCALING_FILE, POLICY_FILE = "scaling.json", "policy.json"

# The raw fields a customer record needs. Nothing else is used: no bill amounts,
# no age or education, and never sex or marriage.
REQUIRED_FIELDS: list[str] = ["limit_bal", *STATUS_COLUMNS, *PAYMENT_COLUMNS]
STATUS_RANGE = (-2, 8)
MAX_BATCH_ROWS = 100_000
N_REASONS = 3

STATUS_MEANINGS: dict[int, str] = {
    -2: "card not used", -1: "paid in full", 0: "revolving (minimum paid)",
    **{k: f"{k} month{'s' if k > 1 else ''} late" for k in range(1, 9)},
}
VARIABLE_DESCRIPTIONS: dict[str, str] = {
    "pay_1": "Repayment status in September (latest month)",
    "max_delinquency_6m": "Worst repayment status in the last six months",
    "pay_4": "Repayment status in June",
    "pay_5": "Repayment status in May",
    "pay_6": "Repayment status in April",
    "months_zero_payment_6m": "Months with no payment in the last six",
    "limit_bal": "Credit limit",
    "pay_amt1": "Amount paid in September",
    "pay_amt2": "Amount paid in August",
}
_STATUS_VARIABLES = {"pay_1", "pay_4", "pay_5", "pay_6", "max_delinquency_6m"}
_MONEY_VARIABLES = {"limit_bal", "pay_amt1", "pay_amt2"}


@dataclass(frozen=True)
class ScorecardArtifacts:
    """Everything needed to score a customer, as loaded from ``models/``."""

    bins: dict[str, Binning]
    points: pd.DataFrame
    scaling: dict[str, float]
    policy: dict[str, float]

    @property
    def variables(self) -> list[str]:
        """Scorecard variables in scorecard order."""
        return list(self.points["variable"].drop_duplicates())

    @property
    def best_points(self) -> pd.Series:
        """Highest points available for each variable."""
        return self.points.groupby("variable", sort=False)["points"].max()


@dataclass(frozen=True)
class CustomerScore:
    """Result of scoring one customer."""

    points: pd.DataFrame
    score: int
    pd: float
    decision: str
    reasons: list[str]


# --- export and load ------------------------------------------------------------------

def export_artifacts(
    binnings: dict[str, Binning],
    scorecard: pd.DataFrame,
    scaling: dict[str, float],
    policy: dict[str, float],
    models_dir: Path = MODELS_DIR,
) -> None:
    """Write the frozen scorecard to ``models_dir``.

    Args:
        binnings: Fitted binnings (at least every scorecard variable).
        scorecard: Points table (``variable``, ``bin``, ``woe``, ``points``).
        scaling: ``base_score``, ``base_odds``, ``pdo``, ``factor``, ``offset``.
        policy: ``decline_below`` and ``approve_from``.
        models_dir: Destination folder.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    variables = list(scorecard["variable"].drop_duplicates())
    bins = {
        v: {
            "integer_valued": binnings[v].integer_valued,
            "interior_edges": [float(e) for e in binnings[v].edges[1:-1]],  # -inf and +inf are implied
            "labels": bin_labels(binnings[v].edges, binnings[v].integer_valued),
        }
        for v in variables
    }
    (models_dir / BINS_FILE).write_text(json.dumps(bins, indent=2))
    scorecard[["variable", "bin", "woe", "points"]].to_csv(models_dir / POINTS_FILE, index=False)
    (models_dir / SCALING_FILE).write_text(json.dumps(scaling, indent=2))
    (models_dir / POLICY_FILE).write_text(json.dumps(policy, indent=2))
    logger.info("Exported scorecard artifacts for %d variables to %s", len(variables), models_dir)


def load_artifacts(models_dir: Path = MODELS_DIR) -> ScorecardArtifacts:
    """Load the frozen scorecard and check its parts are consistent.

    Args:
        models_dir: Folder written by :func:`export_artifacts`.

    Returns:
        The loaded artifacts.

    Raises:
        ValueError: If the bin definitions and the points table don't match.
    """
    raw_bins = json.loads((models_dir / BINS_FILE).read_text())
    points = pd.read_csv(models_dir / POINTS_FILE)
    bins = {}
    for variable, spec in raw_bins.items():
        edges = [-np.inf, *spec["interior_edges"], np.inf]
        bins[variable] = Binning(variable, edges, pd.DataFrame(), spec["integer_valued"])
        expected = bin_labels(edges, spec["integer_valued"])
        listed = points.loc[points["variable"] == variable, "bin"].tolist()
        if expected != listed or expected != spec["labels"]:
            raise ValueError(f"{variable}: bin edges give {expected}, points table has {listed}")
    if set(bins) != set(points["variable"]):
        raise ValueError("bins.json and scorecard_points.csv cover different variables")
    scaling = json.loads((models_dir / SCALING_FILE).read_text())
    policy = json.loads((models_dir / POLICY_FILE).read_text())
    return ScorecardArtifacts(bins, points, scaling, policy)


# --- validation and features ----------------------------------------------------------

def _rows(mask: pd.Series) -> str:
    """First few 1-based row numbers where ``mask`` is true."""
    rows = (np.flatnonzero(mask.to_numpy()) + 1).tolist()
    shown = ", ".join(map(str, rows[:5]))
    return shown + (f" and {len(rows) - 5} more" if len(rows) > 5 else "")


def validate_customers(df: pd.DataFrame) -> list[str]:
    """Check raw customer records before scoring.

    Args:
        df: One row per customer with the raw fields.

    Returns:
        Plain-language error messages; empty if the data can be scored.
    """
    if df.empty:
        return ["The file has no customer rows."]
    if len(df) > MAX_BATCH_ROWS:
        return [f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}."]
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        return [f"Missing required column(s): {', '.join(missing)}."]
    errors = []
    numeric = df[REQUIRED_FIELDS].apply(pd.to_numeric, errors="coerce")
    for column in REQUIRED_FIELDS:
        bad = numeric[column].isna()
        if bad.any():
            errors.append(f"Column '{column}' has missing or non-numeric values in row(s) {_rows(bad)}.")
            continue
        values = numeric[column]
        if column in STATUS_COLUMNS:
            lo, hi = STATUS_RANGE
            bad = (values % 1 != 0) | (values < lo) | (values > hi)
            if bad.any():
                errors.append(f"Column '{column}' must be a whole number from {lo} to {hi} "
                              f"(repayment status); check row(s) {_rows(bad)}.")
        elif column == "limit_bal":
            if (bad := values <= 0).any():
                errors.append(f"Column 'limit_bal' must be above 0; check row(s) {_rows(bad)}.")
        elif (bad := values < 0).any():
            errors.append(f"Column '{column}' can't be negative; check row(s) {_rows(bad)}.")
    return errors


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw fields into the scorecard variables, with the same functions as training.

    Args:
        df: Validated raw customer records.

    Returns:
        One column per scorecard variable, same index as ``df``.
    """
    raw = df[REQUIRED_FIELDS].apply(pd.to_numeric)
    features = raw[["pay_1", "pay_4", "pay_5", "pay_6", "limit_bal", "pay_amt1", "pay_amt2"]].copy()
    features["max_delinquency_6m"] = max_delinquency(raw)
    features["months_zero_payment_6m"] = months_zero_payment(raw)
    return features


# --- reason codes ---------------------------------------------------------------------

def _money(text: str) -> str:
    """Format a number string as NT$ with thousands separators."""
    return f"NT${int(float(text)):,}"


def describe_bin(variable: str, label: str) -> str:
    """Plain-language description of a bin label, e.g. ``">= 2"`` -> ``"2+ months late"``.

    Args:
        variable: Scorecard variable.
        label: Bin label as in the points table.

    Returns:
        A short description.
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", label)
    if variable in _STATUS_VARIABLES:
        if label.startswith(">="):
            return f"{int(numbers[0])}+ months late"
        if label.startswith("<="):
            return STATUS_MEANINGS[int(numbers[0])]
        if " to " in label:
            return " or ".join(STATUS_MEANINGS[int(n)] for n in numbers)
        return STATUS_MEANINGS[int(numbers[0])]
    if variable in _MONEY_VARIABLES:
        if label.startswith("<="):
            return "nothing" if float(numbers[0]) <= 0 else f"{_money(numbers[0])} or less"
        if label.startswith(">="):
            return f"{_money(numbers[0])} or more"
        if " to " in label:
            return f"{_money(numbers[0])} to {_money(numbers[1])}"
    if variable == "months_zero_payment_6m":
        if label.startswith("<="):
            return "none"
        if label.startswith(">="):
            return f"{numbers[0]} or more months"
        return f"{numbers[0]} month{'s' if numbers[0] != '1' else ''}"
    return label


def reason_codes(points_row: pd.Series, bins_row: pd.Series, artifacts: ScorecardArtifacts,
                 n_reasons: int = N_REASONS) -> list[str]:
    """The variables where a customer lost the most points versus the best band.

    Args:
        points_row: The customer's points per variable.
        bins_row: The customer's bin label per variable.
        artifacts: Loaded scorecard.
        n_reasons: How many reasons to return.

    Returns:
        Up to ``n_reasons`` plain-language reasons, biggest loss first. Variables
        where the customer is already in the best band are never reasons.
    """
    best = artifacts.best_points
    lost = (best[points_row.index] - points_row).sort_values(ascending=False, kind="stable")
    reasons = []
    for variable, points_lost in lost[lost > 0].head(n_reasons).items():
        block = artifacts.points[artifacts.points["variable"] == variable]
        best_bands = " or ".join(describe_bin(variable, b) for b in block.loc[block["points"] == best[variable], "bin"])
        reasons.append(
            f"{VARIABLE_DESCRIPTIONS.get(variable, variable)}: {describe_bin(variable, bins_row[variable])} "
            f"({int(points_lost)} points below the best band: {best_bands})"
        )
    return reasons


# --- scoring --------------------------------------------------------------------------

def _score_features(features: pd.DataFrame, artifacts: ScorecardArtifacts) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin labels and points per variable for every customer."""
    labels, points = {}, {}
    for variable in artifacts.variables:
        labels[variable] = assign_bins(features[variable], artifacts.bins[variable])
        lookup = artifacts.points[artifacts.points["variable"] == variable].set_index("bin")["points"]
        points[variable] = labels[variable].map(lookup).astype(int)
    return pd.DataFrame(labels, index=features.index), pd.DataFrame(points, index=features.index)


def score_frame(df: pd.DataFrame, artifacts: ScorecardArtifacts) -> pd.DataFrame:
    """Score many customers at once.

    Args:
        df: Raw customer records (extra columns are ignored).
        artifacts: Loaded scorecard.

    Returns:
        ``score``, ``pd``, ``decision``, ``reason_1..3`` and ``points_<variable>``
        for each customer, same index as ``df``.

    Raises:
        ValueError: If the records fail :func:`validate_customers`.
    """
    errors = validate_customers(df)
    if errors:
        raise ValueError(" ".join(errors))
    original_index = df.index
    df = df.reset_index(drop=True)  # a clean 0..n index, so joins can't duplicate rows
    features = build_features(df)
    labels, points = _score_features(features, artifacts)
    scaling, policy = artifacts.scaling, artifacts.policy
    out = pd.DataFrame(index=df.index)
    out["score"] = points.sum(axis=1)
    out["pd"] = score_to_pd(out["score"], scaling["factor"], scaling["offset"])
    out["decision"] = [three_zone_decision(s, policy["decline_below"], policy["approve_from"])
                       for s in out["score"]]
    reasons = [reason_codes(points.iloc[k], labels.iloc[k], artifacts) for k in range(len(df))]
    for k in range(N_REASONS):
        out[f"reason_{k + 1}"] = [r[k] if len(r) > k else "" for r in reasons]
    out = out.join(points.add_prefix("points_"))
    out.index = original_index
    return out


def score_customer(fields: Mapping[str, float], artifacts: ScorecardArtifacts) -> CustomerScore:
    """Score one customer from their raw fields.

    Args:
        fields: ``limit_bal``, ``pay_1..pay_6`` and ``pay_amt1..pay_amt6``.
        artifacts: Loaded scorecard.

    Returns:
        Points per variable (with band and the best available points), total score,
        PD implied by the score, decision zone and up to three reason codes.

    Raises:
        ValueError: If the fields are missing or invalid.
    """
    df = pd.DataFrame([dict(fields)])
    result = score_frame(df, artifacts).iloc[0]
    features = build_features(df)
    labels, points = _score_features(features, artifacts)
    table = pd.DataFrame({
        "variable": artifacts.variables,
        "description": [VARIABLE_DESCRIPTIONS.get(v, v) for v in artifacts.variables],
        "value": [features.iloc[0][v] for v in artifacts.variables],
        "band": [describe_bin(v, labels.iloc[0][v]) for v in artifacts.variables],
        "points": [int(points.iloc[0][v]) for v in artifacts.variables],
        "best_points": [int(artifacts.best_points[v]) for v in artifacts.variables],
    })
    table["points_lost"] = table["best_points"] - table["points"]
    reasons = [result[f"reason_{k + 1}"] for k in range(N_REASONS) if result[f"reason_{k + 1}"]]
    return CustomerScore(table, int(result["score"]), float(result["pd"]), str(result["decision"]), reasons)
