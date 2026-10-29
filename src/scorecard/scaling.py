"""Turn the logistic regression into a points-based scorecard.

A score is a rescaled version of the model's log-odds of being *good*:

    score = Offset + Factor x ln(odds of good)

- ``base_score`` and ``base_odds`` fix one point on the scale: a score of 600 means
  odds of 50 goods for every bad (PD = 1/51, about 2%).
- ``pdo`` ("points to double the odds"): every extra 20 points doubles the odds of
  being good. So 620 means 100:1, 580 means 25:1.
- Factor = pdo / ln(2) and Offset = base_score - Factor x ln(base_odds).

The model predicts log-odds of *default* = intercept + sum(coef x WoE), which is
minus ln(odds of good). So each bin's points are -Factor x coef x WoE, plus an equal
share of (Offset - Factor x intercept) so the points add up to the full score.
Higher score = safer customer.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from scorecard.binning import Binning, assign_bins

logger = logging.getLogger(__name__)

CONSTANT = "const"
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20


def scaling_constants(
    base_score: float = BASE_SCORE, base_odds: float = BASE_ODDS, pdo: float = PDO
) -> tuple[float, float]:
    """Factor and Offset of the score scale.

    Args:
        base_score: Score at which the odds of good equal ``base_odds``.
        base_odds: Odds of good (goods per bad) at ``base_score``.
        pdo: Points needed to double the odds.

    Returns:
        ``(factor, offset)``.
    """
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    return factor, offset


def bin_points(
    coefficient: float, woe: float, intercept: float, n_variables: int, factor: float, offset: float
) -> float:
    """Unrounded points for one bin.

    Args:
        coefficient: The variable's logistic regression coefficient.
        woe: The bin's Weight of Evidence.
        intercept: The model's intercept.
        n_variables: Number of variables in the model (the intercept is shared equally).
        factor: From :func:`scaling_constants`.
        offset: From :func:`scaling_constants`.

    Returns:
        Points for the bin.
    """
    return -factor * coefficient * woe + (offset - factor * intercept) / n_variables


def build_scorecard(
    coefficients: pd.DataFrame, binning_tables: pd.DataFrame, factor: float, offset: float
) -> pd.DataFrame:
    """Points for every bin of every model variable.

    Args:
        coefficients: ``variable`` and ``coefficient`` columns, including the ``const`` row.
        binning_tables: Long table of bins (``variable``, ``bin``, ``woe``).
        factor: From :func:`scaling_constants`.
        offset: From :func:`scaling_constants`.

    Returns:
        Columns ``variable``, ``bin``, ``woe``, ``points`` (whole numbers), in model order
        and bin order.
    """
    coefs = coefficients.set_index("variable")["coefficient"]
    intercept = coefs[CONSTANT]
    variables = [v for v in coefs.index if v != CONSTANT]
    blocks = []
    for variable in variables:
        bins = binning_tables.loc[binning_tables["variable"] == variable, ["variable", "bin", "woe"]].copy()
        bins["points"] = [
            round(bin_points(coefs[variable], w, intercept, len(variables), factor, offset))
            for w in bins["woe"]
        ]
        blocks.append(bins)
    scorecard = pd.concat(blocks, ignore_index=True)
    scorecard["points"] = scorecard["points"].astype(int)
    logger.info("Scorecard: %d variables, %d bins, possible scores %d to %d",
                len(variables), len(scorecard), *score_range(scorecard))
    return scorecard


def score_range(scorecard: pd.DataFrame) -> tuple[int, int]:
    """Lowest and highest possible total score.

    Args:
        scorecard: Output of :func:`build_scorecard`.

    Returns:
        ``(min_score, max_score)``.
    """
    by_variable = scorecard.groupby("variable", sort=False)["points"]
    return int(by_variable.min().sum()), int(by_variable.max().sum())


def points_spread(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Lowest and highest points per variable, and the gap between them.

    The gap is how many points a variable can move a score, a simple measure of
    its weight in the scorecard.

    Args:
        scorecard: Output of :func:`build_scorecard`.

    Returns:
        One row per variable with ``min_points``, ``max_points`` and ``spread``,
        largest spread first.
    """
    by_variable = scorecard.groupby("variable", sort=False)["points"]
    table = pd.DataFrame({"min_points": by_variable.min(), "max_points": by_variable.max()})
    table["spread"] = table["max_points"] - table["min_points"]
    return table.sort_values("spread", ascending=False).reset_index()


def points_by_variable(
    df: pd.DataFrame, scorecard: pd.DataFrame, binnings: dict[str, Binning]
) -> pd.DataFrame:
    """Points each customer gets for each variable, from their raw values.

    Args:
        df: Customers with the raw (not WoE) model variables.
        scorecard: Output of :func:`build_scorecard`.
        binnings: Fitted binnings for the model variables.

    Returns:
        One column of points per variable, same index as ``df``.

    Raises:
        KeyError: If a customer falls in a bin that has no points (should never happen).
    """
    out = {}
    for variable, block in scorecard.groupby("variable", sort=False):
        lookup = block.set_index("bin")["points"]
        bins = assign_bins(df[variable], binnings[variable])
        unknown = set(bins) - set(lookup.index)
        if unknown:
            raise KeyError(f"{variable}: bins {unknown} not in the scorecard")
        out[variable] = bins.map(lookup).astype(int)
    return pd.DataFrame(out, index=df.index)


def score_customers(
    df: pd.DataFrame, scorecard: pd.DataFrame, binnings: dict[str, Binning]
) -> pd.Series:
    """Total score of each customer: the sum of their points.

    Args:
        df: Customers with the raw model variables.
        scorecard: Output of :func:`build_scorecard`.
        binnings: Fitted binnings for the model variables.

    Returns:
        Whole-number scores, same index as ``df``.
    """
    return points_by_variable(df, scorecard, binnings).sum(axis=1).rename("score")


def customer_breakdown(
    customer: pd.Series, scorecard: pd.DataFrame, binnings: dict[str, Binning]
) -> pd.DataFrame:
    """How one customer's score is built: value, bin, WoE and points per variable.

    Args:
        customer: One row of raw data.
        scorecard: Output of :func:`build_scorecard`.
        binnings: Fitted binnings for the model variables.

    Returns:
        One row per variable with ``variable``, ``value``, ``bin``, ``woe``, ``points``.
    """
    rows = []
    for variable, block in scorecard.groupby("variable", sort=False):
        value = customer[variable]
        label = assign_bins(pd.Series([value]), binnings[variable]).iloc[0]
        row = block.set_index("bin").loc[label]
        rows.append({"variable": variable, "value": value, "bin": label,
                     "woe": row["woe"], "points": int(row["points"])})
    return pd.DataFrame(rows)


def score_to_pd(score: float | pd.Series, factor: float, offset: float) -> float | pd.Series:
    """Probability of default implied by a score.

    ln(odds of good) = (score - Offset) / Factor, and PD = 1 / (1 + odds of good).

    Args:
        score: Score(s).
        factor: From :func:`scaling_constants`.
        offset: From :func:`scaling_constants`.

    Returns:
        PD(s) between 0 and 1.
    """
    return 1 / (1 + np.exp((score - offset) / factor))


def pd_to_score(pd_value: float | pd.Series, factor: float, offset: float) -> float | pd.Series:
    """Unrounded score implied by a probability of default (inverse of :func:`score_to_pd`).

    Args:
        pd_value: PD(s) strictly between 0 and 1.
        factor: From :func:`scaling_constants`.
        offset: From :func:`scaling_constants`.

    Returns:
        Score(s).
    """
    return offset + factor * np.log((1 - pd_value) / pd_value)


def check_score_pd_agreement(
    scores: pd.Series, model_pd: pd.Series, n_variables: int, factor: float, offset: float
) -> float:
    """Check that rounded scores agree with the model's PDs.

    Each variable's points are rounded to the nearest whole number, so a total can be
    off by at most 0.5 points per variable from the exact score of the model's PD.

    Args:
        scores: Scorecard totals.
        model_pd: PDs predicted by the logistic regression for the same customers.
        n_variables: Number of variables in the scorecard.
        factor: From :func:`scaling_constants`.
        offset: From :func:`scaling_constants`.

    Returns:
        The largest absolute difference in points.

    Raises:
        ValueError: If any difference exceeds the rounding allowance.
    """
    gap = (scores - pd_to_score(model_pd, factor, offset)).abs()
    allowance = 0.5 * n_variables
    largest = float(gap.max())
    if largest > allowance + 1e-9:
        raise ValueError(f"Score and PD disagree by {largest:.2f} points (allowed {allowance})")
    logger.info("Score and PD agree: largest gap %.2f points (rounding allowance %.1f)", largest, allowance)
    return largest


def score_band_table(scores: pd.Series, default: pd.Series, width: int = 20) -> pd.DataFrame:
    """Customers, bads and bad rate per score band (e.g. 540-559, 560-579).

    Args:
        scores: Scores.
        default: 0/1 outcomes for the same customers.
        width: Band width in points.

    Returns:
        One row per band, lowest scores first, with ``band``, ``customers``, ``bads``,
        ``bad_rate``, ``share`` and ``cumulative_share``.
    """
    lower = (scores // width) * width
    grouped = default.groupby(lower)
    table = pd.DataFrame({"customers": grouped.size(), "bads": grouped.sum()}).sort_index()
    table["bad_rate"] = table["bads"] / table["customers"]
    table["share"] = table["customers"] / table["customers"].sum()
    table["cumulative_share"] = table["share"].cumsum()
    table.index = [f"{int(lo)}-{int(lo) + width - 1}" for lo in table.index]
    return table.rename_axis("band").reset_index()
