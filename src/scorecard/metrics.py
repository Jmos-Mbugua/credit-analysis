"""Performance measures: how well the scorecard ranks and how accurate its PDs are.

- AUC: pick one good and one bad customer at random; AUC is the chance the good one
  gets the higher score. 0.5 = coin toss, 1 = always right.
- Gini = 2 x AUC - 1: the same thing on a 0-to-1 scale (0 = random, 1 = perfect).
  It is the headline number in credit risk.
- KS: the largest gap between the share of bads and the share of goods scoring at or
  below a given score. It says how well a single cut-off could separate the two groups.
- Calibration: whether predicted PDs match actual default rates, group by group.

Scores follow the scorecard convention (higher = safer); PDs are higher = riskier.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

# Train-minus-test Gini gaps (in Gini points) used for the overfitting verdict.
GINI_GAP_OK = 0.03
GINI_GAP_WATCH = 0.05


def auc(default: pd.Series, score: pd.Series) -> float:
    """Area under the ROC curve for a score where higher means safer.

    Ties count as half right.

    Args:
        default: 0/1 outcomes (1 = bad).
        score: Scores (higher = safer). Pass ``-pd`` to evaluate PDs.

    Returns:
        AUC between 0 and 1.
    """
    return float(roc_auc_score(default, -np.asarray(score, dtype=float)))


def gini(default: pd.Series, score: pd.Series) -> float:
    """Gini coefficient = 2 x AUC - 1.

    Args:
        default: 0/1 outcomes (1 = bad).
        score: Scores (higher = safer).

    Returns:
        Gini between -1 and 1 (1 = perfect ranking, 0 = random).
    """
    return 2 * auc(default, score) - 1


def ks_table(default: pd.Series, score: pd.Series) -> pd.DataFrame:
    """Cumulative share of goods and bads at or below each score.

    Customers with the same score are kept together, because a cut-off can't
    split them.

    Args:
        default: 0/1 outcomes (1 = bad).
        score: Scores (higher = safer).

    Returns:
        One row per distinct score (lowest first) with ``cum_bads``, ``cum_goods``
        and ``gap`` (cum_bads - cum_goods).
    """
    counts = pd.crosstab(pd.Series(np.asarray(score), name="score"),
                         pd.Series(np.asarray(default), name="default"))
    counts = counts.reindex(columns=[0, 1], fill_value=0)
    table = pd.DataFrame({
        "cum_bads": counts[1].cumsum() / counts[1].sum(),
        "cum_goods": counts[0].cumsum() / counts[0].sum(),
    })
    table["gap"] = table["cum_bads"] - table["cum_goods"]
    return table.reset_index()


def ks(default: pd.Series, score: pd.Series) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and the score where it occurs.

    Args:
        default: 0/1 outcomes (1 = bad).
        score: Scores (higher = safer).

    Returns:
        ``(ks, score_at_max)``: the largest absolute gap between the cumulative
        shares of bads and goods, and the score (cut-off "at or below") where it is largest.
    """
    table = ks_table(default, score)
    i = table["gap"].abs().idxmax()
    return float(abs(table.loc[i, "gap"])), float(table.loc[i, "score"])


def roc_points(default: pd.Series, score: pd.Series) -> pd.DataFrame:
    """Points of the ROC curve.

    Args:
        default: 0/1 outcomes (1 = bad).
        score: Scores (higher = safer).

    Returns:
        ``false_positive_rate`` (share of goods flagged) and ``true_positive_rate``
        (share of bads flagged) as the cut-off moves from the lowest score upwards.
    """
    fpr, tpr, _ = roc_curve(default, -np.asarray(score, dtype=float))
    return pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr})


def bootstrap_gini(
    default: pd.Series, score: pd.Series, n_resamples: int = 500, seed: int = 42, level: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for Gini by resampling customers with replacement.

    A Gini measured on 9,000 customers would come out a little different on another
    9,000. Re-drawing the sample many times shows how much it moves by chance.

    Args:
        default: 0/1 outcomes (1 = bad).
        score: Scores (higher = safer).
        n_resamples: Number of resamples.
        seed: Random seed.
        level: Confidence level (0.95 = middle 95% of resampled Ginis).

    Returns:
        ``(low, high)`` bounds of the interval.
    """
    rng = np.random.default_rng(seed)
    y, s = np.asarray(default), np.asarray(score, dtype=float)
    ginis = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():  # need both goods and bads
            ginis.append(2 * roc_auc_score(y[idx], -s[idx]) - 1)
    tail = (1 - level) / 2
    low, high = np.quantile(ginis, [tail, 1 - tail])
    return float(low), float(high)


def performance_summary(default: pd.Series, score: pd.Series) -> dict[str, float]:
    """AUC, Gini and KS for one sample.

    Args:
        default: 0/1 outcomes (1 = bad).
        score: Scores (higher = safer).

    Returns:
        ``customers``, ``default_rate``, ``auc``, ``gini``, ``ks`` and ``ks_score``.
    """
    ks_value, ks_score = ks(default, score)
    return {
        "customers": len(default),
        "default_rate": float(np.mean(default)),
        "auc": auc(default, score),
        "gini": gini(default, score),
        "ks": ks_value,
        "ks_score": ks_score,
    }


def overfitting_verdict(train_gini: float, test_gini: float) -> str:
    """One-line verdict on overfitting from the train-test Gini gap.

    A model that memorised its training data scores much better on train than on
    test. Rule of thumb: a drop under 3 Gini points is normal noise, 3-5 is worth
    watching, above 5 points is overfitting.

    Args:
        train_gini: Gini on the training set.
        test_gini: Gini on the test set.

    Returns:
        A short verdict.
    """
    gap = train_gini - test_gini
    if gap < GINI_GAP_OK:
        return f"No overfitting: test Gini is within {GINI_GAP_OK * 100:.0f} points of train ({gap * 100:+.1f})."
    if gap < GINI_GAP_WATCH:
        return f"Watch: test Gini is {gap * 100:.1f} points below train."
    return f"Overfitting: test Gini is {gap * 100:.1f} points below train."


def calibration_table(default: pd.Series, pd_hat: pd.Series, n_groups: int = 10) -> pd.DataFrame:
    """Predicted vs actual default rate in groups of equal size, ordered by predicted PD.

    Customers with the same PD may be split between neighbouring groups so that
    every group has the same size.

    Args:
        default: 0/1 outcomes (1 = bad).
        pd_hat: Predicted probabilities of default.
        n_groups: Number of groups (10 = deciles).

    Returns:
        One row per group (1 = lowest PD) with ``customers``, ``min_pd``, ``max_pd``,
        ``predicted_default_rate``, ``actual_default_rate`` and ``difference``
        (predicted minus actual).
    """
    pd_hat = pd.Series(np.asarray(pd_hat, dtype=float))
    default = pd.Series(np.asarray(default))
    group = pd.qcut(pd_hat.rank(method="first"), n_groups, labels=range(1, n_groups + 1))
    grouped = pd.DataFrame({"pd": pd_hat, "default": default, "group": group}).groupby("group", observed=True)
    table = pd.DataFrame({
        "customers": grouped.size(),
        "min_pd": grouped["pd"].min(),
        "max_pd": grouped["pd"].max(),
        "predicted_default_rate": grouped["pd"].mean(),
        "actual_default_rate": grouped["default"].mean(),
    })
    table["difference"] = table["predicted_default_rate"] - table["actual_default_rate"]
    return table.rename_axis("decile").reset_index()
