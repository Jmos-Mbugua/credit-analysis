"""Cut-off strategy: who to approve, refer or decline, and what each choice is worth.

Convention: a customer is **approved if their score is at or above the cut-off**,
declined below it. In a behavioural (existing-customer) setting, "decline" stands for
actions such as blocking further spending, cutting the limit or not renewing the card.

Profit uses deliberately simple, configurable assumptions (see :class:`ProfitAssumptions`):
a good customer earns a yearly margin on their balance, and a bad customer loses
part of their balance (loss given default). Declined customers earn and lose nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BILL_COLUMNS = [f"bill_amt{i}" for i in range(1, 7)]
DECLINE, REFER, APPROVE = "Decline", "Refer", "Approve"

# Proposed three-zone policy (notebook 07): decline below 520, refer 520-549, approve from 550.
DECLINE_BELOW = 520
APPROVE_FROM = 550


@dataclass(frozen=True)
class ProfitAssumptions:
    """Business assumptions behind the profit calculation.

    Attributes:
        margin_rate: Yearly margin earned on a good customer, as a share of their
            average bill balance (interest and fees minus funding and running costs).
        lgd: Loss given default: share of a bad customer's average bill balance that
            is lost when they default (after collections and recoveries).
    """

    margin_rate: float = 0.08
    lgd: float = 0.60


def break_even_pd(assumptions: ProfitAssumptions) -> float:
    """PD at which a customer's expected profit is zero (for a given balance).

    Expected profit per unit of balance = (1 - PD) x margin - PD x LGD, which is
    zero at PD = margin / (margin + LGD). Customers riskier than this lose money
    on average.

    Args:
        assumptions: Margin and LGD.

    Returns:
        The break-even probability of default.
    """
    return assumptions.margin_rate / (assumptions.margin_rate + assumptions.lgd)


def average_balance(df: pd.DataFrame) -> pd.Series:
    """Each customer's average bill balance over the six months, floored at zero.

    A negative balance (credit) means the bank owes the customer money, so there is
    nothing to earn a margin on or to lose; it counts as zero exposure.

    Args:
        df: Customers with ``bill_amt1..6``.

    Returns:
        Average balance per customer (NT$), same index as ``df``.
    """
    return df[BILL_COLUMNS].mean(axis=1).clip(lower=0).rename("average_balance")


def strategy_table(scores: pd.Series, defaults: pd.Series, cutoffs: list[float]) -> pd.DataFrame:
    """What happens to approvals and bads at each cut-off.

    Args:
        scores: Scores (higher = safer).
        defaults: 0/1 outcomes (1 = bad) for the same customers.
        cutoffs: Cut-offs to evaluate (approve if score >= cut-off).

    Returns:
        One row per cut-off with ``approved``, ``approval_rate``, ``approved_bad_rate``,
        ``declined_bad_rate``, ``bads_declined_share`` (% of all bads declined) and
        ``goods_declined_share`` (% of all goods declined). Bad rates are NaN when the
        group is empty.
    """
    scores, defaults = np.asarray(scores, dtype=float), np.asarray(defaults)
    n_bads, n_goods = defaults.sum(), (1 - defaults).sum()
    rows = []
    for cutoff in cutoffs:
        approved = scores >= cutoff
        bads_declined = defaults[~approved].sum()
        rows.append({
            "cutoff": cutoff,
            "approved": int(approved.sum()),
            "approval_rate": approved.mean(),
            "approved_bad_rate": defaults[approved].mean() if approved.any() else np.nan,
            "declined_bad_rate": defaults[~approved].mean() if (~approved).any() else np.nan,
            "bads_declined_share": bads_declined / n_bads,
            "goods_declined_share": (1 - defaults)[~approved].sum() / n_goods,
        })
    return pd.DataFrame(rows)


def expected_profit(
    scores: pd.Series,
    defaults: pd.Series,
    balances: pd.Series,
    cutoffs: list[float],
    assumptions: ProfitAssumptions = ProfitAssumptions(),
) -> pd.DataFrame:
    """Profit from the approved customers at each cut-off, given actual outcomes.

    Each approved good earns ``margin_rate x balance``; each approved bad loses
    ``lgd x balance``. Declined customers contribute nothing.

    Args:
        scores: Scores (higher = safer).
        defaults: 0/1 outcomes (1 = bad).
        balances: Average balance per customer (see :func:`average_balance`).
        cutoffs: Cut-offs to evaluate (approve if score >= cut-off).
        assumptions: Margin and LGD.

    Returns:
        One row per cut-off with ``approval_rate``, ``margin_from_goods``,
        ``loss_from_bads`` and ``profit`` (NT$).
    """
    scores, defaults = np.asarray(scores, dtype=float), np.asarray(defaults)
    balances = np.asarray(balances, dtype=float)
    margin = np.where(defaults == 0, assumptions.margin_rate * balances, 0.0)
    loss = np.where(defaults == 1, assumptions.lgd * balances, 0.0)
    rows = []
    for cutoff in cutoffs:
        approved = scores >= cutoff
        rows.append({
            "cutoff": cutoff,
            "approval_rate": approved.mean(),
            "margin_from_goods": margin[approved].sum(),
            "loss_from_bads": loss[approved].sum(),
        })
    table = pd.DataFrame(rows)
    table["profit"] = table["margin_from_goods"] - table["loss_from_bads"]
    return table


def best_cutoff(profit_table: pd.DataFrame) -> pd.Series:
    """The profit-maximising row of :func:`expected_profit`.

    If several cut-offs tie, the lowest one (highest approval rate) is chosen.

    Args:
        profit_table: Output of :func:`expected_profit`.

    Returns:
        The best row.
    """
    best = profit_table["profit"].max()
    return profit_table[profit_table["profit"] == best].sort_values("cutoff").iloc[0]


def three_zone_decision(score: float, decline_below: float, approve_from: float) -> str:
    """Decision for one score under a three-zone policy.

    - score < ``decline_below``: Decline (too risky).
    - ``decline_below`` <= score < ``approve_from``: Refer (a person reviews the case).
    - score >= ``approve_from``: Approve.

    Args:
        score: Customer's score.
        decline_below: Lowest score that is not automatically declined.
        approve_from: Lowest score that is automatically approved.

    Returns:
        ``"Decline"``, ``"Refer"`` or ``"Approve"``.

    Raises:
        ValueError: If ``decline_below`` is above ``approve_from``.
    """
    if decline_below > approve_from:
        raise ValueError(f"decline_below ({decline_below}) must not exceed approve_from ({approve_from})")
    if score < decline_below:
        return DECLINE
    if score < approve_from:
        return REFER
    return APPROVE


def zone_summary(
    scores: pd.Series, defaults: pd.Series, decline_below: float, approve_from: float
) -> pd.DataFrame:
    """Customers, share, bads and bad rate in each zone of a three-zone policy.

    Args:
        scores: Scores (higher = safer).
        defaults: 0/1 outcomes (1 = bad).
        decline_below: Lowest score that is not automatically declined.
        approve_from: Lowest score that is automatically approved.

    Returns:
        One row per zone (Decline, Refer, Approve) with ``score_range``, ``customers``,
        ``share``, ``bads``, ``bad_rate`` and ``share_of_all_bads``.
    """
    zones = pd.Series([three_zone_decision(s, decline_below, approve_from) for s in scores])
    defaults = pd.Series(np.asarray(defaults))
    grouped = defaults.groupby(zones)
    table = pd.DataFrame({"customers": grouped.size(), "bads": grouped.sum()})
    table = table.reindex([DECLINE, REFER, APPROVE], fill_value=0)
    table["share"] = table["customers"] / table["customers"].sum()
    table["bad_rate"] = table["bads"] / table["customers"].replace(0, np.nan)
    table["share_of_all_bads"] = table["bads"] / table["bads"].sum()
    table.insert(0, "score_range", [f"< {decline_below:g}", f"{decline_below:g} to {approve_from - 1:g}",
                                    f">= {approve_from:g}"])
    return table.rename_axis("zone").reset_index()


def profit_by_zone(
    scores: pd.Series,
    defaults: pd.Series,
    balances: pd.Series,
    decline_below: float,
    approve_from: float,
    assumptions: ProfitAssumptions = ProfitAssumptions(),
) -> pd.Series:
    """Profit each zone would make if all of its customers were kept.

    Shows whether a zone pays for itself: a zone with negative profit loses
    money on average under these assumptions.

    Args:
        scores: Scores (higher = safer).
        defaults: 0/1 outcomes (1 = bad).
        balances: Average balance per customer (see :func:`average_balance`).
        decline_below: Lowest score that is not automatically declined.
        approve_from: Lowest score that is automatically approved.
        assumptions: Margin and LGD.

    Returns:
        Profit (NT$) indexed by zone (Decline, Refer, Approve).
    """
    defaults = np.asarray(defaults)
    balances = np.asarray(balances, dtype=float)
    value = np.where(defaults == 0, assumptions.margin_rate * balances, -assumptions.lgd * balances)
    zones = pd.Series([three_zone_decision(s, decline_below, approve_from) for s in scores])
    return pd.Series(value).groupby(zones).sum().reindex([DECLINE, REFER, APPROVE], fill_value=0.0)
