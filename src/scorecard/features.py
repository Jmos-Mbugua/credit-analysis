"""Behavioural features built from the six monthly snapshots, and the candidate list.

Suffix 1 is September 2005 (most recent), 6 is April 2005 (oldest).
"""

from __future__ import annotations

import logging

import pandas as pd

from scorecard.binning import BinSpec

logger = logging.getLogger(__name__)

MONTHS = range(1, 7)
STATUS_COLUMNS = [f"pay_{i}" for i in MONTHS]
BILL_COLUMNS = [f"bill_amt{i}" for i in MONTHS]
PAYMENT_COLUMNS = [f"pay_amt{i}" for i in MONTHS]

# Every candidate for the model and the kind of behaviour it captures.
# sex, marriage and id are deliberately absent: they are never model features.
FEATURE_TYPES: dict[str, str] = {
    **{c: "delinquency" for c in STATUS_COLUMNS},
    "max_delinquency_6m": "delinquency",
    "months_delinquent_6m": "delinquency",
    "util_1": "utilisation",
    "avg_util_6m": "utilisation",
    "max_util_6m": "utilisation",
    "util_trend": "utilisation",
    **{c: "utilisation" for c in BILL_COLUMNS},
    "pay_ratio_1": "payment",
    "avg_pay_ratio_6m": "payment",
    "months_zero_payment_6m": "payment",
    **{c: "payment" for c in PAYMENT_COLUMNS},
    "limit_bal": "limit",
    "age": "demographic",
    "education": "demographic",
}
CANDIDATES: list[str] = list(FEATURE_TYPES)

# One-sentence business reason for variables that may be selected (see notebook 03).
FEATURE_REASONS: dict[str, str] = {
    "pay_1": "Last month's repayment status is the most direct sign of current financial stress.",
    "pay_2": "August repayment status shows whether arrears are recent or persistent.",
    "pay_3": "July repayment status adds history of missed payments beyond the last two months.",
    "pay_4": "June repayment status adds older payment history.",
    "pay_5": "May repayment status adds older payment history.",
    "pay_6": "April repayment status adds the oldest payment history in the window.",
    "max_delinquency_6m": "The worst arrears in six months captures past trouble even if the customer has since caught up.",
    "months_zero_payment_6m": "Customers who repeatedly pay nothing are not servicing their debt, whatever their balance.",
    "pay_ratio_1": "Paying little of last month's bill signals difficulty keeping up with the balance.",
    "avg_pay_ratio_6m": "Paying a consistently high share of each bill shows a stable habit of clearing debt.",
    "limit_bal": "The credit limit reflects the bank's earlier assessment of the customer's income and creditworthiness.",
    "pay_amt1": "The amount paid last month shows the customer's current capacity to pay.",
    "pay_amt2": "The amount paid in August adds a second month of payment capacity.",
    "avg_util_6m": "Customers who keep their card close to the limit have little room to absorb a shock.",
    "util_trend": "A rising balance relative to the limit suggests growing reliance on credit.",
}

# Business-rule bins. Repayment status: -2 no use, -1 paid in full, 0 revolving,
# 1, 2 and 3+ months late. These are categories rather than amounts, so the default
# rate is not forced to be monotonic across -2/-1/0. Bins under 5% are still merged.
_STATUS_SPEC = BinSpec(fixed_edges=(-2, -1, 0, 1, 2), monotonic=False)
BIN_SPECS: dict[str, BinSpec] = {
    **{c: _STATUS_SPEC for c in STATUS_COLUMNS},
    "max_delinquency_6m": _STATUS_SPEC,
    "months_delinquent_6m": BinSpec(fixed_edges=(0, 1, 2, 3)),
    "education": BinSpec(fixed_edges=(1, 2, 3), monotonic=False),
    "age": BinSpec(fixed_edges=(25, 30, 35, 40, 50, 60), monotonic=False),
}


def utilisation(df: pd.DataFrame) -> pd.DataFrame:
    """Bill amount divided by credit limit for each month (how "full" the card is).

    Args:
        df: Cleaned table with ``bill_amt1..6`` and ``limit_bal``.

    Returns:
        Columns ``util1..util6``. Negative values mean a credit balance.
    """
    return pd.DataFrame(
        {f"util{i}": df[f"bill_amt{i}"] / df["limit_bal"] for i in MONTHS}, index=df.index
    )


def payment_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Share of the previous statement paid each month, capped to [0, 1].

    ``pay_amt{i}`` is matched with ``bill_amt{i+1}``: the payment made in a month
    settles the bill issued the month before. This gives five ratios. When that
    bill is zero or negative there is nothing to pay, so the ratio is NaN
    ("no balance due") rather than an invented number. Paying more than the bill
    counts as 1 (paid in full).

    Args:
        df: Cleaned table with ``pay_amt1..5`` and ``bill_amt2..6``.

    Returns:
        Columns ``pay_ratio1..pay_ratio5``.
    """
    ratios = {}
    for i in range(1, 6):
        bill = df[f"bill_amt{i + 1}"]
        ratios[f"pay_ratio{i}"] = (df[f"pay_amt{i}"] / bill.where(bill > 0)).clip(0, 1)
    return pd.DataFrame(ratios, index=df.index)


def max_delinquency(df: pd.DataFrame) -> pd.Series:
    """Worst repayment status over the six months (``max_delinquency_6m``).

    Args:
        df: Table with ``pay_1..pay_6``.

    Returns:
        The highest status per customer.
    """
    return df[STATUS_COLUMNS].max(axis=1).rename("max_delinquency_6m")


def months_zero_payment(df: pd.DataFrame) -> pd.Series:
    """Number of months in which nothing was paid (``months_zero_payment_6m``).

    Args:
        df: Table with ``pay_amt1..pay_amt6``.

    Returns:
        A count from 0 to 6 per customer.
    """
    return (df[PAYMENT_COLUMNS] == 0).sum(axis=1).rename("months_zero_payment_6m")


def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add summary features that describe six months of behaviour.

    - ``max_delinquency_6m``: worst repayment status in the six months.
    - ``months_delinquent_6m``: months with status 1 or more (at least one month late).
    - ``util_1``, ``avg_util_6m``, ``max_util_6m``: latest, average and peak utilisation.
    - ``util_trend``: latest minus oldest utilisation (positive = balance growing).
    - ``pay_ratio_1``: share of the August bill paid in September (NaN if nothing was due).
    - ``avg_pay_ratio_6m``: average share paid over the five bill/payment pairs (NaN if never due).
    - ``months_zero_payment_6m``: months in which nothing was paid.

    Args:
        df: Cleaned table from :func:`scorecard.data.clean`.

    Returns:
        A copy with the new columns added.
    """
    out = df.copy()
    status = out[STATUS_COLUMNS]
    out["max_delinquency_6m"] = max_delinquency(out)
    out["months_delinquent_6m"] = (status >= 1).sum(axis=1)

    util = utilisation(out)
    out["util_1"] = util["util1"]
    out["avg_util_6m"] = util.mean(axis=1)
    out["max_util_6m"] = util.max(axis=1)
    out["util_trend"] = util["util1"] - util["util6"]

    ratios = payment_ratios(out)
    out["pay_ratio_1"] = ratios["pay_ratio1"]
    out["avg_pay_ratio_6m"] = ratios.mean(axis=1)
    out["months_zero_payment_6m"] = months_zero_payment(out)

    logger.info(
        "Added behavioural features; no balance due: %d (pay_ratio_1), %d (avg_pay_ratio_6m)",
        int(out["pay_ratio_1"].isna().sum()), int(out["avg_pay_ratio_6m"].isna().sum()),
    )
    return out
