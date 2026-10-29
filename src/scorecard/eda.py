"""Exploratory calculations and charts: default rates by group, bands, bill checks."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "default"
BILL_COLUMNS = [f"bill_amt{i}" for i in range(1, 7)]

# Chart styling (reference palette: slot-1 blue, muted ink, recessive grid).
BAR_COLOR = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def default_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count customers and defaulters in each group and compute the default rate.

    Args:
        df: Table with a 0/1 ``default`` column.
        column: Column to group by.

    Returns:
        One row per group with ``customers``, ``defaults``, ``default_rate`` and
        ``share`` (fraction of all customers in the group).
    """
    grouped = df.groupby(column, observed=True)[TARGET]
    table = pd.DataFrame({"customers": grouped.size(), "defaults": grouped.sum()})
    table["default_rate"] = table["defaults"] / table["customers"]
    table["share"] = table["customers"] / table["customers"].sum()
    return table


def quantile_bands(series: pd.Series, q: int = 5) -> pd.Series:
    """Split a numeric column into ``q`` groups of (roughly) equal size.

    Args:
        series: Numeric values, e.g. credit limits.
        q: Number of bands (5 = quintiles).

    Returns:
        Ordered categorical labels such as ``"10,000-50,000"``.
    """
    _, edges = pd.qcut(series, q=q, retbins=True, duplicates="drop")
    labels = [f"{lo:,.0f}-{hi:,.0f}" for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)


def fixed_bands(series: pd.Series, edges: list[float]) -> pd.Series:
    """Split integer values into bands with right-closed edges, e.g. 21-25, 26-30, 61+.

    Args:
        series: Integer values, e.g. ages.
        edges: Band edges; a band covers (lower, upper]. Use ``np.inf`` as the last
            edge for an open-ended top band.

    Returns:
        Ordered categorical labels.
    """
    labels = [
        f"{lo + 1:.0f}+" if np.isinf(hi) else f"{lo + 1:.0f}-{hi:.0f}"
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    return pd.cut(series, bins=edges, labels=labels)


def negative_bill_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count negative bill amounts per month.

    A negative bill means the customer has a credit balance: they paid more than
    they owed (or received a refund), so the bank owes them money.

    Args:
        df: Cleaned table with ``bill_amt1``..``bill_amt6``.

    Returns:
        One row per bill column with the number and share of negative values.
    """
    negative = df[BILL_COLUMNS] < 0
    return pd.DataFrame({"negative_bills": negative.sum(), "share": negative.mean()})


def customers_with_negative_bill(df: pd.DataFrame) -> int:
    """Number of customers with at least one negative bill in the six months.

    Args:
        df: Cleaned table with ``bill_amt1``..``bill_amt6``.

    Returns:
        Count of customers.
    """
    return int((df[BILL_COLUMNS] < 0).any(axis=1).sum())


def plot_default_rate(
    table: pd.DataFrame, title: str, xlabel: str, overall_rate: float
) -> plt.Axes:
    """Bar chart of default rate per group, with the overall rate as a dashed line.

    Args:
        table: Output of :func:`default_rate_by`.
        title: Chart title.
        xlabel: Label for the groups axis.
        overall_rate: Portfolio default rate, drawn for comparison.

    Returns:
        The matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [str(i) for i in table.index]
    bars = ax.bar(labels, table["default_rate"], color=BAR_COLOR, width=0.7)
    ax.bar_label(bars, labels=[f"{r:.0%}" for r in table["default_rate"]],
                 padding=3, fontsize=9, color=INK_SECONDARY)
    ax.axhline(overall_rate, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate(f"All customers\n{overall_rate:.1%}", xy=(1, overall_rate),
                xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, loc="left", color=INK_PRIMARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel("Default rate", color=INK_SECONDARY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, max(table["default_rate"].max(), overall_rate) * 1.15)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    fig.tight_layout()
    return ax
