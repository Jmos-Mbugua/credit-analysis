"""Fine and coarse classing, Weight of Evidence (WoE) and Information Value (IV).

Conventions (standard in bank scorecards):
- "good" = ``default`` 0, "bad" = ``default`` 1.
- WoE = ln(% of all goods in the bin / % of all bads in the bin).
  Positive WoE = safer than average, negative = riskier.
- IV = sum over bins of (% goods - % bads) * WoE: how well the variable separates
  goods from bads overall.
- Bins are right-closed intervals ``(lower, upper]``; missing values get their own bin.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "default"
MISSING_LABEL = "Missing"
ZERO_COUNT_ADJUSTMENT = 0.5  # used in place of a zero count so WoE stays finite

# Upper IV limit of each strength band (Siddiqi's rule of thumb).
IV_STRENGTH_BANDS: list[tuple[float, str]] = [
    (0.02, "useless"),
    (0.1, "weak"),
    (0.3, "medium"),
    (0.5, "strong"),
    (np.inf, "suspicious"),
]


@dataclass(frozen=True)
class BinSpec:
    """How to bin one variable.

    Attributes:
        fixed_edges: Interior edges set by a business rule (e.g. repayment statuses).
            If ``None``, fine bins are quantiles.
        n_fine: Number of quantile fine bins when ``fixed_edges`` is ``None``.
        monotonic: Whether coarse bins must have a default rate that only rises or only falls.
        min_share: Smallest share of all customers a (non-missing) bin may hold.
    """

    fixed_edges: tuple[float, ...] | None = None
    n_fine: int = 20
    monotonic: bool = True
    min_share: float = 0.05


@dataclass
class Binning:
    """A fitted binning: the edges and the per-bin table learned on training data."""

    variable: str
    edges: list[float]
    table: pd.DataFrame
    integer_valued: bool = False
    woe_by_bin: dict[str, float] = field(default_factory=dict)

    @property
    def iv(self) -> float:
        """Information Value of the variable."""
        return float(self.table["iv"].sum())


def iv_strength(iv: float) -> str:
    """Name the predictive strength of an IV value.

    Args:
        iv: Information Value.

    Returns:
        One of ``useless``, ``weak``, ``medium``, ``strong``, ``suspicious``.
    """
    return next(label for upper, label in IV_STRENGTH_BANDS if iv < upper)


def woe_iv(goods: np.ndarray, bads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weight of Evidence and IV contribution of each bin.

    Args:
        goods: Number of goods per bin.
        bads: Number of bads per bin.

    Returns:
        ``(woe, iv)`` arrays, one value per bin.
    """
    goods = np.asarray(goods, dtype=float)
    bads = np.asarray(bads, dtype=float)
    pct_good = np.where(goods == 0, ZERO_COUNT_ADJUSTMENT, goods) / goods.sum()
    pct_bad = np.where(bads == 0, ZERO_COUNT_ADJUSTMENT, bads) / bads.sum()
    woe = np.log(pct_good / pct_bad)
    return woe, (pct_good - pct_bad) * woe


def fine_edges(x: pd.Series, spec: BinSpec) -> list[float]:
    """Starting edges before merging, including -inf and +inf at the ends.

    Quantile edges always include 0 when the variable has values on both sides,
    so credit balances (negative) are never mixed with positive balances at this stage.

    Args:
        x: Values of the variable (missing values are ignored).
        spec: Binning settings.

    Returns:
        Sorted unique edges.
    """
    values = x.dropna()
    if spec.fixed_edges is not None:
        interior = list(spec.fixed_edges)
    else:
        interior = list(np.quantile(values, np.linspace(0, 1, spec.n_fine + 1)[1:-1]))
        if values.min() < 0 < values.max():
            interior.append(0.0)
    interior = [e for e in interior if values.min() <= e < values.max()]
    return [-np.inf, *sorted(set(interior)), np.inf]


def _counts(x: pd.Series, y: pd.Series, edges: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Goods and bads in each interval defined by ``edges`` (missing values excluded)."""
    idx = pd.cut(x, bins=edges, labels=False)
    known = idx.notna()
    n_bins = len(edges) - 1
    bads = np.bincount(idx[known].astype(int), weights=y[known], minlength=n_bins)
    totals = np.bincount(idx[known].astype(int), minlength=n_bins)
    return totals - bads, bads


def _trend(x: pd.Series, y: pd.Series) -> int:
    """+1 if the default rate rises with ``x`` overall, -1 if it falls."""
    known = x.notna()
    corr = x[known].rank().corr(y[known])
    return 1 if corr >= 0 else -1


def merge_bins(
    edges: list[float],
    goods: np.ndarray,
    bads: np.ndarray,
    n_total: int,
    spec: BinSpec,
    trend: int = 1,
) -> list[float]:
    """Coarse classing: merge neighbouring bins until every rule is met.

    Rules, checked in this order:
    1. Every bin has at least one good and one bad, and at least ``spec.min_share``
       of all customers. The smallest offending bin is merged with the neighbour
       whose default rate is closest.
    2. If ``spec.monotonic``, the default rate moves in the direction ``trend`` from
       bin to bin. Of the pairs that break this, the one with the smallest
       difference in default rate is merged first.

    Args:
        edges: Fine edges, including -inf and +inf.
        goods: Goods per fine bin.
        bads: Bads per fine bin.
        n_total: Number of customers used for bin shares (including missing).
        spec: Binning settings.
        trend: +1 if the default rate should rise with the variable, -1 if it should fall.

    Returns:
        The coarse edges.
    """
    edges = list(edges)
    goods = list(np.asarray(goods, dtype=float))
    bads = list(np.asarray(bads, dtype=float))

    def merge(i: int) -> None:
        """Merge bin i with bin i + 1."""
        goods[i] += goods.pop(i + 1)
        bads[i] += bads.pop(i + 1)
        del edges[i + 1]

    while len(goods) > 1:
        g, b = np.array(goods), np.array(bads)
        share = (g + b) / n_total
        rate = np.divide(b, g + b, out=np.zeros_like(b), where=(g + b) > 0)  # empty bins: 0
        too_small = (g == 0) | (b == 0) | (share < spec.min_share)
        if too_small.any():
            i = int(np.argmin(np.where(too_small, share, np.inf)))
            if i == 0:
                merge(0)
            elif i == len(goods) - 1:
                merge(i - 1)
            else:
                closer_left = abs(rate[i] - rate[i - 1]) <= abs(rate[i] - rate[i + 1])
                merge(i - 1 if closer_left else i)
            continue
        if spec.monotonic:
            steps = trend * np.diff(rate)
            violations = np.flatnonzero(steps < 0)
            if violations.size:
                merge(int(violations[np.argmin(np.abs(np.diff(rate))[violations])]))
                continue
        break
    return edges


def bin_labels(edges: list[float], integer_valued: bool) -> list[str]:
    """Readable labels for intervals ``(lower, upper]``.

    Integer variables get labels like ``"0"``, ``"1 to 3"``, ``"<= -1"``, ``">= 3"``;
    continuous ones get ``"<= 0.1"``, ``"(0.1, 0.5]"``, ``"> 0.5"``.

    Args:
        edges: Bin edges including -inf and +inf.
        integer_valued: Whether the variable only takes whole numbers.

    Returns:
        One label per bin.
    """
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if integer_valued:
            first, last = np.floor(lo) + 1, np.floor(hi)
            if np.isinf(lo) and np.isinf(hi):
                labels.append("all")
            elif np.isinf(lo):
                labels.append(f"<= {last:.0f}")
            elif np.isinf(hi):
                labels.append(f">= {first:.0f}")
            elif first == last:
                labels.append(f"{last:.0f}")
            else:
                labels.append(f"{first:.0f} to {last:.0f}")
        elif np.isinf(lo) and np.isinf(hi):
            labels.append("all")
        elif np.isinf(lo):
            labels.append(f"<= {hi:.4g}")
        elif np.isinf(hi):
            labels.append(f"> {lo:.4g}")
        else:
            labels.append(f"({lo:.4g}, {hi:.4g}]")
    return labels


def binning_table(
    x: pd.Series, y: pd.Series, edges: list[float], integer_valued: bool
) -> pd.DataFrame:
    """Counts, default rate, WoE and IV per bin, plus a missing bin if needed.

    Args:
        x: Values of the variable.
        y: 0/1 target.
        edges: Coarse edges.
        integer_valued: Whether to use integer-style labels.

    Returns:
        One row per bin, indexed by bin label.
    """
    goods, bads = _counts(x, y, edges)
    labels = bin_labels(edges, integer_valued)
    missing = x.isna()
    if missing.any():
        bads_missing = y[missing].sum()
        goods = np.append(goods, missing.sum() - bads_missing)
        bads = np.append(bads, bads_missing)
        labels.append(MISSING_LABEL)
    woe, iv = woe_iv(goods, bads)
    customers = goods + bads
    table = pd.DataFrame(
        {
            "customers": customers.astype(int),
            "goods": goods.astype(int),
            "bads": bads.astype(int),
            "default_rate": bads / customers,
            "share": customers / customers.sum(),
            "woe": woe,
            "iv": iv,
        },
        index=pd.Index(labels, name="bin"),
    )
    return table


def fit_binning(x: pd.Series, y: pd.Series, spec: BinSpec | None = None) -> Binning:
    """Fine-class, coarse-class and compute WoE/IV for one variable.

    Args:
        x: Training values of the variable.
        y: Training 0/1 target.
        spec: Binning settings (defaults to quantile bins, monotonic, 5% minimum).

    Returns:
        The fitted binning.
    """
    spec = spec or BinSpec()
    integer_valued = bool((x.dropna() % 1 == 0).all())
    edges = fine_edges(x, spec)
    goods, bads = _counts(x, y, edges)
    edges = merge_bins(edges, goods, bads, n_total=len(x), spec=spec, trend=_trend(x, y))
    table = binning_table(x, y, edges, integer_valued)
    binning = Binning(x.name, edges, table, integer_valued, table["woe"].to_dict())
    logger.info("%s: %d fine -> %d coarse bins, IV %.3f", x.name, len(goods), len(edges) - 1, binning.iv)
    return binning


def assign_bins(x: pd.Series, binning: Binning) -> pd.Series:
    """Label of the bin each value falls into (``Missing`` for missing values).

    Args:
        x: Values of the variable (training or new data).
        binning: Fitted binning for this variable.

    Returns:
        Bin labels with the same index as ``x``.
    """
    labels = bin_labels(binning.edges, binning.integer_valued)
    codes = pd.cut(x, bins=binning.edges, labels=False)
    out = pd.Series(MISSING_LABEL, index=x.index, name=x.name, dtype=object)
    known = codes.notna()
    out[known] = np.array(labels, dtype=object)[codes[known].astype(int)]
    return out


def transform_woe(x: pd.Series, binning: Binning) -> pd.Series:
    """Replace each value by the WoE of its bin (learned on training data).

    Missing values take the WoE of the missing bin; if training had no missing
    values, they get 0 (neutral, "no evidence either way").

    Args:
        x: Values of the variable (training or new data).
        binning: Fitted binning for this variable.

    Returns:
        WoE values with the same index as ``x``.
    """
    labels = bin_labels(binning.edges, binning.integer_valued)
    codes = pd.cut(x, bins=binning.edges, labels=False)
    woe_values = np.array([binning.woe_by_bin[label] for label in labels])
    out = pd.Series(np.nan, index=x.index, name=x.name)
    known = codes.notna()
    out[known] = woe_values[codes[known].astype(int)]
    out[~known] = binning.woe_by_bin.get(MISSING_LABEL, 0.0)
    return out


def fit_all(
    df: pd.DataFrame, variables: list[str], specs: dict[str, BinSpec] | None = None
) -> dict[str, Binning]:
    """Fit a binning for each variable on the training data.

    Args:
        df: Training table with the variables and ``default``.
        variables: Variables to bin.
        specs: Per-variable settings; variables not listed use the default.

    Returns:
        Fitted binnings keyed by variable name.
    """
    specs = specs or {}
    return {v: fit_binning(df[v], df[TARGET], specs.get(v)) for v in variables}


def transform_all(df: pd.DataFrame, binnings: dict[str, Binning]) -> pd.DataFrame:
    """WoE-transform every binned variable, keeping ``id`` (if present) and the target.

    Args:
        df: Table with the raw variables and ``default``.
        binnings: Output of :func:`fit_all`.

    Returns:
        ``id``, one WoE column per variable and ``default``, same index as ``df``.
    """
    woe = pd.DataFrame({v: transform_woe(df[v], b) for v, b in binnings.items()}, index=df.index)
    if "id" in df.columns:
        woe.insert(0, "id", df["id"])
    woe[TARGET] = df[TARGET]
    return woe


def iv_table(binnings: dict[str, Binning], feature_types: dict[str, str]) -> pd.DataFrame:
    """Summary of all variables ranked by IV.

    Args:
        binnings: Output of :func:`fit_all`.
        feature_types: Behaviour type of each variable.

    Returns:
        Columns ``variable``, ``iv``, ``strength``, ``n_bins``, ``type``, sorted by IV.
    """
    rows = [
        {
            "variable": v,
            "iv": b.iv,
            "strength": iv_strength(b.iv),
            "n_bins": len(b.table),
            "type": feature_types.get(v, "other"),
        }
        for v, b in binnings.items()
    ]
    return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)


def all_binning_tables(binnings: dict[str, Binning]) -> pd.DataFrame:
    """Every variable's binning table stacked into one long table.

    Args:
        binnings: Output of :func:`fit_all`.

    Returns:
        One row per (variable, bin).
    """
    return pd.concat(
        {v: b.table for v, b in binnings.items()}, names=["variable", "bin"]
    ).reset_index()


def save_binnings(binnings: dict[str, Binning], path: Path) -> None:
    """Save fitted binnings (edges and tables) so new customers can be binned the same way.

    Args:
        binnings: Output of :func:`fit_all`.
        path: Destination pickle file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(binnings, f)
    logger.info("Saved %d binnings to %s", len(binnings), path)


def load_binnings(path: Path) -> dict[str, Binning]:
    """Load binnings saved with :func:`save_binnings`.

    Only load files this project created: pickle files can run code when loaded.

    Args:
        path: File written by :func:`save_binnings`.

    Returns:
        Fitted binnings keyed by variable name.
    """
    with path.open("rb") as f:
        return pickle.load(f)
