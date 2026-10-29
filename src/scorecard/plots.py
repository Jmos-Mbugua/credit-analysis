"""Charts for binning and feature selection, styled like :func:`scorecard.eda.plot_default_rate`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from scorecard.eda import BAR_COLOR, GRID_COLOR, INK_PRIMARY, INK_SECONDARY

MUTED_BAR = "#c3c2b7"
GOOD_COLOR = "#2a78d6"  # categorical slot 1
BAD_COLOR = "#eb6834"  # categorical slot 2
# Diverging: red (negative) - neutral grey - blue (positive).
CORRELATION_CMAP = LinearSegmentedColormap.from_list("corr", ["#c0392f", "#f0efec", "#1f5fae"])


def _style(ax: plt.Axes) -> None:
    """Recessive axes: no box, light grid, muted ticks."""
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)


def plot_iv(iv_table: pd.DataFrame, min_iv: float = 0.02) -> plt.Axes:
    """Horizontal bars of IV per variable with the strength band boundaries.

    Variables below ``min_iv`` are drawn in grey (they will be dropped).

    Args:
        iv_table: Output of :func:`scorecard.binning.iv_table`.
        min_iv: Threshold below which a variable is useless.

    Returns:
        The matplotlib axes.
    """
    table = iv_table.sort_values("iv")
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(table) + 1.2))
    colors = [BAR_COLOR if v >= min_iv else MUTED_BAR for v in table["iv"]]
    ax.barh(table["variable"], table["iv"], color=colors, height=0.7)
    for x, label in [(0.02, "0.02"), (0.1, "weak | medium 0.1"), (0.3, "0.3 strong"), (0.5, "0.5 suspicious")]:
        ax.axvline(x, color=INK_SECONDARY, linestyle="--", linewidth=0.8)
        ax.annotate(label, xy=(x, 1), xycoords=("data", "axes fraction"), xytext=(3, 2),
                    textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    ax.set_title("Information Value per candidate variable (training data)",
                 loc="left", color=INK_PRIMARY, pad=18)
    ax.set_xlabel("IV", color=INK_SECONDARY)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    return ax


def plot_correlation_heatmap(corr: pd.DataFrame, threshold: float = 0.7) -> plt.Axes:
    """Heatmap of a correlation matrix; cells above ``threshold`` in absolute value are boxed.

    Args:
        corr: Square correlation matrix.
        threshold: Absolute correlation to highlight.

    Returns:
        The matplotlib axes.
    """
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.5 * n + 3, 0.5 * n + 2))
    image = ax.imshow(corr.values, cmap=CORRELATION_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}".replace("0.", ".").replace("1.00", "1"),
                    ha="center", va="center", fontsize=6.5,
                    color="white" if abs(value) > 0.6 else INK_PRIMARY)
            if i != j and abs(value) > threshold:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=INK_PRIMARY, linewidth=1.5))
    fig.colorbar(image, ax=ax, shrink=0.6, label="Correlation of WoE values")
    ax.set_title(f"Correlation of WoE variables (boxed: |r| > {threshold})",
                 loc="left", color=INK_PRIMARY)
    _style(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return ax


def plot_pd_distribution(samples: dict[str, tuple[pd.Series, pd.Series]], bins: int = 40) -> plt.Figure:
    """Histogram of predicted PD for goods and bads, one panel per sample.

    Each group is drawn as a share of that group (not a count), so goods and bads
    are comparable even though there are about 3.5 times more goods.

    Args:
        samples: ``{"Train": (pd_hat, default), "Test": (...)}``.
        bins: Number of histogram bins between 0 and 1.

    Returns:
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    edges = np.linspace(0, 1, bins + 1)
    for ax, (name, (pd_hat, actual)) in zip(axes, samples.items()):
        for label, mask, color in [("Goods (default = 0)", actual == 0, GOOD_COLOR),
                                   ("Bads (default = 1)", actual == 1, BAD_COLOR)]:
            weights = np.full(int(mask.sum()), 1 / mask.sum())
            ax.hist(pd_hat[mask], bins=edges, weights=weights, histtype="step",
                    linewidth=2, color=color, label=label)
        ax.set_title(name, loc="left", color=INK_PRIMARY)
        ax.set_xlabel("Predicted probability of default", color=INK_SECONDARY)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0].set_ylabel("Share of the group", color=INK_SECONDARY)
    axes[0].legend(frameon=False, labelcolor=INK_SECONDARY)
    fig.suptitle("Predicted PD: goods vs bads", x=0.01, ha="left", color=INK_PRIMARY)
    fig.tight_layout()
    return fig


def plot_score_distribution(scores: pd.Series, default: pd.Series, width: int = 5) -> plt.Axes:
    """Overlapping histograms of scores for goods and bads.

    Each group is drawn as a share of that group, so the two shapes are comparable.

    Args:
        scores: Scorecard totals.
        default: 0/1 outcomes for the same customers.
        width: Histogram bin width in points.

    Returns:
        The matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    edges = np.arange(scores.min() // width * width, scores.max() + 2 * width, width)
    for label, mask, color in [("Goods (default = 0)", default == 0, GOOD_COLOR),
                               ("Bads (default = 1)", default == 1, BAD_COLOR)]:
        weights = np.full(int(mask.sum()), 1 / mask.sum())
        ax.hist(scores[mask], bins=edges, weights=weights, color=color, alpha=0.35)
        ax.hist(scores[mask], bins=edges, weights=weights, histtype="step", linewidth=1.5,
                color=color, label=label)
    ax.set_title("Score distribution on the test set: goods vs bads", loc="left", color=INK_PRIMARY)
    ax.set_xlabel("Score (higher = safer)", color=INK_SECONDARY)
    ax.set_ylabel("Share of the group", color=INK_SECONDARY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return ax


def _percent_axis(ax: plt.Axes, axis: str = "both") -> None:
    """Format one or both axes as percentages."""
    formatter = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def plot_roc(points: pd.DataFrame, gini_value: float, title: str) -> plt.Axes:
    """ROC curve with the random-model diagonal.

    Args:
        points: Output of :func:`scorecard.metrics.roc_points`.
        gini_value: Gini to show in the title.
        title: Sample name, e.g. "test set".

    Returns:
        The matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(points["false_positive_rate"], points["true_positive_rate"], color=GOOD_COLOR, linewidth=2)
    ax.fill_between(points["false_positive_rate"], points["false_positive_rate"],
                    points["true_positive_rate"], color=GOOD_COLOR, alpha=0.12)
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate("Random model (Gini 0)", xy=(0.55, 0.47), color=INK_SECONDARY, fontsize=9, rotation=42)
    ax.set_title(f"ROC curve, {title} (Gini {gini_value:.3f})", loc="left", color=INK_PRIMARY)
    ax.set_xlabel("Share of goods flagged (false positive rate)", color=INK_SECONDARY)
    ax.set_ylabel("Share of bads flagged (true positive rate)", color=INK_SECONDARY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _percent_axis(ax)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    return ax


def plot_ks(table: pd.DataFrame, ks_value: float, ks_score: float, title: str) -> plt.Axes:
    """Cumulative share of goods and bads by score, with the KS gap marked.

    Args:
        table: Output of :func:`scorecard.metrics.ks_table`.
        ks_value: KS statistic.
        ks_score: Score where the gap is largest.
        title: Sample name, e.g. "test set".

    Returns:
        The matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(table["score"], table["cum_bads"], where="post", color=BAD_COLOR, linewidth=2,
            label="Bads: share at or below score")
    ax.step(table["score"], table["cum_goods"], where="post", color=GOOD_COLOR, linewidth=2,
            label="Goods: share at or below score")
    row = table.loc[table["score"] == ks_score].iloc[0]
    ax.vlines(ks_score, row["cum_goods"], row["cum_bads"], color=INK_PRIMARY, linewidth=1.5)
    ax.annotate(f"KS = {ks_value:.2f} at score {ks_score:.0f}\n"
                f"{row['cum_bads']:.0%} of bads vs {row['cum_goods']:.0%} of goods",
                xy=(ks_score, (row["cum_goods"] + row["cum_bads"]) / 2), xytext=(0.03, 0.72),
                textcoords="axes fraction", ha="left", va="center", fontsize=9, color=INK_PRIMARY,
                arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "linewidth": 0.8})
    ax.set_title(f"KS chart, {title}", loc="left", color=INK_PRIMARY)
    ax.set_xlabel("Score (higher = safer)", color=INK_SECONDARY)
    ax.set_ylabel("Cumulative share of the group", color=INK_SECONDARY)
    ax.set_ylim(0, 1.02)
    _percent_axis(ax, "y")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return ax


def plot_calibration(table: pd.DataFrame, title: str) -> plt.Axes:
    """Predicted vs actual default rate per decile of predicted PD.

    Args:
        table: Output of :func:`scorecard.metrics.calibration_table`.
        title: Sample name, e.g. "test set".

    Returns:
        The matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = table["decile"].astype(int)
    ax.bar(x, table["actual_default_rate"], color=GOOD_COLOR, width=0.65, label="Actual default rate")
    ax.plot(x, table["predicted_default_rate"], color=BAD_COLOR, marker="o", markersize=7,
            linewidth=2, label="Predicted (average PD)")
    ax.set_title(f"Calibration by decile of predicted PD, {title}", loc="left", color=INK_PRIMARY)
    ax.set_xlabel("Decile of predicted PD (1 = safest 10%)", color=INK_SECONDARY)
    ax.set_ylabel("Default rate", color=INK_SECONDARY)
    ax.set_xticks(x)
    _percent_axis(ax, "y")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return ax


def plot_strategy_curves(table: pd.DataFrame, ks_score: float | None = None) -> plt.Axes:
    """Approval rate and bad rates against cut-off.

    Args:
        table: Output of :func:`scorecard.strategy.strategy_table`.
        ks_score: Optional score to mark (the KS cut-off).

    Returns:
        The matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    series = [("approval_rate", "Approval rate", GOOD_COLOR, "-"),
              ("approved_bad_rate", "Bad rate among approved", BAD_COLOR, "-"),
              ("declined_bad_rate", "Bad rate among declined", BAD_COLOR, "--")]
    for column, label, color, style in series:
        ax.plot(table["cutoff"], table[column], color=color, linestyle=style, marker="o",
                markersize=5, linewidth=2, label=label)
    if ks_score is not None:
        ax.axvline(ks_score, color=INK_SECONDARY, linestyle=":", linewidth=1)
        ax.annotate(f"KS cut-off {ks_score:.0f}", xy=(ks_score, 1.0), xytext=(4, -2),
                    textcoords="offset points", fontsize=9, color=INK_SECONDARY, va="top")
    ax.set_title("Approval rate and bad rates by cut-off (test set)", loc="left", color=INK_PRIMARY)
    ax.set_xlabel("Cut-off: approve if score >= cut-off", color=INK_SECONDARY)
    ax.set_ylabel("Share of customers", color=INK_SECONDARY)
    ax.set_ylim(0, 1.02)
    _percent_axis(ax, "y")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="center left")
    _style(ax)
    fig.tight_layout()
    return ax


def plot_profit_curve(profit_table: pd.DataFrame, best: pd.Series, note: str) -> plt.Axes:
    """Expected profit against cut-off, with the profit-maximising cut-off marked.

    Args:
        profit_table: Output of :func:`scorecard.strategy.expected_profit`.
        best: Output of :func:`scorecard.strategy.best_cutoff`.
        note: Assumptions to print under the title.

    Returns:
        The matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    millions = profit_table["profit"] / 1e6
    ax.plot(profit_table["cutoff"], millions, color=GOOD_COLOR, linewidth=2)
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.plot(best["cutoff"], best["profit"] / 1e6, marker="o", markersize=9, color=BAD_COLOR)
    ax.annotate(f"Best cut-off {best['cutoff']:.0f}\n"
                f"profit NT${best['profit'] / 1e6:.1f}M, approval {best['approval_rate']:.0%}",
                xy=(best["cutoff"], best["profit"] / 1e6), xytext=(0.97, 0.45), textcoords="axes fraction",
                ha="right", va="center", fontsize=9, color=INK_PRIMARY,
                arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "linewidth": 0.8})
    ax.set_title("Expected profit by cut-off (test set)", loc="left", color=INK_PRIMARY, pad=22)
    ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("Cut-off: approve if score >= cut-off", color=INK_SECONDARY)
    ax.set_ylabel("Profit (NT$ millions)", color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    return ax
