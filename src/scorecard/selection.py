"""Feature selection: IV filter, correlation filter and VIF filter.

Each filter answers a different question:
- IV: does the variable separate goods from bads at all?
- Correlation: do two variables tell the same story? Keep the stronger one.
- VIF: can a variable be predicted from a *combination* of the others? If so, its
  coefficient in the logistic regression becomes unstable and hard to explain.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

logger = logging.getLogger(__name__)

MIN_IV = 0.02
WEAK_IV = 0.1
MAX_CORRELATION = 0.7
MAX_VIF = 5.0
MAX_VARIABLES = 12
TARGET = "default"


def filter_by_iv(
    iv: pd.Series, min_iv: float = MIN_IV, weak_below: float = WEAK_IV
) -> pd.DataFrame:
    """Keep variables with enough predictive power and flag the weak ones.

    Args:
        iv: IV per variable (index = variable name).
        min_iv: Variables below this are dropped (no useful information).
        weak_below: Kept variables below this are flagged as weak.

    Returns:
        One row per variable with ``iv``, ``kept`` and ``weak`` columns, sorted by IV.
    """
    table = pd.DataFrame({"iv": iv}).sort_values("iv", ascending=False)
    table["kept"] = table["iv"] >= min_iv
    table["weak"] = table["kept"] & (table["iv"] < weak_below)
    for variable, row in table.iterrows():
        if not row["kept"]:
            logger.info("IV filter: drop %s (IV %.3f < %.2f)", variable, row["iv"], min_iv)
        elif row["weak"]:
            logger.info("IV filter: keep %s but flag as weak (IV %.3f < %.2f)", variable, row["iv"], weak_below)
    logger.info("IV filter: %d of %d variables kept", int(table["kept"].sum()), len(table))
    return table


def drop_correlated(
    woe: pd.DataFrame, iv: pd.Series, threshold: float = MAX_CORRELATION
) -> tuple[list[str], pd.DataFrame]:
    """When two variables have |correlation| above ``threshold``, keep only the one with higher IV.

    Variables are visited from highest to lowest IV. Each is kept unless it is too
    correlated with a variable already kept, so every drop is justified by a
    stronger variable that stays in the list. (Working through pairs in order of
    correlation instead can drop a variable because of a partner that is itself
    dropped later.)

    Args:
        woe: WoE-transformed training variables (one column per variable).
        iv: IV per variable, used to decide which of a pair to keep.
        threshold: Largest absolute correlation allowed between kept variables.

    Returns:
        ``(kept, decisions)``: the kept variables in their original column order, and
        one row per dropped variable with ``dropped``, ``kept_instead`` (the kept
        variable it is most correlated with), ``correlation`` and ``reason``.
    """
    corr = woe.corr()
    kept: list[str] = []
    decisions = []
    for variable in iv[woe.columns].sort_values(ascending=False).index:
        conflicts = corr.loc[variable, kept].abs()
        conflicts = conflicts[conflicts > threshold]
        if conflicts.empty:
            kept.append(variable)
            continue
        winner = conflicts.idxmax()
        value = float(corr.loc[variable, winner])
        reason = f"r = {value:.2f} with {winner}; IV {iv[variable]:.3f} < {iv[winner]:.3f}"
        decisions.append({"dropped": variable, "kept_instead": winner,
                          "correlation": value, "reason": reason})
        logger.info("Correlation filter: drop %s (%s)", variable, reason)
    kept = [c for c in woe.columns if c in kept]
    logger.info("Correlation filter: %d of %d variables kept", len(kept), woe.shape[1])
    columns = ["dropped", "kept_instead", "correlation", "reason"]
    return kept, pd.DataFrame(decisions, columns=columns)


def compute_vif(df: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor of each column.

    VIF = 1 / (1 - R²), where R² comes from predicting the column from all the
    others. VIF 1 means unrelated; VIF 5 means 80% of its variation is explained
    by the other variables.

    Args:
        df: Numeric columns (no target).

    Returns:
        VIF per column, sorted from highest to lowest.
    """
    x = add_constant(df.astype(float), has_constant="add")
    vif = {col: variance_inflation_factor(x.values, i) for i, col in enumerate(x.columns) if col != "const"}
    return pd.Series(vif, name="vif").sort_values(ascending=False)


def drop_high_vif(
    df: pd.DataFrame, max_vif: float = MAX_VIF
) -> tuple[list[str], pd.DataFrame]:
    """Repeatedly drop the variable with the highest VIF until all are at or below ``max_vif``.

    VIF is recomputed after each drop, because removing one variable lowers the
    VIF of the variables it was entangled with.

    Args:
        df: WoE-transformed training variables.
        max_vif: Largest VIF allowed.

    Returns:
        ``(kept, steps)``: the kept variables, and one row per round with the
        ``step``, ``variable`` with the highest VIF, its ``vif`` and whether it was ``dropped``.
    """
    kept = list(df.columns)
    steps = []
    while len(kept) > 1:
        vif = compute_vif(df[kept])
        worst, value = vif.index[0], float(vif.iloc[0])
        dropped = value > max_vif
        steps.append({"step": len(steps) + 1, "variable": worst, "vif": value, "dropped": dropped})
        if not dropped:
            logger.info("VIF filter: highest VIF is %s (%.2f) <= %.1f, stop", worst, value, max_vif)
            break
        logger.info("VIF filter step %d: drop %s (VIF %.2f > %.1f)", len(steps), worst, value, max_vif)
        kept.remove(worst)
    logger.info("VIF filter: %d of %d variables kept", len(kept), df.shape[1])
    return kept, pd.DataFrame(steps, columns=["step", "variable", "vif", "dropped"])


def cap_by_iv(
    variables: list[str], iv: pd.Series, max_variables: int = MAX_VARIABLES
) -> list[str]:
    """Keep at most ``max_variables``, choosing those with the highest IV.

    Args:
        variables: Variables that passed the earlier filters.
        iv: IV per variable.
        max_variables: Largest number of variables to keep.

    Returns:
        The kept variables, highest IV first.
    """
    ranked = iv[variables].sort_values(ascending=False)
    for variable, value in ranked.iloc[max_variables:].items():
        logger.info("IV cap: drop %s (IV %.3f, rank > %d)", variable, value, max_variables)
    return ranked.index[:max_variables].tolist()


def _logistic_regression() -> LogisticRegression:
    """Logistic regression for quick checks (its default L2 penalty is negligible with 21,000 rows)."""
    return LogisticRegression(solver="newton-cholesky")


def cross_validated_gini(
    woe: pd.DataFrame, variables: list[str], folds: int = 5, seed: int = 42
) -> float:
    """Average Gini of a logistic regression across cross-validation folds of the training data.

    Gini = 2 * AUC - 1: 0 means no better than random, 1 means perfect ranking.
    Cross-validation (fit on 4/5, score on the other 1/5, five times) gives an
    honest estimate without touching the test set.

    Args:
        woe: WoE-transformed training data including ``default``.
        variables: Variables to use.
        folds: Number of folds.
        seed: Random seed for the fold split.

    Returns:
        Mean Gini across folds.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    auc = cross_val_score(_logistic_regression(), woe[variables], woe[TARGET], cv=cv, scoring="roc_auc")
    return float(2 * auc.mean() - 1)


def wrong_sign_variables(woe: pd.DataFrame, variables: list[str]) -> list[str]:
    """Variables whose logistic regression coefficient has the unexpected sign.

    Higher WoE means safer, so every coefficient for P(default) should be negative.
    A positive one means the variable's information overlaps with the others and
    the model is using it to "correct" them: hard to explain and unstable.

    Args:
        woe: WoE-transformed training data including ``default``.
        variables: Variables to use.

    Returns:
        Variables with a positive coefficient.
    """
    model = _logistic_regression().fit(woe[variables], woe[TARGET])
    return [v for v, coef in zip(variables, model.coef_[0]) if coef > 0]


def compare_feature_sets(
    woe: pd.DataFrame, sets: dict[str, list[str]], iv: pd.Series, feature_types: dict[str, str]
) -> pd.DataFrame:
    """Side-by-side summary of candidate variable lists.

    Args:
        woe: WoE-transformed training data including ``default``.
        sets: Named lists of variables.
        iv: IV per variable.
        feature_types: Behaviour type of each variable.

    Returns:
        One row per set: size, number of delinquency variables, delinquency share
        of total IV, cross-validated Gini and wrong-sign variables.
    """
    rows = []
    for name, variables in sets.items():
        delinquency = [v for v in variables if feature_types[v] == "delinquency"]
        rows.append({
            "set": name,
            "n_variables": len(variables),
            "n_delinquency": len(delinquency),
            "delinquency_iv_share": iv[delinquency].sum() / iv[variables].sum(),
            "cv_gini": cross_validated_gini(woe, variables),
            "wrong_sign": ", ".join(wrong_sign_variables(woe, variables)) or "none",
        })
    return pd.DataFrame(rows).set_index("set")


def selection_table(
    variables: list[str], iv_table: pd.DataFrame, reasons: dict[str, str]
) -> pd.DataFrame:
    """Final list with IV, strength, behaviour type and the business reason for each variable.

    Args:
        variables: Selected variables.
        iv_table: Output of :func:`scorecard.binning.iv_table`.
        reasons: One-sentence business reason per variable.

    Returns:
        Columns ``variable``, ``iv``, ``strength``, ``type``, ``business_reason``, highest IV first.

    Raises:
        KeyError: If a selected variable has no business reason.
    """
    missing = [v for v in variables if v not in reasons]
    if missing:
        raise KeyError(f"No business reason recorded for: {missing}")
    table = iv_table.set_index("variable").loc[variables, ["iv", "strength", "type"]]
    table["business_reason"] = [reasons[v] for v in variables]
    return table.sort_values("iv", ascending=False).reset_index()
