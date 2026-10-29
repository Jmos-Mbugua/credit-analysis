"""Logistic regression on WoE variables, with sign and significance checks.

Model: log-odds of default = intercept + sum of (coefficient x WoE).

Why statsmodels: it reports standard errors and p-values, which a scorecard needs
to show that every variable genuinely contributes.

Why no class rebalancing (no oversampling, undersampling or class weights): the
model's output is used as a probability of default (PD). Logistic regression fitted
on the real mix of goods and bads produces PDs that match the real default rate.
Rebalancing would push every PD towards 50% and the points scale, cut-offs and
expected-loss numbers built on top would all be wrong. A 22% default rate is
not extreme imbalance, so there is no need for it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper

logger = logging.getLogger(__name__)

TARGET = "default"
CONSTANT = "const"
MAX_P_VALUE = 0.05


def fit_logit(woe: pd.DataFrame, variables: list[str]) -> BinaryResultsWrapper:
    """Fit an (unpenalised) logistic regression of ``default`` on WoE variables.

    Args:
        woe: WoE-transformed training data including ``default``.
        variables: WoE columns to use.

    Returns:
        The fitted statsmodels result.
    """
    x = sm.add_constant(woe[variables], has_constant="add")
    return sm.Logit(woe[TARGET], x).fit(disp=0)


def find_offender(
    result: BinaryResultsWrapper, max_p_value: float = MAX_P_VALUE
) -> tuple[str, str] | None:
    """Find the weakest variable that breaks a rule, if any.

    Rules: every coefficient must be negative (higher WoE = safer = lower risk),
    and every p-value below ``max_p_value`` (the variable genuinely adds information).
    Among the offenders, the one with the highest p-value is the weakest: the data
    gives least evidence that it belongs in the model.

    Args:
        result: Fitted model.
        max_p_value: Significance level.

    Returns:
        ``(variable, reason)`` for the weakest offender, or ``None`` if all rules pass.
    """
    coefs = result.params.drop(CONSTANT)
    p_values = result.pvalues.drop(CONSTANT)
    offenders = coefs.index[(coefs >= 0) | (p_values >= max_p_value)]
    if offenders.empty:
        return None
    weakest = p_values[offenders].idxmax()
    problems = []
    if coefs[weakest] >= 0:
        problems.append(f"positive coefficient {coefs[weakest]:+.3f}")
    if p_values[weakest] >= max_p_value:
        problems.append(f"p-value {p_values[weakest]:.3f} >= {max_p_value}")
    return weakest, " and ".join(problems)


def fit_with_checks(
    woe: pd.DataFrame, variables: list[str], max_p_value: float = MAX_P_VALUE
) -> tuple[BinaryResultsWrapper, list[str], pd.DataFrame]:
    """Fit, then drop the weakest rule-breaking variable and refit until all rules pass.

    Args:
        woe: WoE-transformed training data including ``default``.
        variables: Starting variables.
        max_p_value: Significance level.

    Returns:
        ``(result, kept, steps)``: the final model, its variables, and one row per
        removed variable with ``step``, ``variable`` and ``reason``.
    """
    kept = list(variables)
    steps = []
    while True:
        result = fit_logit(woe, kept)
        offender = find_offender(result, max_p_value)
        if offender is None:
            logger.info("All %d coefficients negative and significant", len(kept))
            return result, kept, pd.DataFrame(steps, columns=["step", "variable", "reason"])
        variable, reason = offender
        steps.append({"step": len(steps) + 1, "variable": variable, "reason": reason})
        logger.info("Step %d: drop %s (%s), refit with %d variables",
                    len(steps), variable, reason, len(kept) - 1)
        kept.remove(variable)
        if not kept:
            raise ValueError("Every variable broke a rule; nothing left to model")


def predict_pd(result: BinaryResultsWrapper, woe: pd.DataFrame, variables: list[str]) -> pd.Series:
    """Probability of default for each customer.

    Args:
        result: Fitted model.
        woe: WoE-transformed data (train or test).
        variables: The model's variables.

    Returns:
        PD between 0 and 1, same index as ``woe``.
    """
    x = sm.add_constant(woe[variables], has_constant="add")
    return pd.Series(result.predict(x), index=woe.index, name="pd")


def average_pd_vs_actual(pd_hat: pd.Series, actual: pd.Series) -> dict[str, float]:
    """Compare the average predicted PD with the actual default rate.

    If the model is well calibrated overall, the two are close.

    Args:
        pd_hat: Predicted probabilities of default.
        actual: Observed 0/1 defaults.

    Returns:
        ``customers``, ``average_pd``, ``actual_default_rate`` and ``difference`` (PD minus actual).
    """
    average_pd, actual_rate = float(pd_hat.mean()), float(actual.mean())
    return {
        "customers": len(actual),
        "average_pd": average_pd,
        "actual_default_rate": actual_rate,
        "difference": average_pd - actual_rate,
    }


def coefficients_table(result: BinaryResultsWrapper) -> pd.DataFrame:
    """Coefficient, standard error, z-statistic, p-value and odds ratio per term.

    Args:
        result: Fitted model.

    Returns:
        One row per term (intercept first).
    """
    return pd.DataFrame({
        "coefficient": result.params,
        "std_error": result.bse,
        "z": result.tvalues,
        "p_value": result.pvalues,
        "odds_ratio": np.exp(result.params),
    }).rename_axis("variable").reset_index()


def odds_effect_by_variable(coefs: pd.DataFrame, binning_tables: pd.DataFrame) -> pd.DataFrame:
    """How much each variable moves the odds of default between its riskiest and safest bin.

    With WoE inputs, a coefficient is hard to read on its own. Moving from the
    lowest-WoE bin (riskiest) to the highest-WoE bin (safest) multiplies the odds of
    default by exp(coefficient x WoE range), with all other variables unchanged.

    Args:
        coefs: Output of :func:`coefficients_table`.
        binning_tables: Long table of bins (``variable``, ``bin``, ``woe``).

    Returns:
        Per variable: ``riskiest_bin``, ``safest_bin``, ``woe_range`` and ``odds_divided_by``
        (how many times lower the odds are in the safest bin).
    """
    rows = []
    for _, row in coefs[coefs["variable"] != CONSTANT].iterrows():
        bins = binning_tables[binning_tables["variable"] == row["variable"]]
        riskiest, safest = bins.loc[bins["woe"].idxmin()], bins.loc[bins["woe"].idxmax()]
        woe_range = safest["woe"] - riskiest["woe"]
        rows.append({
            "variable": row["variable"],
            "riskiest_bin": riskiest["bin"],
            "safest_bin": safest["bin"],
            "woe_range": woe_range,
            "odds_divided_by": float(np.exp(-row["coefficient"] * woe_range)),
        })
    return pd.DataFrame(rows)


def save_model(result: BinaryResultsWrapper, path: Path) -> None:
    """Save the fitted model (without the training data) to a pickle file.

    Args:
        result: Fitted model.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(path), remove_data=True)
    logger.info("Saved model to %s", path)


def load_model(path: Path) -> BinaryResultsWrapper:
    """Load a model saved with :func:`save_model`.

    Only load files this project created: pickle files can run code when loaded.

    Args:
        path: File written by :func:`save_model`.

    Returns:
        The fitted model.
    """
    return sm.load(str(path))
