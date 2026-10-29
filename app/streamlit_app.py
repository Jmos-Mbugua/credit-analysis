"""Streamlit app: apply the frozen credit scorecard to one customer or a batch.

Nothing is trained here. The app loads the scorecard exported to ``models/`` and the
performance tables in ``reports/``.

Run locally from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # import the package without `pip install -e .`

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from scorecard.scorer import (  # noqa: E402
    REQUIRED_FIELDS,
    STATUS_MEANINGS,
    ScorecardArtifacts,
    load_artifacts,
    score_customer,
    score_frame,
    validate_customers,
)

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SAMPLE_PATH = ROOT / "data" / "sample" / "sample_customers.csv"

PAGES = ["Score a customer", "Score a batch", "Model performance"]
MONTHS = ["September", "August", "July", "June", "May", "April"]  # suffix 1..6
DECISION_STYLE = {  # status colours with an icon and a label, so colour is never the only signal
    "Approve": ("#0ca30c", "#ffffff", "✔"),
    "Refer": ("#fab219", "#1a1a19", "⚠"),
    "Decline": ("#d03b3b", "#ffffff", "✖"),
}
DATA_CREDIT = (
    "Data: Yeh, I. C. & Lien, C. H. (2009). *Default of Credit Card Clients* [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. Licensed under "
    "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)."
)
PERCENT = "{:.1%}"


@st.cache_resource
def get_artifacts() -> ScorecardArtifacts:
    """Load the frozen scorecard once per app process."""
    return load_artifacts(MODELS_DIR)


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV shipped with the app (cached)."""
    return pd.read_csv(path)


def decision_badge(decision: str) -> str:
    """HTML badge for a decision."""
    background, text, icon = DECISION_STYLE[decision]
    return (f"<span style='background:{background};color:{text};padding:0.35rem 0.9rem;"
            f"border-radius:0.5rem;font-weight:600;font-size:1.25rem'>{icon} {decision}</span>")


def sidebar() -> str:
    """Navigation and the 'about' text. Returns the selected page."""
    with st.sidebar:
        page = st.radio("Page", PAGES, key="page")
        st.divider()
        st.markdown(
            "**What this is.** A behavioural credit scorecard built the way banks build them "
            "(binning, Weight of Evidence, logistic regression, points). It predicts whether an existing "
            "credit card customer will miss next month's payment."
        )
        st.markdown(
            "**Data.** The public UCI *Default of Credit Card Clients* dataset: 30,000 customers of a "
            "Taiwanese bank, April-September 2005. Sex, marriage, age and education are not used."
        )
        st.markdown(
            "**Illustrative only.** The decision policy (decline below 520, refer 520-549, approve from 550) "
            "rests on assumed margin and loss figures. It is a portfolio demonstration, not a lending tool."
        )
        st.caption(DATA_CREDIT)
    return page


# --- Score a customer ---------------------------------------------------------------

def manual_inputs(defaults: pd.Series) -> dict:
    """Widgets for the raw fields, pre-filled with ``defaults``."""
    statuses = list(STATUS_MEANINGS)
    fields = {"limit_bal": st.number_input("Credit limit (NT$)", min_value=1, step=10_000,
                                           value=int(defaults["limit_bal"]))}
    st.markdown("**Repayment status by month**")
    columns = st.columns(3)
    for i, month in enumerate(MONTHS, start=1):
        with columns[(i - 1) % 3]:
            fields[f"pay_{i}"] = st.selectbox(
                month, statuses, index=statuses.index(int(defaults[f"pay_{i}"])),
                format_func=lambda s: f"{s}: {STATUS_MEANINGS[s]}", key=f"pay_{i}")
    st.markdown("**Amount paid by month (NT$)**")
    columns = st.columns(3)
    for i, month in enumerate(MONTHS, start=1):
        with columns[(i - 1) % 3]:
            fields[f"pay_amt{i}"] = st.number_input(month, min_value=0, step=1_000,
                                                    value=int(defaults[f"pay_amt{i}"]), key=f"pay_amt{i}")
    return fields


def page_score_customer(artifacts: ScorecardArtifacts) -> None:
    """Score one sample or hand-entered customer."""
    st.header("Score a customer")
    sample = read_csv(SAMPLE_PATH)
    mode = st.radio("Customer", ["Pick a sample customer", "Enter values manually"], horizontal=True,
                    key="customer_mode")
    if mode == "Pick a sample customer":
        customer_id = st.selectbox("Sample customer (real test-set customers)", sample["id"],
                                   format_func=lambda i: f"Customer {i}")
        row = sample.loc[sample["id"] == customer_id].iloc[0]
        fields = {f: row[f] for f in REQUIRED_FIELDS}
        with st.expander("Raw fields for this customer"):
            st.dataframe(pd.DataFrame([fields]), hide_index=True)
        outcome = "defaulted" if row["default"] == 1 else "did not default"
        st.caption(f"What actually happened: this customer **{outcome}** in October 2005.")
    else:
        fields = manual_inputs(sample.iloc[0])

    try:
        result = score_customer(fields, artifacts)
    except ValueError as error:
        st.error(str(error))
        return

    st.divider()
    left, middle, right = st.columns(3)
    left.metric("Score", result.score, help="Higher = safer. 600 means odds of 50 goods per bad; "
                                            "every 20 points doubles the odds.")
    middle.metric("Probability of default", PERCENT.format(result.pd),
                  help="Chance of missing next month's payment, implied by the score.")
    with right:
        st.markdown("Decision")
        st.markdown(decision_badge(result.decision), unsafe_allow_html=True)

    st.subheader("Why: top reasons")
    if result.reasons:
        st.markdown("\n".join(f"{k}. {reason}" for k, reason in enumerate(result.reasons, start=1)))
    else:
        st.markdown("The customer is in the best band on every variable.")

    st.subheader("Points by variable")
    st.dataframe(
        result.points.drop(columns="variable"),
        hide_index=True,
        column_config={
            "description": "Variable", "value": "Customer value", "band": "Band",
            "points": "Points", "best_points": "Best possible", "points_lost": "Points lost",
        },
    )
    st.caption(f"Total = {result.score} points. Policy: decline below {artifacts.policy['decline_below']}, "
               f"refer {artifacts.policy['decline_below']}-{artifacts.policy['approve_from'] - 1}, "
               f"approve from {artifacts.policy['approve_from']}.")


# --- Score a batch ------------------------------------------------------------------

def page_score_batch(artifacts: ScorecardArtifacts) -> None:
    """Score an uploaded CSV of customers."""
    st.header("Score a batch")
    st.markdown(
        "Upload a CSV with one row per customer and these columns: "
        + ", ".join(f"`{c}`" for c in REQUIRED_FIELDS)
        + ". An optional `id` column is carried through to the results. Other columns are ignored."
    )
    sample = read_csv(SAMPLE_PATH)
    template = sample[["id", *REQUIRED_FIELDS]].head(3)
    left, right = st.columns(2)
    left.download_button("Download template (3 example rows)", template.to_csv(index=False),
                         "scorecard_template.csv", "text/csv")
    right.download_button("Download the 200 sample customers", sample[["id", *REQUIRED_FIELDS]].to_csv(index=False),
                          "sample_customers.csv", "text/csv")

    upload = st.file_uploader("Customer file (CSV)", type="csv")
    if upload is None:
        return
    try:
        customers = pd.read_csv(upload)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        st.error(f"Couldn't read the file as CSV: {error}")
        return
    errors = validate_customers(customers)
    if errors:
        st.error("The file can't be scored yet:\n\n" + "\n".join(f"- {e}" for e in errors))
        return

    scored = score_frame(customers, artifacts)
    if "id" in customers.columns:
        scored.insert(0, "id", customers["id"])
    counts = scored["decision"].value_counts()
    columns = st.columns(4)
    columns[0].metric("Customers", f"{len(scored):,}")
    for column, decision in zip(columns[1:], ["Approve", "Refer", "Decline"]):
        column.metric(decision, f"{counts.get(decision, 0):,}")
    st.dataframe(scored, hide_index=True, column_config={"pd": st.column_config.NumberColumn("PD", format="%.3f")})
    st.download_button("Download results (CSV)", scored.to_csv(index=False), "scored_customers.csv", "text/csv")


# --- Model performance --------------------------------------------------------------

def page_performance() -> None:
    """Test-set performance, score bands, strategy and policy from reports/."""
    st.header("Model performance")
    performance = read_csv(REPORTS_DIR / "performance.csv").set_index("sample")
    test, train = performance.loc["Test"], performance.loc["Train"]
    columns = st.columns(3)
    columns[0].metric("Gini (test)", f"{test['gini']:.3f}", f"{test['gini'] - train['gini']:+.3f} vs train")
    columns[1].metric("KS (test)", f"{test['ks']:.3f}", f"at score {test['ks_score']:.0f}", delta_color="off")
    columns[2].metric("AUC (test)", f"{test['auc']:.3f}", f"{test['auc'] - train['auc']:+.3f} vs train")
    st.caption(
        "Measured on 9,000 customers the model never saw. Gini: 0 = random, 1 = perfect ranking. "
        "KS: the largest gap between the share of bads and the share of goods below a score. "
        f"Test Gini 95% range: {test['gini_95_low']:.3f}-{test['gini_95_high']:.3f}. The 3.5-point drop from train "
        "is in the 'watch' zone, not overfitting."
    )

    st.subheader("Score bands (test set)")
    bands = read_csv(REPORTS_DIR / "score_bands.csv")
    st.dataframe(bands.style.format({"bad_rate": PERCENT, "share": PERCENT, "cumulative_share": PERCENT}),
                 hide_index=True)

    st.subheader("Strategy table: approve if score >= cut-off")
    strategy = read_csv(REPORTS_DIR / "strategy_table.csv")
    rates = ["approval_rate", "approved_bad_rate", "declined_bad_rate", "bads_declined_share", "goods_declined_share"]
    st.dataframe(strategy.style.format({**{c: PERCENT for c in rates}, "cutoff": "{:.0f}"}, na_rep="n/a"),
                 hide_index=True)

    st.subheader("Proposed three-zone policy (illustrative)")
    policy = read_csv(REPORTS_DIR / "cutoff_policy.csv")
    profit_columns = [c for c in policy.columns if c.startswith("profit_")]
    st.dataframe(policy.style.format({"share": PERCENT, "bad_rate": PERCENT, "share_of_all_bads": PERCENT,
                                      **{c: "{:+.1f}" for c in profit_columns}}), hide_index=True)
    st.caption(
        "Profit in NT$ millions on the test set, assuming a yearly margin of 8% of balance on good customers and "
        "a loss of 40%, 60% or 80% of balance on bad ones. The refer zone is where profit depends on that loss "
        "assumption. The margin is yearly but the default flag covers one month, so the thresholds are illustrative."
    )


def main() -> None:
    """Lay out the app."""
    st.set_page_config(page_title="Credit Risk Scorecard", page_icon="💳", layout="wide")
    page = sidebar()
    st.title("Credit Risk Scorecard")
    artifacts = get_artifacts()
    if page == PAGES[0]:
        page_score_customer(artifacts)
    elif page == PAGES[1]:
        page_score_batch(artifacts)
    else:
        page_performance()


main()
