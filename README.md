# credit-risk-scorecard

A behavioural credit scorecard built the way banks build them. Using six months of
repayment history for 30,000 credit card customers, the project predicts which customers
will default next month and turns that prediction into a points-based score. The steps are
binning, Weight of Evidence and Information Value, logistic regression, points scaling,
validation with Gini and KS, a cut-off strategy, and Population Stability Index (PSI) monitoring.

## Data

Yeh, I. C. & Lien, C. H. (2009). The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients. *Expert Systems
with Applications*, 36(2), 2473–2480.

Dataset: Yeh, I. C. (2009). "Default of Credit Card Clients", UCI Machine Learning Repository.
https://doi.org/10.24432/C55S3H. Licensed under
[Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The full dataset is not stored in this repository. Download it with `python scripts/download_data.py`.
The only data committed is `data/sample/sample_customers.csv`: 200 test-set customers with the
13 raw fields the scorecard uses (no sex, marriage, age or education), shared under the same CC BY 4.0
licence so the app has real examples.

## Streamlit app

`app/streamlit_app.py` applies the frozen scorecard. It never retrains anything. It has three pages:
score one customer (a sample customer or typed-in values), score an uploaded CSV, and model
performance. The decision policy shown is illustrative.

Run it locally (WSL, from the project root):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501.

### Deploy on Streamlit Community Cloud

1. Push the repository to GitHub. `models/*.json`, `models/scorecard_points.csv`, `reports/` and
   `data/sample/` must be committed: the app reads them at start-up.
2. On https://share.streamlit.io choose **Create app**, select the repository and branch, and set
   **Main file path** to `app/streamlit_app.py`.
3. Under **Advanced settings**, choose Python 3.12.
4. Deploy. Streamlit Cloud installs `requirements.txt` (numpy, pandas, streamlit only). The app
   imports the `scorecard` package straight from `src/`, so no `pip install -e .` is needed.

## Rebuild the scorecard

```bash
pip install -r requirements-dev.txt
pip install -e .
python scripts/download_data.py
# run notebooks/01_eda.ipynb ... notebooks/07_strategy.ipynb in order
python scripts/export_app_artifacts.py   # freeze the scorecard for the app
pytest
```

## Data notes

Quirks found while exploring the data (`notebooks/01_eda.ipynb`) and how each is handled
by `scorecard.data.clean()`, which writes `data/processed/clean.csv`:

| Quirk | Handling |
|---|---|
| Repayment status columns are named `PAY_0, PAY_2 … PAY_6` (no `PAY_1`) | `PAY_0` renamed `pay_1`, so `pay_1` (Sep 2005) … `pay_6` (Apr 2005) line up with `bill_amt1..6` and `pay_amt1..6` |
| `EDUCATION` has undocumented codes 0, 5, 6 (345 customers) | Grouped into 4 ("other") |
| `MARRIAGE` has undocumented code 0 (54 customers) | Grouped into 3 ("other"); never a model feature anyway |
| Repayment status uses -2 (no use) and 0 (revolving), which the original documentation doesn't describe | Kept as they are; meanings documented in the notebook. They will get their own bins |
| Status 1 ("one month late") appears 3,688 times in `pay_1` but almost never in `pay_2`–`pay_6` | Kept; noted as a recording inconsistency to handle when binning |
| Statuses 5–8 have very few customers (9–76 each in `pay_1`) | Kept; will be merged with neighbouring statuses when binning |
| 1,930 customers (6.4%) have at least one negative bill amount (credit balance) | Kept: legitimate overpayments, not errors |
| Default rate is 22.1%, far above a typical card portfolio | Kept; predicted probabilities apply to this sample, not to a real portfolio |
| Column names are uppercase | All columns lowercased |

No duplicate customer IDs and no missing values were found.
