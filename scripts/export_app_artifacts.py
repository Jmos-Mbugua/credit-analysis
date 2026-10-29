"""Freeze the fitted scorecard for the app: models/ artifacts, a sample of customers and a test fixture.

Run after notebooks 02-07 (it reads models/binnings.pkl and reports/scorecard.csv):
    python scripts/export_app_artifacts.py

Writes:
- models/bins.json, models/scorecard_points.csv, models/scaling.json, models/policy.json
- data/sample/sample_customers.csv: 200 test-set customers, raw scorecard fields only
- tests/fixtures/notebook_scores.csv: those customers' scores from the notebook pipeline,
  used by the tests to prove the app's scorer gives identical results
"""

import logging

import pandas as pd

from scorecard.binning import load_binnings
from scorecard.data import MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from scorecard.scaling import BASE_ODDS, BASE_SCORE, PDO, scaling_constants, score_customers
from scorecard.scorer import REQUIRED_FIELDS, export_artifacts
from scorecard.strategy import APPROVE_FROM, DECLINE_BELOW

SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "sample_customers.csv"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "notebook_scores.csv"
SAMPLE_SIZE = 200
SEED = 42

logger = logging.getLogger(__name__)


def main() -> None:
    """Export the artifacts, the sample and the regression fixture."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    binnings = load_binnings(MODELS_DIR / "binnings.pkl")
    scorecard = pd.read_csv(REPORTS_DIR / "scorecard.csv")
    factor, offset = scaling_constants(BASE_SCORE, BASE_ODDS, PDO)
    export_artifacts(
        binnings,
        scorecard,
        scaling={"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO, "factor": factor, "offset": offset},
        policy={"decline_below": DECLINE_BELOW, "approve_from": APPROVE_FROM},
    )

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    test_pd = pd.read_csv(PROCESSED_DIR / "test_pd.csv")
    sample = test.sample(n=SAMPLE_SIZE, random_state=SEED).sort_values("id")
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample[["id", *REQUIRED_FIELDS, "default"]].to_csv(SAMPLE_PATH, index=False)
    logger.info("Saved %d sample customers to %s", len(sample), SAMPLE_PATH)

    fixture = pd.DataFrame({"id": sample["id"], "score": score_customers(sample, scorecard, binnings)})
    fixture = fixture.merge(test_pd[["id", "pd"]].rename(columns={"pd": "model_pd"}), on="id")
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(FIXTURE_PATH, index=False)
    logger.info("Saved notebook scores for %d customers to %s", len(fixture), FIXTURE_PATH)


if __name__ == "__main__":
    main()
