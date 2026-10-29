"""Download, standardise and validate the UCI "Default of Credit Card Clients" dataset."""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_default.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

UCI_DATASET_ID = 350
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
)
EXPECTED_ROWS = 30_000
TARGET = "default"
ID_COLUMN = "ID"

# ucimlrepo labels the columns X1..X23 and Y; these are the original names from the paper.
# Note the original data has PAY_0 and then PAY_2 (there is no PAY_1).
FEATURE_NAMES: list[str] = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
RENAME_MAP: dict[str, str] = {
    **{f"X{i}": name for i, name in enumerate(FEATURE_NAMES, start=1)},
    "Y": TARGET,
    "default payment next month": TARGET,
}


def fetch_from_ucimlrepo(dataset_id: int = UCI_DATASET_ID) -> pd.DataFrame:
    """Download the dataset with the ucimlrepo package.

    Args:
        dataset_id: UCI repository id of the dataset.

    Returns:
        The raw table including id, feature and target columns.
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=dataset_id)
    return dataset.data.original


def fetch_from_uci_website(url: str = UCI_ZIP_URL, timeout: int = 60) -> pd.DataFrame:
    """Download the zipped .xls file directly from the UCI website.

    Args:
        url: Address of the zip archive.
        timeout: Seconds to wait for the server before giving up.

    Returns:
        The raw table read from the Excel file inside the archive.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    xls_name = next(name for name in archive.namelist() if name.endswith(".xls"))
    # The sheet has two header rows (X1..X23, then the real names); header=1 keeps the real ones.
    return pd.read_excel(archive.open(xls_name), header=1, engine="xlrd")


def fetch_raw_data() -> pd.DataFrame:
    """Download the dataset, falling back to the UCI website if ucimlrepo fails.

    Returns:
        The raw, not yet standardised table.
    """
    try:
        logger.info("Downloading UCI dataset %d with ucimlrepo", UCI_DATASET_ID)
        return fetch_from_ucimlrepo()
    except Exception as exc:  # any failure (network, API change) triggers the fallback
        logger.warning("ucimlrepo failed (%s); falling back to %s", exc, UCI_ZIP_URL)
        return fetch_from_uci_website()


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give the columns their original names and a fixed order: ID, features, default.

    Works for both download routes: ucimlrepo (X1..X23, Y) and the Excel file
    (LIMIT_BAL..., "default payment next month").

    Args:
        df: Raw table from either download route.

    Returns:
        A copy with columns ``ID``, the 23 features and ``default``, all integers.
    """
    out = df.rename(columns=RENAME_MAP)
    if ID_COLUMN not in out.columns:
        logger.warning("No %s column found; numbering customers 1..n", ID_COLUMN)
        out.insert(0, ID_COLUMN, range(1, len(out) + 1))
    missing = [c for c in [ID_COLUMN, *FEATURE_NAMES, TARGET] if c not in out.columns]
    if missing:
        raise ValueError(f"Columns missing after renaming: {missing}")
    return out[[ID_COLUMN, *FEATURE_NAMES, TARGET]].astype("int64")


def validate(df: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> None:
    """Check the table is what we expect before saving it.

    Args:
        df: Standardised table.
        expected_rows: Number of customers the dataset should contain.

    Raises:
        ValueError: If the row count, target values, IDs or missing values are wrong.
    """
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"Target '{TARGET}' must only contain 0 and 1")
    check_integrity(df, id_column=ID_COLUMN)


def check_integrity(df: pd.DataFrame, id_column: str) -> None:
    """Check that every customer appears once and no value is missing.

    Args:
        df: Table with one row per customer.
        id_column: Name of the customer identifier column.

    Raises:
        ValueError: If any ID is duplicated or any value is missing.
    """
    n_duplicates = int(df[id_column].duplicated().sum())
    n_missing = int(df.isna().sum().sum())
    if n_duplicates:
        raise ValueError(f"Column '{id_column}' contains {n_duplicates} duplicate customers")
    if n_missing:
        raise ValueError(f"Dataset contains {n_missing} missing values")


def default_rate(df: pd.DataFrame) -> float:
    """Share of customers who defaulted (the "bad rate").

    Args:
        df: Table with a 0/1 ``default`` column.

    Returns:
        Number of defaulters divided by number of customers.
    """
    return float(df[TARGET].mean())


def download_dataset(output_path: Path) -> pd.DataFrame:
    """Download, standardise, validate and save the dataset as CSV.

    Args:
        output_path: Where to write the CSV file.

    Returns:
        The saved table.
    """
    df = standardise_columns(fetch_raw_data())
    validate(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %s", output_path)
    logger.info("Rows: %d | Columns: %d", *df.shape)
    logger.info("Default rate: %.2f%%", 100 * default_rate(df))
    return df


# Undocumented codes and the documented "other" category they are folded into.
EDUCATION_OTHER = 4  # documented: 1 graduate school, 2 university, 3 high school, 4 other
EDUCATION_UNDOCUMENTED = [0, 5, 6]
MARRIAGE_OTHER = 3  # documented: 1 married, 2 single, 3 other
MARRIAGE_UNDOCUMENTED = [0]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the downloaded dataset.

    Args:
        path: Location of the raw CSV written by ``scripts/download_data.py``.

    Returns:
        The raw table with the original column names.
    """
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return df


def _recode(series: pd.Series, codes: list[int], target: int) -> pd.Series:
    """Replace each of ``codes`` with ``target`` and log how many values changed."""
    mask = series.isin(codes)
    logger.info("%s: recoded %d values %s -> %d", series.name, int(mask.sum()), codes, target)
    return series.mask(mask, target)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known quirks of the raw data.

    - ``PAY_0`` becomes ``PAY_1`` so the repayment columns run PAY_1 (September 2005,
      most recent) to PAY_6 (April 2005), matching BILL_AMT1..6 and PAY_AMT1..6.
    - Undocumented EDUCATION codes 0, 5, 6 are grouped into 4 ("other").
    - Undocumented MARRIAGE code 0 is grouped into 3 ("other").
    - Column names become lowercase.

    Args:
        df: Raw table as returned by :func:`load_raw`.

    Returns:
        A cleaned copy; the input is not modified.

    Raises:
        ValueError: If any customer ID is duplicated or any value is missing.
    """
    out = df.rename(columns={"PAY_0": "PAY_1"})
    out["EDUCATION"] = _recode(out["EDUCATION"], EDUCATION_UNDOCUMENTED, EDUCATION_OTHER)
    out["MARRIAGE"] = _recode(out["MARRIAGE"], MARRIAGE_UNDOCUMENTED, MARRIAGE_OTHER)
    out.columns = out.columns.str.lower()
    check_integrity(out, id_column=ID_COLUMN.lower())
    logger.info("Integrity check passed: %d unique customers, no missing values", len(out))
    return out


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Save the cleaned table as CSV.

    Args:
        df: Cleaned table from :func:`clean`.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split customers into a training and a test set with the same default rate.

    All binning, WoE and selection decisions are learned on the training set only;
    the test set is kept untouched for validation, so its results are honest.

    Args:
        df: Cleaned table with a 0/1 ``default`` column.
        test_size: Fraction of customers put in the test set.
        seed: Random seed, so the split is the same every run.

    Returns:
        ``(train, test)``.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    logger.info("Train: %d customers, default rate %.2f%% | Test: %d customers, default rate %.2f%%",
                len(train), 100 * default_rate(train), len(test), 100 * default_rate(test))
    return train, test
