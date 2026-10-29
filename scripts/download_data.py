"""Download the UCI credit card default dataset to data/raw/credit_card_default.csv.

Usage (from the project root, with the virtual environment active):
    python scripts/download_data.py
"""

import logging

from scorecard.data import RAW_PATH, download_dataset


def main() -> None:
    """Configure logging and run the download."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    download_dataset(RAW_PATH)


if __name__ == "__main__":
    main()
