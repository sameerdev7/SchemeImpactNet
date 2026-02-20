"""
extract.py
----------
Loads and validates raw MNREGA CSV data.
"""

import pandas as pd

REQUIRED_COLUMNS = {"state", "district", "financial_year", "person_days_lakhs", "expenditure_lakhs"}


def load_csv(filepath: str) -> pd.DataFrame:
    print(f"[extract] Loading: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"[extract] File not found: {filepath}")

    print(f"[extract] Loaded {len(df)} rows × {len(df.columns)} columns")
    _validate_columns(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    actual = {c.strip().lower() for c in df.columns}
    missing = REQUIRED_COLUMNS - actual
    if missing:
        raise ValueError(f"[extract] Missing required columns: {missing}")
    print(f"[extract] Validation passed ✓")
