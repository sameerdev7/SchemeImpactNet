"""
extract.py
----------
Loads and validates the unified MNREGA CSV.
Supports both the synthetic unified dataset and any real CSV
that matches the schema.
"""

import pandas as pd

REQUIRED_COLUMNS = {
    "state", "district", "financial_year",
    "person_days_lakhs", "expenditure_lakhs", "avg_wage_rate"
}

STAGE1_COLUMNS = REQUIRED_COLUMNS
STAGE2_COLUMNS = STAGE1_COLUMNS | {"rainfall_mm", "crop_season_index", "rural_population_lakhs", "poverty_rate_pct"}
STAGE3_COLUMNS = STAGE2_COLUMNS | {"pmkisan_beneficiaries", "pmay_houses_sanctioned", "budget_allocated_lakhs"}


def load_csv(filepath: str, state_filter: str = None) -> pd.DataFrame:
    """
    Load unified MNREGA CSV.

    Args:
        filepath     : Path to CSV file.
        state_filter : If provided, filter to a single state e.g. "Maharashtra".
                       Pass None for all-India (Stage 2+).

    Returns:
        Raw DataFrame.
    """
    print(f"[extract] Loading: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"[extract] File not found: {filepath}")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    _validate_columns(df)

    if state_filter:
        before = len(df)
        df = df[df["state"] == state_filter].reset_index(drop=True)
        print(f"[extract] Filtered to '{state_filter}': {before} → {len(df)} rows")

    print(f"[extract] Loaded {len(df)} rows | {df['state'].nunique()} state(s) | {df['district'].nunique()} districts | {df['financial_year'].nunique()} years")
    print(f"[extract] Validation passed ✓")
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    actual = set(df.columns)
    missing = REQUIRED_COLUMNS - actual
    if missing:
        raise ValueError(f"[extract] Missing required columns: {missing}")
