"""
clean.py
--------
Cleans and standardizes MNREGA data.

Handles the realistic schema:
    state, district, financial_year,
    households_demanded, households_offered, households_availed,
    person_days_lakhs, expenditure_lakhs, avg_wage_rate, works_completed
"""

import pandas as pd
import numpy as np

NUMERIC_COLS = [
    "households_demanded", "households_offered", "households_availed",
    "person_days_lakhs", "expenditure_lakhs", "avg_wage_rate", "works_completed"
]

REQUIRED_COLS = {
    "state", "district", "financial_year",
    "person_days_lakhs", "expenditure_lakhs"
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[clean] Starting cleaning pipeline...")
    df = _standardize_columns(df)
    df = _strip_strings(df)
    df = _parse_financial_year(df)
    df = _cast_numerics(df)
    df = _handle_missing(df)
    print(f"[clean] Done. Shape: {df.shape}")
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[clean] Missing required columns: {missing}")
    print(f"[clean] Columns: {list(df.columns)}")
    return df


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def _parse_financial_year(df: pd.DataFrame) -> pd.DataFrame:
    """Convert '2018-19' → integer 2018."""
    def _parse(val):
        val = str(val).strip()
        return int(val.split("-")[0]) if "-" in val else int(val)

    df["financial_year"] = df["financial_year"].apply(_parse)
    print(f"[clean] financial_year range: {df['financial_year'].min()} – {df['financial_year'].max()}")
    return df


def _cast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
    - Critical cols (person_days_lakhs, expenditure_lakhs): forward-fill
      within district, drop if still null.
    - Non-critical cols (households_*, works_completed): forward-fill,
      leave remaining NaN — not dropped, model handles them.
    """
    df = df.sort_values(["district", "financial_year"])

    critical = ["person_days_lakhs", "expenditure_lakhs"]
    non_critical = ["households_demanded", "households_offered",
                    "households_availed", "works_completed"]

    for col in critical + non_critical:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df.groupby("district")[col].transform(lambda s: s.ffill())
            filled = before - df[col].isna().sum()
            if filled > 0:
                print(f"[clean] '{col}': forward-filled {filled} value(s)")

    before = len(df)
    df = df.dropna(subset=critical).reset_index(drop=True)
    if len(df) < before:
        print(f"[clean] Dropped {before - len(df)} rows with unresolvable critical nulls")

    return df
