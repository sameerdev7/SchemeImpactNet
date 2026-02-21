"""
clean.py
--------
Cleans and standardizes the unified MNREGA dataset.
Works for Stage 1 (Maharashtra) through Stage 3 (All-India + scheme data).
"""

import pandas as pd
import numpy as np

CRITICAL_COLS = ["person_days_lakhs", "expenditure_lakhs", "avg_wage_rate"]

NON_CRITICAL_COLS = [
    "households_demanded", "households_offered", "households_availed",
    "works_completed", "rainfall_mm", "crop_season_index",
    "rural_population_lakhs", "poverty_rate_pct",
    "pmkisan_beneficiaries", "pmkisan_amount_lakhs",
    "pmay_houses_sanctioned", "pmay_houses_completed",
    "pmay_expenditure_lakhs", "budget_allocated_lakhs"
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[clean] Starting cleaning pipeline...")
    df = _strip_strings(df)
    df = _parse_financial_year(df)
    df = _cast_numerics(df)
    df = _handle_missing(df)
    df = _enforce_logical_constraints(df)
    print(f"[clean] Done. Shape: {df.shape}")
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
    all_numeric = CRITICAL_COLS + NON_CRITICAL_COLS
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Critical cols   → forward-fill within district, drop if still null.
    Non-critical    → forward-fill within district, leave remaining NaN.
    """
    df = df.sort_values(["state", "district", "financial_year"])

    for col in CRITICAL_COLS + NON_CRITICAL_COLS:
        if col not in df.columns:
            continue
        before = df[col].isna().sum()
        if before > 0:
            df[col] = df.groupby(["state", "district"])[col].transform(lambda s: s.ffill())
            filled = before - df[col].isna().sum()
            if filled > 0:
                print(f"[clean] '{col}': forward-filled {filled} value(s)")

    before = len(df)
    df = df.dropna(subset=CRITICAL_COLS).reset_index(drop=True)
    if len(df) < before:
        print(f"[clean] Dropped {before - len(df)} rows with unresolvable critical nulls")

    return df


def _enforce_logical_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Clip any constraint violations that slipped through generation."""
    if all(c in df.columns for c in ["households_offered", "households_demanded"]):
        violations = (df["households_offered"] > df["households_demanded"]).sum()
        if violations:
            df["households_offered"] = df[["households_offered", "households_demanded"]].min(axis=1)
            print(f"[clean] Fixed {violations} households_offered > households_demanded")

    if all(c in df.columns for c in ["households_availed", "households_offered"]):
        violations = (df["households_availed"] > df["households_offered"]).sum()
        if violations:
            df["households_availed"] = df[["households_availed", "households_offered"]].min(axis=1)
            print(f"[clean] Fixed {violations} households_availed > households_offered")

    return df
