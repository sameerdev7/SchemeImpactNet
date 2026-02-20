"""
features.py
-----------
Feature engineering for MNREGA district-level performance forecasting.

Features created:
    lag_person_days         : person_days_lakhs from previous year (per district)
    lag_expenditure         : expenditure_lakhs from previous year
    expenditure_per_personday: cost efficiency metric (₹ lakhs per lakh persondays)
    demand_fulfillment_rate : households_availed / households_demanded
    yoy_growth              : year-on-year % change in person_days per district
    district_avg_persondays : rolling historical mean per district (proxy for base capacity)
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[features] Building features...")
    df = df.sort_values(["district", "financial_year"]).reset_index(drop=True)

    df = _lag_features(df)
    df = _efficiency_features(df)
    df = _demand_fulfillment(df)
    df = _yoy_growth(df)
    df = _district_avg(df)

    # Drop rows missing lag features — can't train without them
    before = len(df)
    df = df.dropna(subset=["lag_person_days", "lag_expenditure"]).reset_index(drop=True)
    print(f"[features] Dropped {before - len(df)} first-year rows (no lag available)")
    print(f"[features] Done. Final shape: {df.shape}")
    return df


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lag_person_days"]  = df.groupby("district")["person_days_lakhs"].shift(1)
    df["lag_expenditure"]  = df.groupby("district")["expenditure_lakhs"].shift(1)
    return df


def _efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cost per lakh persondays — lower = more efficient delivery."""
    safe_pd = df["person_days_lakhs"].replace(0, np.nan)
    df["expenditure_per_personday"] = (df["expenditure_lakhs"] / safe_pd).round(3)
    return df


def _demand_fulfillment(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of households who got work vs those who asked for it."""
    if "households_demanded" in df.columns and "households_availed" in df.columns:
        safe_demand = df["households_demanded"].replace(0, np.nan)
        df["demand_fulfillment_rate"] = (
            df["households_availed"] / safe_demand
        ).clip(0, 1).round(4)
    return df


def _yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year % change in person_days per district."""
    lag = df.groupby("district")["person_days_lakhs"].shift(1)
    df["yoy_growth"] = ((df["person_days_lakhs"] - lag) / lag).round(4)
    return df


def _district_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding mean of person_days per district — captures base capacity."""
    df["district_avg_persondays"] = (
        df.groupby("district")["person_days_lakhs"]
        .transform(lambda s: s.expanding().mean().shift(1))
        .round(3)
    )
    return df
