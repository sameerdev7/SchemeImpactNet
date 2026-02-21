"""
features.py
-----------
Feature engineering for MNREGA district-level forecasting.
Designed to work for Stage 1 through Stage 3 — extra columns
are added automatically when present in the data.

Core features (Stage 1):
    lag_person_days           : person_days_lakhs previous year per district
    lag_expenditure           : expenditure_lakhs previous year
    expenditure_per_personday : cost efficiency metric
    demand_fulfillment_rate   : households_availed / households_demanded
    yoy_growth                : year-on-year % change in person_days
    district_avg_persondays   : expanding historical mean (base capacity proxy)

Stage 2 additions (when columns exist):
    drought_flag              : 1 if rainfall_mm < 25th percentile of that state
    high_poverty_flag         : 1 if poverty_rate_pct > 35%

Stage 3 additions (when columns exist):
    scheme_overlap_score      : combined PMKISAN + PMAY activity index per district
    budget_utilization_rate   : expenditure / budget_allocated
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[features] Building features...")
    df = df.sort_values(["state", "district", "financial_year"]).reset_index(drop=True)

    df = _lag_features(df)
    df = _efficiency_features(df)
    df = _demand_fulfillment(df)
    df = _yoy_growth(df)
    df = _district_avg(df)

    # Stage 2 features — only if columns exist
    if "rainfall_mm" in df.columns:
        df = _drought_flag(df)
    if "poverty_rate_pct" in df.columns:
        df = _poverty_flag(df)

    # Stage 3 features — only if columns exist
    if "pmkisan_beneficiaries" in df.columns and "pmay_houses_completed" in df.columns:
        df = _scheme_overlap(df)
    if "budget_allocated_lakhs" in df.columns:
        df = _budget_utilization(df)

    # Drop rows with no lag — can't train on them
    before = len(df)
    df = df.dropna(subset=["lag_person_days", "lag_expenditure"]).reset_index(drop=True)
    print(f"[features] Dropped {before - len(df)} first-year rows (no lag available)")
    print(f"[features] Done. Final shape: {df.shape}")
    return df


# ── Core features ─────────────────────────────────────────────────────────────

def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["state", "district"])
    df["lag_person_days"] = grp["person_days_lakhs"].shift(1)
    df["lag_expenditure"] = grp["expenditure_lakhs"].shift(1)
    return df


def _efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    safe = df["person_days_lakhs"].replace(0, np.nan)
    df["expenditure_per_personday"] = (df["expenditure_lakhs"] / safe).round(3)
    return df


def _demand_fulfillment(df: pd.DataFrame) -> pd.DataFrame:
    if "households_demanded" in df.columns and "households_availed" in df.columns:
        safe = df["households_demanded"].replace(0, np.nan)
        df["demand_fulfillment_rate"] = (df["households_availed"] / safe).clip(0, 1).round(4)
    return df


def _yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    lag = df.groupby(["state", "district"])["person_days_lakhs"].shift(1)
    df["yoy_growth"] = ((df["person_days_lakhs"] - lag) / lag).round(4)
    return df


def _district_avg(df: pd.DataFrame) -> pd.DataFrame:
    df["district_avg_persondays"] = (
        df.groupby(["state", "district"])["person_days_lakhs"]
        .transform(lambda s: s.expanding().mean().shift(1))
        .round(3)
    )
    return df


# ── Stage 2 features ──────────────────────────────────────────────────────────

def _drought_flag(df: pd.DataFrame) -> pd.DataFrame:
    """1 if district rainfall is below the 25th percentile of its state that year."""
    threshold = df.groupby(["state", "financial_year"])["rainfall_mm"].transform(lambda s: s.quantile(0.25))
    df["drought_flag"] = (df["rainfall_mm"] < threshold).astype(int)
    print(f"[features] drought_flag: {df['drought_flag'].sum()} drought district-years flagged")
    return df


def _poverty_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["high_poverty_flag"] = (df["poverty_rate_pct"] > 35).astype(int)
    print(f"[features] high_poverty_flag: {df['high_poverty_flag'].sum()} high-poverty district-years")
    return df


# ── Stage 3 features ──────────────────────────────────────────────────────────

def _scheme_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalized index (0–1) combining PMKISAN and PMAY activity.
    High score = district has strong presence of both schemes.
    """
    pk = df["pmkisan_beneficiaries"].fillna(0)
    pmay = df["pmay_houses_completed"].fillna(0)

    pk_norm   = (pk   - pk.min())   / (pk.max()   - pk.min()   + 1e-9)
    pmay_norm = (pmay - pmay.min()) / (pmay.max() - pmay.min() + 1e-9)

    df["scheme_overlap_score"] = ((pk_norm + pmay_norm) / 2).round(4)
    print("[features] scheme_overlap_score created")
    return df


def _budget_utilization(df: pd.DataFrame) -> pd.DataFrame:
    safe = df["budget_allocated_lakhs"].replace(0, np.nan)
    df["budget_utilization_rate"] = (df["expenditure_lakhs"] / safe).clip(0, 1).round(4)
    print("[features] budget_utilization_rate created")
    return df
