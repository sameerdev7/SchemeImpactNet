"""
pipeline.py
-----------
Pipeline orchestrator for SchemeImpactNet.

Data sources:
    Real:      data/raw/20063- Dataful/mnrega-...-persondays-...csv
               -> person_days_lakhs, households_availed (real gov data)
               -> avg_wage_rate (official wage schedule, exogenous)
"""

import os
import pandas as pd
import numpy as np

from src.clean    import clean
from src.features import build_features
from src.eda      import run_eda
from src.model    import run_model

DATAFUL_PATH = os.path.join(
    "data", "raw", "20063- Dataful",
    "mnrega-year-month-state-and-district-wise-total-persondays-"
    "and-households-engaged-in-work.csv"
)
UNIFIED_PATH   = os.path.join("data", "raw", "mnrega_india_unified.csv")
PROCESSED_PATH = os.path.join("data", "processed", "mnrega_cleaned.csv")
MODEL_PATH     = os.path.join("models", "mnrega_best_model.pkl")


def run_pipeline() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  SchemeImpactNet — Pipeline")
    print("  Scope: All-India")
    print("=" * 60)

    print(f"\n[pipeline] Step 1: Extract")
    df = _load_real_data()

    print(f"\n[pipeline] Step 2: Clean")
    df = _clean_real(df)

    print(f"\n[pipeline] Step 3: Feature Engineering")
    df = build_features(df)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n[pipeline] Processed data saved -> {PROCESSED_PATH}")

    print(f"\n[pipeline] Step 4: EDA")
    run_eda(df, scope="All-India")

    print(f"\n[pipeline] Step 5: Model")
    predictions = run_model(df)

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print(f"  Processed   : {PROCESSED_PATH}")
    print(f"  Model       : {MODEL_PATH}")
    print(f"  Figures     : reports/figures/")
    print(f"  Predictions : data/processed/mnrega_predictions.csv")
    print(f"  Report      : reports/model_report.txt")
    print("=" * 60 + "\n")

    return predictions


def _load_real_data(state_filter: str = None) -> pd.DataFrame:
    if os.path.exists(DATAFUL_PATH):
        print(f"[pipeline] Loading Dataful government CSV")
        df_raw = pd.read_csv(DATAFUL_PATH)
        df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]

        df_raw["fy"] = df_raw["fiscal_year"].apply(
            lambda v: int(str(v).split("-")[0]) if "-" in str(v) else int(v)
        )
        df_raw = df_raw[df_raw["fy"] <= 2024]

        pivot = df_raw.pivot_table(
            index=["fiscal_year", "fy", "state", "district"],
            columns="category",
            values="value",
            aggfunc="sum"
        ).reset_index()
        pivot.columns.name = None

        pivot = pivot.rename(columns={
            "Persondays": "person_days",
            "Household":  "households_availed",
            "fy":         "financial_year",
        })
        pivot["person_days_lakhs"] = (pivot["person_days"] / 1e5).round(3)

        if os.path.exists(UNIFIED_PATH):
            df_uni = pd.read_csv(UNIFIED_PATH)
            df_uni.columns = [c.strip().lower().replace(" ", "_") for c in df_uni.columns]
            df_uni["financial_year"] = df_uni["financial_year"].apply(
                lambda v: int(str(v).split("-")[0]) if "-" in str(v) else int(v)
            )
            wage_map = df_uni[["state", "financial_year", "avg_wage_rate"]].drop_duplicates()
            pivot = pivot.merge(wage_map, on=["state", "financial_year"], how="left")

        keep = ["state", "district", "financial_year",
                "person_days_lakhs", "households_availed", "avg_wage_rate"]
        df = pivot[[c for c in keep if c in pivot.columns]].copy()

    else:
        print(f"[pipeline] Dataful CSV not found, falling back to unified CSV")
        df = pd.read_csv(UNIFIED_PATH)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["financial_year"] = df["financial_year"].apply(
            lambda v: int(str(v).split("-")[0]) if "-" in str(v) else int(v)
        )

    if state_filter:
        before = len(df)
        df = df[df["state"] == state_filter].reset_index(drop=True)
        print(f"[pipeline] Filtered to {state_filter}: {before} -> {len(df)} rows")

    print(f"[pipeline] Loaded {len(df):,} rows | "
          f"{df['state'].nunique()} states | "
          f"{df['district'].nunique()} districts | "
          f"{df['financial_year'].nunique()} years "
          f"({df['financial_year'].min()}–{df['financial_year'].max()})")
    return df


def _clean_real(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["state", "district", "financial_year"]).reset_index(drop=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    for col in ["person_days_lakhs", "households_availed", "avg_wage_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "avg_wage_rate" in df.columns:
        df["avg_wage_rate"] = df.groupby("state")["avg_wage_rate"].transform(
            lambda s: s.ffill().bfill()
        )

    before = len(df)
    df = df.dropna(subset=["person_days_lakhs"]).reset_index(drop=True)
    if len(df) < before:
        print(f"[pipeline] Dropped {before - len(df)} rows with null person_days_lakhs")

    print(f"[pipeline] Cleaned. Shape: {df.shape}")
    return df


def run_optimizer_step(scope_state: str = None) -> None:
    from src.optimize import run_optimizer
    run_optimizer(
        predictions_path=os.path.join("data", "processed", "mnrega_predictions.csv"),
        raw_path=UNIFIED_PATH,
        scope_state=scope_state,
        target_year=2024,
    )
