"""
pipeline.py
-----------
V3 pipeline orchestrator for SchemeImpactNet.

Changes from original:
    - RAW_PATH now points to the real Dataful government CSV
      (confirmed 99% match with mnrega_india_unified.csv, <0.005L diff)
    - Feature engineering uses V3 leak-free features (src/features.py)
    - Model uses GBR V3 with walk-forward CV (src/model.py)
    - Model saved to models/mnrega_gbr_v3.pkl
    - Removed generate_synthetic dependency from Stage 1
    - Stage 3 model comparison retained but flags honest metrics

Data sources:
    Real:      data/raw/20063- Dataful/mnrega-...-persondays-...csv
               → person_days_lakhs, households_availed (real gov data)
               → avg_wage_rate (official wage schedule, exogenous)
    Synthetic: all other columns (rainfall, poverty, pmkisan, pmay)
               → EXCLUDED from V3 model features
"""

import os
import pandas as pd
import numpy as np

from src.clean   import clean
from src.features import build_features
from src.eda     import run_eda
from src.model   import run_model

# ── Data paths ────────────────────────────────────────────────────────────────
DATAFUL_PATH   = os.path.join(
    "data", "raw", "20063- Dataful",
    "mnrega-year-month-state-and-district-wise-total-persondays-"
    "and-households-engaged-in-work.csv"
)
UNIFIED_PATH   = os.path.join("data", "raw", "mnrega_india_unified.csv")
PROCESSED_PATH = os.path.join("data", "processed", "mnrega_cleaned.csv")
MODEL_PATH     = os.path.join("models", "mnrega_best_model.pkl")

SCOPE_LABEL = {
    1: "Maharashtra",
    2: "All-India",
    3: "All-India (V3 leak-free)",
}


def run_pipeline(stage: int = 3) -> pd.DataFrame:
    assert stage in [1, 2, 3], "Stage must be 1, 2, or 3"

    print("\n" + "=" * 60)
    print(f"  SchemeImpactNet V3 — Stage {stage} Pipeline")
    print(f"  Scope : {SCOPE_LABEL[stage]}")
    print("=" * 60)

    # ── Extract ───────────────────────────────────────────────────
    print(f"\n[pipeline] Step 1: Extract (real government data)")
    df = _load_real_data(state_filter="Maharashtra" if stage == 1 else None)

    # ── Clean ─────────────────────────────────────────────────────
    print(f"\n[pipeline] Step 2: Clean")
    df = _clean_real(df)

    # ── Features ──────────────────────────────────────────────────
    print(f"\n[pipeline] Step 3: V3 Feature Engineering (leak-free)")
    df = build_features(df)

    # ── Save processed ────────────────────────────────────────────
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n[pipeline] Processed data saved → {PROCESSED_PATH}")

    # ── EDA ───────────────────────────────────────────────────────
    print(f"\n[pipeline] Step 4: EDA")
    run_eda(df, scope=SCOPE_LABEL[stage])

    # ── Model ─────────────────────────────────────────────────────
    print(f"\n[pipeline] Step 5: V3 Model (walk-forward CV + pkl save)")
    predictions = run_model(df)

    print("\n" + "=" * 60)
    print(f"  Stage {stage} Complete!")
    print(f"  Processed   : {PROCESSED_PATH}")
    print(f"  Model       : {MODEL_PATH}")
    print(f"  Figures     : reports/figures/")
    print(f"  Predictions : data/processed/mnrega_predictions.csv")
    print(f"  Report      : reports/model_report.txt")
    print("=" * 60 + "\n")

    return predictions


# ── Real data loader ──────────────────────────────────────────────────────────

def _load_real_data(state_filter: str = None) -> pd.DataFrame:
    """
    Load and pivot the Dataful government CSV from long format
    (one row per district-month-category) to annual wide format
    (one row per district-year with person_days_lakhs + households_availed).

    Falls back to unified CSV if Dataful not found.
    """
    if os.path.exists(DATAFUL_PATH):
        print(f"[pipeline] Loading Dataful government CSV: {DATAFUL_PATH}")
        df_raw = pd.read_csv(DATAFUL_PATH)
        df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]

        # Parse fiscal year start integer from '2014-15' → 2014
        df_raw["fy"] = df_raw["fiscal_year"].apply(
            lambda v: int(str(v).split("-")[0]) if "-" in str(v) else int(v)
        )
        # Exclude incomplete current fiscal year
        df_raw = df_raw[df_raw["fy"] <= 2024]

        # Pivot: sum monthly values to annual per district
        pivot = df_raw.pivot_table(
            index=["fiscal_year", "fy", "state", "district"],
            columns="category",
            values="value",
            aggfunc="sum"
        ).reset_index()
        pivot.columns.name = None

        # Rename to match model schema
        pivot = pivot.rename(columns={
            "Persondays": "person_days",
            "Household":  "households_availed",
            "fy":         "financial_year",
        })
        pivot["person_days_lakhs"] = (pivot["person_days"] / 1e5).round(3)

        # Bring in avg_wage_rate from unified CSV (official schedule, exogenous)
        if os.path.exists(UNIFIED_PATH):
            df_uni = pd.read_csv(UNIFIED_PATH)
            df_uni.columns = [c.strip().lower().replace(" ", "_") for c in df_uni.columns]
            df_uni["financial_year"] = df_uni["financial_year"].apply(
                lambda v: int(str(v).split("-")[0]) if "-" in str(v) else int(v)
            )
            wage_map = df_uni[["state", "financial_year", "avg_wage_rate"]].drop_duplicates()
            pivot = pivot.merge(wage_map, on=["state", "financial_year"], how="left")

        # Keep only needed columns
        keep = ["state", "district", "financial_year",
                "person_days_lakhs", "households_availed", "avg_wage_rate"]
        df = pivot[[c for c in keep if c in pivot.columns]].copy()

    else:
        print(f"[pipeline] Dataful CSV not found, falling back to unified CSV")
        print(f"[pipeline] NOTE: unified CSV contains synthetic columns — "
              f"V3 features ignore them")
        df = pd.read_csv(UNIFIED_PATH)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["financial_year"] = df["financial_year"].apply(
            lambda v: int(str(v).split("-")[0]) if "-" in str(v) else int(v)
        )

    if state_filter:
        before = len(df)
        df = df[df["state"] == state_filter].reset_index(drop=True)
        print(f"[pipeline] Filtered to {state_filter}: {before} → {len(df)} rows")

    print(f"[pipeline] Loaded {len(df):,} rows | "
          f"{df['state'].nunique()} states | "
          f"{df['district'].nunique()} districts | "
          f"{df['financial_year'].nunique()} years "
          f"({df['financial_year'].min()}–{df['financial_year'].max()})")
    return df


def _clean_real(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight clean for the real Dataful data.
    The full clean() from src/clean.py expects synthetic columns —
    we do a minimal version here.
    """
    df = df.sort_values(["state", "district", "financial_year"]).reset_index(drop=True)

    # Strip strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Numeric cast
    for col in ["person_days_lakhs", "households_availed", "avg_wage_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill wage within state (official schedule rarely changes mid-year)
    if "avg_wage_rate" in df.columns:
        df["avg_wage_rate"] = df.groupby("state")["avg_wage_rate"].transform(
            lambda s: s.ffill().bfill()
        )

    # Drop rows with no person_days_lakhs
    before = len(df)
    df = df.dropna(subset=["person_days_lakhs"]).reset_index(drop=True)
    if len(df) < before:
        print(f"[pipeline] Dropped {before - len(df)} rows with null person_days_lakhs")

    print(f"[pipeline] Cleaned. Shape: {df.shape}")
    return df


def run_optimizer_step(scope_state: str = None) -> None:
    """Run the budget optimizer after predictions are generated."""
    from src.optimize import run_optimizer
    run_optimizer(
        predictions_path=os.path.join("data", "processed", "mnrega_predictions.csv"),
        raw_path=UNIFIED_PATH,
        scope_state=scope_state,
        target_year=2024,
    )
