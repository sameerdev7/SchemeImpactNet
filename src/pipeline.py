"""
pipeline.py
-----------
Stage-aware pipeline orchestrator.

Stage 1: Maharashtra only  → python main.py --stage 1
Stage 2: All-India         → python main.py --stage 2
Stage 3: All-India + schemes → python main.py --stage 3
"""

import os
import pandas as pd

from src.extract import load_csv
from src.clean import clean
from src.features import build_features
from src.eda import run_eda
from src.model import run_model

RAW_PATH       = os.path.join("data", "raw", "mnrega_india_unified.csv")
PROCESSED_PATH = os.path.join("data", "processed", "mnrega_cleaned.csv")

STATE_FILTER = {
    1: "Maharashtra",   # Stage 1: single state
    2: None,            # Stage 2: all-India
    3: None,            # Stage 3: all-India + scheme features
}

SCOPE_LABEL = {
    1: "Maharashtra",
    2: "All-India",
    3: "All-India (with Scheme Interdependency)",
}


def run_pipeline(stage: int = 1) -> pd.DataFrame:
    assert stage in [1, 2, 3], "Stage must be 1, 2, or 3"

    print("\n" + "=" * 60)
    print(f"  SchemeImpactNet — Stage {stage} Pipeline")
    print(f"  Scope : {SCOPE_LABEL[stage]}")
    print("=" * 60)

    # ── Extract ───────────────────────────────────────────────────
    print(f"\n[pipeline] Step 1: Extract")
    df = load_csv(RAW_PATH, state_filter=STATE_FILTER[stage])

    # ── Clean ─────────────────────────────────────────────────────
    print(f"\n[pipeline] Step 2: Clean")
    df = clean(df)

    # ── Features ──────────────────────────────────────────────────
    print(f"\n[pipeline] Step 3: Feature Engineering")
    df = build_features(df)

    # ── Save processed ────────────────────────────────────────────
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n[pipeline] Processed data saved → {PROCESSED_PATH}")

    # ── EDA ───────────────────────────────────────────────────────
    print(f"\n[pipeline] Step 4: EDA")
    run_eda(df, scope=SCOPE_LABEL[stage])

    # ── Model ─────────────────────────────────────────────────────
    print(f"\n[pipeline] Step 5: Model")
    predictions = run_model(df)

    print("\n" + "=" * 60)
    print(f"  Stage {stage} Complete!")
    print(f"  Processed : {PROCESSED_PATH}")
    print(f"  Figures   : reports/figures/")
    print(f"  Predictions: data/processed/mnrega_predictions.csv")
    print(f"  Report    : reports/model_report.txt")
    print("=" * 60 + "\n")

    return predictions


def run_optimizer_step(scope_state: str = None) -> None:
    """Run the budget optimizer — called as part of Stage 3."""
    from src.optimize import run_optimizer
    run_optimizer(
        predictions_path=os.path.join("data", "processed", "mnrega_predictions.csv"),
        raw_path=RAW_PATH,
        scope_state=scope_state,
        target_year=2023,
    )
