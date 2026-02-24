"""
pipeline.py
-----------
Stage-aware pipeline orchestrator.

Stage 1: Maharashtra only  → python main.py --stage 1
Stage 2: All-India         → python main.py --stage 2
Stage 3: All-India + schemes → python main.py --stage 3

Changes from original:
  - run_model_comparison() now called automatically in Stage 3 (Rao et al. 2025)
  - target_year aligned to 2024 (was 2023, mismatched src/optimize.py default)
  - W&B: WANDB_ENABLED=True in src/model.py — set False to skip during demo
"""

import os
import pandas as pd

from src.extract import load_csv
from src.clean import clean
from src.features import build_features
from src.eda import run_eda
from src.model import run_model, _encode_categoricals, _prepare_xy, _temporal_split

RAW_PATH       = os.path.join("data", "raw", "mnrega_india_unified.csv")
PROCESSED_PATH = os.path.join("data", "processed", "mnrega_cleaned.csv")

STATE_FILTER = {
    1: "Maharashtra",
    2: None,
    3: None,
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
    print(f"\n[pipeline] Step 5: Model training + evaluation")
    predictions = run_model(df)

    # ── Model Comparison (Stage 3 only — Rao et al. 2025 Table I) ─
    if stage == 3:
        print(f"\n[pipeline] Step 5b: Model Comparison (Rao et al. 2025)")
        try:
            from src.model import run_model_comparison
            df_enc = _encode_categoricals(df.copy())
            X, y = _prepare_xy(df_enc)
            X_train, X_test, y_train, y_test, test_idx = _temporal_split(df_enc, X, y)
            run_model_comparison(X_train, X_test, y_train, y_test, df_enc, test_idx)
        except Exception as e:
            print(f"[pipeline] Model comparison failed (non-critical): {e}")

    print("\n" + "=" * 60)
    print(f"  Stage {stage} Complete!")
    print(f"  Processed   : {PROCESSED_PATH}")
    print(f"  Figures     : reports/figures/")
    print(f"  Predictions : data/processed/mnrega_predictions.csv")
    print(f"  Report      : reports/model_report.txt")
    if stage == 3:
        print(f"  Comparison  : reports/model_comparison.csv  (Rao et al. 2025)")
        print(f"  W&B         : https://wandb.ai/schemeimpactnet")
    print("=" * 60 + "\n")

    return predictions


def run_optimizer_step(scope_state: str = None) -> None:
    """Run the budget optimizer — called as part of Stage 3.
    
    NOTE: target_year aligned to 2024 (matching src/optimize.py default).
    Original had 2023 — this caused mismatch with optimizer predictions.
    """
    from src.optimize import run_optimizer
    run_optimizer(
        predictions_path=os.path.join("data", "processed", "mnrega_predictions.csv"),
        raw_path=RAW_PATH,
        scope_state=scope_state,
        target_year=2024,   # FIX: was 2023, now matches optimize.py default
    )
