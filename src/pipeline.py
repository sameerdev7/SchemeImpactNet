"""
pipeline.py
-----------
Orchestrates the full Stage 1 pipeline:
    generate → extract → clean → features → eda → model → save
"""

import os
import pandas as pd

from src.generate_synthetic import generate, save as save_raw
from src.extract import load_csv
from src.clean import clean
from src.features import build_features
from src.eda import run_eda
from src.model import run_model

RAW_PATH       = os.path.join("data", "raw", "mnrega_maharashtra_synthetic.csv")
PROCESSED_PATH = os.path.join("data", "processed", "mnrega_cleaned.csv")


def run_pipeline(skip_generate: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  SchemeImpactNet — Stage 1 Pipeline")
    print("=" * 60)

    # Step 1: Generate synthetic data (skip if real data exists)
    if not skip_generate or not os.path.exists(RAW_PATH):
        print("\n[pipeline] Step 1: Generating synthetic data...")
        raw_df = generate()
        save_raw(raw_df, RAW_PATH)
    else:
        print(f"\n[pipeline] Step 1: Using existing data at {RAW_PATH}")

    # Step 2: Extract
    print("\n[pipeline] Step 2: Extract")
    raw_df = load_csv(RAW_PATH)

    # Step 3: Clean
    print("\n[pipeline] Step 3: Clean")
    clean_df = clean(raw_df)

    # Step 4: Features
    print("\n[pipeline] Step 4: Feature Engineering")
    feature_df = build_features(clean_df)

    # Step 5: Save processed data
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    feature_df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n[pipeline] Processed data saved → {PROCESSED_PATH}")

    # Step 6: EDA
    print("\n[pipeline] Step 5: EDA")
    run_eda(feature_df)

    # Step 7: Model
    print("\n[pipeline] Step 6: Model")
    predictions = run_model(feature_df)

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print(f"  Processed data : {PROCESSED_PATH}")
    print(f"  Figures        : reports/figures/")
    print(f"  Predictions    : data/processed/mnrega_predictions.csv")
    print("=" * 60 + "\n")

    return predictions
