"""
fix_optimizer.py
----------------
Standalone script to re-run the optimizer and regenerate
data/processed/optimized_budget_allocation.csv.

Usage:
    python fix_optimizer.py

Then reseed the database:
    rm data/schemeimpactnet.db
    ./start.sh
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.optimize import run_optimizer

if __name__ == "__main__":
    print("=" * 60)
    print("SchemeImpactNet — Optimizer Fix")
    print("=" * 60)

    result = run_optimizer(
        predictions_path="data/processed/mnrega_predictions.csv",
        raw_path="data/raw/mnrega_real_data_final_clean.csv",
        scope_state=None,
        target_year=2024,
    )

    print(f"\n✅ Done. {len(result)} districts processed.")
    print(f"   budget_change_pct range: {result['budget_change_pct'].min():.1f}% to {result['budget_change_pct'].max():.1f}%")
    print(f"   Unique values: {result['budget_change_pct'].nunique()}")
    print("\nNext steps:")
    print("  rm data/schemeimpactnet.db")
    print("  ./start.sh")
