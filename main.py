"""
main.py
-------
Entry point for SchemeImpactNet.

Usage:
    python main.py                        # Run full pipeline
    python main.py --state Maharashtra    # Run for one state
    python main.py --optimize-only        # Run optimizer on existing predictions
"""

import sys
from src.pipeline import run_pipeline, run_optimizer_step

if __name__ == "__main__":
    args = sys.argv[1:]

    scope_state = None
    if "--state" in args:
        scope_state = args[args.index("--state") + 1]

    optimize_only = "--optimize-only" in args

    if optimize_only:
        print("\nRunning optimizer on existing predictions...")
        run_optimizer_step(scope_state=scope_state)
    else:
        predictions = run_pipeline()

        print(f"\nTop 10 predicted districts (2023):")
        latest = predictions[predictions["financial_year"] == 2023]
        print(
            latest[["state", "district", "person_days_lakhs", "predicted_persondays"]]
            .sort_values("predicted_persondays", ascending=False)
            .head(10)
            .to_string(index=False)
        )

        print("\n" + "─" * 60)
        print("  Running Budget Optimizer...")
        print("─" * 60)
        run_optimizer_step(scope_state=scope_state)
