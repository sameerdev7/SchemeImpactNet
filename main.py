"""
main.py
-------
Entry point for SchemeImpactNet Stage 1.

Usage:
    python main.py                  # generate synthetic data + run full pipeline
    python main.py --skip-generate  # use existing CSV in data/raw/
"""

import sys
from src.pipeline import run_pipeline

if __name__ == "__main__":
    skip = "--skip-generate" in sys.argv
    predictions = run_pipeline(skip_generate=skip)

    print("Sample predictions (last year per district):")
    latest = predictions[predictions["financial_year"] == predictions["financial_year"].max()]
    print(latest[["district", "financial_year", "person_days_lakhs", "predicted_persondays"]]
          .sort_values("predicted_persondays", ascending=False)
          .head(10)
          .to_string(index=False))
