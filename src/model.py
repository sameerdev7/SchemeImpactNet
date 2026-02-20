"""
model.py
--------
Train and evaluate a gradient boosting model to predict
next year's person_days_lakhs per district.

Target  : person_days_lakhs (next year)
Features: lag values, efficiency metrics, district encoding, year, wage rate

NOTE: Uses sklearn GradientBoostingRegressor by default.
      To switch to XGBoost (after `pip install xgboost`), change
      the MODEL_BACKEND constant to "xgboost". Interface is identical.

Outputs:
    - Evaluation metrics (RMSE, MAE, R²)
    - Feature importance plot
    - Predictions CSV for all districts
    - reports/figures/06_predictions_vs_actual.png
    - reports/figures/07_feature_importance.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Switch backend here when XGBoost is available ────────────────────────────
MODEL_BACKEND = "xgboost"   # "sklearn" | "xgboost"

FIGURES_DIR = os.path.join("reports", "figures")
OUTPUT_DIR  = os.path.join("data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "financial_year",
    "avg_wage_rate",
    "lag_person_days",
    "lag_expenditure",
    "expenditure_per_personday",
    "yoy_growth",
    "district_avg_persondays",
    "demand_fulfillment_rate",   # may be NaN — model handles it
    "district_encoded",
]

TARGET = "person_days_lakhs"


def run_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full model pipeline: prepare → split → train → evaluate → predict.

    Args:
        df: Feature-engineered DataFrame from features.py

    Returns:
        DataFrame with predictions appended.
    """
    print("\n[model] ── Starting Model Pipeline ────────────────────────────")

    df = _encode_district(df)
    X, y = _prepare_xy(df)

    X_train, X_test, y_train, y_test, test_idx = _temporal_split(df, X, y)

    model = _build_model()
    model.fit(X_train, y_train)
    print(f"[model] Trained {model.__class__.__name__} on {len(X_train)} samples")

    results_df = _evaluate(model, X_test, y_test, df, test_idx)
    _plot_predictions(results_df)
    _plot_feature_importance(model, X.columns.tolist())

    predictions_df = _predict_all(model, df, X)
    _save_predictions(predictions_df)

    print("[model] ── Model Pipeline Complete ────────────────────────────\n")
    return predictions_df


# ── Data preparation ─────────────────────────────────────────────────────────

def _encode_district(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    df["district_encoded"] = le.fit_transform(df["district"])
    return df


def _prepare_xy(df: pd.DataFrame) -> tuple:
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()

    # Fill NaN in non-critical features with column median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    y = df[TARGET].copy()
    return X, y


# ── Temporal train/test split ─────────────────────────────────────────────────
# Train on years up to 2021, test on 2022 and 2023
# This mimics real forecasting — never train on future data

def _temporal_split(df, X, y):
    test_years = [2022, 2023]
    test_mask  = df["financial_year"].isin(test_years)
    train_mask = ~test_mask

    print(f"[model] Train years: {sorted(df[train_mask]['financial_year'].unique())}")
    print(f"[model] Test  years: {sorted(df[test_mask]['financial_year'].unique())}")
    print(f"[model] Train size : {train_mask.sum()} | Test size: {test_mask.sum()}")

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        df[test_mask].index
    )


# ── Model ────────────────────────────────────────────────────────────────────

def _build_model():
    if MODEL_BACKEND == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    else:
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )


# ── Evaluation ───────────────────────────────────────────────────────────────

def _evaluate(model, X_test, y_test, df, test_idx) -> pd.DataFrame:
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"\n[model] ── Evaluation Metrics ──────────────────────────────")
    print(f"[model]   RMSE : {rmse:.4f} lakh persondays")
    print(f"[model]   MAE  : {mae:.4f} lakh persondays")
    print(f"[model]   R²   : {r2:.4f}")
    print(f"[model] ────────────────────────────────────────────────────")

    results = df.loc[test_idx, ["district", "financial_year", TARGET]].copy()
    results["predicted"] = preds
    results["error"]     = results[TARGET] - results["predicted"]
    results["abs_error"] = results["error"].abs()

    print("\n[model] Worst predictions (highest absolute error):")
    worst = results.nlargest(5, "abs_error")[
        ["district", "financial_year", TARGET, "predicted", "abs_error"]
    ]
    print(worst.to_string(index=False))

    return results


def _plot_predictions(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for year, grp in results.groupby("financial_year"):
        ax.scatter(grp[TARGET], grp["predicted"], label=str(year), alpha=0.75, s=60)

    lims = [
        min(results[TARGET].min(), results["predicted"].min()) * 0.95,
        max(results[TARGET].max(), results["predicted"].max()) * 1.05,
    ]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Person Days (lakh)")
    ax.set_ylabel("Predicted Person Days (lakh)")
    ax.set_title("Actual vs Predicted — Person Days per District")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "06_predictions_vs_actual.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[model] Saved: {path}")


def _plot_feature_importance(model, feature_names: list) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="#5C6BC0")
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "07_feature_importance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[model] Saved: {path}")

    print("\n[model] Top 3 most important features:")
    for feat, val in importances.sort_values(ascending=False).head(3).items():
        print(f"        {feat}: {val:.4f}")


# ── Predict all districts for latest year ────────────────────────────────────

def _predict_all(model, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Predict for ALL rows so we have a complete forecast table."""
    preds = model.predict(X)
    out = df[["state", "district", "financial_year",
              TARGET, "expenditure_lakhs"]].copy()
    out["predicted_persondays"] = preds.round(3)
    out["prediction_error"]     = (out[TARGET] - out["predicted_persondays"]).round(3)
    return out


def _save_predictions(df: pd.DataFrame) -> None:
    path = os.path.join(OUTPUT_DIR, "mnrega_predictions.csv")
    df.to_csv(path, index=False)
    print(f"[model] Predictions saved → {path}")
