"""
model.py
--------
XGBoost model to predict next year's person_days_lakhs per district.

Stage 1: Maharashtra features only
Stage 2+: Automatically uses extra features when present

Temporal split: train on years up to 2021, test on 2022–2023.
This strictly mimics real forecasting — never train on future data.

Switch to XGBoost from sklearn by changing MODEL_BACKEND = "xgboost"
after running: pip install xgboost
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

MODEL_BACKEND = "xgboost"   # "sklearn" | "xgboost"

FIGURES_DIR = os.path.join("reports", "figures")
OUTPUT_DIR  = os.path.join("data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "person_days_lakhs"

# All possible features — model uses whichever exist in the dataframe
ALL_FEATURE_COLS = [
    # Core (Stage 1)
    "financial_year", "avg_wage_rate",
    "lag_person_days", "lag_expenditure",
    "expenditure_per_personday", "yoy_growth",
    "district_avg_persondays", "demand_fulfillment_rate",
    "district_encoded", "state_encoded",
    # Stage 2
    "rainfall_mm", "crop_season_index",
    "rural_population_lakhs", "poverty_rate_pct",
    "drought_flag", "high_poverty_flag",
    # Stage 3
    "scheme_overlap_score", "budget_utilization_rate",
]

TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS  = [2022, 2023]


def run_model(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[model] ── Starting Model Pipeline ──────────────────────────")

    df = _encode_categoricals(df)
    X, y = _prepare_xy(df)
    X_train, X_test, y_train, y_test, test_idx = _temporal_split(df, X, y)

    model = _build_model()
    model.fit(X_train, y_train)
    print(f"[model] Trained {model.__class__.__name__} | features used: {len(X.columns)}")

    results_df = _evaluate(model, X_test, y_test, df, test_idx)
    _plot_predictions(results_df)
    _plot_feature_importance(model, X.columns.tolist())
    _save_model_report(model, results_df, X.columns.tolist())

    predictions_df = _predict_all(model, df, X)
    _save_predictions(predictions_df)

    print("[model] ── Model Pipeline Complete ──────────────────────────\n")
    return predictions_df


# ── Preparation ───────────────────────────────────────────────────────────────

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col, enc_col in [("district", "district_encoded"), ("state", "state_encoded")]:
        if col in df.columns:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col].astype(str))
    return df


def _prepare_xy(df: pd.DataFrame) -> tuple:
    available = [c for c in ALL_FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    # Fill remaining NaN with column median — keeps model robust
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    y = df[TARGET].copy()
    return X, y


def _temporal_split(df, X, y):
    test_mask  = df["financial_year"].isin(TEST_YEARS)
    train_mask = df["financial_year"].isin(TRAIN_YEARS)

    print(f"[model] Train years : {TRAIN_YEARS}")
    print(f"[model] Test years  : {TEST_YEARS}")
    print(f"[model] Train size  : {train_mask.sum()} | Test size: {test_mask.sum()}")

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        df[test_mask].index
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_model():
    params = dict(n_estimators=300, learning_rate=0.05,
                  max_depth=4, subsample=0.8, random_state=42)
    if MODEL_BACKEND == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(**params, colsample_bytree=0.8, verbosity=0)
        except ImportError:
            print("[model] XGBoost not found, falling back to sklearn GradientBoosting")
    return GradientBoostingRegressor(**params)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model, X_test, y_test, df, test_idx) -> pd.DataFrame:
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"\n[model] ── Evaluation (held-out {TEST_YEARS}) ──────────────")
    print(f"[model]   RMSE : {rmse:.4f} lakh persondays")
    print(f"[model]   MAE  : {mae:.4f} lakh persondays")
    print(f"[model]   R2   : {r2:.4f}")
    print(f"[model] ────────────────────────────────────────────────────")

    results = df.loc[test_idx, ["state", "district", "financial_year", TARGET]].copy()
    results["predicted"]  = preds
    results["error"]      = (results[TARGET] - results["predicted"]).round(3)
    results["abs_error"]  = results["error"].abs()

    print("\n[model] Worst 5 predictions:")
    print(results.nlargest(5, "abs_error")[
        ["state", "district", "financial_year", TARGET, "predicted", "abs_error"]
    ].to_string(index=False))

    return results


def _plot_predictions(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for year, grp in results.groupby("financial_year"):
        ax.scatter(grp[TARGET], grp["predicted"], label=str(year), alpha=0.65, s=50)
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
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(5, len(imp) * 0.3)))
    imp.plot(kind="barh", ax=ax, color="#5C6BC0")
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "07_feature_importance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[model] Saved: {path}")

    print("\n[model] Top 5 features:")
    for feat, val in imp.sort_values(ascending=False).head(5).items():
        print(f"        {feat:35s}: {val:.4f}")


def _save_model_report(model, results: pd.DataFrame, features: list) -> None:
    """Save a plain text report every run — lets you track improvement over time."""
    path = os.path.join("reports", "model_report.txt")
    os.makedirs("reports", exist_ok=True)
    rmse = np.sqrt(mean_squared_error(results[TARGET], results["predicted"]))
    mae  = mean_absolute_error(results[TARGET], results["predicted"])
    r2   = r2_score(results[TARGET], results["predicted"])

    with open(path, "w") as f:
        f.write("SchemeImpactNet — Model Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model     : {model.__class__.__name__}\n")
        f.write(f"Test years: {TEST_YEARS}\n")
        f.write(f"RMSE      : {rmse:.4f}\n")
        f.write(f"MAE       : {mae:.4f}\n")
        f.write(f"R2        : {r2:.4f}\n\n")
        f.write("Features used:\n")
        for feat in features:
            f.write(f"  - {feat}\n")
    print(f"[model] Report saved → {path}")


def _predict_all(model, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(X)
    out = df[["state", "district", "financial_year", TARGET, "expenditure_lakhs"]].copy()
    out["predicted_persondays"] = preds.round(3)
    out["prediction_error"]     = (out[TARGET] - out["predicted_persondays"]).round(3)
    return out


def _save_predictions(df: pd.DataFrame) -> None:
    path = os.path.join(OUTPUT_DIR, "mnrega_predictions.csv")
    df.to_csv(path, index=False)
    print(f"[model] Predictions saved → {path}")
