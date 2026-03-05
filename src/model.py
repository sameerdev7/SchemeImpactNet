"""
model.py
--------
V4 Multi-Algorithm Model Selection for MNREGA district-level forecasting.

Algorithms compared via walk-forward CV:
    - GradientBoostingRegressor  (current champion)
    - RandomForestRegressor
    - XGBoost
    - LightGBM
    - Ridge (linear baseline)
    - ElasticNet (regularised linear baseline)

Selection criterion: mean R² across walk-forward CV years (excl. 2022 anomaly).
Best model is saved to models/mnrega_best_model.pkl.

W&B logging:
    - Each algorithm gets its own W&B run (group="mnrega_model_selection")
    - Per-year CV metrics logged as time-series
    - Feature importance logged as bar chart
    - Model comparison summary table logged
    - Best model flagged with tag "champion"

Usage:
    export WANDB_API_KEY=your_key   # or wandb login
    python main.py --stage 3
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Optional imports — graceful fallback if not installed
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[model] xgboost not installed — skipping")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[model] lightgbm not installed — skipping")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[model] wandb not installed — metrics will be logged locally only")

from src.features import FEATURE_COLS

TARGET      = "person_days_lakhs"
FIGURES_DIR = os.path.join("reports", "figures")
OUTPUT_DIR  = os.path.join("data", "processed")
MODELS_DIR  = "models"
MODEL_PATH  = os.path.join(MODELS_DIR, "mnrega_best_model.pkl")
WANDB_PROJECT = "SchemeImpactNet"
WANDB_GROUP   = "mnrega_model_selection"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# Walk-forward CV test years
WF_TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# ── Algorithm registry ────────────────────────────────────────────────────────
def _build_candidates() -> dict:
    """
    Returns dict of {name: estimator}.
    Each estimator is either a plain sklearn estimator or a Pipeline
    (for linear models that need scaling).
    """
    candidates = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.7, min_samples_leaf=10, random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            n_jobs=-1, random_state=42,
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)),
        ]),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
        )
    if HAS_LGB:
        candidates["LightGBM"] = LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=-1,
        )
    return candidates


# ── Main entry point ──────────────────────────────────────────────────────────

def run_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full model selection pipeline:
      1. Walk-forward CV for each algorithm candidate
      2. Select best by mean R² (excl. 2022)
      3. Train winner on all data
      4. Save model + metadata pkl
      5. Generate figures + W&B logs
      6. Return predictions DataFrame
    """
    print("\n[model] ── V4 Multi-Algorithm Model Selection ───────────────")

    features = _get_features(df)
    print(f"[model] Features ({len(features)}): {features}")
    print(f"[model] Algorithms: {list(_build_candidates().keys())}")

    candidates = _build_candidates()

    # ── Walk-forward CV for all candidates ────────────────────────────────
    all_cv_results = {}
    for name, estimator in candidates.items():
        print(f"\n[model] ── {name} ──")
        cv = _walk_forward_cv(df, features, estimator, name)
        all_cv_results[name] = cv

    # ── Select best model ─────────────────────────────────────────────────
    best_name, best_cv = _select_best(all_cv_results)
    print(f"\n[model] ✓ Best model: {best_name}")

    # ── Print full comparison table ───────────────────────────────────────
    _print_comparison_table(all_cv_results)

    # ── Train winner on all data ──────────────────────────────────────────
    print(f"\n[model] Training {best_name} on all {len(df):,} district-years...")
    best_estimator = candidates[best_name]
    X_all = df[features].fillna(0)
    y_all = df[TARGET]
    best_estimator.fit(X_all, y_all)

    # ── Log to W&B ────────────────────────────────────────────────────────
    if HAS_WANDB:
        _wandb_log_all(all_cv_results, best_name, best_estimator, features, df)

    # ── Save best model ───────────────────────────────────────────────────
    _save_model(best_name, best_estimator, features, best_cv, all_cv_results, df)

    # ── Figures ───────────────────────────────────────────────────────────
    _plot_model_comparison(all_cv_results, best_name)
    _plot_cv_per_year(all_cv_results, best_name)
    _plot_feature_importance(best_name, best_estimator, features)

    # ── Predictions + report ──────────────────────────────────────────────
    predictions_df = _predict_all(best_estimator, df, features)
    _save_predictions(predictions_df)
    _save_model_report(best_name, best_cv, all_cv_results, features, best_estimator)

    print("\n[model] ── V4 Pipeline Complete ─────────────────────────────\n")
    return predictions_df


# ── Walk-forward CV ───────────────────────────────────────────────────────────

def _walk_forward_cv(
    df: pd.DataFrame,
    features: list,
    estimator,
    name: str,
) -> pd.DataFrame:
    """Walk-forward CV: train on years < T, evaluate on T."""
    print(f"  {'Year':<6} {'n':>5}  {'R²':>8}  {'MAE':>8}  {'RMSE':>8}  {'Naive R²':>10}  {'R² gain':>8}")
    print(f"  {'-'*68}")

    rows = []
    for test_yr in WF_TEST_YEARS:
        tr = df[df["financial_year"] < test_yr]
        te = df[df["financial_year"] == test_yr]
        if len(tr) < 200 or len(te) < 50:
            continue

        import copy
        m = copy.deepcopy(estimator)
        m.fit(tr[features].fillna(0), tr[TARGET])
        pred  = m.predict(te[features].fillna(0))
        naive = te["lag1_pd"].fillna(te[TARGET].mean()).values

        r2      = r2_score(te[TARGET], pred)
        mae     = mean_absolute_error(te[TARGET], pred)
        rmse    = np.sqrt(mean_squared_error(te[TARGET], pred))
        naive_r2  = r2_score(te[TARGET], naive)
        naive_mae = mean_absolute_error(te[TARGET], naive)
        mape    = np.mean(np.abs((te[TARGET].values - pred) / (te[TARGET].values + 1e-9))) * 100

        print(f"  {test_yr:<6} {len(te):>5}  {r2:>8.4f}  {mae:>8.3f}  {rmse:>8.3f}  "
              f"{naive_r2:>10.4f}  {r2-naive_r2:>+8.4f}")

        rows.append({
            "year": test_yr, "n": len(te),
            "r2": round(r2, 4),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "mape": round(mape, 3),
            "naive_r2": round(naive_r2, 4),
            "naive_mae": round(naive_mae, 3),
            "r2_gain": round(r2 - naive_r2, 4),
            "mae_gain": round(naive_mae - mae, 3),
        })

    cv = pd.DataFrame(rows)
    ex22 = cv[cv["year"] != 2022]
    print(f"  → Mean R²={cv['r2'].mean():.4f}  excl.2022 R²={ex22['r2'].mean():.4f}  "
          f"MAE={cv['mae'].mean():.3f}L")
    return cv


# ── Model selection ───────────────────────────────────────────────────────────

def _select_best(all_cv: dict) -> tuple:
    """Select best model by mean R² excluding 2022 anomaly year."""
    scores = {}
    for name, cv in all_cv.items():
        ex22 = cv[cv["year"] != 2022]
        scores[name] = ex22["r2"].mean()

    best_name = max(scores, key=scores.get)
    print(f"\n[model] Model selection (mean R² excl. 2022):")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:<20}: {score:.4f}{marker}")

    return best_name, all_cv[best_name]


def _print_comparison_table(all_cv: dict) -> None:
    print(f"\n[model] Full comparison (all years):")
    print(f"  {'Model':<20}  {'R²':>8}  {'excl22 R²':>10}  {'MAE':>8}  {'RMSE':>8}  {'R²gain':>8}")
    print(f"  {'-'*72}")
    for name, cv in all_cv.items():
        ex22 = cv[cv["year"] != 2022]
        print(f"  {name:<20}  {cv['r2'].mean():>8.4f}  {ex22['r2'].mean():>10.4f}  "
              f"{cv['mae'].mean():>8.3f}  {cv['rmse'].mean():>8.3f}  "
              f"{cv['r2_gain'].mean():>+8.4f}")


# ── W&B logging ───────────────────────────────────────────────────────────────

def _wandb_log_all(
    all_cv: dict,
    best_name: str,
    best_estimator,
    features: list,
    df: pd.DataFrame,
) -> None:
    """Log all model results to W&B — one run per algorithm + one summary run."""

    # ── Per-algorithm runs ────────────────────────────────────────────────
    for name, cv in all_cv.items():
        ex22 = cv[cv["year"] != 2022]
        tags = ["champion"] if name == best_name else []

        run = wandb.init(
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            name=name,
            tags=tags,
            config={
                "algorithm": name,
                "n_features": len(features),
                "features": features,
                "wf_test_years": WF_TEST_YEARS,
                "target": TARGET,
                "is_best": name == best_name,
            },
            reinit=True,
        )

        # Per-year CV metrics as time series
        for _, row in cv.iterrows():
            run.log({
                "year": int(row["year"]),
                "r2": row["r2"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "mape": row["mape"],
                "naive_r2": row["naive_r2"],
                "r2_gain": row["r2_gain"],
                "mae_gain": row["mae_gain"],
                "is_anomaly_year": int(row["year"]) == 2022,
            })

        # Summary metrics
        run.summary.update({
            "cv_mean_r2":     round(cv["r2"].mean(), 4),
            "cv_ex22_r2":     round(ex22["r2"].mean(), 4),
            "cv_mean_mae":    round(cv["mae"].mean(), 3),
            "cv_mean_rmse":   round(cv["rmse"].mean(), 3),
            "cv_mean_mape":   round(cv["mape"].mean(), 3),
            "cv_r2_gain":     round(cv["r2_gain"].mean(), 4),
            "n_districts":    df["district"].nunique(),
            "n_states":       df["state"].nunique(),
            "train_years":    len(df["financial_year"].unique()),
        })

        # Feature importance (tree-based only)
        fi = _get_feature_importance(name, best_estimator if name == best_name else None, features)
        if fi is not None and name == best_name:
            fi_table = wandb.Table(
                columns=["feature", "importance"],
                data=[[f, v] for f, v in sorted(fi.items(), key=lambda x: -x[1])]
            )
            run.log({"feature_importance": wandb.plot.bar(
                fi_table, "feature", "importance",
                title=f"Feature Importance — {name}"
            )})

        # CV R² chart per year
        cv_table = wandb.Table(dataframe=cv[["year","r2","naive_r2","mae","rmse","r2_gain"]])
        run.log({
            "cv_results_table": cv_table,
            "cv_r2_chart": wandb.plot.line_series(
                xs=cv["year"].tolist(),
                ys=[cv["r2"].tolist(), cv["naive_r2"].tolist()],
                keys=["Model R²", "Naive R²"],
                title=f"Walk-Forward CV R² — {name}",
                xname="Financial Year",
            ),
        })

        run.finish()

    # ── Summary comparison run ────────────────────────────────────────────
    run = wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name="model_selection_summary",
        tags=["summary"],
        reinit=True,
    )

    summary_rows = []
    for name, cv in all_cv.items():
        ex22 = cv[cv["year"] != 2022]
        summary_rows.append([
            name,
            round(cv["r2"].mean(), 4),
            round(ex22["r2"].mean(), 4),
            round(cv["mae"].mean(), 3),
            round(cv["rmse"].mean(), 3),
            round(cv["mape"].mean(), 3),
            round(cv["r2_gain"].mean(), 4),
            name == best_name,
        ])

    summary_table = wandb.Table(
        columns=["model", "mean_r2", "ex22_r2", "mean_mae",
                 "mean_rmse", "mean_mape", "r2_gain", "is_best"],
        data=summary_rows,
    )
    run.log({
        "model_comparison": summary_table,
        "best_model": best_name,
        "best_ex22_r2": round(all_cv[best_name][all_cv[best_name]["year"] != 2022]["r2"].mean(), 4),
    })

    # Comparison bar chart
    run.log({
        "r2_comparison": wandb.plot.bar(
            wandb.Table(
                columns=["model", "ex22_r2"],
                data=[[r[0], r[2]] for r in summary_rows]
            ),
            "model", "ex22_r2",
            title="Model Comparison — R² excl. 2022",
        )
    })

    run.finish()
    print(f"[model] W&B logs complete → project: {WANDB_PROJECT} / group: {WANDB_GROUP}")


# ── Figures ───────────────────────────────────────────────────────────────────

def _plot_model_comparison(all_cv: dict, best_name: str) -> None:
    """Bar chart comparing all models on mean R² (all years and excl. 2022)."""
    names = list(all_cv.keys())
    mean_r2  = [all_cv[n]["r2"].mean() for n in names]
    ex22_r2  = [all_cv[n][all_cv[n]["year"] != 2022]["r2"].mean() for n in names]
    mean_mae = [all_cv[n]["mae"].mean() for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(x - w/2, mean_r2,  w, label="All years",    alpha=0.8, color="#42A5F5")
    bars2 = ax1.bar(x + w/2, ex22_r2, w, label="excl. 2022",   alpha=0.8, color="#26A69A")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Mean R² (Walk-Forward CV)")
    ax1.set_title("Model Comparison — R² Score")
    ax1.set_ylim(0, 1)
    ax1.legend()
    # Annotate best
    best_idx = names.index(best_name)
    ax1.annotate("★ BEST", xy=(best_idx + w/2, ex22_r2[best_idx] + 0.01),
                 ha="center", color="#E53935", fontsize=9, fontweight="bold")

    bars3 = ax2.bar(x, mean_mae, alpha=0.8,
                    color=["#E53935" if n == best_name else "#78909C" for n in names])
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Mean MAE (lakh person-days)")
    ax2.set_title("Model Comparison — MAE")
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("SchemeImpactNet V4 — Algorithm Selection Results", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "06_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[model] Saved: {path}")


def _plot_cv_per_year(all_cv: dict, best_name: str) -> None:
    """Line chart: R² per year for every algorithm."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_cv)))
    for (name, cv), color in zip(all_cv.items(), colors):
        lw  = 2.5 if name == best_name else 1.2
        ls  = "-"  if name == best_name else "--"
        alpha = 1.0 if name == best_name else 0.65
        axes[0].plot(cv["year"], cv["r2"], marker="o", label=name,
                     linewidth=lw, linestyle=ls, alpha=alpha, color=color)
        axes[1].plot(cv["year"], cv["mae"], marker="o", label=name,
                     linewidth=lw, linestyle=ls, alpha=alpha, color=color)

    for ax in axes:
        ax.axvspan(2021.5, 2022.5, alpha=0.08, color="red", label="2022 anomaly")
        ax.axvspan(2019.5, 2020.5, alpha=0.05, color="orange", label="COVID-2020")
        ax.set_xticks(WF_TEST_YEARS)
        ax.set_xlabel("Financial Year")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("R²"); axes[0].set_title("Walk-Forward CV R² by Year")
    axes[1].set_ylabel("MAE (lakh PD)"); axes[1].set_title("Walk-Forward CV MAE by Year")

    plt.suptitle("All Models — Walk-Forward CV Results", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "07_cv_per_year.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[model] Saved: {path}")


def _plot_feature_importance(name: str, estimator, features: list) -> None:
    fi = _get_feature_importance(name, estimator, features)
    if fi is None:
        return
    imp = pd.Series(fi).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(5, len(imp) * 0.35)))
    colors = ["#E53935" if imp[f] > imp.quantile(0.75) else "#42A5F5" for f in imp.index]
    imp.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Feature Importances — {name} (Best Model)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "08_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[model] Saved: {path}")
    print(f"\n[model] Top 5 features ({name}):")
    for feat, val in imp.sort_values(ascending=False).head(5).items():
        print(f"  {feat:<35}: {val:.4f}")


def _get_feature_importance(name: str, estimator, features: list):
    """Extract feature importance — works for tree models and linear models."""
    if estimator is None:
        return None
    try:
        # Tree-based: direct feature_importances_
        if hasattr(estimator, "feature_importances_"):
            return dict(zip(features, estimator.feature_importances_))
        # Pipeline with tree inside
        if hasattr(estimator, "named_steps"):
            inner = list(estimator.named_steps.values())[-1]
            if hasattr(inner, "feature_importances_"):
                return dict(zip(features, inner.feature_importances_))
            if hasattr(inner, "coef_"):
                return dict(zip(features, np.abs(inner.coef_)))
        # XGBoost / LightGBM
        if hasattr(estimator, "feature_importances_"):
            return dict(zip(features, estimator.feature_importances_))
    except Exception:
        pass
    return None


# ── Model persistence ─────────────────────────────────────────────────────────

def _save_model(
    best_name: str,
    best_estimator,
    features: list,
    best_cv: pd.DataFrame,
    all_cv: dict,
    df: pd.DataFrame,
) -> None:
    ex22 = best_cv[best_cv["year"] != 2022]

    # Build comparison summary for the bundle
    comparison = {}
    for name, cv in all_cv.items():
        e22 = cv[cv["year"] != 2022]
        comparison[name] = {
            "mean_r2": round(cv["r2"].mean(), 4),
            "ex22_r2": round(e22["r2"].mean(), 4),
            "mean_mae": round(cv["mae"].mean(), 3),
            "mean_rmse": round(cv["rmse"].mean(), 3),
        }

    bundle = {
        "model":              best_estimator,
        "model_name":         best_name,
        "features":           features,
        "target":             TARGET,
        "covid_multiplier":   1.447,
        "train_years":        sorted(df["financial_year"].unique().tolist()),
        "n_districts":        df["district"].nunique(),
        "n_states":           df["state"].nunique(),
        "feature_importance": _get_feature_importance(best_name, best_estimator, features),
        "cv_results":         best_cv.to_dict(),
        "cv_mean_r2":         round(best_cv["r2"].mean(), 4),
        "cv_ex22_r2":         round(ex22["r2"].mean(), 4),
        "cv_mean_mae":        round(best_cv["mae"].mean(), 3),
        "all_model_comparison": comparison,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[model] Model saved → {MODEL_PATH}")
    print(f"[model] Best: {best_name}  |  ex22 R²={ex22['r2'].mean():.4f}  |  MAE={best_cv['mae'].mean():.3f}L")


def load_model(path: str = MODEL_PATH) -> dict:
    """Load the saved best model bundle."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[model] Loaded: {bundle['model_name']} from {path}")
    print(f"[model] ex22 R²={bundle['cv_ex22_r2']}  |  MAE={bundle['cv_mean_mae']}L")
    return bundle


# ── Prediction helpers ────────────────────────────────────────────────────────

def _predict_all(estimator, df: pd.DataFrame, features: list) -> pd.DataFrame:
    preds = estimator.predict(df[features].fillna(0))
    out = df[["state", "district", "financial_year", TARGET]].copy()
    out["predicted_persondays"] = preds.round(3)
    out["prediction_error"]     = (out[TARGET] - out["predicted_persondays"]).round(3)
    out["abs_error"]            = out["prediction_error"].abs()
    return out


def _save_predictions(df: pd.DataFrame) -> None:
    path = os.path.join(OUTPUT_DIR, "mnrega_predictions.csv")
    df.to_csv(path, index=False)
    print(f"[model] Predictions saved → {path}")


# ── Report ────────────────────────────────────────────────────────────────────

def _save_model_report(
    best_name: str,
    best_cv: pd.DataFrame,
    all_cv: dict,
    features: list,
    best_estimator,
) -> None:
    ex22 = best_cv[best_cv["year"] != 2022]
    path = os.path.join("reports", "model_report.txt")
    os.makedirs("reports", exist_ok=True)
    with open(path, "w") as f:
        f.write("SchemeImpactNet — V4 Model Selection Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Model    : {best_name}\n")
        f.write(f"Selection     : max mean R² excl. 2022 (walk-forward CV)\n")
        f.write(f"Features      : {len(features)}\n")
        f.write(f"Evaluation    : Walk-forward CV (2018–2024)\n\n")

        f.write("Algorithm Comparison:\n")
        f.write(f"  {'Model':<20}  {'R²':>8}  {'ex22 R²':>10}  {'MAE':>8}  {'RMSE':>8}\n")
        f.write(f"  {'-'*60}\n")
        for name, cv in all_cv.items():
            e22 = cv[cv["year"] != 2022]
            marker = " ← BEST" if name == best_name else ""
            f.write(f"  {name:<20}  {cv['r2'].mean():>8.4f}  "
                    f"{e22['r2'].mean():>10.4f}  {cv['mae'].mean():>8.3f}  "
                    f"{cv['rmse'].mean():>8.3f}{marker}\n")

        f.write(f"\nBest Model ({best_name}) Walk-Forward CV:\n")
        f.write(f"  Mean R²     : {best_cv['r2'].mean():.4f}\n")
        f.write(f"  excl.2022 R²: {ex22['r2'].mean():.4f}\n")
        f.write(f"  Mean MAE    : {best_cv['mae'].mean():.3f} lakh\n")
        f.write(f"  Mean RMSE   : {best_cv['rmse'].mean():.3f} lakh\n")
        f.write(f"  R² gain     : {best_cv['r2_gain'].mean():+.4f} vs naive lag-1\n\n")

        f.write(f"Previous (leaked) R²: 0.9963\n")
        f.write(f"Leakage source: works_completed (r=1.0 with target)\n\n")
        f.write(f"2022 anomaly: West Bengal -93 to -98% reporting drop. Excl. R²={ex22['r2'].mean():.4f}\n\n")

        fi = _get_feature_importance(best_name, best_estimator, features)
        if fi:
            f.write("Feature Importances:\n")
            for feat, val in sorted(fi.items(), key=lambda x: -x[1]):
                f.write(f"  {feat:<35} {val:.4f}\n")

        f.write(f"\nYear-by-year CV ({best_name}):\n")
        f.write(best_cv.to_string(index=False))
    print(f"[model] Report saved → {path}")


# ── Feature list helper ───────────────────────────────────────────────────────

def _get_features(df: pd.DataFrame) -> list:
    available = [f for f in FEATURE_COLS if f in df.columns]
    missing   = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"[model] Warning: {len(missing)} features not in df: {missing}")
    return available
