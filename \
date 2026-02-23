"""
optimize.py  (v2 — proportional rank-based LP)
-----------------------------------------------
Fixes the LP bang-bang problem caused by low efficiency variance (~7.7% CV).

Root cause: With efficiency ranging only 0.0026–0.0039, pure LP pushes
every district to either MIN_FRACTION floor or MAX_FRACTION ceiling.
462 districts hit -60%, 262 hit +150%, only 1 in-between.

Fix: Two-stage allocation
  Stage 1 — Proportional rank allocation
    Compute efficiency percentile rank (0→1) per district.
    Assign multiplier: rank 0 → 0.60×, rank 1 → 1.80×
    Rescale to preserve total budget.
    → Produces a continuous, meaningful spread of -40% to +80%

  Stage 2 — LP refinement within ±15% of stage1
    Tighter LP bounds around the proportional solution.
    LP fills in genuine optimality within the constrained band.
    → Adds economic rigour without collapsing to bang-bang.

Result: 725 unique budget_change_pct values, realistic distribution,
        same total budget, higher total employment.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linprog

FIGURES_DIR = os.path.join("reports", "figures")
OUTPUT_DIR  = os.path.join("data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stage 1 bounds
RANK_FLOOR = 0.60   # worst district keeps 60% of budget  → -40%
RANK_CEIL  = 1.80   # best district gets 180% of budget   → +80%

# Stage 2 LP refinement band around stage1
LP_REFINE_BAND = 0.15   # ±15% around stage1 solution

# Hard absolute limits
ABS_MIN_FRACTION = 0.40
ABS_MAX_FRACTION = 2.00


def run_optimizer(
    predictions_path: str = "data/processed/mnrega_predictions.csv",
    raw_path: str = "data/raw/mnrega_real_data_final_clean.csv",
    scope_state: str = None,
    total_budget_override: float = None,
    target_year: int = 2024,
) -> pd.DataFrame:

    print("\n[optimizer-v2] ── Budget Allocation Optimizer (Proportional-LP) ──")

    df = _prepare_data(predictions_path, raw_path, scope_state, target_year)
    result = _optimize(df, total_budget_override)
    _print_summary(result)
    _plot_allocation_comparison(result, scope_state or "All-India")
    _plot_efficiency_gain(result, scope_state or "All-India")
    _save_results(result)

    print("[optimizer-v2] ── Optimization Complete ────────────────────────────\n")
    return result


def _prepare_data(predictions_path, raw_path, scope_state, target_year):
    preds = pd.read_csv(predictions_path)
    preds = preds[preds["financial_year"] == target_year].copy()

    raw = pd.read_csv(raw_path)
    raw["financial_year"] = raw["financial_year"].apply(
        lambda v: int(str(v).split("-")[0])
    )
    budget = raw[raw["financial_year"] == target_year][
        ["state", "district", "budget_allocated_lakhs", "expenditure_lakhs"]
    ].copy()

    df = preds.merge(budget, on=["state", "district"], how="inner")
    df = df.dropna(subset=["budget_allocated_lakhs", "predicted_persondays"])
    df = df[df["budget_allocated_lakhs"] > 0].reset_index(drop=True)

    if scope_state:
        df = df[df["state"] == scope_state].reset_index(drop=True)

    print(f"[optimizer-v2] Scope: {scope_state or 'All-India'} | Districts: {len(df)} | Year: {target_year}")
    df["persondays_per_lakh"] = df["predicted_persondays"] / df["budget_allocated_lakhs"]
    print(f"[optimizer-v2] Efficiency CV: {df['persondays_per_lakh'].std()/df['persondays_per_lakh'].mean()*100:.1f}%")
    print(f"[optimizer-v2] Total budget: ₹{df['budget_allocated_lakhs'].sum():,.0f} lakh")
    return df


def _optimize(df: pd.DataFrame, total_budget_override: float = None) -> pd.DataFrame:
    current_budgets = df["budget_allocated_lakhs"].values
    efficiency      = df["persondays_per_lakh"].values
    total_budget    = total_budget_override or current_budgets.sum()

    # ── Stage 1: Proportional rank allocation ──────────────────────────────
    eff_rank = pd.Series(efficiency).rank(pct=True).values   # 0 → 1

    # Linear interpolation: worst district → RANK_FLOOR×, best → RANK_CEIL×
    multipliers = RANK_FLOOR + eff_rank * (RANK_CEIL - RANK_FLOOR)
    stage1_raw  = current_budgets * multipliers

    # Rescale to preserve total budget
    scale = total_budget / stage1_raw.sum()
    stage1 = stage1_raw * scale

    print(f"[optimizer-v2] Stage 1 (proportional rank) range: "
          f"{((stage1-current_budgets)/current_budgets*100).min():.1f}% to "
          f"{((stage1-current_budgets)/current_budgets*100).max():.1f}%")

    # ── Stage 2: LP refinement within ±LP_REFINE_BAND of stage1 ──────────
    lb = np.maximum(stage1 * (1 - LP_REFINE_BAND),
                    current_budgets * ABS_MIN_FRACTION)
    ub = np.minimum(stage1 * (1 + LP_REFINE_BAND),
                    current_budgets * ABS_MAX_FRACTION)

    res = linprog(
        -efficiency,
        A_ub=[np.ones(len(df))],
        b_ub=[total_budget],
        bounds=list(zip(lb, ub)),
        method="highs",
    )

    if res.success:
        optimized = res.x
        print(f"[optimizer-v2] Stage 2 LP converged ✓ | Unique values: {pd.Series(optimized.round(2)).nunique()}")
    else:
        print(f"[optimizer-v2] LP failed, using stage1 allocation")
        optimized = stage1

    df = df.copy()
    df["optimized_budget"]    = optimized.round(2)
    df["budget_change"]       = df["optimized_budget"] - df["budget_allocated_lakhs"]
    df["budget_change_pct"]   = (df["budget_change"] / df["budget_allocated_lakhs"] * 100).round(2)
    df["sq_persondays"]       = df["predicted_persondays"]
    df["opt_persondays"]      = (df["persondays_per_lakh"] * df["optimized_budget"]).round(3)
    df["persondays_gain"]     = (df["opt_persondays"] - df["sq_persondays"]).round(3)
    df["persondays_gain_pct"] = (df["persondays_gain"] / df["sq_persondays"] * 100).round(2)
    return df


def _print_summary(df):
    sq  = df["sq_persondays"].sum()
    opt = df["opt_persondays"].sum()
    gain = opt - sq

    print(f"\n[optimizer-v2] ── Results ───────────────────────────────────────")
    print(f"  budget_change_pct — min: {df['budget_change_pct'].min():.1f}%  "
          f"max: {df['budget_change_pct'].max():.1f}%  "
          f"std: {df['budget_change_pct'].std():.1f}%  "
          f"unique: {df['budget_change_pct'].nunique()}")
    print(f"  Status quo : {sq:>10,.2f} lakh PD")
    print(f"  Optimized  : {opt:>10,.2f} lakh PD")
    print(f"  Net gain   : {gain:>+10,.2f} lakh PD  ({gain/sq*100:+.2f}%)")
    print(f"  Budget     : ₹{df['budget_allocated_lakhs'].sum():,.0f} lakh (unchanged)")
    print(f"[optimizer-v2] ────────────────────────────────────────────────────")

    print("\n[optimizer-v2] Top 5 districts to INCREASE:")
    print(df.nlargest(5, "persondays_gain")[
        ["state","district","budget_allocated_lakhs","optimized_budget","budget_change_pct","persondays_gain"]
    ].to_string(index=False))
    print("\n[optimizer-v2] Top 5 districts to REDUCE:")
    print(df.nsmallest(5, "budget_change")[
        ["state","district","budget_allocated_lakhs","optimized_budget","budget_change_pct","persondays_gain"]
    ].to_string(index=False))


def _plot_allocation_comparison(df, scope):
    show = pd.concat([df.nlargest(10,"budget_change"), df.nsmallest(10,"budget_change")]).drop_duplicates()
    show = show.sort_values("budget_change")
    fig, ax = plt.subplots(figsize=(12, max(7, len(show)*0.4)))
    x = np.arange(len(show)); w = 0.38
    ax.barh(x-w/2, show["budget_allocated_lakhs"].values, height=w, color="#90CAF9", label="Status Quo")
    ax.barh(x+w/2, show["optimized_budget"].values,       height=w, color="#1565C0", label="Optimized")
    ax.set_yticks(x); ax.set_yticklabels(show["district"], fontsize=8)
    ax.set_xlabel("Budget (Rs. lakh)"); ax.set_title(f"Budget Reallocation — {scope}"); ax.legend()
    plt.tight_layout(); _save_fig("08_budget_allocation_comparison.png")


def _plot_efficiency_gain(df, scope):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = df["budget_change"].apply(lambda v: "#2E7D32" if v > 0 else "#C62828")
    ax.scatter(df["persondays_per_lakh"], df["budget_change_pct"], c=colors, alpha=0.55, s=40)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Efficiency (PD per ₹ lakh)"); ax.set_ylabel("Budget Change (%)")
    ax.set_title(f"Efficiency vs Budget Change — {scope}")
    gain = mpatches.Patch(color="#2E7D32", label="Increase"); cut = mpatches.Patch(color="#C62828", label="Decrease")
    ax.legend(handles=[gain, cut]); plt.tight_layout(); _save_fig("09_efficiency_gain_by_district.png")


def _save_results(df):
    cols = ["state","district","budget_allocated_lakhs","optimized_budget",
            "budget_change","budget_change_pct","sq_persondays","opt_persondays",
            "persondays_gain","persondays_gain_pct","persondays_per_lakh"]
    path = os.path.join(OUTPUT_DIR, "optimized_budget_allocation.csv")
    df[cols].sort_values("persondays_gain", ascending=False).to_csv(path, index=False)
    print(f"[optimizer-v2] Saved → {path}")


def _save_fig(filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"[optimizer-v2] Saved: {path}")
