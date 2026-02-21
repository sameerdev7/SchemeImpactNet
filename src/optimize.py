"""
optimize.py
-----------
Stage 3: Budget Allocation Optimizer for MNREGA districts.

Problem it solves:
    Given a fixed total budget and predicted person_days per rupee
    for each district — how should we allocate funds across districts
    to MAXIMIZE total employment generated?

This is a linear programming problem:
    Maximize  : sum(persondays_per_rupee[i] * budget[i])
    Subject to: sum(budget[i]) <= total_budget
                min_budget[i] <= budget[i] <= max_budget[i]  for all i

Two strategies compared:
    1. Status Quo    : current allocation from the data
    2. Optimized     : LP-optimal reallocation

Uses scipy.optimize.linprog (no extra install needed).
Switch to PuLP by setting BACKEND = "pulp" after: pip install pulp

Outputs:
    - data/processed/optimized_budget_allocation.csv
    - reports/figures/08_budget_allocation_comparison.png
    - reports/figures/09_efficiency_gain_by_district.png
    - Printed summary of total employment gain
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.optimize import linprog

FIGURES_DIR = os.path.join("reports", "figures")
OUTPUT_DIR  = os.path.join("data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BACKEND = "scipy"   # "scipy" | "pulp"

# Budget constraints per district (as fraction of average budget)
MIN_BUDGET_FRACTION = 0.40   # no district gets less than 40% of its current budget
MAX_BUDGET_FRACTION = 2.50   # no district gets more than 2.5x its current budget


def run_optimizer(
    predictions_path: str = "data/processed/mnrega_predictions.csv",
    raw_path: str = "data/raw/mnrega_india_unified.csv",
    scope_state: str = None,       # None = All-India, "Maharashtra" = state only
    total_budget_override: float = None,   # override total budget (Rs. lakh)
    target_year: int = 2023,
) -> pd.DataFrame:
    """
    Run budget allocation optimization.

    Args:
        predictions_path     : Path to model predictions CSV.
        raw_path             : Path to raw unified CSV (for budget data).
        scope_state          : Filter to one state, or None for All-India.
        total_budget_override: Override total budget in Rs. lakh.
        target_year          : Year to optimize for.

    Returns:
        DataFrame with status_quo and optimized budget per district.
    """
    print("\n[optimizer] ── Budget Allocation Optimizer ─────────────────")

    df = _prepare_data(predictions_path, raw_path, scope_state, target_year)
    result = _optimize(df, total_budget_override)
    _print_summary(result)
    _plot_allocation_comparison(result, scope_state or "All-India")
    _plot_efficiency_gain(result, scope_state or "All-India")
    _save_results(result)

    print("[optimizer] ── Optimization Complete ───────────────────────\n")
    return result


# ── Data preparation ──────────────────────────────────────────────────────────

def _prepare_data(
    predictions_path: str,
    raw_path: str,
    scope_state: str,
    target_year: int
) -> pd.DataFrame:
    """
    Merge predictions with budget data to build the optimizer input table.

    Each row = one district with:
        predicted_persondays   : model forecast for target_year
        current_budget         : budget_allocated_lakhs from raw data
        persondays_per_lakh    : efficiency = predicted_persondays / current_budget
    """
    # Load predictions
    preds = pd.read_csv(predictions_path)
    preds = preds[preds["financial_year"] == target_year].copy()

    # Load budget from raw
    raw = pd.read_csv(raw_path)
    raw["financial_year"] = raw["financial_year"].apply(
        lambda v: int(str(v).split("-")[0])
    )
    budget = raw[raw["financial_year"] == target_year][
        ["state", "district", "budget_allocated_lakhs", "expenditure_lakhs"]
    ].copy()

    df = preds.merge(budget, on=["state", "district"], how="inner")

    if scope_state:
        df = df[df["state"] == scope_state].reset_index(drop=True)
        print(f"[optimizer] Scope: {scope_state} | Districts: {len(df)}")
    else:
        print(f"[optimizer] Scope: All-India | Districts: {len(df)}")

    # Efficiency: predicted persondays per Rs. lakh of budget
    safe_budget = df["budget_allocated_lakhs"].replace(0, np.nan)
    df["persondays_per_lakh"] = (df["predicted_persondays"] / safe_budget).fillna(0)

    print(f"[optimizer] Target year: {target_year}")
    print(f"[optimizer] Total current budget: Rs. {df['budget_allocated_lakhs'].sum():,.0f} lakh")
    print(f"[optimizer] Total predicted persondays (status quo): {df['predicted_persondays'].sum():,.1f} lakh")

    return df


# ── Optimizer ─────────────────────────────────────────────────────────────────

def _optimize(df: pd.DataFrame, total_budget_override: float = None) -> pd.DataFrame:
    """
    Linear program:
        Maximize  : c @ x          (total persondays)
        Subject to: sum(x) <= B    (total budget constraint)
                    lb[i] <= x[i] <= ub[i]  (per-district bounds)

    scipy.linprog minimizes, so we minimize -c @ x.
    """
    n = len(df)
    current_budgets = df["budget_allocated_lakhs"].values
    efficiency      = df["persondays_per_lakh"].values

    total_budget = total_budget_override or current_budgets.sum()

    lb = current_budgets * MIN_BUDGET_FRACTION
    ub = current_budgets * MAX_BUDGET_FRACTION

    # Objective: minimize -efficiency (= maximize efficiency)
    c = -efficiency

    # Constraint: sum of all budgets <= total_budget
    A_ub = np.ones((1, n))
    b_ub = np.array([total_budget])

    bounds = list(zip(lb, ub))

    if BACKEND == "pulp":
        result_x = _solve_pulp(c, A_ub, b_ub, bounds, n)
    else:
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs"
        )
        if not res.success:
            raise RuntimeError(f"[optimizer] LP failed: {res.message}")
        result_x = res.x

    df = df.copy()
    df["optimized_budget"]     = result_x.round(2)
    df["budget_change"]        = df["optimized_budget"] - df["budget_allocated_lakhs"]
    df["budget_change_pct"]    = (df["budget_change"] / df["budget_allocated_lakhs"] * 100).round(2)

    df["sq_persondays"]        = df["predicted_persondays"]
    df["opt_persondays"]       = (df["persondays_per_lakh"] * df["optimized_budget"]).round(3)
    df["persondays_gain"]      = (df["opt_persondays"] - df["sq_persondays"]).round(3)
    df["persondays_gain_pct"]  = (df["persondays_gain"] / df["sq_persondays"] * 100).round(2)

    return df


def _solve_pulp(c, A_ub, b_ub, bounds, n):
    """PuLP backend — used when pip install pulp is available."""
    import pulp
    prob  = pulp.LpProblem("MNREGA_Budget_Optimizer", pulp.LpMaximize)
    x     = [pulp.LpVariable(f"x_{i}", lowBound=bounds[i][0], upBound=bounds[i][1]) for i in range(n)]
    prob += pulp.lpSum(-c[i] * x[i] for i in range(n))
    prob += pulp.lpSum(x[i] for i in range(n)) <= b_ub[0]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return np.array([pulp.value(xi) for xi in x])


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    sq_total  = df["sq_persondays"].sum()
    opt_total = df["opt_persondays"].sum()
    gain      = opt_total - sq_total
    gain_pct  = gain / sq_total * 100

    print(f"\n[optimizer] ── Results ──────────────────────────────────────")
    print(f"[optimizer]   Status quo persondays : {sq_total:>10,.2f} lakh")
    print(f"[optimizer]   Optimized persondays  : {opt_total:>10,.2f} lakh")
    print(f"[optimizer]   Net gain              : {gain:>+10,.2f} lakh (+{gain_pct:.2f}%)")
    print(f"[optimizer]   Same total budget     : Rs. {df['budget_allocated_lakhs'].sum():,.0f} lakh")
    print(f"[optimizer] ────────────────────────────────────────────────────")

    top_gainers = df.nlargest(5, "persondays_gain")[
        ["state", "district", "budget_allocated_lakhs",
         "optimized_budget", "budget_change_pct", "persondays_gain"]
    ]
    print("\n[optimizer] Top 5 districts to INCREASE budget:")
    print(top_gainers.to_string(index=False))

    top_cutters = df.nsmallest(5, "budget_change")[
        ["state", "district", "budget_allocated_lakhs",
         "optimized_budget", "budget_change_pct", "persondays_gain"]
    ]
    print("\n[optimizer] Top 5 districts to REDUCE budget (lower efficiency):")
    print(top_cutters.to_string(index=False))


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_allocation_comparison(df: pd.DataFrame, scope: str) -> None:
    """
    Bar chart: Status quo vs optimized budget for top/bottom 20 districts
    by budget change.
    """
    show = pd.concat([
        df.nlargest(10, "budget_change"),
        df.nsmallest(10, "budget_change")
    ]).drop_duplicates()

    show = show.sort_values("budget_change", ascending=True)
    labels = show["district"].str.replace("_District_", " D", regex=False)

    fig, ax = plt.subplots(figsize=(12, max(7, len(show) * 0.4)))

    x = np.arange(len(show))
    w = 0.38
    ax.barh(x - w/2, show["budget_allocated_lakhs"].values,
            height=w, color="#90CAF9", label="Status Quo Budget")
    ax.barh(x + w/2, show["optimized_budget"].values,
            height=w, color="#1565C0", label="Optimized Budget")

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Budget (Rs. lakh)")
    ax.set_title(f"Budget Reallocation — {scope}\n(Top gainers & losers)")
    ax.legend()
    plt.tight_layout()
    _save_fig("08_budget_allocation_comparison.png")


def _plot_efficiency_gain(df: pd.DataFrame, scope: str) -> None:
    """
    Scatter: efficiency (persondays per lakh) vs budget change %.
    Shows why high-efficiency districts get more budget.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = df["budget_change"].apply(lambda v: "#2E7D32" if v > 0 else "#C62828")

    ax.scatter(
        df["persondays_per_lakh"],
        df["budget_change_pct"],
        c=colors, alpha=0.65, s=55
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Efficiency (predicted persondays per Rs. lakh)")
    ax.set_ylabel("Budget Change (%)")
    ax.set_title(f"Efficiency vs Budget Change — {scope}\n"
                 f"(Green = budget increase, Red = budget decrease)")

    gain_patch = mpatches.Patch(color="#2E7D32", label="Budget increase")
    cut_patch  = mpatches.Patch(color="#C62828", label="Budget decrease")
    ax.legend(handles=[gain_patch, cut_patch])

    plt.tight_layout()
    _save_fig("09_efficiency_gain_by_district.png")


# ── Save ──────────────────────────────────────────────────────────────────────

def _save_results(df: pd.DataFrame) -> None:
    cols = [
        "state", "district",
        "budget_allocated_lakhs", "optimized_budget",
        "budget_change", "budget_change_pct",
        "sq_persondays", "opt_persondays",
        "persondays_gain", "persondays_gain_pct",
        "persondays_per_lakh"
    ]
    out = df[cols].sort_values("persondays_gain", ascending=False)
    path = os.path.join(OUTPUT_DIR, "optimized_budget_allocation.csv")
    out.to_csv(path, index=False)
    print(f"[optimizer] Saved → {path}")


def _save_fig(filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[optimizer] Saved: {path}")
