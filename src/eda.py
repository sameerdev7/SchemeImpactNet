"""
eda.py
------
Exploratory Data Analysis for MNREGA Maharashtra data.

Answers:
    1. Which districts are top/bottom performers by person_days?
    2. How has Maharashtra trended year-over-year?
    3. Which districts are most/least cost-efficient?
    4. COVID-year (2020-21) impact across districts
    5. Correlation heatmap of all numeric features

All plots saved to reports/figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Clean consistent style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def run_eda(df: pd.DataFrame) -> None:
    print("\n[eda] ── Starting EDA ──────────────────────────────────────────")
    _summary_stats(df)
    _plot_statewide_trend(df)
    _plot_top_bottom_districts(df)
    _plot_efficiency_ranking(df)
    _plot_covid_impact(df)
    _plot_correlation_heatmap(df)
    print(f"[eda] All figures saved to: {FIGURES_DIR}/")
    print("[eda] ── EDA Complete ──────────────────────────────────────────\n")


# ── 1. Summary stats ─────────────────────────────────────────────────────────

def _summary_stats(df: pd.DataFrame) -> None:
    print("\n[eda] Dataset overview:")
    print(f"      Rows         : {len(df)}")
    print(f"      Districts    : {df['district'].nunique()}")
    print(f"      Years        : {sorted(df['financial_year'].unique())}")
    print(f"      Total persondays (lakh): {df['person_days_lakhs'].sum():,.1f}")
    print(f"      Total expenditure (₹ lakh): {df['expenditure_lakhs'].sum():,.1f}")

    by_year = df.groupby("financial_year")["person_days_lakhs"].sum()
    print("\n[eda] State-level person_days_lakhs by year:")
    for yr, val in by_year.items():
        bar = "█" * int(val / by_year.max() * 30)
        print(f"      {yr}: {bar} {val:,.1f}")


# ── 2. Statewide trend ───────────────────────────────────────────────────────

def _plot_statewide_trend(df: pd.DataFrame) -> None:
    yearly = df.groupby("financial_year").agg(
        total_persondays=("person_days_lakhs", "sum"),
        total_expenditure=("expenditure_lakhs", "sum")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "#2196F3"
    ax1.bar(yearly["financial_year"], yearly["total_persondays"],
            color=color1, alpha=0.75, label="Person Days (lakh)")
    ax1.set_xlabel("Financial Year")
    ax1.set_ylabel("Total Person Days (lakh)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#F44336"
    ax2.plot(yearly["financial_year"], yearly["total_expenditure"],
             color=color2, marker="o", linewidth=2.5, label="Expenditure (₹ lakh)")
    ax2.set_ylabel("Total Expenditure (₹ lakh)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Maharashtra MNREGA: Statewide Trend (Person Days & Expenditure)")
    fig.tight_layout()
    _save("01_statewide_trend.png")


# ── 3. Top/Bottom district performers ────────────────────────────────────────

def _plot_top_bottom_districts(df: pd.DataFrame) -> None:
    avg_by_district = (
        df.groupby("district")["person_days_lakhs"]
        .mean()
        .sort_values(ascending=False)
    )
    top10    = avg_by_district.head(10)
    bottom10 = avg_by_district.tail(10).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top 10
    axes[0].barh(top10.index, top10.values, color="#4CAF50")
    axes[0].set_title("Top 10 Districts (Avg Person Days/Year)")
    axes[0].set_xlabel("Avg Person Days (lakh)")
    axes[0].invert_yaxis()

    # Bottom 10
    axes[1].barh(bottom10.index, bottom10.values, color="#FF7043")
    axes[1].set_title("Bottom 10 Districts (Avg Person Days/Year)")
    axes[1].set_xlabel("Avg Person Days (lakh)")
    axes[1].invert_yaxis()

    plt.suptitle("MNREGA District Performance Ranking — Maharashtra", fontsize=13)
    plt.tight_layout()
    _save("02_district_performance_ranking.png")

    print("\n[eda] Top 5 districts by avg person_days_lakhs:")
    for d, v in avg_by_district.head(5).items():
        print(f"      {d:20s}: {v:.2f} lakh")
    print("[eda] Bottom 5 districts:")
    for d, v in avg_by_district.tail(5).items():
        print(f"      {d:20s}: {v:.2f} lakh")


# ── 4. Cost efficiency ranking ───────────────────────────────────────────────

def _plot_efficiency_ranking(df: pd.DataFrame) -> None:
    """expenditure_per_personday: lower = more efficient."""
    eff = (
        df.groupby("district")["expenditure_per_personday"]
        .mean()
        .sort_values()
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#43A047" if v <= eff.median() else "#EF5350" for v in eff.values]
    ax.barh(eff.index, eff.values, color=colors)
    ax.axvline(eff.median(), color="navy", linestyle="--",
               linewidth=1.5, label=f"Median: {eff.median():.2f}")
    ax.set_title("District Cost Efficiency\n(Expenditure per Lakh Persondays — lower is better)")
    ax.set_xlabel("₹ lakh per lakh persondays")
    ax.legend()
    plt.tight_layout()
    _save("03_efficiency_ranking.png")

    print(f"\n[eda] Most efficient district : {eff.idxmin()} ({eff.min():.2f})")
    print(f"[eda] Least efficient district: {eff.idxmax()} ({eff.max():.2f})")


# ── 5. COVID impact ──────────────────────────────────────────────────────────

def _plot_covid_impact(df: pd.DataFrame) -> None:
    """Compare 2019 vs 2020 person_days per district — shows COVID spike."""
    pre  = df[df["financial_year"] == 2019].set_index("district")["person_days_lakhs"]
    post = df[df["financial_year"] == 2020].set_index("district")["person_days_lakhs"]
    common = pre.index.intersection(post.index)

    change = ((post[common] - pre[common]) / pre[common] * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#388E3C" if v >= 0 else "#D32F2F" for v in change.values]
    ax.barh(change.index, change.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("COVID Impact: % Change in Person Days (2019-20 → 2020-21)")
    ax.set_xlabel("% Change")
    plt.tight_layout()
    _save("04_covid_impact.png")

    print(f"\n[eda] COVID spike — biggest increase : {change.idxmax()} (+{change.max():.1f}%)")
    print(f"[eda] COVID spike — least impacted   : {change.idxmin()} ({change.min():.1f}%)")


# ── 6. Correlation heatmap ───────────────────────────────────────────────────

def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_cols = [
        "person_days_lakhs", "expenditure_lakhs", "avg_wage_rate",
        "expenditure_per_personday", "lag_person_days", "yoy_growth",
        "demand_fulfillment_rate", "district_avg_persondays"
    ]
    cols = [c for c in num_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    _save("05_correlation_heatmap.png")


# ── Helper ───────────────────────────────────────────────────────────────────

def _save(filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[eda] Saved: {path}")
