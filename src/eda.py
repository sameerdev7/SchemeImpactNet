"""
eda.py
------
Exploratory Data Analysis for MNREGA unified dataset.
Automatically adapts to Maharashtra-only or All-India data.

Figures produced:
    01_statewide_trend.png
    02_district_performance_ranking.png
    03_efficiency_ranking.png
    04_covid_impact.png
    05_correlation_heatmap.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# Use a font that supports the rupee symbol if available, else fallback
def _get_font():
    available = [f.name for f in fm.fontManager.ttflist]
    for font in ["DejaVu Sans", "FreeSans", "Liberation Sans", "Arial"]:
        if font in available:
            return font
    return None

FONT = _get_font()
if FONT:
    plt.rcParams["font.family"] = FONT


def run_eda(df: pd.DataFrame, scope: str = "Maharashtra") -> None:
    print(f"\n[eda] Starting EDA — scope: {scope}")
    _summary_stats(df)
    _plot_trend(df, scope)
    _plot_top_bottom_districts(df, scope)
    _plot_efficiency_ranking(df, scope)
    _plot_covid_impact(df)
    _plot_correlation_heatmap(df)
    print(f"[eda] All figures saved to: {FIGURES_DIR}/")


# ── 1. Summary ────────────────────────────────────────────────────────────────

def _summary_stats(df: pd.DataFrame) -> None:
    print(f"\n[eda] {'─'*50}")
    print(f"[eda] Rows            : {len(df)}")
    print(f"[eda] States          : {df['state'].nunique()}")
    print(f"[eda] Districts       : {df['district'].nunique()}")
    print(f"[eda] Years           : {df['financial_year'].min()} – {df['financial_year'].max()}")
    print(f"[eda] Total persondays: {df['person_days_lakhs'].sum():,.1f} lakh")
    print(f"[eda] Total expenditure: Rs. {df['expenditure_lakhs'].sum():,.1f} lakh")

    print(f"\n[eda] Person days by year (state-aggregated mean):")
    by_year = df.groupby("financial_year")["person_days_lakhs"].mean()
    max_val = by_year.max()
    for yr, val in by_year.items():
        bar = "█" * int(val / max_val * 28)
        print(f"      {yr}: {bar} {val:.2f}")
    print(f"[eda] {'─'*50}")


# ── 2. Trend ──────────────────────────────────────────────────────────────────

def _plot_trend(df: pd.DataFrame, scope: str) -> None:
    yearly = df.groupby("financial_year").agg(
        total_persondays=("person_days_lakhs", "sum"),
        total_expenditure=("expenditure_lakhs", "sum")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(yearly["financial_year"], yearly["total_persondays"],
            color="#2196F3", alpha=0.75, label="Person Days (lakh)")
    ax1.set_ylabel("Total Person Days (lakh)", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax1.set_xlabel("Financial Year")

    ax2 = ax1.twinx()
    ax2.plot(yearly["financial_year"], yearly["total_expenditure"],
             color="#F44336", marker="o", linewidth=2.5, label="Expenditure (Rs. lakh)")
    ax2.set_ylabel("Total Expenditure (Rs. lakh)", color="#F44336")
    ax2.tick_params(axis="y", labelcolor="#F44336")

    plt.title(f"MNREGA Trend — {scope} (Person Days & Expenditure)")
    fig.tight_layout()
    _save("01_statewide_trend.png")


# ── 3. District rankings ──────────────────────────────────────────────────────

def _plot_top_bottom_districts(df: pd.DataFrame, scope: str) -> None:
    avg = df.groupby("district")["person_days_lakhs"].mean().sort_values(ascending=False)
    n = min(10, len(avg) // 2)
    top = avg.head(n)
    bot = avg.tail(n).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n * 0.55)))
    axes[0].barh(top.index, top.values, color="#4CAF50")
    axes[0].set_title(f"Top {n} Districts")
    axes[0].set_xlabel("Avg Person Days (lakh)")
    axes[0].invert_yaxis()

    axes[1].barh(bot.index, bot.values, color="#FF7043")
    axes[1].set_title(f"Bottom {n} Districts")
    axes[1].set_xlabel("Avg Person Days (lakh)")
    axes[1].invert_yaxis()

    plt.suptitle(f"MNREGA District Performance — {scope}", fontsize=13)
    plt.tight_layout()
    _save("02_district_performance_ranking.png")

    print(f"\n[eda] Top 5 districts:")
    for d, v in avg.head(5).items():
        print(f"      {d:35s}: {v:.2f} lakh")
    print(f"[eda] Bottom 5 districts:")
    for d, v in avg.tail(5).items():
        print(f"      {d:35s}: {v:.2f} lakh")


# ── 4. Efficiency ranking ─────────────────────────────────────────────────────

def _plot_efficiency_ranking(df: pd.DataFrame, scope: str) -> None:
    eff = (
        df.groupby("district")["expenditure_per_personday"]
        .mean().sort_values().dropna()
    )
    # Show top/bottom 15 only if too many districts
    if len(eff) > 30:
        eff = pd.concat([eff.head(15), eff.tail(15)])

    fig, ax = plt.subplots(figsize=(10, max(6, len(eff) * 0.3)))
    colors = ["#43A047" if v <= eff.median() else "#EF5350" for v in eff.values]
    ax.barh(eff.index, eff.values, color=colors)
    ax.axvline(eff.median(), color="navy", linestyle="--",
               linewidth=1.5, label=f"Median: {eff.median():.1f}")
    ax.set_title(f"Cost Efficiency — {scope}\n(Rs. expenditure per lakh persondays — lower is better)")
    ax.set_xlabel("Rs. lakh per lakh persondays")
    ax.legend()
    plt.tight_layout()
    _save("03_efficiency_ranking.png")

    print(f"\n[eda] Most efficient : {eff.idxmin()} ({eff.min():.1f})")
    print(f"[eda] Least efficient: {eff.idxmax()} ({eff.max():.1f})")


# ── 5. COVID impact ───────────────────────────────────────────────────────────

def _plot_covid_impact(df: pd.DataFrame) -> None:
    pre  = df[df["financial_year"] == 2019].groupby("district")["person_days_lakhs"].mean()
    post = df[df["financial_year"] == 2020].groupby("district")["person_days_lakhs"].mean()
    common = pre.index.intersection(post.index)
    change = ((post[common] - pre[common]) / pre[common] * 100).sort_values(ascending=False)

    # Cap at 20 districts for readability
    show = pd.concat([change.head(10), change.tail(10)]) if len(change) > 20 else change

    fig, ax = plt.subplots(figsize=(10, max(6, len(show) * 0.35)))
    colors = ["#388E3C" if v >= 0 else "#D32F2F" for v in show.values]
    ax.barh(show.index, show.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("COVID Impact: % Change in Person Days\n(2019-20 to 2020-21)")
    ax.set_xlabel("% Change")
    plt.tight_layout()
    _save("04_covid_impact.png")

    print(f"\n[eda] COVID — biggest spike   : {change.idxmax()} (+{change.max():.1f}%)")
    print(f"[eda] COVID — least impacted  : {change.idxmin()} ({change.min():.1f}%)")


# ── 6. Correlation heatmap ────────────────────────────────────────────────────

def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    candidates = [
        "person_days_lakhs", "expenditure_lakhs", "avg_wage_rate",
        "expenditure_per_personday", "lag_person_days", "yoy_growth",
        "demand_fulfillment_rate", "district_avg_persondays",
        "rainfall_mm", "poverty_rate_pct", "scheme_overlap_score",
        "budget_utilization_rate"
    ]
    cols = [c for c in candidates if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    _save("05_correlation_heatmap.png")


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[eda] Saved: {path}")
