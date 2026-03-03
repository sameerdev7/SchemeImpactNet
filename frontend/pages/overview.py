# pages/overview.py — National MNREGA trend overview.


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from theme import inject_theme, page_header, section_label, PLOTLY_LAYOUT, SAFFRON, GREEN, RED
from utils.api_client import fetch_stats, fetch_states, fetch_yearly_trend, fetch_top_districts

inject_theme()
page_header(
    "◈ Module 01",
    "Overview",
    "Longitudinal MNREGA performance across India — employment, expenditure, and efficiency trends",
)

# ── Stats KPIs ────────────────────────────────────────────────────────────────
stats = fetch_stats()
if stats:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("States",         stats.get("total_states", "—"))
    c2.metric("Districts",      stats.get("total_districts", "—"))
    c3.metric("Period",         stats.get("year_range", "—"))
    c4.metric("Total PD",       f"{stats.get('total_persondays_lakhs', 0):,.0f}L")
    c5.metric("COVID Spike",    f"{stats.get('covid_spike_pct', 0):.1f}%", delta="2020 peak")
else:
    st.warning("⚠️ Backend offline — run `uvicorn backend.main:app --port 8000`")
    st.stop()

st.markdown("---")

# ── Scope selector ────────────────────────────────────────────────────────────
states_list = fetch_states()
col_sel, _ = st.columns([1, 2])
with col_sel:
    scope = st.selectbox("Geographic Scope", ["All-India"] + states_list)
state_param = None if scope == "All-India" else scope

# ── Trend chart ───────────────────────────────────────────────────────────────
section_label("Employment & Expenditure Trend")
df_trend = fetch_yearly_trend(state_param)

if not df_trend.empty:
    fig = go.Figure()

    # Bars for person-days
    fig.add_bar(
        x=df_trend["financial_year"],
        y=df_trend["total_persondays"],
        name="Person-Days (lakh)",
        marker=dict(color=SAFFRON, opacity=0.78),
    )

    # Line for expenditure on secondary axis
    fig.add_scatter(
        x=df_trend["financial_year"],
        y=df_trend["total_expenditure"],
        name="Expenditure (Rs. lakh)",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#1C1917", width=2.5),
        marker=dict(size=6, color="#1C1917"),
    )

    if 2020 in df_trend["financial_year"].values:
        fig.add_vline(
            x=2020, line_dash="dot", line_color=RED, line_width=1.5,
            annotation_text="COVID-19",
            annotation_font=dict(color=RED, size=9, family="DM Mono, monospace"),
            annotation_position="top right",
        )

    layout = {**PLOTLY_LAYOUT}
    layout.update(dict(
        title=dict(
            text=f"MNREGA National Trend — {scope}",
            font=dict(family="Fraunces, serif", size=15, color="#1C1917"),
        ),
        hovermode="x unified",
        height=420,
        bargap=0.35,
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Person-Days (lakh)"),
        yaxis2=dict(
            title="Expenditure (Rs. lakh)", overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#78716C", size=10),
            title_font=dict(color="#57534E", size=11),
        ),
        legend=dict(**PLOTLY_LAYOUT["legend"], orientation="h", y=1.08, x=0),
    ))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Source: MNREGA MIS · Ministry of Rural Development · Annual district-level aggregates")
else:
    st.info("No trend data — API offline or pipeline not yet run.")

st.markdown("---")

# ── District ranking ──────────────────────────────────────────────────────────
section_label("District Performance Benchmarking")

cm, cn = st.columns([2, 1])
with cm:
    metric_options = {
        "Employment Volume (Lakh Person-Days)":        "person_days_lakhs",
        "Total Expenditure (Rs. Lakh)":                "expenditure_lakhs",
        "Cost per Person-Day (Rs. Lakh / Lakh PD)":   "expenditure_per_personday",
    }
    sel = st.selectbox("Ranking Metric", list(metric_options.keys()))
    metric = metric_options[sel]
with cn:
    n_top = st.slider("Top N Districts", 5, 30, 15)

df_top = fetch_top_districts(state_param, metric, n_top)

if not df_top.empty:
    # Determine which column to plot
    col_map = {
        "person_days_lakhs":         "avg_persondays",
        "expenditure_lakhs":         "avg_expenditure",
        "expenditure_per_personday": "avg_efficiency",
    }
    plot_col = col_map.get(metric, "avg_persondays")
    if plot_col not in df_top.columns:
        plot_col = df_top.select_dtypes("number").columns[0]

    df_top["label"] = df_top["district"] + " · " + df_top["state"]

    fig2 = go.Figure()
    fig2.add_bar(
        x=df_top[plot_col],
        y=df_top["label"],
        orientation="h",
        marker=dict(
            color=df_top[plot_col],
            colorscale=[[0, "#FED7AA"], [1, "#9A3412"]],
            showscale=False,
        ),
        customdata=list(zip(
            df_top["state"],
            df_top["district"],
            df_top[plot_col].round(2),
        )),
        hovertemplate=(
            "<b>%{customdata[1]}</b> · %{customdata[0]}<br>"
            f"{sel}: <b>%{{customdata[2]}}</b><extra></extra>"
        ),
    )
    layout2 = {**PLOTLY_LAYOUT}
    layout2.update(dict(
        title=dict(
            text=f"Top {n_top} Districts — {sel}",
            font=dict(family="Fraunces, serif", size=14, color="#1C1917"),
        ),
        height=max(380, n_top * 30),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title=sel),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], autorange="reversed"),
    ))
    fig2.update_layout(**layout2)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("No ranking data available.")
