# pages/overview.py — National MNREGA trend overview.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go

from theme import inject_theme, page_header, section_label, PLOTLY_LAYOUT, SAFFRON, GREEN, RED
from utils.api_client import fetch_stats, fetch_states, fetch_yearly_trend, fetch_top_districts

inject_theme()
page_header(
    "◈ Module 01",
    "Overview",
    "Longitudinal MNREGA performance across India — employment and wage trends",
)

# ── Stats KPIs ────────────────────────────────────────────────────────────────
stats = fetch_stats()
if stats:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("States",      stats.get("total_states", "—"))
    c2.metric("Districts",   stats.get("total_districts", "—"))
    c3.metric("Period",      stats.get("year_range", "—"))
    c4.metric("Total PD",    f"{stats.get('total_persondays_lakhs', 0):,.0f}L")
    c5.metric("COVID Spike", f"{stats.get('covid_spike_pct', 0):.1f}%", delta="2020 peak")
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
section_label("Employment Trend")
df_trend = fetch_yearly_trend(state_param)

if not df_trend.empty:
    fig = go.Figure()

    fig.add_bar(
        x=df_trend["financial_year"],
        y=df_trend["total_persondays"],
        name="Person-Days (lakh)",
        marker=dict(color=SAFFRON, opacity=0.78),
    )

    # Wage on secondary axis if available
    if "avg_wage" in df_trend.columns:
        fig.add_scatter(
            x=df_trend["financial_year"],
            y=df_trend["avg_wage"],
            name="Avg Wage Rate (₹/day)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=6, color=GREEN),
        )

    if 2020 in df_trend["financial_year"].values:
        fig.add_vline(
            x=2020, line_dash="dot", line_color=RED, line_width=1.5,
            annotation_text="COVID-19",
            annotation_font=dict(color=RED, size=9, family="DM Mono, monospace"),
            annotation_position="top right",
        )
    if 2022 in df_trend["financial_year"].values:
        fig.add_vline(
            x=2022, line_dash="dot", line_color="#A8A29E", line_width=1,
            annotation_text="WB anomaly",
            annotation_font=dict(color="#A8A29E", size=9, family="DM Mono, monospace"),
            annotation_position="top left",
        )

    layout = {**PLOTLY_LAYOUT}
    layout.update(dict(
        title=dict(
            text=f"MNREGA Employment Trend — {scope}",
            font=dict(family="Fraunces, serif", size=15, color="#1C1917"),
        ),
        hovermode="x unified",
        height=420,
        bargap=0.35,
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Person-Days (lakh)"),
        yaxis2=dict(
            title="Avg Wage Rate (₹/day)", overlaying="y", side="right",
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
    # V3: only person_days_lakhs is a real non-synthetic column
    metric = "person_days_lakhs"
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace; font-size:0.65rem; '
        'letter-spacing:1.5px; text-transform:uppercase; color:#78716C; margin-bottom:4px;">'
        'Ranking Metric</p>'
        '<p style="font-size:0.9rem; color:#1C1917; margin:0;">Employment Volume (Lakh Person-Days)</p>',
        unsafe_allow_html=True
    )
with cn:
    n_top = st.slider("Top N Districts", 5, 30, 15)

df_top = fetch_top_districts(state_param, metric, n_top)

if not df_top.empty:
    df_top["label"] = df_top["district"] + " · " + df_top["state"]

    fig2 = go.Figure()
    fig2.add_bar(
        x=df_top["avg_persondays"],
        y=df_top["label"],
        orientation="h",
        marker=dict(
            color=df_top["avg_persondays"],
            colorscale=[[0, "#FED7AA"], [1, "#9A3412"]],
            showscale=False,
        ),
        customdata=list(zip(df_top["state"], df_top["district"], df_top["avg_persondays"].round(2))),
        hovertemplate=(
            "<b>%{customdata[1]}</b> · %{customdata[0]}<br>"
            "Avg Person-Days: <b>%{customdata[2]}L</b><extra></extra>"
        ),
    )
    layout2 = {**PLOTLY_LAYOUT}
    layout2.update(dict(
        title=dict(
            text=f"Top {n_top} Districts — Employment Volume",
            font=dict(family="Fraunces, serif", size=14, color="#1C1917"),
        ),
        height=max(380, n_top * 30),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Avg Lakh Person-Days"),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], autorange="reversed"),
    ))
    fig2.update_layout(**layout2)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("No ranking data available.")


from utils.ai_summary import render_ai_summary 
render_ai_summary("overview", state_param=state_param)
