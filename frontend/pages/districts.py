# pages/districts.py — District deep-dive explorer.


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go

from theme import inject_theme, page_header, section_label, PLOTLY_LAYOUT, SAFFRON, GREEN, RED, AMBER
from utils.api_client import fetch_states, fetch_districts, fetch_district_history

inject_theme()
page_header(
    "◈ Module 02",
    "District Explorer",
    "Full historical MNREGA performance deep-dive for any district",
)

# ── Selectors ─────────────────────────────────────────────────────────────────
states = fetch_states()
if not states:
    st.error("⚠️ API offline — run `uvicorn backend.main:app --port 8000`")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("State", states)
with col2:
    districts = fetch_districts(state)
    if not districts:
        st.warning("No districts found for this state.")
        st.stop()
    district = st.selectbox("District", districts)

# ── Fetch district history ────────────────────────────────────────────────────
df = fetch_district_history(state, district)

if df.empty:
    st.warning("No historical data for this district.")
    st.stop()

df = df.sort_values("financial_year").reset_index(drop=True)

# ── District headline ─────────────────────────────────────────────────────────
latest = df.iloc[-1]
prev   = df.iloc[-2] if len(df) > 1 else latest

st.markdown(f"""
<div style="margin:0.5rem 0 1.5rem;">
  <p style="font-family:'Fraunces',serif; font-size:1.65rem; font-weight:600;
            color:#1C1917; margin:0;">
    {district}
    <span style="font-size:1rem; font-weight:300; color:#78716C;">· {state}</span>
  </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Person-Days (latest)",
    f"{latest['person_days_lakhs']:.2f}L",
    delta=f"{latest['person_days_lakhs'] - prev['person_days_lakhs']:+.2f}",
)
c2.metric(
    "Expenditure",
    f"₹{latest['expenditure_lakhs']:,.0f}L",
    delta=f"{latest['expenditure_lakhs'] - prev['expenditure_lakhs']:+.0f}",
)
c3.metric("Avg Wage Rate", f"₹{latest['avg_wage_rate']:.0f}/day")

dfr = latest.get("demand_fulfillment_rate")
if dfr and dfr == dfr:  # not NaN
    c4.metric("Demand Fulfillment", f"{dfr * 100:.1f}%")
else:
    c4.metric("Cost Efficiency", f"₹{latest['expenditure_per_personday']:.1f}L/LPD")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    section_label("Person-Days Trend")
    fig1 = go.Figure()
    fig1.add_scatter(
        x=df["financial_year"], y=df["person_days_lakhs"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(251,146,60,0.07)",
        line=dict(color=SAFFRON, width=2.5),
        marker=dict(size=6, color=SAFFRON, line=dict(width=1.5, color="#FFFFFF")),
        name="Person-Days",
        hovertemplate="FY%{x}<br>PD: <b>%{y:.2f}L</b><extra></extra>",
    )
    if 2020 in df["financial_year"].values:
        fig1.add_vline(
            x=2020, line_dash="dot", line_color=RED, line_width=1.5,
            annotation_text="COVID",
            annotation_font=dict(color=RED, size=9, family="DM Mono, monospace"),
        )
    l1 = {**PLOTLY_LAYOUT}
    l1.update(dict(
        height=300,
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Lakh PD"),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Financial Year", dtick=1),
    ))
    fig1.update_layout(**l1)
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with col_b:
    section_label("Expenditure & Cost Efficiency")
    fig2 = go.Figure()
    fig2.add_scatter(
        x=df["financial_year"], y=df["expenditure_lakhs"],
        mode="lines+markers",
        name="Expenditure (Rs. lakh)",
        line=dict(color="#1C1917", width=2),
        marker=dict(size=5, color="#1C1917"),
        hovertemplate="FY%{x}<br>₹%{y:,.0f}L<extra></extra>",
    )
    if "expenditure_per_personday" in df.columns:
        fig2.add_scatter(
            x=df["financial_year"], y=df["expenditure_per_personday"],
            mode="lines+markers",
            name="Cost per Lakh PD",
            yaxis="y2",
            line=dict(color=AMBER, width=2, dash="dot"),
            marker=dict(size=5, color=AMBER),
            hovertemplate="FY%{x}<br>Efficiency: %{y:.2f}<extra></extra>",
        )
    l2 = {**PLOTLY_LAYOUT}
    l2.update(dict(
        height=300,
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Expenditure (Rs. lakh)"),
        yaxis2=dict(
            title="Cost per Lakh PD", overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#78716C", size=10),
            title_font=dict(color="#57534E", size=11),
        ),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Financial Year", dtick=1),
        legend=dict(**PLOTLY_LAYOUT["legend"]),
    ))
    fig2.update_layout(**l2)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── Wage trend ────────────────────────────────────────────────────────────────
if "avg_wage_rate" in df.columns:
    section_label("Wage Rate History")
    fig3 = go.Figure()
    fig3.add_scatter(
        x=df["financial_year"], y=df["avg_wage_rate"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(22,163,74,0.06)",
        line=dict(color=GREEN, width=2),
        marker=dict(size=6, color=GREEN),
        hovertemplate="FY%{x}<br>₹%{y:.0f}/day<extra></extra>",
    )
    l3 = {**PLOTLY_LAYOUT}
    l3.update(dict(
        height=220,
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="₹/day"),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Financial Year", dtick=1),
    ))
    fig3.update_layout(**l3)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

# ── Demand vs availed ─────────────────────────────────────────────────────────
if "demand_fulfillment_rate" in df.columns and df["demand_fulfillment_rate"].notna().any():
    section_label("Demand Fulfillment Rate")
    fig4 = go.Figure()
    fig4.add_bar(
        x=df["financial_year"],
        y=(df["demand_fulfillment_rate"] * 100).round(1),
        marker=dict(
            color=(df["demand_fulfillment_rate"] * 100),
            colorscale=[[0, "#FEE2E2"], [0.5, "#FB923C"], [1, "#16A34A"]],
            showscale=False,
        ),
        hovertemplate="FY%{x}<br>Fulfillment: <b>%{y:.1f}%</b><extra></extra>",
    )
    fig4.add_hline(y=100, line_dash="dot", line_color="#1C1917", line_width=1)
    l4 = {**PLOTLY_LAYOUT}
    l4.update(dict(
        height=200,
        bargap=0.3,
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="%", range=[0, 115]),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Financial Year", dtick=1),
    ))
    fig4.update_layout(**l4)
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋 Raw Data Table"):
    display_cols = [
        c for c in [
            "financial_year", "person_days_lakhs", "expenditure_lakhs",
            "avg_wage_rate", "expenditure_per_personday", "demand_fulfillment_rate",
        ] if c in df.columns
    ]
    st.dataframe(df[display_cols].round(3), use_container_width=True, hide_index=True)
