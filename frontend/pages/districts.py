# pages/districts.py — District deep-dive explorer.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go

from theme import inject_theme, page_header, section_label, PLOTLY_LAYOUT, SAFFRON, GREEN, RED
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

pd_delta = latest['person_days_lakhs'] - prev['person_days_lakhs']
wage_delta = latest['avg_wage_rate'] - prev['avg_wage_rate']

c1, c2, c3 = st.columns(3)
c1.metric(
    "Person-Days (latest yr)",
    f"{latest['person_days_lakhs']:.2f}L",
    delta=f"{pd_delta:+.2f}L",
)
c2.metric(
    "Avg Wage Rate",
    f"₹{latest['avg_wage_rate']:.0f}/day",
    delta=f"₹{wage_delta:+.0f}",
)
c3.metric(
    "Years on Record",
    f"{len(df)}",
)

st.markdown("---")

# ── Person-Days Trend ─────────────────────────────────────────────────────────
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
if 2022 in df["financial_year"].values:
    fig1.add_vline(
        x=2022, line_dash="dot", line_color="#A8A29E", line_width=1,
        annotation_text="2022 anomaly",
        annotation_font=dict(color="#A8A29E", size=9, family="DM Mono, monospace"),
    )
l1 = {**PLOTLY_LAYOUT}
l1.update(dict(
    height=320,
    yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Lakh Person-Days"),
    xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Financial Year", dtick=1),
))
fig1.update_layout(**l1)
st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

# ── YoY Change ────────────────────────────────────────────────────────────────
section_label("Year-on-Year Change")
df["yoy"] = df["person_days_lakhs"].pct_change() * 100

fig2 = go.Figure()
fig2.add_bar(
    x=df["financial_year"],
    y=df["yoy"],
    marker=dict(
        color=[GREEN if v >= 0 else RED for v in df["yoy"].fillna(0)],
        opacity=0.8,
    ),
    hovertemplate="FY%{x}<br>YoY: <b>%{y:+.1f}%</b><extra></extra>",
)
fig2.add_hline(y=0, line_dash="solid", line_color="#1C1917", line_width=1)
l2 = {**PLOTLY_LAYOUT}
l2.update(dict(
    height=220,
    bargap=0.3,
    yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="% Change"),
    xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Financial Year", dtick=1),
))
fig2.update_layout(**l2)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── Wage Rate Trend ───────────────────────────────────────────────────────────
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

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋 Raw Data Table"):
    display_cols = [c for c in [
        "financial_year", "person_days_lakhs", "avg_wage_rate",
    ] if c in df.columns]
    st.dataframe(df[display_cols].round(3), use_container_width=True, hide_index=True)



from utils.ai_summary import render_ai_summary
state_param = state  # state is already defined from the selectbox above
render_ai_summary("districts", state_param=state_param)
