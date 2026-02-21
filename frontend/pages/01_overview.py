"""Page 1 — Overview Dashboard."""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API = "http://localhost:8000"
st.set_page_config(page_title="Overview", page_icon="🌍", layout="wide")
st.title("🌍 Overview Dashboard")
st.markdown("Statewide MNREGA trends across India")


@st.cache_data(ttl=300)
def fetch(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params, timeout=5)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Stats cards ───────────────────────────────────────────────────────────────
stats = fetch("/districts/stats")
if stats:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("States",     stats["total_states"])
    c2.metric("Districts",  stats["total_districts"])
    c3.metric("Years",      stats["year_range"])
    c4.metric("Total Person Days", f"{stats['total_persondays_lakhs']:,.0f}L")
    c5.metric("COVID Spike", f"+{stats['covid_spike_pct']:.1f}%", delta="2020-21")

st.markdown("---")

# ── Scope filter ──────────────────────────────────────────────────────────────
states_list = fetch("/districts/states") or []
scope = st.selectbox("Filter by State (or All-India)", ["All-India"] + states_list)
state_param = None if scope == "All-India" else scope

# ── Trend chart ───────────────────────────────────────────────────────────────
trend = fetch("/districts/trend", {"state": state_param} if state_param else {})
if trend:
    df_trend = pd.DataFrame(trend)
    fig = go.Figure()
    fig.add_bar(x=df_trend["financial_year"], y=df_trend["total_persondays"],
                name="Person Days (lakh)", marker_color="#2196F3", opacity=0.8)
    fig.add_scatter(x=df_trend["financial_year"], y=df_trend["total_expenditure"],
                    name="Expenditure (Rs. lakh)", yaxis="y2",
                    line=dict(color="#F44336", width=3), mode="lines+markers")
    fig.update_layout(
        title=f"MNREGA Trend — {scope}",
        yaxis=dict(title="Person Days (lakh)"),
        yaxis2=dict(title="Expenditure (Rs. lakh)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Top districts ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top Performing Districts")

col_metric, col_n = st.columns([2, 1])
metric = col_metric.selectbox("Rank by", ["person_days_lakhs", "expenditure_lakhs", "expenditure_per_personday"])
n      = col_n.slider("Show top N", 5, 25, 10)

top = fetch("/districts/top", {"state": state_param, "metric": metric, "n": n})
if top:
    df_top = pd.DataFrame(top)
    fig2 = px.bar(
        df_top, x="avg_persondays", y="district",
        orientation="h", color="state",
        title=f"Top {n} Districts by {metric}",
        labels={"avg_persondays": "Avg Person Days (lakh)", "district": ""},
        height=max(350, n * 32)
    )
    fig2.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)
