"""
Page 1 — Overview Dashboard
---------------------------
Formalized analytical interface for statewide MNREGA trends.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API = "http://localhost:8000"

st.set_page_config(
    page_title="Analytics Overview", 
    page_icon="📊", 
    layout="wide"
)

st.title("Overview")
st.markdown("#### Longitudinal MNREGA Performance Metrics — All India")
st.markdown("---")

@st.cache_data(ttl=300)
def fetch(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params, timeout=5)
        return r.json()
    except Exception as e:
        st.error(f"System Communication Error: {e}")
        return None

# ── KEY PERFORMANCE INDICATORS (KPIs) ─────────────────────────────────────────
stats = fetch("/districts/stats")
if stats:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total States", stats["total_states"])
    with c2:
        st.metric("Districts Surveyed", stats["total_districts"])
    with c3:
        st.metric("Analysis Period", stats["year_range"])
    with c4:
        st.metric("Aggregate Person-Days", f"{stats['total_persondays_lakhs']:,.0f}L")
    with c5:
        st.metric("2020-21 Variance (COVID)", f"{stats['covid_spike_pct']:.1f}%", delta="Historical High")

st.markdown("---")

# ── FILTRATION & SCOPE ────────────────────────────────────────────────────────
# Using a cleaner layout for the filter
f1, f2 = st.columns([1, 2])
with f1:
    states_list = fetch("/districts/states") or []
    scope = st.selectbox("Geographic Scope", ["All-India"] + states_list)
    state_param = None if scope == "All-India" else scope

# ── TREND ANALYSIS ────────────────────────────────────────────────────────────
trend = fetch("/districts/trend", {"state": state_param} if state_param else {})
if trend:
    df_trend = pd.DataFrame(trend)
    fig = go.Figure()
    
    # Primary Axis: Person-Days
    fig.add_bar(
        x=df_trend["financial_year"], 
        y=df_trend["total_persondays"],
        name="Employment (Lakh Person-Days)", 
        marker_color="#264653", 
        opacity=0.8
    )
    
    # Secondary Axis: Expenditure
    fig.add_scatter(
        x=df_trend["financial_year"], 
        y=df_trend["total_expenditure"],
        name="Fiscal Expenditure (Rs. Lakh)", 
        yaxis="y2",
        line=dict(color="#E76F51", width=3), 
        mode="lines+markers"
    )
    
    fig.update_layout(
        title=f"Comparative Performance Trend: {scope}",
        hovermode="x unified",
        yaxis=dict(title="Employment (Lakh Person-Days)", gridcolor='lightgrey'),
        yaxis2=dict(title="Fiscal Expenditure (Rs. Lakh)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=0, r=0, t=80, b=0),
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── DISTRICT RANKING & BENCHMARKING ──────────────────────────────────────────
st.markdown("---")
st.subheader("Regional Performance Benchmarking")

# Formalized selection labels
col_metric, col_n = st.columns([2, 1])
with col_metric:
    metric_map = {
        "person_days_lakhs": "Employment Volume (Lakh Person-Days)",
        "expenditure_lakhs": "Total Expenditure (Rs. Lakh)",
        "expenditure_per_personday": "Cost Efficiency (Expenditure per Person-Day)"
    }
    selected_display = st.selectbox("Ranking Metric", options=list(metric_map.values()))
    # Inverse map to get original key for API
    metric = [k for k, v in metric_map.items() if v == selected_display][0]

with col_n:
    n = st.slider("Data Points (Top N Districts)", 5, 25, 10)

top = fetch("/districts/top", {"state": state_param, "metric": metric, "n": n})
if top:
    df_top = pd.DataFrame(top)
    fig2 = px.bar(
        df_top, 
        x="avg_persondays", 
        y="district",
        orientation="h", 
        color="state",
        title=f"Ranking: Top {n} Districts by {selected_display}",
        labels={"avg_persondays": "Mean Annual Person-Days (Lakh)", "district": "District Authority"},
        color_discrete_sequence=px.colors.qualitative.Prism,
        height=max(400, n * 35)
    )
    fig2.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Average Metric Value"),
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig2, use_container_width=True)
