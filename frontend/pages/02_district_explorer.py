"""Page 2 — District Explorer."""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API = "http://localhost:8000"
st.set_page_config(page_title="District Explorer", page_icon="🔍", layout="wide")
st.title("🔍 District Explorer")
st.markdown("Drill into any district's full historical performance")


@st.cache_data(ttl=300)
def fetch(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params, timeout=5)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Selectors ─────────────────────────────────────────────────────────────────
states = fetch("/districts/states") or []
col1, col2 = st.columns(2)
state    = col1.selectbox("State", states)
districts = fetch("/districts/list", {"state": state}) or []
district  = col2.selectbox("District", districts)

if not district:
    st.info("Select a district to explore")
    st.stop()

# ── Load history ──────────────────────────────────────────────────────────────
hist = fetch("/districts/history", {"state": state, "district": district})
if not hist:
    st.warning("No data found for this district")
    st.stop()

df = pd.DataFrame(hist)

# ── KPI cards ─────────────────────────────────────────────────────────────────
latest = df.iloc[-1]
prev   = df.iloc[-2] if len(df) > 1 else latest

st.markdown(f"### {district} — {state}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Person Days (latest)",
          f"{latest['person_days_lakhs']:.2f}L",
          f"{latest['person_days_lakhs'] - prev['person_days_lakhs']:+.2f}")
c2.metric("Expenditure (latest)",
          f"Rs.{latest['expenditure_lakhs']:,.0f}L",
          f"{latest['expenditure_lakhs'] - prev['expenditure_lakhs']:+.0f}")
c3.metric("Avg Wage Rate",
          f"Rs.{latest['avg_wage_rate']:.0f}/day")
c4.metric("Demand Fulfillment",
          f"{latest['demand_fulfillment_rate']*100:.1f}%" if latest['demand_fulfillment_rate'] else "N/A")

st.markdown("---")

# ── Person days trend ─────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    fig1 = px.area(df, x="financial_year", y="person_days_lakhs",
                   title="Person Days Over Years",
                   labels={"person_days_lakhs": "Person Days (lakh)", "financial_year": "Year"},
                   color_discrete_sequence=["#2196F3"])
    fig1.add_vline(x=2020, line_dash="dash", line_color="red",
                   annotation_text="COVID spike")
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    fig2 = go.Figure()
    fig2.add_scatter(x=df["financial_year"], y=df["expenditure_lakhs"],
                     mode="lines+markers", name="Expenditure",
                     line=dict(color="#F44336", width=2))
    fig2.add_scatter(x=df["financial_year"], y=df["expenditure_per_personday"],
                     mode="lines+markers", name="Cost per Lakh PD",
                     yaxis="y2", line=dict(color="#FF9800", width=2, dash="dot"))
    fig2.update_layout(
        title="Expenditure & Efficiency",
        yaxis=dict(title="Expenditure (Rs. lakh)"),
        yaxis2=dict(title="Rs./lakh persondays", overlaying="y", side="right"),
        height=380
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("📋 Raw data table"):
    st.dataframe(df, use_container_width=True)
