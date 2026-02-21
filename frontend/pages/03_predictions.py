"""Page 3 — Model Predictions."""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API = "http://localhost:8000"
st.set_page_config(page_title="Predictions", page_icon="🤖", layout="wide")
st.title("🤖 Model Predictions")
st.markdown("XGBoost forecasts for person-days per district (test years: 2022–2023)")


@st.cache_data(ttl=300)
def fetch(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params or {}, timeout=5)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Filters ───────────────────────────────────────────────────────────────────
states = fetch("/districts/states") or []
col1, col2, col3 = st.columns(3)
state    = col1.selectbox("State", ["All-India"] + states)
year     = col2.selectbox("Year", [None, 2022, 2023], format_func=lambda x: "All years" if x is None else str(x))
district = col3.text_input("District (optional)")

params = {}
if state != "All-India": params["state"] = state
if year:                  params["year"]  = year
if district:              params["district"] = district

preds = fetch("/predictions/", params)
if not preds:
    st.info("No predictions found for the selected filters.")
    st.stop()

df = pd.DataFrame(preds)

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Districts shown", df["district"].nunique())
c2.metric("Avg actual PD",   f"{df['person_days_lakhs'].mean():.2f}L")
c3.metric("Avg predicted PD",f"{df['predicted_persondays'].mean():.2f}L")
mae = df["prediction_error"].abs().mean()
c4.metric("Mean Abs Error",  f"{mae:.3f}L")

st.markdown("---")

# ── Actual vs Predicted scatter ───────────────────────────────────────────────
col_a, col_b = st.columns([3, 2])

with col_a:
    fig = px.scatter(
        df, x="person_days_lakhs", y="predicted_persondays",
        color="financial_year", hover_data=["state", "district"],
        labels={"person_days_lakhs": "Actual (lakh PD)", "predicted_persondays": "Predicted (lakh PD)"},
        title="Actual vs Predicted Person Days",
        color_discrete_sequence=["#2196F3", "#F44336"]
    )
    lim = [df[["person_days_lakhs","predicted_persondays"]].min().min() * 0.95,
           df[["person_days_lakhs","predicted_persondays"]].max().max() * 1.05]
    fig.add_scatter(x=lim, y=lim, mode="lines",
                    line=dict(color="black", dash="dash"), name="Perfect prediction")
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    # Error distribution
    fig2 = px.histogram(df, x="prediction_error", nbins=40,
                        title="Prediction Error Distribution",
                        labels={"prediction_error": "Error (lakh PD)"},
                        color_discrete_sequence=["#9C27B0"])
    fig2.add_vline(x=0, line_color="black", line_dash="dash")
    fig2.update_layout(height=430)
    st.plotly_chart(fig2, use_container_width=True)

# ── Top over/under predicted ──────────────────────────────────────────────────
st.markdown("---")
col_over, col_under = st.columns(2)

with col_over:
    st.subheader("⚠️ Most Overestimated")
    over = df.nsmallest(8, "prediction_error")[
        ["state", "district", "financial_year", "person_days_lakhs", "predicted_persondays", "prediction_error"]
    ]
    st.dataframe(over, use_container_width=True, hide_index=True)

with col_under:
    st.subheader("⚠️ Most Underestimated")
    under = df.nlargest(8, "prediction_error")[
        ["state", "district", "financial_year", "person_days_lakhs", "predicted_persondays", "prediction_error"]
    ]
    st.dataframe(under, use_container_width=True, hide_index=True)

# ── Full table ────────────────────────────────────────────────────────────────
with st.expander("📋 Full predictions table"):
    st.dataframe(df.sort_values("prediction_error", key=abs, ascending=False),
                 use_container_width=True, hide_index=True)
