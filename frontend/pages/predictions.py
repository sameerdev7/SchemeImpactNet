# pages/predictions.py — GBR V3 model predictions and error analysis.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from theme import inject_theme, page_header, section_label, kpi_html, PLOTLY_LAYOUT, SAFFRON, GREEN, RED
from utils.api_client import fetch_states, fetch_districts, fetch_predictions

inject_theme()
page_header(
    "◈ Module 03",
    "Predictions",
    "GBR V3 district-level employment forecasts — walk-forward CV R²≈0.91 (excl. 2022 anomaly)",
)

# ── Filters ───────────────────────────────────────────────────────────────────
states = fetch_states()
if not states:
    st.error("⚠️ API offline — run `uvicorn backend.main:app --port 8000`")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    scope = st.selectbox("State", ["All States"] + states)
with c2:
    state_param = None if scope == "All States" else scope
    districts = ["All Districts"] + fetch_districts(state_param) if state_param else ["All Districts"]
    dist_sel = st.selectbox("District", districts)
with c3:
    df_all = fetch_predictions(state=state_param)
    years  = sorted(df_all["financial_year"].unique().tolist()) if not df_all.empty else []
    yr_sel = st.selectbox("Year", ["All Years"] + years)

# Apply filters
df = fetch_predictions(
    state=state_param,
    district=None if dist_sel == "All Districts" else dist_sel,
    year=None if yr_sel == "All Years" else int(yr_sel),
)

if df.empty:
    st.info("No prediction data for selected filters.")
    st.stop()

# ── Model KPIs ────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

try:
    rmse = np.sqrt(mean_squared_error(df["person_days_lakhs"], df["predicted_persondays"]))
    mae  = mean_absolute_error(df["person_days_lakhs"], df["predicted_persondays"])
    r2   = r2_score(df["person_days_lakhs"], df["predicted_persondays"])
    bias = (df["predicted_persondays"] - df["person_days_lakhs"]).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score",  f"{r2:.4f}")
    c2.metric("RMSE",      f"{rmse:.3f}L")
    c3.metric("MAE",       f"{mae:.3f}L")
    c4.metric("Mean Bias", f"{bias:+.3f}L")
except Exception:
    pass

# ── Model info callout ────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#F0FDF4; border:1px solid #BBF7D0; border-left:3px solid #16A34A;
            border-radius:8px; padding:0.9rem 1.1rem; margin:1rem 0;">
  <p style="font-family:'DM Mono',monospace; font-size:0.56rem; letter-spacing:2px;
            text-transform:uppercase; color:#16A34A; margin:0 0 6px 0;">V3 Leak-Free Model</p>
  <p style="font-family:'Source Serif 4',serif; font-size:0.85rem; color:#14532D;
            line-height:1.65; margin:0;">
    GradientBoostingRegressor · 17 lag-based features · Walk-forward CV
    · R²=0.91 excl. 2022 · Previous R²=0.9963 was data leakage
    (<code>works_completed</code> r=1.0 with target).
    2022 West Bengal reporting anomaly (−93 to −98% drop) is structurally unpredictable.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns(2)

# ── Actual vs Predicted scatter ───────────────────────────────────────────────
with col_left:
    section_label("Actual vs Predicted")

    fig1 = go.Figure()
    lim_mn = min(df["person_days_lakhs"].min(), df["predicted_persondays"].min()) * 0.92
    lim_mx = max(df["person_days_lakhs"].max(), df["predicted_persondays"].max()) * 1.06

    fig1.add_scatter(
        x=[lim_mn, lim_mx], y=[lim_mn, lim_mx],
        mode="lines",
        line=dict(color="#E7E5E4", width=1.5, dash="dot"),
        name="Perfect prediction",
        hoverinfo="skip",
    )
    fig1.add_scatter(
        x=df["person_days_lakhs"],
        y=df["predicted_persondays"],
        mode="markers",
        marker=dict(
            color=df["prediction_error"].abs(),
            colorscale=[[0, SAFFRON], [1, RED]],
            size=5, opacity=0.65,
            colorbar=dict(
                title=dict(text="|Error|L", font=dict(color="#78716C", size=9)),
                tickfont=dict(color="#78716C", size=8),
                thickness=8, len=0.5,
            ),
        ),
        customdata=list(zip(
            df["state"], df["district"],
            df["financial_year"],
            df["person_days_lakhs"].round(2),
            df["predicted_persondays"].round(2),
            df["prediction_error"].round(2),
        )),
        hovertemplate=(
            "<b>%{customdata[1]}</b> · %{customdata[0]}<br>"
            "FY: %{customdata[2]}<br>"
            "Actual: <b>%{customdata[3]}L</b><br>"
            "Predicted: <b>%{customdata[4]}L</b><br>"
            "Error: %{customdata[5]}L"
            "<extra></extra>"
        ),
        name="Districts",
    )

    l1 = {**PLOTLY_LAYOUT}
    l1.update(dict(
        height=370,
        title=dict(text="Actual vs Predicted Person-Days",
                   font=dict(family="Fraunces, serif", size=13, color="#1C1917")),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Actual (Lakh PD)", range=[lim_mn, lim_mx]),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Predicted (Lakh PD)", range=[lim_mn, lim_mx]),
        showlegend=False,
    ))
    fig1.update_layout(**l1)
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

# ── Error distribution ────────────────────────────────────────────────────────
with col_right:
    section_label("Prediction Error Distribution")

    errors = df["prediction_error"]
    fig2 = go.Figure()
    fig2.add_histogram(
        x=errors, nbinsx=40,
        marker=dict(color=SAFFRON, opacity=0.75, line=dict(color="#FFFFFF", width=0.5)),
        hovertemplate="Error: %{x:.2f}L<br>Count: %{y}<extra></extra>",
    )
    fig2.add_vline(x=0, line_dash="dot", line_color="#1C1917", line_width=1.5)
    fig2.add_vline(x=errors.mean(), line_dash="dash", line_color=RED, line_width=1,
                   annotation_text=f"Mean={errors.mean():+.2f}",
                   annotation_font=dict(color=RED, size=9, family="DM Mono, monospace"))

    l2 = {**PLOTLY_LAYOUT}
    l2.update(dict(
        height=370,
        title=dict(text="Error Distribution (Actual − Predicted)",
                   font=dict(family="Fraunces, serif", size=13, color="#1C1917")),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Error (Lakh PD)"),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Count"),
        showlegend=False, bargap=0.05,
    ))
    fig2.update_layout(**l2)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# ── Year-on-year prediction vs actual trend ───────────────────────────────────
section_label("Year-on-Year Prediction Accuracy")

trend = df.groupby("financial_year", as_index=False).agg(
    actual   =("person_days_lakhs",   "sum"),
    predicted=("predicted_persondays", "sum"),
)

fig3 = go.Figure()
fig3.add_bar(
    x=trend["financial_year"], y=trend["actual"],
    name="Actual",
    marker=dict(color="#E7E5E4", opacity=0.9),
)
fig3.add_scatter(
    x=trend["financial_year"], y=trend["predicted"],
    name="Predicted",
    mode="lines+markers",
    line=dict(color=SAFFRON, width=2.5),
    marker=dict(size=7, color=SAFFRON, line=dict(width=1.5, color="#FFFFFF")),
)

l3 = {**PLOTLY_LAYOUT}
l3.update(dict(
    height=300,
    barmode="overlay", bargap=0.35,
    title=dict(text="Aggregated Actual vs Predicted by Year",
               font=dict(family="Fraunces, serif", size=13, color="#1C1917")),
    xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Financial Year", dtick=1),
    yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Total Lakh PD"),
    legend=dict(**PLOTLY_LAYOUT["legend"], orientation="h", y=1.08, x=0),
))

# Annotate known anomalies
if 2020 in trend["financial_year"].values:
    fig3.add_vline(x=2020, line_dash="dot", line_color=RED, line_width=1.5,
                   annotation_text="COVID", annotation_font=dict(color=RED, size=9, family="DM Mono, monospace"))
if 2022 in trend["financial_year"].values:
    fig3.add_vline(x=2022, line_dash="dot", line_color="#A8A29E", line_width=1,
                   annotation_text="WB anomaly", annotation_font=dict(color="#A8A29E", size=9, family="DM Mono, monospace"))

fig3.update_layout(**l3)
st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# ── Walk-forward CV summary ───────────────────────────────────────────────────
section_label("Walk-Forward CV Performance (Honest Evaluation)")

cv_data = {
    "Year": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "R²":   [0.916, 0.926, 0.835, 0.926, 0.510, 0.909, 0.935],
    "MAE":  [6.639, 6.380, 12.681, 7.150, 13.954, 7.403, 5.673],
    "vs Naive R²": ["+0.004", "+0.061", "+0.083", "−0.012", "+0.330", "−0.014", "+0.065"],
    "Note": ["", "", "COVID spike", "", "WB reporting anomaly", "", ""],
}
import pandas as pd
cv_df = pd.DataFrame(cv_data)
st.dataframe(cv_df, use_container_width=True, hide_index=True)
st.caption("Walk-forward CV: model trained on years before test year only. Mean R²=0.851, excl. 2022: R²=0.908.")

st.markdown("---")

# ── Worst predictions table ───────────────────────────────────────────────────
section_label("Largest Prediction Errors")
worst = (
    df.assign(abs_error=df["prediction_error"].abs())
    .nlargest(20, "abs_error")[
        ["state", "district", "financial_year",
         "person_days_lakhs", "predicted_persondays", "prediction_error"]
    ]
    .rename(columns={
        "person_days_lakhs":   "actual_L",
        "predicted_persondays":"predicted_L",
        "prediction_error":    "error_L",
    })
    .round(3)
)
st.dataframe(worst, use_container_width=True, hide_index=True)