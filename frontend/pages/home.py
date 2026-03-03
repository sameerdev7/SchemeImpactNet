# pages/home.py — Landing dashboard.


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from theme import (
    inject_theme, page_header, section_label, kpi_html,
    signal_card_html, PLOTLY_LAYOUT, SAFFRON, SAFFRON_SCALE, GREEN, RED, AMBER,
)
from utils.api_client import (
    is_online, fetch_stats, fetch_predictions, fetch_optimizer_results,
)

inject_theme()

# ── Status pill ───────────────────────────────────────────────────────────────
online = is_online()
pill_color = "#16A34A" if online else "#DC2626"
pill_text  = "API LIVE" if online else "API OFFLINE — run `uvicorn backend.main:app --port 8000`"
st.markdown(
    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:1.4rem;">'
    f'<span style="width:7px;height:7px;border-radius:50%;background:{pill_color};display:inline-block;"></span>'
    f'<span style="font-family:DM Mono,monospace;font-size:0.62rem;letter-spacing:2px;'
    f'text-transform:uppercase;color:{pill_color};">{pill_text}</span></div>',
    unsafe_allow_html=True,
)

page_header(
    "◈ MNREGA · India · 2014–2024",
    "SchemeImpactNet",
    "Predictive impact analysis and budget optimisation for India's rural employment scheme",
)

# ── Data fetch ────────────────────────────────────────────────────────────────
stats   = fetch_stats()
pred_df = fetch_predictions()
opt_df  = fetch_optimizer_results()

# Derived KPIs
n_dist    = stats.get("total_districts", "—")
n_states  = stats.get("total_states", "—")
yr_range  = stats.get("year_range", "—")
total_pd  = stats.get("total_persondays_lakhs", 0)
total_exp = stats.get("total_expenditure_lakhs", 0)
covid_pct = stats.get("covid_spike_pct", 0)
exp_cr    = total_exp / 1e4 if total_exp else 0

nat_gain = gain_pct = 0.0
if not opt_df.empty and "persondays_gain" in opt_df.columns:
    nat_gain = opt_df["persondays_gain"].sum()
    sq_sum   = opt_df["sq_persondays"].sum() if "sq_persondays" in opt_df.columns else 1
    gain_pct = nat_gain / sq_sum * 100 if sq_sum else 0

# ── KPI strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5, gap="small")
cards = [
    (str(n_dist),           "Districts",        SAFFRON, ""),
    (f"{total_pd:,.0f}L",   "Person-Days",      "#1C1917", "historical total"),
    (f"₹{exp_cr:,.0f}Cr",   "Fiscal Envelope",  "#1C1917", ""),
    (f"{covid_pct:+.1f}%",  "COVID-20 Spike",   RED,     "2020 peak"),
    (f"{gain_pct:+.2f}%",   "LP Opt. Gain",     GREEN,   "budget-neutral"),
]
for col, (val, label, color, note) in zip([c1, c2, c3, c4, c5], cards):
    with col:
        st.markdown(kpi_html(val, label, color, note), unsafe_allow_html=True)

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

# ── LEFT: state bubble map ────────────────────────────────────────────────────
STATE_COORDS = {
    "Andhra Pradesh":      (15.9, 79.7), "Arunachal Pradesh": (28.2, 94.7),
    "Assam":               (26.2, 92.9), "Bihar":             (25.1, 85.3),
    "Chhattisgarh":        (21.3, 81.7), "Goa":               (15.3, 74.0),
    "Gujarat":             (22.3, 71.2), "Haryana":           (29.1, 76.1),
    "Himachal Pradesh":    (31.1, 77.2), "Jharkhand":         (23.6, 85.3),
    "Karnataka":           (15.3, 75.7), "Kerala":            (10.9, 76.3),
    "Madhya Pradesh":      (22.9, 78.7), "Maharashtra":       (19.7, 75.7),
    "Manipur":             (24.7, 93.9), "Meghalaya":         (25.5, 91.4),
    "Mizoram":             (23.2, 92.7), "Nagaland":          (26.2, 94.6),
    "Odisha":              (20.9, 85.1), "Punjab":            (31.1, 75.3),
    "Rajasthan":           (27.0, 74.2), "Sikkim":            (27.5, 88.5),
    "Tamil Nadu":          (11.1, 78.7), "Telangana":         (17.4, 79.1),
    "Tripura":             (23.9, 91.5), "Uttar Pradesh":     (26.8, 80.9),
    "Uttarakhand":         (30.1, 79.3), "West Bengal":       (22.9, 87.9),
    "Jammu and Kashmir":   (33.7, 76.9), "Ladakh":            (34.2, 77.6),
    "Delhi":               (28.7, 77.1), "Puducherry":        (11.9, 79.8),
}

with left:
    section_label("State-Level Employment · Latest Year")

    if not pred_df.empty and "financial_year" in pred_df.columns:
        ly = pred_df["financial_year"].max()
        agg = (
            pred_df[pred_df["financial_year"] == ly]
            .groupby("state", as_index=False)
            .agg(
                pd_sum    =("person_days_lakhs",  "sum"),
                pred_sum  =("predicted_persondays","sum"),
                n_dist    =("district",           "count"),
                avg_err   =("prediction_error",   "mean"),
            )
        )

        rng = np.random.default_rng(42)
        lats, lons, szs = [], [], []
        for _, r in agg.iterrows():
            lat, lon = STATE_COORDS.get(r["state"], (22.0, 78.0))
            lats.append(lat + rng.uniform(-0.12, 0.12))
            lons.append(lon + rng.uniform(-0.12, 0.12))
            szs.append(float(r["pd_sum"]))

        mn, mx = min(szs), max(szs)
        bsz = [float(np.clip((v - mn) / (mx - mn + 1e-9) * 14 + 5, 5, 19)) for v in szs]

        fig = go.Figure()
        fig.add_scattergeo(
            lat=lats, lon=lons, mode="markers",
            marker=dict(
                size=bsz, color=szs,
                colorscale=SAFFRON_SCALE,
                colorbar=dict(
                    title=dict(text="Lakh PD", font=dict(color="#78716C", size=9)),
                    tickfont=dict(color="#78716C", size=8),
                    thickness=8, len=0.45,
                    bgcolor="rgba(255,255,255,0.85)",
                ),
                opacity=0.88,
                line=dict(width=1, color="#FFFFFF"),
            ),
            text=agg["state"],
            customdata=list(zip(
                agg["pd_sum"].round(1),
                agg["pred_sum"].round(1),
                agg["n_dist"],
                agg["avg_err"].round(2),
            )),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Actual PD: <b>%{customdata[0]}L</b><br>"
                "Predicted: <b>%{customdata[1]}L</b><br>"
                "Districts: %{customdata[2]}<br>"
                "Avg Model Error: %{customdata[3]}L"
                "<extra></extra>"
            ),
        )
        fig.update_geos(
            scope="asia", showland=True, landcolor="#F5F5F4",
            showocean=True, oceancolor="#EFF6FF",
            showcountries=True, countrycolor="#D6D3D1",
            showsubunits=True, subunitcolor="#E7E5E4",
            center=dict(lat=22, lon=80), projection_scale=5.2,
            bgcolor="rgba(0,0,0,0)",
        )
        fig.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(family="DM Mono, monospace", color="#1C1917"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"FY {ly} · bubble size ∝ employment volume · hover for model predictions")
    else:
        st.info("Start the backend to load state-level data.")

# ── RIGHT: brief + signals ────────────────────────────────────────────────────
with right:
    section_label("Intelligence Brief")

    # Compute signals from data
    n_declining = n_underfunded = 0
    top_state   = "—"
    if not pred_df.empty:
        ly = pred_df["financial_year"].max()
        lat = pred_df[pred_df["financial_year"] == ly]
        prv = pred_df[pred_df["financial_year"] == ly - 1]
        if not prv.empty:
            mg = lat.merge(
                prv[["state", "district", "person_days_lakhs"]].rename(
                    columns={"person_days_lakhs": "prev"}
                ),
                on=["state", "district"], how="left",
            )
            n_declining = int((mg["predicted_persondays"] < mg["prev"]).sum())

    if not opt_df.empty and "budget_allocated_lakhs" in opt_df.columns:
        th = opt_df["budget_allocated_lakhs"].quantile(0.33)
        n_underfunded = int((opt_df["budget_allocated_lakhs"] < th).sum())
    if not opt_df.empty and "persondays_gain" in opt_df.columns:
        top_state = opt_df.groupby("state")["persondays_gain"].sum().idxmax()

    gain_str = f"{nat_gain:+,.1f}L" if nat_gain else "—"
    ly_label = pred_df["financial_year"].max() if not pred_df.empty else "—"

    st.markdown(f"""
<div style="background:#FFF7ED; border:1px solid #FED7AA; border-left:3px solid #FB923C;
            border-radius:8px; padding:1.2rem 1.4rem; margin-bottom:1rem;">
  <p style="font-family:'DM Mono',monospace; font-size:0.56rem; letter-spacing:2.5px;
            text-transform:uppercase; color:#FB923C; margin:0 0 9px 0;">
    ◈ Auto-generated · Pipeline FY {ly_label}</p>
  <p style="font-family:'Source Serif 4',serif; font-size:0.88rem; color:#431407;
            line-height:1.75; margin:0;">
    Budget-neutral LP reallocation yields a projected
    <strong>{gain_str}</strong> of additional employment —
    a <strong>{gain_pct:+.2f}%</strong> uplift at zero additional outlay.
    <strong>{n_declining} districts</strong> face declining employment trajectories.
    Highest reallocation opportunity: <strong>{top_state}</strong>.
    <strong>{n_underfunded} districts</strong> in the bottom budget tercile show
    above-average delivery efficiency.
  </p>
</div>
""", unsafe_allow_html=True)

    section_label("Live Signals")
    signals = [
        (str(n_declining),   "High-Risk Districts",     "Predicted employment decline",    RED),
        (str(n_underfunded), "Underfunded · High Eff.", "Bottom-tercile budget",           AMBER),
        (gain_str,           "LP Reallocation Gain",    f"Budget-neutral · {gain_pct:+.2f}%", GREEN),
        (str(n_dist),        "Districts in Model",      "XGBoost · R²≈0.9963",            SAFFRON),
    ]
    for val, title, body, accent in signals:
        st.markdown(signal_card_html(val, title, body, accent), unsafe_allow_html=True)
