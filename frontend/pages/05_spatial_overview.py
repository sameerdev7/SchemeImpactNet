"""
pages/05_spatial_overview.py
----------------------------
National Spatial Overview — 3-mode India district visualization.
No GeoJSON dependency. Works 100% offline with API data only.

Modes:
  1. District Bubble Map  — scattergeo with lat/lon, sized+colored by metric
  2. State Treemap        — tile area proportional to metric value
  3. State Rankings       — horizontal bar chart ranked by metric

Literature note (Deshpande & Pandurangi 2018): Their paper compared scheme
popularity across Swachh Bharat, Digital India and Demonetization using
visualizations. This page implements the same comparative ranking concept
for MNREGA district-level outcomes using a composite_performance_score.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Spatial Overview", page_icon="🗺️", layout="wide")

API = "http://localhost:8000"

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.page-title  { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #E8F0FE; margin-bottom: 0; }
.page-subtitle { font-size: 0.8rem; color: #7B8EC8; margin-top: 2px; margin-bottom: 1.5rem; letter-spacing: 0.5px; text-transform: uppercase; }
.metric-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1rem 1.2rem; text-align: center; }
.metric-label { font-size: 0.7rem; color: #7B8EC8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 1.5rem; font-weight: 500; color: #E8F0FE; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params or {}, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def to_df(data):
    if not data: return pd.DataFrame()
    if isinstance(data, list): return pd.DataFrame(data)
    if isinstance(data, dict): return pd.DataFrame([data])
    return pd.DataFrame()

# Approximate state centroids for bubble map
STATE_COORDS = {
    "Andhra Pradesh": (15.9, 79.7), "Arunachal Pradesh": (28.2, 94.7),
    "Assam": (26.2, 92.9), "Bihar": (25.1, 85.3),
    "Chhattisgarh": (21.3, 81.7), "Goa": (15.3, 74.0),
    "Gujarat": (22.3, 71.2), "Haryana": (29.1, 76.1),
    "Himachal Pradesh": (31.1, 77.2), "Jharkhand": (23.6, 85.3),
    "Karnataka": (15.3, 75.7), "Kerala": (10.9, 76.3),
    "Madhya Pradesh": (22.9, 78.7), "Maharashtra": (19.7, 75.7),
    "Manipur": (24.7, 93.9), "Meghalaya": (25.5, 91.4),
    "Mizoram": (23.2, 92.7), "Nagaland": (26.2, 94.6),
    "Odisha": (20.9, 85.1), "Punjab": (31.1, 75.3),
    "Rajasthan": (27.0, 74.2), "Sikkim": (27.5, 88.5),
    "Tamil Nadu": (11.1, 78.7), "Telangana": (17.4, 79.1),
    "Tripura": (23.9, 91.5), "Uttar Pradesh": (26.8, 80.9),
    "Uttarakhand": (30.1, 79.3), "West Bengal": (22.9, 87.9),
    "Jammu and Kashmir": (33.7, 76.9), "Ladakh": (34.2, 77.6),
    "Delhi": (28.7, 77.1), "Puducherry": (11.9, 79.8),
}

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🗺️ National Spatial Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">District-level MNREGA Performance · Real-time Policy Intelligence</div>', unsafe_allow_html=True)

# ── Data ────────────────────────────────────────────────────────────────────────
with st.spinner("Loading national data..."):
    pred_df = to_df(api_get("/predictions/"))
    opt_df  = to_df(api_get("/optimizer/results"))

if pred_df.empty:
    st.error("No prediction data. Run `python main.py --stage 3` first.")
    st.stop()

latest_year = pred_df["financial_year"].max()
latest      = pred_df[pred_df["financial_year"] == latest_year].copy()

if not opt_df.empty:
    opt_slim = opt_df[["state","district","persondays_gain","persondays_gain_pct",
                        "persondays_per_lakh","budget_change_pct"]].copy()
    map_df = latest.merge(opt_slim, on=["state","district"], how="left")
else:
    map_df = latest.copy()
    for c in ["persondays_gain","persondays_gain_pct","persondays_per_lakh","budget_change_pct"]:
        map_df[c] = 0.0

map_df = map_df.fillna(0)

# ── Controls ─────────────────────────────────────────────────────────────────────
METRICS = {
    "Person Days — Latest Year":       ("person_days_lakhs",    "lakh PD", "Blues"),
    "Efficiency (PD per ₹ lakh)":      ("persondays_per_lakh",  "PD/₹L",  "Greens"),
    "Predicted Next-Year Person Days": ("predicted_persondays", "lakh PD", "Purples"),
    "Optimization Gain (%)":           ("persondays_gain_pct",  "%",       "RdYlGn"),
    "Budget Change (%)":               ("budget_change_pct",    "%",       "RdYlGn"),
}

c1, c2, _ = st.columns([2, 1, 1])
selected_label = c1.selectbox("📊 Map Metric", list(METRICS.keys()))
view_mode = c2.selectbox("📐 View", ["District Bubbles", "State Treemap", "State Rankings"])
metric_col, metric_unit, colorscale = METRICS[selected_label]

st.markdown("---")

# ── State-level aggregation ────────────────────────────────────────────────────
state_agg = map_df.groupby("state").agg(
    person_days_lakhs    =("person_days_lakhs",    "sum"),
    predicted_persondays =("predicted_persondays", "sum"),
    persondays_per_lakh  =("persondays_per_lakh",  "mean"),
    persondays_gain_pct  =("persondays_gain_pct",  "mean"),
    budget_change_pct    =("budget_change_pct",    "mean"),
    persondays_gain      =("persondays_gain",      "sum"),
    district_count       =("district",             "count"),
).reset_index()
state_agg["metric_val"] = state_agg.get(metric_col, state_agg["person_days_lakhs"])

# ── View: District Bubbles ─────────────────────────────────────────────────────
if view_mode == "District Bubbles":
    # Add approximate lat/lon by jittering around state centroid
    def get_coords(row):
        lat, lon = STATE_COORDS.get(row["state"], (20.0, 78.0))
        rng = np.random.RandomState(hash(row["district"]) % (2**31))
        return lat + rng.uniform(-1.5, 1.5), lon + rng.uniform(-1.5, 1.5)

    if "lat" not in map_df.columns:
        coords = map_df.apply(get_coords, axis=1)
        map_df["lat"] = [c[0] for c in coords]
        map_df["lon"] = [c[1] for c in coords]

    if metric_col not in map_df.columns:
        map_df[metric_col] = 0.0

    fig = go.Figure()
    vals = map_df[metric_col].fillna(0)
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin: vmax = vmin + 1

    fig.add_scattergeo(
        lat=map_df["lat"],
        lon=map_df["lon"],
        mode="markers",
        marker=dict(
            size=np.clip((vals - vmin) / (vmax - vmin) * 14 + 4, 4, 18),
            color=vals,
            colorscale=colorscale,
            colorbar=dict(title=dict(text=metric_unit, font=dict(color="#E8F0FE")),
                          tickfont=dict(color="#E8F0FE")),
            opacity=0.75,
            line=dict(width=0.3, color="rgba(255,255,255,0.3)"),
        ),
        text=map_df["district"] + ", " + map_df["state"],
        customdata=np.stack([
            map_df["state"],
            map_df["person_days_lakhs"].round(2),
            map_df["predicted_persondays"].round(2),
            map_df.get("persondays_per_lakh", pd.Series(0, index=map_df.index)).round(4),
        ], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Person Days: %{customdata[1]} L<br>"
            "Predicted PD: %{customdata[2]} L<br>"
            "Efficiency: %{customdata[3]}<br>"
            "<extra></extra>"
        ),
    )
    fig.update_geos(
        scope="asia",
        showland=True, landcolor="rgba(30,40,70,0.9)",
        showocean=True, oceancolor="rgba(10,20,50,0.95)",
        showcountries=True, countrycolor="rgba(255,255,255,0.1)",
        showsubunits=True, subunitcolor="rgba(255,255,255,0.08)",
        center=dict(lat=22, lon=80),
        projection_scale=4.5,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=600, paper_bgcolor="rgba(10,20,45,0.95)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8F0FE", family="DM Sans"),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"{selected_label} — {latest_year} (District Bubbles)", font=dict(size=14)),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── View: State Treemap ────────────────────────────────────────────────────────
elif view_mode == "State Treemap":
    tm_df = state_agg[state_agg["metric_val"] > 0].copy()
    fig = px.treemap(
        tm_df,
        path=["state"],
        values="metric_val",
        color="metric_val",
        color_continuous_scale=colorscale,
        hover_data={"district_count": True},
        title=f"{selected_label} — {latest_year} (State Treemap)",
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:.2f}",
        hovertemplate="<b>%{label}</b><br>Value: %{value:.3f}<br>Districts: %{customdata[0]}<extra></extra>",
    )
    fig.update_layout(
        height=580, paper_bgcolor="rgba(10,20,45,0.95)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8F0FE", family="DM Sans"),
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title=dict(text=metric_unit, font=dict(color="#E8F0FE")),
            tickfont=dict(color="#E8F0FE"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── View: State Rankings ───────────────────────────────────────────────────────
else:
    ranked = state_agg.sort_values("metric_val", ascending=True)
    colors = ranked["metric_val"].apply(
        lambda v: "#4ADE80" if v >= ranked["metric_val"].quantile(0.67)
        else "#FBBF24" if v >= ranked["metric_val"].quantile(0.33)
        else "#F87171"
    )
    fig = go.Figure()
    fig.add_bar(
        x=ranked["metric_val"],
        y=ranked["state"],
        orientation="h",
        marker=dict(color=list(colors), opacity=0.85),
        text=ranked["metric_val"].round(2).astype(str),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + selected_label + ": %{x:.3f}<extra></extra>",
    )
    fig.update_layout(
        height=max(500, len(ranked) * 22),
        margin=dict(l=0, r=80, t=50, b=0),
        paper_bgcolor="rgba(10,20,45,0.95)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="DM Sans"),
        title=dict(text=f"State Rankings — {selected_label} ({latest_year})", font=dict(size=14, color="#E8F0FE")),
        xaxis=dict(title=metric_unit, gridcolor="rgba(255,255,255,0.05)",
                   range=[0, ranked["metric_val"].max() * 1.15]),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = Top tier (≥67th percentile)  |  Amber = Mid tier  |  Red = Bottom tier (≤33rd percentile)")

# ── District Drill-Down ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 District Drill-Down")

states_list = sorted(map_df["state"].dropna().unique().tolist())
col_s, col_d = st.columns(2)
sel_state = col_s.selectbox("State", states_list)
districts = sorted(map_df[map_df["state"]==sel_state]["district"].dropna().unique().tolist())
sel_dist  = col_d.selectbox("District", districts)

if sel_dist:
    row = map_df[(map_df["state"]==sel_state)&(map_df["district"]==sel_dist)]
    hist_raw = api_get("/districts/history", {"state": sel_state, "district": sel_dist})
    hist_df  = to_df(hist_raw)

    st.markdown(f"#### 📌 {sel_dist}, {sel_state}")
    if not row.empty:
        r = row.iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        for col, label, fmt in [
            (m1, "Person Days",    f"{r.get('person_days_lakhs', 0):.2f}L"),
            (m2, "Predicted PD",   f"{r.get('predicted_persondays', 0):.2f}L"),
            (m3, "Efficiency",     f"{r.get('persondays_per_lakh', 0):.4f}"),
            (m4, "Opt. Gain",      f"{r.get('persondays_gain_pct', 0):+.2f}%"),
            (m5, "Budget Change",  f"{r.get('budget_change_pct', 0):+.1f}%"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{fmt}</div>
            </div>""", unsafe_allow_html=True)

    if not hist_df.empty and "person_days_lakhs" in hist_df.columns:
        fig3 = go.Figure()
        fig3.add_scatter(x=hist_df["financial_year"], y=hist_df["person_days_lakhs"],
                         mode="lines+markers", fill="tozeroy",
                         fillcolor="rgba(66,133,244,0.12)",
                         line=dict(color="#4285F4", width=2.5),
                         marker=dict(size=6))
        fig3.update_layout(
            height=220, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E8F0FE", size=11),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Lakh PD"),
            title=dict(text="Historical Trend", font=dict(size=13)),
        )
        st.plotly_chart(fig3, use_container_width=True)
