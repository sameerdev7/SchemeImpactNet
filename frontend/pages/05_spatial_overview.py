"""pages/05_spatial_overview.py — National Spatial Overview"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Spatial Overview", page_icon="🗺️", layout="wide")
API = "http://localhost:8000"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.page-title { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #E8F0FE; margin-bottom: 0; }
.page-subtitle { font-size: 0.8rem; color: #7B8EC8; margin-top: 2px; margin-bottom: 1.5rem; letter-spacing: 0.5px; text-transform: uppercase; }
.metric-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1rem 1.2rem; text-align: center; }
.metric-label { font-size: 0.7rem; color: #7B8EC8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 1.5rem; font-weight: 500; color: #E8F0FE; }
</style>
""", unsafe_allow_html=True)

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
    return pd.DataFrame()

st.markdown('<div class="page-title">🗺️ National Spatial Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">District-level MNREGA Performance · Real-time Policy Intelligence</div>', unsafe_allow_html=True)

with st.spinner("Loading..."):
    pred_df = to_df(api_get("/predictions/"))
    opt_df  = to_df(api_get("/optimizer/results"))

if pred_df.empty:
    st.error("No prediction data. Run `python main.py --stage 3` first.")
    st.stop()

latest_year = pred_df["financial_year"].max()
latest = pred_df[pred_df["financial_year"] == latest_year].copy()

if not opt_df.empty:
    opt_slim = opt_df[["state","district","persondays_gain","persondays_gain_pct",
                        "persondays_per_lakh","budget_change_pct"]].copy()
    map_df = latest.merge(opt_slim, on=["state","district"], how="left")
else:
    map_df = latest.copy()
    for c in ["persondays_gain","persondays_gain_pct","persondays_per_lakh","budget_change_pct"]:
        map_df[c] = 0.0
map_df = map_df.fillna(0)

state_agg = map_df.groupby("state").agg(
    person_days_lakhs    = ("person_days_lakhs",    "sum"),
    predicted_persondays = ("predicted_persondays", "sum"),
    persondays_gain_pct  = ("persondays_gain_pct",  "mean"),
    persondays_per_lakh  = ("persondays_per_lakh",  "mean"),
    budget_change_pct    = ("budget_change_pct",    "mean"),
    district_count       = ("district",             "count"),
).reset_index()

METRICS = {
    "Person Days — Latest Year":       ("person_days_lakhs",    "lakh PD", "Blues"),
    "Efficiency (PD per ₹ lakh)":      ("persondays_per_lakh",  "PD/₹L",  "Greens"),
    "Predicted Next-Year Person Days": ("predicted_persondays", "lakh PD", "Purples"),
    "Optimization Gain (%)":           ("persondays_gain_pct",  "%",       "RdYlGn"),
    "Budget Change (%)":               ("budget_change_pct",    "%",       "RdYlGn"),
}
MAP_MODES = ["District Bubble Map", "State Treemap", "State Bar Chart"]

cc1, cc2 = st.columns([2, 1])
selected_label = cc1.selectbox("📊 Map Metric", list(METRICS.keys()))
map_mode = cc2.selectbox("🗺️ View Mode", MAP_MODES)
metric_col, metric_unit, colorscale = METRICS[selected_label]
st.markdown("---")

# ── District Bubble Map (scattergeo with lat/lon — no locationmode needed) ─────
STATE_COORDS = {
    "Andhra Pradesh": (15.9, 79.7), "Arunachal Pradesh": (27.1, 93.6),
    "Assam": (26.2, 92.9), "Bihar": (25.1, 85.3),
    "Chhattisgarh": (21.3, 81.9), "Goa": (15.3, 74.0),
    "Gujarat": (22.3, 71.2), "Haryana": (29.1, 76.1),
    "Himachal Pradesh": (31.1, 77.2), "Jharkhand": (23.6, 85.3),
    "Karnataka": (15.3, 75.7), "Kerala": (10.5, 76.3),
    "Madhya Pradesh": (22.7, 77.7), "Maharashtra": (19.7, 75.7),
    "Manipur": (24.7, 93.9), "Meghalaya": (25.5, 91.4),
    "Mizoram": (23.2, 92.7), "Nagaland": (26.2, 94.6),
    "Odisha": (20.9, 85.1), "Punjab": (31.1, 75.3),
    "Rajasthan": (27.0, 74.2), "Sikkim": (27.5, 88.5),
    "Tamil Nadu": (11.1, 78.7), "Telangana": (17.4, 79.1),
    "Tripura": (23.9, 91.9), "Uttar Pradesh": (27.1, 80.9),
    "Uttarakhand": (30.1, 79.3), "West Bengal": (22.9, 87.9),
    "Jammu and Kashmir": (33.5, 75.3), "Ladakh": (34.2, 77.6),
    "Andaman and Nicobar Islands": (11.7, 92.7),
    "Dadra and Nagar Haveli": (20.1, 73.0), "Lakshadweep": (10.6, 72.6),
    "Delhi": (28.6, 77.2), "Puducherry": (11.9, 79.8),
}

if map_mode == "District Bubble Map":
    bubble_df = map_df.copy()
    rng = np.random.default_rng(42)
    lats, lons = [], []
    for _, row in bubble_df.iterrows():
        base = STATE_COORDS.get(row["state"], (20.0, 78.0))
        lats.append(base[0] + rng.uniform(-2.2, 2.2))
        lons.append(base[1] + rng.uniform(-2.2, 2.2))
    bubble_df["lat"] = lats
    bubble_df["lon"] = lons

    if metric_col not in bubble_df.columns:
        bubble_df[metric_col] = 0
    vmin, vmax = bubble_df[metric_col].min(), bubble_df[metric_col].max()
    vrange = vmax - vmin if vmax != vmin else 1
    bubble_df["bsize"] = ((bubble_df[metric_col] - vmin) / vrange * 22 + 4).clip(4, 26)

    fig = go.Figure()
    fig.add_scattergeo(
        lat=bubble_df["lat"], lon=bubble_df["lon"],
        mode="markers",
        marker=dict(
            size=bubble_df["bsize"],
            color=bubble_df[metric_col],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text=metric_unit, font=dict(color="#E8F0FE")),
                tickfont=dict(color="#E8F0FE"),
                bgcolor="rgba(15,27,61,0.8)",
            ),
            opacity=0.75,
            line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
        ),
        text=bubble_df.apply(lambda r: (
            f"<b>{r['district']}</b><br>State: {r['state']}<br>"
            f"Person Days: {r.get('person_days_lakhs',0):.2f}L<br>"
            f"Predicted: {r.get('predicted_persondays',0):.2f}L<br>"
            f"Efficiency: {r.get('persondays_per_lakh',0):.4f}<br>"
            f"Opt Gain: {r.get('persondays_gain_pct',0):+.2f}%"
        ), axis=1),
        hoverinfo="text",
    )
    fig.update_geos(
        center=dict(lat=22, lon=82),
        lataxis_range=[6, 38], lonaxis_range=[67, 98],
        showland=True,  landcolor="rgba(30,42,65,1)",
        showocean=True, oceancolor="rgba(10,18,40,1)",
        showcountries=True, countrycolor="rgba(255,255,255,0.3)",
        showsubunits=True, subunitcolor="rgba(255,255,255,0.15)",
        showframe=False, bgcolor="rgba(10,18,40,1)",
        showlakes=False,
    )
    fig.update_layout(
        height=620, paper_bgcolor="rgba(10,18,40,1)",
        font=dict(color="#E8F0FE", family="DM Sans"),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"{selected_label} · District Bubbles · {latest_year}",
                   font=dict(color="#E8F0FE", size=14)),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("📍 District positions jittered within state boundaries for visibility.")

elif map_mode == "State Treemap":
    if metric_col not in state_agg.columns:
        state_agg[metric_col] = 0
    plot_df = state_agg[state_agg[metric_col] > 0].copy() if state_agg[metric_col].min() >= 0 \
              else state_agg.copy()
    plot_df["abs_val"] = plot_df[metric_col].abs()

    fig = px.treemap(
        plot_df,
        path=["state"],
        values="abs_val",
        color=metric_col,
        color_continuous_scale=colorscale,
        hover_data={"person_days_lakhs":":.1f", "district_count":True,
                    "persondays_gain_pct":":.2f"},
        title=f"{selected_label} by State — {latest_year}",
        labels={metric_col: metric_unit, "abs_val": "Size",
                "person_days_lakhs": "PD (L)", "district_count": "Districts"},
    )
    fig.update_traces(
        textfont=dict(color="#E8F0FE", size=13),
        hovertemplate="<b>%{label}</b><br>Value: %{color:.2f}<br>Districts: %{customdata[1]}<extra></extra>",
    )
    fig.update_layout(
        height=580, paper_bgcolor="rgba(13,21,38,0.95)",
        font=dict(color="#E8F0FE", family="DM Sans"),
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title=dict(text=metric_unit, font=dict(color="#E8F0FE")),
            tickfont=dict(color="#E8F0FE"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

else:  # State Bar Chart
    sort_col = metric_col if metric_col in state_agg.columns else "person_days_lakhs"
    sorted_df = state_agg.sort_values(sort_col, ascending=True)
    fig = px.bar(
        sorted_df, x=sort_col, y="state", orientation="h",
        color=sort_col, color_continuous_scale=colorscale,
        title=f"{selected_label} by State — {latest_year}",
        labels={sort_col: metric_unit, "state": ""},
        text=sorted_df[sort_col].round(1).astype(str),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(400, len(state_agg)*22), margin=dict(l=0, r=60, t=50, b=0),
        paper_bgcolor="rgba(13,21,38,0.95)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="DM Sans"), coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── District Drill-Down ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 District Drill-Down")
states_list = sorted(map_df["state"].dropna().unique().tolist())
col_s, col_d = st.columns(2)
sel_state = col_s.selectbox("State", states_list)
districts = sorted(map_df[map_df["state"]==sel_state]["district"].dropna().unique().tolist())
sel_dist  = col_d.selectbox("District", districts)

if sel_dist:
    row = map_df[(map_df["state"]==sel_state) & (map_df["district"]==sel_dist)]
    hist_raw = api_get("/districts/history", {"state": sel_state, "district": sel_dist})
    hist_df  = to_df(hist_raw)

    st.markdown(f"#### 📌 {sel_dist}, {sel_state}")
    if not row.empty:
        r = row.iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        for col, label, fmt in [
            (m1, "Person Days",   f"{r.get('person_days_lakhs',0):.2f}L"),
            (m2, "Predicted PD",  f"{r.get('predicted_persondays',0):.2f}L"),
            (m3, "Efficiency",    f"{r.get('persondays_per_lakh',0):.4f}"),
            (m4, "Opt. Gain",     f"{r.get('persondays_gain_pct',0):+.2f}%"),
            (m5, "Budget Change", f"{r.get('budget_change_pct',0):+.1f}%"),
        ]:
            col.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{fmt}</div>
            </div>""", unsafe_allow_html=True)

    if not hist_df.empty and "person_days_lakhs" in hist_df.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        fig4 = go.Figure()
        fig4.add_scatter(
            x=hist_df["financial_year"], y=hist_df["person_days_lakhs"],
            mode="lines+markers", fill="tozeroy",
            fillcolor="rgba(66,133,244,0.12)",
            line=dict(color="#4285F4", width=2.5),
            marker=dict(size=6),
        )
        fig4.update_layout(
            height=220, margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E8F0FE", size=11),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Lakh PD"),
            title=dict(text="Historical Person Days Trend", font=dict(size=13)),
        )
        st.plotly_chart(fig4, use_container_width=True)
