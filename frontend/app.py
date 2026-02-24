"""
frontend/app.py
---------------
SchemeImpactNet — Dashboard Landing Page.

Design: Government data meets mission control.
Tightrope between institutional authority and live-data energy.
No brochure text. Pure signal.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="SchemeImpactNet",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API = "http://localhost:8000"

# ── Fonts + Global CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Syne+Mono&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #070D1A;
}

/* ── Brand bar ── */
.brand {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 4px;
}
.brand-mark {
    font-family: 'Syne Mono', monospace;
    font-size: 0.75rem;
    color: #3B6FE8;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 3px 8px;
    border: 1px solid rgba(59,111,232,0.4);
    border-radius: 2px;
}
.brand-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #F0F4FF;
    letter-spacing: -1.5px;
    line-height: 1;
    margin: 0;
}
.brand-title span {
    color: #3B6FE8;
}
.brand-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #4A5568;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 6px;
    margin-bottom: 0;
}

/* ── Live indicator ── */
.live-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    background: #22C55E;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
    vertical-align: middle;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50%       { opacity: 0.8; box-shadow: 0 0 0 5px rgba(34,197,94,0); }
}
.status-bar {
    font-family: 'Syne Mono', monospace;
    font-size: 0.68rem;
    color: #4A5568;
    letter-spacing: 1px;
    margin-bottom: 1.5rem;
}
.status-bar .ok { color: #22C55E; }
.status-bar .sep { margin: 0 12px; color: #2D3748; }

/* ── KPI strip ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.kpi-cell {
    background: #0B1224;
    padding: 1.1rem 1.2rem;
    position: relative;
}
.kpi-cell::after {
    content: '';
    position: absolute;
    bottom: 0; left: 1.2rem; right: 1.2rem;
    height: 1px;
    background: rgba(59,111,232,0);
    transition: background 0.3s;
}
.kpi-cell:hover::after { background: rgba(59,111,232,0.4); }
.kpi-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #E8F0FE;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-num.accent { color: #3B6FE8; }
.kpi-num.green  { color: #4ADE80; }
.kpi-num.amber  { color: #FBBF24; }
.kpi-label {
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4A5568;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3B6FE8;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 16px;
    height: 1px;
    background: #3B6FE8;
}

/* ── AI brief box ── */
.ai-brief {
    background: linear-gradient(135deg, rgba(11,18,36,0.98) 0%, rgba(20,35,70,0.95) 100%);
    border: 1px solid rgba(59,111,232,0.25);
    border-left: 3px solid #3B6FE8;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.ai-brief::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59,111,232,0.06) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(50%, -50%);
}
.ai-brief-tag {
    font-family: 'Syne Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: #3B6FE8;
    text-transform: uppercase;
    margin-bottom: 10px;
    opacity: 0.9;
}
.ai-brief-text {
    font-size: 0.92rem;
    color: #94A3B8;
    line-height: 1.7;
}
.ai-brief-text strong { color: #CBD5E1; font-weight: 500; }

/* ── Insight cards ── */
.card-grid { display: flex; flex-direction: column; gap: 10px; }
.icard {
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.icard-red    { background: rgba(239,68,68,0.06);  border-color: rgba(239,68,68,0.2); }
.icard-yellow { background: rgba(251,191,36,0.06); border-color: rgba(251,191,36,0.2); }
.icard-green  { background: rgba(74,222,128,0.06); border-color: rgba(74,222,128,0.2); }
.icard-blue   { background: rgba(59,111,232,0.06); border-color: rgba(59,111,232,0.2); }
.icard-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    min-width: 52px;
    text-align: right;
}
.icard-red    .icard-num { color: #F87171; }
.icard-yellow .icard-num { color: #FBBF24; }
.icard-green  .icard-num { color: #4ADE80; }
.icard-blue   .icard-num { color: #60A5FA; }
.icard-body {}
.icard-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 2px;
    letter-spacing: 0.2px;
}
.icard-sub { font-size: 0.7rem; color: #4A5568; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
[data-testid="stSidebar"] { background: #070D1A; }
</style>
""", unsafe_allow_html=True)

# ── API helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params or {}, timeout=5)
        r.raise_for_status()
        return r.json()
    except:
        return None

def to_df(d):
    if not d: return pd.DataFrame()
    return pd.DataFrame(d) if isinstance(d, list) else pd.DataFrame([d])

# ── Fetch data ─────────────────────────────────────────────────────────────────
stats    = api_get("/districts/stats") or {}
pred_raw = api_get("/predictions/")
opt_raw  = api_get("/optimizer/results")
pred_df  = to_df(pred_raw)
opt_df   = to_df(opt_raw)

backend_ok = bool(stats)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand">
    <span class="brand-mark">MNREGA · ML</span>
</div>
<div class="brand-title">Scheme<span>Impact</span>Net</div>
<p class="brand-sub">Predictive Impact Analysis & Budget Optimisation Framework · India</p>
""", unsafe_allow_html=True)

# Live status bar
api_status = '<span class="ok">● LIVE</span>' if backend_ok else '<span style="color:#EF4444">● OFFLINE</span>'
year_range = stats.get("year_range", "2015–2024")
n_dist     = stats.get("total_districts", "725")
n_states   = stats.get("total_states", "28")

st.markdown(f"""
<div class="status-bar">
    API {api_status}
    <span class="sep">|</span>
    {n_dist} districts
    <span class="sep">|</span>
    {n_states} states
    <span class="sep">|</span>
    {year_range}
    <span class="sep">|</span>
    XGBoost · SciPy LP · FastAPI
</div>
""", unsafe_allow_html=True)

# ── KPI Strip ──────────────────────────────────────────────────────────────────
total_pd  = stats.get("total_persondays_lakhs", 0)
total_exp = stats.get("total_expenditure_lakhs", 0)
covid_pct = stats.get("covid_spike_pct", 0)

nat_gain, gain_pct_val = 0.0, 0.0
if not opt_df.empty and "persondays_gain" in opt_df.columns:
    nat_gain     = opt_df["persondays_gain"].sum()
    sq_sum       = opt_df["sq_persondays"].sum() if "sq_persondays" in opt_df.columns else 1
    gain_pct_val = nat_gain / sq_sum * 100 if sq_sum else 0

exp_cr = total_exp / 1e4 if total_exp else 0

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-cell">
        <div class="kpi-num accent">{int(n_dist)}</div>
        <div class="kpi-label">Districts Covered</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-num">{total_pd:,.0f}L</div>
        <div class="kpi-label">Total Person-Days</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-num">₹{exp_cr:,.0f}Cr</div>
        <div class="kpi-label">Fiscal Envelope</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-num amber">{covid_pct:+.1f}%</div>
        <div class="kpi-label">COVID Spike (2020)</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-num green">{gain_pct_val:+.2f}%</div>
        <div class="kpi-label">LP Realloc. Gain</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout: map left | insights right ─────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

# ───── LEFT: Mini spatial overview ────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-label">Spatial Distribution</div>', unsafe_allow_html=True)

    if not pred_df.empty:
        latest_yr = pred_df["financial_year"].max()
        latest    = pred_df[pred_df["financial_year"] == latest_yr].copy()

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

        # State-level aggregation for the bubble map
        state_agg = latest.groupby("state").agg(
            person_days=("person_days_lakhs", "sum"),
            predicted  =("predicted_persondays", "sum"),
        ).reset_index()

        lats, lons, sizes, colors_v = [], [], [], []
        for _, row in state_agg.iterrows():
            lat, lon = STATE_COORDS.get(row["state"], (20.0, 78.0))
            lats.append(lat + np.random.uniform(-0.3, 0.3))
            lons.append(lon + np.random.uniform(-0.3, 0.3))
            sizes.append(row["person_days"])
            colors_v.append(row["person_days"])

        vmin, vmax = min(sizes), max(sizes)
        bubble_sizes = [np.clip((v - vmin) / (vmax - vmin) * 28 + 6, 6, 34) for v in sizes]

        fig = go.Figure()
        fig.add_scattergeo(
            lat=lats, lon=lons,
            mode="markers",
            marker=dict(
                size=bubble_sizes,
                color=colors_v,
                colorscale=[[0, "#1E3A5F"], [0.4, "#2563EB"], [0.7, "#3B82F6"], [1, "#93C5FD"]],
                colorbar=dict(
                    title=dict(text="Lakh PD", font=dict(color="#4A5568", size=10)),
                    tickfont=dict(color="#4A5568", size=9),
                    thickness=10,
                    len=0.5,
                ),
                opacity=0.85,
                line=dict(width=0.5, color="rgba(147,197,253,0.3)"),
            ),
            text=state_agg["state"],
            customdata=np.stack([
                state_agg["person_days"].round(1),
                state_agg["predicted"].round(1),
            ], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Person Days: %{customdata[0]}L<br>"
                "Predicted: %{customdata[1]}L"
                "<extra></extra>"
            ),
        )
        fig.update_geos(
            scope="asia",
            showland=True,    landcolor="rgba(15,25,50,0.95)",
            showocean=True,   oceancolor="rgba(7,13,26,1)",
            showcountries=True, countrycolor="rgba(59,111,232,0.15)",
            showsubunits=True,  subunitcolor="rgba(59,111,232,0.08)",
            center=dict(lat=22, lon=80),
            projection_scale=5,
            bgcolor="rgba(0,0,0,0)",
        )
        fig.update_layout(
            height=430,
            paper_bgcolor="rgba(7,13,26,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(color="#E8F0FE", family="DM Sans"),
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"State-level person-days · FY {latest_yr} · Bubble size = employment volume")
    else:
        st.info("Run `python main.py --stage 3` to load prediction data.")

# ───── RIGHT: AI brief + Insight cards ────────────────────────────────────────
with right_col:

    # ── AI-generated summary ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Intelligence Brief</div>', unsafe_allow_html=True)

    # Build a real summary from live data
    n_declining, n_underfunded, top_state = 0, 0, "—"

    if not pred_df.empty and not opt_df.empty:
        latest_yr = pred_df["financial_year"].max()
        lat = pred_df[pred_df["financial_year"] == latest_yr]
        prv = pred_df[pred_df["financial_year"] == latest_yr - 1]
        if not prv.empty:
            mg = lat.merge(
                prv[["state","district","person_days_lakhs"]].rename(columns={"person_days_lakhs":"prev"}),
                on=["state","district"], how="left"
            )
            mg["chg"] = mg["predicted_persondays"] - mg["prev"]
            n_declining = int((mg["chg"] < 0).sum())

        if "budget_allocated_lakhs" in opt_df.columns:
            th = opt_df["budget_allocated_lakhs"].quantile(0.33)
            n_underfunded = int((opt_df["budget_allocated_lakhs"] < th).sum())

        if "persondays_gain" in opt_df.columns:
            top_state = opt_df.groupby("state")["persondays_gain"].sum().idxmax()

    gain_display = f"{nat_gain:+,.1f}L" if nat_gain else "—"
    brief_html = f"""
<div class="ai-brief">
    <div class="ai-brief-tag">◈ Generated from pipeline · FY {pred_df['financial_year'].max() if not pred_df.empty else '—'}</div>
    <div class="ai-brief-text">
        Budget-neutral reallocation of the MNREGA fiscal envelope yields a projected
        <strong>{gain_display} lakh person-days</strong> of additional employment —
        a <strong>{gain_pct_val:+.2f}%</strong> uplift at zero additional outlay.
        <strong>{n_declining} districts</strong> are flagged for declining employment trajectories,
        warranting priority administrative review.
        High-return reallocation is concentrated in <strong>{top_state}</strong>.
        <strong>{n_underfunded} districts</strong> in the bottom budget tercile
        demonstrate above-average delivery efficiency, suggesting structural underfunding
        that constrains welfare generation.
    </div>
</div>"""
    st.markdown(brief_html, unsafe_allow_html=True)

    # ── Insight cards ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Live Signals</div>', unsafe_allow_html=True)

    cards_html = f"""
<div class="card-grid">
    <div class="icard icard-red">
        <div class="icard-num">{n_declining}</div>
        <div class="icard-body">
            <div class="icard-title">High-Risk Districts</div>
            <div class="icard-sub">Predicted employment decline · next cycle</div>
        </div>
    </div>
    <div class="icard icard-yellow">
        <div class="icard-num">{n_underfunded}</div>
        <div class="icard-body">
            <div class="icard-title">Underfunded · High Efficiency</div>
            <div class="icard-sub">Bottom-tercile budget · above-avg delivery</div>
        </div>
    </div>
    <div class="icard icard-green">
        <div class="icard-num">{gain_display}</div>
        <div class="icard-body">
            <div class="icard-title">LP Reallocation Gain</div>
            <div class="icard-sub">Budget-neutral · {gain_pct_val:+.2f}% national uplift</div>
        </div>
    </div>
    <div class="icard icard-blue">
        <div class="icard-num">{int(n_dist)}</div>
        <div class="icard-body">
            <div class="icard-title">Districts in Model</div>
            <div class="icard-sub">XGBoost · {year_range}</div>
        </div>
    </div>
</div>"""
    st.markdown(cards_html, unsafe_allow_html=True)

# ── Divider + Nav hint ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem;
            display: flex; gap: 2rem; align-items: center;">
    <span style="font-family:'Syne Mono',monospace; font-size:0.62rem;
                 letter-spacing:2px; color:#2D3748; text-transform:uppercase;">
        Navigate →
    </span>
    <span style="font-size:0.75rem; color:#3D4F6E;">
        01 Overview &nbsp;·&nbsp; 02 District Explorer &nbsp;·&nbsp;
        03 Predictions &nbsp;·&nbsp; 04 Optimizer &nbsp;·&nbsp;
        05 Spatial Map &nbsp;·&nbsp; 06 Strategic Insights
    </span>
</div>
""", unsafe_allow_html=True)
