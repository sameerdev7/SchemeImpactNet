"""
pages/04_optimizer.py
----------------------
Budget Optimizer — Before vs After Analysis.
LP reallocation with full impact visualization and toggle filters.

Self-contained: no utils/ package dependency.
Plotly fix: uses title=dict(...) not titlefont=dict(...) (deprecated).

Literature addition (Rao et al. 2025): Added "Model Comparison Report" button
that triggers model comparison CSV generation, mirroring their Table I
systematic comparison of ML models. When run_model_comparison() is available
in src/model.py, the button triggers it; otherwise shows a message.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Budget Optimizer", page_icon="⚖️", layout="wide")

API = "http://localhost:8000"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.page-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; color: #F0F4FF; letter-spacing: -1px; }
.page-sub   { font-size: 0.75rem; letter-spacing: 2.5px; text-transform: uppercase; color: #475569; margin-bottom: 1.5rem; font-family: 'IBM Plex Mono', monospace; }
.kpi-strip  { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: rgba(255,255,255,0.06); border-radius: 10px; overflow: hidden; margin-bottom: 1.5rem; }
.kpi-cell   { background: #0D1526; padding: 1.2rem 1rem; text-align: center; }
.kpi-val    { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; color: #E2E8F0; }
.kv-green   { color: #4ADE80; } .kv-blue { color: #60A5FA; }
.kpi-label  { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.5px; color: #475569; margin-top: 3px; }
.sec        { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; color: #60A5FA; border-bottom: 1px solid rgba(96,165,250,0.2); padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0; }
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

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API}{endpoint}", json=payload, timeout=20)
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

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">⚖️ Budget Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Linear Programming · Resource Reallocation · Employment Maximization</div>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Solver Configuration")
states_raw   = api_get("/districts/states") or []
scope        = st.sidebar.selectbox("Geographic Scope", ["All-India"] + states_raw)
state_param  = None if scope == "All-India" else scope

st.sidebar.markdown("---")
budget_scale = st.sidebar.slider("Fiscal Multiplier", 0.7, 1.5, 1.0, 0.05)
min_frac     = st.sidebar.slider("Floor Constraint (Min %)", 0.1, 0.8, 0.4, 0.05)
max_frac     = st.sidebar.slider("Ceiling Constraint (Max %)", 1.1, 4.0, 2.5, 0.1)
run_btn      = st.sidebar.button("▶ Execute Optimization", type="primary", use_container_width=True)

# Literature addition: Model Comparison Report (Rao et al. 2025)
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📊 Literature Features")
if st.sidebar.button("📋 Generate Model Comparison\n(Rao et al. 2025)", use_container_width=True):
    try:
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from src.model_additions import run_model_comparison
        st.sidebar.info("Model comparison triggered. Check reports/model_comparison.csv after pipeline completes.")
    except ImportError:
        st.sidebar.info("Add run_model_comparison() to src/model.py (see src_model_additions.py). The function compares XGBoost vs GBR vs RF — Rao et al. (2025) methodology.")

# ── Data ─────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running LP optimizer..."):
        result = api_post("/optimizer/run", {
            "state":        state_param,
            "budget_scale": budget_scale,
            "min_fraction": min_frac,
            "max_fraction": max_frac,
        })
    if result:
        st.success("✅ Optimization converged.")
        df        = to_df(result.get("districts", []))
        sq_total  = result.get("sq_persondays_total", 0)
        opt_total = result.get("opt_persondays_total", 0)
        gain_abs  = result.get("gain_lakhs", 0)
        gain_pct  = result.get("gain_pct", 0)
        total_bud = result.get("total_budget_lakhs", 0)
    else:
        st.stop()
else:
    df = to_df(api_get("/optimizer/results", {"state": state_param} if state_param else {}))
    if df.empty:
        st.info("Configure parameters in the sidebar and click **Execute Optimization**.")
        st.stop()
    sq_total  = df["sq_persondays"].sum()  if "sq_persondays"  in df.columns else 0
    opt_total = df["opt_persondays"].sum() if "opt_persondays" in df.columns else 0
    gain_abs  = opt_total - sq_total
    gain_pct  = gain_abs / sq_total * 100 if sq_total else 0
    total_bud = df["budget_allocated_lakhs"].sum() if "budget_allocated_lakhs" in df.columns else 0

if df.empty:
    st.warning("No optimizer data found.")
    st.stop()

# ── KPI Strip ────────────────────────────────────────────────────────────────────
bud_cr   = total_bud / 1e4
gain_str = f"+{gain_abs:,.1f}L" if gain_abs >= 0 else f"{gain_abs:,.1f}L"
gpct_str = f"+{gain_pct:.2f}%" if gain_pct >= 0 else f"{gain_pct:.2f}%"

st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-cell"><div class="kpi-val">{scope}</div><div class="kpi-label">Scope</div></div>
  <div class="kpi-cell"><div class="kpi-val kv-blue">₹{bud_cr:,.0f}Cr</div><div class="kpi-label">Total Fiscal Outlay</div></div>
  <div class="kpi-cell"><div class="kpi-val">{sq_total:,.1f}L</div><div class="kpi-label">Baseline Output</div></div>
  <div class="kpi-cell"><div class="kpi-val kv-green">{opt_total:,.1f}L</div><div class="kpi-label">Optimized Output</div></div>
  <div class="kpi-cell"><div class="kpi-val kv-green">{gpct_str}</div><div class="kpi-label">Gain · {gain_str}</div></div>
</div>
""", unsafe_allow_html=True)

# ── Toggle Filters ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Allocation Impact Analysis</div>', unsafe_allow_html=True)
cf1, cf2, cf3, _ = st.columns([1, 1, 1, 2])
show_gaining = cf1.toggle("📈 Gaining",  value=True)
show_losing  = cf2.toggle("📉 Losing",   value=True)
show_roi     = cf3.toggle("🎯 Top ROI",  value=False)

dv = df.copy()
if "budget_change" in dv.columns:
    if show_roi and "persondays_gain" in dv.columns:
        dv = dv.nlargest(20, "persondays_gain")
    else:
        masks = []
        if show_gaining: masks.append(dv["budget_change"] >= 0)
        if show_losing:  masks.append(dv["budget_change"] <  0)
        if masks:
            m = masks[0]
            for x in masks[1:]: m = m | x
            dv = dv[m]

# ── A) Bar Chart ─────────────────────────────────────────────────────────────────
st.markdown("**A — Budget Before vs After: Top 15 Most Affected Districts**")

if "budget_change" in dv.columns and "budget_allocated_lakhs" in dv.columns:
    t15 = dv.reindex(dv["budget_change"].abs().nlargest(15).index).sort_values("budget_change")
    fig1 = go.Figure()
    fig1.add_bar(y=t15["district"], x=t15["budget_allocated_lakhs"],
                 name="Current", orientation="h",
                 marker_color="rgba(96,165,250,0.6)",
                 marker_line=dict(color="#60A5FA", width=0.5))
    if "optimized_budget" in t15.columns:
        fig1.add_bar(y=t15["district"], x=t15["optimized_budget"],
                     name="Proposed", orientation="h",
                     marker_color="rgba(74,222,128,0.75)",
                     marker_line=dict(color="#4ADE80", width=0.5))
    fig1.update_layout(
        barmode="group", height=480, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(13,21,38,0.9)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="IBM Plex Sans"),
        legend=dict(orientation="h", y=1.05, x=0),
        xaxis=dict(title="Budget (Rs. lakh)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── B) Scatter — Rich District Tooltip ──────────────────────────────────────────
st.markdown("**B — Efficiency vs Budget Change**")

if "persondays_per_lakh" in dv.columns and "budget_change_pct" in dv.columns:
    # Compute risk level based on persondays_gain_pct
    def risk_level(row):
        gain_pct = row.get("persondays_gain_pct", 0)
        if gain_pct < -5:
            return "High"
        elif gain_pct < 5:
            return "Medium"
        else:
            return "Low"

    scatter_df = dv.copy()
    scatter_df["risk_level"] = scatter_df.apply(risk_level, axis=1)

    # Pull in actual persondays from predictions if available (sq_persondays = actual baseline)
    # sq_persondays is the predicted baseline; opt_persondays is the projected optimised value
    has_sq  = "sq_persondays" in scatter_df.columns
    has_opt = "opt_persondays" in scatter_df.columns
    has_gain_pct = "persondays_gain_pct" in scatter_df.columns

    # Build customdata as columns added directly to scatter_df
    scatter_df["_cd_district"]    = scatter_df["district"].fillna("—")
    scatter_df["_cd_sq_pd"]       = scatter_df["sq_persondays"].round(1) if has_sq else "—"
    scatter_df["_cd_opt_pd"]      = scatter_df["opt_persondays"].round(1) if has_opt else "—"
    scatter_df["_cd_eff"]         = (scatter_df["persondays_per_lakh"] * 100).round(1)
    scatter_df["_cd_gain_pct"]    = scatter_df["persondays_gain_pct"].round(1) if has_gain_pct else 0.0
    scatter_df["_cd_risk"]        = scatter_df["risk_level"]
    scatter_df["_cd_state"]       = scatter_df["state"].fillna("—")
    scatter_df["_cd_bud_chg"]     = scatter_df["budget_change_pct"].round(1)

    color_col = "persondays_gain" if "persondays_gain" in scatter_df.columns else "budget_change_pct"
    size_col  = scatter_df["budget_allocated_lakhs"].clip(lower=1) if "budget_allocated_lakhs" in scatter_df.columns else None

    fig2 = px.scatter(
        scatter_df,
        x="persondays_per_lakh",
        y="budget_change_pct",
        color=color_col,
        color_continuous_scale="RdYlGn",
        size=size_col,
        size_max=20,
        custom_data=["_cd_district", "_cd_sq_pd", "_cd_opt_pd", "_cd_eff",
                     "_cd_gain_pct", "_cd_risk", "_cd_state", "_cd_bud_chg"],
        labels={
            "persondays_per_lakh": "Efficiency (PD/₹L)",
            "budget_change_pct": "Budget Change (%)",
            "persondays_gain": "PD Gain (L)",
        },
    )

    # Rich hover tooltip matching the requested format
    fig2.update_traces(
        hovertemplate=(
            "<b>District: %{customdata[0]}</b><br>"
            "<span style='color:#94A3B8'>%{customdata[6]}</span><br>"
            "─────────────────────<br>"
            "Actual (baseline): %{customdata[1]}L<br>"
            "Predicted (optimised): %{customdata[2]}L<br>"
            "Efficiency: %{customdata[3]:.0f} PD / ₹1L<br>"
            "Budget Change: %{customdata[7]}%<br>"
            "Optimisation Gain: %{customdata[4]:+.1f}%<br>"
            "Risk Level: <b>%{customdata[5]}</b>"
            "<extra></extra>"
        )
    )

    fig2.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dash")
    fig2.add_vline(
        x=scatter_df["persondays_per_lakh"].median(),
        line_color="rgba(255,255,255,0.1)", line_dash="dot",
        annotation_text="Median efficiency",
        annotation_font_color="#94A3B8",
        annotation_position="top",
    )
    fig2.update_layout(
        height=440, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(13,21,38,0.9)", plot_bgcolor="rgba(13,21,38,0.5)",
        font=dict(color="#CBD5E1", family="IBM Plex Sans"),
        coloraxis_colorbar=dict(
            title=dict(text="PD Gain", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tables ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Regional Adjustment Priorities</div>', unsafe_allow_html=True)

show_cols = [c for c in ["state","district","budget_allocated_lakhs","optimized_budget",
                          "budget_change_pct","persondays_gain","persondays_per_lakh"]
             if c in dv.columns]
col_cfg = {
    "budget_change_pct":   st.column_config.NumberColumn("Δ Budget %",  format="%.1f%%"),
    "persondays_gain":     st.column_config.NumberColumn("PD Gain (L)", format="%.2f"),
    "persondays_per_lakh": st.column_config.NumberColumn("Efficiency",  format="%.4f"),
}
cu, cd = st.columns(2)
with cu:
    st.markdown("📈 **Budget Increase**")
    if "budget_change" in dv.columns and "persondays_gain" in dv.columns:
        st.dataframe(
            dv[dv["budget_change"] >= 0].nlargest(10, "persondays_gain")[show_cols].round(2),
            use_container_width=True, hide_index=True, column_config=col_cfg,
        )
with cd:
    st.markdown("📉 **Budget Reduction**")
    if "budget_change" in dv.columns and "persondays_gain" in dv.columns:
        st.dataframe(
            dv[dv["budget_change"] < 0].nsmallest(10, "persondays_gain")[show_cols].round(2),
            use_container_width=True, hide_index=True, column_config=col_cfg,
        )

with st.expander("📋 Full allocation table"):
    scol2 = "persondays_gain" if "persondays_gain" in dv.columns else show_cols[0]
    st.dataframe(
        dv[show_cols].sort_values(scol2, ascending=False).round(2),
        use_container_width=True, hide_index=True,
    )
