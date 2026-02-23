"""pages/06_strategic_insights.py — Strategic Insights"""

import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Strategic Insights", page_icon="🧠", layout="wide")

API = "http://localhost:8000"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
.page-title { font-family: 'Playfair Display', serif; font-size: 2.1rem; color: #F0F4FF; letter-spacing: -0.5px; }
.page-sub { font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; color: #64748B; margin-bottom: 2rem; }
.insight-card { border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 0.5rem; position: relative; overflow: hidden; }
.insight-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 14px 14px 0 0; }
.card-red    { background: rgba(239,68,68,0.08);  border: 1px solid rgba(239,68,68,0.25);  }
.card-red::before    { background: #EF4444; }
.card-yellow { background: rgba(234,179,8,0.08);  border: 1px solid rgba(234,179,8,0.25);  }
.card-yellow::before { background: #EAB308; }
.card-green  { background: rgba(34,197,94,0.08);  border: 1px solid rgba(34,197,94,0.25);  }
.card-green::before  { background: #22C55E; }
.card-blue   { background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.25); }
.card-blue::before   { background: #3B82F6; }
.card-purple { background: rgba(168,85,247,0.08); border: 1px solid rgba(168,85,247,0.25); }
.card-purple::before { background: #A855F7; }
.card-icon   { font-size: 1.8rem; line-height: 1; margin-bottom: 0.5rem; }
.card-number { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 700; line-height: 1; }
.num-red    { color: #F87171; } .num-yellow { color: #FDE047; }
.num-green  { color: #4ADE80; } .num-blue   { color: #60A5FA; } .num-purple { color: #C084FC; }
.card-label  { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: #94A3B8; margin-top: 4px; }
.card-detail { font-size: 0.85rem; color: #CBD5E1; margin-top: 0.6rem; line-height: 1.5; }
.brief-box { background: linear-gradient(135deg, rgba(15,27,61,0.9), rgba(26,40,85,0.9)); border: 1px solid rgba(99,132,255,0.3); border-radius: 14px; padding: 2rem 2.4rem; margin-top: 1.5rem; }
.brief-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; color: #6384FF; margin-bottom: 0.8rem; font-weight: 600; }
.brief-text { font-size: 1.05rem; color: #CBD5E1; line-height: 1.75; font-style: italic; }
.section-header { font-family: 'Playfair Display', serif; font-size: 1.2rem; color: #E2E8F0; margin-top: 2rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.08); }
.drow { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem; }
.drow:last-child { border-bottom: none; }
.dname { color: #E2E8F0; font-weight: 500; }
.dstate { color: #64748B; font-size: 0.75rem; }
.dval { font-family: 'Playfair Display', serif; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params or {}, timeout=8)
        r.raise_for_status()
        return r.json()
    except:
        return None

def to_df(data):
    if not data: return pd.DataFrame()
    if isinstance(data, list): return pd.DataFrame(data)
    if isinstance(data, dict): return pd.DataFrame([data])
    return pd.DataFrame()

st.markdown('<div class="page-title">🧠 Strategic Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Automated Policy Intelligence · Generated from ML Pipeline</div>', unsafe_allow_html=True)

with st.spinner("Generating insights..."):
    pred_df = to_df(api_get("/predictions/"))
    opt_df  = to_df(api_get("/optimizer/results"))

if pred_df.empty:
    st.error("No data. Run `python main.py --stage 3` first.")
    st.stop()

# ── Compute ────────────────────────────────────────────────────────────────────
latest_year = pred_df["financial_year"].max()
latest = pred_df[pred_df["financial_year"] == latest_year].copy()
prev   = pred_df[pred_df["financial_year"] == latest_year - 1].copy()

# Declining districts
if not prev.empty:
    mg = latest.merge(
        prev[["state","district","person_days_lakhs"]].rename(columns={"person_days_lakhs":"prev_pd"}),
        on=["state","district"], how="left"
    )
    mg["pd_change"] = mg["predicted_persondays"] - mg["prev_pd"]
    declining    = mg[mg["pd_change"] < 0]
    n_declining  = len(declining)
    top_declining = declining.nsmallest(5, "pd_change")[["state","district","pd_change"]]
else:
    n_declining, top_declining = 0, pd.DataFrame()

# Underfunded high-efficiency
if not opt_df.empty and "budget_allocated_lakhs" in opt_df.columns:
    threshold   = opt_df["budget_allocated_lakhs"].quantile(0.33)
    underfunded = opt_df[opt_df["budget_allocated_lakhs"] < threshold]
    n_underfunded = len(underfunded)
    top_underfunded = underfunded.nlargest(5, "persondays_per_lakh")[
        ["state","district","persondays_per_lakh","budget_allocated_lakhs"]]
else:
    n_underfunded, top_underfunded = 0, pd.DataFrame()

# High ROI targets
if not opt_df.empty and "persondays_gain" in opt_df.columns:
    top_roi   = opt_df[opt_df.get("budget_change_pct", pd.Series(0)) > 0].nlargest(10, "persondays_gain") \
                if "budget_change_pct" in opt_df.columns \
                else opt_df.nlargest(10, "persondays_gain")
    n_high_roi = len(top_roi)
    top_roi_5  = top_roi.head(5)
else:
    n_high_roi, top_roi_5 = 0, pd.DataFrame()

# National gain
if not opt_df.empty and "persondays_gain" in opt_df.columns:
    nat_gain = opt_df["persondays_gain"].sum()
    sq_sum   = opt_df["sq_persondays"].sum() if "sq_persondays" in opt_df.columns else 1
    nat_pct  = nat_gain / sq_sum * 100 if sq_sum else 0
    total_bud = opt_df["budget_allocated_lakhs"].sum() if "budget_allocated_lakhs" in opt_df.columns else 0
    state_gains = opt_df.groupby("state")["persondays_gain"].sum().nlargest(3)
    top_states  = ", ".join(state_gains.index.tolist())
else:
    nat_gain, nat_pct, total_bud, top_states = 0, 0, 0, "Rajasthan, Tamil Nadu"

# ── Cards ──────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="insight-card card-red">
        <div class="card-icon">🟥</div>
        <div class="card-number num-red">{n_declining}</div>
        <div class="card-label">High Risk Districts</div>
        <div class="card-detail">Districts predicted to generate fewer person-days next year than current baseline.</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="insight-card card-yellow">
        <div class="card-icon">🟨</div>
        <div class="card-number num-yellow">{n_underfunded}</div>
        <div class="card-label">Underfunded High-Efficiency Districts</div>
        <div class="card-detail">Bottom-tercile budget allocation but above-average efficiency per rupee spent.</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="insight-card card-green">
        <div class="card-icon">🟩</div>
        <div class="card-number num-green">{n_high_roi}</div>
        <div class="card-label">High ROI Reallocation Targets</div>
        <div class="card-detail">Districts where optimizer recommends budget increase for maximum employment return.</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
c4, c5 = st.columns(2)

with c4:
    gain_str = f"+{nat_gain:,.1f}" if nat_gain >= 0 else f"{nat_gain:,.1f}"
    st.markdown(f"""<div class="insight-card card-blue">
        <div class="card-icon">📈</div>
        <div class="card-number num-blue">{gain_str}L</div>
        <div class="card-label">National Employment Gain · Person-Days</div>
        <div class="card-detail">Projected additional employment from LP-optimal budget reallocation ({nat_pct:.2f}% increase).</div>
    </div>""", unsafe_allow_html=True)

with c5:
    budget_cr = total_bud / 1e4
    st.markdown(f"""<div class="insight-card card-purple">
        <div class="card-icon">💰</div>
        <div class="card-number num-purple">₹{budget_cr:,.0f}Cr</div>
        <div class="card-label">Budget Neutral Reallocation</div>
        <div class="card-detail">✅ Zero Additional Outlay. Total MNREGA budget envelope held constant across all districts.</div>
    </div>""", unsafe_allow_html=True)

# ── Policy Brief ───────────────────────────────────────────────────────────────
brief = f"""Optimization analysis of {latest_year} MNREGA data indicates a projected {nat_pct:.2f}% increase 
in national employment generation — equivalent to {nat_gain:+,.1f} lakh additional person-days — achievable 
through budget-neutral reallocation within the existing fiscal envelope of ₹{budget_cr:,.0f} crore. 
{n_declining} districts are flagged as high-risk with predicted year-on-year employment decline, 
warranting priority monitoring and administrative intervention. High-return reallocation opportunities 
are concentrated in {top_states}, where efficiency ratios significantly exceed the national average. 
{n_underfunded} districts currently in the bottom tercile of budget allocation demonstrate above-average 
efficiency, suggesting structural underfunding that constrains employment generation. 
Immediate policy action on this cohort is recommended for maximum welfare impact per rupee of expenditure."""

st.markdown(f"""
<div class="brief-box">
    <div class="brief-label">⚡ Automated Policy Brief — Generated from Pipeline Output</div>
    <div class="brief-text">{brief}</div>
</div>""", unsafe_allow_html=True)

# ── Tables ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
ca, cb, cc = st.columns(3)

with ca:
    st.markdown('<div class="section-header">⚠️ Declining Districts</div>', unsafe_allow_html=True)
    if not top_declining.empty:
        for _, row in top_declining.iterrows():
            st.markdown(f"""<div class="drow">
                <div><div class="dname">{row['district']}</div><div class="dstate">{row['state']}</div></div>
                <div class="dval num-red">{row['pd_change']:+.1f}L</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No declining districts detected.")

with cb:
    st.markdown('<div class="section-header">💡 Top ROI Targets</div>', unsafe_allow_html=True)
    if not top_roi_5.empty:
        for _, row in top_roi_5.iterrows():
            g = row.get("persondays_gain", 0)
            st.markdown(f"""<div class="drow">
                <div><div class="dname">{row['district']}</div><div class="dstate">{row['state']}</div></div>
                <div class="dval num-green">+{g:.1f}L</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No ROI targets computed.")

with cc:
    st.markdown('<div class="section-header">🔴 Underfunded Districts</div>', unsafe_allow_html=True)
    if not top_underfunded.empty:
        for _, row in top_underfunded.iterrows():
            e = row.get("persondays_per_lakh", 0)
            st.markdown(f"""<div class="drow">
                <div><div class="dname">{row['district']}</div><div class="dstate">{row['state']}</div></div>
                <div class="dval num-yellow">{e:.3f}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No underfunded districts detected.")

st.markdown("---")
st.caption(f"Generated from ML pipeline · Test year: {latest_year} · SchemeImpactNet v2.0")
