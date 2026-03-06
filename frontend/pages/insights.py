# pages/insights.py — Strategic Insights & Policy Brief.


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from theme import inject_theme, page_header, section_label, kpi_html, signal_card_html, PLOTLY_LAYOUT, SAFFRON, GREEN, RED, AMBER, BLUE
from utils.api_client import fetch_states, fetch_predictions, fetch_optimizer_results, fetch_yearly_trend

inject_theme()
page_header(
    "◈ Module 06",
    "Strategic Insights",
    "Auto-generated policy intelligence — high-risk districts, efficiency leaders, and reallocation priorities",
)

states = fetch_states()
if not states:
    st.error("⚠️ API offline — run `uvicorn backend.main:app --port 8000`")
    st.stop()

cs, _ = st.columns([1, 2])
with cs:
    scope = st.selectbox("State Scope", ["All India"] + states)
state_param = None if scope == "All India" else scope

pred_df = fetch_predictions(state=state_param)
opt_df  = fetch_optimizer_results(state=state_param)
trend   = fetch_yearly_trend(state_param)

if pred_df.empty:
    st.info("No data — run the pipeline first.")
    st.stop()

st.markdown("---")

# ── Section A: Declining districts ───────────────────────────────────────────
section_label("A. High-Risk Districts — Declining Employment Trajectory")

ly  = pred_df["financial_year"].max()
prv = ly - 1

lat = pred_df[pred_df["financial_year"] == ly].copy()
prv_df = pred_df[pred_df["financial_year"] == prv].copy()

if not prv_df.empty:
    mg = lat.merge(
        prv_df[["state", "district", "person_days_lakhs"]].rename(
            columns={"person_days_lakhs": "prev_actual"}
        ),
        on=["state", "district"], how="inner",
    )
    mg["predicted_chg"]    = mg["predicted_persondays"] - mg["prev_actual"]
    mg["predicted_chg_pct"]= (mg["predicted_chg"] / mg["prev_actual"] * 100).round(2)

    declining = mg[mg["predicted_chg"] < 0].copy().nsmallest(20, "predicted_chg")
    declining["label"] = declining["district"] + " · " + declining["state"]

    if not declining.empty:
        col_risk, col_info = st.columns([2, 1])
        with col_risk:
            fig1 = go.Figure()
            fig1.add_bar(
                x=declining["predicted_chg"],
                y=declining["label"],
                orientation="h",
                marker=dict(
                    color=declining["predicted_chg_pct"],
                    colorscale=[[0, "#7F1D1D"], [1, "#FCA5A5"]],
                    showscale=False,
                    opacity=0.85,
                ),
                customdata=list(zip(
                    declining["state"], declining["district"],
                    declining["prev_actual"].round(2),
                    declining["predicted_persondays"].round(2),
                    declining["predicted_chg"].round(2),
                    declining["predicted_chg_pct"],
                )),
                hovertemplate=(
                    "<b>%{customdata[1]}</b> · %{customdata[0]}<br>"
                    "Actual: %{customdata[2]}L<br>"
                    "Predicted: %{customdata[3]}L<br>"
                    "Change: <b>%{customdata[4]:+.2f}L</b> (%{customdata[5]:+.1f}%)"
                    "<extra></extra>"
                ),
            )
            l1 = {**PLOTLY_LAYOUT}
            l1.update(dict(
                height=max(380, len(declining) * 26),
                title=dict(text=f"Districts with Declining Predicted Employment · FY{prv}→{ly}",
                           font=dict(family="Fraunces, serif", size=13, color="#1C1917")),
                xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Predicted Change (Lakh PD)"),
                yaxis=dict(**PLOTLY_LAYOUT["yaxis"]),
                bargap=0.28, showlegend=False,
            ))
            fig1.update_layout(**l1)
            st.plotly_chart(fig1, width="stretch", config={"displayModeBar": False})

        with col_info:
            st.markdown(f"""
<div style="background:#FEF2F2; border:1px solid #FECACA; border-left:3px solid #DC2626;
            border-radius:8px; padding:1.1rem 1.2rem; margin-bottom:0.8rem;">
  <p style="font-family:'DM Mono',monospace; font-size:0.56rem; letter-spacing:2px;
            text-transform:uppercase; color:#DC2626; margin:0 0 8px 0;">Risk Alert</p>
  <p style="font-family:'Fraunces',serif; font-size:1.6rem; font-weight:600;
            color:#7F1D1D; margin:0 0 4px 0;">{len(declining)}</p>
  <p style="font-family:'Source Serif 4',serif; font-size:0.82rem; color:#991B1B;
            margin:0; line-height:1.5;">
    Districts predicted to see employment decline next cycle.
    Avg change: <strong>{declining['predicted_chg'].mean():+.2f}L</strong> person-days.
  </p>
</div>

<div style="background:#FFFFFF; border:1px solid #E7E5E4;
            border-radius:8px; padding:1rem 1.1rem;">
  <p style="font-family:'DM Mono',monospace; font-size:0.56rem; letter-spacing:2px;
            text-transform:uppercase; color:#A8A29E; margin:0 0 8px 0;">Worst Decline</p>
  <p style="font-family:'Fraunces',serif; font-size:1.1rem; font-weight:600;
            color:#1C1917; margin:0 0 2px 0;">{declining.iloc[0]['district']}</p>
  <p style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#78716C; margin:0;">
    {declining.iloc[0]['state']} · {declining.iloc[0]['predicted_chg']:+.2f}L
  </p>
</div>
""", unsafe_allow_html=True)
    else:
        st.success("✅ No districts show predicted employment decline.")
else:
    st.info("Previous year data unavailable for trend comparison.")

st.markdown("---")

# ── Section B: Efficiency leaders & laggards ──────────────────────────────────
section_label("B. Cost Efficiency — Leaders & Laggards")

eff_df = (
    pred_df.groupby(["state", "district"], as_index=False)
    .agg(
        avg_actual      =("person_days_lakhs",   "mean"),
        avg_predicted   =("predicted_persondays", "mean"),
        avg_error       =("prediction_error",     "mean"),
    )
)

if not opt_df.empty and "persondays_per_lakh" in opt_df.columns:
    eff_sub = opt_df[["state", "district", "persondays_per_lakh"]].drop_duplicates(["state", "district"])
    eff_df  = eff_df.merge(eff_sub, on=["state", "district"], how="left")

    top_eff = eff_df.nlargest(12, "persondays_per_lakh")
    bot_eff = eff_df.nsmallest(12, "persondays_per_lakh")

    col_e1, col_e2 = st.columns(2)
    for col_e, sub, title_str, c in [
        (col_e1, top_eff, "Top 12 Most Efficient", GREEN),
        (col_e2, bot_eff, "Bottom 12 Least Efficient", RED),
    ]:
        with col_e:
            sub = sub.copy()
            sub["label"] = sub["district"] + " · " + sub["state"]
            fig_e = go.Figure()
            fig_e.add_bar(
                x=sub["persondays_per_lakh"],
                y=sub["label"],
                orientation="h",
                marker=dict(color=c, opacity=0.78),
                hovertemplate="<b>%{y}</b><br>%{x:.4f} PD/₹L<extra></extra>",
            )
            l_e = {**PLOTLY_LAYOUT}
            l_e.update(dict(
                height=340,
                title=dict(text=title_str, font=dict(family="Fraunces, serif", size=13, color="#1C1917")),
                xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="PD per ₹ Lakh"),
                yaxis=dict(**PLOTLY_LAYOUT["yaxis"], autorange="reversed"),
                bargap=0.25, showlegend=False,
            ))
            fig_e.update_layout(**l_e)
            st.plotly_chart(fig_e, width="stretch", config={"displayModeBar": False})
else:
    st.info("Run optimizer pipeline to see efficiency rankings.")

st.markdown("---")

# ── Section C: State-level LP opportunities ───────────────────────────────────
section_label("C. LP Reallocation Opportunities by State")

if not opt_df.empty and "persondays_gain" in opt_df.columns:
    state_gain = (
        opt_df.groupby("state", as_index=False)
        .agg(
            total_gain =("persondays_gain",        "sum"),
            n_districts=("district",               "count"),
            avg_eff    =("persondays_per_lakh",    "mean"),
            total_bud  =("budget_allocated_lakhs", "sum"),
        )
        .sort_values("total_gain", ascending=False)
    )
    state_gain["gain_per_dist"] = (state_gain["total_gain"] / state_gain["n_districts"]).round(3)

    fig_s = go.Figure()
    fig_s.add_bar(
        x=state_gain["state"],
        y=state_gain["total_gain"],
        marker=dict(
            color=state_gain["total_gain"],
            colorscale=[[0, "#FEF3C7"], [0.5, "#FB923C"], [1, "#7C2D12"]],
            showscale=False,
            opacity=0.85,
        ),
        customdata=list(zip(
            state_gain["state"],
            state_gain["total_gain"].round(2),
            state_gain["n_districts"],
            state_gain["avg_eff"].round(4),
            state_gain["total_bud"].round(0),
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Total PD Gain: <b>%{customdata[1]:+.2f}L</b><br>"
            "Districts: %{customdata[2]}<br>"
            "Avg Efficiency: %{customdata[3]} PD/₹L<br>"
            "Total Budget: ₹%{customdata[4]:,.0f}L"
            "<extra></extra>"
        ),
    )
    l_s = {**PLOTLY_LAYOUT}
    l_s.update(dict(
        height=360,
        title=dict(text="Total LP Person-Day Gain by State",
                   font=dict(family="Fraunces, serif", size=14, color="#1C1917")),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="State", tickangle=-35),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Total PD Gain (Lakh)"),
        bargap=0.3,
    ))
    fig_s.update_layout(**l_s)
    st.plotly_chart(fig_s, width="stretch", config={"displayModeBar": False})

    with st.expander("📋 State-Level Summary Table"):
        st.dataframe(state_gain.round(3), width="stretch", hide_index=True)
else:
    st.info("No optimizer data — run `python main.py --stage 3`.")

st.markdown("---")

# ── Section D: National trend analysis ───────────────────────────────────────
section_label("D. National Employment Trend & COVID Impact")

if not trend.empty:
    fig_t = go.Figure()
    fig_t.add_scatter(
        x=trend["financial_year"], y=trend["total_persondays"],
        name="Total PD (Lakh)", mode="lines+markers",
        fill="tozeroy", fillcolor="rgba(251,146,60,0.07)",
        line=dict(color=SAFFRON, width=2.5),
        marker=dict(size=7, color=SAFFRON),
    )
    if 2020 in trend["financial_year"].values:
        fig_t.add_vline(
            x=2020, line_dash="dot", line_color=RED, line_width=1.5,
            annotation_text="COVID surge",
            annotation_font=dict(color=RED, size=9, family="DM Mono, monospace"),
        )
    l_t = {**PLOTLY_LAYOUT}
    l_t.update(dict(
        height=260,
        title=dict(text="National Person-Days Trend",
                   font=dict(family="Fraunces, serif", size=13, color="#1C1917")),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Financial Year", dtick=1),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Lakh PD"),
        showlegend=False,
    ))
    fig_t.update_layout(**l_t)
    st.plotly_chart(fig_t, width="stretch", config={"displayModeBar": False})

from utils.ai_summary import render_ai_summary 
render_ai_summary("overview", state_param=state_param)
