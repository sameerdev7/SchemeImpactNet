# pages/optimizer.py — Budget reallocation optimizer results and live LP runner.


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from theme import inject_theme, page_header, section_label, kpi_html, PLOTLY_LAYOUT, SAFFRON, GREEN, RED, AMBER
from utils.api_client import fetch_states, fetch_optimizer_results, run_optimizer_live

inject_theme()
page_header(
    "◈ Module 04",
    "Budget Optimizer",
    "SciPy LP two-stage proportional reallocation — maximize employment at zero additional cost",
)

# ── Tabs: pre-computed vs live ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Pre-Computed Results", "Run Live Optimizer"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pre-computed results
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    states = fetch_states()
    if not states:
        st.error("⚠️ API offline — run `uvicorn backend.main:app --port 8000`")
        st.stop()

    cs, _ = st.columns([1, 2])
    with cs:
        scope = st.selectbox("State Filter", ["All-India"] + states, key="pre_scope")
    state_param = None if scope == "All-India" else scope

    df = fetch_optimizer_results(state_param)

    if df.empty:
        st.info("No optimizer results — run the pipeline first: `python main.py --stage 3`")
    else:
        # ── Summary KPIs ──────────────────────────────────────────────────────
        sq_total  = df["sq_persondays"].sum()
        opt_total = df["opt_persondays"].sum() if "opt_persondays" in df.columns else sq_total + df["persondays_gain"].sum()
        gain      = df["persondays_gain"].sum()
        gain_pct  = gain / sq_total * 100 if sq_total else 0
        tot_bud   = df["budget_allocated_lakhs"].sum() if "budget_allocated_lakhs" in df.columns else 0
        n_gain    = int((df["persondays_gain"] > 0).sum())
        n_cut     = int((df["persondays_gain"] <= 0).sum())

        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        with kc1: st.markdown(kpi_html(f"{sq_total:,.0f}L",  "Status Quo PD",   "#1C1917"), unsafe_allow_html=True)
        with kc2: st.markdown(kpi_html(f"{opt_total:,.0f}L", "Optimized PD",    GREEN),     unsafe_allow_html=True)
        with kc3: st.markdown(kpi_html(f"{gain:+,.1f}L",     "Net Gain",        GREEN,      "lakh person-days"), unsafe_allow_html=True)
        with kc4: st.markdown(kpi_html(f"{gain_pct:+.2f}%",  "% Uplift",        GREEN,      "budget-neutral"), unsafe_allow_html=True)
        with kc5: st.markdown(kpi_html(f"₹{tot_bud:,.0f}L",  "Total Budget",    "#1C1917",  "unchanged"), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

        # ── Budget change waterfall — top movers ──────────────────────────────
        section_label("Top Budget Movers")

        top_gain = df.nlargest(10, "persondays_gain").copy()
        top_cut  = df.nsmallest(10, "persondays_gain").copy()
        show     = pd.concat([top_gain, top_cut]).drop_duplicates().sort_values("persondays_gain")
        show["label"] = show["district"] + " · " + show["state"]

        fig1 = go.Figure()
        fig1.add_bar(
            x=show["persondays_gain"],
            y=show["label"],
            orientation="h",
            marker=dict(
                color=[GREEN if v > 0 else RED for v in show["persondays_gain"]],
                opacity=0.8,
            ),
            customdata=list(zip(
                show["state"], show["district"],
                show["budget_allocated_lakhs"].round(0) if "budget_allocated_lakhs" in show else [0]*len(show),
                show.get("budget_change_pct", pd.Series([0]*len(show))).round(1),
                show["persondays_gain"].round(2),
                show.get("persondays_per_lakh", pd.Series([0]*len(show))).round(4),
            )),
            hovertemplate=(
                "<b>%{customdata[1]}</b> · %{customdata[0]}<br>"
                "Budget: ₹%{customdata[2]:,.0f}L → %{customdata[3]:+.1f}%<br>"
                "PD Gain: <b>%{customdata[4]:+.2f}L</b><br>"
                "Efficiency: %{customdata[5]} PD/₹L"
                "<extra></extra>"
            ),
        )
        fig1.add_vline(x=0, line_dash="solid", line_color="#1C1917", line_width=1)
        l1 = {**PLOTLY_LAYOUT}
        l1.update(dict(
            height=520,
            title=dict(text="Person-Day Gain by District (Top 10 + Bottom 10)",
                       font=dict(family="Fraunces, serif", size=14, color="#1C1917")),
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Person-Day Gain (Lakh)"),
            yaxis=dict(**PLOTLY_LAYOUT["yaxis"]),
            showlegend=False,
            bargap=0.3,
        ))
        fig1.update_layout(**l1)
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        # ── Efficiency vs budget change scatter ───────────────────────────────
        section_label("Efficiency vs Budget Reallocation")

        if "persondays_per_lakh" in df.columns and "budget_change_pct" in df.columns:
            fig2 = go.Figure()
            fig2.add_scatter(
                x=df["persondays_per_lakh"],
                y=df["budget_change_pct"],
                mode="markers",
                marker=dict(
                    color=df["persondays_gain"],
                    colorscale=[[0, RED], [0.5, "#FED7AA"], [1, GREEN]],
                    size=5, opacity=0.65,
                    colorbar=dict(
                        title=dict(text="PD Gain", font=dict(color="#78716C", size=9)),
                        tickfont=dict(color="#78716C", size=8),
                        thickness=8, len=0.5,
                    ),
                ),
                customdata=list(zip(
                    df["state"], df["district"],
                    df["budget_change_pct"].round(1),
                    df["persondays_gain"].round(2),
                )),
                hovertemplate=(
                    "<b>%{customdata[1]}</b> · %{customdata[0]}<br>"
                    "Budget Δ: %{customdata[2]:+.1f}%<br>"
                    "PD Gain: %{customdata[3]:+.2f}L"
                    "<extra></extra>"
                ),
            )
            fig2.add_hline(y=0, line_dash="dot", line_color="#1C1917", line_width=1)
            l2 = {**PLOTLY_LAYOUT}
            l2.update(dict(
                height=340,
                title=dict(text="Efficiency (PD/₹ Lakh) vs Budget Change %",
                           font=dict(family="Fraunces, serif", size=13, color="#1C1917")),
                xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="PD per ₹ Lakh"),
                yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Budget Change (%)"),
                showlegend=False,
            ))
            fig2.update_layout(**l2)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # ── Full table ────────────────────────────────────────────────────────
        with st.expander("📋 Full Reallocation Table"):
            show_cols = [c for c in [
                "state", "district", "budget_allocated_lakhs", "optimized_budget",
                "budget_change_pct", "sq_persondays", "opt_persondays",
                "persondays_gain", "persondays_gain_pct", "persondays_per_lakh",
            ] if c in df.columns]
            styled = df[show_cols].round(3).sort_values("persondays_gain", ascending=False)
            st.dataframe(styled, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live optimizer
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
<p style="font-family:'Source Serif 4',serif; font-size:0.9rem; color:#57534E;
          line-height:1.7; margin-bottom:1.5rem;">
  Run the SciPy linear-programming optimizer live with custom parameters.
  Results are computed in real-time using the latest district predictions from the database.
</p>
""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    states2 = fetch_states() or []
    with ca:
        scope2      = st.selectbox("State (or All-India)", ["All-India"] + states2, key="live_scope")
        budget_scale = st.slider("Budget Scale", 0.8, 1.5, 1.0, 0.05,
                                  help="1.0 = same total budget; 1.1 = +10% more funds")
    with cb:
        min_frac = st.slider("Min Allocation (floor)", 0.10, 0.60, 0.40, 0.05,
                              help="No district drops below this fraction of its current budget")
        max_frac = st.slider("Max Allocation (cap)", 1.5, 3.0, 2.5, 0.1,
                              help="No district exceeds this multiple of its current budget")

    if st.button("▶ Run Optimizer", type="primary"):
        with st.spinner("Running LP optimization…"):
            result = run_optimizer_live(
                state=None if scope2 == "All-India" else scope2,
                budget_scale=budget_scale,
                min_fraction=min_frac,
                max_fraction=max_frac,
            )

        if result:
            st.success(
                f"✅ Optimization complete — "
                f"Gain: **{result['gain_lakhs']:+,.2f}L** person-days "
                f"({result['gain_pct']:+.2f}%) · "
                f"Total budget: ₹{result['total_budget_lakhs']:,.0f}L"
            )

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SQ Person-Days",  f"{result['sq_persondays_total']:,.1f}L")
            m2.metric("Opt Person-Days", f"{result['opt_persondays_total']:,.1f}L")
            m3.metric("Net Gain",        f"{result['gain_lakhs']:+,.2f}L")
            m4.metric("% Uplift",        f"{result['gain_pct']:+.2f}%")

            # District breakdown
            if result.get("districts"):
                dist_df = pd.DataFrame(result["districts"])

                section_label("District Reallocation Details")
                top10 = dist_df.nlargest(10, "persondays_gain")
                top10["label"] = top10["district"] + " · " + top10["state"]

                fig_live = go.Figure()
                fig_live.add_bar(
                    x=top10["persondays_gain"], y=top10["label"],
                    orientation="h",
                    marker=dict(color=GREEN, opacity=0.8),
                    hovertemplate=(
                        "<b>%{y}</b><br>PD Gain: <b>%{x:+.2f}L</b><extra></extra>"
                    ),
                )
                l_live = {**PLOTLY_LAYOUT}
                l_live.update(dict(
                    height=380, showlegend=False, bargap=0.3,
                    title=dict(text="Top 10 Districts to Increase",
                               font=dict(family="Fraunces, serif", size=13, color="#1C1917")),
                    xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="PD Gain (Lakh)"),
                    yaxis=dict(**PLOTLY_LAYOUT["yaxis"]),
                ))
                fig_live.update_layout(**l_live)
                st.plotly_chart(fig_live, use_container_width=True,
                                config={"displayModeBar": False})

                with st.expander("📋 Full Live Results Table"):
                    st.dataframe(dist_df.round(3), use_container_width=True, hide_index=True)


from utils.ai_summary import render_ai_summary 
render_ai_summary("overview", state_param=state_param)
