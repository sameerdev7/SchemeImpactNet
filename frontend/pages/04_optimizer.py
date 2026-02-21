"""Page 4 — Budget Optimizer (interactive what-if)."""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API = "http://localhost:8000"
st.set_page_config(page_title="Budget Optimizer", page_icon="💰", layout="wide")
st.title("💰 Budget Optimizer")
st.markdown("Reallocate MNREGA budget across districts to **maximize employment generated**")


@st.cache_data(ttl=300)
def fetch(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params or {}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({endpoint}): {e}")
        return None


def run_optimizer_api(payload):
    try:
        r = requests.post(f"{API}/optimizer/run", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Optimizer API error: {e}")
        return None


def safe_df(data) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, list) and len(data) > 0:
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


def resolve_cols(df: pd.DataFrame):
    sq  = "sq_persondays"  if "sq_persondays"  in df.columns else None
    opt = "opt_persondays" if "opt_persondays"  in df.columns else None
    return sq, opt


def _render_charts(df: pd.DataFrame, scope: str):
    sq_col, opt_col = resolve_cols(df)
    gain_col = "persondays_gain" if "persondays_gain" in df.columns else None

    needed = {"budget_allocated_lakhs", "optimized_budget", "budget_change",
              "persondays_per_lakh", "budget_change_pct"}
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"Missing columns for charts: {missing}")
        st.dataframe(df.head(10))
        return

    col_a, col_b = st.columns(2)

    with col_a:
        show = pd.concat([
            df.nlargest(10, "budget_change"),
            df.nsmallest(10, "budget_change"),
        ]).drop_duplicates().sort_values("budget_change")

        fig = go.Figure()
        fig.add_bar(y=show["district"], x=show["budget_allocated_lakhs"],
                    name="Current", orientation="h", marker_color="#90CAF9")
        fig.add_bar(y=show["district"], x=show["optimized_budget"],
                    name="Optimized", orientation="h", marker_color="#1565C0")
        fig.update_layout(barmode="group", title="Budget Reallocation",
                          xaxis_title="Budget (Rs. lakh)", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        color_col = gain_col if gain_col else "budget_change_pct"
        hover_cols = {c: True for c in ["state", "district", gain_col]
                      if c and c in df.columns}
        fig2 = px.scatter(
            df, x="persondays_per_lakh", y="budget_change_pct",
            color=color_col, color_continuous_scale="RdYlGn",
            hover_data=hover_cols,
            title="Efficiency vs Budget Change",
            labels={"persondays_per_lakh": "Efficiency (PD per Rs. lakh)",
                    "budget_change_pct": "Budget Change (%)"},
            height=500,
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig2, use_container_width=True)

    show_cols = [c for c in ["state", "district", "budget_allocated_lakhs",
                              "optimized_budget", "budget_change_pct", gain_col]
                 if c and c in df.columns]

    col_up, col_down = st.columns(2)
    with col_up:
        st.subheader("📈 Top districts to increase")
        if gain_col:
            st.dataframe(df.nlargest(8, gain_col)[show_cols].round(2),
                         use_container_width=True, hide_index=True)
    with col_down:
        st.subheader("📉 Top districts to reduce")
        if gain_col:
            st.dataframe(df.nsmallest(8, gain_col)[show_cols].round(2),
                         use_container_width=True, hide_index=True)

    with st.expander("📋 Full table"):
        sort_col = gain_col or "budget_change"
        st.dataframe(df.sort_values(sort_col, ascending=False),
                     use_container_width=True, hide_index=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Optimizer Settings")

states      = fetch("/districts/states") or []
scope       = st.sidebar.selectbox("Scope", ["All-India"] + states)
state_param = None if scope == "All-India" else scope

budget_scale = st.sidebar.slider("Budget scale", 0.7, 1.5, 1.0, 0.05,
    help="1.0 = same budget. 1.1 = +10%.")
min_frac = st.sidebar.slider("Min alloc per district", 0.1, 0.8, 0.4, 0.05)
max_frac = st.sidebar.slider("Max alloc per district", 1.1, 4.0, 2.5, 0.1)

run_btn = st.sidebar.button("🚀 Run Optimizer", type="primary", use_container_width=True)

# ── Pre-computed results (default view) ───────────────────────────────────────
params      = {"state": state_param} if state_param else {}
precomputed = fetch("/optimizer/results", params)
df_pre      = safe_df(precomputed)

if not df_pre.empty and not run_btn:
    sq_col, opt_col = resolve_cols(df_pre)
    if sq_col and opt_col:
        sq_total  = df_pre[sq_col].sum()
        opt_total = df_pre[opt_col].sum()
        gain      = opt_total - sq_total
        gain_pct  = gain / sq_total * 100 if sq_total else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scope",           scope)
        c2.metric("Status Quo PD",   f"{sq_total:,.1f}L")
        c3.metric("Optimized PD",    f"{opt_total:,.1f}L", f"+{gain:.1f}L")
        c4.metric("Efficiency Gain", f"+{gain_pct:.2f}%")
    else:
        st.warning(f"Unexpected columns: {list(df_pre.columns)}")
    st.markdown("---")
    _render_charts(df_pre, scope)

elif df_pre.empty and not run_btn:
    st.info("Click **Run Optimizer** to generate results.")

# ── Live run ──────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running LP optimizer..."):
        result = run_optimizer_api({
            "state": state_param,
            "budget_scale": budget_scale,
            "min_fraction": min_frac,
            "max_fraction": max_frac,
        })

    if result:
        st.success("✅ Optimization complete!")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Scope",         result.get("scope", ""))
        c2.metric("Total Budget",  f"Rs.{result.get('total_budget_lakhs', 0):,.0f}L")
        c3.metric("Status Quo PD", f"{result.get('sq_persondays_total', 0):,.1f}L")
        c4.metric("Optimized PD",  f"{result.get('opt_persondays_total', 0):,.1f}L",
                  f"+{result.get('gain_lakhs', 0):.1f}L")
        c5.metric("Gain",          f"+{result.get('gain_pct', 0):.2f}%")
        st.markdown("---")
        df_live = safe_df(result.get("districts", []))
        if not df_live.empty:
            _render_charts(df_live, result.get("scope", scope))
