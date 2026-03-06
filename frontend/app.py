# frontend/app.py — SchemeImpactNet entry point
# Run from project root: streamlit run frontend/app.py

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="SchemeImpactNet",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS first — before anything else ───────────────────────────────────
# Must happen before st.navigation() so sidebar styles are present when nav renders.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,300&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600&family=DM+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif !important; }
.stApp { background-color: #FAF9F7 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1320px !important; }

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #1C1917 !important;
    border-right: none !important;
    min-width: 220px !important;
}
[data-testid="stSidebarContent"] {
    background: #1C1917 !important;
}
section[data-testid="stSidebar"] > div {
    background: #1C1917 !important;
}

/* ── Sidebar text ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #A8A29E !important;
}

/* ── Nav links from st.navigation() ── */
[data-testid="stSidebarNavLink"] {
    border-radius: 5px !important;
    padding: 0.48rem 1rem !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.5px !important;
    color: #A8A29E !important;
    border-left: 2px solid transparent !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebarNavLink"]:hover {
    background: rgba(251,146,60,0.1) !important;
    color: #FB923C !important;
    border-left-color: rgba(251,146,60,0.5) !important;
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(251,146,60,0.15) !important;
    color: #FB923C !important;
    border-left-color: #FB923C !important;
}
[data-testid="stSidebarNavLink"] svg { display: none !important; }

/* ── Sidebar nav section label ── */
[data-testid="stSidebarNavSeparator"] {
    border-color: rgba(255,255,255,0.07) !important;
}

/* ── Collapse button ── */
[data-testid="collapsedControl"] {
    background: #1C1917 !important;
    color: #A8A29E !important;
    border-right: 1px solid #292524 !important;
}
button[kind="header"] { background: transparent !important; }

/* ── Main area typography ── */
h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #1C1917 !important; }
h1 { font-size: 2.2rem !important; font-weight: 600 !important; line-height: 1.15 !important; }
h2 { font-size: 1.5rem !important; font-weight: 600 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }
p  { font-family: 'Source Serif 4', serif !important; color: #292524 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #FFFFFF !important; border: 1px solid #E7E5E4 !important;
    border-radius: 8px !important; padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'DM Mono', monospace !important; font-size: 0.62rem !important;
    letter-spacing: 2px !important; text-transform: uppercase !important; color: #78716C !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important; font-size: 1.85rem !important;
    font-weight: 600 !important; color: #1C1917 !important; line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important; }

/* ── Inputs ── */
[data-testid="stSelectbox"] label p,
[data-testid="stSlider"] label p,
[data-testid="stTextInput"] label p {
    font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important; color: #78716C !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
    background: #1C1917 !important; color: #FAF9F7 !important;
    border: none !important; border-radius: 6px !important; padding: 0.5rem 1.2rem !important;
}
.stButton > button:hover { background: #FB923C !important; color: #C1917 !important;}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #E7E5E4 !important; border-radius: 8px !important; overflow: hidden !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #E7E5E4 !important; border-radius: 8px !important; background: #FFFFFF !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
    font-family: 'DM Mono', monospace !important; font-size: 0.63rem !important;
    color: #A8A29E !important; letter-spacing: 0.3px !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #E7E5E4 !important; margin: 1.5rem 0 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Mono', monospace !important; font-size: 0.68rem !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
}

/* ── Sidebar Gemini key input ── */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: #292524 !important;
    border: 1px solid #44403C !important;
    color: #FAF9F7 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    border-radius: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: #57534E !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar brand + Gemini key input ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:1.4rem 0.75rem 1.2rem 0.75rem;
            border-bottom:1px solid rgba(255,255,255,0.07);
            margin-bottom:0.75rem;">
  <p style="font-family:'DM Mono',monospace; font-size:0.52rem; letter-spacing:4px;
            text-transform:uppercase; color:#FB923C; margin:0 0 8px 0; line-height:1;">
    Policy Analytics
  </p>
  <p style="font-family:'Fraunces',serif; font-size:1.35rem; font-weight:600;
            color:#FAF9F7; line-height:1.1; margin:0;">
    Scheme<br>Impact<em style="color:#FB923C;">Net</em>
  </p>
  <p style="font-family:'DM Mono',monospace; font-size:0.55rem; color:#57534E;
            margin:10px 0 0 0; letter-spacing:0.4px; line-height:1.65;">
    MNREGA · XGBoost · SciPy LP<br>
    7,758 district-years · 2014–2024
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Gemini API key — persisted in session_state, available on every page ──
    st.markdown("""
<div style="padding:0.9rem 0.75rem 0.3rem 0.75rem;">
  <p style="font-family:'DM Mono',monospace; font-size:0.5rem; letter-spacing:3px;
            text-transform:uppercase; color:#57534E; margin:0 0 5px 0;">
    ✦ AI Insights Key
  </p>
</div>
""", unsafe_allow_html=True)

    key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.get("gemini_api_key", ""),
        type="password",
        placeholder="AIza... (optional)",
        help="Enables AI summaries on every page. Free at aistudio.google.com/apikey",
        key="sidebar_gemini_key",
        label_visibility="collapsed",
    )
    if key_input:
        st.session_state["gemini_api_key"] = key_input

    # Status indicator
    if st.session_state.get("gemini_api_key"):
        st.markdown("""
<p style="font-family:'DM Mono',monospace; font-size:0.52rem; color:#16A34A;
          margin:4px 0.75rem 0.5rem 0.75rem; letter-spacing:1px;">
  ✓ AI summaries active
</p>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<p style="font-family:'DM Mono',monospace; font-size:0.5rem; color:#57534E;
          margin:4px 0.75rem 1rem 0.75rem; letter-spacing:0.3px; line-height:1.6;">
  Add key to unlock AI insights<br>
  <a href="https://aistudio.google.com/apikey" target="_blank"
     style="color:#FB923C; text-decoration:none;">aistudio.google.com ↗</a>
</p>""", unsafe_allow_html=True)

# ── Page registry ─────────────────────────────────────────────────────────────
pages = [
    st.Page("pages/home.py",             title="Home",               icon="🏛️", default=True),
    st.Page("pages/overview.py",         title="Overview",           icon="📊"),
    st.Page("pages/districts.py",        title="District Explorer",  icon="🔍"),
    st.Page("pages/predictions.py",      title="Predictions",        icon="🤖"),
    st.Page("pages/optimizer.py",        title="Budget Optimizer",   icon="⚖️"),
    st.Page("pages/spatial.py",          title="Spatial Map",        icon="🗺️"),
    st.Page("pages/insights.py",         title="Strategic Insights", icon="🧠"),
    st.Page("pages/gemini_insights.py",  title="AI Insights", icon="✨"),
]

pg = st.navigation(pages, position="sidebar")
pg.run()
