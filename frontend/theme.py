"""
theme.py
--------
Shared design system for SchemeImpactNet.
Import and call inject_theme() at the top of every page.

Design: Light, professional, policy-brief aesthetic.
Fonts:  Crimson Pro (headings) + DM Sans (body/mono)
Colors: Slate whites, deep navy text, electric blue accent
"""

import streamlit as st

ACCENT   = "#2563EB"
ACCENT_L = "#EFF6FF"
SUCCESS  = "#16A34A"
WARN     = "#D97706"
DANGER   = "#DC2626"
TEXT     = "#1E293B"
MUTED    = "#64748B"
BORDER   = "#E2E8F0"
BG       = "#F8FAFC"
CARD     = "#FFFFFF"

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F8FAFC;
    color: #1E293B;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px;
}
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Crimson Pro', serif;
    color: #1E293B;
    letter-spacing: -0.3px;
}

/* ── Page header ── */
.sin-page-header {
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 1.25rem;
    margin-bottom: 2rem;
}
.sin-page-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 6px;
}
.sin-page-title {
    font-family: 'Crimson Pro', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #1E293B;
    line-height: 1.15;
    margin: 0 0 4px 0;
}
.sin-page-sub {
    font-size: 0.85rem;
    color: #64748B;
    margin: 0;
}

/* ── KPI cards ── */
.sin-kpi-row {
    display: grid;
    gap: 1px;
    background: #E2E8F0;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.sin-kpi-cell {
    background: #FFFFFF;
    padding: 1.25rem 1.5rem;
}
.sin-kpi-num {
    font-family: 'Crimson Pro', serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #1E293B;
    line-height: 1;
    margin-bottom: 3px;
}
.sin-kpi-num.accent  { color: #2563EB; }
.sin-kpi-num.success { color: #16A34A; }
.sin-kpi-num.warn    { color: #D97706; }
.sin-kpi-num.danger  { color: #DC2626; }
.sin-kpi-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94A3B8;
    font-weight: 500;
}

/* ── Insight cards ── */
.sin-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid transparent;
}
.sin-card.blue   { border-left-color: #2563EB; }
.sin-card.green  { border-left-color: #16A34A; }
.sin-card.amber  { border-left-color: #D97706; }
.sin-card.red    { border-left-color: #DC2626; }
.sin-card.purple { border-left-color: #7C3AED; }
.sin-card-num {
    font-family: 'Crimson Pro', serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 2px;
}
.sin-card.blue   .sin-card-num { color: #2563EB; }
.sin-card.green  .sin-card-num { color: #16A34A; }
.sin-card.amber  .sin-card-num { color: #D97706; }
.sin-card.red    .sin-card-num { color: #DC2626; }
.sin-card.purple .sin-card-num { color: #7C3AED; }
.sin-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #475569;
    margin-bottom: 4px;
}
.sin-card-body {
    font-size: 0.83rem;
    color: #64748B;
    line-height: 1.5;
}

/* ── Section label ── */
.sin-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #F1F5F9;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sin-section::before {
    content: '';
    display: inline-block;
    width: 20px;
    height: 2px;
    background: #2563EB;
    border-radius: 2px;
}

/* ── Brief box ── */
.sin-brief {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-left: 4px solid #2563EB;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.5rem;
}
.sin-brief-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 10px;
    font-weight: 500;
}
.sin-brief-text {
    font-family: 'Crimson Pro', serif;
    font-size: 1.05rem;
    color: #1E3A5F;
    line-height: 1.75;
}
.sin-brief-text strong { color: #1E293B; font-weight: 600; }

/* ── Data rows ── */
.sin-drow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.85rem;
}
.sin-drow:last-child { border-bottom: none; }
.sin-dname { color: #1E293B; font-weight: 500; }
.sin-dstate { color: #94A3B8; font-size: 0.75rem; }
.sin-dval { font-family: 'Crimson Pro', serif; font-size: 1rem; font-weight: 600; }

/* ── Status badge ── */
.sin-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.sin-badge.live  { background: #DCFCE7; color: #16A34A; }
.sin-badge.warn  { background: #FEF3C7; color: #D97706; }
.sin-badge.error { background: #FEE2E2; color: #DC2626; }

/* ── Sidebar nav ── */
.sin-nav-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #F1F5F9;
}

/* ── Status bar ── */
.sin-status {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #94A3B8;
    letter-spacing: 0.5px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}
.sin-status .dot-live  { color: #16A34A; }
.sin-status .dot-off   { color: #DC2626; }
.sin-status .sep { color: #CBD5E1; }

/* ── Metric card (small) ── */
.sin-metric {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.sin-metric-val {
    font-family: 'Crimson Pro', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #1E293B;
    line-height: 1;
    margin-bottom: 3px;
}
.sin-metric-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94A3B8;
}

/* ── State ranking rows ── */
.sin-rank-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 0.84rem;
}
.sin-rank-row.top    { background: #F0FDF4; border: 1px solid #BBF7D0; }
.sin-rank-row.mid    { background: #FFFBEB; border: 1px solid #FDE68A; }
.sin-rank-row.bottom { background: #FFF1F2; border: 1px solid #FECDD3; }

/* ── Plotly overrides for light theme ── */
.js-plotly-plot .plotly .modebar { background: transparent !important; }
</style>
"""


def inject_theme():
    """Call at top of every page to apply the shared design system."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    st.markdown(f"""
<div class="sin-page-header">
    <div class="sin-page-eyebrow">{eyebrow}</div>
    <div class="sin-page-title">{title}</div>
    {"" if not subtitle else f'<p class="sin-page-sub">{subtitle}</p>'}
</div>
""", unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="sin-section">{text}</div>', unsafe_allow_html=True)


def kpi_row(cells: list, cols: int = None):
    """cells = list of (value, label, color_class)"""
    n = cols or len(cells)
    items = "".join(
        f'<div class="sin-kpi-cell"><div class="sin-kpi-num {c}">{v}</div><div class="sin-kpi-label">{l}</div></div>'
        for v, l, c in cells
    )
    st.markdown(f'<div class="sin-kpi-row" style="grid-template-columns:repeat({n},1fr)">{items}</div>',
                unsafe_allow_html=True)


def insight_card(value, title: str, body: str, color: str = "blue"):
    st.markdown(f"""
<div class="sin-card {color}">
    <div class="sin-card-num">{value}</div>
    <div class="sin-card-title">{title}</div>
    <div class="sin-card-body">{body}</div>
</div>""", unsafe_allow_html=True)


def brief_box(label: str, text: str):
    st.markdown(f"""
<div class="sin-brief">
    <div class="sin-brief-label">{label}</div>
    <div class="sin-brief-text">{text}</div>
</div>""", unsafe_allow_html=True)


# Shared Plotly layout for light theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFAFA",
    font=dict(family="DM Sans", color="#1E293B", size=12),
    margin=dict(l=0, r=0, t=36, b=0),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        font=dict(size=11),
    ),
    coloraxis_colorbar=dict(
        tickfont=dict(color="#64748B", size=10),
        title=dict(font=dict(color="#64748B", size=11)),
        thickness=12,
    ),
)

PLOTLY_AXES = dict(
    gridcolor="#F1F5F9",
    linecolor="#E2E8F0",
    tickfont=dict(color="#64748B", size=11),
    title_font=dict(color="#475569", size=12),
    zerolinecolor="#E2E8F0",
)
