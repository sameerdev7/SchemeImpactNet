"""
frontend/app.py
---------------
Streamlit entry point for SchemeImpactNet.

Run with:
    streamlit run frontend/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="SchemeImpactNet",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 SchemeImpactNet")
st.markdown("**MNREGA District-Level Forecasting & Budget Optimization**")
st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.info("👈 Use the sidebar to navigate between pages")
col2.success("🚀 Backend: FastAPI on port 8000")
col3.warning("📦 Data: 725 districts × 10 years across 29 states")

st.markdown("""
### What this system does

| Stage | Capability |
|---|---|
| **Stage 1** | Maharashtra baseline — predict next year's person-days per district |
| **Stage 2** | All-India scale — 725 districts with rainfall, poverty, crop season features |
| **Stage 3** | Budget optimizer — reallocate same budget to maximize employment |

### How to use
- **Overview** — State-level trends and key stats
- **District Explorer** — Drill into any district's historical performance  
- **Predictions** — Model forecasts for 2022–2023, filterable by state/district
- **Budget Optimizer** — Run what-if budget scenarios interactively
""")
