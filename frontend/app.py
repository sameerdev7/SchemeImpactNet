"""
frontend/app.py
---------------
Streamlit entry point for SchemeImpactNet.
"""

import streamlit as st

st.set_page_config(
    page_title="SchemeImpactNet",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HEADER SECTION ---
st.title("📊 SchemeImpactNet")
st.markdown("### Machine Learning Framework for Predictive Impact Analysis & Optimization")
st.markdown("---")

# --- SYSTEM STATUS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("📍 **Navigation**")
    st.caption("Access analytics and optimization via the sidebar.")
with col2:
    st.markdown("⚙️ **Backend Status**")
    st.caption("FastAPI Engine: Active (Port 8000)")
with col3:
    st.markdown("📁 **Dataset Scope**")
    st.caption("725 Districts | 10-Year Longitudinal Data")

# --- PROJECT OVERVIEW ---
st.markdown("## Project Overview")
st.write("""
SchemeImpactNet is a machine learning platform designed for the predictive analysis and optimization of Indian 
government schemes. By integrating data from **data.gov.in** and **mospi.gov.in**, the system transitions 
public policy management from reactive administration to proactive, data-driven decision-making. 
The framework focuses on forecasting socio-economic impacts, identifying regional inefficiencies, 
and recommending resource allocations to maximize welfare outcomes.
""")

# --- THE CORE PROBLEM ---
with st.container():
    st.markdown("### Problem Statement")
    st.markdown("""
    Annual expenditure on MNREGA ranges between **₹70,000–90,000 crore**. Current administrative 
    processes often lack the predictive tools necessary to address:
    * **Demand Forecasting**: Predicting if a district will meet employment requirements in the upcoming cycle.
    * **Efficiency Metrics**: Quantifying the value generated per unit of expenditure across diverse geographies.
    * **Resource Allocation**: Identifying where budget shifts can most effectively mitigate rural distress.
    """)

# --- SYSTEM CAPABILITIES ---
st.markdown("---")
st.markdown("### System Capabilities")
cap1, cap2, cap3 = st.columns(3)

with cap1:
    st.markdown("#### 📉 Predictive Analysis")
    st.write("""
    Forecasts district-level person-days using historical trends, rainfall patterns, 
    and agricultural cycles.
    """)

with cap2:
    st.markdown("#### 📋 Efficiency Benchmarking")
    st.write("""
    Analyzes performance variance between districts to identify high-impact 
    implementation models.
    """)

with cap3:
    st.markdown("#### ⚖️ Budget Optimization")
    st.write("""
    Utilizes linear programming to suggest budget reallocations that maximize 
    total employment within fiscal constraints.
    """)

# --- TECHNICAL ARCHITECTURE ---
with st.expander("🛠️ Technical Architecture & Workflow"):
    st.markdown("""
    #### Data Pipeline & ML Core
    * **Data Layer**: Automated ingestion of beneficiary demographics and expenditure records.
    * **Preprocessing**: Feature engineering for 'Impact Scores' and normalization of regional variables.
    * **ML Core Engine**:
        * **Time-Series**: LSTM and XGBoost for longitudinal forecasting.
        * **Clustering**: K-Means for regional performance categorization.
        * **Optimization**: PuLP-based linear programming for resource distribution.
    """)
    
    st.code("""
    Raw Data -> Preprocessing -> ML Engine (XGBoost/LSTM) -> Optimization Layer -> Visualization
    """, language="text")

# --- SIDEBAR ---
st.sidebar.markdown("### Navigation")
st.sidebar.page_link("app.py", label="Overview", icon="🏠")
# Add your additional pages here as they are created
# st.sidebar.page_link("pages/predictions.py", label="Predictions", icon="📈")

st.sidebar.markdown("---")
st.sidebar.markdown("📜 **Project Documentation**")
st.sidebar.caption("SchemeImpactNet v1.0")
st.sidebar.caption("Final Year Engineering Project")
