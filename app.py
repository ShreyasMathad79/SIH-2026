"""
POLAR ENERGY INTELLIGENCE SYSTEM (PEIS)
AI-Powered Energy Management & SCADA Monitoring Platform for Remote Polar Research Stations
National Centre for Polar and Ocean Research (NCPOR) / Ministry of Earth Sciences (MoES)
Main Application Entrypoint
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE_DIR) in ['pages', 'src']:
    BASE_DIR = os.path.dirname(BASE_DIR)

SRC_DIR = os.path.join(BASE_DIR, "src")

for d in [BASE_DIR, SRC_DIR]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import streamlit as st

st.set_page_config(
    page_title="Polar Energy Intelligence System | MoES / NCPOR",
    page_icon="❄",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from state_helper import initialize_system_state
except ModuleNotFoundError:
    from src.state_helper import initialize_system_state

initialize_system_state()

# Global Sidebar System Context & Engine Status
with st.sidebar:
    st.markdown("### ❄ NCPOR Operations")
    st.info("""
**Station Profile**:
- **Target**: Maitri / Bharati Station
- **Solar Peak**: 60 kW PV
- **Wind Peak**: 80 kW Turbine
- **BESS Capacity**: 200 kWh
- **Diesel Backup**: 100 kW Gen
""")
    st.markdown("---")
    st.markdown("### 📊 Engine Status")
    st.success("✓ ML Forecasting Engine: Active")
    st.success("✓ PuLP MILP Optimizer: Active")
    st.success("✓ Satellite Weather API: Connected")
    st.markdown("---")

# Navigation setup using st.Page and st.navigation
pages = {
    "📊 Core Operations": [
        st.Page("pages/01_Main_Dashboard.py", title="Main Dashboard", icon="📊", default=True),
        st.Page("pages/10_Realtime_Monitoring.py", title="Real-Time SCADA", icon="📡"),
    ],
    "📈 AI & Optimization": [
        st.Page("pages/02_AI_Forecast.py", title="AI Forecast Engine", icon="📈"),
        st.Page("pages/03_Energy_Optimization.py", title="Energy Optimization", icon="⚡"),
        st.Page("pages/04_Energy_Flow.py", title="Microgrid Flow Topology", icon="🔀"),
        st.Page("pages/05_Alerts_and_Recommendations.py", title="Alerts & Recommendations", icon="🚨"),
    ],
    "🧪 Resilience & Simulation": [
        st.Page("pages/06_Simulation_Mode.py", title="Polar Simulation Mode", icon="🧪"),
        st.Page("pages/07_Before_vs_After.py", title="Before vs After Analysis", icon="⚖️"),
    ],
    "📁 Data & Model Ops": [
        st.Page("pages/08_Data_Upload.py", title="Data Upload & Management", icon="📁"),
        st.Page("pages/09_Model_Training.py", title="Model Training & Export", icon="⚙️"),
    ],
    "🛰️ Satellite Weather": [
        st.Page("pages/11_Live_24h_Weather_Predictor.py", title="Live 24h Weather Predictor", icon="🌤️"),
        st.Page("pages/12_Live_7_Day_Weather_Predictor.py", title="Live 7-Day Weather Predictor", icon="📅"),
    ]
}

pg = st.navigation(pages)
pg.run()
