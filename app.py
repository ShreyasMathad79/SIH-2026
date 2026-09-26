import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check root, subfolder, and parent for src
possible_paths = [
    BASE_DIR,
    os.path.join(BASE_DIR, 'src'),
    os.path.join(BASE_DIR, 'polar_energy_system'),
    os.path.join(BASE_DIR, 'polar_energy_system', 'src'),
    os.path.join(BASE_DIR, 'polar energy system'),
    os.path.join(BASE_DIR, 'polar energy system', 'src'),
    os.path.dirname(BASE_DIR),
    os.path.join(os.path.dirname(BASE_DIR), 'src'),
]
for p in possible_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

"""
POLAR ENERGY INTELLIGENCE SYSTEM (PEIS)
AI-Powered Energy Management & SCADA Monitoring Platform for Remote Polar Research Stations
National Centre for Polar and Ocean Research (NCPOR) / Ministry of Earth Sciences (MoES)
Main Application Entrypoint
"""

import os
import sys

# Ensure root directory is on Python path for Cloud deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
try:
    from state_helper import initialize_system_state
    from live_weather_api import fetch_live_polar_weather, STATION_COORDINATES
except ModuleNotFoundError:
    from state_helper import initialize_system_state
    from live_weather_api import fetch_live_polar_weather, STATION_COORDINATES
st.set_page_config(
    page_title="Polar Energy Intelligence System | MoES / NCPOR",
    page_icon="❄",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_system_state()

# Live Satellite Weather Ticker
live_weather = fetch_live_polar_weather("Bharati Station (Antarctica - Larsemann Hills, 69.4°S)")
ticker_icon = "🟢" if live_weather['is_live'] else "🟡"
ticker_text = f"{ticker_icon} {live_weather['status_label']} | Temp: {live_weather['temp_c']}°C | Wind: {live_weather['wind_ms']} m/s | Irradiance: {live_weather['solar_wm2']} W/m² | Cloud: {live_weather['cloud_pct']}%"

st.markdown(f"""
<div style="background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 10px 18px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center;">
    <div style="font-size: 0.9rem; font-weight: 600; color: #38bdf8;">
        {ticker_text}
    </div>
    <div style="font-size: 0.8rem; color: #94a3b8;">
        Updated: {live_weather['timestamp']}
    </div>
</div>
""", unsafe_allow_html=True)

# Glassmorphism Header Banner
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(2, 132, 199, 0.85) 0%, rgba(3, 105, 161, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%); backdrop-filter: blur(16px); padding: 28px; border-radius: 16px; margin-bottom: 24px; border: 1px solid rgba(56, 189, 248, 0.4); box-shadow: 0 10px 30px -5px rgba(0,0,0,0.5);">
    <div>
        <span class="badge-ncpor">MINISTRY OF EARTH SCIENCES (MoES) • NCPOR INDIA</span>
        <h1 style="color: white; margin: 10px 0 6px 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px; text-shadow: 0 0 20px rgba(56,189,248,0.4);">
            ❄ POLAR ENERGY INTELLIGENCE SYSTEM
        </h1>
        <p style="color: #bae6fd; font-size: 1.1rem; margin: 0; font-weight: 500;">
            AI-Driven Decision Support & Microgrid Fuel Optimization under Extreme Polar Conditions
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar System Context & Navigation Guidance
st.sidebar.markdown("### ❄ NCPOR Operations")
st.sidebar.info("""
**Station Profile**:
- **Target**: Maitri / Bharati Station
- **Solar Peak**: 60 kW PV
- **Wind Peak**: 80 kW Turbine
- **BESS Capacity**: 200 kWh
- **Diesel Backup**: 100 kW Gen
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Engine Status")
st.sidebar.success("✓ ML Forecasting Engine: Cached (<10ms)")
st.sidebar.success("✓ PuLP MILP Optimizer: Active")
st.sidebar.success("✓ Satellite Weather API: Connected")

# Interactive Quick Navigation Cards Grid
st.markdown("### 🧭 Quick Control Room Hub")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="polar-card">
        <h3 style="color:#38bdf8; margin-top:0;">📊 Main Dashboard</h3>
        <p style="color:#cbd5e1;">Real-time KPI cards, live 24h trajectory chart, energy survival forecast, and alerts.</p>
    </div>
    """, unsafe_allow_html=True)
    
with c2:
    st.markdown("""
    <div class="polar-card">
        <h3 style="color:#10b981; margin-top:0;">📡 Real-Time SCADA</h3>
        <p style="color:#cbd5e1;">Live IoT sensor telemetry, 32-cell BESS thermal heatmap, engine gauges, and AC waveforms.</p>
    </div>
    """, unsafe_allow_html=True)
    
with c3:
    st.markdown("""
    <div class="polar-card">
        <h3 style="color:#f59e0b; margin-top:0;">⚡ Energy Optimization</h3>
        <p style="color:#cbd5e1;">PuLP MILP hourly dispatch schedule, stacked area breakdown, and fuel savings.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
#### 📌 System Capabilities & Navigation Hub:
- **📊 Main Dashboard**: Real-time KPI summary, energy balance trajectory, and 24-hour survival risk alerts.
- **📡 Real-Time Monitoring**: Live IoT sensor diagnostics, 32-cell BESS thermal heatmap, diesel engine gauges, and SCADA alarm logger.
- **📈 AI Forecast**: Actual vs Predicted Load, Solar, and Wind generation using XGBoost models with strict metrics (MAE, RMSE, R²).
- **⚡ Energy Optimization**: PuLP MILP optimal hourly power dispatch across Solar, Wind, Battery, Diesel, and 3-Tiered Load Shedding.
- **🔀 Energy Flow**: Interactive visual microgrid topology mapping power routing from generation to station consumption.
- **🚨 Alerts & Recommendations**: Real-time AI advisory feed and polar survival warnings.
- **🧪 Simulation Mode**: Test station resilience under 8 extreme polar scenarios (e.g. Severe Blizzard, Polar Night, Fuel Scarcity).
- **⚖️ Before vs After**: Quantitative comparison of Baseline (Diesel-First) vs AI-Optimized strategy proving fuel savings & CO2 reduction.
- **📁 Data Upload**: Upload custom historical station CSV datasets with automated schema validation.
- **⚙️ Model Training**: Retrain load and renewable forecasting models and download `.pkl` binaries.

---
*Developed for National Centre for Polar and Ocean Research (NCPOR), Ministry of Earth Sciences (MoES).*
""")
