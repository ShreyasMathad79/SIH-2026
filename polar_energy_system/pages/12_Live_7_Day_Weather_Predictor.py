import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(BASE_DIR) in ['pages', 'src']:
    BASE_DIR = os.path.dirname(BASE_DIR)

SRC_DIR = os.path.join(BASE_DIR, 'src')
for d in [BASE_DIR, SRC_DIR]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

try:
    from src.state_helper import initialize_system_state, load_ml_models_cached
    from src.live_weather_api import fetch_live_polar_weather, STATION_COORDINATES
    from src.optimization import EnergyOptimizer
except ModuleNotFoundError:
    from state_helper import initialize_system_state, load_ml_models_cached
    from live_weather_api import fetch_live_polar_weather, STATION_COORDINATES
    from optimization import EnergyOptimizer

try:
    st.set_page_config(page_title="Live 7-Day Weather Predictor | PEIS", page_icon="🌐", layout="wide")
except Exception:
    pass
initialize_system_state()

st.title("🌐 Live 7-Day Polar Weather & Renewable Horizon Predictor")
st.markdown("7-Day Extended Forecast for Antactic Expedition Logistics & Fuel Planning.")

station_choice = st.selectbox("Select Polar Station", list(STATION_COORDINATES.keys()), key="7d_station")
weather_data = fetch_live_polar_weather(station_choice)

st.success("7-Day Satellite Weather Feed Synced & Active")
st.write(f"Station: {station_choice} | Coordinates: {STATION_COORDINATES[station_choice]}")
