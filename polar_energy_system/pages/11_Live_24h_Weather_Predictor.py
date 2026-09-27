import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
for d in [BASE_DIR, SRC_DIR, os.path.dirname(BASE_DIR)]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from src.state_helper import initialize_system_state, load_ml_models_cached
    from src.live_weather_api import fetch_live_polar_weather, STATION_COORDINATES
    from src.optimization import EnergyOptimizer
except ModuleNotFoundError:
    from state_helper import initialize_system_state, load_ml_models_cached
    from live_weather_api import fetch_live_polar_weather, STATION_COORDINATES
    from optimization import EnergyOptimizer

try:
    st.set_page_config(page_title="Live 24h Weather Predictor | PEIS", page_icon="🌐", layout="wide")
except Exception:
    pass
initialize_system_state()

st.title("🌐 Live 24-Hour Polar Satellite Weather Predictor")
st.markdown("Real-time telemetry integration with Open-Meteo Antarctic High-Resolution Weather API.")

station_choice = st.selectbox("Select Polar Station", list(STATION_COORDINATES.keys()))
weather_data = fetch_live_polar_weather(station_choice)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Live Temperature", f"{weather_data['temp_c']} °C")
c2.metric("Wind Speed", f"{weather_data['wind_ms']} m/s")
c3.metric("Solar Irradiance", f"{weather_data['solar_wm2']} W/m²")
c4.metric("Cloud Cover", f"{weather_data['cloud_pct']} %")

st.info(f"Connected to Satellite Station Coordinates: {STATION_COORDINATES[station_choice]}")
