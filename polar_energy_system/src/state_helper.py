"""
State Helper Module for Streamlit App
Handles high-efficiency cached loading of ML models, session state initialization, data persistence, and modern Glassmorphism CSS injection.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

from src.preprocessing import load_and_preprocess_data
from src.load_forecasting import LoadForecaster
from src.renewable_forecasting import RenewableForecaster
from src.battery import BatteryStorage
from src.diesel import DieselGenerator
from src.optimization import EnergyOptimizer
from src.simulation import generate_scenario_weather
from src.evaluation import calculate_survival_forecast, compare_baseline_vs_optimized

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(BASE_DIR, "data", "sample", "polar_station_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ACTIVE_CSV = os.path.join(BASE_DIR, "data", "active_dataset.csv")
ACTIVE_META = os.path.join(BASE_DIR, "data", "active_dataset_meta.json")
SYSTEM_STATE_JSON = os.path.join(BASE_DIR, "data", "persistent_system_state.json")

def save_active_dataset(df: pd.DataFrame, label: str):
    """Saves active dataset and metadata to disk so it persists across server and device reboots."""
    os.makedirs(os.path.dirname(ACTIVE_CSV), exist_ok=True)
    df.to_csv(ACTIVE_CSV, index=False)
    meta = {
        'label': label,
        'saved_at': datetime.now().isoformat(),
        'rows': len(df)
    }
    with open(ACTIVE_META, 'w') as f:
        json.dump(meta, f, indent=2)

def load_persistent_dataset():
    """Loads saved active dataset from disk if present."""
    if os.path.exists(ACTIVE_CSV) and os.path.exists(ACTIVE_META):
        try:
            df = pd.read_csv(ACTIVE_CSV)
            df = load_and_preprocess_data(df)
            with open(ACTIVE_META, 'r') as f:
                meta = json.load(f)
            return df, meta.get('label', 'Persistent Active Dataset')
        except Exception:
            pass
    return None, None

def reset_to_sample_dataset():
    """Removes custom persistent dataset and reverts to default station telemetry data."""
    if os.path.exists(ACTIVE_CSV):
        try:
            os.remove(ACTIVE_CSV)
        except Exception:
            pass
    if os.path.exists(ACTIVE_META):
        try:
            os.remove(ACTIVE_META)
        except Exception:
            pass

def save_persistent_state(data: dict):
    """Saves arbitrary key-value session state parameters to disk."""
    os.makedirs(os.path.dirname(SYSTEM_STATE_JSON), exist_ok=True)
    existing = load_persistent_state()
    existing.update(data)
    with open(SYSTEM_STATE_JSON, 'w') as f:
        json.dump(existing, f, indent=2)

def load_persistent_state() -> dict:
    """Loads persistent session state dictionary from disk."""
    if os.path.exists(SYSTEM_STATE_JSON):
        try:
            with open(SYSTEM_STATE_JSON, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def inject_polar_css():
    st.markdown("""
        <style>
        /* Modern Polar Glassmorphism & Cyber Theme */
        .stApp {
            background: radial-gradient(circle at 50% -20%, #0f2b48 0%, #081121 60%, #030712 100%);
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        .stSidebar {
            background: rgba(7, 13, 25, 0.85) !important;
            backdrop-filter: blur(16px);
            border-right: 1px solid rgba(56, 189, 248, 0.15);
        }
        /* Glassmorphism Metric Cards */
        .stMetric {
            background: rgba(15, 23, 42, 0.65) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(56, 189, 248, 0.2) !important;
            border-radius: 12px !important;
            padding: 16px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .stMetric:hover {
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.5) !important;
        }
        .stMetric label {
            color: #94a3b8 !important;
            font-size: 0.80rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .stMetric [data-testid="stMetricValue"] {
            color: #38bdf8 !important;
            font-weight: 800 !important;
            font-size: 1.6rem !important;
            text-shadow: 0 0 12px rgba(56, 189, 248, 0.3);
        }
        /* Custom Polar Cards */
        .polar-card {
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(51, 65, 85, 0.6);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 22px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        }
        .alert-card-critical {
            background: rgba(239, 68, 68, 0.15);
            backdrop-filter: blur(8px);
            border: 1px solid #ef4444;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
            color: #fca5a5;
        }
        .alert-card-warning {
            background: rgba(245, 158, 11, 0.15);
            backdrop-filter: blur(8px);
            border: 1px solid #f59e0b;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
            color: #fde047;
        }
        .alert-card-success {
            background: rgba(16, 185, 129, 0.15);
            backdrop-filter: blur(8px);
            border: 1px solid #10b981;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
            color: #6ee7b7;
        }
        .alert-card-info {
            background: rgba(56, 189, 248, 0.15);
            backdrop-filter: blur(8px);
            border: 1px solid #38bdf8;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
            color: #93c5fd;
        }
        .badge-ncpor {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 0 10px rgba(2, 132, 199, 0.5);
        }
        /* Smooth Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #070d19;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #38bdf8;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_ml_models_cached():
    """Cached singleton loader for trained ML models."""
    load_model = LoadForecaster()
    solar_model = RenewableForecaster(resource_type='solar')
    wind_model = RenewableForecaster(resource_type='wind')
    
    load_path = os.path.join(MODELS_DIR, "load_model.pkl")
    solar_path = os.path.join(MODELS_DIR, "solar_model.pkl")
    wind_path = os.path.join(MODELS_DIR, "wind_model.pkl")
    
    if os.path.exists(load_path):
        load_model.load(load_path)
    if os.path.exists(solar_path):
        solar_model.load(solar_path)
    if os.path.exists(wind_path):
        wind_model.load(wind_path)
        
    return load_model, solar_model, wind_model

@st.cache_data(ttl=600, show_spinner=False)
def load_sample_dataset_cached():
    """Cached dataset loader."""
    if os.path.exists(SAMPLE_CSV):
        return load_and_preprocess_data(SAMPLE_CSV)
    return None

def initialize_system_state():
    inject_polar_css()
    
    if 'data_df' not in st.session_state:
        df, label = load_persistent_dataset()
        if df is not None:
            st.session_state['data_df'] = df
            st.session_state['data_source_label'] = label
        else:
            df = load_sample_dataset_cached()
            st.session_state['data_df'] = df
            st.session_state['data_source_label'] = "Simulated Polar Station Telemetry Dataset (Default)"
        
    load_model, solar_model, wind_model = load_ml_models_cached()
    st.session_state['load_model'] = load_model
    st.session_state['solar_model'] = solar_model
    st.session_state['wind_model'] = wind_model

    p_state = load_persistent_state()

    if 'active_scenario' not in st.session_state:
        st.session_state['active_scenario'] = p_state.get('active_scenario', "Normal Day")
        
    if 'battery' not in st.session_state:
        st.session_state['battery'] = BatteryStorage()
        
    if 'diesel' not in st.session_state:
        st.session_state['diesel'] = DieselGenerator()
        
    if 'optimizer' not in st.session_state:
        st.session_state['optimizer'] = EnergyOptimizer()

def run_system_forecast_and_optimization(scenario_name="Normal Day", horizon_hours=24):
    """
    Executes forecast models + PuLP optimizer for given scenario over horizon_hours with caching.
    """
    df = st.session_state.get('data_df') if hasattr(st, 'session_state') else None
    load_model, solar_model, wind_model = load_ml_models_cached()
    optimizer = st.session_state.get('optimizer') or EnergyOptimizer()
    battery = st.session_state.get('battery') or BatteryStorage()
    diesel = st.session_state.get('diesel') or DieselGenerator()
    
    # 1. Weather Profile
    weather_df = generate_scenario_weather(scenario_name, hours=horizon_hours)
    
    # 2. Last rows for lag calculation
    if df is not None and len(df) >= 24:
        last_rows = df.iloc[-24:].copy()
    else:
        last_rows = weather_df.copy()
        last_rows['historical_load'] = 40.0
        last_rows['solar_generation'] = 10.0
        last_rows['wind_generation'] = 15.0

    # 3. Forecasts
    if load_model.is_trained:
        load_pred = load_model.forecast_horizon(last_rows, weather_df, horizon_hours=horizon_hours)
    else:
        load_pred = (40.0 + (-weather_df['temperature']) * 0.7).values.tolist()
        
    if solar_model.is_trained:
        solar_pred = solar_model.forecast_horizon(last_rows, weather_df, horizon_hours=horizon_hours)
    else:
        solar_pred = (weather_df['solar_irradiance'] * 0.05).values.tolist()
        
    if wind_model.is_trained:
        wind_pred = wind_model.forecast_horizon(last_rows, weather_df, horizon_hours=horizon_hours)
    else:
        wind_pred = np.clip((weather_df['wind_speed'] - 3.0) * 3.0, 0, 80).values.tolist()
        
    init_soc = battery.current_soc
    if scenario_name == "Low Battery Reserve":
        init_soc = 22.0
    elif scenario_name == "Low Diesel Fuel":
        diesel.current_fuel_liters = 400.0

    # 4. PuLP Optimization
    timestamps = [t.strftime("%H:00") for t in weather_df['timestamp']]
    opt_df = optimizer.optimize_dispatch(
        load_forecast_kw=load_pred,
        solar_forecast_kw=solar_pred,
        wind_forecast_kw=wind_pred,
        initial_soc=init_soc,
        timestamps=timestamps
    )
    
    # 5. Risk Survival & Comparison
    survival_metrics = calculate_survival_forecast(
        opt_df,
        current_fuel_liters=diesel.current_fuel_liters,
        battery_capacity_kwh=battery.capacity_kwh,
        min_soc=battery.min_soc
    )
    
    comparison_metrics = compare_baseline_vs_optimized(
        load_forecast_kw=load_pred,
        solar_forecast_kw=solar_pred,
        wind_forecast_kw=wind_pred,
        optimized_df=opt_df,
        initial_soc=init_soc,
        fuel_cost_per_liter=diesel.fuel_cost_per_liter
    )
    
    return {
        'weather_df': weather_df,
        'load_forecast': load_pred,
        'solar_forecast': solar_pred,
        'wind_forecast': wind_pred,
        'optimization_df': opt_df,
        'survival_metrics': survival_metrics,
        'comparison_metrics': comparison_metrics
    }
