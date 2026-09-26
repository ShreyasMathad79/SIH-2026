"""
Model Training Page
Interactive ML model training pipeline for Load, Solar, and Wind forecasting models.
"""

import os
import streamlit as st
import pandas as pd
import joblib

from src.state_helper import initialize_system_state
from src.preprocessing import train_val_test_split_chronological
from src.load_forecasting import LoadForecaster
from src.renewable_forecasting import RenewableForecaster

initialize_system_state()

st.title("⚙️ Train AI Forecasting Models")
st.caption("Chronological Training Pipeline & Hyperparameter Tuning | MoES / NCPOR")

df = st.session_state.get('data_df')

if df is None:
    st.error("No dataset loaded. Please upload a dataset in Data Upload section.")
    st.stop()

st.markdown("### 1. Chronological Data Split Configuration")

c_s1, c_s2 = st.columns(2)
with c_s1:
    train_pct = st.slider("Train Split (%)", 50, 85, 70)
with c_s2:
    val_pct = st.slider("Validation Split (%)", 5, 25, 15)

test_pct = 100 - train_pct - val_pct
st.info(f"**Split Ratio**: Train {train_pct}% | Validation {val_pct}% | Test {test_pct}% (Chronological Order, No Shuffling)")

train_df, val_df, test_df = train_val_test_split_chronological(df, train_ratio=train_pct/100.0, val_ratio=val_pct/100.0)

m_c1, m_c2, m_c3, m_c4 = st.columns(4)
m_c1.metric("Total Records", f"{len(df)} h")
m_c2.metric("Train Samples", f"{len(train_df)} h")
m_c3.metric("Val Samples", f"{len(val_df)} h")
m_c4.metric("Test Samples", f"{len(test_df)} h")

st.markdown("---")

st.markdown("### 2. Model Training Controls")

col_tr1, col_tr2, col_tr3, col_tr4 = st.columns(4)

with col_tr1:
    btn_load = st.button("⚡ Train Load Model", use_container_width=True)
with col_tr2:
    btn_solar = st.button("☀️ Train Solar Model", use_container_width=True)
with col_tr3:
    btn_wind = st.button("💨 Train Wind Model", use_container_width=True)
with col_tr4:
    btn_all = st.button("🚀 Train ALL Models", type="primary", use_container_width=True)

models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(models_dir, exist_ok=True)

if btn_load or btn_all:
    with st.spinner("Training Load Forecasting XGBoost Regressor..."):
        load_model = LoadForecaster(model_type='xgboost')
        load_model.train(train_df, val_df)
        load_metrics = load_model.evaluate(test_df)
        load_model.save(os.path.join(models_dir, "load_model.pkl"))
        st.session_state['load_model'] = load_model
        st.success(f"✓ Load Model Trained! Test MAE: {load_metrics['MAE']} kW | RMSE: {load_metrics['RMSE']} kW | R²: {load_metrics['R2']}")

if btn_solar or btn_all:
    with st.spinner("Training Solar Photovoltaic XGBoost Regressor..."):
        solar_model = RenewableForecaster(resource_type='solar', model_type='xgboost')
        solar_model.train(train_df, val_df)
        solar_metrics = solar_model.evaluate(test_df)
        solar_model.save(os.path.join(models_dir, "solar_model.pkl"))
        st.session_state['solar_model'] = solar_model
        st.success(f"✓ Solar Model Trained! Test MAE: {solar_metrics['MAE']} kW | RMSE: {solar_metrics['RMSE']} kW | R²: {solar_metrics['R2']}")

if btn_wind or btn_all:
    with st.spinner("Training Wind Turbine XGBoost Regressor..."):
        wind_model = RenewableForecaster(resource_type='wind', model_type='xgboost')
        wind_model.train(train_df, val_df)
        wind_metrics = wind_model.evaluate(test_df)
        wind_model.save(os.path.join(models_dir, "wind_model.pkl"))
        st.session_state['wind_model'] = wind_model
        st.success(f"✓ Wind Model Trained! Test MAE: {wind_metrics['MAE']} kW | RMSE: {wind_metrics['RMSE']} kW | R²: {wind_metrics['R2']}")

st.markdown("---")

st.markdown("### 3. Model Binaries & Downloads")
col_d1, col_d2, col_d3 = st.columns(3)

load_path = os.path.join(models_dir, "load_model.pkl")
if os.path.exists(load_path):
    with open(load_path, "rb") as f:
        col_d1.download_button("📥 Download load_model.pkl", f, file_name="load_model.pkl", use_container_width=True)

solar_path = os.path.join(models_dir, "solar_model.pkl")
if os.path.exists(solar_path):
    with open(solar_path, "rb") as f:
        col_d2.download_button("📥 Download solar_model.pkl", f, file_name="solar_model.pkl", use_container_width=True)

wind_path = os.path.join(models_dir, "wind_model.pkl")
if os.path.exists(wind_path):
    with open(wind_path, "rb") as f:
        col_d3.download_button("📥 Download wind_model.pkl", f, file_name="wind_model.pkl", use_container_width=True)
