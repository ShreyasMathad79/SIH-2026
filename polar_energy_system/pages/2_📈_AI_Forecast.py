"""
AI Forecast Page
Evaluates Machine Learning models for Load, Solar, and Wind forecasting.
Calculates MAE, RMSE, R2, MAPE from test set and displays interactive Actual vs Predicted graphs.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.state_helper import initialize_system_state
from src.preprocessing import train_val_test_split_chronological

initialize_system_state()

st.title("📈 AI Forecasting Models & Performance")
st.caption("Machine Learning Load Demand & Renewable Energy Forecasting Engine | MoES / NCPOR")

df = st.session_state.get('data_df')

if df is None:
    st.error("No dataset available. Please upload a dataset in the Data Upload section.")
    st.stop()

train_df, val_df, test_df = train_val_test_split_chronological(df)

# Model Tabs
tab_load, tab_solar, tab_wind = st.tabs(["⚡ Station Electricity Demand", "☀️ Solar Photovoltaic Generation", "💨 Wind Turbine Generation"])

# 1. LOAD MODEL TAB
with tab_load:
    st.subheader("Station Electricity Demand Model (kW)")
    st.caption("Algorithm: XGBoost Regressor with Cyclical Time Encodings & Lag Features")
    
    load_model = st.session_state['load_model']
    
    if load_model.is_trained:
        eval_metrics = load_model.evaluate(test_df)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE", f"{eval_metrics['MAE']} kW", help="Mean Absolute Error on Test Split")
        m2.metric("RMSE", f"{eval_metrics['RMSE']} kW", help="Root Mean Squared Error on Test Split")
        m3.metric("R² Score", f"{eval_metrics['R2']}", help="Coefficient of Determination")
        m4.metric("MAPE", f"{eval_metrics['MAPE']}%", help="Mean Absolute Percentage Error")
        m5.metric("Test Split Size", f"{len(test_df)} hours", delta=f"Train {len(train_df)}h")
        
        st.markdown("#### Actual vs Predicted Load Demand (Test Set)")
        
        test_timestamps = test_df['timestamp'].values
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_timestamps, y=eval_metrics['actuals'],
            mode='lines', name='Actual Station Load (kW)',
            line=dict(color='#38bdf8', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=test_timestamps, y=eval_metrics['predictions'],
            mode='lines', name='AI Predicted Load (kW)',
            line=dict(color='#f43f5e', width=2, dash='dash')
        ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
            height=400, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title='Timeline', showgrid=True, gridcolor='#334155'),
            yaxis=dict(title='Electricity Load (kW)', showgrid=True, gridcolor='#334155'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Load model is not trained. Please visit the Model Training page.")

# 2. SOLAR MODEL TAB
with tab_solar:
    st.subheader("Solar Photovoltaic Generation Model (kW)")
    st.caption("Algorithm: XGBoost Regressor trained on Solar Irradiance, Temperature & Solar Angle Physics")
    
    solar_model = st.session_state['solar_model']
    
    if solar_model.is_trained:
        solar_eval = solar_model.evaluate(test_df)
        
        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("MAE", f"{solar_eval['MAE']} kW")
        sm2.metric("RMSE", f"{solar_eval['RMSE']} kW")
        sm3.metric("R² Score", f"{solar_eval['R2']}")
        sm4.metric("MAPE", f"{solar_eval['MAPE']}%")
        sm5.metric("Peak PV Capacity", "60 kW")
        
        st.markdown("#### Actual vs Predicted Solar Generation (Test Set)")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=test_df['timestamp'].values, y=solar_eval['actuals'],
            mode='lines', name='Actual Solar Output (kW)',
            line=dict(color='#f59e0b', width=2)
        ))
        fig_s.add_trace(go.Scatter(
            x=test_df['timestamp'].values, y=solar_eval['predictions'],
            mode='lines', name='AI Predicted Solar (kW)',
            line=dict(color='#10b981', width=2, dash='dash')
        ))
        fig_s.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
            height=400, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title='Timeline', showgrid=True, gridcolor='#334155'),
            yaxis=dict(title='Solar Power (kW)', showgrid=True, gridcolor='#334155'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.warning("Solar model is not trained.")

# 3. WIND MODEL TAB
with tab_wind:
    st.subheader("Wind Turbine Power Generation Model (kW)")
    st.caption("Algorithm: XGBoost Regressor trained on Weibull Wind Speed Distributions & Turbine Power Curves")
    
    wind_model = st.session_state['wind_model']
    
    if wind_model.is_trained:
        wind_eval = wind_model.evaluate(test_df)
        
        wm1, wm2, wm3, wm4, wm5 = st.columns(5)
        wm1.metric("MAE", f"{wind_eval['MAE']} kW")
        wm2.metric("RMSE", f"{wind_eval['RMSE']} kW")
        wm3.metric("R² Score", f"{wind_eval['R2']}")
        wm4.metric("MAPE", f"{wind_eval['MAPE']}%")
        wm5.metric("Peak Turbine Capacity", "80 kW")
        
        st.markdown("#### Actual vs Predicted Wind Generation (Test Set)")
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            x=test_df['timestamp'].values, y=wind_eval['actuals'],
            mode='lines', name='Actual Wind Output (kW)',
            line=dict(color='#06b6d4', width=2)
        ))
        fig_w.add_trace(go.Scatter(
            x=test_df['timestamp'].values, y=wind_eval['predictions'],
            mode='lines', name='AI Predicted Wind (kW)',
            line=dict(color='#a855f7', width=2, dash='dash')
        ))
        fig_w.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
            height=400, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title='Timeline', showgrid=True, gridcolor='#334155'),
            yaxis=dict(title='Wind Power (kW)', showgrid=True, gridcolor='#334155'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.warning("Wind model is not trained.")

st.markdown("---")
st.subheader("🌦️ Meteorological Factors & Polar Trends")
col_w1, col_w2 = st.columns(2)

with col_w1:
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=test_df['timestamp'], y=test_df['temperature'], name='Ambient Temp (°C)', line=dict(color='#60a5fa')))
    fig_temp.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)', height=250, margin=dict(l=10,r=10,t=20,b=10), title="Sub-Zero Temperature Profile (°C)")
    st.plotly_chart(fig_temp, use_container_width=True)

with col_w2:
    fig_wind = go.Figure()
    fig_wind.add_trace(go.Scatter(x=test_df['timestamp'], y=test_df['wind_speed'], name='Wind Speed (m/s)', line=dict(color='#2dd4bf')))
    fig_wind.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)', height=250, margin=dict(l=10,r=10,t=20,b=10), title="Antarctic Wind Speed Profile (m/s)")
    st.plotly_chart(fig_wind, use_container_width=True)
