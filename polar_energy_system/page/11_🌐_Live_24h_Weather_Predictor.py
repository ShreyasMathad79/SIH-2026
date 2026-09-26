import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
"""
Real-Time 24-Hour Live Weather & Future Predictive Center Page (High-Precision Telemetry)
Fetches live 24h satellite weather forecast from Open-Meteo API and executes ML & PuLP optimization
to predict station load, solar PV, wind generation, battery SOC, and diesel fuel consumption hour-by-hour.
"""

import json
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.state_helper import initialize_system_state, load_ml_models_cached
from src.live_weather_api import fetch_live_polar_weather, STATION_COORDINATES
from src.optimization import EnergyOptimizer

initialize_system_state()

st.set_page_config(
    page_title="24-Hour High-Precision Live Weather Predictor | PEIS MoES / NCPOR",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 High-Precision 24-Hour Live Weather & Predictive Center")
st.caption("Live Satellite Weather Telemetry (Pressure, Wind Chill, Humidity, Gusts) & AI 24h Microgrid Power Forecasting | MoES / NCPOR")

# Top Control Bar
c_fac, c_opt, c_ref = st.columns([3, 2, 1])

with c_fac:
    station_selected = st.selectbox(
        "📍 Select Polar Facility",
        list(STATION_COORDINATES.keys()),
        index=0
    )

with c_opt:
    model_choice = st.radio("AI Prediction Engine", ["XGBoost Regressors (Optimized)", "Rule-Based Physics Model"], horizontal=True)

with c_ref:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Sync Satellite Feed", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 1. Fetch Live High-Precision 24h Weather Data from Open-Meteo Satellite
live_weather = fetch_live_polar_weather(station_name=station_selected, days=1)
weather_df = live_weather['weather_df'].iloc[:24].copy()

# Add time features to weather_df if missing
weather_df['hour'] = weather_df['timestamp'].dt.hour
weather_df['month'] = weather_df['timestamp'].dt.month
weather_df['day_of_year'] = weather_df['timestamp'].dt.dayofyear
weather_df['day_of_week'] = weather_df['timestamp'].dt.dayofweek

# High-Precision Live Satellite Status Ticker
ticker_icon = "🟢" if live_weather['is_live'] else "🟡"
st.markdown(f"""
<div style="background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(14px); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 14px 22px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
    <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8;">
        {ticker_icon} {live_weather['status_label']} | Facility: <b>{station_selected.split('(')[0]}</b>
    </div>
    <div style="font-size: 0.85rem; color: #cbd5e1;">
        Temp: <b>{live_weather['temp_c']}°C</b> (Chill: <b>{live_weather['apparent_temp_c']}°C</b>) | Press: <b>{live_weather['pressure_hpa']} hPa</b> | Humidity: <b>{live_weather['humidity_pct']}%</b> | Wind: <b>{live_weather['wind_ms']} m/s</b> (Gust: <b>{live_weather['wind_gust_ms']} m/s</b>)
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Run ML Models & PuLP Optimizer on 24-Hour Weather Feed
load_model, solar_model, wind_model = load_ml_models_cached()

if model_choice == "XGBoost Regressors (Optimized)" and load_model.is_trained:
    last_rows = weather_df.copy()
    last_rows['historical_load'] = 40.0
    last_rows['solar_generation'] = 10.0
    last_rows['wind_generation'] = 15.0
    
    load_pred = load_model.forecast_horizon(last_rows, weather_df, horizon_hours=24)
    solar_pred = solar_model.forecast_horizon(last_rows, weather_df, horizon_hours=24)
    wind_pred = wind_model.forecast_horizon(last_rows, weather_df, horizon_hours=24)
else:
    load_pred = (38.0 + (-weather_df['temperature']) * 0.75).values.tolist()
    solar_pred = (weather_df['solar_irradiance'] * 0.08).clip(0, 60).values.tolist()
    wind_pred = np.clip((weather_df['wind_speed'] - 3.0) * 4.5, 0, 80).values.tolist()

optimizer = EnergyOptimizer()
timestamps = [t.strftime("%H:00") for t in weather_df['timestamp']]
opt_df = optimizer.optimize_dispatch(
    load_forecast_kw=load_pred,
    solar_forecast_kw=solar_pred,
    wind_forecast_kw=wind_pred,
    initial_soc=75.0,
    timestamps=timestamps
)

# Merge weather and dispatch predictions into single master 24h dataframe
master_24h_df = pd.DataFrame({
    'Hour': timestamps,
    'Timestamp': weather_df['timestamp'],
    'Temperature_C': weather_df['temperature'].round(1),
    'Apparent_Temp_C': weather_df.get('apparent_temperature', weather_df['temperature'] - 4.0).round(1),
    'Pressure_hPa': weather_df.get('surface_pressure', 985.0).round(1),
    'Humidity_Pct': weather_df.get('relative_humidity', 60.0).round(1),
    'Wind_Speed_ms': weather_df['wind_speed'].round(1),
    'Wind_Gust_ms': weather_df.get('wind_gusts', weather_df['wind_speed'] * 1.4).round(1),
    'Solar_Irradiance_Wm2': weather_df['solar_irradiance'].round(1),
    'Predicted_Load_kW': opt_df['demand_kw'],
    'Predicted_Solar_kW': opt_df['solar_used_kw'],
    'Predicted_Wind_kW': opt_df['wind_used_kw'],
    'Battery_Discharge_kW': opt_df['battery_discharge_kw'],
    'Battery_Charge_kW': opt_df['battery_charge_kw'],
    'Battery_SOC_Pct': opt_df['battery_soc'],
    'Diesel_Output_kW': opt_df['diesel_output_kw'],
    'Diesel_Fuel_Liters': opt_df['diesel_fuel_liters']
})

# 3. Top 24-Hour Prediction KPI Summary Cards
st.markdown("### 📊 24-Hour High-Precision Telemetry & Prediction Summary")
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

avg_temp = round(master_24h_df['Temperature_C'].mean(), 1)
avg_press = round(master_24h_df['Pressure_hPa'].mean(), 1)
max_gust = round(master_24h_df['Wind_Gust_ms'].max(), 1)
peak_solar = round(master_24h_df['Solar_Irradiance_Wm2'].max(), 1)
total_load_kwh = round(master_24h_df['Predicted_Load_kW'].sum(), 1)
total_fuel_l = round(master_24h_df['Diesel_Fuel_Liters'].sum(), 1)
ren_pct = round(((master_24h_df['Predicted_Solar_kW'].sum() + master_24h_df['Predicted_Wind_kW'].sum()) / max(1, total_load_kwh)) * 100, 1)

k1.metric("24h Avg Temp", f"{avg_temp} °C", delta=f"Chill {round(master_24h_df['Apparent_Temp_C'].mean(), 1)}°C")
k2.metric("24h Barometric Press", f"{avg_press} hPa", delta="Surface Pressure")
k3.metric("24h Max Gust", f"{max_gust} m/s", delta=f"Base Wind {round(master_24h_df['Wind_Speed_ms'].max(), 1)}m/s")
k4.metric("24h Peak Solar", f"{peak_solar} W/m²", delta="Irradiance Peak")
k5.metric("24h Total Load", f"{total_load_kwh} kWh", delta=f"Avg {round(total_load_kwh/24, 1)} kW")
k6.metric("24h Fuel Demand", f"{total_fuel_l} Liters", delta=f"~{round(total_fuel_l*2.5, 1)}$ Cost")
k7.metric("24h Renewable Share", f"{min(100.0, ren_pct)}%", delta="Target > 70%")

st.markdown("---")

# 4. Multi-Variable 24-Hour Plotly Charts
tab_chart1, tab_chart2 = st.tabs([
    "🌡️ 24-Hour High-Precision Weather Forecast (Temp, Chill, Pressure, Gusts)",
    "⚡ 24-Hour Predictive Microgrid Power Balance"
])

with tab_chart1:
    st.subheader("🌡️ High-Precision Satellite Weather Trajectory (Next 24 Hours)")
    
    fig_w = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_w.add_trace(go.Scatter(
        x=master_24h_df['Hour'], y=master_24h_df['Temperature_C'],
        name="Ambient Temp (°C)", line=dict(color="#ef4444", width=3)
    ), secondary_y=False)
    
    fig_w.add_trace(go.Scatter(
        x=master_24h_df['Hour'], y=master_24h_df['Apparent_Temp_C'],
        name="Wind Chill Temp (°C)", line=dict(color="#f87171", width=2, dash="dot")
    ), secondary_y=False)
    
    fig_w.add_trace(go.Scatter(
        x=master_24h_df['Hour'], y=master_24h_df['Wind_Gust_ms'],
        name="Wind Gusts (m/s)", line=dict(color="#06b6d4", width=2.5, dash="dash")
    ), secondary_y=False)
    
    fig_w.add_trace(go.Scatter(
        x=master_24h_df['Hour'], y=master_24h_df['Pressure_hPa'],
        name="Surface Pressure (hPa)", line=dict(color="#38bdf8", width=2),
        fill='none'
    ), secondary_y=True)
    
    fig_w.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
        height=420, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig_w.update_xaxes(title_text="Hour (UTC)")
    fig_w.update_yaxes(title_text="Temperature (°C) / Gusts (m/s)", secondary_y=False)
    fig_w.update_yaxes(title_text="Barometric Pressure (hPa)", secondary_y=True)
    
    st.plotly_chart(fig_w, use_container_width=True)

with tab_chart2:
    st.subheader("⚡ Predictive Power Generation & Dispatch (Next 24 Hours)")
    
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=master_24h_df['Hour'], y=master_24h_df['Predicted_Solar_kW'], name='Solar PV (kW)', fill='tozeroy', line=dict(color='#f59e0b', width=2), stackgroup='supply'))
    fig_p.add_trace(go.Scatter(x=master_24h_df['Hour'], y=master_24h_df['Predicted_Wind_kW'], name='Wind Turbine (kW)', fill='tonexty', line=dict(color='#06b6d4', width=2), stackgroup='supply'))
    fig_p.add_trace(go.Scatter(x=master_24h_df['Hour'], y=master_24h_df['Battery_Discharge_kW'], name='Battery Discharge (kW)', fill='tonexty', line=dict(color='#10b981', width=2), stackgroup='supply'))
    fig_p.add_trace(go.Scatter(x=master_24h_df['Hour'], y=master_24h_df['Diesel_Output_kW'], name='Diesel Generator (kW)', fill='tonexty', line=dict(color='#ef4444', width=2), stackgroup='supply'))
    fig_p.add_trace(go.Scatter(x=master_24h_df['Hour'], y=master_24h_df['Battery_Charge_kW'], name='Battery Charge (kW)', line=dict(color='#a855f7', width=2, dash='dash')))
    
    fig_p.add_trace(go.Scatter(x=master_24h_df['Hour'], y=master_24h_df['Predicted_Load_kW'], name='Predicted Station Demand (kW)', line=dict(color='#ffffff', width=3.5)))
    
    fig_p.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
        height=420, margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title='Hour (UTC)'), yaxis=dict(title='Power (kW)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_p, use_container_width=True)

st.markdown("---")

# 5. Full 24-Hour High-Precision Predictive Forecast Data Table & CSV Export
st.subheader("📋 24-Hour High-Precision Forecast Data Table")
st.caption("Complete Hour-by-Hour Weather & Microgrid Power Dispatch Schedule")

st.dataframe(
    master_24h_df,
    column_config={
        "Hour": "Hour (UTC)",
        "Temperature_C": st.column_config.NumberColumn("Temp (°C)", format="%.1f °C"),
        "Apparent_Temp_C": st.column_config.NumberColumn("Chill (°C)", format="%.1f °C"),
        "Pressure_hPa": st.column_config.NumberColumn("Press (hPa)", format="%.1f hPa"),
        "Humidity_Pct": st.column_config.NumberColumn("Humidity (%)", format="%.0f %%"),
        "Wind_Speed_ms": st.column_config.NumberColumn("Wind (m/s)", format="%.1f m/s"),
        "Wind_Gust_ms": st.column_config.NumberColumn("Gust (m/s)", format="%.1f m/s"),
        "Solar_Irradiance_Wm2": st.column_config.NumberColumn("Solar (W/m²)", format="%.1f W/m²"),
        "Predicted_Load_kW": st.column_config.NumberColumn("Load (kW)", format="%.1f kW"),
        "Predicted_Solar_kW": st.column_config.NumberColumn("Solar PV (kW)", format="%.1f kW"),
        "Predicted_Wind_kW": st.column_config.NumberColumn("Wind (kW)", format="%.1f kW"),
        "Battery_Discharge_kW": st.column_config.NumberColumn("Batt Dis (kW)", format="%.1f kW"),
        "Battery_Charge_kW": st.column_config.NumberColumn("Batt Chg (kW)", format="%.1f kW"),
        "Battery_SOC_Pct": st.column_config.NumberColumn("Batt SOC (%)", format="%.1f %%"),
        "Diesel_Output_kW": st.column_config.NumberColumn("Diesel (kW)", format="%.1f kW"),
        "Diesel_Fuel_Liters": st.column_config.NumberColumn("Fuel (L)", format="%.1f L")
    },
    hide_index=True,
    use_container_width=True
)

col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    csv_data = master_24h_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download High-Precision 24h Predictions CSV",
        data=csv_data,
        file_name=f"polar_high_precision_forecast_{station_selected.split()[0]}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_exp2:
    json_data = master_24h_df.to_json(orient="records", date_format="iso")
    st.download_button(
        label="📥 Download High-Precision 24h Predictions JSON",
        data=json_data,
        file_name=f"polar_high_precision_forecast_{station_selected.split()[0]}.json",
        mime="application/json",
        use_container_width=True
    )
