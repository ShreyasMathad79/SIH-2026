import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
for d in [BASE_DIR, SRC_DIR, os.path.dirname(BASE_DIR)]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

"""
Main Dashboard Page
Top KPI Cards, Live 24h Trajectory Plotly Chart, Energy Survival Status, and Alert Feed.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

try:
    from state_helper import initialize_system_state, run_system_forecast_and_optimization
    from live_weather_api import fetch_live_polar_weather
except ModuleNotFoundError:
    from state_helper import initialize_system_state, run_system_forecast_and_optimization
    from live_weather_api import fetch_live_polar_weather
initialize_system_state()

try:
    st.set_page_config(
        page_title="Main Dashboard | Polar Energy Intelligence System",
        page_icon="📊",
        layout="wide"
    )
except Exception:
    pass

# Fetch Live Weather API Feed
live_weather = fetch_live_polar_weather("Bharati Station (Antarctica - Larsemann Hills, 69.4°S)")

st.title("📊 Polar Energy Main Dashboard")
st.caption("Real-Time Microgrid Monitoring & 24-Hour Predictive Dispatch | MoES / NCPOR")

# Live Satellite Telemetry Ticker
live_icon = "🟢" if live_weather['is_live'] else "🟡"
st.markdown(f"""
<div style="background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 10px 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
    <div style="font-size: 0.9rem; font-weight: 600; color: #38bdf8;">
        {live_icon} {live_weather['status_label']} | Live Ambient Temp: <b>{live_weather['temp_c']}°C</b> | Wind Speed: <b>{live_weather['wind_ms']} m/s</b> | Solar Irradiance: <b>{live_weather['solar_wm2']} W/m²</b>
    </div>
    <div style="font-size: 0.8rem; color: #94a3b8;">
        Sync: {live_weather['timestamp']}
    </div>
</div>
""", unsafe_allow_html=True)

# Active Scenario Selection Bar
col_sc1, col_sc2 = st.columns([3, 1])
with col_sc1:
    scenario = st.selectbox(
        "Current Operating Scenario",
        [
            "Normal Day", "Polar Night (Zero Solar)", "High Wind Surge",
            "Severe Polar Storm", "Extreme Cold Snap", "High Station Activity",
            "Low Battery Reserve", "Low Diesel Fuel"
        ],
        index=["Normal Day", "Polar Night (Zero Solar)", "High Wind Surge", "Severe Polar Storm", "Extreme Cold Snap", "High Station Activity", "Low Battery Reserve", "Low Diesel Fuel"].index(st.session_state.get('active_scenario', 'Normal Day'))
    )
    st.session_state['active_scenario'] = scenario

with col_sc2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh System State", use_container_width=True):
        st.rerun()

# Run forecast & optimizer for active scenario
res = run_system_forecast_and_optimization(scenario_name=scenario, horizon_hours=24)
opt_df = res['optimization_df']
survival = res['survival_metrics']
weather_df = res['weather_df']

# Current hour slice (hour 0 of forecast)
cur_row = opt_df.iloc[0]
cur_demand = cur_row['demand_kw']
cur_solar = cur_row['solar_used_kw']
cur_wind = cur_row['wind_used_kw']
cur_ren = cur_solar + cur_wind
cur_soc = cur_row['battery_soc']
cur_diesel = cur_row['diesel_output_kw']
diesel_status = "OFF" if cur_diesel < 0.5 else f"ON ({cur_diesel} kW)"
ren_contrib = round((cur_ren / max(1e-5, cur_demand)) * 100, 1)

battery = st.session_state['battery']
diesel = st.session_state['diesel']

# 1. Top KPI Cards
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)

with kpi1:
    st.metric("Current Load", f"{cur_demand} kW", delta=f"{weather_df.iloc[0]['temperature']}°C Temp")

with kpi2:
    st.metric("Renewable Gen", f"{round(cur_ren, 1)} kW", delta=f"Solar {cur_solar} | Wind {cur_wind}")

with kpi3:
    st.metric("Battery SOC", f"{cur_soc}%", delta=f"Cap {battery.capacity_kwh} kWh")

with kpi4:
    st.metric("Diesel Gen", diesel_status, delta="Min 15 kW" if cur_diesel > 0 else "Standby")

with kpi5:
    st.metric("Fuel Remaining", f"{round(diesel.current_fuel_liters, 1)} L", delta=f"~{survival['estimated_fuel_endurance_hours']}h Run")

with kpi6:
    st.metric("Renewable Share", f"{min(100.0, ren_contrib)}%", delta="Target > 70%")

with kpi7:
    pred_24h_sum = round(sum(res['load_forecast']), 1)
    st.metric("24h Predicted Load", f"{pred_24h_sum} kWh", delta=f"Avg {round(pred_24h_sum/24, 1)} kW")

st.markdown("---")

# 2. 24-Hour Energy Balance Trajectory Chart
st.subheader("📈 24-Hour Power Balance Trajectory")
st.caption("Forecasted Station Demand vs Optimized Supply Dispatch (Solar, Wind, Battery, Diesel)")

fig = go.Figure()

# Stacked supply components
fig.add_trace(go.Scatter(
    x=opt_df['timestamp'], y=opt_df['solar_used_kw'],
    mode='lines', name='Solar PV (kW)', fill='tozeroy',
    line=dict(width=2, color='#f59e0b'),
    stackgroup='supply'
))

fig.add_trace(go.Scatter(
    x=opt_df['timestamp'], y=opt_df['wind_used_kw'],
    mode='lines', name='Wind Turbine (kW)', fill='tonexty',
    line=dict(width=2, color='#06b6d4'),
    stackgroup='supply'
))

fig.add_trace(go.Scatter(
    x=opt_df['timestamp'], y=opt_df['battery_discharge_kw'],
    mode='lines', name='Battery Discharge (kW)', fill='tonexty',
    line=dict(width=2, color='#10b981'),
    stackgroup='supply'
))

fig.add_trace(go.Scatter(
    x=opt_df['timestamp'], y=opt_df['diesel_output_kw'],
    mode='lines', name='Diesel Generator (kW)', fill='tonexty',
    line=dict(width=2, color='#ef4444'),
    stackgroup='supply'
))

# Total Demand Line (Overlaid)
fig.add_trace(go.Scatter(
    x=opt_df['timestamp'], y=opt_df['demand_kw'],
    mode='lines+markers', name='Station Load Demand (kW)',
    line=dict(color='#ffffff', width=3.5, dash='solid')
))

# Battery Charge Power (Negative / Downward)
fig.add_trace(go.Scatter(
    x=opt_df['timestamp'], y=-opt_df['battery_charge_kw'],
    mode='lines', name='Battery Charge Power (kW)',
    line=dict(color='#8b5cf6', width=2, dash='dot')
))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.6)',
    height=440,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(title='Timestamp (Hour)', showgrid=True, gridcolor='#334155'),
    yaxis=dict(title='Power (kW)', showgrid=True, gridcolor='#334155'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 3. Energy Survival Forecast & Active Alerts Section
col_surv, col_alerts = st.columns([1, 1])

with col_surv:
    st.subheader("🛡️ 24-Hour Energy Survival Forecast")
    
    surv_color = "#10b981" if survival['critical_loads_protected_pct'] == 100 else "#ef4444"
    st.markdown(f"""
    <div class="polar-card">
        <h4 style="margin: 0; color: #94a3b8;">Critical Loads Protection Status</h4>
        <h1 style="color: {surv_color}; font-size: 2.6rem; margin: 10px 0; font-weight: 800;">{survival['critical_loads_protected_pct']}% Protected</h1>
        <p style="color: #cbd5e1; margin: 0;"><b>Renewable Share:</b> {survival['renewable_share_pct']}%</p>
        <p style="color: #cbd5e1; margin: 4px 0;"><b>Est. Diesel Endurance:</b> {survival['estimated_fuel_endurance_hours']} Operating Hours</p>
        <p style="color: #cbd5e1; margin: 0;"><b>Min Projected Battery SOC:</b> {survival['min_battery_soc_pct']}%</p>
    </div>
    """, unsafe_allow_html=True)

with col_alerts:
    st.subheader("🚨 System AI Advisory & Alerts")
    for alt in survival['alerts']:
        card_class = f"alert-card-{alt['type'].lower()}"
        st.markdown(f"""
        <div class="{card_class}">
            <b>{alt['icon']} {alt['title']}</b><br>
            <span style="font-size: 0.9rem;">{alt['message']}</span>
        </div>
        """, unsafe_allow_html=True)

st.caption(f"📌 **Data Source**: {st.session_state.get('data_source_label')} | Live Satellite Stream: Active")
