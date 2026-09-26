import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
"""
Simulation Mode Page
Interactive Hackathon Scenario Demonstrator for testing polar station resilience under extreme conditions.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.state_helper import initialize_system_state, run_system_forecast_and_optimization

initialize_system_state()

st.title("🧪 Polar Stress-Test Simulation Environment")
st.caption("Hackathon Interactive Scenario Demonstrator | MoES / NCPOR")

st.markdown("""
Select an extreme polar weather or hardware condition below to simulate station behavior.
The AI engine will immediately re-run load forecasts, renewable generation models, and PuLP optimization dispatches.
""")

# Scenario Selection Grid
col_s1, col_s2 = st.columns([2, 1])

with col_s1:
    scenario = st.radio(
        "Choose Extreme Polar Scenario to Trigger",
        [
            "Normal Day",
            "Polar Night (Zero Solar)",
            "High Wind Surge",
            "Severe Polar Storm",
            "Extreme Cold Snap",
            "High Station Activity",
            "Low Battery Reserve",
            "Low Diesel Fuel"
        ],
        index=["Normal Day", "Polar Night (Zero Solar)", "High Wind Surge", "Severe Polar Storm", "Extreme Cold Snap", "High Station Activity", "Low Battery Reserve", "Low Diesel Fuel"].index(st.session_state.get('active_scenario', 'Normal Day')),
        horizontal=True
    )
    st.session_state['active_scenario'] = scenario

with col_s2:
    st.markdown("""
    <div class="polar-card">
        <h4 style="color: #38bdf8; margin:0;">⚡ Active Scenario</h4>
        <h3 style="color: white; margin: 4px 0;">{scenario}</h3>
        <p style="color:#94a3b8; font-size:0.85rem; margin:0;">Forecast Horizon: 24 Hours</p>
    </div>
    """.format(scenario=scenario), unsafe_allow_html=True)

# Run Simulation
res = run_system_forecast_and_optimization(scenario_name=scenario, horizon_hours=24)
opt_df = res['optimization_df']
survival = res['survival_metrics']
weather_df = res['weather_df']

st.markdown("---")

# Scenario Weather Parameters Summary
st.subheader("🌡️ Simulated Environmental & Station Profile")
w_col1, w_col2, w_col3, w_col4 = st.columns(4)

w_col1.metric("Min Temperature", f"{weather_df['temperature'].min()} °C", delta="Sub-Zero Cold")
w_col2.metric("Peak Wind Speed", f"{weather_df['wind_speed'].max()} m/s", delta="Turbine Cut-out > 25 m/s" if weather_df['wind_speed'].max() > 25 else "Normal Wind")
w_col3.metric("Max Solar Irradiance", f"{weather_df['solar_irradiance'].max()} W/m²", delta="Polar Darkness" if weather_df['solar_irradiance'].max() == 0 else "Sunlight Available")
w_col4.metric("Personnel Count", f"{weather_df['personnel_count'].iloc[0]} Occupants", delta="Base Load Modified")

st.markdown("---")

# Results & Trajectory Comparison
st.subheader("📈 Optimization Response & Power Dispatch")

fig = go.Figure()

fig.add_trace(go.Scatter(x=opt_df['timestamp'], y=opt_df['demand_kw'], name='Station Load Demand (kW)', line=dict(color='#ffffff', width=3)))
fig.add_trace(go.Scatter(x=opt_df['timestamp'], y=opt_df['solar_used_kw'], name='Solar Power Used (kW)', line=dict(color='#f59e0b', width=2)))
fig.add_trace(go.Scatter(x=opt_df['timestamp'], y=opt_df['wind_used_kw'], name='Wind Power Used (kW)', line=dict(color='#06b6d4', width=2)))
fig.add_trace(go.Scatter(x=opt_df['timestamp'], y=opt_df['battery_discharge_kw'], name='Battery Discharge (kW)', line=dict(color='#10b981', width=2)))
fig.add_trace(go.Scatter(x=opt_df['timestamp'], y=opt_df['diesel_output_kw'], name='Diesel Output (kW)', line=dict(color='#ef4444', width=2)))
fig.add_trace(go.Scatter(x=opt_df['timestamp'], y=opt_df['total_shed_kw'], name='Load Shedding (kW)', line=dict(color='#94a3b8', width=2, dash='dot')))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.6)',
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(title='Hour', showgrid=True, gridcolor='#334155'),
    yaxis=dict(title='Power (kW)', showgrid=True, gridcolor='#334155')
)

st.plotly_chart(fig, use_container_width=True)

# Scenario Special Highlights (Demonstration Callouts)
if scenario == "Severe Polar Storm":
    st.error("""
    🚨 **HACKATHON DEMO: SEVERE POLAR STORM TRIGGERED**
    - Temperature dropped to **-38°C** → Heating load surged.
    - Solar irradiance dropped to **0 W/m²** (Zero solar availability).
    - High wind (>26 m/s) exceeded turbine cut-out safety threshold.
    
    **AI Optimizer Response**:
    1. Preserved battery emergency reserve (20% SOC).
    2. Started Diesel Generator on minimal loading.
    3. Protected Tier 1 Critical heating & life support loads 100%.
    4. Proactively deferred Tier 3 non-critical loads to avert total station blackout.
    """)
elif scenario == "Polar Night (Zero Solar)":
    st.info("""
    ❄ **POLAR NIGHT SCENARIO ACTIVE**: Solar generation is 0 W/m² continuously.
    Wind turbines & Battery storage handle the primary load, supported by minimal diesel generation.
    """)
