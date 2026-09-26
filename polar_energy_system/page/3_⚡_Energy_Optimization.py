import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
"""
Energy Optimization Page
Displays PuLP MILP solver dispatch schedule, hourly timeline table, stacked area breakdown, and fuel savings summary.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.state_helper import initialize_system_state, run_system_forecast_and_optimization

initialize_system_state()

st.title("⚡ PuLP Energy Optimization Engine")
st.caption("Mixed-Integer Linear Programming (MILP) Microgrid Dispatch Schedule | MoES / NCPOR")

# Sidebar Configuration for Battery & Diesel Specs
st.sidebar.markdown("### ⚙️ Physical Asset Parameters")

b_cap = st.sidebar.number_input("Battery Capacity (kWh)", 50.0, 1000.0, 200.0, step=10.0)
b_min_soc = st.sidebar.slider("Min Reserve SOC (%)", 10.0, 40.0, 20.0)
b_max_soc = st.sidebar.slider("Max Charging SOC (%)", 80.0, 100.0, 95.0)

d_max = st.sidebar.number_input("Diesel Max Output (kW)", 20.0, 300.0, 100.0)
d_min = st.sidebar.number_input("Diesel Min Output (kW)", 5.0, 50.0, 15.0)
fuel_cost = st.sidebar.number_input("Diesel Fuel Cost ($/L)", 1.0, 10.0, 2.50)

# Update optimizer parameters
optimizer = st.session_state['optimizer']
optimizer.b_cap = b_cap
optimizer.min_soc = b_min_soc
optimizer.max_soc = b_max_soc
optimizer.diesel_max = d_max
optimizer.diesel_min = d_min
optimizer.fuel_cost = fuel_cost

scenario = st.session_state.get('active_scenario', 'Normal Day')
res = run_system_forecast_and_optimization(scenario_name=scenario, horizon_hours=24)
opt_df = res['optimization_df']
comp = res['comparison_metrics']

# Summary KPI Header
c1, c2, c3, c4, c5, c6 = st.columns(6)

solar_used = opt_df['solar_used_kw'].sum()
wind_used = opt_df['wind_used_kw'].sum()
ren_used_total = round(solar_used + wind_used, 1)

solar_avail = opt_df['solar_available_kw'].sum()
wind_avail = opt_df['wind_available_kw'].sum()
ren_curtailed = round(max(0.0, (solar_avail + wind_avail) - ren_used_total), 1)

diesel_consumed = round(opt_df['diesel_fuel_liters'].sum(), 1)
diesel_kw_total = round(opt_df['diesel_output_kw'].sum(), 1)
battery_dis_total = round(opt_df['battery_discharge_kw'].sum(), 1)
total_shed = round(opt_df['total_shed_kw'].sum(), 1)

c1.metric("Renewable Utilized", f"{ren_used_total} kWh", delta=f"Solar {round(solar_used,1)} | Wind {round(wind_used,1)}")
c2.metric("Renewable Curtailed", f"{ren_curtailed} kWh", delta="Minimized by PuLP")
c3.metric("Battery Energy Discharged", f"{battery_dis_total} kWh", delta=f"Degradation Cost ${round(battery_dis_total*0.08, 2)}")
c4.metric("Diesel Consumed", f"{diesel_consumed} L", delta=f"${round(diesel_consumed*fuel_cost, 2)}")
c5.metric("Fuel Saved vs Baseline", f"{comp['savings']['fuel_saved_liters']} L", delta=f"{comp['savings']['fuel_saved_pct']}% Reduction")
c6.metric("Critical Loads Status", "100% Protected" if opt_df['shed_critical_kw'].sum() < 0.1 else f"⚠ {round(opt_df['shed_critical_kw'].sum(),1)} kW Shed", delta="Tier 1 Priority")

st.markdown("---")

# Stacked Power Balance Chart
st.subheader("📊 Optimal Power Generation & Load Balancing Stack")
fig = go.Figure()

fig.add_trace(go.Bar(x=opt_df['timestamp'], y=opt_df['solar_used_kw'], name='Solar PV (kW)', marker_color='#f59e0b'))
fig.add_trace(go.Bar(x=opt_df['timestamp'], y=opt_df['wind_used_kw'], name='Wind Power (kW)', marker_color='#06b6d4'))
fig.add_trace(go.Bar(x=opt_df['timestamp'], y=opt_df['battery_discharge_kw'], name='Battery Discharge (kW)', marker_color='#10b981'))
fig.add_trace(go.Bar(x=opt_df['timestamp'], y=opt_df['diesel_output_kw'], name='Diesel Generator (kW)', marker_color='#ef4444'))
fig.add_trace(go.Bar(x=opt_df['timestamp'], y=opt_df['total_shed_kw'], name='Load Management / Shedding (kW)', marker_color='#64748b'))

fig.add_trace(go.Scatter(
    x=opt_df['timestamp'], y=opt_df['demand_kw'],
    mode='lines+markers', name='Target Load Demand (kW)',
    line=dict(color='#ffffff', width=3)
))

fig.update_layout(
    barmode='stack',
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.6)',
    height=420,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(title='Hour', showgrid=True, gridcolor='#334155'),
    yaxis=dict(title='Power (kW)', showgrid=True, gridcolor='#334155'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Hourly Dispatch Timeline Table
st.subheader("📋 Hourly Dispatch Schedule & Actions")
st.caption("Calculated by PuLP MILP Cost Optimizer (Minimizing Fuel + Degradation + Curtailment + Unmet Penalties)")

display_df = opt_df[[
    'timestamp', 'demand_kw', 'solar_available_kw', 'wind_available_kw',
    'solar_used_kw', 'wind_used_kw', 'battery_charge_kw', 'battery_discharge_kw',
    'battery_soc', 'diesel_output_kw', 'diesel_fuel_liters', 'total_shed_kw', 'action'
]].copy()

display_df.columns = [
    'Time', 'Demand (kW)', 'Solar Avail', 'Wind Avail',
    'Solar Used', 'Wind Used', 'Batt Charge', 'Batt Dischg',
    'Batt SOC %', 'Diesel (kW)', 'Fuel (L)', 'Shed (kW)', 'Optimized Action'
]

st.dataframe(
    display_df,
    use_container_width=True,
    height=350,
    hide_index=True
)
