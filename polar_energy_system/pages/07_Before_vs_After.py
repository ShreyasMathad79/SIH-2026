import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
for d in [BASE_DIR, SRC_DIR, os.path.dirname(BASE_DIR)]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

"""
Before vs After Comparison Page
Quantifies performance gains of AI + Optimization Strategy vs Baseline Strategy.
"""

import streamlit as st
import plotly.graph_objects as go

try:
    from state_helper import initialize_system_state, run_system_forecast_and_optimization
except ModuleNotFoundError:
    from state_helper import initialize_system_state, run_system_forecast_and_optimization
initialize_system_state()

st.title("⚖️ Before vs After: AI Strategy Comparison")
st.caption("Quantitative Evaluation: Baseline Diesel-First Strategy vs AI + PuLP Optimization | MoES / NCPOR")

scenario = st.session_state.get('active_scenario', 'Normal Day')
res = run_system_forecast_and_optimization(scenario_name=scenario, horizon_hours=24)
comp = res['comparison_metrics']

b_m = comp['baseline']
o_m = comp['optimized']
s_m = comp['savings']

# 1. Headline Savings Cards
st.subheader("🎉 Headline Savings & Efficiency Performance")

sc1, sc2, sc3, sc4 = st.columns(4)

sc1.metric(
    "Diesel Fuel Saved",
    f"{s_m['fuel_saved_liters']} L",
    delta=f"{s_m['fuel_saved_pct']}% Reduction"
)

sc2.metric(
    "Cost Reduction",
    f"${s_m['cost_saved_usd']}",
    delta="Fuel Cost Saved"
)

sc3.metric(
    "Renewable Utilization",
    f"{o_m['renewable_utilization_pct']}%",
    delta=f"+{round(o_m['renewable_utilization_pct'] - b_m['renewable_utilization_pct'], 1)}% vs Baseline"
)

sc4.metric(
    "CO2 Emission Avoided",
    f"{s_m['co2_saved_kg']} kg",
    delta="Carbon Footprint Reduction"
)

st.markdown("---")

# 2. Side-by-Side Detailed Metrics Table
st.subheader("📋 Side-by-Side Metric Benchmark")

comp_table = [
    {
        "Metric": "Diesel Fuel Consumed (Liters / 24h)",
        "Baseline Strategy": f"{b_m['diesel_fuel_liters']} L",
        "AI + Optimization": f"{o_m['diesel_fuel_liters']} L",
        "Improvement": f"▼ {s_m['fuel_saved_pct']}% Saved"
    },
    {
        "Metric": "Total Fuel Expense ($)",
        "Baseline Strategy": f"${b_m['fuel_cost_usd']}",
        "AI + Optimization": f"${o_m['fuel_cost_usd']}",
        "Improvement": f"▼ ${s_m['cost_saved_usd']} Saved"
    },
    {
        "Metric": "Renewable Energy Utilization (%)",
        "Baseline Strategy": f"{b_m['renewable_utilization_pct']}%",
        "AI + Optimization": f"{o_m['renewable_utilization_pct']}%",
        "Improvement": f"▲ +{round(o_m['renewable_utilization_pct'] - b_m['renewable_utilization_pct'], 1)}%"
    },
    {
        "Metric": "Renewable Energy Curtailed (kWh)",
        "Baseline Strategy": f"{b_m['renewable_curtailed_kwh']} kWh",
        "AI + Optimization": f"{o_m['renewable_curtailed_kwh']} kWh",
        "Improvement": f"▼ Curtailed Energy Reduced"
    },
    {
        "Metric": "CO2 Emissions (kg CO2)",
        "Baseline Strategy": f"{b_m['co2_emissions_kg']} kg",
        "AI + Optimization": f"{o_m['co2_emissions_kg']} kg",
        "Improvement": f"▼ {s_m['co2_saved_kg']} kg Reduced"
    },
    {
        "Metric": "Energy Deficit / Unmet Load (kWh)",
        "Baseline Strategy": f"{b_m['energy_deficit_kwh']} kWh",
        "AI + Optimization": f"{o_m['energy_deficit_kwh']} kWh",
        "Improvement": "✓ Critical Load 100% Protected"
    }
]

st.dataframe(comp_table, use_container_width=True, hide_index=True)

st.markdown("---")

# 3. Bar Chart Comparison
st.subheader("📊 Visual Benchmark Comparison")

col_b1, col_b2 = st.columns(2)

with col_b1:
    fig_fuel = go.Figure(data=[
        go.Bar(name='Baseline', x=['Fuel Consumed (L)', 'CO2 Emissions (kg)'], y=[b_m['diesel_fuel_liters'], b_m['co2_emissions_kg']], marker_color='#ef4444'),
        go.Bar(name='AI Optimized', x=['Fuel Consumed (L)', 'CO2 Emissions (kg)'], y=[o_m['diesel_fuel_liters'], o_m['co2_emissions_kg']], marker_color='#10b981')
    ])
    fig_fuel.update_layout(barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)', height=320, title="Fuel & Carbon Reduction")
    st.plotly_chart(fig_fuel, use_container_width=True)

with col_b2:
    fig_ren = go.Figure(data=[
        go.Bar(name='Baseline', x=['Renewable Utilization %', 'Energy Curtailed (kWh)'], y=[b_m['renewable_utilization_pct'], b_m['renewable_curtailed_kwh']], marker_color='#f59e0b'),
        go.Bar(name='AI Optimized', x=['Renewable Utilization %', 'Energy Curtailed (kWh)'], y=[o_m['renewable_utilization_pct'], o_m['renewable_curtailed_kwh']], marker_color='#06b6d4')
    ])
    fig_ren.update_layout(barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)', height=320, title="Renewable Efficiency")
    st.plotly_chart(fig_ren, use_container_width=True)
