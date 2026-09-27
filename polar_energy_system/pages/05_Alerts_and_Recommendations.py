import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
for d in [BASE_DIR, SRC_DIR, os.path.dirname(BASE_DIR)]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

"""
Alerts & Recommendations Page
Calculates real-time AI advisory recommendations and polar risk alerts based on system physics and ML forecasts.
"""

import streamlit as st
import pandas as pd

try:
    from state_helper import initialize_system_state, run_system_forecast_and_optimization
except ModuleNotFoundError:
    from state_helper import initialize_system_state, run_system_forecast_and_optimization
initialize_system_state()

st.title("🚨 AI Recommendations & Operational Advisories")
st.caption("Intelligent Decision-Support Feed for Polar Station Operators | MoES / NCPOR")

scenario = st.session_state.get('active_scenario', 'Normal Day')
res = run_system_forecast_and_optimization(scenario_name=scenario, horizon_hours=24)
opt_df = res['optimization_df']
survival = res['survival_metrics']
diesel = st.session_state['diesel']

st.markdown(f"### ⚡ Current System Context: **{scenario}**")

# 1. Survival Alert Cards
st.subheader("🛡️ 24-Hour Polar Risk Assessment")

for alt in survival['alerts']:
    card_class = f"alert-card-{alt['type'].lower()}"
    st.markdown(f"""
    <div class="{card_class}">
        <h4 style="margin: 0;">{alt['icon']} {alt['title']}</h4>
        <p style="margin: 6px 0 0 0; font-size: 1rem;">{alt['message']}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 2. Dynamic AI Advisory Generator
st.subheader("🤖 Calculated Operator Action Recommendations")
st.caption("Generated from actual system forecasting equations & PuLP optimization schedules (Not hardcoded)")

recommendations = []

# A. Solar drop check
solar_forecast = opt_df['solar_available_kw'].values
if len(solar_forecast) >= 6:
    next_6h_solar = solar_forecast[:6]
    max_s = max(next_6h_solar)
    min_s = min(next_6h_solar)
    if max_s > 10.0 and min_s / (max_s + 1e-5) < 0.4:
        drop_pct = round((1.0 - min_s / max_s) * 100, 1)
        drop_h = list(next_6h_solar).index(min_s) + 1
        recommendations.append({
            'category': 'SOLAR ADVISORY',
            'icon': '☀️',
            'action': f"Solar generation is expected to fall sharply by {drop_pct}% in the next {drop_h} hours.",
            'detail': "Increase battery reserve charging during current window to buffer the upcoming solar deficit."
        })

# B. Non-critical load deferral check
shed_nc = opt_df['shed_noncritical_kw'].values
if sum(shed_nc) > 0.5:
    shed_hours = opt_df[opt_df['shed_noncritical_kw'] > 0.1]['timestamp'].values
    start_t = shed_hours[0]
    end_t = shed_hours[-1]
    recommendations.append({
        'category': 'LOAD SHEDDING ADVISORY',
        'icon': '⚡',
        'action': f"Defer non-critical equipment (EV charging, non-essential tools) between {start_t}–{end_t}.",
        'detail': "Proactive load shedding will prevent diesel generator activation and preserve battery reserves for heating."
    })
else:
    recommendations.append({
        'category': 'LOAD SHEDDING ADVISORY',
        'icon': '✓',
        'action': "Station loads are fully optimal. No non-critical load deferral required.",
        'detail': "Available renewables and battery capacity are sufficient for all station operations."
    })

# C. Diesel generator necessity check
diesel_kw = opt_df['diesel_output_kw'].values
total_diesel_h = sum(1 for d in diesel_kw if d > 0.5)

if total_diesel_h == 0:
    recommendations.append({
        'category': 'GENERATOR STATUS',
        'icon': '🟢',
        'action': "Diesel generator is currently UNNECESSARY for the next 24 hours.",
        'detail': "Keep generator in auto-standby mode. Zero fuel consumption projected."
    })
else:
    first_d_time = opt_df[opt_df['diesel_output_kw'] > 0.5]['timestamp'].iloc[0]
    avg_d = round(opt_df[opt_df['diesel_output_kw'] > 0.5]['diesel_output_kw'].mean(), 1)
    recommendations.append({
        'category': 'GENERATOR STATUS',
        'icon': '🟡',
        'action': f"Diesel generator start required at {first_d_time} (Expected loading: {avg_d} kW).",
        'detail': "Verify generator pre-heating and lube oil pressure prior to automated start."
    })

# D. Battery reserve check
soc_vals = opt_df['battery_soc'].values
min_soc_val = min(soc_vals)
if min_soc_val < 30.0:
    recommendations.append({
        'category': 'BATTERY ADVISORY',
        'icon': '🔋',
        'action': f"Battery SOC will reach low threshold ({round(min_soc_val,1)}%). Pre-charge recommended.",
        'detail': "Allow wind/diesel surplus to top up battery to 80% before entering extreme cold window."
    })

# E. Fuel endurance check
fuel_used_24h = sum(opt_df['diesel_fuel_liters'].values)
rem_fuel = diesel.current_fuel_liters
if fuel_used_24h > 0:
    days_endurance = round(rem_fuel / (fuel_used_24h), 1)
    recommendations.append({
        'category': 'FUEL ENDURANCE',
        'icon': '⛽',
        'action': f"Current diesel fuel stock ({round(rem_fuel,1)} L) will last approximately {days_endurance} days.",
        'detail': f"Projected 24-hour fuel consumption: {round(fuel_used_24h,1)} Liters."
    })
else:
    recommendations.append({
        'category': 'FUEL ENDURANCE',
        'icon': '⛽',
        'action': f"Fuel reserves ({round(rem_fuel,1)} L) intact. Zero fuel burn projected over 24 hours.",
        'detail': "Station running on 100% clean renewable energy and battery storage."
    })

# Render Recommendations Cards
for rec in recommendations:
    st.markdown(f"""
    <div class="polar-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="color: #38bdf8; margin: 0;">{rec['icon']} {rec['category']}</h4>
            <span style="background: #1e293b; border: 1px solid #475569; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; color: #cbd5e1;">AI Automated Advisory</span>
        </div>
        <p style="color: white; font-size: 1.1rem; font-weight: 600; margin: 8px 0 4px 0;">{rec['action']}</p>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">{rec['detail']}</p>
    </div>
    """, unsafe_allow_html=True)
