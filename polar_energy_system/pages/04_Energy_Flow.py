import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(BASE_DIR) in ['pages', 'src']:
    BASE_DIR = os.path.dirname(BASE_DIR)

SRC_DIR = os.path.join(BASE_DIR, 'src')
for d in [BASE_DIR, SRC_DIR]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

"""
Energy Flow Visualization Page
Visual microgrid topology & interactive energy distribution diagrams.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

try:
    from state_helper import initialize_system_state, run_system_forecast_and_optimization
except ModuleNotFoundError:
    from state_helper import initialize_system_state, run_system_forecast_and_optimization
initialize_system_state()

st.title("🔀 Microgrid Energy Flow Topology")
st.caption("Dynamic Power Routing: Solar & Wind → Energy Bus → Battery / Diesel → Tiered Station Loads")

scenario = st.session_state.get('active_scenario', 'Normal Day')
res = run_system_forecast_and_optimization(scenario_name=scenario, horizon_hours=24)
opt_df = res['optimization_df']

# Select hour index to inspect
hour_idx = st.slider("Select Timeline Hour to Inspect Flow Topology", 0, 23, 12, format="Hour %d:00")
row = opt_df.iloc[hour_idx]

s_used = row['solar_used_kw']
w_used = row['wind_used_kw']
b_dis = row['battery_discharge_kw']
b_chg = row['battery_charge_kw']
d_gen = row['diesel_output_kw']
l_total = row['demand_kw']
soc = row['battery_soc']

# Tiered load breakdown
l_crit = 0.50 * l_total - row['shed_critical_kw']
l_imp = 0.30 * l_total - row['shed_important_kw']
l_noncrit = 0.20 * l_total - row['shed_noncritical_kw']
l_shed = row['total_shed_kw']

st.markdown(f"### 📍 Power Flow Snapshot at {row['timestamp']}")

# 1. Visual Topology Cards
col_gen, col_bus, col_store, col_load = st.columns(4)

with col_gen:
    st.markdown(f"""
    <div class="polar-card">
        <h4 style="color: #f59e0b; margin-top: 0;">☀️ Generation Sources</h4>
        <p><b>Solar PV:</b> <span style="color:#f59e0b; font-size:1.2rem;">{s_used} kW</span></p>
        <p><b>Wind Turbine:</b> <span style="color:#06b6d4; font-size:1.2rem;">{w_used} kW</span></p>
        <hr style="border-color:#334155;">
        <p><b>Total Generation:</b> <span style="color:#10b981; font-weight:bold;">{round(s_used + w_used, 1)} kW</span></p>
    </div>
    """, unsafe_allow_html=True)

with col_bus:
    st.markdown(f"""
    <div class="polar-card" style="border: 2px solid #0284c7;">
        <h4 style="color: #38bdf8; margin-top: 0;">⚡ Main Station Busbar</h4>
        <h2 style="color: white; margin: 10px 0;">{l_total} kW</h2>
        <p style="color: #94a3b8; margin:0;">Active Power Balanced</p>
        <p style="color: #38bdf8; font-size:0.85rem; margin-top:8px;">Frequency: 50.0 Hz (Stable)</p>
    </div>
    """, unsafe_allow_html=True)

with col_store:
    st.markdown(f"""
    <div class="polar-card">
        <h4 style="color: #10b981; margin-top: 0;">🔋 Storage & Backup</h4>
        <p><b>Battery SOC:</b> <span style="color:#10b981; font-weight:bold;">{soc}%</span></p>
        <p><b>Batt Discharge:</b> <span style="color:#34d399;">{b_dis} kW</span></p>
        <p><b>Batt Charge:</b> <span style="color:#a78bfa;">{b_chg} kW</span></p>
        <p><b>Diesel Backup:</b> <span style="color:#f87171;">{d_gen} kW</span></p>
    </div>
    """, unsafe_allow_html=True)

with col_load:
    st.markdown(f"""
    <div class="polar-card">
        <h4 style="color: #e2e8f0; margin-top: 0;">🏠 Station Load Tiers</h4>
        <p><b>Tier 1 Critical:</b> <span style="color:#ef4444; font-weight:bold;">{round(l_crit,1)} kW</span></p>
        <p><b>Tier 2 Important:</b> <span style="color:#f59e0b;">{round(l_imp,1)} kW</span></p>
        <p><b>Tier 3 Non-Critical:</b> <span style="color:#38bdf8;">{round(l_noncrit,1)} kW</span></p>
        <p><b>Load Shedding:</b> <span style="color:#94a3b8;">{round(l_shed,1)} kW</span></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 2. Sankey Power Distribution Diagram
st.subheader("📊 Sankey Energy Distribution Diagram")

label_list = [
    "Solar PV", "Wind Turbine", "Battery Storage", "Diesel Generator",
    "Central Busbar",
    "Tier 1 Critical (Heating/Medical)", "Tier 2 Important (Labs/Comms)", "Tier 3 Non-Critical (EV/Tools)", "Battery Charging"
]

source = []
target = []
value = []
color_list = []

# Solar -> Bus
if s_used > 0.1:
    source.append(0); target.append(4); value.append(s_used)
# Wind -> Bus
if w_used > 0.1:
    source.append(1); target.append(4); value.append(w_used)
# Battery -> Bus
if b_dis > 0.1:
    source.append(2); target.append(4); value.append(b_dis)
# Diesel -> Bus
if d_gen > 0.1:
    source.append(3); target.append(4); value.append(d_gen)

# Bus -> Critical
if l_crit > 0.1:
    source.append(4); target.append(5); value.append(l_crit)
# Bus -> Important
if l_imp > 0.1:
    source.append(4); target.append(6); value.append(l_imp)
# Bus -> Non-Critical
if l_noncrit > 0.1:
    source.append(4); target.append(7); value.append(l_noncrit)
# Bus -> Battery Charge
if b_chg > 0.1:
    source.append(4); target.append(8); value.append(b_chg)

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=label_list,
        color=["#f59e0b", "#06b6d4", "#10b981", "#ef4444", "#0284c7", "#f43f5e", "#fbbf24", "#38bdf8", "#8b5cf6"]
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color="rgba(56, 189, 248, 0.3)"
    )
)])

fig_sankey.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    height=450,
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig_sankey, use_container_width=True)
