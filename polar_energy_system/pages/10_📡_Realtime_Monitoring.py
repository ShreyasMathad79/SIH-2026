import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
"""
Real-Time Telemetry & Hardware SCADA Control Platform Page
Interactive Multi-Station IoT Sensor Feed, SCADA Alarm Manager, 32-Cell Battery Thermal Heatmap,
Remote Control Switchboard, and Electrical Power Quality Monitor.
"""

import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

try:
    from src.state_helper import initialize_system_state, save_persistent_state, load_persistent_state
    from src.sensor_telemetry import generate_live_sensor_telemetry, STATION_PROFILES
except ModuleNotFoundError:
    from state_helper import initialize_system_state, save_persistent_state, load_persistent_state
    from sensor_telemetry import generate_live_sensor_telemetry, STATION_PROFILES
initialize_system_state()

p_state = load_persistent_state()
facility_keys = list(STATION_PROFILES.keys())
def_fac = p_state.get('selected_facility', facility_keys[0])
fac_idx = facility_keys.index(def_fac) if def_fac in facility_keys else 0

scenarios = [
    "Normal Day", "Polar Night (Zero Solar)", "High Wind Surge",
    "Severe Polar Storm", "Extreme Cold Snap", "High Station Activity",
    "Low Battery Reserve", "Low Diesel Fuel"
]
def_scen = p_state.get('active_scenario', "Normal Day")
scen_idx = scenarios.index(def_scen) if def_scen in scenarios else 0

st.set_page_config(
    page_title="Real-Time SCADA Monitoring Platform | NCPOR / MoES",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Real-Time Polar SCADA & Telemetry Control Platform")
st.caption("Multi-Station Fleet Monitoring, Hardware Diagnostics, SCADA Alarms & Remote Circuit Controls | MoES / NCPOR")

# Top Platform Navigation Bar
c_st, c_sc, c_time, c_poll = st.columns([3, 2, 2, 1])

with c_st:
    station_selected = st.selectbox(
        "📍 Select Polar Facility",
        facility_keys,
        index=fac_idx
    )

with c_sc:
    scenario = st.selectbox(
        "Weather & Operational Scenario",
        scenarios,
        index=scen_idx
    )

with c_time:
    hour_inspect = st.slider("Timeline Hour (UTC)", 0, 23, p_state.get('hour_inspect', 14), format="%d:00 UTC")

save_persistent_state({
    'selected_facility': station_selected,
    'active_scenario': scenario,
    'hour_inspect': hour_inspect
})

with c_poll:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Poll Sensors", use_container_width=True):
        st.rerun()

# Station Specifications Bar
profile = STATION_PROFILES[station_selected]
st.info(f"**Facility Assets**: Solar PV: `{profile['solar_capacity']} kW` | Wind Turbine: `{profile['wind_capacity']} kW` | BESS Storage: `{profile['bess_kwh']} kWh` | Diesel Backup: `{profile['diesel_kw']} kW` | Lat: `{profile['lat']}°`")

# Generate Telemetry
telem = generate_live_sensor_telemetry(station_name=station_selected, scenario_name=scenario, hour_offset=hour_inspect)
bess = telem['battery_cells']
eng = telem['diesel_engine']
pv = telem['solar_pv']
wind = telem['wind_turbine']
pq = telem['power_quality']
alarms = telem['scada_alarms']

# Overall Telemetry Status Banner
if scenario == "Severe Polar Storm":
    status_msg = "🚨 SYSTEM ALERT: HIGH WIND CUT-OUT ENGAGED | DIESEL GENERATOR RUNNING"
    status_bg = "rgba(239, 68, 68, 0.2)"
    status_border = "#ef4444"
elif bess['balancing_active']:
    status_msg = "🟡 SYSTEM NOTICE: BESS ACTIVE CELL BALANCING IN PROGRESS"
    status_bg = "rgba(245, 158, 11, 0.2)"
    status_border = "#f59e0b"
else:
    status_msg = "🟢 ALL HARDWARE TELEMETRY NOMINAL | MICROGRID STABLE (50.0 Hz)"
    status_bg = "rgba(16, 185, 129, 0.2)"
    status_border = "#10b981"

st.markdown(f"""
<div style="background: {status_bg}; border: 1px solid {status_border}; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px;">
    <h4 style="margin: 0; color: white;">{status_msg}</h4>
    <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.85rem;">{telem['timestamp_label']}</p>
</div>
""", unsafe_allow_html=True)

# Main Diagnostic Platform Tabs
tab_scada, tab_bess, tab_eng, tab_ren, tab_pq, tab_controls, tab_api = st.tabs([
    "🚨 SCADA Alarms & Event Logs",
    "🔋 BESS 32-Cell Thermal & Voltage",
    "🚜 Diesel Engine Diagnostics",
    "☀️ Solar & 💨 Wind Telemetry",
    "⚡ Electrical Power Quality",
    "🎛️ SCADA Remote Control Room",
    "📡 IoT MQTT Stream API"
])

# 1. SCADA ALARMS & EVENT LOGS TAB
with tab_scada:
    st.subheader("🚨 Live SCADA Alarm & Event Log Manager")
    st.caption("Real-Time Telemetry Alarm Filtering, Event Timestamps & Equipment Trip Signals")
    
    a1, a2, a3, a4 = st.columns(4)
    active_critical = sum(1 for a in alarms if a['severity'] == 'CRITICAL' and a['status'] == 'ACTIVE')
    active_warning = sum(1 for a in alarms if a['severity'] == 'WARNING' and a['status'] == 'ACTIVE')
    
    a1.metric("Active Critical Alarms", f"{active_critical}", delta="Action Required" if active_critical > 0 else "Normal")
    a2.metric("Active Warnings", f"{active_warning}", delta="Monitoring" if active_warning > 0 else "Clear")
    a3.metric("IoT Telemetry Latency", "12 ms", delta="Sub-Second Stream")
    a4.metric("SCADA Gateway Status", "ONLINE", delta="MQTT Port 1883")
    
    st.markdown("#### Live Alarm Log Stream")
    alarms_df = pd.DataFrame(alarms)
    st.dataframe(
        alarms_df,
        column_config={
            "time": "Timestamp",
            "asset": "Equipment Asset",
            "severity": st.column_config.SelectboxColumn("Severity", options=["CRITICAL", "WARNING", "INFO"]),
            "message": "Alarm / Event Description",
            "status": "State"
        },
        hide_index=True,
        use_container_width=True
    )

# 2. BESS 32-CELL THERMAL & VOLTAGE TAB
with tab_bess:
    st.subheader("Battery Energy Storage System (32-Cell Diagnostics & Thermal Heatmap)")
    st.caption(f"LiFePO4 Chemistry • {profile['bess_kwh']} kWh Capacity • 4x8 Module Temperature & Voltage Distribution")
    
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("State of Health (SoH)", f"{bess['bess_soh']}%", delta="Cell Health Excellent")
    b2.metric("Pack Avg Temp", f"{bess['bess_temp']} °C", delta="Thermal Management OK")
    b3.metric("Max Cell Delta V", f"{bess['cell_delta_v']} V", delta="Balancing Active" if bess['balancing_active'] else "Uniform")
    b4.metric("BMS Balancer", "ACTIVE" if bess['balancing_active'] else "IDLE", delta=f"Min {bess['min_cell_v']}V | Max {bess['max_cell_v']}V")
    b5.metric("Cycle Count", f"{bess['cycle_count']} Cycles", delta="Design Life 5,000")
    
    col_hm1, col_hm2 = st.columns(2)
    
    with col_hm1:
        st.markdown("#### 4x8 Module Thermal Heatmap (°C)")
        fig_heat = go.Figure(data=go.Heatmap(
            z=bess['cell_temps_grid'],
            x=[f"Col {j+1}" for j in range(8)],
            y=[f"Row {i+1}" for i in range(4)],
            colorscale='Thermal',
            text=bess['cell_temps_grid'],
            texttemplate="%{text}°C",
            showscale=True
        ))
        fig_heat.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', height=300,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_hm2:
        st.markdown("#### 32-Cell Voltage Distribution (Volts per Cell)")
        cells_df = pd.DataFrame({
            'Cell_ID': [f"Cell {i+1}" for i in range(32)],
            'Voltage': bess['cell_voltages']
        })
        colors = ['#ef4444' if v == bess['max_cell_v'] else ('#38bdf8' if v == bess['min_cell_v'] else '#10b981') for v in bess['cell_voltages']]
        fig_cells = go.Figure(go.Bar(
            x=cells_df['Cell_ID'], y=cells_df['Voltage'],
            marker_color=colors, text=cells_df['Voltage'], textposition='outside'
        ))
        fig_cells.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
            height=300, margin=dict(l=10, r=10, t=20, b=20),
            xaxis=dict(tickangle=-45), yaxis=dict(range=[3.0, 3.6], showgrid=True, gridcolor='#334155')
        )
        st.plotly_chart(fig_cells, use_container_width=True)

# 3. DIESEL ENGINE TAB
with tab_eng:
    st.subheader("Diesel Generator Engine Health & Maintenance Diagnostics")
    st.caption(f"{profile['diesel_kw']} kW Heavy-Duty Polar Diesel Engine • Automated Lube Oil & Coolant Sensors")
    
    st.markdown(f"**Engine Status**: `{eng['engine_status']}`")
    
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Engine RPM", f"{eng['engine_rpm']} RPM", delta="Nominal 1500 RPM")
    e2.metric("Lube Oil Pressure", f"{eng['lube_oil_press_bar']} bar", delta="Normal (3.5 - 4.5 bar)")
    e3.metric("Coolant Temp", f"{eng['coolant_temp_c']} °C", delta="Normal (80 - 92 °C)")
    e4.metric("Exhaust Gas Temp", f"{eng['exhaust_temp_c']} °C", delta="Thermal Loading")
    e5.metric("Service Countdown", f"{eng['hours_to_service']} Hours", delta="500h Interval")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_gauge_press = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = eng['lube_oil_press_bar'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Lube Oil Pressure (bar)"},
            gauge = {
                'axis': {'range': [0, 6]},
                'bar': {'color': "#38bdf8"},
                'steps': [
                    {'range': [0, 2.5], 'color': "rgba(239, 68, 68, 0.4)"},
                    {'range': [2.5, 5.0], 'color': "rgba(16, 185, 129, 0.4)"},
                    {'range': [5.0, 6.0], 'color': "rgba(245, 158, 11, 0.4)"}
                ]
            }
        ))
        fig_gauge_press.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', height=240)
        st.plotly_chart(fig_gauge_press, use_container_width=True)
        
    with col_g2:
        fig_gauge_temp = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = eng['coolant_temp_c'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Coolant Temperature (°C)"},
            gauge = {
                'axis': {'range': [0, 120]},
                'bar': {'color': "#f59e0b"},
                'steps': [
                    {'range': [0, 60], 'color': "rgba(56, 189, 248, 0.4)"},
                    {'range': [60, 95], 'color': "rgba(16, 185, 129, 0.4)"},
                    {'range': [95, 120], 'color': "rgba(239, 68, 68, 0.4)"}
                ]
            }
        ))
        fig_gauge_temp.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', height=240)
        st.plotly_chart(fig_gauge_temp, use_container_width=True)

# 4. SOLAR & WIND SENSOR TAB
with tab_ren:
    st.subheader("Solar PV Array & Wind Turbine Sensor Telemetry")
    
    st.markdown(f"#### ☀️ Photovoltaic Array ({profile['solar_capacity']} kW Peak)")
    pv1, pv2, pv3, pv4 = st.columns(4)
    pv1.metric("Solar Irradiance", f"{pv['pv_irradiance']} W/m²")
    pv2.metric("String Voltage", f"{pv['pv_string_voltage']} V")
    pv3.metric("String Current", f"{pv['pv_string_current']} A")
    pv4.metric("MPPT Tracking Efficiency", f"{pv['mppt_efficiency']}%")
    
    st.markdown("---")
    
    st.markdown(f"#### 💨 Wind Turbine Generator ({profile['wind_capacity']} kW Peak)")
    st.markdown(f"**Turbine Status**: `{wind['wind_status']}`")
    
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Anemometer Wind Speed", f"{wind['wind_speed_ms']} m/s")
    w2.metric("Rotor Speed", f"{wind['turbine_rpm']} RPM")
    w3.metric("Blade Pitch Angle", f"{wind['blade_pitch_deg']}°")
    w4.metric("Gearbox Temp", f"{wind['gearbox_temp_c']} °C", delta=f"Vibration {wind['bearing_vibration_mms']} mm/s")

# 5. POWER QUALITY TAB
with tab_pq:
    st.subheader("3-Phase Station Electrical Busbar Power Quality")
    st.caption("400V AC 3-Phase 50Hz • Real-Time Frequency Stability & Harmonic Distortion Monitor")
    
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Grid Frequency", f"{pq['freq_hz']} Hz", delta="Target 50.00 Hz (±0.05 Hz)")
    p2.metric("3-Phase Voltage (Va/Vb/Vc)", f"{pq['v_a']} V", delta=f"Vb: {pq['v_b']}V | Vc: {pq['v_c']}V")
    p3.metric("Power Factor (cos φ)", f"{pq['power_factor']}", delta="Optimal > 0.95")
    p4.metric("Harmonic Distortion (THD)", f"{pq['thd_pct']}%", delta="IEEE 519 Limit < 5%")
    
    st.markdown("#### 3-Phase AC Voltage Waveform (400V / 50Hz)")
    t_val = np.linspace(0, 0.04, 200)
    omega = 2 * np.pi * pq['freq_hz']
    
    v_a_wave = pq['v_a'] * np.sqrt(2) * np.sin(omega * t_val)
    v_b_wave = pq['v_b'] * np.sqrt(2) * np.sin(omega * t_val - 2*np.pi/3)
    v_c_wave = pq['v_c'] * np.sqrt(2) * np.sin(omega * t_val + 2*np.pi/3)
    
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t_val*1000, y=v_a_wave, name='Phase A (V)', line=dict(color='#ef4444', width=2)))
    fig_wave.add_trace(go.Scatter(x=t_val*1000, y=v_b_wave, name='Phase B (V)', line=dict(color='#f59e0b', width=2)))
    fig_wave.add_trace(go.Scatter(x=t_val*1000, y=v_c_wave, name='Phase C (V)', line=dict(color='#38bdf8', width=2)))
    
    fig_wave.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title='Time (milliseconds)', showgrid=True, gridcolor='#334155'),
        yaxis=dict(title='Instantaneous Voltage (V)', showgrid=True, gridcolor='#334155')
    )
    st.plotly_chart(fig_wave, use_container_width=True)

# 6. SCADA REMOTE CONTROL ROOM TAB
with tab_controls:
    st.subheader("🎛️ SCADA Operator Remote Control Panel")
    st.caption("Interactive Equipment Switchboard for Station Power Operators")
    
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        st.markdown("**Diesel Generator Breaker**")
        d_breaker = st.toggle("Engage Main Gen Breaker CB-101", value=eng['engine_running'])
        st.caption("State: " + ("CLOSED (ENGAGED)" if d_breaker else "OPEN (STANDBY)"))
        
    with ctrl2:
        st.markdown("**BESS Cell Equalizer Mode**")
        b_equal = st.toggle("Enable Active Shunt Equalizer", value=bess['balancing_active'])
        st.caption("State: " + ("ACTIVE BALANCING" if b_equal else "IDLE"))
        
    with ctrl3:
        st.markdown("**Wind Turbine Aerodynamic Brake**")
        w_brake = st.toggle("Lock Turbine Brake CB-202", value=scenario=="Severe Polar Storm")
        st.caption("State: " + ("BRAKE LOCKED" if w_brake else "PITCH AUTO"))
        
    with ctrl4:
        st.markdown("**Diesel Block Pre-Heater**")
        h_override = st.toggle("Manual Preheat Override", value=True)
        st.caption("State: " + ("PRE-HEATER 4kW ON" if h_override else "AUTO TEMP"))
        
    st.markdown("---")
    st.success("✓ Control Telemetry Signals Synchronized with Station Modbus SCADA Controller.")

# 7. IOT MQTT STREAM API TAB
with tab_api:
    st.subheader("📡 IoT Telemetry Stream Payload (MQTT Preview)")
    st.caption("JSON Telemetry Payload Format for Hardware Sensor Integration")
    
    json_str = json.dumps(telem, indent=2)
    st.code(json_str, language='json')
    
    st.download_button(
        label="📥 Download Real-Time Telemetry JSON",
        data=json_str,
        file_name=f"polar_station_telemetry_{hour_inspect:02d}00.json",
        mime="application/json"
    )
