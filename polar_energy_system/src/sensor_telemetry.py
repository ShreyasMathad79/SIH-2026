"""
Real-Time Telemetry & Hardware Sensor Diagnostics Module
Simulates live IoT telemetry feeds for PV strings, wind turbines, 32 BESS battery cell voltages,
diesel generator engine sensors, and 3-phase station electrical busbar stability across polar station profiles.
"""

import numpy as np
import pandas as pd

STATION_PROFILES = {
    "Bharati Station (Antarctica - Larsemann Hills, 69.4°S)": {
        "solar_capacity": 60, "wind_capacity": 80, "bess_kwh": 200, "diesel_kw": 100, "lat": -69.4
    },
    "Maitri Station (Antarctica - Schirmacher Oasis, 70.7°S)": {
        "solar_capacity": 40, "wind_capacity": 100, "bess_kwh": 250, "diesel_kw": 120, "lat": -70.7
    },
    "Himadri Station (Arctic - Ny-Ålesund, Svalbard, 78.9°N)": {
        "solar_capacity": 30, "wind_capacity": 60, "bess_kwh": 150, "diesel_kw": 80, "lat": 78.9
    }
}

def generate_live_sensor_telemetry(station_name="Bharati Station (Antarctica - Larsemann Hills, 69.4°S)", scenario_name="Normal Day", hour_offset=12):
    """
    Generates high-frequency IoT sensor telemetry readings for Antarctic/Arctic station microgrid assets.
    """
    np.random.seed(42 + hour_offset + len(station_name))
    profile = STATION_PROFILES.get(station_name, STATION_PROFILES["Bharati Station (Antarctica - Larsemann Hills, 69.4°S)"])
    
    # 1. BESS 32-Cell Voltage Distribution & 4x8 Thermal Matrix
    base_cell_v = 3.31 + np.random.normal(0, 0.015)
    cell_noises = np.random.normal(0, 0.02, 32)
    cell_voltages = np.clip(base_cell_v + cell_noises, 3.12, 3.48).round(3)
    
    max_cell_v = float(np.max(cell_voltages))
    min_cell_v = float(np.min(cell_voltages))
    cell_delta_v = round(max_cell_v - min_cell_v, 3)
    
    cell_balancing_active = bool(cell_delta_v > 0.035)
    bess_soh = 96.4 # State of Health %
    bess_temp = round(float(18.5 + np.random.normal(0, 1.2)), 1)
    bess_cycle_count = 1420
    
    # 4x8 Thermal Matrix (°C per cell module)
    cell_temps = np.clip(bess_temp + np.random.normal(0, 1.5, (4, 8)), 12.0, 32.0).round(1)
    cell_voltages_grid = cell_voltages.reshape((4, 8))
    
    # 2. Diesel Generator Engine Telemetry
    is_storm = scenario_name in ["Severe Polar Storm", "Extreme Cold Snap", "Polar Night (Zero Solar)"]
    engine_running = is_storm or (hour_offset >= 18 and hour_offset <= 22)
    
    if engine_running:
        engine_rpm = int(np.random.normal(1500, 8))
        lube_oil_press_bar = round(float(4.2 + np.random.normal(0, 0.15)), 2)
        coolant_temp_c = round(float(86.5 + np.random.normal(0, 1.1)), 1)
        exhaust_temp_c = round(float(385.0 + np.random.normal(0, 8.0)), 1)
        fuel_flow_l_h = round(float(14.8 + np.random.normal(0, 0.5)), 1)
        air_filter_dp_mbar = round(float(12.4 + np.random.normal(0, 0.3)), 1)
        engine_status = "RUNNING (OPTIMAL LOADING)"
    else:
        engine_rpm = 0
        lube_oil_press_bar = 0.0
        coolant_temp_c = round(float(35.0 + np.random.normal(0, 0.5)), 1) # Block heater active
        exhaust_temp_c = round(float(35.0 + np.random.normal(0, 0.5)), 1)
        fuel_flow_l_h = 0.0
        air_filter_dp_mbar = 0.0
        engine_status = "STANDBY (AUTO-PREHEAT ACTIVE)"
        
    hours_to_service = 185 # Hours remaining until 500-hour oil/filter maintenance
    
    # 3. Solar PV String & Panel Telemetry
    if scenario_name == "Polar Night (Zero Solar)":
        pv_irradiance = 0.0
        pv_string_voltage = 0.0
        pv_string_current = 0.0
        pv_panel_temp = -24.0
        mppt_efficiency = 0.0
    else:
        pv_irradiance = round(float(max(0, 550.0 * np.sin(np.pi * (hour_offset - 6) / 12) + np.random.normal(0, 15))), 1)
        if pv_irradiance > 5.0:
            pv_string_voltage = round(float(398.0 + np.random.normal(0, 2.5)), 1)
            pv_string_current = round(float((pv_irradiance / 550.0) * 125.0 + np.random.normal(0, 1.5)), 1)
            pv_panel_temp = round(float(-8.0 + (pv_irradiance / 1000.0) * 12.0), 1)
            mppt_efficiency = round(float(98.6 + np.random.normal(0, 0.2)), 1)
        else:
            pv_string_voltage = 0.0
            pv_string_current = 0.0
            pv_panel_temp = -15.0
            mppt_efficiency = 0.0
            
    # 4. Wind Turbine Sensor Telemetry
    if scenario_name == "Severe Polar Storm":
        wind_speed_ms = 26.8 # Exceeds cut-out 25 m/s
        turbine_rpm = 0 # Feathered / brake locked
        blade_pitch_deg = 88.0 # Fully feathered
        gearbox_temp_c = round(float(42.0 + np.random.normal(0, 0.5)), 1)
        bearing_vibration_mms = round(float(3.2 + np.random.normal(0, 0.2)), 2) # High storm turbulence
        wind_status = "SAFETY BRAKE ENGAGED (HIGH WIND CUT-OUT)"
    else:
        wind_speed_ms = round(float(max(1.0, 11.2 + np.random.normal(0, 1.8))), 1)
        if 3.0 <= wind_speed_ms <= 25.0:
            turbine_rpm = int(min(65, (wind_speed_ms / 12.0) * 55 + np.random.normal(0, 1)))
            blade_pitch_deg = round(float(max(0.0, (wind_speed_ms - 12.0) * 2.5)), 1)
            gearbox_temp_c = round(float(52.4 + np.random.normal(0, 1.0)), 1)
            bearing_vibration_mms = round(float(0.85 + np.random.normal(0, 0.05)), 2)
            wind_status = "GENERATING (AUTO-PITCH CONTROL ACTIVE)"
        else:
            turbine_rpm = 0
            blade_pitch_deg = 0.0
            gearbox_temp_c = 25.0
            bearing_vibration_mms = 0.1
            wind_status = "STANDBY (LOW WIND CUT-IN)"
            
    # 5. Station Electrical Busbar Power Quality
    freq_hz = round(float(50.0 + np.random.normal(0, 0.03)), 2)
    v_a = round(float(400.0 + np.random.normal(0, 1.2)), 1)
    v_b = round(float(400.2 + np.random.normal(0, 1.1)), 1)
    v_c = round(float(399.8 + np.random.normal(0, 1.3)), 1)
    power_factor = round(float(min(0.99, 0.978 + np.random.normal(0, 0.005))), 3)
    thd_pct = round(float(max(0.5, 1.8 + np.random.normal(0, 0.1))), 2)
    
    # 6. Live SCADA Alarm & Event Logs
    alarm_events = []
    timestamp_str = f"{hour_offset:02d}:14:22 UTC"
    
    if scenario_name == "Severe Polar Storm":
        alarm_events.append({"time": timestamp_str, "asset": "Wind Turbine #1", "severity": "CRITICAL", "message": "High Wind Cut-Out Engaged (26.8 m/s)", "status": "ACTIVE"})
        alarm_events.append({"time": timestamp_str, "asset": "Diesel Gen #1", "severity": "INFO", "message": "Auto-Start Sequence Initiated", "status": "CLOSED"})
    elif cell_balancing_active:
        alarm_events.append({"time": timestamp_str, "asset": "BESS BMS Rack A", "severity": "WARNING", "message": f"Cell Delta V Exceeds Threshold ({cell_delta_v}V > 0.035V)", "status": "ACTIVE"})
        alarm_events.append({"time": timestamp_str, "asset": "BESS Balancer", "severity": "INFO", "message": "Active Shunt Balancing Initiated", "status": "ACTIVE"})
    
    alarm_events.append({"time": f"{hour_offset:02d}:02:10 UTC", "asset": "Station AC Bus", "severity": "INFO", "message": f"Frequency Lock Stable ({freq_hz} Hz)", "status": "CLOSED"})
    alarm_events.append({"time": f"{hour_offset:02d}:00:00 UTC", "asset": "IoT Sensor Gateway", "severity": "INFO", "message": f"Telemetry Sync OK ({station_name})", "status": "CLOSED"})
    
    return {
        'station_name': station_name,
        'station_profile': profile,
        'timestamp_label': f"Hour {hour_offset:02d}:00 Live SCADA Telemetry Stream",
        'battery_cells': {
            'cell_voltages': cell_voltages.tolist(),
            'cell_voltages_grid': cell_voltages_grid.tolist(),
            'cell_temps_grid': cell_temps.tolist(),
            'max_cell_v': max_cell_v,
            'min_cell_v': min_cell_v,
            'cell_delta_v': cell_delta_v,
            'balancing_active': cell_balancing_active,
            'bess_soh': bess_soh,
            'bess_temp': bess_temp,
            'cycle_count': bess_cycle_count
        },
        'diesel_engine': {
            'engine_status': engine_status,
            'engine_running': engine_running,
            'engine_rpm': engine_rpm,
            'lube_oil_press_bar': lube_oil_press_bar,
            'coolant_temp_c': coolant_temp_c,
            'exhaust_temp_c': exhaust_temp_c,
            'fuel_flow_l_h': fuel_flow_l_h,
            'air_filter_dp_mbar': air_filter_dp_mbar,
            'hours_to_service': hours_to_service
        },
        'solar_pv': {
            'pv_irradiance': pv_irradiance,
            'pv_string_voltage': pv_string_voltage,
            'pv_string_current': pv_string_current,
            'pv_panel_temp': pv_panel_temp,
            'mppt_efficiency': mppt_efficiency
        },
        'wind_turbine': {
            'wind_status': wind_status,
            'wind_speed_ms': wind_speed_ms,
            'turbine_rpm': turbine_rpm,
            'blade_pitch_deg': blade_pitch_deg,
            'gearbox_temp_c': gearbox_temp_c,
            'bearing_vibration_mms': bearing_vibration_mms
        },
        'power_quality': {
            'freq_hz': freq_hz,
            'v_a': v_a,
            'v_b': v_b,
            'v_c': v_c,
            'power_factor': power_factor,
            'thd_pct': thd_pct
        },
        'scada_alarms': alarm_events
    }
