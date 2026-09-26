"""
Real-Time Polar Telemetry & Dataset Exporter Script (7-Day / 168-Hour Horizon)
Fetches live high-precision 7-day satellite telemetry from Open-Meteo API and exports present real-time dataset files.
"""

import os
import json
import pandas as pd
from datetime import datetime

from src.live_weather_api import fetch_live_polar_weather, STATION_COORDINATES
from src.optimization import EnergyOptimizer

def export_present_realtime_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "data", "sample")
    os.makedirs(sample_dir, exist_ok=True)
    
    print("Connecting to Live 7-Day Satellite Telemetry Stream (Open-Meteo)...")
    master_records = []
    
    for station_name in STATION_COORDINATES.keys():
        print(f"--> Fetching live 7-day stream for: {station_name}")
        live_data = fetch_live_polar_weather(station_name, days=7)
        w_df = live_data['weather_df'].copy()
        
        # Prepare 168h predictions
        load_kw = (38.0 + (-w_df['temperature']) * 0.75).values.tolist()
        solar_kw = (w_df['solar_irradiance'] * 0.08).clip(0, 60).values.tolist()
        wind_kw = ((w_df['wind_speed'] - 3.0) * 4.5).clip(0, 80).values.tolist()
        
        optimizer = EnergyOptimizer()
        timestamps = [t.strftime("%Y-%m-%d %H:00") for t in w_df['timestamp']]
        opt_df = optimizer.optimize_dispatch(load_kw, solar_kw, wind_kw, timestamps=timestamps)
        
        for idx, row in w_df.iterrows():
            d_row = opt_df.iloc[idx]
            master_records.append({
                'station_name': station_name,
                'timestamp': row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'temperature_c': row['temperature'],
                'relative_humidity_pct': row.get('relative_humidity', 60.0),
                'apparent_temp_c': row.get('apparent_temperature', row['temperature'] - 4.0),
                'surface_pressure_hpa': row.get('surface_pressure', 985.0),
                'wind_speed_ms': row['wind_speed'],
                'wind_direction_deg': row.get('wind_direction', 180),
                'wind_gust_ms': row.get('wind_gusts', row['wind_speed'] * 1.4),
                'solar_irradiance_wm2': row['solar_irradiance'],
                'cloud_cover_pct': row.get('cloud_cover', 50.0),
                'predicted_demand_kw': d_row['demand_kw'],
                'predicted_solar_pv_kw': d_row['solar_used_kw'],
                'predicted_wind_kw': d_row['wind_used_kw'],
                'battery_discharge_kw': d_row['battery_discharge_kw'],
                'battery_charge_kw': d_row['battery_charge_kw'],
                'battery_soc_pct': d_row['battery_soc'],
                'diesel_output_kw': d_row['diesel_output_kw'],
                'diesel_fuel_liters': d_row['diesel_fuel_liters']
            })
            
    df_export = pd.DataFrame(master_records)
    
    # Save CSV
    csv_path = os.path.join(sample_dir, "live_polar_station_telemetry_7days.csv")
    json_path = os.path.join(sample_dir, "live_polar_station_telemetry_7days.json")
    df_export.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(master_records, f, indent=2)
        
    print(f"Live 7-Day Real-Time Telemetry CSV saved to: {csv_path} ({len(df_export)} rows)")
    print(f"Live 7-Day Real-Time Telemetry JSON saved to: {json_path}")
    return csv_path, json_path

if __name__ == "__main__":
    export_present_realtime_dataset()
