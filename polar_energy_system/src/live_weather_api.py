"""
Live Satellite & Polar Weather API Integration Module (7-Day / 168-Hour Horizon)
Fetches real-time ambient weather data from Open-Meteo API for polar station coordinates
(Bharati -69.4°S, Maitri -70.7°S, Himadri 78.9°N) for up to 7 full days (168 hours).
Includes Surface Pressure (hPa), Humidity (%), Wind Chill (°C), Wind Direction (°), and Gusts (m/s).
"""

import urllib.request
import json
import ssl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

STATION_COORDINATES = {
    "Bharati Station (Antarctica - Larsemann Hills, 69.4°S)": {"lat": -69.4, "lon": 76.2},
    "Maitri Station (Antarctica - Schirmacher Oasis, 70.7°S)": {"lat": -70.7, "lon": 11.7},
    "Himadri Station (Arctic - Ny-Ålesund, Svalbard, 78.9°N)": {"lat": 78.9, "lon": 11.9}
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_polar_weather(station_name="Bharati Station (Antarctica - Larsemann Hills, 69.4°S)", days=7):
    """
    Fetches high-precision live real-time ambient weather for specified polar station for up to `days` (default 7 days = 168 hours).
    """
    coords = STATION_COORDINATES.get(station_name, STATION_COORDINATES["Bharati Station (Antarctica - Larsemann Hills, 69.4°S)"])
    lat = coords["lat"]
    lon = coords["lon"]
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,direct_normal_irradiance,cloud_cover&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,direct_normal_irradiance,cloud_cover&forecast_days={days}"
    
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 PolarEnergySystem/2.0 7DayHorizon'})
        with urllib.request.urlopen(req, context=ctx, timeout=8) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        current = data.get('current', {})
        hourly = data.get('hourly', {})
        
        temp_c = float(current.get('temperature_2m', -15.0))
        humidity_pct = float(current.get('relative_humidity_2m', 60.0))
        apparent_temp_c = float(current.get('apparent_temperature', temp_c - 4.0))
        pressure_hpa = float(current.get('surface_pressure', 985.0))
        wind_kmh = float(current.get('wind_speed_10m', 20.0))
        wind_ms = round(wind_kmh / 3.6, 1)
        wind_dir_deg = int(current.get('wind_direction_10m', 180))
        wind_gust_ms = round(float(current.get('wind_gusts_10m', wind_kmh * 1.5)) / 3.6, 1)
        solar_dni = float(current.get('direct_normal_irradiance', 0.0))
        cloud_pct = float(current.get('cloud_cover', 50.0))
        
        # 168-hour (7 days) hourly weather forecast dataframe
        raw_times = hourly.get('time', [])
        target_hours = days * 24
        
        timestamps = [pd.to_datetime(t) for t in raw_times[:target_hours]]
        temps = hourly.get('temperature_2m', [])[:target_hours]
        hums = hourly.get('relative_humidity_2m', [])[:target_hours]
        app_temps = hourly.get('apparent_temperature', [])[:target_hours]
        pressures = hourly.get('surface_pressure', [])[:target_hours]
        winds = [round(w / 3.6, 1) for w in hourly.get('wind_speed_10m', [])[:target_hours]]
        wind_dirs = hourly.get('wind_direction_10m', [])[:target_hours]
        wind_gusts = [round(g / 3.6, 1) for g in hourly.get('wind_gusts_10m', [])[:target_hours]]
        solars = hourly.get('direct_normal_irradiance', [])[:target_hours]
        clouds = hourly.get('cloud_cover', [])[:target_hours]
        
        n_rows = len(timestamps)
        weather_df = pd.DataFrame({
            'timestamp': timestamps if n_rows == target_hours else [pd.Timestamp.now() + timedelta(hours=i) for i in range(target_hours)],
            'temperature': temps if len(temps) == target_hours else [temp_c] * target_hours,
            'relative_humidity': hums if len(hums) == target_hours else [humidity_pct] * target_hours,
            'apparent_temperature': app_temps if len(app_temps) == target_hours else [apparent_temp_c] * target_hours,
            'surface_pressure': pressures if len(pressures) == target_hours else [pressure_hpa] * target_hours,
            'wind_speed': winds if len(winds) == target_hours else [wind_ms] * target_hours,
            'wind_direction': wind_dirs if len(wind_dirs) == target_hours else [wind_dir_deg] * target_hours,
            'wind_gusts': wind_gusts if len(wind_gusts) == target_hours else [wind_gust_ms] * target_hours,
            'solar_irradiance': solars if len(solars) == target_hours else [solar_dni] * target_hours,
            'cloud_cover': clouds if len(clouds) == target_hours else [cloud_pct] * target_hours
        })
        
        return {
            'is_live': True,
            'status_label': f'🟢 LIVE 7-DAY SATELLITE FEED CONNECTED (Open-Meteo: {len(weather_df)} Hours)',
            'temp_c': temp_c,
            'humidity_pct': humidity_pct,
            'apparent_temp_c': apparent_temp_c,
            'pressure_hpa': pressure_hpa,
            'wind_ms': wind_ms,
            'wind_dir_deg': wind_dir_deg,
            'wind_gust_ms': wind_gust_ms,
            'solar_wm2': solar_dni,
            'cloud_pct': cloud_pct,
            'timestamp': current.get('time', datetime.now().strftime("%Y-%m-%d %H:%M")),
            'weather_df': weather_df
        }
    except Exception as e:
        now_ts = datetime.now()
        target_hours = days * 24
        timestamps = [now_ts + timedelta(hours=i) for i in range(target_hours)]
        return {
            'is_live': False,
            'status_label': f'🟡 SIMULATED FEED (Offline / Fallback)',
            'temp_c': -18.5,
            'humidity_pct': 62.0,
            'apparent_temp_c': -23.1,
            'pressure_hpa': 982.5,
            'wind_ms': 12.4,
            'wind_dir_deg': 210,
            'wind_gust_ms': 18.2,
            'solar_wm2': 140.0,
            'cloud_pct': 45.0,
            'timestamp': now_ts.strftime("%Y-%m-%d %H:%M"),
            'weather_df': pd.DataFrame({
                'timestamp': timestamps,
                'temperature': np.random.normal(-18.5, 3.0, target_hours).round(1),
                'relative_humidity': [62.0] * target_hours,
                'apparent_temperature': np.random.normal(-23.1, 3.5, target_hours).round(1),
                'surface_pressure': np.random.normal(982.5, 4.0, target_hours).round(1),
                'wind_speed': np.random.normal(12.4, 2.0, target_hours).round(1),
                'wind_direction': [210] * target_hours,
                'wind_gusts': np.random.normal(18.2, 2.5, target_hours).round(1),
                'solar_irradiance': np.maximum(0, 450 * np.sin(np.tile(np.linspace(0, np.pi, 24), days))).round(1),
                'cloud_cover': [45.0] * target_hours
            })
        }
