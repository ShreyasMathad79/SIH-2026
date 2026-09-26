"""
Polar Energy Simulation Module
Simulates extreme polar scenarios (blizzards, polar night, severe cold, fuel scarcity)
and re-evaluates forecasts, battery reserves, diesel generator, and optimization dispatch.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_scenario_weather(scenario_name, base_timestamp=None, hours=24):
    """
    Generates a 24-hour future weather & station activity profile based on scenario_name.
    """
    if base_timestamp is None:
        base_timestamp = pd.to_datetime("2025-06-15 00:00:00")
        
    timestamps = [base_timestamp + timedelta(hours=i) for i in range(hours)]
    
    df = pd.DataFrame({'timestamp': timestamps})
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['personnel_count'] = 30
    
    if scenario_name == "Normal Day":
        df['temperature'] = -12.0 + 3.0 * np.sin(2 * np.pi * (df['hour'] - 9) / 24)
        df['wind_speed'] = 8.0 + 2.0 * np.random.normal(0, 1, hours)
        df['solar_irradiance'] = 500.0 * np.maximum(0, np.sin(np.pi * (df['hour'] - 6) / 12))
        
    elif scenario_name == "Polar Night (Zero Solar)":
        df['temperature'] = -28.0 + 1.5 * np.sin(2 * np.pi * (df['hour'] - 9) / 24)
        df['wind_speed'] = 11.0 + 3.0 * np.random.normal(0, 1, hours)
        df['solar_irradiance'] = 0.0 # Complete polar darkness
        
    elif scenario_name == "High Wind Surge":
        df['temperature'] = -15.0 + np.random.normal(0, 1, hours)
        df['wind_speed'] = 19.0 + 4.0 * np.random.normal(0, 1, hours)
        df['solar_irradiance'] = 350.0 * np.maximum(0, np.sin(np.pi * (df['hour'] - 6) / 12))
        
    elif scenario_name == "Severe Polar Storm":
        # Extreme blizzard: -38°C, zero solar, wind turbine cut-out risk (> 25 m/s)
        df['temperature'] = -38.0 + np.random.normal(0, 1.5, hours)
        df['wind_speed'] = 26.5 + 5.0 * np.random.normal(0, 1, hours)
        df['solar_irradiance'] = 0.0
        
    elif scenario_name == "Extreme Cold Snap":
        # Deep freeze -45°C -> maximum heating power demand
        df['temperature'] = -45.0 + np.random.normal(0, 1, hours)
        df['wind_speed'] = 6.0 + 1.5 * np.random.normal(0, 1, hours)
        df['solar_irradiance'] = 150.0 * np.maximum(0, np.sin(np.pi * (df['hour'] - 7) / 10))
        
    elif scenario_name == "High Station Activity":
        df['temperature'] = -10.0 + 2.0 * np.sin(2 * np.pi * (df['hour'] - 9) / 24)
        df['wind_speed'] = 9.0 + np.random.normal(0, 1, hours)
        df['solar_irradiance'] = 600.0 * np.maximum(0, np.sin(np.pi * (df['hour'] - 6) / 12))
        df['personnel_count'] = 52 # Heavy summer research expedition
        
    elif scenario_name == "Low Battery Reserve":
        df['temperature'] = -20.0 + np.random.normal(0, 1, hours)
        df['wind_speed'] = 7.0 + np.random.normal(0, 1, hours)
        df['solar_irradiance'] = 200.0 * np.maximum(0, np.sin(np.pi * (df['hour'] - 6) / 12))
        
    elif scenario_name == "Low Diesel Fuel":
        df['temperature'] = -22.0 + np.random.normal(0, 1, hours)
        df['wind_speed'] = 12.0 + np.random.normal(0, 1, hours)
        df['solar_irradiance'] = 300.0 * np.maximum(0, np.sin(np.pi * (df['hour'] - 6) / 12))
        
    else:
        df['temperature'] = -15.0
        df['wind_speed'] = 10.0
        df['solar_irradiance'] = 300.0

    df['temperature'] = df['temperature'].round(1)
    df['wind_speed'] = np.clip(df['wind_speed'], 0.1, 40.0).round(1)
    df['solar_irradiance'] = np.clip(df['solar_irradiance'], 0.0, 1000.0).round(1)
    
    return df
