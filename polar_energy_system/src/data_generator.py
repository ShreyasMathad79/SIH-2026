"""
Simulated Polar Research Station Dataset Generator
Generates realistic physics-informed time-series data for an isolated Antarctic research station.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_polar_station_data(
    start_date="2025-01-01",
    days=365,
    random_seed=42,
    output_path=None
):
    """
    Generates 1 year (or specified days) of 1-hour resolution data for a polar station.
    Station Specs:
    - Solar PV Peak: 60 kW
    - Wind Turbine Peak: 80 kW
    - Battery Storage: 200 kWh capacity (min SOC 20%, max SOC 95%)
    - Diesel Generator: 100 kW max capacity (min stable load 15 kW)
    """
    np.random.seed(random_seed)
    
    total_hours = days * 24
    dates = [pd.to_datetime(start_date) + timedelta(hours=i) for i in range(total_hours)]
    
    df = pd.DataFrame({'timestamp': dates})
    
    # Time features
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 1. Personnel / Occupants (Summer expedition ~45, Winter team ~25)
    def get_occupants(month):
        if month in [11, 12, 1, 2]:
            return int(np.random.normal(45, 3))
        elif month in [3, 10]:
            return int(np.random.normal(32, 2))
        else:
            return int(np.random.normal(25, 1))
    
    df['personnel_count'] = df['month'].apply(get_occupants)
    
    # 2. Ambient Temperature (°C)
    seasonal_temp = -18 - 14 * np.cos(2 * np.pi * (df['day_of_year'] - 15) / 365)
    diurnal_temp = 2.5 * np.sin(2 * np.pi * (df['hour'] - 9) / 24) * (1 - 0.7 * (df['month'].isin([5, 6, 7, 8])))
    noise_temp = np.random.normal(0, 3.5, total_hours)
    raw_temp = seasonal_temp + diurnal_temp + noise_temp
    df['temperature'] = pd.Series(raw_temp).rolling(window=6, min_periods=1, center=True).mean().round(1)
    
    # 3. Wind Speed (m/s)
    base_wind = np.random.weibull(a=2.1, size=total_hours) * 7.5
    storm_mask = np.zeros(total_hours)
    num_storms = int(days / 8)
    for _ in range(num_storms):
        start_h = np.random.randint(0, total_hours - 48)
        duration = np.random.randint(12, 36)
        storm_mask[start_h:start_h + duration] = np.sin(np.linspace(0, np.pi, duration)) * 18.0
    
    df['wind_speed'] = np.clip(base_wind + storm_mask, 0.2, 38.0).round(1)
    
    # 4. Solar Irradiance (W/m²)
    solar_declination = 23.45 * np.sin(2 * np.pi * (df['day_of_year'] - 80) / 365)
    hour_angle = 15 * (df['hour'] - 12)
    latitude = -69.4  # Larsemann Hills (Bharati Station)
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(solar_declination)
    ha_rad = np.radians(hour_angle)
    
    sin_elevation = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    elevation_deg = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))
    
    cloud_cover = np.random.beta(a=2, b=5, size=total_hours)
    storm_cloud = (df['wind_speed'] > 18).astype(float) * 0.5
    total_cloud = np.clip(cloud_cover + storm_cloud, 0, 0.95)
    
    max_irradiance = 1000 * np.clip(np.sin(np.radians(elevation_deg)), 0, 1)
    df['solar_irradiance'] = (max_irradiance * (1 - 0.75 * total_cloud)).round(1)
    df.loc[elevation_deg < 2, 'solar_irradiance'] = 0.0
    
    # 5. Solar Generation (kW) - Max Capacity 60 kW
    pv_capacity_kw = 60.0
    temp_factor = 1 + 0.003 * (25 - df['temperature'])
    raw_solar = (df['solar_irradiance'] / 1000) * pv_capacity_kw * temp_factor * (1 - 0.1 * total_cloud)
    df['solar_generation'] = np.clip(raw_solar, 0.0, pv_capacity_kw).round(2)
    
    # 6. Wind Generation (kW) - Max Capacity 80 kW
    wind_capacity_kw = 80.0
    def calculate_wind_power(v):
        cut_in = 3.0
        rated = 12.0
        cut_out = 25.0
        if v < cut_in or v > cut_out:
            return 0.0
        elif cut_in <= v < rated:
            return wind_capacity_kw * ((v - cut_in) / (rated - cut_in)) ** 3
        else:
            return wind_capacity_kw
            
    df['wind_generation'] = df['wind_speed'].apply(calculate_wind_power).round(2)
    
    # 7. Station Load Demand (kW)
    base_load = 25.0
    heating_load = np.maximum(0, (-df['temperature']) * 0.75)
    activity_factor = np.where((df['hour'] >= 7) & (df['hour'] <= 21), 1.3, 0.8)
    occupant_load = (df['personnel_count'] / 30.0) * 8.0 * activity_factor
    random_fluctuation = np.random.normal(0, 2.5, total_hours)
    
    total_load = base_load + heating_load + occupant_load + random_fluctuation
    df['historical_load'] = np.clip(total_load, 20.0, 95.0).round(2)
    
    # 8. Weather Conditions Description
    def assign_weather(row):
        ws = row['wind_speed']
        temp = row['temperature']
        sol = row['solar_irradiance']
        
        if ws >= 22.0:
            return "Blizzard / Extreme Storm"
        elif ws >= 14.0:
            return "High Wind Surge"
        elif temp <= -30.0:
            return "Severe Sub-Zero Cold"
        elif sol == 0 and row['month'] in [5, 6, 7]:
            return "Polar Night"
        elif sol > 400:
            return "Clear Solar Day"
        else:
            return "Overcast / Moderate"
            
    df['weather_conditions'] = df.apply(assign_weather, axis=1)
    
    # 9. Initial Simulation of Battery SoC, Diesel Output, Fuel Consumption
    battery_capacity_kwh = 200.0
    battery_soc = 75.0
    
    soc_list = []
    battery_pwr_list = []
    diesel_output_list = []
    fuel_cons_list = []
    
    for idx, row in df.iterrows():
        load = row['historical_load']
        renewable_gen = row['solar_generation'] + row['wind_generation']
        net_demand = load - renewable_gen
        
        b_power = 0.0
        diesel_gen = 0.0
        fuel_used = 0.0
        
        if net_demand < 0:
            surplus = -net_demand
            max_charge_kwh = (95.0 - battery_soc) / 100.0 * battery_capacity_kwh
            charge_kwh = min(surplus, max_charge_kwh, 40.0)
            battery_soc += (charge_kwh * 0.92) / battery_capacity_kwh * 100.0
            b_power = -charge_kwh
        else:
            max_discharge_kwh = (battery_soc - 20.0) / 100.0 * battery_capacity_kwh
            discharge_kwh = min(net_demand, max_discharge_kwh, 50.0)
            
            if discharge_kwh > 0:
                battery_soc -= (discharge_kwh / 0.92) / battery_capacity_kwh * 100.0
                b_power = discharge_kwh
                
            rem_deficit = net_demand - discharge_kwh
            if rem_deficit > 0:
                diesel_gen = max(rem_deficit, 15.0)
                fuel_used = diesel_gen * 0.28
                
        battery_soc = np.clip(battery_soc, 15.0, 98.0)
        soc_list.append(round(battery_soc, 2))
        battery_pwr_list.append(round(b_power, 2))
        diesel_output_list.append(round(diesel_gen, 2))
        fuel_cons_list.append(round(fuel_used, 2))
        
    df['battery_soc'] = soc_list
    df['battery_power_kw'] = battery_pwr_list
    df['diesel_output'] = diesel_output_list
    df['diesel_fuel_consumption'] = fuel_cons_list
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset successfully saved to {output_path} ({len(df)} rows)")
        
    return df

if __name__ == "__main__":
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    output_file = os.path.join(sample_dir, "polar_station_data.csv")
    generate_polar_station_data(output_path=output_file)
