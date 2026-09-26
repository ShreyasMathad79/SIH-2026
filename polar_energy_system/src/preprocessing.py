"""
Data Preprocessing & Feature Engineering Module
Handles CSV loading, cleaning, feature engineering, and chronological splitting.
"""

import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ['timestamp', 'historical_load', 'temperature', 'wind_speed', 'solar_irradiance']

def load_and_preprocess_data(csv_path_or_df):
    """
    Loads raw CSV data or DataFrame, validates required columns, cleans timestamps,
    handles missing values, sorts chronologically, and engineers time-series features.
    """
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    elif isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        raise ValueError("Input must be a CSV file path or a pandas DataFrame.")
        
    # Check for essential columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        
    # Standardize timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Sort chronologically and drop duplicates
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # Numeric column clean up & imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        df[col] = df[col].ffill().bfill()
        
    # 1. Date & Time Feature Extraction with Cyclical Transformations
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical sin/cos encodings for periodic time components
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # 2. Lag Features for Load Forecasting
    df['previous_hour_load'] = df['historical_load'].shift(1)
    df['previous_2_hour_load'] = df['historical_load'].shift(2)
    df['previous_24_hour_load'] = df['historical_load'].shift(24)
    
    # Rolling averages
    df['rolling_mean_load_6h'] = df['historical_load'].shift(1).rolling(window=6, min_periods=1).mean()
    df['rolling_mean_load_24h'] = df['historical_load'].shift(1).rolling(window=24, min_periods=1).mean()
    df['rolling_std_load_24h'] = df['historical_load'].shift(1).rolling(window=24, min_periods=1).std().fillna(0)
    
    # Weather lag features
    df['lag_temp_1h'] = df['temperature'].shift(1)
    df['lag_wind_1h'] = df['wind_speed'].shift(1)
    df['lag_solar_1h'] = df['solar_irradiance'].shift(1)
    
    # Solar and Wind Generation lag features if available
    if 'solar_generation' in df.columns:
        df['previous_hour_solar'] = df['solar_generation'].shift(1)
    else:
        df['solar_generation'] = 0.0
        df['previous_hour_solar'] = 0.0
        
    if 'wind_generation' in df.columns:
        df['previous_hour_wind'] = df['wind_generation'].shift(1)
    else:
        df['wind_generation'] = 0.0
        df['previous_hour_wind'] = 0.0
        
    if 'personnel_count' not in df.columns:
        df['personnel_count'] = 30
        
    if 'battery_soc' not in df.columns:
        df['battery_soc'] = 75.0
        
    # Fill remaining NaNs produced by lag features using backfill
    df = df.bfill().ffill()
    
    return df

def train_val_test_split_chronological(df, train_ratio=0.70, val_ratio=0.15):
    """
    Splits time-series data chronologically without shuffling.
    70% train, 15% validation, 15% testing.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df
