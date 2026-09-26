"""
Initial Model Trainer Script
Trains Load, Solar, and Wind forecasting models on sample polar station data and saves pkl artifacts.
"""

import os
from src.preprocessing import load_and_preprocess_data, train_val_test_split_chronological
from src.load_forecasting import LoadForecaster
from src.renewable_forecasting import RenewableForecaster

def train_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_csv = os.path.join(base_dir, "data", "sample", "polar_station_data.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading data from {sample_csv}...")
    df = load_and_preprocess_data(sample_csv)
    train_df, val_df, test_df = train_val_test_split_chronological(df)
    
    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 1. Load Model
    print("Training Load Forecasting Model...")
    load_model = LoadForecaster(model_type='xgboost')
    load_model.train(train_df, val_df)
    load_metrics = load_model.evaluate(test_df)
    print(f"Load Model Metrics -> MAE: {load_metrics['MAE']} kW, RMSE: {load_metrics['RMSE']} kW, R2: {load_metrics['R2']}")
    load_model.save(os.path.join(models_dir, "load_model.pkl"))
    
    # 2. Solar Model
    print("Training Solar Forecasting Model...")
    solar_model = RenewableForecaster(resource_type='solar', model_type='xgboost')
    solar_model.train(train_df, val_df)
    solar_metrics = solar_model.evaluate(test_df)
    print(f"Solar Model Metrics -> MAE: {solar_metrics['MAE']} kW, RMSE: {solar_metrics['RMSE']} kW, R2: {solar_metrics['R2']}")
    solar_model.save(os.path.join(models_dir, "solar_model.pkl"))
    
    # 3. Wind Model
    print("Training Wind Forecasting Model...")
    wind_model = RenewableForecaster(resource_type='wind', model_type='xgboost')
    wind_model.train(train_df, val_df)
    wind_metrics = wind_model.evaluate(test_df)
    print(f"Wind Model Metrics -> MAE: {wind_metrics['MAE']} kW, RMSE: {wind_metrics['RMSE']} kW, R2: {wind_metrics['R2']}")
    wind_model.save(os.path.join(models_dir, "wind_model.pkl"))
    
    print("All models successfully trained and saved!")

if __name__ == "__main__":
    train_all()
