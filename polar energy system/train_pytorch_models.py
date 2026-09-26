"""
PyTorch Deep Learning Training Script
Trains Load, Solar, and Wind forecasting models using PyTorch Epochs & Mini-Batch Gradient Descent.
"""

import os
from src.preprocessing import load_and_preprocess_data, train_val_test_split_chronological
from src.deep_learning_forecaster import PyTorchDeepLearningForecaster
from src.load_forecasting import LOAD_FEATURES
from src.renewable_forecasting import SOLAR_FEATURES, WIND_FEATURES

def train_all_pytorch():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_csv = os.path.join(base_dir, "data", "sample", "polar_station_data.csv")
    models_dir = os.path.join(base_dir, "models", "pytorch")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading dataset from {sample_csv}...")
    df = load_and_preprocess_data(sample_csv)
    train_df, val_df, test_df = train_val_test_split_chronological(df)
    
    print(f"Chronological Dataset Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")
    
    # --------------------------------------------------------------------------
    # 1. PyTorch Load Demand Model (Epochs + Mini-Batches)
    # --------------------------------------------------------------------------
    print("======================================================================")
    print("Training PyTorch Load Demand Model (30 Epochs, Mini-Batch Size = 64)")
    print("======================================================================")
    load_dl = PyTorchDeepLearningForecaster(LOAD_FEATURES, 'historical_load', hidden_dim=128, lr=0.001)
    load_dl.train_with_epochs_and_minibatches(train_df, val_df, epochs=30, batch_size=64)
    load_metrics = load_dl.evaluate(test_df)
    print(f"--> Test Evaluation Metrics -> MAE: {load_metrics['MAE']} kW | RMSE: {load_metrics['RMSE']} kW | R2: {load_metrics['R2']}\n")
    load_dl.save(os.path.join(models_dir, "pytorch_load_model.pt"))
    
    # --------------------------------------------------------------------------
    # 2. PyTorch Solar Generation Model (Epochs + Mini-Batches)
    # --------------------------------------------------------------------------
    print("======================================================================")
    print("Training PyTorch Solar PV Model (30 Epochs, Mini-Batch Size = 64)")
    print("======================================================================")
    solar_dl = PyTorchDeepLearningForecaster(SOLAR_FEATURES, 'solar_generation', hidden_dim=128, lr=0.001)
    solar_dl.train_with_epochs_and_minibatches(train_df, val_df, epochs=30, batch_size=64)
    solar_metrics = solar_dl.evaluate(test_df)
    print(f"--> Test Evaluation Metrics -> MAE: {solar_metrics['MAE']} kW | RMSE: {solar_metrics['RMSE']} kW | R2: {solar_metrics['R2']}\n")
    solar_dl.save(os.path.join(models_dir, "pytorch_solar_model.pt"))
    
    # --------------------------------------------------------------------------
    # 3. PyTorch Wind Generation Model (Epochs + Mini-Batches)
    # --------------------------------------------------------------------------
    print("======================================================================")
    print("Training PyTorch Wind Turbine Model (30 Epochs, Mini-Batch Size = 64)")
    print("======================================================================")
    wind_dl = PyTorchDeepLearningForecaster(WIND_FEATURES, 'wind_generation', hidden_dim=128, lr=0.001)
    wind_dl.train_with_epochs_and_minibatches(train_df, val_df, epochs=30, batch_size=64)
    wind_metrics = wind_dl.evaluate(test_df)
    print(f"--> Test Evaluation Metrics -> MAE: {wind_metrics['MAE']} kW | RMSE: {wind_metrics['RMSE']} kW | R2: {wind_metrics['R2']}\n")
    wind_dl.save(os.path.join(models_dir, "pytorch_wind_model.pt"))
    
    print("All PyTorch Deep Learning models trained successfully using Epochs & Mini-Batches!")

if __name__ == "__main__":
    train_all_pytorch()
