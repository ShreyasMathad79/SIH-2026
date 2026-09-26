"""
Renewable Energy Forecasting Module
Builds XGBoost / Random Forest models to forecast Solar and Wind generation in kW.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestRegressor

SOLAR_FEATURES = [
    'solar_irradiance', 'temperature', 'hour', 'month', 'day_of_year',
    'previous_hour_solar', 'lag_solar_1h', 'lag_temp_1h'
]

WIND_FEATURES = [
    'wind_speed', 'temperature', 'hour', 'month', 'day_of_year',
    'previous_hour_wind', 'lag_wind_1h', 'lag_temp_1h'
]

class RenewableForecaster:
    def __init__(self, resource_type='solar', model_type='xgboost'):
        """
        resource_type: 'solar' or 'wind'
        """
        self.resource_type = resource_type
        self.model_type = model_type
        self.features = SOLAR_FEATURES if resource_type == 'solar' else WIND_FEATURES
        self.target = 'solar_generation' if resource_type == 'solar' else 'wind_generation'
        
        if HAS_XGBOOST and model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
        else:
            self.model_type = 'random_forest'
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        self.is_trained = False

    def train(self, train_df, val_df=None):
        X_train = train_df[self.features]
        y_train = train_df[self.target]
        
        if self.model_type == 'xgboost' and val_df is not None:
            X_val = val_df[self.features]
            y_val = val_df[self.target]
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
            
        self.is_trained = True

    def evaluate(self, test_df):
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
            
        X_test = test_df[self.features]
        y_test = test_df[self.target]
        
        y_pred = np.clip(self.model.predict(X_test), 0, None)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-5))) * 100)
        
        return {
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'R2': round(r2, 4),
            'MAPE': round(mape, 2),
            'actuals': y_test.values,
            'predictions': np.round(y_pred, 2)
        }

    def predict(self, input_df):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        X_input = input_df[self.features]
        preds = np.clip(self.model.predict(X_input), 0, None)
        return np.round(preds, 2)

    def forecast_horizon(self, last_rows_df, future_weather_df, horizon_hours=24):
        """
        Forecasts solar or wind output for horizon hours.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting.")
            
        history = last_rows_df.copy().reset_index(drop=True)
        predictions = []
        
        for i in range(min(horizon_hours, len(future_weather_df))):
            current_row = future_weather_df.iloc[i].copy()
            
            if self.resource_type == 'solar':
                prev_gen = history['solar_generation'].iloc[-1] if len(history) > 0 else 0.0
                lag_sol = current_row.get('solar_irradiance', 0.0)
                lag_t = current_row.get('temperature', -15.0)
                feat_dict = {
                    'solar_irradiance': current_row['solar_irradiance'],
                    'temperature': current_row['temperature'],
                    'hour': current_row['hour'],
                    'month': current_row['month'],
                    'day_of_year': current_row.get('day_of_year', 1),
                    'previous_hour_solar': prev_gen,
                    'lag_solar_1h': lag_sol,
                    'lag_temp_1h': lag_t
                }
            else: # wind
                prev_gen = history['wind_generation'].iloc[-1] if len(history) > 0 else 0.0
                lag_w = current_row.get('wind_speed', 10.0)
                lag_t = current_row.get('temperature', -15.0)
                feat_dict = {
                    'wind_speed': current_row['wind_speed'],
                    'temperature': current_row['temperature'],
                    'hour': current_row['hour'],
                    'month': current_row['month'],
                    'day_of_year': current_row.get('day_of_year', 1),
                    'previous_hour_wind': prev_gen,
                    'lag_wind_1h': lag_w,
                    'lag_temp_1h': lag_t
                }
                
            single_X = pd.DataFrame([feat_dict])[self.features]
            pred_val = float(np.clip(self.model.predict(single_X)[0], 0, None))
            predictions.append(round(pred_val, 2))
            
            new_hist_entry = feat_dict.copy()
            new_hist_entry[self.target] = pred_val
            new_hist_entry['timestamp'] = current_row.get('timestamp', history['timestamp'].iloc[-1] + pd.Timedelta(hours=1))
            history = pd.concat([history, pd.DataFrame([new_hist_entry])], ignore_index=True)
            
        return predictions

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'resource_type': self.resource_type,
            'model_type': self.model_type,
            'features': self.features
        }, file_path)

    def load(self, file_path):
        data = joblib.load(file_path)
        self.model = data['model']
        self.resource_type = data['resource_type']
        self.model_type = data['model_type']
        self.features = data['features']
        self.target = 'solar_generation' if self.resource_type == 'solar' else 'wind_generation'
        self.is_trained = True
