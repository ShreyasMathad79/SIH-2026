"""
Load Forecasting Module
Predicts future polar station electricity demand using XGBoost / Random Forest Regressor.
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

LOAD_FEATURES = [
    'temperature', 'wind_speed', 'solar_irradiance', 'sin_hour', 'cos_hour',
    'sin_month', 'cos_month', 'sin_day_of_year', 'cos_day_of_year',
    'day_of_week', 'previous_hour_load', 'previous_2_hour_load',
    'previous_24_hour_load', 'rolling_mean_load_6h', 'rolling_mean_load_24h',
    'personnel_count'
]

TARGET = 'historical_load'

class LoadForecaster:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        if HAS_XGBOOST and model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=7,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                random_state=42
            )
        else:
            self.model_type = 'random_forest'
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
        self.features = LOAD_FEATURES
        self.is_trained = False

    def train(self, train_df, val_df=None):
        X_train = train_df[self.features]
        y_train = train_df[TARGET]
        
        if self.model_type == 'xgboost' and val_df is not None:
            X_val = val_df[self.features]
            y_val = val_df[TARGET]
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
        y_test = test_df[TARGET]
        
        y_pred = self.model.predict(X_test)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        
        # MAPE with epsilon protection against zero
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
        preds = self.model.predict(X_input)
        return np.round(preds, 2)

    def forecast_horizon(self, last_rows_df, future_weather_df, horizon_hours=24):
        """
        Iterative recursive forecasting for 24h/48h/72h horizons using future weather estimates.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting.")
            
        history = last_rows_df.copy().reset_index(drop=True)
        predictions = []
        
        for i in range(min(horizon_hours, len(future_weather_df))):
            current_row = future_weather_df.iloc[i].copy()
            
            # Dynamic lag calculation from history
            prev_load_1 = history['historical_load'].iloc[-1]
            prev_load_2 = history['historical_load'].iloc[-2] if len(history) >= 2 else prev_load_1
            prev_load_24 = history['historical_load'].iloc[-24] if len(history) >= 24 else prev_load_1
            roll_6 = history['historical_load'].iloc[-6:].mean() if len(history) >= 6 else prev_load_1
            roll_24 = history['historical_load'].iloc[-24:].mean() if len(history) >= 24 else prev_load_1
            
            h_val = current_row['hour']
            m_val = current_row['month']
            doy_val = current_row.get('day_of_year', 1)
            
            feat_dict = {
                'temperature': current_row['temperature'],
                'wind_speed': current_row['wind_speed'],
                'solar_irradiance': current_row['solar_irradiance'],
                'sin_hour': np.sin(2 * np.pi * h_val / 24.0),
                'cos_hour': np.cos(2 * np.pi * h_val / 24.0),
                'sin_month': np.sin(2 * np.pi * m_val / 12.0),
                'cos_month': np.cos(2 * np.pi * m_val / 12.0),
                'sin_day_of_year': np.sin(2 * np.pi * doy_val / 365.25),
                'cos_day_of_year': np.cos(2 * np.pi * doy_val / 365.25),
                'day_of_week': current_row.get('day_of_week', 0),
                'previous_hour_load': prev_load_1,
                'previous_2_hour_load': prev_load_2,
                'previous_24_hour_load': prev_load_24,
                'rolling_mean_load_6h': roll_6,
                'rolling_mean_load_24h': roll_24,
                'personnel_count': current_row.get('personnel_count', 30)
            }
            
            single_X = pd.DataFrame([feat_dict])[self.features]
            pred_val = float(self.model.predict(single_X)[0])
            pred_val = max(15.0, pred_val) # Minimum physical load constraint
            predictions.append(round(pred_val, 2))
            
            # Append prediction to history dataframe for next iteration lag
            new_hist_entry = feat_dict.copy()
            new_hist_entry['historical_load'] = pred_val
            new_hist_entry['timestamp'] = current_row.get('timestamp', history['timestamp'].iloc[-1] + pd.Timedelta(hours=1))
            history = pd.concat([history, pd.DataFrame([new_hist_entry])], ignore_index=True)
            
        return predictions

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump({'model': self.model, 'model_type': self.model_type, 'features': self.features}, file_path)

    def load(self, file_path):
        data = joblib.load(file_path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.features = data['features']
        self.is_trained = True
