"""
PyTorch Deep Learning Forecasting Module
Uses PyTorch LSTM / Neural Networks trained with Epochs & Mini-batches for Polar Load & Renewable Forecasting.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. PYTORCH CUSTOM DATASET (MINI-BATCH PREPARATION)
# ==============================================================================
class PolarTimeseriesDataset(Dataset):
    """
    Custom PyTorch Dataset for mini-batching time-series features and target values.
    """
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==============================================================================
# 2. PYTORCH DEEP NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class PolarNeuralNetwork(nn.Module):
    """
    Deep Neural Network with Residual Connections for Time-Series Load & Renewable Prediction.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(PolarNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.2)
        
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h1 = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        h2 = self.dropout2(self.relu2(self.bn2(self.fc2(h1))))
        prediction = self.out(h2)
        return prediction

# ==============================================================================
# 3. DEEP LEARNING TRAINER CLASS (EPOCH & MINI-BATCH LOOP)
# ==============================================================================
class PyTorchDeepLearningForecaster:
    def __init__(self, feature_names, target_name, hidden_dim=128, lr=0.001):
        self.features = feature_names
        self.target = target_name
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.is_trained = False
        
    def train_with_epochs_and_minibatches(self, train_df, val_df=None, epochs=50, batch_size=64):
        """
        Trains the Deep Neural Network using explicit Epochs and Mini-Batches.
        """
        # Prepare inputs
        X_train_raw = train_df[self.features].values
        y_train_raw = train_df[self.target].values.reshape(-1, 1)
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train_raw)
        y_train_scaled = self.scaler_y.fit_transform(y_train_raw).flatten()
        
        # Create PyTorch Mini-Batch DataLoader
        train_dataset = PolarTimeseriesDataset(X_train_scaled, y_train_scaled)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_df is not None:
            X_val_scaled = self.scaler_X.transform(val_df[self.features].values)
            y_val_scaled = self.scaler_y.transform(val_df[self.target].values.reshape(-1, 1)).flatten()
            val_dataset = PolarTimeseriesDataset(X_val_scaled, y_val_scaled)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
        # Initialize Model, Loss Function & Adam Optimizer
        input_dim = len(self.features)
        self.model = PolarNeuralNetwork(input_dim=input_dim, hidden_dim=self.hidden_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        print(f"Starting Training: {epochs} Epochs | Mini-Batch Size: {batch_size} | Batches per Epoch: {len(train_loader)}")
        
        # ----------------------------------------------------------------------
        # MAIN TRAINING LOOP: EPOCHS & MINI-BATCHES
        # ----------------------------------------------------------------------
        history_log = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            total_batches = 0
            
            # Mini-Batch Loop
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()            # Clear previous gradients
                predictions = self.model(batch_X) # Forward pass
                loss = criterion(predictions, batch_y) # Compute loss
                loss.backward()                  # Backward pass (Backpropagation)
                optimizer.step()                 # Update weights
                
                running_loss += loss.item()
                total_batches += 1
                
            epoch_loss = running_loss / total_batches
            
            # Validation Loss check
            val_loss_str = ""
            if val_df is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_preds = self.model(val_X)
                        v_loss = criterion(val_preds, val_y)
                        val_loss += v_loss.item()
                        val_batches += 1
                val_loss_str = f" | Val Loss: {val_loss/val_batches:.4f}"
                
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] -> Mini-Batch Train Loss: {epoch_loss:.4f}{val_loss_str}")
                
            history_log.append({'epoch': epoch, 'train_loss': epoch_loss})
            
        self.is_trained = True
        return history_log

    def evaluate(self, test_df):
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
            
        self.model.eval()
        X_test_scaled = self.scaler_X.transform(test_df[self.features].values)
        y_test_actual = test_df[self.target].values
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            scaled_preds = self.model(X_tensor).numpy()
            y_pred = self.scaler_y.inverse_transform(scaled_preds).flatten()
            
        y_pred = np.clip(y_pred, 0, None)
        
        mae = float(mean_absolute_error(y_test_actual, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test_actual, y_pred)))
        r2 = float(r2_score(y_test_actual, y_pred))
        
        return {
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'R2': round(r2, 4),
            'actuals': y_test_actual,
            'predictions': np.round(y_pred, 2)
        }

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'features': self.features,
            'target': self.target,
            'hidden_dim': self.hidden_dim
        }, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.features = checkpoint['features']
        self.target = checkpoint['target']
        self.hidden_dim = checkpoint['hidden_dim']
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        
        self.model = PolarNeuralNetwork(input_dim=len(self.features), hidden_dim=self.hidden_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.is_trained = True
