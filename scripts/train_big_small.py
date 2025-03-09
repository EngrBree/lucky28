import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_architecture import MLP
from utils.focal_loss import FocalLoss

# ✅ Define Dataset Loader

def get_dataloaders(historical_train, historical_test, realtime_file, target_column, batch_size=32):
    train_data = pd.read_csv(historical_train)
    try:
        realtime_data = pd.read_csv(realtime_file)
        train_data = pd.concat([train_data, realtime_data], ignore_index=True)
    except FileNotFoundError:
        print("⚠️ No real-time data found. Training on historical data only.")

    test_data = pd.read_csv(historical_test)

    train_data = train_data.select_dtypes(include=['number'])
    test_data = test_data.select_dtypes(include=['number'])

    if target_column not in train_data.columns or target_column not in test_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    missing_cols = set(train_data.columns) - set(test_data.columns)
    for col in missing_cols:
        test_data[col] = 0
    test_data = test_data[train_data.columns]

    X_train = train_data.drop(columns=[target_column]).values
    y_train = train_data[target_column].values.reshape(-1, 1)
    X_test = test_data.drop(columns=[target_column]).values
    y_test = test_data[target_column].values.reshape(-1, 1)

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset, test_dataset

# ✅ Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"⛔ Early stopping triggered! Best validation loss: {self.best_loss:.4f}")
                return True
        return False

# ✅ Adaptive Retraining Function

def adaptive_retrain_loop(interval_minutes=60):
    while True:
        print("\n🔄 Starting adaptive retraining...")

        # ✅ Load Data
        train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
            historical_train="data/train.csv",
            historical_test="data/test.csv",
            realtime_file="data/real_time_preprocessed.csv",
            target_column="big_small_1",
            batch_size=8
        )

        input_dim = train_dataset.tensors[0].shape[1]
        model = MLP(input_dim=input_dim, dropout_rate=0.6)
        criterion = FocalLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-3)

        scheduler1 = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=50)
        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)

        early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
        best_loss = float('inf')

        for epoch in range(50):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).view(-1)
                loss = criterion(outputs, y_batch.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch).view(-1)
                    val_loss += criterion(outputs, y_batch.view(-1)).item()
            val_loss /= len(test_loader)

            print(f"Epoch {epoch+1} | Training Loss: {epoch_loss / len(train_loader):.4f} | Validation Loss: {val_loss:.4f}")
            scheduler1.step()
            scheduler2.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "models/best_big_small_model.pth")
                print("✅ Best model saved!")

            if early_stopping(val_loss):
                break

        torch.save(model.state_dict(), "models/big_small_model.pth")
        print("✅ Final Model Saved!")

        # ✅ Evaluation
        def evaluate_model(model, test_loader):
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch).view(-1)
                    predictions = (torch.sigmoid(outputs) >= 0.5).float()
                    correct += (predictions == y_batch.view(-1)).sum().item()
                    total += y_batch.size(0)
            print(f"🎯 Test Accuracy: {correct/total:.4f}")

        evaluate_model(model, test_loader)

        print(f"⏳ Waiting {interval_minutes} minutes for next retraining...")
        time.sleep(interval_minutes * 60)

# ✅ Start Adaptive Loop
if __name__ == "__main__":
    adaptive_retrain_loop(interval_minutes=60)
