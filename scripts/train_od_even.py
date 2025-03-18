# ✅ Corrected train_big_small.py with Integrated Real-Time Evaluation & F1 Metrics (No Ablation Testing)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_architecture import MLP

REQUIRED_FEATURES = ['sum', 'sum_mod3', 'parity_last_digit', 'parity_sum_digits']

# ✅ Define Dataset Loader
def get_dataloaders(historical_train, historical_test, realtime_file, target_column, feature_subset=None, batch_size=32):
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

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column].values.reshape(-1, 1)
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column].values.reshape(-1, 1)

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    if feature_subset is None:
        feature_subset = REQUIRED_FEATURES

    scaler = StandardScaler()
    X_train_df = pd.DataFrame(X_train, columns=train_data.drop(columns=[target_column]).columns)
    X_test_df = pd.DataFrame(X_test, columns=test_data.drop(columns=[target_column]).columns)

    X_train = scaler.fit_transform(X_train_df[feature_subset])
    X_test = scaler.transform(X_test_df[feature_subset])
    joblib.dump(scaler, "scripts/odd_even_scaler.pkl")

    print("📊 Class Distribution (Real-Time + Historical):")
    print(pd.Series(y_train.flatten()).value_counts(normalize=True))

    y_train_flat = y_train.flatten()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flat), y=y_train_flat)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    global weighted_loss_fn
    weighted_loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    return train_loader, test_loader, train_dataset, test_dataset, y_test

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

# ✅ Evaluation Function
def evaluate_model(model, test_loader, y_test):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).view(-1)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(y_batch.view(-1).cpu().numpy())
    print(f"🎯 Test Accuracy: {(np.array(predictions) == np.array(true_labels)).mean():.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(true_labels, predictions))
    print("🧩 Confusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    return (np.array(predictions) == np.array(true_labels)).mean()

# ✅ Evaluate on Real-Time Window
def evaluate_on_realtime_window(model, scaler, window_size=200):
    try:
        df = pd.read_csv("data/real_time_preprocessed.csv")
        if len(df) < window_size:
            print(f"⚠️ Not enough real-time data for evaluation window (only {len(df)} rows). Skipping...")
            return
        df = df.tail(window_size)
        if "odd_even_1" not in df.columns:
            print("⚠️ Target column 'odd_even_1' not found in real-time data. Skipping evaluation.")
            return

        X_eval = df[REQUIRED_FEATURES].values
        y_eval = df["odd_even_1"].values

        X_eval = scaler.transform(X_eval)
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
        y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            preds = model(X_eval_tensor).view(-1)
            predictions = (torch.sigmoid(preds) >= 0.5).float()
            accuracy = (predictions == y_eval_tensor).float().mean().item()
            print(f"🧪 Real-Time Evaluation (last {window_size} rows): Accuracy = {accuracy:.4f}")
    except FileNotFoundError:
        print("⚠️ Real-time data not found. Skipping real-time evaluation.")

# ✅ Adaptive Retraining Loop
def adaptive_retrain_loop(interval_minutes=60):
    train_loader, test_loader, train_dataset, test_dataset, y_test = get_dataloaders(
        historical_train="data/train.csv",
        historical_test="data/test.csv",
        realtime_file="data/real_time_preprocessed.csv",
        target_column="odd_even_1",
        feature_subset=REQUIRED_FEATURES,
        batch_size=16
    )

    input_dim = train_dataset.tensors[0].shape[1]
    model = MLP(input_dim=input_dim, dropout_rate=0.3)
    criterion = weighted_loss_fn
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-3)

    scheduler1 = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=50)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)

    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    best_loss = float('inf')

    for epoch in range(50):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == y_batch.view(-1)).sum().item()
            train_total += y_batch.size(0)

        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch).view(-1)
                val_loss += criterion(outputs, y_batch.view(-1)).item()
        val_loss /= len(test_loader)
        train_acc = train_correct / train_total

        print(f"Epoch {epoch+1} | Training Loss: {epoch_loss / len(train_loader):.4f} | Validation Loss: {val_loss:.4f} | Training Accuracy: {train_acc:.4f}")

        scheduler1.step()
        scheduler2.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print("✅ Best model saved!")

        if early_stopping(val_loss):
            break

    torch.save(model.state_dict(), "models/odd_even_model.pth")
    print("✅ Final Model Saved!")
    evaluate_model(model, test_loader, y_test)
    evaluate_on_realtime_window(model, scaler=joblib.load("scripts/odd_even_scaler.pkl"))

if __name__ == "__main__":
    adaptive_retrain_loop(interval_minutes=60)
