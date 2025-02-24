#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import ast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold

# Data Processing Functions #

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['code'] = df['code'].apply(ast.literal_eval)
    
    # Extract individual numbers
    df['num1'] = df['code'].apply(lambda x: int(x[0]))
    df['num2'] = df['code'].apply(lambda x: int(x[1]))
    df['num3'] = df['code'].apply(lambda x: int(x[2]))
    
    # Compute sum and derived features
    df['sum'] = df['num1'] + df['num2'] + df['num3']
    df['odd_even'] = df['sum'] % 2  # 0 = Even, 1 = Odd
    df['big_small'] = (df['sum'] >= 14).astype(int)  # 0 = Small, 1 = Big

    # Rolling mean of sum (captures trends)
    df['rolling_sum_mean'] = df['sum'].rolling(window=3, min_periods=1).mean()
    
    # Lag features (previous round’s sum and classifications)
    df['lag1_sum'] = df['sum'].shift(1)
    df['lag1_odd_even'] = df['odd_even'].shift(1)
    df['lag1_big_small'] = df['big_small'].shift(1)

    # Difference between consecutive numbers (new features)
    df['diff1'] = df['num1'] - df['num2']
    df['diff2'] = df['num2'] - df['num3']

    # Handle missing values (for lagged and rolling features)
    missing_cols = ['rolling_sum_mean', 'lag1_sum', 'lag1_odd_even', 'lag1_big_small']
    df[missing_cols] = df[missing_cols].fillna(df[missing_cols].median())

    return df[['num1', 'num2', 'num3', 'sum', 'rolling_sum_mean', 'lag1_sum',
               'odd_even', 'big_small', 'lag1_odd_even', 'lag1_big_small', 'diff1', 'diff2']]

# Load data
data_file = "newlucky28.csv"
df = load_data(data_file)

# Split data (80% train, 20% test; ensure at least 100 draws in test set, chronological order)
test_size = max(0.2, 100 / len(df))
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=False)

# Fit scaler only on train set
# Update the numerical features list
numerical_features = ['sum', 'rolling_sum_mean', 'lag1_sum', 'diff1', 'diff2']


scaler = StandardScaler()
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])  # Transform only

# One-Hot Encode categorical features
categorical_features = ['odd_even', 'big_small', 'lag1_odd_even', 'lag1_big_small']
combined_cat = pd.concat([train_df[categorical_features], test_df[categorical_features]], axis=0)
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cat = encoder.fit_transform(combined_cat)
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))
train_encoded_df = encoded_cat_df.iloc[:len(train_df)].reset_index(drop=True)
test_encoded_df = encoded_cat_df.iloc[len(train_df):].reset_index(drop=True)
train_df = pd.concat([train_df.drop(columns=categorical_features).reset_index(drop=True), train_encoded_df], axis=1)
test_df = pd.concat([test_df.drop(columns=categorical_features).reset_index(drop=True), test_encoded_df], axis=1)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)


train_df = pd.read_csv("train.csv")
print(train_df["big_small_1"].value_counts())  # See if one class dominates


# PyTorch Dataset & DataLoader

class LotteryDataset(TensorDataset):
    def __init__(self, file_path, target_column):
        data = pd.read_csv(file_path)
        X = data.drop(columns=[target_column]).values
        y = data[target_column].values.reshape(-1, 1)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        super().__init__(X, y)

def get_dataloaders(train_file, test_file, target_column, batch_size=64):
    train_dataset = LotteryDataset(train_file, target_column)
    test_dataset = LotteryDataset(test_file, target_column)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

print("Columns in train.csv:", pd.read_csv("train.csv").columns.tolist())


# Model Definition

class MLP(nn.Module):
    def __init__(self, input_dim=11): 
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 1)  
        self.dropout = nn.Dropout(0.5) 
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

# Training & Evaluation Functions

def load_data_for_training(train_path, test_path, target_column, val_split=0.1):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop(columns=[target_column]).values
    y_train = train[target_column].values.reshape(-1, 1)
    X_test = test.drop(columns=[target_column]).values
    y_test = test[target_column].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    val_size = int(len(X_train) * val_split)
    train_size = len(X_train) - val_size
    X_train, X_val = torch.tensor(X_train[:train_size], dtype=torch.float32), torch.tensor(X_train[train_size:], dtype=torch.float32)
    y_train, y_val = torch.tensor(y_train[:train_size], dtype=torch.float32), torch.tensor(y_train[train_size:], dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataloader_from_tensors(X, y, batch_size):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(train_path, test_path, target_column, model_path, model, epochs=50, batch_size=64, lr=0.001, weight_decay=1e-3, patience=7):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_for_training(train_path, test_path, target_column)
    train_loader = create_dataloader_from_tensors(X_train, y_train, batch_size)
    val_loader = create_dataloader_from_tensors(X_val, y_val, batch_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None  # Store the best model state
    
    device = torch.device("cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_loss += criterion(model(X_val_batch).view(-1), y_val_batch.view(-1)).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Store best model
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. Restoring best model.")
                model.load_state_dict(best_model_state)  # Restore best model
                break

    return model


def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch.view(-1))
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == y_batch.view(-1)).sum().item()
            total += y_batch.size(0)
    total_loss /= len(test_loader)
    accuracy = correct / float(total)
    print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {accuracy:.4f}")

def load_and_evaluate(model_path, test_path, target_column, batch_size=64):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_for_training("train.csv", test_path, target_column)
    test_loader = create_dataloader_from_tensors(X_test, y_test, batch_size)
    input_dim = X_test.shape[1]
    model = MLP(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    evaluate_model(model, test_loader)

def cross_validate_big_small(train_file, target_column, n_splits=5, epochs=50, batch_size=64, lr=0.001, weight_decay=5e-4, patience=5):
    # Load the full training set
    data = pd.read_csv(train_file)
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values.reshape(-1, 1)
    
    # Scale features using the entire training set
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Setup KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_losses = []
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]
        
        # Create DataLoaders for this fold
        train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_fold, y_val_fold), batch_size=batch_size, shuffle=False)
        
        # Initialize model (for Big/Small task, input_dim should be 9)
        model = MLP(input_dim=X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        best_loss, patience_counter = float('inf'), 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).view(-1)
                loss = criterion(outputs, y_batch.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # Evaluate on validation set
            model.eval()
            val_loss = 0
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch).view(-1)
                    loss = criterion(outputs, y_batch.view(-1))
                    val_loss += loss.item()
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (preds == y_batch.view(-1)).sum().item()
                    total += y_batch.size(0)
            val_loss /= len(val_loader)
            accuracy = correct / total
            print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
            if val_loss < best_loss:
                best_loss, patience_counter = val_loss, 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping for this fold.")
                    break
        fold_losses.append(best_loss)
        fold_accuracies.append(accuracy)
    
    avg_loss = sum(fold_losses)/len(fold_losses)
    avg_accuracy = sum(fold_accuracies)/len(fold_accuracies)
    print(f"\nCross-Validation Results: Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

# Training Functions

def train_odd_even():
    model = MLP(input_dim=11)
    train_model("train.csv", "test.csv", "odd_even_1", "model_odd_even_trained.pth", model, epochs=50, batch_size=64, lr=0.001, weight_decay=5e-4, patience=7)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_for_training("train.csv", "test.csv", "odd_even_1")
    test_loader = create_dataloader_from_tensors(X_test, y_test, 64)
    evaluate_model(model, test_loader)

def train_big_small():
    model = MLP(input_dim=11)
    train_model("train.csv", "test.csv", "big_small_1", "model_big_small_trained.pth", model, epochs=50, batch_size=64, lr=0.001, weight_decay=5e-4, patience=7)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_for_training("train.csv", "test.csv", "big_small_1")
    test_loader = create_dataloader_from_tensors(X_test, y_test, 64)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    train_odd_even()
    train_big_small()


# Final Prediction Function

def final_prediction(input_vector, odd_even_model_path, big_small_model_path, input_dim=11):
    model_oe = MLP(input_dim=input_dim)
    model_bs = MLP(input_dim=input_dim)
    model_oe.load_state_dict(torch.load(odd_even_model_path, map_location=torch.device('cpu')))
    model_bs.load_state_dict(torch.load(big_small_model_path, map_location=torch.device('cpu')))
    model_oe.eval()
    model_bs.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).view(1, -1)
        output_oe = torch.sigmoid(model_oe(input_tensor)).item()
        output_bs = torch.sigmoid(model_bs(input_tensor)).item()
    conf_oe = abs(output_oe - 0.5)
    conf_bs = abs(output_bs - 0.5)
    if conf_oe > conf_bs:
        prediction = 'Even' if output_oe >= 0.5 else 'Odd'
        print(f"Final Prediction: {prediction} (Odd/Even Model, confidence {conf_oe:.2f})")
    else:
        prediction = 'Big' if output_bs >= 0.5 else 'Small'
        print(f"Final Prediction: {prediction} (Big/Small Model, confidence {conf_bs:.2f})")
    return prediction

