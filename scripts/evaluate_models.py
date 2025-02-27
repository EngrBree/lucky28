import torch
import torch.nn as nn
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_architecture import MLP

def load_data_for_testing(file_path, target_column):
    """Loads test data from a CSV file and returns tensors."""
    data = pd.read_csv(file_path)

    # ✅ Drop non-numeric columns (e.g., datetime, text)
    data = data.select_dtypes(include=['number'])

    # ✅ Ensure target column is included
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in {file_path}. Available columns: {data.columns.tolist()}")

    X = data.drop(columns=[target_column])  # Features
    y = data[target_column].values.reshape(-1, 1)  # Target

    # ✅ Convert all NaN values to the column median
    X = X.fillna(X.median())

    # ✅ Convert to float to avoid object dtype issues
    X = X.astype(float)

    # ✅ Convert to PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y

def evaluate_model(model_path, test_file, target_column):
    """Loads a trained model and evaluates it using test data."""
    
    # ✅ Load test data
    test_X, test_y = load_data_for_testing(test_file, target_column)

    # ✅ Load trained model
    model = MLP(input_dim=test_X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # ✅ Compute predictions
    with torch.no_grad():
        outputs = model(test_X).view(-1)
        probabilities = torch.sigmoid(outputs).numpy()
        predictions = (probabilities >= 0.5).astype(int)

    # ✅ Compute accuracy
    accuracy = (predictions == test_y.numpy().flatten()).mean()
    print(f"✅ Model Accuracy on {target_column}: {accuracy:.4f}")

# ✅ Evaluate both models
evaluate_model("models/odd_even_model.pth", "data/test.csv", "odd_even_1")
evaluate_model("models/big_small_model.pth", "data/test.csv", "big_small_1")
