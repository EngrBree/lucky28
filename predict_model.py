#!/usr/bin/env python
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Dynamic Paths Setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
ODD_EVEN_MODEL_PATH = os.path.join(MODEL_DIR, "odd_even_model.pth")
BIG_SMALL_MODEL_PATH = os.path.join(MODEL_DIR, "big_small_model.pth")

# Set expected input dimension based on training (dropping target gives 9 features)
INPUT_DIM = 9

# -------------------------------
# Import Model Architecture
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_architecture import MLP  # This should exactly match your training definition

# -------------------------------
# Data Preparation Functions
# -------------------------------
def load_latest_draws(csv_path, num_rows=20):
    """
    Load the last `num_rows` from test.csv (assumed to be in chronological order).
    """
    df = pd.read_csv(csv_path)
    latest_df = df.tail(num_rows).reset_index(drop=True)
    return latest_df

def prepare_features(df, target_column):
    """
    Drop the target column from the DataFrame and return a scaled tensor.
    """
    X = df.drop(columns=[target_column])
    # Ensure all values are numeric and fill NaNs with 0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if X_scaled.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} input features, but got {X_scaled.shape[1]}")
    return torch.tensor(X_scaled.astype(np.float32), dtype=torch.float32)

# -------------------------------
# Model Loading Functions
# -------------------------------
def load_odd_even_model(input_dim):
    model = MLP(input_dim=input_dim, dropout_rate=0.6)
    model.load_state_dict(torch.load(ODD_EVEN_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

def load_big_small_model(input_dim):
    model = MLP(input_dim=input_dim, dropout_rate=0.8)
    model.load_state_dict(torch.load(BIG_SMALL_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# -------------------------------
# Prediction Functions
# -------------------------------
def make_predictions(model, data):
    """
    Run the model on the provided data and return predictions and probabilities.
    """
    with torch.no_grad():
        outputs = model(data).view(-1)
        probabilities = torch.sigmoid(outputs).numpy().flatten()
        predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities

def predict_latest_draws(test_csv=TEST_FILE, num_latest=20,
                         odd_even_target="odd_even_1", big_small_target="big_small_1",
                         confidence_threshold=0.6):
    # Load the latest draws from test CSV
    latest_df = load_latest_draws(test_csv, num_rows=num_latest)
    
    # Prepare feature inputs for both models
    X_oe = prepare_features(latest_df, target_column=odd_even_target)
    X_bs = prepare_features(latest_df, target_column=big_small_target)
    
    # Verify dimensions
    if X_oe.shape[1] != INPUT_DIM or X_bs.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected input dimension {INPUT_DIM}, but got {X_oe.shape[1]} and {X_bs.shape[1]}")
    
    # Load trained models
    odd_even_model = load_odd_even_model(input_dim=INPUT_DIM)
    big_small_model = load_big_small_model(input_dim=INPUT_DIM)
    
    # Get predictions and probabilities
    _, prob_oe = make_predictions(odd_even_model, X_oe)
    _, prob_bs = make_predictions(big_small_model, X_bs)
    
    final_results = []
    for i in range(num_latest):
        p_oe_val = float(prob_oe[i])
        p_bs_val = float(prob_bs[i])
        
        # Simpler confidence calculation: use the absolute probability difference from 0.5
        conf_oe = abs(p_oe_val - 0.5) * 2  # Confidence scaled between 0 and 1
        conf_bs = abs(p_bs_val - 0.5) * 2
        
        # Map binary predictions to labels
        label_oe = "Odd" if p_oe_val >= 0.5 else "Even"
        label_bs = "Big" if p_bs_val >= 0.5 else "Small"
        
        # Refined decision logic:
        if conf_oe > conf_bs:
            final_pred = label_oe
            chosen_model = "Odd/Even"
            final_conf = conf_oe
        else:
            final_pred = label_bs
            chosen_model = "Big/Small"
            final_conf = conf_bs
        
        # Recommend bet if final confidence exceeds threshold
        recommended = "Yes" if final_conf >= confidence_threshold else "No"
        
        result = {
            "Draw Index": i + 1,
            "sum": float(latest_df.loc[i, "sum"]),
            "Odd/Even Prediction": label_oe,
            "Odd/Even Probability": round(p_oe_val, 4),
            "Big/Small Prediction": label_bs,
            "Big/Small Probability": round(p_bs_val, 4),
            "Final Estimated Result": final_pred,
            "Chosen Model": chosen_model,
            "Confidence": round(final_conf, 4),
            "Recommended Bet": recommended,
            "Prediction Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        final_results.append(result)
    
    results_df = pd.DataFrame(final_results)
    print("Latest Predictions:")
    print(results_df)
    
    # Save results to CSV and JSON in the project root
    results_csv = os.path.join(BASE_DIR, "latest_predictions.csv")
    results_json = os.path.join(BASE_DIR, "latest_predictions.json")
    results_df.to_csv(results_csv, index=False)
    with open(results_json, "w") as f:
        json.dump(final_results, f, indent=4, default=lambda o: float(o) if isinstance(o, np.float32) else o)
    print(f"Predictions saved to '{results_csv}' and '{results_json}'")
    
    return results_df

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    predict_latest_draws()
