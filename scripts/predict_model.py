#!/usr/bin/env python
import os
import sys
import torch
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

# Expected input dimensions based on training
INPUT_DIM = 14

# -------------------------------
# Import Model Architecture
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_architecture import MLP  # Ensure this matches training script definition

# -------------------------------
# Data Preparation Functions
# -------------------------------
def load_latest_draws(csv_path, num_rows=20):
    """Load last `num_rows` from test.csv."""
    df_full = pd.read_csv(csv_path)
    df_numeric = df_full.select_dtypes(include=[np.number]).copy()
    
    latest_full = df_full.tail(num_rows).reset_index(drop=True)
    latest_numeric = df_numeric.tail(num_rows).reset_index(drop=True)
    
    return latest_full, latest_numeric

def prepare_features(df_numeric, target_column):
    """Prepares features for prediction by scaling and converting to tensors."""
    X = df_numeric.drop(columns=[target_column], errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if X_scaled.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} input features, got {X_scaled.shape[1]}")
    
    return torch.tensor(X_scaled.astype(np.float32), dtype=torch.float32)

# -------------------------------
# Model Loading Functions
# -------------------------------
def load_model(model_path, input_dim, dropout_rate):
    """Loads and returns a trained model."""
    model = MLP(input_dim=input_dim, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# -------------------------------
# Prediction Functions
# -------------------------------
def make_predictions(model, data):
    """Runs the model and returns binary predictions with probabilities."""
    with torch.no_grad():
        outputs = model(data).view(-1)
        probabilities = torch.sigmoid(outputs).numpy().flatten()
        predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities

def predict_latest_draws(test_csv=TEST_FILE, num_latest=20, 
                         odd_even_target="odd_even_1", big_small_target="big_small_1"):
    """Runs predictions on the latest draws and saves results."""
    
    # Load latest draws
    full_df, numeric_df = load_latest_draws(test_csv, num_rows=num_latest)
    
    # Prepare features
    X_oe = prepare_features(numeric_df, target_column=odd_even_target)
    X_bs = prepare_features(numeric_df, target_column=big_small_target)
    
    # Load models
    odd_even_model = load_model(ODD_EVEN_MODEL_PATH, input_dim=INPUT_DIM, dropout_rate=0.6)
    big_small_model = load_model(BIG_SMALL_MODEL_PATH, input_dim=INPUT_DIM, dropout_rate=0.8)
    
    # Generate predictions
    oe_preds, prob_oe = make_predictions(odd_even_model, X_oe)
    bs_preds, prob_bs = make_predictions(big_small_model, X_bs)
    
    final_results = []
    
    for i in range(num_latest):
        p_oe = float(prob_oe[i])
        p_bs = float(prob_bs[i])
        conf_oe = abs(p_oe - 0.5)
        conf_bs = abs(p_bs - 0.5)

        # Map predictions to labels
        label_oe = "Odd" if p_oe >= 0.5 else "Even"
        label_bs = "Big" if p_bs >= 0.5 else "Small"

        # Final prediction decision logic
        if abs(p_oe - 0.5) < 0.02:
            final_pred = label_bs
            chosen_model = "Big/Small"
            final_conf = conf_bs
        else:
            if conf_oe >= conf_bs:
                final_pred = label_oe
                chosen_model = "Odd/Even"
                final_conf = conf_oe
            else:
                final_pred = label_bs
                chosen_model = "Big/Small"
                final_conf = conf_bs

        # **Improved Betting Recommendation Logic**
        if final_conf >= 0.15 or (p_oe >= 0.6 or p_bs >= 0.6):
            recommended = "Yes"
        elif (label_oe == label_bs) and (p_oe >= 0.55 and p_bs >= 0.55):
            recommended = "Yes"
        else:
            recommended = "No"

        result = {
            "Draw Index": i + 1,
            "sum": float(full_df.loc[i, "sum"]),
            "Odd/Even Prediction": label_oe,
            "Odd/Even Probability": float(round(p_oe, 4)),
            "Big/Small Prediction": label_bs,
            "Big/Small Probability": float(round(p_bs, 4)),
            "Final Estimated Result": final_pred,
            "Chosen Model": chosen_model,
            "Confidence": float(round(final_conf, 4)),
            "Recommended Bet": recommended,
            "Prediction Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        final_results.append(result)
    
    # Save results
    results_df = pd.DataFrame(final_results)
    print("Latest Predictions:")
    print(results_df)
    
    results_csv = os.path.join(BASE_DIR, "latest_predictions.csv")
    results_json = os.path.join(BASE_DIR, "latest_predictions.json")
    results_df.to_csv(results_csv, index=False)
    with open(results_json, "w") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Predictions saved to '{results_csv}' and '{results_json}'")
    
    return results_df

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    predict_latest_draws()
