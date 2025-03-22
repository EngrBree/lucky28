# scripts/predict_model.py

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
import random

# ✅ Import MLP from models folder
from utils.model_architecture import MLP

# ---------------------------------------------
# Load the scalers

odd_even_scaler = joblib.load("scripts/odd_even_scaler.pkl")
big_small_scaler = joblib.load("scripts/big_small_scaler.pkl")

# Define input dimensions per model
odd_even_input_dim = 4
big_small_input_dim = 5

# Load trained model weights using correct input_dim
odd_even_model = MLP(odd_even_input_dim)
odd_even_model.load_state_dict(torch.load("models/odd_even_model.pth"))
odd_even_model.eval()

big_small_model = MLP(big_small_input_dim)
big_small_model.load_state_dict(torch.load("models/big_small_model.pth"))
big_small_model.eval()

# Define model-specific features
ODD_EVEN_FEATURES = ['sum', 'sum_mod3', 'parity_last_digit', 'parity_sum_digits']
BIG_SMALL_FEATURES = ['sum', 'parity_sum_digits', 'rolling_sum_median', 'parity_last_digit', 'sum_mod3']
# API URL Template for fetching latest draw
API_URL_TEMPLATE = "https://www.apigx.cn/token/c5c808c4f81511ef9a5eafbf7b4e6e4c/code/jnd28/rows/{rows}.json"

def fetch_latest_draws(target_rows=100, batch_size=20):
    """Fetches the latest lottery draws in multiple API requests (max 20 draws per request)."""
    all_draws = []
    num_requests = target_rows // batch_size  # How many times to request

    for i in range(num_requests):
        try:
            print(f"🔄 Fetching batch {i+1}/{num_requests} ({batch_size} draws)...")
            response = requests.get(API_URL_TEMPLATE.format(rows=batch_size))
            response.raise_for_status()
            data = response.json()

            draws = data.get("data", [])
            if not draws:
                print("❌ No draws found in API response.")
                break  # Stop if no more draws are available

            all_draws.extend(draws)  # Add new batch to list
            time.sleep(1)  # Prevent hitting API limits too fast

        except Exception as e:
            print(f"⚠️ Error fetching data: {e}")
            break  # Stop on error

    if not all_draws:
        return None

    df = pd.DataFrame(all_draws)

    # Debugging output
    print("🔍 Raw API data =", all_draws[:2])  # Print first 2 draws for debugging

    # Handle draw ID
    if "expect" in df.columns:
        df["draw_id"] = df["expect"]
    else:
        df["draw_id"] = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")

    # Extract numbers from 'opencode'
    if "opencode" in df.columns:
        df[["draw_number1", "draw_number2", "draw_number3"]] = df["opencode"].str.split(",", expand=True).astype(int)
    else:
        print("❌ 'opencode' field missing in API response.")
        return None

    # Compute sum
    df["sum"] = df["draw_number1"] + df["draw_number2"] + df["draw_number3"]

    return df

def prepare_features(df):
    # Compute all needed features
    df["rolling_sum_mean"] = df["sum"].rolling(window=3, min_periods=1).mean()
    df["rolling_sum_median"] = df["sum"].rolling(3, min_periods=1).median()
    df["lag1_sum"] = df["sum"].shift(1)
    df["diff1"] = df["draw_number1"] - df["draw_number2"]
    df["diff2"] = df["draw_number2"] - df["draw_number3"]
    df["sum_mod3"] = df["sum"] % 3
    df["last_digit"] = df["sum"] % 10
    df["sum_digits"] = df["sum"].astype(str).apply(lambda x: sum(int(ch) for ch in x))
    df["parity_last_digit"] = df["last_digit"] % 2
    df["parity_sum_digits"] = df["sum_digits"] % 2

    # Fill any missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df

def predict():
    df = fetch_latest_draws()
    if df is None:
        print("❌ No data available for prediction.")
        return None

    # Add engineered features
    df = prepare_features(df)

    # Ensure all required features exist
    for feature in set(ODD_EVEN_FEATURES + BIG_SMALL_FEATURES):
        if feature not in df.columns:
            print(f"⚠️ Missing feature: {feature} — auto-filled with 0.")
            df[feature] = 0

    # ✅ Step 1: Print Raw Data for Debugging
    print("📌 Debug: Raw Data After Feature Engineering\n", df.head())

    # Scale & predict using correct feature sets
    try:
        print("📌 Debug: Odd-Even Feature Values Before Scaling\n", df[ODD_EVEN_FEATURES].head())
        print("📌 Debug: Big-Small Feature Values Before Scaling\n", df[BIG_SMALL_FEATURES].head())

        X_odd_even = odd_even_scaler.transform(df[ODD_EVEN_FEATURES])
        X_big_small = big_small_scaler.transform(df[BIG_SMALL_FEATURES])

        print("📌 Debug: Odd-Even Features After Scaling\n", X_odd_even[:5])
        print("📌 Debug: Big-Small Features After Scaling\n", X_big_small[:5])
    except Exception as e:
        print(f"⚠️ Scaling Error: {e}")
        return None

    # ✅ Step 2: Check Shape Before Model Prediction
    print(f"📌 Debug: Odd-Even Features Shape: {X_odd_even.shape}")
    print(f"📌 Debug: Big-Small Features Shape: {X_big_small.shape}")

    # Ensure correct tensor shapes
    try:
        X_odd_even_tensor = torch.tensor(X_odd_even, dtype=torch.float32)
        X_big_small_tensor = torch.tensor(X_big_small, dtype=torch.float32)
    except Exception as e:
        print(f"❌ Tensor Conversion Error: {e}")
        return None

    # ✅ Step 3: Check Model Output
    with torch.no_grad():
        try:
            odd_even_probs = torch.sigmoid(odd_even_model(X_odd_even_tensor)).numpy().flatten()
            big_small_probs = torch.sigmoid(big_small_model(X_big_small_tensor)).numpy().flatten()
            print(f"📌 Debug: Odd-Even Model Probabilities:\n{odd_even_probs[:5]}")
            print(f"📌 Debug: Big-Small Model Probabilities:\n{big_small_probs[:5]}")
        except Exception as e:
            print(f"❌ Model Prediction Error: {e}")
            return None

        predictions = []

        for i, row in df.iterrows():
            oe_prob = odd_even_probs[i]  
            bs_prob = big_small_probs[i]  

            print(f"📌 Debug: Row {i} - Odd/Even Probability = {oe_prob:.4f}, Big/Small Probability = {bs_prob:.4f}")

            # 🎯 Optimized Thresholds
            upper_threshold = 0.65
            lower_threshold = 0.45

            if bs_prob > upper_threshold:
                final_prediction = "Big"
                final_conf = min(bs_prob * 100, 95)
            elif bs_prob < lower_threshold:
                final_prediction = "Small"
                final_conf = min((1 - bs_prob) * 105, 95)  # 🔥 Scale Small higher
            else:
                if oe_prob > 0.52:  # 🔥 Trigger Odd earlier
                    final_prediction = "Odd"
                    final_conf = min(oe_prob * 100, 95)
                else:
                    final_prediction = "Even"
                    final_conf = min((1 - oe_prob) * 100, 95)

            # Normalize confidence between 50% - 95%
            final_conf = min(max(final_conf, 50), 95)

            print(f"📌 Debug: Row {i} - Final Prediction = {final_prediction} ({final_conf:.2f}%)")

            predictions.append({
                "Draw ID": row.get("draw_id", "Unknown"),
                "Final Prediction": final_prediction,
                "Confidence (%)": round(final_conf, 2),
            })

        # ✅ Save final predictions to CSV
        output_path = "data/final_predictions.csv"
        prediction_df = pd.DataFrame(predictions)
        
        try:
            prediction_df.to_csv(output_path, index=False)
            print(f"✅ Final predictions saved to: {os.path.abspath(output_path)}")
        except Exception as e:
            print(f"❌ Failed to save predictions: {e}")

        return predictions


if __name__ == "__main__":
    predict()