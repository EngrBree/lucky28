# scripts/predict_model.py

import sys
import os
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

# ---------------------------------------------
# API URL for fetching latest draw
API_URL = "https://www.apigx.cn/token/c5c808c4f81511ef9a5eafbf7b4e6e4c/code/jnd28/rows/20.json"

def fetch_latest_draws():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
        draws = data.get("data", [])

        if not draws:
            print("❌ No draws found in API response.")
            return None

        df = pd.DataFrame(draws)

        print("🔍 Raw API data =", data)
        print("🔍 Sample draw =", draws[0])

        if "expect" in df.columns:
            df["draw_id"] = df["expect"]
        else:
            df["draw_id"] = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")

        if "opencode" in df.columns:
            df[["draw_number1", "draw_number2", "draw_number3"]] = df["opencode"].str.split(",", expand=True).astype(int)
        else:
            print("❌ 'opencode' field missing in API response.")
            return None

        df["sum"] = df["draw_number1"] + df["draw_number2"] + df["draw_number3"]

        return df

    except Exception as e:
        print(f"⚠️ Error fetching data: {e}")
        return None

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

    X_odd_even_tensor = torch.tensor(X_odd_even, dtype=torch.float32)
    X_big_small_tensor = torch.tensor(X_big_small, dtype=torch.float32)

    with torch.no_grad():
        odd_even_probs = torch.sigmoid(odd_even_model(X_odd_even_tensor)).numpy().flatten()
        big_small_probs = torch.sigmoid(big_small_model(X_big_small_tensor)).numpy().flatten()

    predictions = []

    for i, row in df.iterrows():
        oe_prob = odd_even_probs[i]   # Odd-Even model probability
        bs_prob = big_small_probs[i]    # Big-Small model probability

        # Use big-small prediction if the model is confident enough:
        if bs_prob > 0.6:
            final_prediction = "big"
            final_conf = bs_prob * 100
        elif bs_prob < 0.4:
            final_prediction = "small"
            final_conf = (1 - bs_prob) * 100
        else:
            # Otherwise, use the odd-even model's prediction:
            if oe_prob > 0.5:
                final_prediction = "Odd"
                final_conf = oe_prob * 100
            else:
                final_prediction = "Even"
                final_conf = (1 - oe_prob) * 100

        predictions.append({
            "Draw ID": row.get("draw_id", "Unknown"),
            "Prediction": final_prediction,
            "Accuracy (%)": round(final_conf, 2)
        })

    # Save predictions to CSV
    prediction_df = pd.DataFrame(predictions)
    output_path = "data/latest_prediction_with_percentages.csv"
    try:
        prediction_df.to_csv(output_path, index=False)
        print(f"✅ Prediction results saved to: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"❌ Failed to save predictions: {e}")

    return predictions


if __name__ == "__main__":
    predict()