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
odd_even_features = ['sum', 'sum_mod3', 'parity_last_digit', 'parity_sum_digits']
big_small_features = ['sum', 'last_digit', 'sum_digits', 'sum_mod3','rolling_sum_median']

# ---------------------------------------------
# API URL for fetching latest draw
API_URL = "https://www.apigx.cn/token/81dc4724ff3111ef8d0c01bd2828daeb/code/jnd28/rows/20.json"

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
    for feature in set(odd_even_features + big_small_features):
        if feature not in df.columns:
            print(f"⚠️ Missing feature: {feature} — auto-filled with 0.")
            df[feature] = 0

    # Scale & predict using correct feature sets
    try:
        print("📌 Features in df before odd-even scaling:", df[odd_even_features].columns.tolist())
        print("📌 Features used to train odd-even scaler:", odd_even_scaler.feature_names_in_)

        X_odd_even = odd_even_scaler.transform(df[odd_even_features])

        print("📌 Features in df before big-small scaling:", df[big_small_features].columns.tolist())
        print("📌 Features used to train big-small scaler:", big_small_scaler.feature_names_in_)

        X_big_small = big_small_scaler.transform(df[big_small_features])
    except Exception as e:
        print(f"⚠️ Feature scaling error: {e}")
        return None

    X_odd_even_tensor = torch.tensor(X_odd_even, dtype=torch.float32)
    X_big_small_tensor = torch.tensor(X_big_small, dtype=torch.float32)

    with torch.no_grad():
        odd_even_probs = torch.sigmoid(odd_even_model(X_odd_even_tensor)).numpy().flatten()
        big_small_probs = torch.sigmoid(big_small_model(X_big_small_tensor)).numpy().flatten()

    # Add class predictions
    df["pred_odd_even"] = (odd_even_probs >= 0.5).astype(int)
    df["pred_big_small"] = (big_small_probs >= 0.5).astype(int)

    # Save to CSV
       # 🎯 Format and Save Prediction Summary
    prediction_results = []

    for index, row in df.iterrows():
        draw_time = row.get("opentime", row.get("draw_id", "Unknown"))

        odd_even_pred = "Odd" if row["pred_odd_even"] == 1 else "Even"
        odd_even_prob = odd_even_probs[index]
        big_small_pred = "Big" if row["pred_big_small"] == 1 else "Small"
        big_small_prob = big_small_probs[index]

        if odd_even_prob >= big_small_prob:
            final_result = f"{odd_even_pred} ({int(odd_even_prob * 100)}%)"
        else:
            final_result = f"{big_small_pred} ({int(big_small_prob * 100)}%)"

        recommend_bet = "Yes" if odd_even_prob > 0.7 and big_small_prob > 0.7 else "No"

        prediction_results.append({
            "Draw Time": draw_time,
            "Odd/Even (Prediction & Accuracy %)": f"Prediction: {odd_even_pred} ({int(odd_even_prob * 100)}%)",
            "Big/Small (Prediction & Accuracy %)": f"Prediction: {big_small_pred} ({int(big_small_prob * 100)}%)",
            "Final Estimated Result": final_result,
            "Recommended Bet": recommend_bet
        })

    prediction_df = pd.DataFrame(prediction_results)

    try:
        output_path = "data/latest_prediction.csv"
        prediction_df.to_csv(output_path, index=False)
        print(f"✅ Prediction results saved to: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"❌ Failed to save formatted predictions: {e}")


    except Exception as e:
        print(f"❌ Failed to save predictions: {e}")

    return df

if __name__ == "__main__":
    predict()
