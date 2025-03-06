import torch
import pandas as pd
import numpy as np
import requests  # For fetching API data
from datetime import datetime

# Load models
odd_even_model = torch.load("models/odd_even_model.pth")
big_small_model = torch.load("models/big_small_model.pth")

# Define API URL
API_URL = "https://www.apigx.cn/history/code/pcdd.html"  # Replace with actual API

# Function to fetch real-time data
def fetch_latest_draws():
    """
    Fetches real-time draw data from the API.
    Returns a Pandas DataFrame formatted for prediction.
    """
    try:
        response = requests.get(API_URL)  # Fetch data
        response.raise_for_status()  # Raise an error if request fails
        
        data = response.json()  # Convert response to JSON
        draws = data.get("data", [])  # Extract relevant data

        if not draws:
            print("❌ No draws found in API response.")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(draws)

        # Adjust column names based on actual API response
        df = df.rename(columns={
            "issue": "draw_id",
            "code1": "draw_number1",
            "code2": "draw_number2",
            "code3": "draw_number3"
        })

        # Convert necessary columns to integers
        df[["draw_number1", "draw_number2", "draw_number3"]] = df[["draw_number1", "draw_number2", "draw_number3"]].astype(int)

        # Create target columns (these may need adjustments)
        df["odd_even"] = (df["draw_number1"] + df["draw_number2"] + df["draw_number3"]) % 2  # 0 = Even, 1 = Odd
        df["big_small"] = ((df["draw_number1"] + df["draw_number2"] + df["draw_number3"]) >= 14).astype(int)  # 0 = Small, 1 = Big
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return None

# Function to prepare features
def prepare_features(df, target_column):
    """
    Prepares input features by removing the target column.
    """
    X = df.drop(columns=[target_column])
    return torch.tensor(X.to_numpy(), dtype=torch.float32)

# Function to make predictions
def predict_latest_draws():
    """
    Fetches real-time data and makes predictions.
    """
    latest_df = fetch_latest_draws()  # Fetch data from API
    if latest_df is None or latest_df.empty:
        print("❌ No data available for prediction.")
        return None

    odd_even_target = "odd_even"
    big_small_target = "big_small"

    X_oe = prepare_features(latest_df, target_column=odd_even_target)
    X_bs = prepare_features(latest_df, target_column=big_small_target)

    # Predictions
    oe_logits = odd_even_model(X_oe)
    bs_logits = big_small_model(X_bs)
    
    oe_probs = torch.softmax(oe_logits, dim=1)
    bs_probs = torch.softmax(bs_logits, dim=1)
    
    oe_pred = torch.argmax(oe_probs, dim=1).numpy()
    bs_pred = torch.argmax(bs_probs, dim=1).numpy()
    
    # Confidence scores
    oe_confidence = (oe_probs.max(dim=1).values * 100).numpy()
    bs_confidence = (bs_probs.max(dim=1).values * 100).numpy()
    
    # Convert predictions to labels
    oe_labels = ["Even" if pred == 0 else "Odd" for pred in oe_pred]
    bs_labels = ["Small" if pred == 0 else "Big" for pred in bs_pred]
    
    # Create DataFrame
    results = pd.DataFrame({
        "Draw Time": [datetime.now().strftime("%H:%M:%S") for _ in range(len(latest_df))],
        "Odd/Even (Prediction & Accuracy %)": [f"Prediction: {oe_labels[i]} ({oe_confidence[i]:.0f}%)" for i in range(len(oe_labels))],
        "Big/Small (Prediction & Accuracy %)": [f"Prediction: {bs_labels[i]} ({bs_confidence[i]:.0f}%)" for i in range(len(bs_labels))],
        "Final Estimated Result": [f"Prediction: {oe_labels[i] if oe_confidence[i] > bs_confidence[i] else bs_labels[i]} ({max(oe_confidence[i], bs_confidence[i]):.0f}%)" for i in range(len(oe_labels))],
        "Recommended Bet": ["Yes" if max(oe_confidence[i], bs_confidence[i]) >= 70 else "No" for i in range(len(oe_labels))]
    })
    
    return results

# Run real-time prediction
if __name__ == "__main__":
    predictions = predict_latest_draws()
    if predictions is not None:
        print(predictions)
