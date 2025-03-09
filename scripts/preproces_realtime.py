import pandas as pd
import numpy as np
import requests
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ✅ API URL for real-time data
API_URL = "https://www.apigx.cn/token/c5c808c4f81511ef9a5eafbf7b4e6e4c/code/jnd28/rows/10.json"
REALTIME_CSV = "data/real_time_preprocessed.csv"
FETCH_INTERVAL = 600  # Fetch data every 60 seconds

def fetch_realtime_draw():
    """Fetch real-time draw data from the API and return as a DataFrame."""
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            print("❌ No new data available.")
            return None

        df = pd.DataFrame(data["data"])
        df["num1"], df["num2"], df["num3"] = zip(*df["opencode"].apply(lambda x: map(int, x.split(","))))
        df["sum"] = df["num1"] + df["num2"] + df["num3"]
        df["odd_even"] = df["sum"] % 2  # 0 = Even, 1 = Odd
        df["big_small"] = (df["sum"] >= 14).astype(int)  # 0 = Small, 1 = Big

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return None

def preprocess_and_update():
    """Fetch and preprocess real-time data, then update the dataset."""
    df = fetch_realtime_draw()
    if df is None:
        return

    # ✅ Compute Rolling & Lag Features
    df["rolling_sum_mean"] = df["sum"].rolling(window=3, min_periods=1).mean()
    df["rolling_sum_median"] = df["sum"].rolling(3, min_periods=1).median()
    df["lag1_sum"] = df["sum"].shift(1)
    df["lag1_odd_even"] = df["odd_even"].shift(1)
    df["lag1_big_small"] = df["big_small"].shift(1)

    # ✅ Differences between consecutive numbers
    df["diff1"] = df["num1"] - df["num2"]
    df["diff2"] = df["num2"] - df["num3"]
    df["sum_mod3"] = df["sum"] % 3  

    # ✅ Handle missing values
    missing_cols = ["rolling_sum_mean", "rolling_sum_median", "lag1_sum", "lag1_odd_even", "lag1_big_small"]
    df[missing_cols] = df[missing_cols].fillna(df[missing_cols].median())

    # ✅ Scale numerical features
    numerical_features = ["sum", "rolling_sum_mean", "lag1_sum", "diff1", "diff2", "rolling_sum_median", "sum_mod3"]
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # ✅ One-Hot Encoding categorical features
    categorical_features = ["odd_even", "big_small", "lag1_odd_even", "lag1_big_small"]
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_cat = encoder.fit_transform(df[categorical_features])
    
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))
    df = pd.concat([df.drop(columns=categorical_features).reset_index(drop=True), encoded_cat_df], axis=1)

    # ✅ Append new data to the existing dataset
    try:
        existing_data = pd.read_csv(REALTIME_CSV)
        df = pd.concat([existing_data, df], ignore_index=True)
    except FileNotFoundError:
        print("ℹ️ Creating new real-time dataset.")

    # ✅ Save updated real-time dataset
    df.to_csv(REALTIME_CSV, index=False)
    print(f"✅ Real-time data updated successfully in {REALTIME_CSV}")

def start_continuous_fetching():
    """Continuously fetch and update real-time data."""
    while True:
        print("\n🔄 Fetching real-time data...")
        preprocess_and_update()
        print(f"⏳ Waiting {FETCH_INTERVAL} seconds before next fetch...")
        time.sleep(FETCH_INTERVAL)  # Wait before next fetch

if __name__ == "__main__":
    start_continuous_fetching()
