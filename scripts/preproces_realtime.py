# ✅ Updated preproces_realtime.py — Parity Features Only (No Target Columns)
import pandas as pd
import numpy as np
import requests
import time
import joblib

REALTIME_CSV = "data/real_time_preprocessed.csv"
API_URL = "https://www.apigx.cn/token/c5c808c4f81511ef9a5eafbf7b4e6e4c/code/jnd28/rows/10.json"
FETCH_INTERVAL = 600

def fetch_realtime_draw():
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
        df["odd_even"] = df["sum"] % 2
        df["big_small"] = (df["sum"] >= 14).astype(int)
        return df
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return None

def preprocess_and_update():
    df = fetch_realtime_draw()
    if df is None:
        return

    df["rolling_sum_mean"] = df["sum"].rolling(window=3, min_periods=1).mean()
    df["rolling_sum_median"] = df["sum"].rolling(3, min_periods=1).median()
    df["lag1_sum"] = df["sum"].shift(1)
    df["lag1_odd_even"] = df["odd_even"].shift(1)
    df["lag1_big_small"] = df["big_small"].shift(1)
    df["diff1"] = df["num1"] - df["num2"]
    df["diff2"] = df["num2"] - df["num3"]
    df["sum_mod3"] = df["sum"] % 3

    # ✅ Add parity features only
    df["last_digit"] = df["sum"] % 10
    df["sum_digits"] = df["sum"].astype(str).apply(lambda x: sum(int(ch) for ch in x))
    df["parity_last_digit"] = df["last_digit"] % 2
    df["parity_sum_digits"] = df["sum_digits"] % 2

    missing_cols = ["rolling_sum_mean", "rolling_sum_median", "lag1_sum", "lag1_odd_even", "lag1_big_small"]
    df[missing_cols] = df[missing_cols].fillna(df[missing_cols].median())

    scaler = joblib.load("scripts/scaler.pkl")
    numerical_features = [
        "sum", "rolling_sum_mean", "lag1_sum", "diff1", "diff2",
        "rolling_sum_median", "sum_mod3", "parity_last_digit", "parity_sum_digits"
    ]
    df[numerical_features] = scaler.transform(df[numerical_features])

    encoder = joblib.load("scripts/encoder.pkl")
    categorical_features = ["odd_even", "big_small", "lag1_odd_even", "lag1_big_small"]
    encoded_cat = encoder.transform(df[categorical_features])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))
    df = pd.concat([df.drop(columns=categorical_features).reset_index(drop=True), encoded_cat_df], axis=1)

    try:
        existing_data = pd.read_csv(REALTIME_CSV)
        df = pd.concat([existing_data, df], ignore_index=True)
    except FileNotFoundError:
        print("ℹ️ Creating new real-time dataset.")

    df.to_csv(REALTIME_CSV, index=False)
    print(f"✅ Real-time data updated successfully in {REALTIME_CSV}")

def start_continuous_fetching():
    while True:
        print("\n🔄 Fetching real-time data...")
        preprocess_and_update()
        print(f"⏳ Waiting {FETCH_INTERVAL} seconds before next fetch...")
        time.sleep(FETCH_INTERVAL)

if __name__ == "__main__":
    start_continuous_fetching()
