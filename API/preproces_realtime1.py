# ✅ Updated preproces_realtime.py — Parity Features & Dual Scaler Setup
import pandas as pd
import numpy as np
import requests
import time
import joblib
import os

REALTIME_CSV = "data/real_time_preprocessed.csv"
API_URL = "https://www.apigx.cn/token/81dc4724ff3111ef8d0c01bd2828daeb/code/jnd28/rows/20.json"
FETCH_INTERVAL = 600

ODD_EVEN_SCALER_PATH = "scripts/odd_even_scaler.pkl"
BIG_SMALL_SCALER_PATH = "scripts/big_small_scaler.pkl"
ENCODER_PATH = "scripts/encoder.pkl"

# Define feature sets
odd_even_features = ["sum", "sum_mod3", "parity_last_digit", "parity_sum_digits"]
big_small_features = ["sum", "parity_sum_digits", "rolling_sum_median", "parity_last_digit", "sum_mod3"]

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
        df["odd_even_1"] = df["sum"] % 2
        df["big_small_1"] = (df["sum"] >= 14).astype(int)
        return df
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return None


def preprocess_and_update():
    df = fetch_realtime_draw()
    if df is None:
        return

    # ➕ Feature Engineering (same as in preprocessing script)
    df["rolling_sum_mean"] = df["sum"].rolling(window=3, min_periods=1).mean()
    df["rolling_sum_median"] = df["sum"].rolling(3, min_periods=1).median()
    df["lag1_sum"] = df["sum"].shift(1)
    df["lag1_odd_even"] = df["odd_even_1"].shift(1)
    df["lag1_big_small"] = df["big_small_1"].shift(1)
    df["diff1"] = df["num1"] - df["num2"]
    df["diff2"] = df["num2"] - df["num3"]
    df["sum_mod3"] = df["sum"] % 3
    df["last_digit"] = df["sum"] % 10
    df["sum_digits"] = df["sum"].astype(str).apply(lambda x: sum(int(ch) for ch in x))
    df["parity_last_digit"] = df["last_digit"] % 2
    df["parity_sum_digits"] = df["sum_digits"] % 2

    missing_cols = ['rolling_sum_mean', 'rolling_sum_median', 'lag1_sum', 'lag1_odd_even', 'lag1_big_small']
    df[missing_cols] = df[missing_cols].fillna(df[missing_cols].median(numeric_only=True))

    # ✅ Load scalers
    try:
        odd_even_scaler = joblib.load(ODD_EVEN_SCALER_PATH)
        big_small_scaler = joblib.load(BIG_SMALL_SCALER_PATH)
    except Exception as e:
        print(f"❌ Error loading scalers: {e}")
        return

    # ✅ Apply each scaler to its own features
    for col in odd_even_features:
        if col not in df.columns:
            df[col] = 0
    for col in big_small_features:
        if col not in df.columns:
            df[col] = 0

    try:
        df_odd_scaled = odd_even_scaler.transform(df[odd_even_features])
        df_big_scaled = big_small_scaler.transform(df[big_small_features])

        # Optionally store scaled versions separately if needed
        odd_scaled_df = pd.DataFrame(df_odd_scaled, columns=[f"oe_{col}" for col in odd_even_features])
        big_scaled_df = pd.DataFrame(df_big_scaled, columns=[f"bs_{col}" for col in big_small_features])
        df = pd.concat([df.reset_index(drop=True), odd_scaled_df, big_scaled_df], axis=1)
    except Exception as e:
        print(f"❌ Feature scaling error: {e}")
        return

    # ✅ Apply Encoder to lag features
    try:
        encoder = joblib.load(ENCODER_PATH)
        cat_features = ["lag1_odd_even", "lag1_big_small"]
        cat_encoded = encoder.transform(df[cat_features])
        cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_features))
        df = pd.concat([df.drop(columns=cat_features).reset_index(drop=True), cat_df], axis=1)
    except Exception as e:
        print(f"❌ Encoder error: {e}")

    # ✅ Save to CSV
    try:
        if os.path.exists(REALTIME_CSV):
            old_data = pd.read_csv(REALTIME_CSV)
            df = pd.concat([old_data, df], ignore_index=True)
        else:
            print("ℹ️ Creating new real-time dataset...")

        df.to_csv(REALTIME_CSV, index=False)
        print(f"✅ Real-time data updated successfully in {REALTIME_CSV}")
    except Exception as e:
        print(f"❌ Error saving real-time CSV: {e}")


def start_continuous_fetching():
    while True:
        print("\n🔄 Fetching real-time data...")
        preprocess_and_update()
        print(f"⏳ Waiting {FETCH_INTERVAL} seconds before next fetch...")
        time.sleep(FETCH_INTERVAL)


if __name__ == "__main__":
    start_continuous_fetching()
