import os
import pandas as pd
import numpy as np
import requests
import joblib
import time

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ODD_EVEN_SCALER_PATH = os.path.join(BASE_DIR, "odd_even_scaler.pkl")
BIG_SMALL_SCALER_PATH = os.path.join(BASE_DIR, "big_small_scaler.pkl")

# Real-time data API URL
REALTIME_API_URL = "https://www.apigx.cn/token/c5c808c4f81511ef9a5eafbf7b4e6e4c/code/jnd28/rows/10.json"

# Features used for preprocessing and prediction
ODD_EVEN_FEATURES = ['sum', 'sum_mod3', 'parity_last_digit', 'parity_sum_digits']
BIG_SMALL_FEATURES = ['sum', 'parity_sum_digits', 'rolling_sum_median', 'parity_last_digit', 'sum_mod3']

# Load saved scalers
odd_even_scaler = joblib.load(ODD_EVEN_SCALER_PATH)
big_small_scaler = joblib.load(BIG_SMALL_SCALER_PATH)

def fetch_realtime_data(api_url):
    """Fetch real-time data from the given API URL."""
    print("🚀 Fetching real-time data...")
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Debug: Print the raw data structure
        print("🔍 Raw data from API:", data)

        # Extract the 'data' field
        rows = data.get('data', [])
        if not rows:
            raise ValueError("No 'data' found in API response")

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Debug: Display the structure of the DataFrame
        print("🔍 DataFrame created from API response:")
        print(df.head())

        return df
    except Exception as e:
        print(f"🚨 Error: {e}")
        return None

def preprocess_realtime_data(data):
    """Preprocess real-time data for prediction."""
    try:
        # Extract numbers from 'opencode' and split them
        data['code'] = data['opencode'].apply(lambda x: list(map(int, x.split(','))))
        data['num1'] = data['code'].apply(lambda x: x[0])
        data['num2'] = data['code'].apply(lambda x: x[1])
        data['num3'] = data['code'].apply(lambda x: x[2])
        data['sum'] = data['num1'] + data['num2'] + data['num3']
        data['odd_even_1'] = data['sum'] % 2
        data['big_small_1'] = (data['sum'] >= 14).astype(int)
        data['rolling_sum_mean'] = data['sum'].rolling(window=3, min_periods=1).mean()
        data['rolling_sum_median'] = data['sum'].rolling(3, min_periods=1).median()
        data['lag1_sum'] = data['sum'].shift(1)
        data['lag1_odd_even'] = data['odd_even_1'].shift(1)
        data['lag1_big_small'] = data['big_small_1'].shift(1)
        data['diff1'] = data['num1'] - data['num2']
        data['diff2'] = data['num2'] - data['num3']
        data['sum_mod3'] = data['sum'] % 3
        data['last_digit'] = data['sum'] % 10
        data['sum_digits'] = data['sum'].astype(str).apply(lambda x: sum(int(ch) for ch in x))
        data['parity_last_digit'] = data['last_digit'] % 2
        data['parity_sum_digits'] = data['sum_digits'] % 2

        # Fill missing rolling/lag values
        missing_cols = ['rolling_sum_mean', 'rolling_sum_median', 'lag1_sum', 'lag1_odd_even', 'lag1_big_small']
        data[missing_cols] = data[missing_cols].fillna(data[missing_cols].median(numeric_only=True))

        # Apply scalers
        data[ODD_EVEN_FEATURES] = odd_even_scaler.transform(data[ODD_EVEN_FEATURES])
        data[BIG_SMALL_FEATURES] = big_small_scaler.transform(data[BIG_SMALL_FEATURES])
        return data
    except Exception as e:
        print(f"🚨 Error during preprocessing: {e}")
        return None

def main():
    """Main function to fetch, preprocess, and process real-time data."""
    FETCH_INTERVAL = 300  # Fetch data every 60 seconds

    while True:
        # Fetch data
        raw_data = fetch_realtime_data(REALTIME_API_URL)
        if raw_data is None or raw_data.empty:
            print("🚨 No data to process. Retrying in 60 seconds...")
            time.sleep(FETCH_INTERVAL)
            continue

        # Preprocess data
        processed_data = preprocess_realtime_data(raw_data)
        if processed_data is None or processed_data.empty:
            print("🚨 Preprocessing failed. Retrying in 60 seconds...")
            time.sleep(FETCH_INTERVAL)
            continue

        # Display preprocessed data (replace with prediction logic if needed)
        print("✅ Real-time data preprocessed successfully!")
        print(processed_data.head())

        # Wait before fetching the next batch of data
        print(f"⏳ Waiting {FETCH_INTERVAL} seconds before the next fetch...")
        time.sleep(FETCH_INTERVAL)

if __name__ == "__main__":
    main()


