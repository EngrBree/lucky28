import requests
import pandas as pd
import time

API_URL = "https://www.apigx.cn/token/c5c808c4f81511ef9a5eafbf7b4e6e4c/code/jnd28/rows/20.json"

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

        # Keep only required columns
        df = df[["expect", "num1", "num2", "num3", "sum", "odd_even", "big_small"]]

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return None

def update_realtime_data():
    """Fetches new data and updates the real-time dataset file."""
    new_data = fetch_realtime_draw()
    if new_data is not None:
        try:
            # Load existing real-time dataset
            real_time_path = "data/real_time.csv"
            try:
                existing_data = pd.read_csv(real_time_path)
                df = pd.concat([existing_data, new_data], ignore_index=True)
            except FileNotFoundError:
                df = new_data  # First-time creation

            # Remove duplicates
            df = df.drop_duplicates(subset=["expect"], keep="last")

            # Save updated dataset
            df.to_csv(real_time_path, index=False)
            print(f"✅ Real-time dataset updated. {len(new_data)} new draws added.")

        except Exception as e:
            print(f"⚠️ Failed to update real-time data: {e}")

# Run every 5 minutes
if __name__ == "__main__":
    while True:
        update_realtime_data()
        time.sleep(120)  # Wait 5 minutes before fetching again
