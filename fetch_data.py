import requests
import pandas as pd
import time

# ✅ API Endpoint for fetching real-time data
API_URL = "https://www.apigx.cn/history/code/pcdd.html"  # Replace with actual API URL
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}

# ✅ Function to fetch data from API
def fetch_latest_draws():
    """
    Fetches real-time lottery draw data from the API.
    Returns a Pandas DataFrame with structured data.
    """
    try:
        response = requests.get(API_URL)  # Make API request
        response.raise_for_status()  # Check for HTTP errors
        
        data = response.json()  # Convert response to JSON
        draws = data.get("data", [])  # Extract draw data

        if not draws:
            print("❌ No draw results found in API response.")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(draws)

        # ✅ Adjust column names to match expected format
        df = df.rename(columns={
            "issue": "draw_id",
            "code1": "draw_number1",
            "code2": "draw_number2",
            "code3": "draw_number3",
            "opentime": "draw_time"  # Ensure correct time format
        })

        # ✅ Convert necessary columns to numeric values
        df[["draw_number1", "draw_number2", "draw_number3"]] = df[["draw_number1", "draw_number2", "draw_number3"]].astype(int)

        # ✅ Compute additional features
        df["sum"] = df["draw_number1"] + df["draw_number2"] + df["draw_number3"]
        df["odd_even"] = df["sum"] % 2  # 0 = Even, 1 = Odd
        df["big_small"] = (df["sum"] >= 14).astype(int)  # 0 = Small, 1 = Big
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return None

# ✅ Function to save fetched data to CSV
def save_to_csv(df, filename="latest_draws.csv"):
    """
    Saves the latest fetched data to a CSV file.
    """
    if df is not None:
        df.to_csv(filename, index=False)
        print(f"✅ Latest draw results saved to {filename}")

# ✅ Run script to fetch and save data
if __name__ == "__main__":
    while True:
        print("🔄 Fetching latest draw data...")
        latest_df = fetch_latest_draws()
        
        if latest_df is not None:
            save_to_csv(latest_df)
            print(latest_df.head())  # Show first few rows of the fetched data
        
        print("⏳ Waiting for next update...\n")
        time.sleep(30)  # Fetch new data every 30 seconds
