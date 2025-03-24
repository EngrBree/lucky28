from fastapi import FastAPI
import json
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.predict_model1 import predict, fetch_latest_draws

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Lucky28 Prediction API"}
@app.post("/predict/")
def run_prediction():
    predictions = predict()
    if predictions is None:
        return {"error": "Prediction failed or no data available"}

    df = fetch_latest_draws()
    if df is None or df.empty:
        return {"error": "No draw data available"}

    # Parse opencode (e.g., "8,9,2") into num1, num2, num3
    try:
        code_split = df["opencode"].str.split(",", expand=True)
        df["num1"] = code_split[0].astype(int)
        df["num2"] = code_split[1].astype(int)
        df["num3"] = code_split[2].astype(int)
    except Exception as e:
        return {"error": f"Failed to parse opencode: {str(e)}"}

    results = []
    for pred in predictions:
        draw_id = pred.get("Draw ID")
        draw_row = df[df["expect"] == str(draw_id)]  # Match draw ID
        if draw_row.empty:
            actual_size = "Unknown"
            actual_parity = "Unknown"
            draw_time = ""
        else:
            draw_numbers = draw_row[["num1", "num2", "num3"]].values.flatten().tolist()
            total = sum(draw_numbers)
            actual_size = "Big" if total >= 14 else "Small"
            actual_parity = "Even" if total % 2 == 0 else "Odd"
            draw_time = draw_row["opentime"].values[0]  # ✅ Add Draw Time from opentime

        results.append({
            "Draw ID": draw_id,
            "Timestamp": draw_time,  # ✅ Add Timestamp field here
            "Prediction": pred.get("Prediction", "N/A"),
            "Accuracy (%)": float(pred.get("Accuracy (%)", 0)),
            "Actual Size": actual_size,
            "Actual Parity": actual_parity
        })

    return results
