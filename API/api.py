from fastapi import FastAPI, BackgroundTasks
import uvicorn
import os
import joblib
import torch
import pandas as pd
from predict_model1 import predict
from preproces_realtime1 import preprocess_and_update
from train_od_even import adaptive_retrain_loop as retrain_odd_even
from train_big_small import adaptive_retrain_loop as retrain_big_small

app = FastAPI()

# ✅ 1️⃣ Endpoint: Fetch and preprocess real-time data
@app.get("/fetch-realtime")
def fetch_realtime(background_tasks: BackgroundTasks):
    background_tasks.add_task(preprocess_and_update)
    return {"status": "Fetching and preprocessing real-time data started."}

# ✅ 2️⃣ Endpoint: Predict from latest real-time data
@app.get("/predict")
def make_predictions():
    predictions = predict()
    return {"predictions": predictions}

# ✅ 3️⃣ Endpoint: Retrain models on latest data
@app.get("/retrain")
def retrain_models(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_odd_even)
    background_tasks.add_task(retrain_big_small)
    return {"status": "Model retraining started in the background."}

# ✅ 4️⃣ Endpoint: Check latest predictions
@app.get("/latest-predictions")
def get_latest_predictions():
    file_path = "data/latest_prediction_with_percentages.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    return {"error": "No predictions found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
