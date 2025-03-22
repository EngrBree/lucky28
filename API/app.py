from fastapi import FastAPI
import pandas as pd
from scripts.predict_model1 import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Lucky 28 Prediction API Running"}

@app.get("/predict")
def get_prediction():
    results = predict()  # Run prediction automatically
    if results is None:
        return {"error": "No prediction available"}

    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
