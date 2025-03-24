from fastapi import FastAPI
import subprocess
import json

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/predict/")
def run_prediction():
    try:
        # Run the prediction script
        result = subprocess.run(["python", "-m", "scripts.predict_model1"], capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"error": f"Prediction script error: {result.stderr}"}
        
        # Load the prediction output from JSON file
        with open("prediction_results.json", "r") as f:
            prediction_output = json.load(f)
        
        return prediction_output
    except Exception as e:
        return {"error": str(e)}
