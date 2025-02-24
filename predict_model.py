#!/usr/bin/env python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json

  
# Model Definition (Same as in Training)
  
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 1)  # Raw logits for BCEWithLogitsLoss
        self.dropout = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

  
# Data Preparation Functions
  
def load_latest_draws(csv_path, num_rows=20):
    """
    Load the last num_rows from test.csv.
    Since no timestamp is present, we assume the rows are in chronological order.
    """
    df = pd.read_csv(csv_path)
    latest_df = df.tail(num_rows).reset_index(drop=True)
    return latest_df

def prepare_features(df, target_column):
    """
    Drop the target column from the DataFrame and convert the remaining data to a tensor.
    """
    X = df.drop(columns=[target_column]).values
    return torch.tensor(X, dtype=torch.float32)

  
# Prediction Functions
  
def make_predictions(model, data):
    """
    Run the model on data and return both binary predictions and probabilities.
    """
    with torch.no_grad():
        outputs = model(data).view(-1)
        probabilities = torch.sigmoid(outputs).numpy().flatten()
        predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities

def predict_latest_draws(test_csv="test.csv", num_latest=20,
                         odd_even_target="odd_even_1", big_small_target="big_small_1",
                         threshold=0.7):
    # Load the latest 20 draws
    latest_df = load_latest_draws(test_csv, num_rows=num_latest)
    
    # Prepare features for both models (drop the corresponding target column)
    X_oe = prepare_features(latest_df, target_column=odd_even_target)
    X_bs = prepare_features(latest_df, target_column=big_small_target)
    
    # Determine input dimension (should be same for both models)
    input_dim = X_oe.shape[1]
    
    # Define model paths
    odd_even_model_path = "model_odd_even_trained.pth"
    big_small_model_path = "model_big_small_trained.pth"
    
    # Load trained models
    odd_even_model = MLP(input_dim=input_dim)
    big_small_model = MLP(input_dim=input_dim)
    odd_even_model.load_state_dict(torch.load(odd_even_model_path, map_location=torch.device("cpu")))
    big_small_model.load_state_dict(torch.load(big_small_model_path, map_location=torch.device("cpu")))
    odd_even_model.eval()
    big_small_model.eval()
    
    # Make predictions
    odd_even_preds, p_oe = make_predictions(odd_even_model, X_oe)
    big_small_preds, p_bs = make_predictions(big_small_model, X_bs)
    
    final_results = []
    for i in range(num_latest):
        # Ensure probabilities are Python floats
        prob_oe = float(p_oe[i])
        prob_bs = float(p_bs[i])
        conf_oe = abs(prob_oe - 0.5)
        conf_bs = abs(prob_bs - 0.5)
        
        # Determine individual model predictions
        pred_oe = "Odd" if prob_oe >= 0.5 else "Even"
        pred_bs = "Big" if prob_bs >= 0.5 else "Small"
        
        # Refined decision logic:
        if prob_bs >= 0.99 and 0.45 <= prob_oe <= 0.55:
            final_pred = pred_oe
            chosen_model = "Odd/Even"
            final_conf = conf_oe
        else:
            if conf_oe >= conf_bs:
                final_pred = pred_oe
                chosen_model = "Odd/Even"
                final_conf = conf_oe
            else:
                final_pred = pred_bs
                chosen_model = "Big/Small"
                final_conf = conf_bs
        
        recommended = "Yes" if final_conf >= threshold else "No"
        
        result = {
            "Draw Index": i + 1,
            "sum": float(latest_df.loc[i, "sum"]),
            "Odd/Even Prediction": pred_oe,
            "Odd/Even Probability": float(round(prob_oe, 4)),
            "Big/Small Prediction": pred_bs,
            "Big/Small Probability": float(round(prob_bs, 4)),
            "Final Estimated Result": final_pred,
            "Chosen Model": chosen_model,
            "Confidence": float(round(final_conf, 4)),
            "Recommended Bet": recommended
        }
        final_results.append(result)
    
    results_df = pd.DataFrame(final_results)
    print("Latest Predictions:")
    print(results_df)
    
    results_df.to_csv("latest_predictions.csv", index=False)
    with open("latest_predictions.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("Predictions saved to 'latest_predictions.csv' and 'latest_predictions.json'")
    
    return results_df

  
# Main Execution
  
if __name__ == "__main__":
    predict_latest_draws()
