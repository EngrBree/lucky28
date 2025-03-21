import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt

# Ensure you can import your MLP model architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_architecture import MLP

# File paths (adjust if necessary)
TRAIN_CSV = "data/train.csv"
SCALER_PATH = "scripts/big_small_scaler.pkl"
MODEL_PATH = "models/big_small_model.pth"

# Define the required features
FEATURES = ['sum', 'parity_sum_digits', 'rolling_sum_median', 'parity_last_digit', 'sum_mod3']

def load_training_data():
    """Load the training dataset."""
    try:
        df = pd.read_csv(TRAIN_CSV)
        print(f"Loaded training data from {TRAIN_CSV} with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)

def inspect_class_distribution(df):
    """Print class distributions for big_small and odd_even labels."""
    if 'big_small' in df.columns:
        print("Big-Small Distribution:")
        print(df['big_small'].value_counts())
    else:
        print("Column 'big_small' not found in training data.")

    if 'odd_even' in df.columns:
        print("\nOdd-Even Distribution:")
        print(df['odd_even'].value_counts())
    else:
        print("Column 'odd_even' not found in training data.")

def encode_labels(df):
    """Encode big_small to numeric values: 0 for Small, 1 for Big."""
    if 'big_small' in df.columns:
        df['big_small_numeric'] = df['big_small'].map({'Small': 0, 'Big': 1})
    else:
        print("Column 'big_small' not found. Cannot encode labels.")
    return df

def compute_and_print_class_weights(df):
    """Compute class weights based on the encoded labels."""
    if 'big_small_numeric' not in df.columns:
        print("big_small_numeric column missing. Encoding labels first.")
        df = encode_labels(df)
    
    y = df['big_small_numeric'].values
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    print("\nComputed class weights (for [Small, Big]):", weights)
    # For BCEWithLogitsLoss, we use the weight for the positive class ("Big")
    pos_weight = torch.tensor(weights[1], dtype=torch.float32)
    print("Using pos_weight for loss function:", pos_weight.item())
    return pos_weight

def inspect_feature_statistics(df):
    """Print statistics of the required features."""
    print("\nFeature statistics (raw values):")
    if set(FEATURES).issubset(df.columns):
        print(df[FEATURES].describe())
    else:
        missing = set(FEATURES) - set(df.columns)
        print("Missing features:", missing)

def inspect_scaler(scaler, df):
    """Transform features using the scaler and print post-scaling stats."""
    if set(FEATURES).issubset(df.columns):
        X = df[FEATURES].values
        X_scaled = scaler.transform(X)
        df_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
        print("\nScaled feature statistics:")
        print(df_scaled.describe())
    else:
        print("Cannot scale features because some are missing.")

def load_scaler():
    """Load the scaler from file."""
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Loaded scaler from {SCALER_PATH}")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        sys.exit(1)

def load_model(input_dim):
    """Load the big_small model."""
    try:
        model = MLP(input_dim=input_dim)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print(f"Loaded model from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def evaluate_sample_predictions(model, scaler, df, threshold=0.5):
    """Run sample predictions, print raw logits and probabilities, and plot distributions."""
    if not set(FEATURES).issubset(df.columns):
        print("Missing required features for prediction.")
        return
    
    X = df[FEATURES].values
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(X_tensor).view(-1)
        probs = torch.sigmoid(logits).numpy()
    
    print("\nSample predictions:")
    for i in range(min(10, len(probs))):
        big_prob = probs[i]
        small_prob = 1 - big_prob
        pred_label = "Big" if big_prob > threshold else "Small"
        print(f"Row {i+1}: Logit={logits[i]:.4f}, Big prob={big_prob:.4f}, Small prob={small_prob:.4f} -> Prediction: {pred_label}")
    
    # Plot histograms for logits and probabilities
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(logits.numpy(), bins=50, color='skyblue', edgecolor='black')
    plt.title("Histogram of Logits")
    plt.xlabel("Logit Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(probs, bins=50, color='salmon', edgecolor='black')
    plt.title("Histogram of Probabilities for 'Big'")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def main():
    # Step 1: Load data
    df = load_training_data()
    
    # Step 2: Inspect class distributions
    inspect_class_distribution(df)
    
    # Step 3: Encode labels and compute class weights
    df = encode_labels(df)
    pos_weight = compute_and_print_class_weights(df)
    
    # Step 4: Inspect feature statistics
    inspect_feature_statistics(df)
    
    # Step 5: Load scaler and inspect scaled features
    scaler = load_scaler()
    inspect_scaler(scaler, df)
    
    # Step 6: Load the model and evaluate sample predictions
    input_dim = len(FEATURES)
    model = load_model(input_dim)
    evaluate_sample_predictions(model, scaler, df, threshold=0.55)  # Adjust threshold as needed

if __name__ == "__main__":
    main()
