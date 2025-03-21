# ✅ Updated preprocess_data.py — Dual Scalers Setup
import os
import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("📂 Current Working Directory:", os.getcwd())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ODD_EVEN_SCALER_PATH = os.path.join(BASE_DIR, "odd_even_scaler.pkl")
BIG_SMALL_SCALER_PATH = os.path.join(BASE_DIR, "big_small_scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['code'] = df['code'].apply(ast.literal_eval)
    df['num1'] = df['code'].apply(lambda x: int(x[0]))
    df['num2'] = df['code'].apply(lambda x: int(x[1]))
    df['num3'] = df['code'].apply(lambda x: int(x[2]))
    df['sum'] = df['num1'] + df['num2'] + df['num3']
    df['odd_even_1'] = df['sum'] % 2
    df['big_small_1'] = (df['sum'] >= 14).astype(int)
    df['rolling_sum_mean'] = df['sum'].rolling(window=3, min_periods=1).mean()
    df['rolling_sum_median'] = df['sum'].rolling(3, min_periods=1).median()
    df['lag1_sum'] = df['sum'].shift(1)
    df['lag1_odd_even'] = df['odd_even_1'].shift(1)
    df['lag1_big_small'] = df['big_small_1'].shift(1)
    df['diff1'] = df['num1'] - df['num2']
    df['diff2'] = df['num2'] - df['num3']
    df['sum_mod3'] = df['sum'] % 3
    df['last_digit'] = df['sum'] % 10
    df['sum_digits'] = df['sum'].astype(str).apply(lambda x: sum(int(ch) for ch in x))
    df['parity_last_digit'] = df['last_digit'] % 2
    df['parity_sum_digits'] = df['sum_digits'] % 2

    # Fill missing rolling/lag values
    missing_cols = ['rolling_sum_mean', 'rolling_sum_median', 'lag1_sum', 'lag1_odd_even', 'lag1_big_small']
    df[missing_cols] = df[missing_cols].fillna(df[missing_cols].median(numeric_only=True))
    return df

def preprocess_data(input_csv="data/newlucky28.csv", train_csv="data/train.csv", test_csv="data/test.csv"):
    df = load_data(input_csv)

    # 🧪 Create train-test split
    test_size = max(0.2, 100 / len(df))
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=False)

    # 🧠 Define feature groups for each model
    odd_even_features = ['sum', 'sum_mod3', 'parity_last_digit', 'parity_sum_digits']
    big_small_features = ['sum', 'parity_sum_digits', 'rolling_sum_median', 'parity_last_digit', 'sum_mod3']

    # 🔢 Apply separate scalers for each feature group
    odd_even_scaler = StandardScaler()
    train_df[odd_even_features] = odd_even_scaler.fit_transform(train_df[odd_even_features])
    test_df[odd_even_features] = odd_even_scaler.transform(test_df[odd_even_features])
    joblib.dump(odd_even_scaler, ODD_EVEN_SCALER_PATH)
    print(f"✅ Odd-Even Scaler saved at {ODD_EVEN_SCALER_PATH}")

    big_small_scaler = StandardScaler()
    train_df[big_small_features] = big_small_scaler.fit_transform(train_df[big_small_features])
    test_df[big_small_features] = big_small_scaler.transform(test_df[big_small_features])
    joblib.dump(big_small_scaler, BIG_SMALL_SCALER_PATH)
    print(f"✅ Big-Small Scaler saved at {BIG_SMALL_SCALER_PATH}")

    # 🎯 One-Hot Encode lag targets (optional but consistent)
    categorical_features = ['lag1_odd_even', 'lag1_big_small']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(train_df[categorical_features])
    train_encoded = encoder.transform(train_df[categorical_features])
    test_encoded = encoder.transform(test_df[categorical_features])

    joblib.dump(encoder, ENCODER_PATH)
    print(f"✅ Encoder saved at {ENCODER_PATH}")

    # Merge encoded categorical features
    train_cat_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_features))
    test_cat_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_features))

    train_df = pd.concat([train_df.drop(columns=categorical_features).reset_index(drop=True), train_cat_df], axis=1)
    test_df = pd.concat([test_df.drop(columns=categorical_features).reset_index(drop=True), test_cat_df], axis=1)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print("✅ Preprocessing complete! Train & test sets saved successfully.")

if __name__ == "__main__":
    preprocess_data()
