# ✅ Updated preprocess_data.py — With Parity Feature Engineering
import os
import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("📂 Current Working Directory:", os.getcwd())

FEATURES_BIG_SMALL = [
    'sum', 'rolling_sum_mean', 'lag1_sum', 'lag1_big_small_1.0', 'diff1'
]

FEATURES_ODD_EVEN = [
    'sum', 'rolling_sum_mean', 'lag1_sum', 'lag1_odd_even_1.0', 'diff2', 'rolling_sum_median', 'sum_mod3',
    'parity_last_digit', 'parity_sum_digits'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['code'] = df['code'].apply(ast.literal_eval)
    df['num1'] = df['code'].apply(lambda x: int(x[0]))
    df['num2'] = df['code'].apply(lambda x: int(x[1]))
    df['num3'] = df['code'].apply(lambda x: int(x[2]))
    df['sum'] = df['num1'] + df['num2'] + df['num3']
    df['odd_even'] = df['sum'] % 2
    df['big_small'] = (df['sum'] >= 14).astype(int)
    df['rolling_sum_mean'] = df['sum'].rolling(window=3, min_periods=1).mean()
    df['rolling_sum_median'] = df['sum'].rolling(3, min_periods=1).median()
    df['lag1_sum'] = df['sum'].shift(1)
    df['lag1_odd_even'] = df['odd_even'].shift(1)
    df['lag1_big_small'] = df['big_small'].shift(1)
    df['diff1'] = df['num1'] - df['num2']
    df['diff2'] = df['num2'] - df['num3']
    df['sum_mod3'] = df['sum'] % 3

    # ✅ Add Parity Feature Engineering
    df['last_digit'] = df['sum'] % 10
    df['sum_digits'] = df['sum'].astype(str).apply(lambda x: sum(int(ch) for ch in x))
    df['parity_last_digit'] = df['last_digit'] % 2
    df['parity_sum_digits'] = df['sum_digits'] % 2

    missing_cols = ['rolling_sum_mean', 'rolling_sum_median', 'lag1_sum', 'lag1_odd_even', 'lag1_big_small']
    df[missing_cols] = df[missing_cols].fillna(df[missing_cols].median())
    return df

def preprocess_data(input_csv="data/newlucky28.csv", train_csv="data/train.csv", test_csv="data/test.csv"):
    df = load_data(input_csv)
    test_size = max(0.2, 100 / len(df))
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=False)

    numerical_features = [
        'sum', 'rolling_sum_mean', 'lag1_sum', 'diff1', 'diff2',
        'rolling_sum_median', 'sum_mod3', 'parity_last_digit', 'parity_sum_digits'
    ]

    scaler = StandardScaler()
    train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    test_df[numerical_features] = scaler.transform(test_df[numerical_features])

    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved at {SCALER_PATH}")

    categorical_features = ['odd_even', 'big_small', 'lag1_odd_even', 'lag1_big_small']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(train_df[categorical_features])
    train_encoded = encoder.transform(train_df[categorical_features])
    test_encoded = encoder.transform(test_df[categorical_features])

    joblib.dump(encoder, ENCODER_PATH)
    print(f"✅ Encoder saved at {ENCODER_PATH}")

    train_cat_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_features))
    test_cat_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_features))

    train_df = pd.concat([train_df.drop(columns=categorical_features).reset_index(drop=True), train_cat_df], axis=1)
    test_df = pd.concat([test_df.drop(columns=categorical_features).reset_index(drop=True), test_cat_df], axis=1)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print("✅ Preprocessing complete! Train & test sets saved successfully.")

if __name__ == "__main__":
    preprocess_data()
