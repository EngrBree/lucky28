#!/usr/bin/env python
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ✅ Define features used by both models
FEATURES_BIG_SMALL = [
    'sum', 'rolling_sum_mean', 'lag1_sum', 'lag1_big_small_1.0', 'diff1'
]  # Exclude odd/even features

FEATURES_ODD_EVEN = [
    'sum', 'rolling_sum_mean', 'lag1_sum', 'lag1_odd_even_1.0', 'diff2', 'rolling_sum_median', 'sum_mod3'
]  # Exclude big/small features


# ✅ Data Processing Function
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['code'] = df['code'].apply(ast.literal_eval)
    
    # ✅ Extract individual numbers
    df['num1'] = df['code'].apply(lambda x: int(x[0]))
    df['num2'] = df['code'].apply(lambda x: int(x[1]))
    df['num3'] = df['code'].apply(lambda x: int(x[2]))
    
    # ✅ Compute sum and derived features
    df['sum'] = df['num1'] + df['num2'] + df['num3']
    df['odd_even'] = df['sum'] % 2  # 0 = Even, 1 = Odd
    df['big_small'] = (df['sum'] >= 14).astype(int)  # 0 = Small, 1 = Big

    # ✅ Rolling & lagging features
    df['rolling_sum_mean'] = df['sum'].rolling(window=3, min_periods=1).mean()
    df['rolling_sum_median'] = df['sum'].rolling(3, min_periods=1).median()
    df['lag1_sum'] = df['sum'].shift(1)
    df['lag1_odd_even'] = df['odd_even'].shift(1)
    df['lag1_big_small'] = df['big_small'].shift(1)

    # ✅ Differences between consecutive numbers
    df['diff1'] = df['num1'] - df['num2']
    df['diff2'] = df['num2'] - df['num3']
    df['sum_mod3'] = df['sum'] % 3  # Detect alternating sum patterns

    # ✅ Handle missing values
    missing_cols = ['rolling_sum_mean', 'rolling_sum_median', 'lag1_sum', 'lag1_odd_even', 'lag1_big_small']
    df[missing_cols] = df[missing_cols].fillna(df[missing_cols].median())

    
    return df






# ✅ Main Preprocessing Function
def preprocess_data(input_csv="data/newlucky28.csv", train_csv="data/train.csv", test_csv="data/test.csv"):
    # ✅ Load full dataset
    df = load_data(input_csv)
    

    # ✅ Split into train & test (80% train, 20% test; chronological order)
    test_size = max(0.2, 100 / len(df))  # Ensure at least 100 test samples
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=False)

    # ✅ Scale numerical features
    numerical_features = ['sum', 'rolling_sum_mean', 'lag1_sum', 'diff1', 'diff2', 'rolling_sum_median', 'sum_mod3']
    scaler = StandardScaler()
    train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    test_df[numerical_features] = scaler.transform(test_df[numerical_features])  # Transform only

    # ✅ One-Hot Encode categorical features
    categorical_features = ['odd_even', 'big_small', 'lag1_odd_even', 'lag1_big_small']
    combined_cat = pd.concat([train_df[categorical_features], test_df[categorical_features]], axis=0)
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cat = encoder.fit_transform(combined_cat)
    
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))
    train_encoded_df = encoded_cat_df.iloc[:len(train_df)].reset_index(drop=True)
    test_encoded_df = encoded_cat_df.iloc[len(train_df):].reset_index(drop=True)

    # ✅ Merge encoded data with train & test sets
    train_df = pd.concat([train_df.drop(columns=categorical_features).reset_index(drop=True), train_encoded_df], axis=1)
    test_df = pd.concat([test_df.drop(columns=categorical_features).reset_index(drop=True), test_encoded_df], axis=1)


    

    # ✅ Save preprocessed datasets
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("✅ Preprocessing complete! Train & test sets saved successfully.")
    print(train_df["big_small_1"].value_counts())  # Check class balance


# ✅ Run script if executed directly
if __name__ == "__main__":
    preprocess_data()
