import pandas as pd
import numpy as np

print("\n📊 VALIDATING DATA INTEGRITY CHECKS...\n")

# === Load your datasets ===
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
real_time = pd.read_csv("data/real_time_preprocessed.csv")
df = pd.read_csv("data/test.csv")
print(df['big_small_1'].value_counts(normalize=True))


# ✅ 1. Check class distribution
def class_distribution(df, label_col, name="Dataset"):
    print(f"\n👉 Class distribution in {name}:")
    print(df[label_col].value_counts(normalize=True).round(3) * 100)
    print(df[label_col].value_counts())

class_distribution(train, "big_small_1", "TRAIN SET")
class_distribution(test, "big_small_1", "TEST SET")
class_distribution(real_time, "big_small_1", "REAL-TIME SET")

# ✅ 2. Check for duplicate rows (possible leakage risk)
duplicates_in_test = pd.merge(test, train, how='inner', on=train.columns.tolist()).shape[0]
print(f"\n🔍 Potential duplicates between TRAIN and TEST: {duplicates_in_test}")

# ✅ 3. Check if features are consistent
train_features = set(train.columns)
test_features = set(test.columns)
realtime_features = set(real_time.columns)

print("\n✅ Feature consistency check:")
print(f"Features in Train but not in Test: {train_features - test_features}")
print(f"Features in Test but not in Train: {test_features - train_features}")
print(f"Features in Train but not in Real-Time: {train_features - realtime_features}")

# ✅ 4. Compare distributions (mean/std per feature)
numerical_cols = ['sum', 'rolling_sum_mean', 'lag1_sum', 'diff1', 'diff2', 'rolling_sum_median', 'sum_mod3']

print("\n📐 Feature Distribution Comparison (Mean ± Std):")
for col in numerical_cols:
    if col in train.columns:
        print(f"\n🔸 {col}")
        print(f"Train     → mean: {train[col].mean():.4f}, std: {train[col].std():.4f}")
        print(f"Test      → mean: {test[col].mean():.4f}, std: {test[col].std():.4f}")
        print(f"Real-Time → mean: {real_time[col].mean():.4f}, std: {real_time[col].std():.4f}")
    else:
        print(f"{col} missing in data.")

print("\n✅ Validation completed.")
