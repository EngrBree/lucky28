
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/train.csv")

# Define target column
target_col = 'big_small_1'
if target_col not in df.columns:
    raise ValueError("Target column 'big_small_1' not found in dataset.")

# Select numeric features
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

# Plot distribution per feature
for feature in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df, x=feature, hue=target_col, common_norm=False, fill=True, palette='Set1')
    plt.title(f"Distribution of '{feature}' by Class (Big=1, Small=0)")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Optional: Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
