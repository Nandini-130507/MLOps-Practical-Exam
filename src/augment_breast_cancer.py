
import pandas as pd
import numpy as np

# Load Version 1 dataset
file_path = "data/raw/breast_cancer_v1.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Separate features and target
target_col = "target"

if target_col not in df.columns:
    raise ValueError("Target column not found. Check your dataset columns.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Generate 20 synthetic records
np.random.seed(42)

synthetic_rows = []

for i in range(20):
    # Randomly select one existing record
    index = np.random.randint(0, len(df))

    # Copy its feature values
    new_features = X.iloc[index].copy()

    # Add small noise to numerical features
    for col in X.columns:
        noise = np.random.normal(0, X[col].std() * 0.01)
        new_features[col] = new_features[col] + noise

    # Preserve the original target label
    new_features[target_col] = y.iloc[index]

    synthetic_rows.append(new_features)

# Combine original and synthetic records
new_df = pd.concat(
    [df, pd.DataFrame(synthetic_rows)],
    ignore_index=True
)

# Save updated dataset to the SAME tracked file
new_df.to_csv(file_path, index=False)

print("Version 2 created successfully!")
print("New dataset shape:", new_df.shape)
print("Total records:", len(new_df))