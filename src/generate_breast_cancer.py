import os
from sklearn.datasets import load_breast_cancer

os.makedirs("data/raw", exist_ok=True)
cancer = load_breast_cancer(as_frame=True)
df = cancer.frame
df.to_csv("data/raw/breast_cancer_v1.csv", index=False)
print(f"Saved {len(df)} rows to data/raw/breast_cancer_v1.csv")
print(df.head())
python src/generate_breast_cancer.py
