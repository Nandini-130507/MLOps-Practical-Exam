import os, mlflow.sklearn, pandas as pd

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
model = mlflow.sklearn.load_model("models:/breast-cancer-prod/Staging")
df = pd.read_csv("data/processed/cancer_features.csv")
cols = [c for c in df.columns if c not in ["diagnosis", "radius_category"]]
preds = model.predict(df[cols].fillna(df[cols].median()).head(5))
print("Sample predictions:", preds.tolist())
