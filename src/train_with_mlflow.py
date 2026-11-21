import os, mlflow, mlflow.sklearn, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("breast-cancer-baseline")

df = pd.read_csv("data/processed/cancer_features.csv")
cols = [c for c in df.columns if c not in ["diagnosis", "radius_category"]]
X = df[cols].fillna(df[cols].median())
y = LabelEncoder().fit_transform(df["diagnosis"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    "logistic_regression": LogisticRegression(max_iter=5000, C=1.0),
    "random_forest_shallow": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
    "random_forest_deep": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("f1_macro", f1_score(y_test, preds, average="macro", zero_division=0))
        mlflow.sklearn.log_model(model, artifact_path="model")
