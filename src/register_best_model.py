import os, mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()
exp = client.get_experiment_by_name("breast-cancer-baseline")
runs = client.search_runs([exp.experiment_id], order_by=["metrics.f1_macro DESC", "metrics.accuracy DESC"])
best = runs[0]
model_uri = f"runs:/{best.info.run_id}/model"
reg_name = "breast-cancer-prod"
mlflow.register_model(model_uri, reg_name)
versions = client.search_model_versions(f"name='{reg_name}'")
latest_ver = max(int(v.version) for v in versions)
client.transition_model_version_stage(name=reg_name, version=str(latest_ver), stage="Staging", archive_existing_versions=True)
print(f"Registered {reg_name} v{latest_ver} from run {best.info.run_name} in Staging")
