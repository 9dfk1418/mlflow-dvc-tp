import os
import shutil

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/9dfk1418/mlflow-dvc-tp.mlflow")
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "RandomForest_V1.0.1")

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found on DagsHub")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=20,
)

for run in runs:
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model", dst_path="/tmp/dl"
        )
        if not os.path.exists(os.path.join(local, "MLmodel")):
            continue
        shutil.copytree(local, "/app/model", dirs_exist_ok=True)
        print(f"Model downloaded from run: {run.info.run_id}")
        break
    except Exception as error:
        print(f"Could not use run {run.info.run_id}: {error}")
else:
    raise RuntimeError("No trained model found on DagsHub")
