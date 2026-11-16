import shutil
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dagshub.init(repo_owner="9dfk1418", repo_name="mlflow-dvc-tp", mlflow=True)
mlflow.set_experiment("RandomForest_V1.0.1")

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)


def train_and_log_model(n_estimators, max_depth, run_name):
    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        shutil.rmtree("tmp_model", ignore_errors=True)
        mlflow.sklearn.save_model(
            model, "tmp_model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        mlflow.log_artifacts("tmp_model", artifact_path="model")
        print(f"Run {run_name}: accuracy={accuracy:.4f}")


train_and_log_model(20, 5, "random_forest_run_1")
train_and_log_model(100, 10, "random_forest_run_2")
