FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MLFLOW_TRACKING_URI=https://dagshub.com/9dfk1418/mlflow-dvc-tp.mlflow
ENV MLFLOW_TRACKING_USERNAME=9dfk1418
ENV MLFLOW_EXPERIMENT_NAME=RandomForest_V1.0.1

COPY fetch_model.py .

# Le token Dagshub est un secret de build : il ne reste pas dans l'image
RUN --mount=type=secret,id=dagshub_token \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    python fetch_model.py

COPY app/ ./app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
