from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.pyfunc
import numpy as np

app = FastAPI(
    title="Digits Prediction API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

model = mlflow.pyfunc.load_model("model")


class PredictionInput(BaseModel):
    features: list[float] = Field(min_length=64, max_length=64)


@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}
