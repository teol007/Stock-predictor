from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import mlflow
import tempfile
import joblib

router = APIRouter()

def download_model_artifact(run_id, artifact_path):
    local_dir = tempfile.mkdtemp()
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=local_dir)
    return local_path

run_id = "3b49233ba9ca4d328133af028583ce69"
artifact_subpath = "model_activity_AAPL.pkl"
model_path = download_model_artifact(run_id, artifact_subpath)
model = joblib.load(model_path)

VOLUME_CLASSES = [
    "Very Low Activity",
    "Low Activity",
    "Moderate Activity",
    "High Activity",
    "Very High Activity"
]

class Candle(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class MetaData(BaseModel):
    symbol: str
    interval: str
    currency: str
    exchange_timezone: str
    exchange: str
    mic_code: str
    type: str

class CandleData(BaseModel):
    meta: MetaData
    values: List[Candle]

class PredictionInput(BaseModel):
    source: str
    data: CandleData

@router.post("/predict/activity")
async def predict_activity(input_data: PredictionInput):
    try:
        candles = input_data.data.values
        if not candles:
            raise HTTPException(status_code=400, detail="No candle data provided.")

        # Sort descending by datetime (latest first)
        candles_sorted = sorted(candles, key=lambda x: x.datetime, reverse=True)

        # Convert to DataFrame (only the latest candle, or all if you want but predict only first)
        df = pd.DataFrame([{
            "open_price": c.open,
            "high_price": c.high,
            "low_price": c.low,
            "close_price": c.close
        } for c in candles_sorted])

        # Predict only on the latest candle (first row)
        pred = model.predict(df.iloc[[0]])  # Note the double brackets for DataFrame slice
        predicted_class = int(pred[0])
        label = VOLUME_CLASSES[predicted_class]

        return {
            "predicted_class": predicted_class,
            "label": label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
