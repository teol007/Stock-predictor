from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta, UTC
import pandas as pd


from src.routes.stock_price_route import router as stock_router
from src.routes.production_metrics_route import router as production_metrics_router
from src.download_model import load_registered_model_and_pipeline
from src.db import db

model, pipeline = load_registered_model_and_pipeline()
predictions_collection = db['predictions']

# Define input schema
class PriceInput(BaseModel):
    data: List[List[float]]  # 2D array of recent price sequences, length == window_size

app = FastAPI()
app.include_router(stock_router)
app.include_router(production_metrics_router)

class ValueItem(BaseModel):
    datetime: str
    open: str
    high: str
    low: str
    close: str
    volume: str

class MetaData(BaseModel):
    symbol: str
    interval: str
    currency: str
    exchange_timezone: str
    exchange: str
    mic_code: str
    type: str

class Data(BaseModel):
    meta: MetaData
    values: List[ValueItem]

class PredictRequest(BaseModel):
    source: str
    data: Data

@app.post("/predict/price")
def predict(request: PredictRequest):
    values = request.data.values

    if len(values) < 79:
        raise HTTPException(status_code=400, detail="At least 79 data points are required")

    # Use the most recent 79 entries (they are ordered from most recent to oldest)
    values_79 = values[:79][::-1]  # reverse to chronological order

    # Convert to DataFrame using actual JSON values including datetime
    df = pd.DataFrame([v.model_dump() for v in values_79])

    # Rename columns to match training
    df.rename(columns={
        "open": "open_price",
        "high": "high_price",
        "low": "low_price",
        "close": "close_price",
        "volume": "volume"
    }, inplace=True)

    # Reorder columns to match pipeline expectations
    df = df[["open_price", "high_price", "low_price", "close_price"]]

    predictions = []
    for i in range(8):
        # Apply pipeline
        try:
            X, _ = pipeline.transform(df)
        except Exception as e:
            print(f"Transform failed: {e}")
            raise HTTPException(status_code=500, detail=f"Pipeline transformation failed: {e}")

        # Predict
        prediction = model.predict(X)

        # Inverse transform to original scale
        try:
            scaler = pipeline.named_steps["preprocess"] \
                            .named_transformers_["numeric_transformer"] \
                            .named_steps["normalize"]
            prediction_inv = scaler.inverse_transform(prediction)
        except Exception as e:
            print(f"Inverse transform failed: {e}")
            raise HTTPException(status_code=500, detail=f"Inverse transform failed: {e}")
        
        last_pred = prediction_inv[-1]  # last predicted timestep, shape (4,)
        predictions.append({
            "open": float(last_pred[0]),
            "high": float(last_pred[1]),
            "low": float(last_pred[2]),
            "close": float(last_pred[3]),
        })

        new_row = {
            "open_price": float(last_pred[0]),
            "high_price": float(last_pred[1]),
            "low_price": float(last_pred[2]),
            "close_price": float(last_pred[3])
        }

        # Append new row to df (create DataFrame from dict, then concat)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Keep only the last 79 rows for the next iteration (rolling window)
        df = df.tail(79).reset_index(drop=True)

    record = {
        "timestamp": datetime.now(UTC),
        "model_type": "price",
        "symbol": request.data.meta.symbol,
        "predictions": predictions,
    }
    predictions_collection.insert_one(record)

    return {"prediction": predictions}
