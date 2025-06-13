from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta, UTC
import pytz
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import dateutil.parser
from src.routes.stock_price_route import stock_price_apple
from src.db import db

predictions_collection = db['production-metrics']

router = APIRouter()

nasdaq_tz = pytz.timezone("America/New_York")
predictions_collection = db['predictions']
metrics_collection = db['metrics']

def round_to_next_quarter(dt: datetime) -> datetime:
    """
    Round datetime up to next quarter hour (00, 15, 30, 45).
    If exactly on quarter, returns the same datetime.
    """
    minute = dt.minute
    remainder = minute % 15
    if remainder == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.replace(second=0, microsecond=0)
    add_minutes = 15 - remainder
    dt = dt + timedelta(minutes=add_minutes)
    return dt.replace(second=0, microsecond=0)

@router.get("/metrics/price/calculate")
def calculate_metrics():
    # 1. Get real price data from existing endpoint function
    real_data_response = stock_price_apple()
    real_values = real_data_response["data"]["values"]
    if not real_values:
        raise HTTPException(status_code=404, detail="No real price data found")

    # Parse and index real data by NASDAQ tz datetime for quick lookup
    real_data_map = {}
    for val in real_values:
        dt = dateutil.parser.parse(val["datetime"])
        dt = dt.astimezone(nasdaq_tz)
            
        rounded_dt = round_to_next_quarter(dt)
        real_data_map[rounded_dt] = {
            "open": float(val["open"]),
            "high": float(val["high"]),
            "low": float(val["low"]),
            "close": float(val["close"])
        }

    # 2. Iterate over all prediction records to compute metrics
    prediction_records = predictions_collection.find({"model_type": "price"})

    all_metrics = []

    for record in prediction_records:
        pred_timestamp_utc = record["timestamp"]  # UTC datetime
        pred_timestamp_nasdaq = pred_timestamp_utc.astimezone(nasdaq_tz)
        start_time = round_to_next_quarter(pred_timestamp_nasdaq)

        predicted = record.get("predictions", [])
        symbol = record.get("symbol", "unknown")

        aligned_real = []
        aligned_pred = []

        # For each predicted point, find real data at 15 min increments from start_time
        for i, pred_point in enumerate(predicted):
            real_time = start_time + timedelta(minutes=15 * i)
            real_point = real_data_map.get(real_time)
            if real_point is None:
                # Missing real data point; skip or stop comparison here
                break
            aligned_real.append(real_point)
            aligned_pred.append(pred_point)

        if not aligned_real or not aligned_pred:
            continue  # no data to compare for this record

        # Extract numpy arrays for metrics calculation
        def extract_arr(key):
            return np.array([p[key] for p in aligned_real]), np.array([p[key] for p in aligned_pred])

        metrics = {}
        for key in ["open", "high", "low", "close"]:
            real_arr, pred_arr = extract_arr(key)
            mae = mean_absolute_error(real_arr, pred_arr)
            mse = mean_squared_error(real_arr, pred_arr)
            rmse = np.sqrt(mse)
            metrics[key] = {"mae": mae, "mse": mse, "rmse": rmse}

        metrics_record = {
            "symbol": symbol,
            "model_type": "price",
            "mae": mae,
            "mse": mse,
            "rmse": rmse
        }
        metrics_collection.insert_one(metrics_record)
        all_metrics.append(metrics_record)

    if not all_metrics:
        print("No metrics could be calculated - market is not open")
        raise HTTPException(status_code=404, detail="No metrics could be calculated - market is not open")

    return {"metrics": all_metrics}


@router.get("/metrics/price")
def get_latest_metrics():
    record = metrics_collection.find_one(
        {"symbol": "AAPL", "model_type": "price"}
    )
    if not record:
        raise HTTPException(status_code=404, detail="Metrics record not found")

    # Convert ObjectId and datetime fields to str if needed (optional)
    record["_id"] = str(record["_id"])
    if "calculated_at" in record:
        record["calculated_at"] = record["calculated_at"].isoformat()

    return record
