from fastapi import APIRouter
from datetime import datetime, timedelta, UTC
import os
import requests
import hashlib
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "../../.env")
load_dotenv(dotenv_path, override=False)

stock_api_key = os.getenv('STOCK_PRICE_API_KEY', None)
if stock_api_key is None:
    raise ValueError("Environment variable STOCK_PRICE_API_KEY must be set.")

router = APIRouter()
response_cache = {}

def date_weeks_ago(number_of_weeks):
    return datetime.now() - timedelta(weeks=number_of_weeks)

def round_to_15_minutes(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)

@router.get("/stock/apple")
def stock_price_apple():
    try:
        symbol = "AAPL"
        exchange = "NASDAQ"
        timezone = "Exchange"
        interval = "15min"
        weeks_ago = 1
        date_from = date_weeks_ago(weeks_ago).strftime("%Y-%m-%d")

        cache_key = hashlib.md5(round_to_15_minutes(datetime.now(UTC)).isoformat().encode()).hexdigest()
        if cache_key in response_cache:
            return {"source": "cache", "data": response_cache[cache_key]}

        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&exchange={exchange}&timezone={timezone}&interval={interval}&start_date={date_from}&apikey={stock_api_key}"

        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        response_cache[cache_key] = data

        return {"source": "api", "data": data}

    except Exception as e:
        return {"error": str(e)}
