import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path, override=False) # If enviromental variables are not already set, it will get them from .env file


def date_weeks_ago(number_of_weeks):
    one_week_ago = datetime.now() - timedelta(weeks=number_of_weeks)
    return one_week_ago


def fetch_price_data():
    try:
        api_key = os.getenv('STOCK_PRICE_API_KEY', None)
        
        if api_key == None:
            raise ValueError("Enviroment variable STOCK_PRICE_API_KEY must be set.")
        
        symbol = "AAPL" # stock ticker for Apple inc.
        exchange = "NASDAQ"
        timezone = "Exchange" # Datetime will be in timezone that exchange has
        interval = "15min"
        date_from = date_weeks_ago(1).strftime("%Y-%m-%d") # Format date as string ("2025-03-15")
        
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&exchange={exchange}&timezone={timezone}&interval={interval}&start_date={date_from}&apikey={api_key}"

        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        file_path = "data/raw/price/AAPL_price_data.json"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"Fetching successful. Data saved to {file_path} at {datetime.now()}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while trying to fetch data: {http_err}")
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")


if __name__ == "__main__":
    fetch_price_data()
