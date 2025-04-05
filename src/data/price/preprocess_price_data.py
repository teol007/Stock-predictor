import json
import numpy as np
import pandas as pd
import os

def preprocess_price_data():
    with open("data/raw/price/AAPL_price_data.json") as json_file:
        price_data_json = json.load(json_file)
    
    #print(price_data_json)
    print("Stock symbol:", price_data_json["meta"]["symbol"])
    print("Exchange:", price_data_json["meta"]["exchange"])
    print("Exchange timezone (datetime timezone):", price_data_json["meta"]["exchange_timezone"])
    print("Interval:", price_data_json["meta"]["interval"])
    print("Currency:", price_data_json["meta"]["currency"])
    print("Financial instrument type:", price_data_json["meta"]["type"])
    print("JSON data status:", price_data_json["status"])
    
    columns = ["datetime", "open_price", "high_price", "low_price", "close_price", "volume"]
    df = pd.DataFrame(columns=columns)
    
    existing_file_path = f"data/preprocessed/price/AAPL.csv"
    if os.path.exists(existing_file_path): # Check if csv file already exists
        df = pd.read_csv(existing_file_path)
    
    prices_json = price_data_json["values"]
    for price in prices_json:
        if price["datetime"] == "":
            raise ValueError("datetime string in JSON is empty")
        else:
            datetime = price["datetime"]
        
        open_price = price.get("open", np.nan)
        high_price = price.get("high", np.nan)
        low_price = price.get("low", np.nan)
        close_price = price.get("close", np.nan)
        volume = price.get("volume", np.nan)
        
        df = pd.concat([df, pd.DataFrame([[datetime, open_price, high_price, low_price, close_price, volume]], columns=columns)], ignore_index=True)
        
    df = df.drop_duplicates(subset=["datetime"]) # Filter unique "datetime" values
    df = df.sort_values(by="datetime")
    df = df.replace("", np.nan) # Replace empty string values
    
    file_path = "data/preprocessed/price/AAPL.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, sep=",", decimal=".", index=False)
    print(f"Preprocessing 'price' successful. Data saved to {file_path}")


if __name__ == "__main__":
    preprocess_price_data()
