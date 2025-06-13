from pymongo import MongoClient
from dotenv import load_dotenv
import os

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path, override=False) # If enviromental variables are not already set, it will get them from .env file
mongo_db_uri = os.getenv('MONGO_DB_URI', None) # Should be MLFlow tracking access token
if mongo_db_uri == None:
    raise ValueError("Enviroment variable MONGO_DB_URI must be set.")

client = MongoClient(mongo_db_uri)
db = client['stock-predictor']
