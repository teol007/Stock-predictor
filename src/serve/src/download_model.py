import mlflow
import joblib
import os
from tempfile import TemporaryDirectory
from dotenv import load_dotenv

import sys
import src.preprocess as real_preprocess
sys.modules['preprocess'] = real_preprocess

# Load enviromental variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path, override=False) # If enviromental variables are not already set, it will get them from .env file
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', None)
mlflow_tracking_username = os.getenv('MLFLOW_TRACKING_USERNAME', None)
mlflow_tracking_password = os.getenv('MLFLOW_TRACKING_PASSWORD', None) # Should be MLFlow tracking access token
if mlflow_tracking_uri == None:
    raise ValueError("Enviroment variable MLFLOW_TRACKING_URI must be set.")
if mlflow_tracking_username == None:
    raise ValueError("Enviroment variable MLFLOW_TRACKING_USERNAME must be set.")
if mlflow_tracking_password == None:
    raise ValueError("Enviroment variable MLFLOW_TRACKING_PASSWORD must be set.")

def load_registered_model_and_pipeline(
    model_name: str = "price_AAPL_model",
    pipeline_filename: str = "pipeline_price_AAPL.pkl",
    alias: str = "champion"
):
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(model_uri)

    pipeline_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipelines", pipeline_filename)
    pipeline = joblib.load(pipeline_filepath) if os.path.exists(pipeline_filepath) else None
    
    if pipeline is None:
        raise Exception(f"Pipeline could not be downloaded from: {pipeline_filepath}")

    return model, pipeline
