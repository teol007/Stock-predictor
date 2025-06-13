# Model description: https://arxiv.org/abs/2207.01848

import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
import os
import random
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
from dotenv import load_dotenv
import joblib
from pathlib import Path

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

def classify_volume(volume: float) -> str:
    if volume <= 674598.2:
        return "Very Low Activity"
    elif volume <= 907964.2:
        return "Low Activity"
    elif volume <= 1241288.4:
        return "Moderate Activity"
    elif volume <= 1918822.2:
        return "High Activity"
    else:
        return "Very High Activity"

# Load configuration from YAML file
params = yaml.safe_load(open("params.yaml"))["train_price"]

dataset_filename = params["dataset_filename"]
test_size = params["test_size"]
random_state = params["random_state"]

os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)

# Engagement class labels
VOLUME_CLASSES = [
    "Very Low Activity",
    "Low Activity",     
    "Moderate Activity",
    "High Activity",    
    "Very High Activity"
]

# Load dataset
df = pd.read_csv(f"data/preprocessed/price/{dataset_filename}")
df = df[["open_price", "high_price", "low_price", "close_price", "volume"]]
df = df.tail(1000).reset_index(drop=True)

def assign_volume_class(v):
    if v <= 675000:
        return 0
    elif v <= 900000:
        return 1
    elif v <= 1250000:
        return 2
    elif v <= 1900000:
        return 3
    else:
        return 4

df["volume_class"] = df["volume"].apply(assign_volume_class)

# Prepare features and labels
X = df[["open_price", "high_price", "low_price", "close_price"]].values
y = df["volume_class"].values

# Start MLflow experiment
mlflow.set_experiment("activity_predictor_train")
with mlflow.start_run(run_name="train_activity"):
    mlflow.log_params({
        "dataset_filename": dataset_filename,
        "test_size": test_size,
        "random_state": random_state
    })
    
    # Enable MLflow autologging
    mlflow.sklearn.autolog()
    
    model = TabPFNClassifier(device="cpu")
    model.fit(X, y) # Additionally fit preffited model to my data

    # Predict
    pred = model.predict(X)
    pred_labels = [VOLUME_CLASSES[i] for i in pred]

    # Classification metrics
    accuracy = accuracy_score(y, pred)
    f1_macro = f1_score(y, pred, average="macro")
    precision_weighted = precision_score(y, pred, average="weighted")
    recall_weighted = recall_score(y, pred, average="weighted")
    
    # Print metrics
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted): {recall_weighted:.4f}")
    
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("precision_weighted", precision_weighted)
    mlflow.log_metric("recall_weighted", recall_weighted)

    # Save model
    Path("models/activity").mkdir(parents=True, exist_ok=True) # Create directories if don't exist
    dataset_name = os.path.splitext(dataset_filename)[0]
    model_path = f"models/activity/model_activity_{dataset_name}.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    