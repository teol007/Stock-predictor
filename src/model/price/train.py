import os
import joblib
import random
import yaml

import numpy as np
import mlflow
import mlflow.tensorflow
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # If IDE shows that dependency couldn't be resolved, ignore it, because program executes ok
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import DatePreprocessor, SlidingWindowTransformer
import tf2onnx
import onnx

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

# Load configuration from YAML file
params = yaml.safe_load(open("params.yaml"))["train_price"]

dataset_filename = params["dataset_filename"]
target_cols = params["target_cols"]
test_size = params["test_size"]
random_state = params["random_state"]
window_size = params["window_size"]

# MLFlow init
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set PYTHONHASHSEED environment variable for reproducibility
os.environ["PYTHONHASHSEED"] = str(random_state)

# Set seeds for Python, NumPy, and TensorFlow
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

# Start MLflow experiment
mlflow.set_experiment("stock_predictor_train")
with mlflow.start_run(run_name="train_price"):
    # Log parameters
    mlflow.log_param("dataset_filename", dataset_filename)
    mlflow.log_param("target_cols", target_cols)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("window_size", window_size)
    mlflow.log_param("random_state", random_state)

    # Load the preprocessed data
    df = pd.read_csv(f"data/preprocessed/price/{dataset_filename}")
    df = df[["datetime", "open_price", "high_price", "low_price", "close_price", "volume"]]

    # Fill missing timestamps
    date_preprocessor = DatePreprocessor("datetime")
    df = date_preprocessor.fit_transform(df)
    df = df.drop(columns=["datetime"], axis=1)

    # Use test_size rows for testing and the rest for training
    df_test = df.iloc[-test_size:]
    df_train = df.iloc[:-test_size]

    # Preprocess the numeric features
    numeric_transformer = Pipeline([
        ("fillna", SimpleImputer(strategy="mean")),
        ("normalize", MinMaxScaler())
    ])

    preprocess = ColumnTransformer([
        ("numeric_transformer", numeric_transformer, target_cols),
    ])

    # Add the sliding window transformer to the pipeline
    sliding_window_transformer = SlidingWindowTransformer(window_size)

    # Create the pipeline
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("sliding_window_transformer", sliding_window_transformer),
    ])

    # Apply the pipeline to the dataframe
    X_train, y_train = pipeline.fit_transform(df_train)
    X_test, y_test = pipeline.transform(df_test)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Build and train tensorflow keras model

    # Define the model
    def build_model(input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1])) # As many output neurons as number of features we predict
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # Enable MLflow autologging for TensorFlow
    mlflow.tensorflow.autolog()

    # Train the model
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping]) #! Ephocs change
    
    # Make predictions
    y_pred = model.predict(X_test)
    print("y_pred:", y_pred[-1])
    print("y_test:", y_test[-1])

    # Invert the scaling
    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric_transformer"].named_steps["normalize"]
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)
    print("y_pred_inv:", y_pred_inv[-1])
    print("y_test_inv:", y_test_inv[-1])

    # Evaluate the model
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    print(f"Test MAE: {mae}")
    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {rmse}")

    # Log metrics
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_rmse", rmse)


    # Train the model on the entire dataset
    X_full, y_full = pipeline.fit_transform(df)
    model = build_model((X_full.shape[1], X_full.shape[2]))
    model.fit(X_full, y_full, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping]) #! Ephocs change
    
    # Make predictions
    y_pred_full = model.predict(X_full)
    print("y_pred_full:", y_pred_full[-1])
    print("y_full:", y_full[-1])

    # Invert the scaling
    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric_transformer"].named_steps["normalize"]
    y_pred_full_inv = scaler.inverse_transform(y_pred_full)
    y_full_inv = scaler.inverse_transform(y_full)
    print("y_pred_full_inv:", y_pred_full_inv[-1])
    print("y_full_inv:", y_full_inv[-1])

    # Evaluate the model on the entire dataset
    mse_full = mean_squared_error(y_full_inv, y_pred_full_inv)
    mae_full = mean_absolute_error(y_full_inv, y_pred_full_inv)
    rmse_full = np.sqrt(mse_full)
    print(f"Full dataset MAE: {mae_full}")
    print(f"Full dataset MSE: {mse_full}")
    print(f"Full dataset RMSE: {rmse_full}")

    # Log full dataset metrics
    mlflow.log_metric("full_mae", mae_full)
    mlflow.log_metric("full_mse", mse_full)
    mlflow.log_metric("full_rmse", rmse_full)

    # Save the model in keras format
    Path("models/price").mkdir(parents=True, exist_ok=True) # Create directories if don't exist
    dataset_name = Path(dataset_filename).stem # Get name without extension
    keras_model_path = f"models/price/model_price_{dataset_name}.keras"
    model.save(keras_model_path)
    mlflow.log_artifact(keras_model_path)
    
    # Save the model in ONNX format
    inputs = tf.keras.Input(shape=input_shape)
    outputs = model(inputs)
    model_functional_api = tf.keras.Model(inputs=inputs, outputs=outputs)

    onnx_model_path = f"models/price/model_price_{dataset_name}.onnx"
    input_signature = (tf.TensorSpec((None,) + input_shape, tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model_functional_api, input_signature=input_signature, opset=13)
    onnx.save(onnx_model, onnx_model_path)
    mlflow.log_artifact(onnx_model_path)

    # Save the pipeline
    pipeline_path = f"models/price/pipeline_price_{dataset_name}.pkl"
    joblib.dump(pipeline, pipeline_path)
    mlflow.log_artifact(pipeline_path)
