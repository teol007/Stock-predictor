import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PriceFeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[["open_price", "high_price", "low_price", "close_price"]].values

class VolumeClassBinner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bins = [0, 675000, 900000, 1250000, 1900000, float('inf')]
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # transform volume to class labels
        return pd.cut(X["volume"], bins=self.bins, labels=False).astype(int).values
    def inverse_transform(self, y):
        # Convert back from class index to bin range (optional)
        return y  # or map back to labels as needed

class FullPipeline:
    def __init__(self, model):
        self.feature_selector = PriceFeatureSelector()
        self.label_binner = VolumeClassBinner()
        self.model = model
    
    def fit(self, df: pd.DataFrame):
        X = self.feature_selector.fit_transform(df)
        y = self.label_binner.fit_transform(df)
        self.model.fit(X, y)
        return self
    
    def predict(self, df: pd.DataFrame):
        X = self.feature_selector.transform(df)
        return self.model.predict(X)
    
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
