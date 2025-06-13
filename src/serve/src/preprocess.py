import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert datetime to datetime
        X[self.col] = pd.to_datetime(X[self.col])
        # Sort by datetime
        X = X.sort_values(by=self.col)

        # Extract all dates without time
        X['date_only'] = X[self.col].dt.date
        

        # Add missing 15 minute data between 09:30 and 15:45 for each day
        last_datetime = X[self.col].max()
        all_expected_times = []
        for date in X['date_only'].unique():
            start = pd.Timestamp(f"{date} 09:30")
            end = pd.Timestamp(f"{date} 15:45")
            
            if end > last_datetime:
                end = last_datetime # Limit end to last available datetime in dataset
            
            times = pd.date_range(start=start, end=end, freq="15min")
            all_expected_times.extend(times)

        expected_df = pd.DataFrame(all_expected_times, columns=[self.col])

        # Merge with original data
        X = X.drop(columns=["date_only"])
        X = pd.merge(expected_df, X, on=self.col, how="left")
        return X

class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed, y_transformed = self.create_sliding_windows(X, self.window_size)
        return X_transformed, y_transformed

    @staticmethod
    def create_sliding_windows(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)
