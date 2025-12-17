import numpy as np
import pandas as pd

def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:

   data = data.copy()

   data['Pct_Change'] = data['Close'].pct_change()

   data['MA_3'] = data['Close'].rolling(window=3).mean()

   data['MA_5'] = data['Close'].rolling(window=5).mean()

   data['MA_10'] = data['Close'].rolling(window=10).mean()

   data['Volatility'] = data['Close'].rolling(window=10).std()

   data['Direction'] = np.where(
       data['Close'].shift(-1) > data['Close'], 'Up', 'Down'
    )

   return data

def add_return_lags(data: pd.DataFrame, lags=None) -> pd.DataFrame:

   if lags is None:

      lags = [1, 2, 3, 5, 7]

   data = data.copy()

   data['return_1d'] = data['Close'].pct_change()

   for lag in lags:
      data[f'return_lag_{lag}d'] = data['return_1d'].shift(lag)

   return data



def add_rolling_volatility_10d(data: pd.DataFrame, window=10) -> pd.DataFrame:

   data = data.copy()

   data[f'rolling_vol_{window}d'] = data['return_1d'].rolling(window = window).std()

   return data



def add_ma_10d(data: pd.DataFrame, window=10) -> pd.DataFrame:

   data = data.copy()

   data[f'ma_return_{window}d'] = data['return_1d'].rolling(window=window).mean()

   return data



def sharpe_like_10d(data: pd.DataFrame, window=10) -> pd.DataFrame:

   data = data.copy()

   data[f'sharpe_like_{window}d']= data['return_1d'] / data[f'rolling_vol_{window}d']

   return data



def remove_non_modeling_columns(data: pd.DataFrame, cols_to_remove=['Date']) -> pd.DataFrame:

   data = data.drop(columns=cols_to_remove, errors='ignore')

   return data



def remove_low_variance_columns(data: pd.DataFrame, threshold=1e-5) -> pd.DataFrame:

   low_variance = data.var()[data.var() < threshold].index

   data = data.drop(columns=low_variance)
   
   return data



def calculate_corr(data: pd.DataFrame) -> pd.DataFrame:

   return data.corr()


def find_high_corr_pairs(data: pd.DataFrame, threshold=0.9) -> pd.DataFrame:

   correlation_matrix = data.corr(numeric_only=True)

   high_corr_pairs = correlation_matrix[(correlation_matrix > threshold) & (correlation_matrix != 1.0)]

   return high_corr_pairs


   
def find_low_variance_cols(data: pd.DataFrame, variance_threshold=1e-6) -> list:

   feature_std = data.std(numeric_only=True)

   low_variance_cols = feature_std[feature_std < variance_threshold].index.tolist()

   return low_variance_cols



def remove_low_variance_cols(data: pd.DataFrame, low_variance_cols: list) -> pd.DataFrame:

   data = data.drop(columns=low_variance_cols)

   return data





