import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

#for EDA/early steps

def time_train_test_split(X, y=None, split_frac=0.8):

   split = int(len(X)*split_frac)

   X_train = X.iloc[:split].copy()

   X_test  = X.iloc[split:].copy()

   if y is None:

      return X_train, X_test

   y_train = y.iloc[:split].copy()

   y_test  = y.iloc[split:].copy()

   return X_train, X_test, y_train, y_test


def drop_missing_rows(data: pd.DataFrame) -> pd.DataFrame:

   return data.dropna().copy()


def direction_encoding(data: pd.DataFrame, col='Direction') -> pd.DataFrame:

   data = data.copy()

   data[col] = data[col].astype(str)

   data[col] = data[col].fillna('Down').map({'Up': 1, 'Down': 0})

   return data

def prepare_for_unsupervised(df, target_col="Direction", drop_cols=None):

   drop_cols = drop_cols or []

   X = df.drop(columns = drop_cols + [target_col], errors="ignore")

   y = df[target_col]

   X = X.select_dtypes(exclude=["datetime64[ns]"])

   X = X.select_dtypes(include=["number"])

   X_train, X_test, y_train, y_test = time_train_test_split(X, y)

   X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

   return (
        X_train_scaled

        , X_test_scaled

        , X_train

        , X_test

        , y_train

        , y_test
    )


   
#splitting into train and test sets for models


def make_time_split_xy(data, split_frac=0.8, target_column="Direction", drop_cols=None):

   drop_cols = drop_cols or []

   X = data.drop(columns = drop_cols + [target_column], errors="ignore")

   y = data[target_column]

   split = int(len(X) * split_frac)

   X_train = X.iloc[:split].copy()

   X_test = X.iloc[split:].copy()

   y_train = y.iloc[:split].copy()

   y_test = y.iloc[split:].copy()

   return X_train, X_test, y_train, y_test



def applying_pca(X_train_scaled, X_test_scaled, n_components=0.95):

   pca = PCA(n_components=n_components)

   X_train_pca = pca.fit_transform(X_train_scaled)

   X_test_pca = pca.transform(X_test_scaled)

   plt.figure(figsize=(6,4))

   plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')

   plt.xlabel("Number of CComponents")

   plt.ylabel("Cumulative Explained Variance")

   plt.title("PCA Explained Variance (Cumulative)")

   plt.show()

   return X_train_pca, X_test_pca, pca


def scale_data(X_train, X_test, scale = True):

   if not scale:

       return X_train, X_test

   scaler = StandardScaler()

   X_train_scaled_or = scaler.fit_transform(X_train)

   X_test_scaled_or = scaler.transform(X_test)

   X_train_scaled = pd.DataFrame(X_train_scaled_or, columns = X_train.columns, index = X_train.index)

   X_test_scaled = pd.DataFrame(X_test_scaled_or, columns=X_train.columns, index = X_test.index)


   return X_train_scaled, X_test_scaled



   