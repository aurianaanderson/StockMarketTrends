from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

import pandas as pd


def train_and_predict(X_train, X_test, y_train, y_test, scale=True):


   if isinstance(X_train, pd.DataFrame):

       print("X_train is a pandas DataFrame.")

   else:
       print("Warning: X_train is not a pandas DataFrame!")

   if isinstance(X_test, pd.DataFrame):
        pass

   X_train = X_train.select_dtypes(include=["number"]).copy()

   X_test = X_test.select_dtypes(include=["number"]).copy()

   X_train = X_train.dropna()

   X_test = X_test.dropna()

   y_train = y_train.loc[X_train.index]

   y_test = y_test.loc[X_test.index]

   if scale:

      scaler = StandardScaler()

      X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

      X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

   else:
      X_train_scaled = X_train

      X_test_scaled = X_test




   # Logistic Regression

   log_model = LogisticRegression(max_iter=500, solver='lbfgs')

   log_model.fit(X_train_scaled, y_train)

   log_preds = log_model.predict(X_test_scaled)




   # Random Forest

   rf_model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)

   rf_model.fit(X_train_scaled, y_train)

   rf_preds = rf_model.predict(X_test_scaled)


   return log_model, log_preds, rf_model, rf_preds, X_train_scaled, X_test_scaled