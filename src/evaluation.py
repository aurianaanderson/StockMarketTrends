from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score, confusion_matrix, classification_report)

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

def plot_roc_curve(y_test, y_pred_probs):

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

    plt.figure(figsize=(6,4))

    plt.plot(fpr, tpr)

    plt.plot([0,1], [0,1], linestyle='--')

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve - Logistic Regression")

    plt.show()

    print("AUC Score:", roc_auc_score(y_test, y_pred_probs))


def evaluate_model(name, model, X_test, y_test):

   preds = model.predict(X_test)

   probs = model.predict_proba(X_test)[:, 1]

   acc = accuracy_score(y_test, preds)

   roc_auc = roc_auc_score(y_test, probs)

   print(f"\n{name} Accuracy: {acc:.4f}")

   print(f"{name} ROC-AUC: {roc_auc:.4f}")

   print(classification_report(y_test, preds))

   cm = confusion_matrix(y_test, preds)

   sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

   plt.title(name + " Confusion Matrix")

   plt.show()

   return {'accuracy': acc, 'roc_auc': roc_auc}