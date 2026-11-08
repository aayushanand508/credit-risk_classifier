"""Shared metric helpers used by both the training script and the Streamlit app."""
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, y_proba):
    """Return the six assignment metrics (positive class = 1 = bad credit risk)."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }
