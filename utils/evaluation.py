"""Classification metrics shared by all team models (no model fitting)."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, num_classes: int) -> dict:
    """Return accuracy, a confusion matrix, and one metrics row per class.

    Rows of the confusion matrix are true classes; columns are predictions.
    Include every class, even if never predicted; undefined precision is zero.
    """
    labels = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "per_class": pd.DataFrame({
            "class": labels, "precision": precision, "recall": recall,
            "f1": f1, "support": support,
        }),
    }
