"""Evaluation helpers shared by the classifiers: plain NumPy, no scikit-learn."""

import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of predictions that match the true label."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels=None):
    """Counts of true label (row) against predicted label (column)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    index = {v: i for i, v in enumerate(labels)}
    C = len(labels)
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm, labels


def per_class_report(y_true, y_pred, labels=None):
    """Precision, recall, F1 and support for each class, plus macro F1."""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_n = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp),
                   where=denom > 0)
    return {"labels": labels, "precision": precision, "recall": recall,
            "f1": f1, "support": support,
            "macro_f1": float(f1.mean()),
            "accuracy": float(tp.sum() / max(cm.sum(), 1))}


def overall_competition_score(image_accuracy, sentiment_accuracy):
    """The competition metric: the plain mean of the two task accuracies."""
    return (image_accuracy + sentiment_accuracy) / 2.0
