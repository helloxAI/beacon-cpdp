"""Evaluation metrics at the default 0.5 decision threshold."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def confusion_at(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5):
    pred = (prob >= threshold).astype(int)
    y = y_true.astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def evaluate(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=np.float64)
    tp, fp, tn, fn = confusion_at(y_true, prob, threshold)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    g_mean = float(np.sqrt(recall * specificity))
    auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    pred = (prob >= threshold).astype(int)
    mcc = float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0
    return {
        "AUC": auc,
        "MCC": mcc,
        "F1": float(f1),
        "Recall": float(recall),
        "G-Mean": g_mean,
    }


METRIC_NAMES = ["AUC", "MCC", "F1", "Recall", "G-Mean"]
