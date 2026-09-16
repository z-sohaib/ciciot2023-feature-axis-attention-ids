from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_fpr = {}
    total = matrix.sum()
    for idx, name in enumerate(class_names):
        fp = matrix[:, idx].sum() - matrix[idx, idx]
        tp = matrix[idx, idx]
        fn = matrix[idx, :].sum() - tp
        tn = total - tp - fp - fn
        per_class_fpr[name] = float(fp / (fp + tn)) if (fp + tn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "per_class_false_positive_rate": per_class_fpr,
        "confusion_matrix": matrix.tolist(),
    }
