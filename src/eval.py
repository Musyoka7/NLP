"""Evaluation utilities — all models report through these functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV = PROJECT_ROOT / "results" / "metrics.csv"
LABELS = ["activist", "sceptic"]


def evaluate(model_name: str, y_true, y_pred, y_proba=None) -> dict:
    """Compute standard metrics for binary stance classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    auc = None
    if y_proba is not None:
        y_bin = (y_true == "sceptic").astype(int)
        auc = float(roc_auc_score(y_bin, y_proba))

    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_activist": float(p[0]),
        "recall_activist": float(r[0]),
        "f1_activist": float(f[0]),
        "precision_sceptic": float(p[1]),
        "recall_sceptic": float(r[1]),
        "f1_sceptic": float(f[1]),
        "roc_auc": auc,
    }
    return metrics, cm


def append_metrics(metrics: dict) -> pd.DataFrame:
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if METRICS_CSV.exists():
        df = pd.read_csv(METRICS_CSV)
        df = df[df["model"] != metrics["model"]]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    df.to_csv(METRICS_CSV, index=False)
    return df


def pretty_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0)
