
from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc
from .Core import apply_basic_style, new_fig

def roc_plot(y_true, y_score, *, size=(6,4)) -> Tuple[object, object]:
    apply_basic_style()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = new_fig(size)
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0,1],[0,1], linestyle='--', linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return fig, ax

def pr_plot(y_true, y_score, *, size=(6,4)) -> Tuple[object, object]:
    apply_basic_style()
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = new_fig(size)
    ax.plot(r, p, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    return fig, ax

def enrichment_plot(y_true, y_score, fractions=(0.01, 0.02, 0.05, 0.1), *, size=(6,4)) -> Tuple[object, object]:
    apply_basic_style()
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    order = np.argsort(-y_score)
    cum_hits = np.cumsum((y_true[order] == 1).astype(int))
    xs = np.linspace(1/n, 1.0, n)
    fig, ax = new_fig(size)
    ax.plot(xs, cum_hits / max(1, cum_hits[-1]), label="Cumulative hits (norm)")
    for f in fractions:
        k = max(1, int(round(f*n)))
        ax.axvline(x=k/n, linestyle=':', linewidth=1)
    ax.set_xlabel("Top fraction of ranked list"); ax.set_ylabel("Normalized hits")
    ax.set_title("Enrichment Curve")
    ax.legend()
    return fig, ax
