
from __future__ import annotations
from typing import Iterable, Tuple, Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc

def _validate(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have same length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes for AUC metrics")
    return y_true, y_score

def roc_auc(y_true, y_score) -> float:
    y_true, y_score = _validate(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))

def pr_auc(y_true, y_score) -> float:
    y_true, y_score = _validate(y_true, y_score)
    return float(average_precision_score(y_true, y_score))

def top_k_precision(y_true, y_score, k: int) -> float:
    y_true, y_score = _validate(y_true, y_score)
    k = max(1, min(k, len(y_true)))
    idx = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[idx] == 1))

def top_fraction_precision(y_true, y_score, frac: float) -> float:
    y_true, y_score = _validate(y_true, y_score)
    frac = min(max(frac, 0.0), 1.0)
    k = max(1, int(round(frac * len(y_true))))
    return top_k_precision(y_true, y_score, k)

def enrichment_factor(y_true, y_score, fraction: float) -> float:
    """EF@fraction (e.g., 0.01 for 1%). EF = hits_in_top_fraction / expected_hits_random."""
    y_true, y_score = _validate(y_true, y_score)
    n = len(y_true)
    m = int(np.sum(y_true == 1))
    if m == 0:
        return float('nan')
    k = max(1, int(max(1, round(fraction * n))))
    top_idx = np.argsort(-y_score)[:k]
    hits = int(np.sum(y_true[top_idx] == 1))
    expected = m * (k / n)
    if expected == 0:
        return float('nan')
    return float(hits / expected)

def bedroc(y_true, y_score, alpha: float = 20.0) -> float:
    """BEDROC per Truchon & Bayly (2007), ranking by descending score."""
    y_true, y_score = _validate(y_true, y_score)
    n = len(y_true)
    m = np.sum(y_true == 1)
    if m == 0 or m == n:
        return float('nan')
    order = np.argsort(-y_score)
    ranks = np.arange(1, n+1)[order]
    pos_ranks = ranks[y_true[order] == 1]
    x = (pos_ranks - 0.5) / n
    s = np.sum(np.exp(-alpha * x))
    ka = alpha / (1 - np.exp(-alpha))
    m_float = float(m)
    bed = (s * ka / m_float - 1) / (np.exp(ka) - 1)
    return float(bed)

def riep(y_true, y_score, k: int) -> float:
    y_true, y_score = _validate(y_true, y_score)
    k = max(1, min(k, len(y_true)))
    order = np.argsort(-y_score)[:k]
    return float(np.sum(y_true[order] == 1) / max(1, np.sum(y_true == 1)))

def threshold_at_precision(y_true, y_score, target_precision: float) -> Tuple[float, float, float]:
    y_true, y_score = _validate(y_true, y_score)
    p, r, t = precision_recall_curve(y_true, y_score)
    idx = np.where(p[:-1] >= target_precision)[0]
    if len(idx) == 0:
        return float('nan'), float('nan'), float('nan')
    j = idx[0]
    return float(t[j]), float(p[j]), float(r[j])

def groupwise(y_true, y_score, groups: Iterable) -> Dict[str, float]:
    y_true, y_score = _validate(y_true, y_score)
    groups = np.asarray(list(groups))
    uniq = np.unique(groups)
    vals_roc, vals_pr = [], []
    for g in uniq:
        idx = groups == g
        y_g, s_g = y_true[idx], y_score[idx]
        if len(np.unique(y_g)) < 2:
            continue
        vals_roc.append(roc_auc(y_g, s_g))
        vals_pr.append(pr_auc(y_g, s_g))
    macro_roc = float(np.mean(vals_roc)) if len(vals_roc) else float('nan')
    macro_pr  = float(np.mean(vals_pr)) if len(vals_pr) else float('nan')
    micro_roc = roc_auc(y_true, y_score)
    micro_pr  = pr_auc(y_true, y_score)
    return {
        "roc_auc_macro": macro_roc,
        "pr_auc_macro": macro_pr,
        "roc_auc_micro": micro_roc,
        "pr_auc_micro": micro_pr,
    }
