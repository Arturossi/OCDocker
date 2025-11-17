
#!/usr/bin/env python3

# Description
###############################################################################
'''
Core ranking metrics used across Analysis (ROC AUC, PR AUC, EF, BEDROC, etc.).

They are imported as:

from OCDocker.OCScore.Analysis.Metrics import Ranking as Rank
'''

# Imports
###############################################################################

from __future__ import annotations
from typing import Iterable, Tuple, Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc



# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Methods
###############################################################################


def _validate(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Validate arrays and coerce types; ensure both classes present.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Validated (y_true, y_score) as numpy arrays of type (int, float).
    '''

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.shape[0] != y_score.shape[0]:
        # User-facing error: mismatched array lengths
        ocerror.Error.value_error(f"y_true and y_score must have same length. Got y_true length {y_true.shape[0]}, y_score length {y_score.shape[0]}") # type: ignore
        raise ValueError("y_true and y_score must have same length")
    
    if len(np.unique(y_true)) < 2:
        # User-facing error: insufficient classes for AUC
        ocerror.Error.value_error(f"y_true must contain both classes for AUC metrics. Found {len(np.unique(y_true))} unique class(es)") # type: ignore
        raise ValueError("y_true must contain both classes for AUC metrics")
    
    return y_true, y_score


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    '''Compute ROC AUC with defensive validation.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).

    Returns
    -------
    float
        ROC AUC score (0.0 ~ 1.0).
    '''
    
    y_true, y_score = _validate(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    '''Compute average precision (area under PR curve).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).

    Returns
    -------
    float
        Average precision score (0.0 ~ 1.0).
    '''

    y_true, y_score = _validate(y_true, y_score)
    return float(average_precision_score(y_true, y_score))


def top_k_precision(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    '''Precision among the top-k scored samples (descending by score).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).
    k : int
        Number of top-scoring samples to consider.

    Returns
    -------
    float
        Precision among top-k (0.0 ~ 1.0).
    '''

    y_true, y_score = _validate(y_true, y_score)
    k = max(1, min(k, len(y_true)))
    idx = np.argsort(-y_score)[:k]

    return float(np.mean(y_true[idx] == 1))


def top_fraction_precision(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> float:
    '''Precision among the top fraction (e.g., 0.01 for top-1%).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).
    frac : float
        Fraction of top-scoring samples to consider (0.0 ~ 1.0).

    Returns
    -------
    float
        Precision among top fraction (0.0 ~ 1.0).
    '''

    y_true, y_score = _validate(y_true, y_score)
    frac = min(max(frac, 0.0), 1.0)
    k = max(1, int(round(frac * len(y_true))))

    return top_k_precision(y_true, y_score, k)


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    '''EF@fraction (e.g., 0.01 for 1%). EF = hits_in_top_fraction / expected_hits_random.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).
    fraction : float
        Fraction of top-scoring samples to consider (0.0 ~ 1.0).

    Returns
    -------
    float
        Enrichment factor (>= 0.0, or NaN if no positives).
    '''

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


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    '''BEDROC per Truchon & Bayly (2007), ranking by descending score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).
    alpha : float, optional
        Exponential weighting factor; higher = more early recognition. Default is 20.0.

    Returns
    -------
    float
        BEDROC score (0.0 ~ 1.0, or NaN if no positives).
    '''

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


def riep(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    '''Relative enrichment among the top-k versus total positives.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).
    k : int
        Number of top-scoring samples to consider.

    Returns
    -------
    float
        RIEP score (0.0 ~ 1.0, or NaN if no positives).
    '''

    y_true, y_score = _validate(y_true, y_score)
    k = max(1, min(k, len(y_true)))
    order = np.argsort(-y_score)[:k]

    return float(np.sum(y_true[order] == 1) / max(1, np.sum(y_true == 1)))


def threshold_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float) -> Tuple[float, float, float]:
    '''Find first threshold achieving at least the given precision.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).
    target_precision : float
        Desired precision level (0.0 ~ 1.0).

    Returns
    -------
    tuple(float, float, float)
        (threshold, precision, recall) at first point where precision >= target_precision,
        or (NaN, NaN, NaN) if target_precision not achievable.
    '''

    y_true, y_score = _validate(y_true, y_score)
    p, r, t = precision_recall_curve(y_true, y_score)
    idx = np.where(p[:-1] >= target_precision)[0]

    if len(idx) == 0:
        return float('nan'), float('nan'), float('nan')
    
    j = idx[0]

    return float(t[j]), float(p[j]), float(r[j])


def groupwise(y_true: np.ndarray, y_score: np.ndarray, groups: Iterable) -> Dict[str, float]:
    '''Compute macro/micro ROC/PR AUC across discrete groups.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).
    groups : Iterable
        Group labels for each sample (same length as y_true/y_score).

    Returns
    -------
    dict[str, float]
        Dictionary with keys "roc_auc_macro", "pr_auc_macro", "roc_auc_micro",
        "pr_auc_micro" and corresponding float values (or NaN if undefined).
    '''

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
