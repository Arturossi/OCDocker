#!/usr/bin/env python3

# Description
###############################################################################
'''Metrics utilities for the future DNN pipeline.'''

# Imports
###############################################################################
from __future__ import annotations

import numpy as np

from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from typing import Dict, Iterable, List, Sequence, Tuple

import OCDocker.OCScore.Analysis.Metrics.Ranking as ocrank

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

def compute_classification_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_ids: np.ndarray | None = None,
        k_fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75)
    ) -> Dict[str, float]:
    '''Compute classification and ranking metrics for a labeled ranking dataset.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_score : np.ndarray
        Predicted scores.
    target_ids : np.ndarray | None, optional
        Target identifiers per sample, by default None.
    k_fractions : Sequence[float], optional
        Fractions for top-k metrics, by default (0.01, 0.05, 0.10, 0.25, 0.50, 0.75).

    Returns
    -------
    Dict[str, float]
        Metrics dictionary.
    '''

    metrics: Dict[str, float] = {}

    metrics["AUC"] = safe_auc(y_true, y_score)
    metrics["PR_AUC"] = safe_pr_auc(y_true, y_score)
    metrics["log_loss"] = safe_log_loss(y_true, y_score)
    metrics["pAUC@1%"] = partial_auc(y_true, y_score, max_fpr=0.01)
    metrics["pAUC@5%"] = partial_auc(y_true, y_score, max_fpr=0.05)

    if target_ids is not None:
        # Only compute group-based metrics when target ids are available.
        group_metrics = compute_group_metrics(y_true, y_score, target_ids, k_fractions)
        metrics.update(group_metrics)

    return metrics


def compute_group_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_ids: np.ndarray,
        k_fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75)
    ) -> Dict[str, float]:
    '''Compute ranking metrics per target and macro-average them.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_score : np.ndarray
        Predicted scores.
    target_ids : np.ndarray
        Target identifiers per sample.
    k_fractions : Sequence[float], optional
        Fractions for top-k metrics, by default (0.01, 0.05, 0.10, 0.25, 0.50, 0.75).

    Returns
    -------
    Dict[str, float]
        Macro-averaged ranking metrics.
    '''

    # Macro-average across targets to avoid dominance by large groups.
    unique_targets = np.unique(target_ids)
    fractions = tuple(float(f) for f in k_fractions)
    if len(fractions) == 0:
        return {}

    ef_by_frac: dict[float, list[float]] = {f: [] for f in fractions}
    ndcg_by_frac: dict[float, list[float]] = {f: [] for f in fractions}

    for tid in unique_targets:
        mask = target_ids == tid
        yt = y_true[mask]
        ys = y_score[mask]

        if yt.size == 0:
            continue

        # Enrichment factor is undefined if a group has a single class.
        if len(np.unique(yt)) < 2:
            for frac in fractions:
                ef_by_frac[frac].append(np.nan)
        else:
            # Enrichment factor uses a fraction-of-list cutoff.
            for frac in fractions:
                ef_by_frac[frac].append(ocrank.enrichment_factor(yt, ys, frac))

        for frac in fractions:
            k = max(1, int(round(frac * yt.size)))
            ndcg_by_frac[frac].append(ndcg_at_k(yt, ys, k))

    def _safe_mean(values: List[float]) -> float:
        '''Compute mean while ignoring NaNs.

        Parameters
        ----------
        values : list[float]
            Input values.

        Returns
        -------
        float
            Mean of non-NaN values or 0.0 if empty.
        '''

        vals = [v for v in values if not np.isnan(v)]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _label(frac: float) -> str:
        pct = frac * 100.0
        if abs(pct - round(pct)) < 1e-6:
            return f"{int(round(pct))}%"
        return f"{pct:.2f}%"

    metrics: Dict[str, float] = {}
    for frac in fractions:
        metrics[f"EF@{_label(frac)}"] = _safe_mean(ef_by_frac[frac])
    for frac in fractions:
        metrics[f"NDCG@{_label(frac)}"] = _safe_mean(ndcg_by_frac[frac])
    return metrics


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    '''Compute NDCG@k.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_score : np.ndarray
        Predicted scores.
    k : int
        Rank cutoff.

    Returns
    -------
    float
        NDCG@k value.
    '''

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.size == 0:
        return 0.0

    k = max(1, min(int(k), y_true.size))
    order = np.argsort(-y_score)
    # Standard DCG uses exponential gains and log2 discounts.
    gains = (2.0 ** y_true) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    dcg = np.sum(gains[order][:k] * discounts)

    ideal_order = np.argsort(-y_true)
    idcg = np.sum(gains[ideal_order][:k] * discounts)

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def partial_auc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.05) -> float:
    '''Compute partial AUC up to max_fpr.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_score : np.ndarray
        Predicted scores.
    max_fpr : float, optional
        Maximum false positive rate, by default 0.05.

    Returns
    -------
    float
        Normalized partial AUC.
    '''

    try:
        if len(np.unique(y_true)) < 2:
            return 0.0

        fpr, tpr, _ = roc_curve(y_true, y_score)
        # Ensure max_fpr within bounds
        max_fpr = max(1e-6, min(1.0, float(max_fpr)))

        # Add point at max_fpr by linear interpolation if needed.
        if max_fpr not in fpr:
            idx = np.searchsorted(fpr, max_fpr)
            if idx == 0:
                fpr = np.insert(fpr, 0, max_fpr)
                tpr = np.insert(tpr, 0, tpr[0])
            elif idx >= len(fpr):
                fpr = np.append(fpr, max_fpr)
                tpr = np.append(tpr, tpr[-1])
            else:
                fpr0, fpr1 = fpr[idx - 1], fpr[idx]
                tpr0, tpr1 = tpr[idx - 1], tpr[idx]
                tpr_interp = tpr0 + (tpr1 - tpr0) * (max_fpr - fpr0) / (fpr1 - fpr0 + 1e-8)
                fpr = np.insert(fpr, idx, max_fpr)
                tpr = np.insert(tpr, idx, tpr_interp)

        mask = fpr <= max_fpr
        p_auc = auc(fpr[mask], tpr[mask])

        # Normalize by max_fpr to keep scale 0-1
        return float(p_auc / max_fpr)
    except Exception:
        return 0.0


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    '''Compute ROC AUC with guards for degenerate labels.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_score : np.ndarray
        Predicted scores.

    Returns
    -------
    float
        ROC AUC value or 0.0 if undefined.
    '''

    try:
        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Guard against degenerate label distributions or numerical issues.
        return 0.0


def safe_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    '''Compute log-loss with guards for degenerate labels.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_score : np.ndarray
        Predicted scores.

    Returns
    -------
    float
        Log-loss value or inf if undefined.
    '''

    try:
        if len(np.unique(y_true)) < 2:
            return float("inf")
        return float(log_loss(y_true, y_score))
    except Exception:
        return float("inf")


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    '''Compute PR AUC with guards for degenerate labels.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_score : np.ndarray
        Predicted scores.

    Returns
    -------
    float
        PR AUC value or 0.0 if undefined.
    '''

    try:
        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0
