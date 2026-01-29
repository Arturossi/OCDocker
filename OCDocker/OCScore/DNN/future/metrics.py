#!/usr/bin/env python3

# Description
###############################################################################
'''Metrics utilities for the future DNN pipeline.'''

# Imports
###############################################################################

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    auc,
    log_loss,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score
)

import OCDocker.OCScore.Analysis.Metrics.Ranking as ocrank

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

# Methods
###############################################################################
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


def compute_group_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_ids: np.ndarray,
        k_fractions: Tuple[float, float] = (0.01, 0.05)
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
    k_fractions : tuple[float, float], optional
        Fractions for top-k metrics, by default (0.01, 0.05).

    Returns
    -------
    Dict[str, float]
        Macro-averaged ranking metrics.
    '''

    # Macro-average across targets to avoid dominance by large groups.
    unique_targets = np.unique(target_ids)
    ef_1 = []
    ef_5 = []
    ndcg_1 = []
    ndcg_5 = []

    for tid in unique_targets:
        mask = target_ids == tid
        yt = y_true[mask]
        ys = y_score[mask]

        if yt.size == 0:
            continue

        # Compute per-target top-k cutoffs based on group size.
        k1 = max(1, int(round(k_fractions[0] * yt.size)))
        k2 = max(1, int(round(k_fractions[1] * yt.size)))

        # Enrichment factor uses a fraction-of-list cutoff.
        ef_1.append(ocrank.enrichment_factor(yt, ys, k_fractions[0]))
        ef_5.append(ocrank.enrichment_factor(yt, ys, k_fractions[1]))

        ndcg_1.append(ndcg_at_k(yt, ys, k1))
        ndcg_5.append(ndcg_at_k(yt, ys, k2))

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

    return {
        "EF@1%": _safe_mean(ef_1),
        "EF@5%": _safe_mean(ef_5),
        "NDCG@1%": _safe_mean(ndcg_1),
        "NDCG@5%": _safe_mean(ndcg_5)
    }


def compute_classification_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_ids: np.ndarray | None = None,
        k_fractions: Tuple[float, float] = (0.01, 0.05)
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
    k_fractions : tuple[float, float], optional
        Fractions for top-k metrics, by default (0.01, 0.05).

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

