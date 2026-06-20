#!/usr/bin/env python3

# Description
###############################################################################
'''
Core ranking metrics used across Analysis (ROC AUC, PR AUC, EF, BEDROC, etc.).

Usage:

from OCDocker.OCScore.Analysis.Metrics import Ranking as Rank
'''

# Imports
###############################################################################
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import OCDocker.Error as ocerror

SCREENING_CLASSIFICATION_METRICS = ("Precision", "Recall", "F1", "MCC")
SCREENING_CONFUSION_METRICS = ("TP", "FP", "TN", "FN")
DEFAULT_SCREENING_RANKING_METRICS = (
    "BEDROC",
    "ROC-AUC",
    "PR-AUC",
    "EF1%",
    "EF5%",
    "NDCG@1%",
    "NDCG@5%",
)
DEFAULT_SCREENING_COMPARISON_METRICS = (
    *DEFAULT_SCREENING_RANKING_METRICS,
    *SCREENING_CLASSIFICATION_METRICS,
    *SCREENING_CONFUSION_METRICS,
)


# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

DEFAULT_RANKING_SCORE_EPSILON = 1e-8

# Functions
###############################################################################
## Private ##

def _top_k_count(n: int, fraction: float) -> int:
    '''Return the number of top-ranked items for a list fraction.'''

    if n <= 0:
        return 0
    return max(1, int(max(1, round(fraction * n))))


def orient_scores(y_score: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    '''Orient scores so that larger values indicate a better active candidate.

    Ranking helpers in this module assume ``higher_is_better=True``. Classifier
    logits and probabilities should be passed unchanged. Lower-is-better docking
    scores must be negated before ranking.

    Parameters
    ----------
    y_score : np.ndarray
        Raw target scores.
    higher_is_better : bool, optional
        Whether larger raw scores indicate better actives, by default True.

    Returns
    -------
    np.ndarray
        Scores oriented for descending ranking.
    '''

    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if higher_is_better:
        return scores
    return -scores


def _validate(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Validate arrays and coerce types; ensure both classes present.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores where **larger values indicate better actives** after
        optional orientation via :func:`orient_scores`.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Validated (y_true, y_score) as numpy arrays of type (int, float).
    '''

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.shape[0] != y_score.shape[0]:
        # User-facing error: mismatched array lengths
        ocerror.Error.value_error(f"y_true and y_score must have same length. Got y_true length {y_true.shape[0]}, y_score length {y_score.shape[0]}")
        raise ValueError("y_true and y_score must have same length")

    if len(np.unique(y_true)) < 2:
        # User-facing error: insufficient classes for AUC
        ocerror.Error.value_error(f"y_true must contain both classes for AUC metrics. Found {len(np.unique(y_true))} unique class(es)")
        raise ValueError("y_true must contain both classes for AUC metrics")

    return y_true, y_score


def is_valid_ranking_scores(
        y_score: np.ndarray,
        epsilon: float = DEFAULT_RANKING_SCORE_EPSILON,
    ) -> bool:
    '''Return whether scores provide a meaningful ranking for early-enrichment metrics.

    Constant or near-constant scores are invalid because tie-breaking would depend
    on row order rather than model quality.

    Parameters
    ----------
    y_score : np.ndarray
        Oriented screening scores where larger is better.
    epsilon : float, optional
        Minimum standard deviation required for a valid ranking.

    Returns
    -------
    bool
        True when at least two unique finite scores exist and ``std >= epsilon``.
    '''

    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if scores.size == 0:
        return False
    if not np.all(np.isfinite(scores)):
        return False
    if len(np.unique(scores)) < 2:
        return False
    return float(np.std(scores)) >= float(epsilon)


def score_ranking_diagnostics(
        y_score: np.ndarray,
        epsilon: float = DEFAULT_RANKING_SCORE_EPSILON,
    ) -> Dict[str, float]:
    '''Summarize score distribution for ranking-metric validity checks.

    Parameters
    ----------
    y_score : np.ndarray
        Oriented screening scores.
    epsilon : float, optional
        Minimum standard deviation required for a valid ranking.

    Returns
    -------
    dict[str, float]
        Diagnostics including ``score_std``, ``n_unique_scores``, ``min_score``,
        ``max_score``, and ``ranking_valid``.
    '''

    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if scores.size == 0:
        return {
            "score_std": float("nan"),
            "n_unique_scores": 0.0,
            "min_score": float("nan"),
            "max_score": float("nan"),
            "ranking_valid": 0.0,
        }
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return {
            "score_std": float("nan"),
            "n_unique_scores": 0.0,
            "min_score": float("nan"),
            "max_score": float("nan"),
            "ranking_valid": 0.0,
        }
    return {
        "score_std": float(np.std(finite)),
        "n_unique_scores": float(len(np.unique(finite))),
        "min_score": float(np.min(finite)),
        "max_score": float(np.max(finite)),
        "ranking_valid": float(is_valid_ranking_scores(finite, epsilon=epsilon)),
    }


## Public ##

def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    '''BEDROC per Truchon & Bayly (2007), ranking by descending score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores where **larger values indicate better actives**.
    alpha : float, optional
        Exponential weighting factor; higher = more early recognition. Default is 20.0.

    Returns
    -------
    float
        BEDROC score (0.0 ~ 1.0, or NaN if no positives).
    '''

    return _finalize_bedroc(float(_bedroc_from_rie(y_true, y_score, alpha=alpha)))


def rie(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    '''Robust Initial Enhancement (RIE) for early recognition.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores where **larger values indicate better actives**.
    alpha : float, optional
        Exponential weighting factor, by default 20.0.

    Returns
    -------
    float
        RIE score, or NaN when there are no positives.
    '''

    y_true, y_score = _validate(y_true, y_score)
    if not is_valid_ranking_scores(y_score):
        return float("nan")
    n = len(y_true)
    m = int(np.sum(y_true == 1))
    if m == 0:
        return float('nan')

    order = np.argsort(-y_score)
    active_ranks = np.flatnonzero(y_true[order] == 1).astype(float) + 1.0
    observed = float(np.sum(np.exp(-alpha * active_ranks / n)))
    active_fraction = m / n
    expected = active_fraction * (1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0)
    return float(observed / expected)


def _bedroc_from_rie(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    '''Compute BEDROC from validated labels and oriented scores.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores where **larger values indicate better actives**.
    alpha : float, optional
        Exponential weighting factor, by default 20.0.

    Returns
    -------
    float
        BEDROC score on [0, 1], or NaN when undefined.
    '''

    y_true, y_score = _validate(y_true, y_score)
    m = int(np.sum(y_true == 1))
    n = len(y_true)
    if m == 0 or m == n:
        return float('nan')

    active_fraction = m / n
    inactive_fraction = 1.0 - active_fraction
    rie_score = rie(y_true, y_score, alpha=alpha)
    return float(
        rie_score * active_fraction * np.sinh(alpha / 2.0)
        / (np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * active_fraction))
        + 1.0 / (1.0 - np.exp(alpha * inactive_fraction))
    )


def _finalize_bedroc(value: float) -> float:
    '''Clamp tiny negative numerical noise around zero after a valid computation.

    Parameters
    ----------
    value : float
        Raw BEDROC value.

    Returns
    -------
    float
        Final BEDROC value, or NaN when the input is non-finite.
    '''

    if np.isnan(value) or np.isinf(value):
        return float("nan")
    if abs(value) < 1e-8:
        return 0.0
    return float(value)


def aggregate_group_metric(
        y_true: np.ndarray,
        y_score: np.ndarray,
        groups: Iterable,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        epsilon: float = DEFAULT_RANKING_SCORE_EPSILON,
    ) -> Tuple[float, int, int, Dict[str, int]]:
    '''Aggregate one ranking metric as the mean over valid groups.

    Groups with fewer than two classes, zero actives, zero decoys, or invalid
    constant/tied scores are skipped instead of contributing invalid values.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_score : np.ndarray
        Oriented scores where larger is better.
    groups : Iterable
        Group label per sample.
    metric_fn : Callable[[np.ndarray, np.ndarray], float]
        Metric function accepting ``(y_true, y_score)``.
    epsilon : float, optional
        Minimum within-group score standard deviation for ranking validity.

    Returns
    -------
    tuple[float, int, int, dict[str, int]]
        Mean metric across valid groups, number of valid groups, total unique
        groups observed, and per-reason skip counts.
    '''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    group_values = np.asarray(list(groups)).reshape(-1)
    if group_values.shape[0] != y_true.shape[0]:
        ocerror.Error.value_error(
            "groups must have the same length as y_true and y_score."
        )
        raise ValueError("groups must have the same length as y_true and y_score")

    unique_groups = np.unique(group_values)
    values: list[float] = []
    skip_counts = {
        "n_groups_invalid_one_class": 0,
        "n_groups_invalid_constant_score": 0,
        "n_groups_invalid_nonfinite": 0,
    }
    for group in unique_groups:
        mask = group_values == group
        labels = y_true[mask]
        group_scores = y_score[mask]
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        if positives == 0 or negatives == 0 or len(np.unique(labels)) < 2:
            skip_counts["n_groups_invalid_one_class"] += 1
            continue
        if not is_valid_ranking_scores(group_scores, epsilon=epsilon):
            skip_counts["n_groups_invalid_constant_score"] += 1
            continue
        try:
            value = float(metric_fn(labels, group_scores))
        except (ValueError, ZeroDivisionError):
            skip_counts["n_groups_invalid_nonfinite"] += 1
            continue
        if np.isnan(value) or np.isinf(value):
            skip_counts["n_groups_invalid_nonfinite"] += 1
            continue
        values.append(value)

    if not values:
        return float("nan"), 0, int(len(unique_groups)), skip_counts
    return float(np.mean(values)), int(len(values)), int(len(unique_groups)), skip_counts


def evaluate_screening_metrics_by_group(
        y_true: np.ndarray,
        y_score: np.ndarray,
        groups: Iterable,
        *,
        higher_is_better: bool = True,
        metric_names: Optional[Sequence[str]] = None,
        bedroc_alpha: float = 20.0,
    ) -> pd.DataFrame:
    '''Evaluate screening metrics separately for each receptor/group.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_score : np.ndarray
        Raw scores (oriented via ``higher_is_better``).
    groups : Iterable
        Group label per sample.
    higher_is_better : bool, optional
        Whether larger raw scores favor actives, by default True.
    metric_names : Sequence[str] | None, optional
        Metrics to compute per group. Defaults to BEDROC, ROC-AUC, PR-AUC,
        EF, and NDCG variants.
    bedroc_alpha : float, optional
        Exponential BEDROC weighting factor for per-group BEDROC, by default 20.0.

    Returns
    -------
    pandas.DataFrame
        One row per valid group with a ``group`` column and metric columns.
    '''

    if metric_names is None:
        metric_names = DEFAULT_SCREENING_COMPARISON_METRICS
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    oriented_scores = orient_scores(y_score, higher_is_better=higher_is_better)
    group_values = np.asarray(list(groups)).reshape(-1)
    if group_values.shape[0] != y_true.shape[0]:
        ocerror.Error.value_error(
            "groups must have the same length as y_true and y_score."
        )
        raise ValueError("groups must have the same length as y_true and y_score.")

    metric_fns = _screening_metric_functions(bedroc_alpha=bedroc_alpha)
    rows: list[dict[str, Any]] = []
    for group in np.unique(group_values):
        mask = group_values == group
        labels = y_true[mask]
        group_scores = oriented_scores[mask]
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        if positives == 0 or negatives == 0 or len(np.unique(labels)) < 2:
            continue
        if not is_valid_ranking_scores(group_scores):
            continue
        row: dict[str, Any] = {"group": str(group)}
        if "ROC-AUC" in metric_names:
            row["ROC-AUC"] = _safe_metric(
                roc_auc_score,
                labels,
                group_scores,
                default=float("nan"),
            )
        if "PR-AUC" in metric_names:
            row["PR-AUC"] = _safe_metric(
                average_precision_score,
                labels,
                group_scores,
                default=float("nan"),
            )
        for metric_name in metric_names:
            if metric_name in ("ROC-AUC", "PR-AUC"):
                continue
            metric_fn = metric_fns.get(metric_name)
            if metric_fn is None:
                continue
            row[metric_name] = _safe_metric(metric_fn, labels, group_scores, default=float("nan"))
        classification_keys = {
            *SCREENING_CLASSIFICATION_METRICS,
            *SCREENING_CONFUSION_METRICS,
            "classification_threshold",
        }
        if set(metric_names) & classification_keys:
            cls_metrics = classification_metrics_at_threshold(labels, group_scores)
            for key in metric_names:
                if key in cls_metrics:
                    row[key] = cls_metrics[key]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_scoring_functions_by_group(
        dataframe: pd.DataFrame,
        validation_indices: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        columns: Sequence[str],
        *,
        metric_names: Sequence[str],
        column_higher_is_better: Optional[Mapping[str, bool]] = None,
        bedroc_alpha: float = 20.0,
    ) -> pd.DataFrame:
    '''Evaluate scoring-function columns with per-group screening metrics.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Full reduced dataframe containing scoring-function columns.
    validation_indices : np.ndarray
        Row indices for the validation fold.
    labels : np.ndarray
        Binary labels aligned with ``dataframe`` rows.
    groups : np.ndarray
        Receptor/group labels for validation rows.
    columns : Sequence[str]
        Scoring-function column names.
    metric_names : Sequence[str]
        Metrics to retain in the output.
    column_higher_is_better : Mapping[str, bool] | None, optional
        Per-column score orientation. Missing columns default to ``True``.
    bedroc_alpha : float, optional
        Exponential BEDROC weighting factor for per-group BEDROC, by default 20.0.

    Returns
    -------
    pandas.DataFrame
        Long-format rows with ``group``, ``scorer``, ``scorer_type``, and metrics.
    '''

    val_idx = np.asarray(validation_indices, dtype=np.int64)
    y_val = np.asarray(labels, dtype=int).reshape(-1)[val_idx]
    g_val = np.asarray(groups).reshape(-1)[val_idx]
    frames: list[pd.DataFrame] = []
    orientations = column_higher_is_better or {}
    for column in columns:
        if column not in dataframe.columns:
            continue
        raw_scores = dataframe[column].to_numpy(dtype=float)[val_idx]
        if float(np.mean(np.isfinite(raw_scores))) <= 0.0:
            continue
        group_metrics = evaluate_screening_metrics_by_group(
            y_val,
            raw_scores,
            g_val,
            higher_is_better=bool(orientations.get(column, True)),
            metric_names=metric_names,
            bedroc_alpha=bedroc_alpha,
        )
        if group_metrics.empty:
            continue
        group_metrics = group_metrics.copy()
        group_metrics["scorer"] = column
        group_metrics["scorer_type"] = "sf"
        frames.append(group_metrics)
    if not frames:
        return pd.DataFrame(columns=["group"])
    return pd.concat(frames, ignore_index=True)


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    '''EF@fraction (e.g., 0.01 for 1%). EF = hits_in_top_fraction / expected_hits_random.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1 or boolean).
    y_score : np.ndarray
        Target scores where **larger values indicate better actives**.
    fraction : float
        Fraction of top-scoring samples to consider (0.0 ~ 1.0).

    Returns
    -------
    float
        Enrichment factor (>= 0.0, or NaN if no positives).
    '''

    y_true, y_score = _validate(y_true, y_score)
    if not is_valid_ranking_scores(y_score):
        return float("nan")
    n = len(y_true)
    m = int(np.sum(y_true == 1))

    if m == 0:
        return float('nan')

    k = _top_k_count(n, fraction)
    top_idx = np.argsort(-y_score)[:k]
    hits = int(np.sum(y_true[top_idx] == 1))
    expected = m * (k / n)

    if expected == 0:
        return float('nan')

    return float(hits / expected)


def ndcg_at_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    '''Compute NDCG at the top fraction of a ranked list.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Oriented screening scores where larger is better.
    fraction : float
        Top-ranked fraction to evaluate.

    Returns
    -------
    float
        NDCG score at the requested fraction, or NaN when ranking is invalid.
    '''

    y_true, y_score = _validate(y_true, y_score)
    if not is_valid_ranking_scores(y_score):
        return float("nan")
    k = _top_k_count(len(y_true), fraction)
    order = np.argsort(-y_score)
    gains = (2.0 ** y_true) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order][:k] * discounts))
    ideal_order = np.argsort(-y_true)
    idcg = float(np.sum(gains[ideal_order][:k] * discounts))
    if idcg == 0.0:
        return float("nan")
    return float(dcg / idcg)


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

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    groups = np.asarray(list(groups)).reshape(-1)
    if y_true.shape[0] != y_score.shape[0]:
        ocerror.Error.value_error(
            f"y_true and y_score must have same length. Got y_true length {y_true.shape[0]}, y_score length {y_score.shape[0]}"
        )
        raise ValueError("y_true and y_score must have same length")
    if groups.shape[0] != y_true.shape[0]:
        ocerror.Error.value_error(
            "groups must have the same length as y_true and y_score."
        )
        raise ValueError("groups must have the same length as y_true and y_score")
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc_macro": float("nan"),
            "pr_auc_macro": float("nan"),
            "roc_auc_micro": float("nan"),
            "pr_auc_micro": float("nan"),
        }

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


def _safe_metric(
        metric_fn: Callable[..., float],
        y_true: np.ndarray,
        y_score: np.ndarray,
        default: float = 0.0,
    ) -> float:
    '''Return ``metric_fn(y_true, y_score)`` or ``default`` when undefined.'''

    try:
        if len(np.unique(y_true)) < 2:
            return default
        value = float(metric_fn(y_true, y_score))
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def binary_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    '''Pick the score threshold that maximizes Youden's J (TPR - FPR).'''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if len(thresholds) == 0:
        return float("nan")
    j_scores = tpr - fpr
    return float(thresholds[int(np.argmax(j_scores))])


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    '''Return TP/FP/TN/FN counts for binary labels and predictions.'''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TP": float(tp), "FP": float(fp), "TN": float(tn), "FN": float(fn)}


def classification_metrics_at_threshold(
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: Optional[float] = None,
    ) -> dict[str, float]:
    '''Compute threshold-based classification metrics and confusion counts.

    When ``threshold`` is omitted, Youden's J on the ROC curve selects the cutoff.
    ``y_score`` must already be oriented so larger values favor actives.
    '''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    nan_result = {
        **{key: float("nan") for key in SCREENING_CLASSIFICATION_METRICS},
        **{key: float("nan") for key in SCREENING_CONFUSION_METRICS},
        "classification_threshold": float("nan"),
    }
    if len(np.unique(y_true)) < 2:
        return nan_result

    if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
        threshold = binary_threshold_youden(y_true, y_score)
    if np.isnan(threshold):
        return nan_result

    y_pred = (y_score >= threshold).astype(int)
    result = {
        "classification_threshold": float(threshold),
        **confusion_counts(y_true, y_pred),
    }
    result["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    result["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    result["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if len(np.unique(y_pred)) < 2:
        result["MCC"] = float("nan")
    else:
        result["MCC"] = float(matthews_corrcoef(y_true, y_pred))
    return result


def aggregate_group_classification_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        groups: Iterable,
        *,
        metric_keys: Sequence[str] = SCREENING_CLASSIFICATION_METRICS,
    ) -> Tuple[dict[str, float], int, int]:
    '''Macro-average classification metrics over groups with both classes.'''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    group_values = np.asarray(list(groups)).reshape(-1)
    if group_values.shape[0] != y_true.shape[0]:
        ocerror.Error.value_error(
            "groups must have the same length as y_true and y_score."
        )
        raise ValueError("groups must have the same length as y_true and y_score")

    unique_groups = np.unique(group_values)
    per_metric: dict[str, list[float]] = {key: [] for key in metric_keys}
    for group in unique_groups:
        mask = group_values == group
        labels = y_true[mask]
        group_scores = y_score[mask]
        if len(np.unique(labels)) < 2:
            continue
        cls_metrics = classification_metrics_at_threshold(labels, group_scores)
        for key in metric_keys:
            value = cls_metrics.get(key, float("nan"))
            if np.isfinite(value):
                per_metric[key].append(float(value))

    means = {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in per_metric.items()
    }
    used = max((len(values) for values in per_metric.values()), default=0)
    return means, int(used), int(len(unique_groups))


def _group_classification_metric_std(
        y_true: np.ndarray,
        y_score: np.ndarray,
        groups: np.ndarray,
        metric_key: str,
    ) -> float:
    group_values = np.asarray(groups).reshape(-1)
    per_group: list[float] = []
    for group in np.unique(group_values):
        mask = group_values == group
        labels = y_true[mask]
        group_scores = y_score[mask]
        if len(np.unique(labels)) < 2:
            continue
        value = classification_metrics_at_threshold(labels, group_scores).get(metric_key, float("nan"))
        if np.isfinite(value):
            per_group.append(float(value))
    if not per_group:
        return float("nan")
    return float(np.std(per_group, ddof=0))


def _apply_global_classification_metrics(metrics: dict[str, float], y_true: np.ndarray, y_score: np.ndarray) -> None:
    cls_global = classification_metrics_at_threshold(y_true, y_score)
    for key in (*SCREENING_CLASSIFICATION_METRICS, *SCREENING_CONFUSION_METRICS):
        metrics[f"{key}_global"] = cls_global[key]
    metrics["classification_threshold_global"] = cls_global["classification_threshold"]


def _promote_global_classification_metrics(metrics: dict[str, float]) -> None:
    for key in (*SCREENING_CLASSIFICATION_METRICS, *SCREENING_CONFUSION_METRICS):
        metrics[key] = metrics[f"{key}_global"]
    metrics["classification_threshold"] = metrics["classification_threshold_global"]


def _screening_metric_functions(
        bedroc_alpha: float = 20.0,
    ) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "BEDROC": lambda yt, ys: bedroc(yt, ys, alpha=bedroc_alpha),
        "EF1%": lambda yt, ys: enrichment_factor(yt, ys, 0.01),
        "EF5%": lambda yt, ys: enrichment_factor(yt, ys, 0.05),
        "NDCG@1%": lambda yt, ys: ndcg_at_fraction(yt, ys, 0.01),
        "NDCG@5%": lambda yt, ys: ndcg_at_fraction(yt, ys, 0.05),
    }


def _group_metric_std(
        y_true: np.ndarray,
        y_score: np.ndarray,
        groups: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> float:
    group_values = np.asarray(groups).reshape(-1)
    per_group: list[float] = []
    for group in np.unique(group_values):
        mask = group_values == group
        labels = y_true[mask]
        group_scores = y_score[mask]
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        if positives == 0 or negatives == 0 or len(np.unique(labels)) < 2:
            continue
        if not is_valid_ranking_scores(group_scores):
            continue
        try:
            value = float(metric_fn(labels, group_scores))
        except (ValueError, ZeroDivisionError):
            continue
        if np.isnan(value) or np.isinf(value):
            continue
        per_group.append(value)
    if not per_group:
        return float("nan")
    return float(np.std(per_group, ddof=0))


def evaluate_screening_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        groups: Optional[np.ndarray] = None,
        higher_is_better: bool = True,
        bedroc_alpha: float = 20.0,
    ) -> dict[str, float]:
    '''Evaluate DUDEz classification and early-recognition metrics.

    Classifier logits and probabilities should use ``higher_is_better=True``.
    Lower-is-better docking scores must set ``higher_is_better=False``.

    When ``groups`` is provided, BEDROC, EF, and NDCG are averaged across
    targets/receptors with both actives and decoys present.

    Parameters
    ----------
    bedroc_alpha : float, optional
        Exponential BEDROC weighting factor, by default 20.0.
    '''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    oriented_scores = orient_scores(y_score, higher_is_better=higher_is_better)
    score_diagnostics = score_ranking_diagnostics(oriented_scores)
    global_ranking_valid = bool(
        len(np.unique(y_true)) >= 2 and score_diagnostics["ranking_valid"] >= 1.0
    )
    metric_fns = _screening_metric_functions(bedroc_alpha=bedroc_alpha)
    metrics: dict[str, float] = {
        "ROC-AUC": _safe_metric(roc_auc_score, y_true, oriented_scores, default=float("nan")),
        "PR-AUC": _safe_metric(average_precision_score, y_true, oriented_scores, default=float("nan")),
        "score_std": score_diagnostics["score_std"],
        "n_unique_scores": score_diagnostics["n_unique_scores"],
        "min_score": score_diagnostics["min_score"],
        "max_score": score_diagnostics["max_score"],
        "ranking_metrics_valid": float(global_ranking_valid),
        "n_groups_invalid_constant_score": 0.0,
        "n_groups_invalid_one_class": 0.0,
        "n_groups_invalid_nonfinite": 0.0,
    }

    for metric_name, metric_fn in metric_fns.items():
        metrics[f"{metric_name}_global"] = _safe_metric(
            metric_fn,
            y_true,
            oriented_scores,
            default=float("nan"),
        )

    _apply_global_classification_metrics(metrics, y_true, oriented_scores)

    if groups is None:
        for metric_name in metric_fns:
            metrics[metric_name] = metrics[f"{metric_name}_global"]
        _promote_global_classification_metrics(metrics)
        metrics["n_groups_total"] = float("nan")
        metrics["n_groups_used"] = float("nan")
        return metrics

    group_array = np.asarray(groups).reshape(-1)
    grouped_auc = groupwise(y_true, oriented_scores, group_array)
    metrics["ROC-AUC_group_mean"] = grouped_auc["roc_auc_macro"]
    metrics["PR-AUC_group_mean"] = grouped_auc["pr_auc_macro"]

    n_groups_total = int(len(np.unique(group_array)))
    n_groups_used = 0
    group_skip_totals = {
        "n_groups_invalid_constant_score": 0,
        "n_groups_invalid_one_class": 0,
        "n_groups_invalid_nonfinite": 0,
    }
    for metric_name, metric_fn in metric_fns.items():
        mean_value, used_count, total_count, skip_counts = aggregate_group_metric(
            y_true,
            oriented_scores,
            group_array,
            metric_fn,
        )
        n_groups_total = max(n_groups_total, total_count)
        n_groups_used = max(n_groups_used, used_count)
        metrics[metric_name] = mean_value
        metrics[f"{metric_name}_group_std"] = _group_metric_std(
            y_true,
            oriented_scores,
            group_array,
            metric_fn,
        )
        for key, count in skip_counts.items():
            group_skip_totals[key] = max(group_skip_totals[key], int(count))

    group_cls_means, cls_groups_used, _ = aggregate_group_classification_metrics(
        y_true,
        oriented_scores,
        group_array,
    )
    for metric_key in SCREENING_CLASSIFICATION_METRICS:
        metrics[metric_key] = group_cls_means[metric_key]
        metrics[f"{metric_key}_group_std"] = _group_classification_metric_std(
            y_true,
            oriented_scores,
            group_array,
            metric_key,
        )
    for key in SCREENING_CONFUSION_METRICS:
        metrics[key] = metrics[f"{key}_global"]
    metrics["classification_threshold"] = metrics["classification_threshold_global"]

    metrics["n_groups_total"] = float(n_groups_total)
    metrics["n_groups_used"] = float(n_groups_used)
    metrics["n_groups_classification_used"] = float(cls_groups_used)
    metrics.update({key: float(value) for key, value in group_skip_totals.items()})
    metrics["ranking_metrics_valid"] = float(n_groups_used > 0 and global_ranking_valid)
    return metrics

