#!/usr/bin/env python3

# Description
###############################################################################
'''
Simple row-wise baseline scores for OCScore comparisons.

Two families are supported:

- **Descriptor aggregates** (``descriptor_aggregate``): mean/median/max/min over
  all model input features (same matrix passed to the network, after export scaler).
- **SF consensus** (``sf_consensus``): mean/median/max/min over scoring-function
  columns only (Vina, Gnina, Smina, PLANTS, ODDT, …), matching historical
  ``SimpleConsensus`` semantics.
'''

# Imports
###############################################################################
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_screening_metrics
from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_screening_metrics_by_group

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

ROW_AGGREGATOR_NAMES = ("mean", "median", "max", "min")
DESCRIPTOR_AGGREGATE_SCORER_NAMES = ROW_AGGREGATOR_NAMES
DESCRIPTOR_AGGREGATE_SCORER_TYPE = "descriptor_aggregate"
SF_CONSENSUS_SCORER_TYPE = "sf_consensus"
DESCRIPTOR_AGGREGATE_NAME_PREFIX = "desc_"
SF_CONSENSUS_NAME_PREFIX = "sf_"


def format_descriptor_aggregate_scorer(aggregator: str) -> str:
    '''Return display key for a descriptor-row aggregate (e.g. ``desc_mean``).'''

    return f"{DESCRIPTOR_AGGREGATE_NAME_PREFIX}{aggregator}"


def format_sf_consensus_scorer(aggregator: str) -> str:
    '''Return display key for an SF-only row aggregate (e.g. ``sf_mean``).'''

    return f"{SF_CONSENSUS_NAME_PREFIX}{aggregator}"


def scorer_type_for_baseline_name(scorer: str) -> Optional[str]:
    '''Map a baseline scorer label to its ``scorer_type``, or None if not a baseline.'''

    if str(scorer).startswith(DESCRIPTOR_AGGREGATE_NAME_PREFIX):
        return DESCRIPTOR_AGGREGATE_SCORER_TYPE
    if str(scorer).startswith(SF_CONSENSUS_NAME_PREFIX):
        return SF_CONSENSUS_SCORER_TYPE
    if str(scorer) in ROW_AGGREGATOR_NAMES:
        return DESCRIPTOR_AGGREGATE_SCORER_TYPE
    return None


def row_aggregate_feature_scores(
        feature_matrix: np.ndarray,
        aggregators: Sequence[str] = ROW_AGGREGATOR_NAMES,
    ) -> dict[str, np.ndarray]:
    '''Aggregate each row of a feature matrix with simple reducers.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape ``(n_rows, n_features)``.
    aggregators : Sequence[str], optional
        Reducers to apply. Supported: ``mean``, ``median``, ``max``, ``min``.

    Returns
    -------
    dict[str, np.ndarray]
        One score vector per aggregator name (unprefixed keys).
    '''

    values = np.asarray(feature_matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {values.shape}.")
    reducers: dict[str, np.ndarray] = {
        "mean": np.nanmean(values, axis=1),
        "median": np.nanmedian(values, axis=1),
        "max": np.nanmax(values, axis=1),
        "min": np.nanmin(values, axis=1),
    }
    return {name: reducers[name] for name in aggregators if name in reducers}


def row_aggregate_sf_scores(
        dataframe: pd.DataFrame,
        scoring_columns: Sequence[str],
        row_indices: Optional[np.ndarray] = None,
        aggregators: Sequence[str] = ROW_AGGREGATOR_NAMES,
    ) -> dict[str, np.ndarray]:
    '''Aggregate each row across scoring-function columns only.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Reduced DUDEz (or full) table containing SF columns.
    scoring_columns : Sequence[str]
        Scoring-function column names (e.g. ``vina_vina``).
    row_indices : np.ndarray | None, optional
        Optional positional row subset (``iloc``), matching export split indices.
        When None, use all rows.
    aggregators : Sequence[str], optional
        Row reducers to apply.

    Returns
    -------
    dict[str, np.ndarray]
        One score vector per aggregator name (unprefixed keys).
    '''

    missing = [column for column in scoring_columns if column not in dataframe.columns]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Dataframe is missing scoring-function columns: {preview}{suffix}")
    if not scoring_columns:
        return {}

    if row_indices is not None:
        subset = dataframe.iloc[np.asarray(row_indices, dtype=np.int64)]
    else:
        subset = dataframe
    sf_matrix = subset[list(scoring_columns)].to_numpy(dtype=np.float64)
    return row_aggregate_feature_scores(sf_matrix, aggregators)


def _evaluate_named_row_aggregates_on_fold(
        aggregates: dict[str, np.ndarray],
        labels: np.ndarray,
        groups: Optional[np.ndarray],
        *,
        metric_names: Sequence[str],
        bedroc_alpha: float = 20.0,
        infer_higher_is_better: Callable[[np.ndarray, np.ndarray], bool],
        name_formatter: Callable[[str], str],
    ) -> dict[str, dict[str, float]]:
    y_val = np.asarray(labels, dtype=int).reshape(-1)
    g_val = None if groups is None else np.asarray(groups).reshape(-1)
    results: dict[str, dict[str, float]] = {}
    for aggregator, raw_scores in aggregates.items():
        if float(np.mean(np.isfinite(raw_scores))) <= 0.0:
            continue
        orientation = bool(infer_higher_is_better(raw_scores, y_val))
        metrics = evaluate_screening_metrics(
            y_val,
            raw_scores,
            groups=g_val,
            higher_is_better=orientation,
            bedroc_alpha=bedroc_alpha,
        )
        scorer_name = name_formatter(aggregator)
        results[scorer_name] = {
            metric_name: float(metrics[metric_name])
            for metric_name in metric_names
            if metric_name in metrics
        }
        results[scorer_name]["ranking_metrics_valid"] = float(metrics.get("ranking_metrics_valid", 0.0))
    return results


def _evaluate_named_row_aggregates_by_group(
        aggregates: dict[str, np.ndarray],
        labels: np.ndarray,
        groups: np.ndarray,
        *,
        metric_names: Sequence[str],
        bedroc_alpha: float = 20.0,
        infer_higher_is_better: Callable[[np.ndarray, np.ndarray], bool],
        name_formatter: Callable[[str], str],
    ) -> dict[str, pd.DataFrame]:
    y_val = np.asarray(labels, dtype=int).reshape(-1)
    g_val = np.asarray(groups).reshape(-1)
    frames: dict[str, pd.DataFrame] = {}
    for aggregator, raw_scores in aggregates.items():
        if float(np.mean(np.isfinite(raw_scores))) <= 0.0:
            continue
        group_df = evaluate_screening_metrics_by_group(
            y_val,
            raw_scores,
            g_val,
            higher_is_better=bool(infer_higher_is_better(raw_scores, y_val)),
            metric_names=metric_names,
            bedroc_alpha=bedroc_alpha,
        )
        if group_df.empty:
            continue
        frames[name_formatter(aggregator)] = group_df
    return frames


def evaluate_descriptor_aggregate_baselines_on_fold(
        feature_matrix: np.ndarray,
        validation_indices: np.ndarray,
        labels: np.ndarray,
        groups: Optional[np.ndarray],
        *,
        metric_names: Sequence[str],
        bedroc_alpha: float = 20.0,
        infer_higher_is_better: Callable[[np.ndarray, np.ndarray], bool],
        aggregators: Sequence[str] = ROW_AGGREGATOR_NAMES,
    ) -> dict[str, dict[str, float]]:
    '''Evaluate descriptor-row aggregates on one validation fold.'''

    val_idx = np.asarray(validation_indices, dtype=np.int64)
    X_val = np.asarray(feature_matrix, dtype=np.float64)[val_idx]
    aggregates = row_aggregate_feature_scores(X_val, aggregators)
    return _evaluate_named_row_aggregates_on_fold(
        aggregates,
        labels[val_idx],
        None if groups is None else groups[val_idx],
        metric_names=metric_names,
        bedroc_alpha=bedroc_alpha,
        infer_higher_is_better=infer_higher_is_better,
        name_formatter=format_descriptor_aggregate_scorer,
    )


def evaluate_descriptor_aggregates_by_group(
        feature_matrix: np.ndarray,
        validation_indices: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        *,
        metric_names: Sequence[str],
        bedroc_alpha: float = 20.0,
        infer_higher_is_better: Callable[[np.ndarray, np.ndarray], bool],
        aggregators: Sequence[str] = ROW_AGGREGATOR_NAMES,
    ) -> Mapping[str, pd.DataFrame]:
    '''Per-receptor metrics for descriptor-row aggregates.'''

    val_idx = np.asarray(validation_indices, dtype=np.int64)
    X_val = np.asarray(feature_matrix, dtype=np.float64)[val_idx]
    aggregates = row_aggregate_feature_scores(X_val, aggregators)
    return _evaluate_named_row_aggregates_by_group(
        aggregates,
        labels[val_idx],
        groups[val_idx],
        metric_names=metric_names,
        bedroc_alpha=bedroc_alpha,
        infer_higher_is_better=infer_higher_is_better,
        name_formatter=format_descriptor_aggregate_scorer,
    )


def evaluate_sf_consensus_baselines_on_fold(
        dataframe: pd.DataFrame,
        validation_indices: np.ndarray,
        scoring_columns: Sequence[str],
        labels: np.ndarray,
        groups: Optional[np.ndarray],
        *,
        metric_names: Sequence[str],
        bedroc_alpha: float = 20.0,
        infer_higher_is_better: Callable[[np.ndarray, np.ndarray], bool],
        aggregators: Sequence[str] = ROW_AGGREGATOR_NAMES,
    ) -> dict[str, dict[str, float]]:
    '''Evaluate SF-only row aggregates on one validation fold.'''

    val_idx = np.asarray(validation_indices, dtype=np.int64)
    aggregates = row_aggregate_sf_scores(dataframe, scoring_columns, val_idx, aggregators)
    return _evaluate_named_row_aggregates_on_fold(
        aggregates,
        labels[val_idx],
        None if groups is None else groups[val_idx],
        metric_names=metric_names,
        bedroc_alpha=bedroc_alpha,
        infer_higher_is_better=infer_higher_is_better,
        name_formatter=format_sf_consensus_scorer,
    )


def evaluate_sf_consensus_by_group(
        dataframe: pd.DataFrame,
        validation_indices: np.ndarray,
        scoring_columns: Sequence[str],
        labels: np.ndarray,
        groups: np.ndarray,
        *,
        metric_names: Sequence[str],
        bedroc_alpha: float = 20.0,
        infer_higher_is_better: Callable[[np.ndarray, np.ndarray], bool],
        aggregators: Sequence[str] = ROW_AGGREGATOR_NAMES,
    ) -> Mapping[str, pd.DataFrame]:
    '''Per-receptor metrics for SF-only row aggregates.'''

    val_idx = np.asarray(validation_indices, dtype=np.int64)
    aggregates = row_aggregate_sf_scores(dataframe, scoring_columns, val_idx, aggregators)
    return _evaluate_named_row_aggregates_by_group(
        aggregates,
        labels[val_idx],
        groups[val_idx],
        metric_names=metric_names,
        bedroc_alpha=bedroc_alpha,
        infer_higher_is_better=infer_higher_is_better,
        name_formatter=format_sf_consensus_scorer,
    )


__all__ = [
    "DESCRIPTOR_AGGREGATE_NAME_PREFIX",
    "DESCRIPTOR_AGGREGATE_SCORER_NAMES",
    "DESCRIPTOR_AGGREGATE_SCORER_TYPE",
    "ROW_AGGREGATOR_NAMES",
    "SF_CONSENSUS_NAME_PREFIX",
    "SF_CONSENSUS_SCORER_TYPE",
    "evaluate_descriptor_aggregate_baselines_on_fold",
    "evaluate_descriptor_aggregates_by_group",
    "evaluate_sf_consensus_baselines_on_fold",
    "evaluate_sf_consensus_by_group",
    "format_descriptor_aggregate_scorer",
    "format_sf_consensus_scorer",
    "row_aggregate_feature_scores",
    "row_aggregate_sf_scores",
    "scorer_type_for_baseline_name",
]
