#!/usr/bin/env python3

# Description
###############################################################################
'''
K-fold cross-validation for exported OCScore PDBbind and DUDEz models.

Uses fixed hyperparameters from an exported ``best_model/`` bundle and retrains a
fresh model on each fold. DUDEz defaults to receptor-grouped folds so validation
receptors are never seen during training, matching staged screening evaluation.
'''

# Imports
###############################################################################
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import OCDocker.Toolbox.Logging as oclogging

from OCDocker.OCScore.Optimization import ModelExport as ocexport
from OCDocker.OCScore.Optimization.StagedOptuna import DUDEzScreeningModel
from OCDocker.OCScore.Optimization.StagedOptuna import FeatureExtractor
from OCDocker.OCScore.Optimization.StagedOptuna import PDBbindRegressionModel
from OCDocker.OCScore.Optimization.StagedOptuna import TabularDataset
from OCDocker.OCScore.Optimization.StagedOptuna import _apply_dae_noise
from OCDocker.OCScore.Optimization.StagedOptuna import _positive_class_weight
from OCDocker.OCScore.Optimization.StagedOptuna import _predict_regression
from OCDocker.OCScore.Optimization.StagedOptuna import _predict_screening
from OCDocker.OCScore.Optimization.StagedOptuna import _set_random_seed
from OCDocker.OCScore.Optimization.StagedOptuna import build_dudez_model
from OCDocker.OCScore.Optimization.StagedOptuna import build_pdbbind_model
from OCDocker.OCScore.Optimization.StagedOptuna import compute_regression_reconstruction_loss
from OCDocker.OCScore.Optimization.StagedOptuna import derive_dudez_labels
from OCDocker.OCScore.Optimization.StagedOptuna import evaluate_regression_metrics
from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_scoring_functions_by_group
from OCDocker.OCScore.Analysis.Metrics.Calibration import ProbabilityCalibrator
from OCDocker.OCScore.Analysis.Metrics.Calibration import merge_calibration_metrics
from OCDocker.OCScore.Analysis.Metrics.Ranking import DEFAULT_SCREENING_COMPARISON_METRICS
from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_screening_metrics
from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_screening_metrics_by_group
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import DESCRIPTOR_AGGREGATE_SCORER_TYPE
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import SF_CONSENSUS_SCORER_TYPE
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import evaluate_descriptor_aggregate_baselines_on_fold
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import evaluate_descriptor_aggregates_by_group
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import evaluate_sf_consensus_baselines_on_fold
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import evaluate_sf_consensus_by_group
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import scorer_type_for_baseline_name
from OCDocker.OCScore.Utils.FeatureReduction import DEFAULT_SCORING_PATTERNS
from OCDocker.OCScore.Utils.FeatureReduction import split_descriptor_blocks

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

LOGGER = oclogging.get_logger("ocscore.optimization.model_cross_validation")

CROSS_VALIDATION_STRATEGIES = ("auto", "receptor_grouped", "row_kfold")

PDBBIND_CV_METRICS = ("RMSE", "MAE", "Pearson r", "Spearman rho", "R2")
DUDEZ_CV_COMPARISON_METRICS = DEFAULT_SCREENING_COMPARISON_METRICS
DUDEZ_CV_METRICS = DUDEZ_CV_COMPARISON_METRICS
OCSCORE_MODEL_SCORER_NAME = "OCScore"


@dataclass
class CrossValidationConfig:
    """Configuration for exported-model cross-validation.

    Parameters
    ----------
    n_folds : int, optional
        Number of cross-validation folds, by default 5.
    epochs : int, optional
        Training epochs per fold, by default 100.
    random_seed : int, optional
        Seed for fold shuffling and weight initialization, by default 42.
    shuffle : bool, optional
        Shuffle fold assignments, by default True.
    strategy : str, optional
        ``auto``, ``receptor_grouped``, or ``row_kfold``. ``auto`` uses
        receptor-grouped folds for DUDEz when ``group_column`` is present.
    group_column : str, optional
        Receptor/target column for grouped DUDEz CV, by default ``"receptor"``.
    kind_column : str, optional
        DUDEz kind column for label derivation, by default ``"kind"``.
    include_scoring_function_baselines : bool, optional
        For DUDEz screening exports, also evaluate every post-filter scoring-function
        column on each fold's validation split, by default True.
    include_descriptor_aggregate_baselines : bool, optional
        For DUDEz screening exports, also evaluate row-wise mean/median/max/min over
        model input features on each validation fold (``desc_*`` scorers), by default True.
    include_sf_consensus_baselines : bool, optional
        For DUDEz screening exports, also evaluate row-wise mean/median/max/min across
        scoring-function columns only (``sf_*`` scorers), by default True.
    report_entity_overlap : bool, optional
        When True, record train/validation duplicate entity keys in fold diagnostics
        and log a warning, by default True.
    entity_columns : tuple of str, optional
        Columns checked for duplicate entities across train and validation splits.
    include_calibration_metrics : bool, optional
        Compute calibration metrics on validation folds, by default True.
    calibration_method : str, optional
        Calibration method name (e.g. ``"platt"``), by default ``"platt"``.
    """

    n_folds: int = 5
    epochs: int = 100
    random_seed: int = 42
    shuffle: bool = True
    strategy: str = "auto"
    group_column: str = "receptor"
    kind_column: str = "kind"
    include_scoring_function_baselines: bool = True
    include_descriptor_aggregate_baselines: bool = True
    include_sf_consensus_baselines: bool = True
    report_entity_overlap: bool = True
    entity_columns: tuple[str, ...] = ("name", "ligand_name", "smiles")
    include_calibration_metrics: bool = True
    calibration_method: str = "platt"


@dataclass
class CrossValidationFoldResult:
    """Metrics and metadata for one cross-validation fold.

    Parameters
    ----------
    fold_index : int
        Zero-based fold index.
    n_train : int
        Number of training rows in this fold.
    n_validation : int
        Number of validation rows in this fold.
    train_indices : list of int
        Row indices used for training.
    validation_indices : list of int
        Row indices used for validation.
    validation_metrics : dict
        Primary model metrics on the validation split.
    scoring_function_metrics : dict, optional
        Per-scoring-function metrics on validation, by default empty dict.
    per_target_metrics : list of dict, optional
        Per-target breakdown when applicable, by default empty list.
    diagnostics : dict, optional
        Fold-level integrity and overlap diagnostics, by default empty dict.
    """

    fold_index: int
    n_train: int
    n_validation: int
    train_indices: list[int]
    validation_indices: list[int]
    validation_metrics: dict[str, float]
    scoring_function_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    per_target_metrics: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossValidationResult:
    """Aggregated cross-validation output for one exported model.

    Parameters
    ----------
    export_dir : str
        Path to the exported model bundle directory.
    task : str
        Task type (e.g. regression or screening).
    n_folds : int
        Requested number of folds.
    effective_folds : int
        Folds actually evaluated after grouping constraints.
    strategy : str
        Split strategy used (``row_kfold`` or ``receptor_grouped``).
    epochs : int
        Training epochs per fold.
    random_seed : int
        Random seed used for splits and training.
    objective_metric : str
        Primary metric optimized during training.
    fold_results : list of CrossValidationFoldResult
        Per-fold metrics and diagnostics.
    aggregate_validation_metrics : dict
        Mean/std (or similar) aggregates across folds for the primary model.
    model_config : dict
        Serialized model configuration from the export bundle.
    scoring_function_columns : list of str, optional
        Scoring-function columns evaluated as baselines, by default empty list.
    aggregate_scoring_function_metrics : dict, optional
        Aggregated baseline metrics by scorer, by default empty dict.
    scorer_comparison_summary : dict, optional
        Summary comparing OCScore vs baselines, by default empty dict.
    diagnostics : dict, optional
        Run-level diagnostics, by default empty dict.
    """

    export_dir: str
    task: str
    n_folds: int
    effective_folds: int
    strategy: str
    epochs: int
    random_seed: int
    objective_metric: str
    fold_results: list[CrossValidationFoldResult]
    aggregate_validation_metrics: dict[str, dict[str, float]]
    model_config: dict[str, Any]
    scoring_function_columns: list[str] = field(default_factory=list)
    aggregate_scoring_function_metrics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    scorer_comparison_summary: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return str(value)


def _validate_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().lower()
    if normalized not in CROSS_VALIDATION_STRATEGIES:
        raise ValueError(
            f"Unsupported cross-validation strategy {strategy!r}. "
            f"Expected one of {CROSS_VALIDATION_STRATEGIES}."
        )
    return normalized


def _resolve_strategy(task: str, strategy: str, group_column: str, df: pd.DataFrame) -> str:
    normalized = _validate_strategy(strategy)
    if normalized != "auto":
        return normalized
    if task == "dudez_screening" and group_column in df.columns:
        return "receptor_grouped"
    return "row_kfold"


def _effective_fold_count(n_folds: int, n_units: int) -> int:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if n_units < 2:
        raise ValueError("At least two units (rows or groups) are required for cross-validation.")
    return min(int(n_folds), int(n_units))


def iter_row_kfold_indices(
        n_samples: int,
        n_folds: int,
        *,
        random_seed: int,
        shuffle: bool,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
    '''Return train/validation index pairs using row-wise K-fold.'''

    effective_folds = _effective_fold_count(n_folds, n_samples)
    splitter = KFold(
        n_splits=effective_folds,
        shuffle=shuffle,
        random_state=random_seed if shuffle else None,
    )
    indices = np.arange(n_samples, dtype=np.int64)
    return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(indices)]


def iter_receptor_group_kfold_indices(
        groups: np.ndarray,
        n_folds: int,
        *,
        random_seed: int,
        shuffle: bool,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
    '''Return train/validation index pairs by holding out whole receptor groups.'''

    group_array = np.asarray(groups).reshape(-1)
    unique_groups = np.unique(group_array)
    effective_folds = _effective_fold_count(n_folds, len(unique_groups))
    splitter = KFold(
        n_splits=effective_folds,
        shuffle=shuffle,
        random_state=random_seed if shuffle else None,
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(group_array), dtype=np.int64)
    for train_group_pos, val_group_pos in splitter.split(unique_groups):
        val_groups = unique_groups[val_group_pos]
        train_groups = unique_groups[train_group_pos]
        val_mask = np.isin(group_array, val_groups)
        train_mask = np.isin(group_array, train_groups)
        folds.append((all_indices[train_mask], all_indices[val_mask]))
    return folds


_ENTITY_OVERLAP_EXAMPLE_CAP = 5


def validate_fold_indices(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        *,
        fold_index: int,
    ) -> None:
    '''Assert train and validation row indices are disjoint.'''

    train_set = set(np.asarray(train_idx, dtype=np.int64).tolist())
    val_set = set(np.asarray(val_idx, dtype=np.int64).tolist())
    overlap = train_set & val_set
    if overlap:
        raise ValueError(
            f"Cross-validation fold {fold_index}: train and validation indices overlap "
            f"({len(overlap)} row(s))."
        )
    if not val_set:
        raise ValueError(f"Cross-validation fold {fold_index}: validation indices are empty.")


def validate_receptor_group_split(
        groups: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        *,
        fold_index: int,
    ) -> None:
    '''Assert held-out receptors do not appear in training rows.'''

    group_array = np.asarray(groups).reshape(-1)
    train_receptors = set(group_array[np.asarray(train_idx, dtype=np.int64)].astype(str))
    val_receptors = set(group_array[np.asarray(val_idx, dtype=np.int64)].astype(str))
    shared = train_receptors & val_receptors
    if shared:
        preview = sorted(shared)[:_ENTITY_OVERLAP_EXAMPLE_CAP]
        raise ValueError(
            f"Cross-validation fold {fold_index}: train and validation share receptor(s): "
            f"{preview}"
        )


def diagnose_entity_overlap(
        dataframe: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        entity_columns: Sequence[str],
    ) -> dict[str, Any]:
    '''Report duplicate entity keys between train and validation splits.'''

    train_positions = np.asarray(train_idx, dtype=np.int64)
    val_positions = np.asarray(val_idx, dtype=np.int64)
    overlap_report: dict[str, Any] = {}
    for column in entity_columns:
        if column not in dataframe.columns:
            continue
        train_values = set(
            dataframe[column].iloc[train_positions].dropna().astype(str).tolist()
        )
        val_values = set(dataframe[column].iloc[val_positions].dropna().astype(str).tolist())
        shared = train_values & val_values
        if not shared:
            continue
        examples = sorted(shared)[:_ENTITY_OVERLAP_EXAMPLE_CAP]
        overlap_report[column] = {"count": len(shared), "examples": examples}
    return overlap_report


def validate_fold_split(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        *,
        fold_index: int,
        strategy: str,
        groups: Optional[np.ndarray] = None,
    ) -> None:
    '''Run fold integrity checks before training.'''

    validate_fold_indices(train_idx, val_idx, fold_index=fold_index)
    if strategy == "receptor_grouped":
        if groups is None:
            raise ValueError(
                f"Cross-validation fold {fold_index}: receptor_grouped strategy requires groups."
            )
        validate_receptor_group_split(groups, train_idx, val_idx, fold_index=fold_index)


def _fit_transform_pdbbind_fold(
        X_all: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    '''Fit ``StandardScaler`` on training rows only and transform validation rows.'''

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[train_idx]).astype(np.float32)
    X_val = scaler.transform(X_all[val_idx]).astype(np.float32)
    return X_train, X_val, scaler


def _dataframe_rows_from_group_metrics(
        group_df: pd.DataFrame,
        *,
        fold_index: int,
        scorer: str,
        scorer_type: str,
        metric_names: Sequence[str],
    ) -> list[dict[str, Any]]:
    '''Convert per-group metric rows into serializable fold records.'''

    if group_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, record in group_df.iterrows():
        entry: dict[str, Any] = {
            "fold_index": int(fold_index),
            "group": str(record["group"]),
            "scorer": scorer,
            "scorer_type": scorer_type,
        }
        for metric_name in metric_names:
            if metric_name in record and pd.notna(record[metric_name]):
                entry[metric_name] = float(record[metric_name])
        rows.append(entry)
    return rows


def identify_scoring_function_columns(selected_features: Sequence[str]) -> list[str]:
    '''Return scoring-function descriptor columns from the selected feature list.'''

    blocks = split_descriptor_blocks(
        selected_features,
        scoring_patterns=DEFAULT_SCORING_PATTERNS,
        use_scoring_model_descriptors=True,
    )
    return list(blocks.scoring)


def infer_higher_is_better(scores: np.ndarray, labels: np.ndarray) -> bool:
    '''Infer whether larger raw scores favor actives on one validation fold.'''

    mask = np.isfinite(scores)
    if int(mask.sum()) < 10:
        return False
    active_mean = float(np.nanmean(scores[mask & (labels == 1)]))
    decoy_mean = float(np.nanmean(scores[mask & (labels == 0)]))
    if not np.isfinite(active_mean) or not np.isfinite(decoy_mean):
        return False
    return active_mean >= decoy_mean


def evaluate_scoring_function_baselines_on_fold(
        dataframe: pd.DataFrame,
        validation_indices: np.ndarray,
        labels: np.ndarray,
        groups: Optional[np.ndarray],
        scoring_columns: Sequence[str],
        *,
        metric_names: Sequence[str] = DUDEZ_CV_METRICS,
        bedroc_alpha: float = 20.0,
    ) -> dict[str, dict[str, float]]:
    '''Evaluate individual scoring functions on one CV validation fold.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full reduced DUDEz dataframe.
    validation_indices : np.ndarray
        Row indices for the validation fold.
    labels : np.ndarray
        Binary active/decoy labels aligned with ``dataframe`` rows.
    groups : np.ndarray | None
        Receptor groups for grouped screening metrics.
    scoring_columns : Sequence[str]
        Scoring-function columns to evaluate.
    metric_names : Sequence[str], optional
        Metrics retained in each returned scorer dictionary.
    bedroc_alpha : float, optional
        BEDROC alpha used for scorer baseline BEDROC, by default 20.0.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from scorer name to validation metrics on this fold.
    '''

    y_val = np.asarray(labels, dtype=int).reshape(-1)[validation_indices]
    g_val = None if groups is None else np.asarray(groups).reshape(-1)[validation_indices]
    results: dict[str, dict[str, float]] = {}
    for column in scoring_columns:
        raw_scores = dataframe[column].to_numpy(dtype=float)[validation_indices]
        if float(np.mean(np.isfinite(raw_scores))) <= 0.0:
            continue
        metrics = evaluate_screening_metrics(
            y_val,
            raw_scores,
            groups=g_val,
            higher_is_better=infer_higher_is_better(raw_scores, y_val),
            bedroc_alpha=bedroc_alpha,
        )
        results[column] = {
            metric_name: float(metrics[metric_name])
            for metric_name in metric_names
            if metric_name in metrics
        }
        results[column]["ranking_metrics_valid"] = float(metrics.get("ranking_metrics_valid", 0.0))
    return results


def _coerce_numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    '''Return only float-coercible entries from a metrics mapping.'''

    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            numeric[key] = float(value)
            continue
        try:
            numeric[key] = float(value)
        except (TypeError, ValueError):
            continue
    return numeric


def _aggregate_metric_dicts(
        fold_metrics: Sequence[Mapping[str, float]],
        metric_names: Sequence[str],
    ) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [
            float(metrics[metric_name])
            for metrics in fold_metrics
            if metric_name in metrics and np.isfinite(float(metrics[metric_name]))
        ]
        if not values:
            aggregated[metric_name] = {"mean": float("nan"), "std": float("nan"), "n_folds": 0.0}
            continue
        array = np.asarray(values, dtype=float)
        aggregated[metric_name] = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
            "n_folds": float(len(array)),
        }
    return aggregated


def _has_both_binary_classes(labels: np.ndarray) -> bool:
    values = np.asarray(labels, dtype=int).reshape(-1)
    return bool(len(np.unique(values)) >= 2)


def _merge_cv_calibration_metrics(
        val_metrics: dict[str, float],
        *,
        train_true: np.ndarray,
        train_score: np.ndarray,
        val_true: np.ndarray,
        val_score: np.ndarray,
        method: str,
    ) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "calibration_method": str(method),
        "calibration_fit_split": "train",
    }
    merge_calibration_metrics(
        val_metrics,
        val_true,
        val_score,
        calibrator=None,
        include_uncalibrated=True,
        include_calibrated=False,
    )
    if not _has_both_binary_classes(train_true):
        val_metrics["calibration_fit_valid"] = 0.0
        diagnostics["calibration_status"] = "skipped"
        diagnostics["calibration_skip_reason"] = "fit_split_one_class"
        diagnostics["calibration_fit_unique_classes"] = int(len(np.unique(np.asarray(train_true, dtype=int))))
        return diagnostics

    calibrator = ProbabilityCalibrator.fit(
        train_true,
        train_score,
        method=str(method),
        scores_are_logits=True,
    )
    merge_calibration_metrics(
        val_metrics,
        val_true,
        val_score,
        calibrator=calibrator,
        include_uncalibrated=False,
        include_calibrated=True,
    )
    val_metrics["calibration_fit_valid"] = 1.0
    diagnostics["calibration_status"] = "computed"
    return diagnostics


def _fold_scorer_metric_map(fold: CrossValidationFoldResult) -> dict[str, dict[str, float]]:
    '''Return validation metrics for OCScore and each scoring function on one fold.'''

    return {
        OCSCORE_MODEL_SCORER_NAME: dict(fold.validation_metrics),
        **{scorer: dict(metrics) for scorer, metrics in fold.scoring_function_metrics.items()},
    }


def _format_mean_pm_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(std):
        return f"{mean:.6g}"
    return f"{mean:.6g} ± {std:.6g}"


def build_scorer_comparison_summary(
        result: CrossValidationResult,
        *,
        comparison_metrics: Optional[Sequence[str]] = None,
        reference_scorer: str = OCSCORE_MODEL_SCORER_NAME,
    ) -> dict[str, Any]:
    '''Summarize OCScore vs scoring-function baselines across CV folds.

    Computes mean ± std per scorer/metric, per-fold rankings (1 = best), and how
    often ``reference_scorer`` achieves the top value on each fold.

    Parameters
    ----------
    result : CrossValidationResult
        Completed cross-validation output.
    comparison_metrics : Sequence[str] | None, optional
        Metrics to summarize. Defaults to :data:`DUDEZ_CV_COMPARISON_METRICS` for
        DUDEz screening and :data:`PDBBIND_CV_METRICS` for PDBbind regression.
    reference_scorer : str, optional
        Scorer used for win counting, by default ``OCScore``.

    Returns
    -------
    dict[str, Any]
        Summary tables: ``mean_std``, ``ocscore_wins``, ``fold_rankings``.
    '''

    if comparison_metrics is None:
        comparison_metrics = (
            DUDEZ_CV_COMPARISON_METRICS
            if result.task == "dudez_screening"
            else PDBBIND_CV_METRICS
        )
    metric_list = [str(metric) for metric in comparison_metrics]
    scorer_names = [OCSCORE_MODEL_SCORER_NAME, *result.scoring_function_columns]

    mean_std_rows: list[dict[str, Any]] = []
    for scorer in scorer_names:
        for metric_name in metric_list:
            values: list[float] = []
            for fold in result.fold_results:
                scorer_metrics = _fold_scorer_metric_map(fold).get(scorer, {})
                value = scorer_metrics.get(metric_name)
                if value is None:
                    continue
                numeric = float(value)
                if np.isfinite(numeric):
                    values.append(numeric)
            if not values:
                mean = float("nan")
                std = float("nan")
            else:
                array = np.asarray(values, dtype=float)
                mean = float(np.mean(array))
                std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
            mean_std_rows.append(
                {
                    "scorer": scorer,
                    "metric": metric_name,
                    "mean": mean,
                    "std": std,
                    "mean_pm_std": _format_mean_pm_std(mean, std),
                    "n_folds_with_value": len(values),
                }
            )

    win_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    n_effective_folds = len(result.fold_results)
    for metric_name in metric_list:
        wins = 0
        comparable_folds = 0
        for fold in result.fold_results:
            fold_values: dict[str, float] = {}
            for scorer, scorer_metrics in _fold_scorer_metric_map(fold).items():
                value = scorer_metrics.get(metric_name)
                if value is None:
                    continue
                numeric = float(value)
                if np.isfinite(numeric):
                    fold_values[scorer] = numeric
            if not fold_values:
                continue
            comparable_folds += 1
            ranked = sorted(fold_values.items(), key=lambda item: item[1], reverse=True)
            if ranked[0][0] == reference_scorer:
                wins += 1
            for rank, (scorer, value) in enumerate(ranked, start=1):
                ranking_rows.append(
                    {
                        "fold_index": fold.fold_index,
                        "metric": metric_name,
                        "rank": rank,
                        "scorer": scorer,
                        "value": value,
                        "n_scorers_ranked": len(ranked),
                    }
                )
        win_rows.append(
            {
                "metric": metric_name,
                "reference_scorer": reference_scorer,
                "n_folds_won": wins,
                "n_folds_compared": comparable_folds,
                "n_folds_total": n_effective_folds,
            }
        )

    return {
        "reference_scorer": reference_scorer,
        "comparison_metrics": metric_list,
        "scorers": scorer_names,
        "mean_std": mean_std_rows,
        "ocscore_wins": win_rows,
        "fold_rankings": ranking_rows,
    }


def _aggregate_scoring_function_metrics(
        fold_results: Sequence[CrossValidationFoldResult],
        scoring_columns: Sequence[str],
        metric_names: Sequence[str],
    ) -> dict[str, dict[str, dict[str, float]]]:
    '''Aggregate per-fold scoring-function metrics to mean/std per scorer.'''

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for column in scoring_columns:
        fold_metric_dicts = [
            fold.scoring_function_metrics[column]
            for fold in fold_results
            if column in fold.scoring_function_metrics
        ]
        aggregated[column] = _aggregate_metric_dicts(fold_metric_dicts, metric_names)
    return aggregated


def _train_pdbbind_fold(
        model_config: Mapping[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        device: torch.device,
        epochs: int,
        fold_seed: int,
    ) -> tuple[PDBbindRegressionModel, dict[str, float]]:
    _set_random_seed(fold_seed)
    params = dict(model_config)
    model = build_pdbbind_model(input_size=X_train.shape[1], params=params).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(params["optimizer_learning_rate"]),
        weight_decay=float(params["optimizer_weight_decay"]),
    )
    batch_size = int(params["optimizer_batch_size"])
    train_loader = DataLoader(
        TabularDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    if params.get("pdbbind_regression_loss") == "huber":
        regression_loss = nn.HuberLoss(delta=float(params["pdbbind_huber_delta"]))
    else:
        regression_loss = nn.MSELoss()
    reconstruction_loss = nn.MSELoss()

    best_state = None
    best_val_rmse = float("inf")
    for _epoch in range(int(epochs)):
        model.train()
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            dae_enabled = float(params["decoder_lambda_rec"]) > 0.0
            model_input = features
            if dae_enabled:
                model_input = _apply_dae_noise(
                    features,
                    noise_type=str(params.get("dae_noise_type", "none")),
                    mask_prob=float(params.get("dae_mask_prob", 0.0)),
                    gaussian_std=float(params.get("dae_gaussian_std", 0.0)),
                )
            outputs = model(model_input, return_reconstruction=dae_enabled)
            loss = compute_regression_reconstruction_loss(
                prediction=outputs["prediction"],
                target=target,
                reconstruction=outputs["reconstruction"],
                features=features,
                regression_loss=regression_loss,
                reconstruction_loss=reconstruction_loss,
                lambda_rec=float(params["decoder_lambda_rec"]),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_pred, val_true = _predict_regression(model, X_val, y_val, device)
        val_rmse = evaluate_regression_metrics(val_true, val_pred)["RMSE"]
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred, val_true = _predict_regression(model, X_val, y_val, device)
    return model, evaluate_regression_metrics(val_true, val_pred)


def _train_dudez_fold(
        model_config: Mapping[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_groups: Optional[np.ndarray],
        *,
        device: torch.device,
        epochs: int,
        fold_seed: int,
        transferred_extractor: Optional[FeatureExtractor],
        feature_extractor_architecture: Optional[Mapping[str, Any]],
        primary_metric: str,
        bedroc_alpha: float,
    ) -> tuple[DUDEzScreeningModel, dict[str, float]]:
    _set_random_seed(fold_seed)
    params = dict(model_config)
    model = build_dudez_model(
        input_size=X_train.shape[1],
        params=params,
        transferred_extractor=transferred_extractor,
        feature_extractor_architecture=dict(feature_extractor_architecture or {}),
    ).to(device)
    optimizer = optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(params["optimizer_learning_rate"]),
        weight_decay=float(params["optimizer_weight_decay"]),
    )
    batch_size = int(params["optimizer_batch_size"])
    train_loader = DataLoader(
        TabularDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    use_class_weighting = bool(params.get("dudez_use_class_weighting", False))
    pos_weight = _positive_class_weight(y_train, device) if use_class_weighting else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_metric = -float("inf")
    for _epoch in range(int(epochs)):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_score, val_true = _predict_screening(model, X_val, y_val, device)
        val_metrics = evaluate_screening_metrics(
            val_true,
            val_score,
            groups=val_groups,
            higher_is_better=True,
            bedroc_alpha=bedroc_alpha,
        )
        metric_value = float(val_metrics.get(primary_metric, float("nan")))
        if np.isfinite(metric_value) and metric_value > best_metric:
            best_metric = metric_value
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    val_score, val_true = _predict_screening(model, X_val, y_val, device)
    val_metrics = evaluate_screening_metrics(
        val_true,
        val_score,
        groups=val_groups,
        higher_is_better=True,
        bedroc_alpha=bedroc_alpha,
    )
    return model, val_metrics


def run_cross_validation_from_export(
        export_dir: str | Path,
        dataframe: pd.DataFrame,
        *,
        config: Optional[CrossValidationConfig] = None,
        device: Optional[torch.device | str] = None,
        output_dir: Optional[str | Path] = None,
        transferred_extractor: Optional[FeatureExtractor] = None,
    ) -> CrossValidationResult:
    '''Run K-fold cross-validation using a fixed exported model configuration.

    Parameters
    ----------
    export_dir : str | Path
        Exported ``best_model/`` directory.
    dataframe : pd.DataFrame
        Reduced PDBbind or DUDEz dataframe aligned with the export features.
    config : CrossValidationConfig | None, optional
        Cross-validation settings including ``n_folds``.
    device : torch.device | str | None, optional
        Training device, by default CPU.
    output_dir : str | Path | None, optional
        Directory for ``cross_validation_results.json`` and fold CSV. Defaults
        to ``<export_dir>/cross_validation``.
    transferred_extractor : FeatureExtractor | None, optional
        Optional transferred extractor for DUDEz transfer exports.

    Returns
    -------
    CrossValidationResult
        Per-fold and aggregate validation metrics.
    '''

    cv_config = config or CrossValidationConfig()
    export_path = Path(export_dir)
    bundle = ocexport.load_exported_model(export_path, device=device or "cpu")
    retrain_config = bundle["retrain_config"]
    task = str(retrain_config["task"])
    selected_features = list(bundle["selected_features"])
    model_config = dict(retrain_config["resolved_model_config"])
    stage_config = dict(retrain_config.get("stage_config", {}))
    objective_metric = str(retrain_config.get("objective_metric", "RMSE"))
    resolved_device = bundle["device"]
    bedroc_alpha = float(stage_config.get("bedroc_alpha", 20.0))

    strategy = _resolve_strategy(task, cv_config.strategy, cv_config.group_column, dataframe)
    missing_features = [column for column in selected_features if column not in dataframe.columns]
    if missing_features:
        raise ValueError(f"Dataframe is missing exported features: {missing_features[:5]}")

    X_all = dataframe[selected_features].to_numpy(dtype=np.float32)
    if task == "pdbbind_regression":
        target_column = str(
            (retrain_config.get("split_config") or {}).get("target_column", "experimental")
        )
        if target_column not in dataframe.columns:
            raise ValueError(f"PDBbind target column is missing: {target_column!r}")
        y_all = dataframe[target_column].to_numpy(dtype=np.float32)
        fold_pairs = iter_row_kfold_indices(
            len(dataframe),
            cv_config.n_folds,
            random_seed=cv_config.random_seed,
            shuffle=cv_config.shuffle,
        )
        metric_names = PDBBIND_CV_METRICS
        groups_all = None
        scoring_function_columns: list[str] = []
        feature_extractor_architecture = None
    elif task == "dudez_screening":
        y_all = derive_dudez_labels(dataframe, kind_column=cv_config.kind_column)
        groups_all = (
            dataframe[cv_config.group_column].to_numpy()
            if cv_config.group_column in dataframe.columns
            else None
        )
        if strategy == "receptor_grouped":
            if groups_all is None:
                raise ValueError(
                    f"Receptor-grouped cross-validation requires column {cv_config.group_column!r}."
                )
            fold_pairs = iter_receptor_group_kfold_indices(
                groups_all,
                cv_config.n_folds,
                random_seed=cv_config.random_seed,
                shuffle=cv_config.shuffle,
            )
        else:
            fold_pairs = iter_row_kfold_indices(
                len(dataframe),
                cv_config.n_folds,
                random_seed=cv_config.random_seed,
                shuffle=cv_config.shuffle,
            )
        metric_names = DUDEZ_CV_METRICS
        if transferred_extractor is None and bool(model_config.get("dudez_use_transfer", True)):
            extra = retrain_config.get("extra") or {}
            pdbbind_export = extra.get("pdbbind_best_model_export_dir")
            if pdbbind_export:
                transferred_extractor = ocexport.load_exported_model(
                    Path(pdbbind_export),
                    device=resolved_device,
                )["model"].feature_extractor
        feature_extractor_architecture = bundle["architecture"].get("feature_extractor")
        scoring_function_columns: list[str] = []
        if cv_config.include_scoring_function_baselines:
            scoring_function_columns = identify_scoring_function_columns(selected_features)
            if not scoring_function_columns:
                LOGGER.warning(
                    "No scoring-function columns found among selected features; skipping SF baselines."
                )
    else:
        raise ValueError(f"Unsupported export task for cross-validation: {task}")

    fold_results: list[CrossValidationFoldResult] = []
    for fold_index, (train_idx, val_idx) in enumerate(fold_pairs):
        validate_fold_split(
            train_idx,
            val_idx,
            fold_index=fold_index,
            strategy=strategy,
            groups=groups_all,
        )
        fold_seed = int(cv_config.random_seed) + fold_index
        diagnostics: dict[str, Any] = {}
        if cv_config.report_entity_overlap:
            entity_overlap = diagnose_entity_overlap(
                dataframe,
                train_idx,
                val_idx,
                cv_config.entity_columns,
            )
            if entity_overlap:
                diagnostics["entity_overlap"] = entity_overlap
                LOGGER.warning(
                    "Cross-validation fold %s: entity overlap between train and validation: %s",
                    fold_index,
                    entity_overlap,
                )

        per_target_metrics: list[dict[str, Any]] = []
        if task == "pdbbind_regression":
            LOGGER.info(
                "CV fold %s: scaler fit on n_train=%s rows, transform on n_val=%s rows",
                fold_index,
                len(train_idx),
                len(val_idx),
            )
            X_train, X_val, _scaler = _fit_transform_pdbbind_fold(X_all, train_idx, val_idx)
            _, val_metrics = _train_pdbbind_fold(
                model_config,
                X_train,
                y_all[train_idx],
                X_val,
                y_all[val_idx],
                device=resolved_device,
                epochs=cv_config.epochs,
                fold_seed=fold_seed,
            )
            diagnostics["scaler"] = "StandardScaler"
            diagnostics["scaler_fit_n_train"] = int(len(train_idx))
            diagnostics["scaler_transform_n_val"] = int(len(val_idx))
        else:
            X_train = X_all[train_idx]
            X_val = X_all[val_idx]
            val_groups = None if groups_all is None else groups_all[val_idx]
            model, val_metrics = _train_dudez_fold(
                model_config,
                X_train,
                y_all[train_idx],
                X_val,
                y_all[val_idx],
                val_groups,
                device=resolved_device,
                epochs=cv_config.epochs,
                fold_seed=fold_seed,
                transferred_extractor=transferred_extractor,
                feature_extractor_architecture=feature_extractor_architecture,
                primary_metric=objective_metric,
                    bedroc_alpha=bedroc_alpha,
            )
            val_score: Optional[np.ndarray] = None
            val_true: Optional[np.ndarray] = None
            if cv_config.include_calibration_metrics or groups_all is not None:
                val_score, val_true = _predict_screening(
                    model,
                    X_val,
                    y_all[val_idx],
                    resolved_device,
                )
            if cv_config.include_calibration_metrics and val_score is not None and val_true is not None:
                train_score, train_true = _predict_screening(
                    model,
                    X_train,
                    y_all[train_idx],
                    resolved_device,
                )
                calibration_diagnostics = _merge_cv_calibration_metrics(
                    val_metrics,
                    train_true=train_true,
                    train_score=train_score,
                    val_true=val_true,
                    val_score=val_score,
                    method=str(cv_config.calibration_method),
                )
                diagnostics.update(calibration_diagnostics)
                if calibration_diagnostics.get("calibration_status") == "skipped":
                    LOGGER.warning(
                        "Cross-validation fold %s: calibration skipped (%s).",
                        fold_index,
                        calibration_diagnostics.get("calibration_skip_reason"),
                    )
            if groups_all is not None and val_score is not None and val_true is not None:
                diagnostics["validation_receptors"] = sorted(
                    np.unique(groups_all[val_idx]).astype(str).tolist()
                )
                oc_group_df = evaluate_screening_metrics_by_group(
                    val_true,
                    val_score,
                    val_groups,
                    higher_is_better=True,
                    metric_names=metric_names,
                    bedroc_alpha=bedroc_alpha,
                )
                per_target_metrics.extend(
                    _dataframe_rows_from_group_metrics(
                        oc_group_df,
                        fold_index=fold_index,
                        scorer=OCSCORE_MODEL_SCORER_NAME,
                        scorer_type="model",
                        metric_names=metric_names,
                    )
                )

        sf_metrics: dict[str, dict[str, float]] = {}
        if task == "dudez_screening" and scoring_function_columns:
            sf_metrics = evaluate_scoring_function_baselines_on_fold(
                dataframe,
                val_idx,
                y_all,
                groups_all,
                scoring_function_columns,
                metric_names=metric_names,
                bedroc_alpha=bedroc_alpha,
            )
            if groups_all is not None:
                column_orientations = {
                    column: infer_higher_is_better(
                        dataframe[column].to_numpy(dtype=float)[val_idx],
                        y_all[val_idx],
                    )
                    for column in scoring_function_columns
                }
                sf_group_df = evaluate_scoring_functions_by_group(
                    dataframe,
                    val_idx,
                    y_all,
                    groups_all,
                    scoring_function_columns,
                    metric_names=metric_names,
                    column_higher_is_better=column_orientations,
                    bedroc_alpha=bedroc_alpha,
                )
                if not sf_group_df.empty:
                    for column in scoring_function_columns:
                        column_df = sf_group_df.loc[sf_group_df["scorer"] == column]
                        per_target_metrics.extend(
                            _dataframe_rows_from_group_metrics(
                                column_df,
                                fold_index=fold_index,
                                scorer=column,
                                scorer_type="sf",
                                metric_names=metric_names,
                            )
                        )

        if task == "dudez_screening" and cv_config.include_descriptor_aggregate_baselines:
            desc_metrics = evaluate_descriptor_aggregate_baselines_on_fold(
                X_all,
                val_idx,
                y_all,
                groups_all,
                metric_names=metric_names,
                bedroc_alpha=bedroc_alpha,
                infer_higher_is_better=infer_higher_is_better,
            )
            sf_metrics.update(desc_metrics)
            if groups_all is not None:
                desc_group_frames = evaluate_descriptor_aggregates_by_group(
                    X_all,
                    val_idx,
                    y_all,
                    groups_all,
                    metric_names=metric_names,
                    bedroc_alpha=bedroc_alpha,
                    infer_higher_is_better=infer_higher_is_better,
                )
                for scorer_name, group_df in desc_group_frames.items():
                    per_target_metrics.extend(
                        _dataframe_rows_from_group_metrics(
                            group_df,
                            fold_index=fold_index,
                            scorer=scorer_name,
                            scorer_type=DESCRIPTOR_AGGREGATE_SCORER_TYPE,
                            metric_names=metric_names,
                        )
                    )

        if (
            task == "dudez_screening"
            and cv_config.include_sf_consensus_baselines
            and scoring_function_columns
        ):
            consensus_metrics = evaluate_sf_consensus_baselines_on_fold(
                dataframe,
                val_idx,
                scoring_function_columns,
                y_all,
                groups_all,
                metric_names=metric_names,
                bedroc_alpha=bedroc_alpha,
                infer_higher_is_better=infer_higher_is_better,
            )
            sf_metrics.update(consensus_metrics)
            if groups_all is not None:
                consensus_group_frames = evaluate_sf_consensus_by_group(
                    dataframe,
                    val_idx,
                    scoring_function_columns,
                    y_all,
                    groups_all,
                    metric_names=metric_names,
                    bedroc_alpha=bedroc_alpha,
                    infer_higher_is_better=infer_higher_is_better,
                )
                for scorer_name, group_df in consensus_group_frames.items():
                    per_target_metrics.extend(
                        _dataframe_rows_from_group_metrics(
                            group_df,
                            fold_index=fold_index,
                            scorer=scorer_name,
                            scorer_type=SF_CONSENSUS_SCORER_TYPE,
                            metric_names=metric_names,
                        )
                    )

        fold_results.append(
            CrossValidationFoldResult(
                fold_index=fold_index,
                n_train=int(len(train_idx)),
                n_validation=int(len(val_idx)),
                train_indices=train_idx.astype(int).tolist(),
                validation_indices=val_idx.astype(int).tolist(),
                validation_metrics=_coerce_numeric_metrics(val_metrics),
                scoring_function_metrics=sf_metrics,
                per_target_metrics=per_target_metrics,
                diagnostics=diagnostics,
            )
        )
        LOGGER.info(
            "Cross-validation fold %s/%s (%s): n_train=%s n_val=%s",
            fold_index + 1,
            len(fold_pairs),
            task,
            len(train_idx),
            len(val_idx),
        )

    baseline_scorer_columns = list(scoring_function_columns)
    for fold in fold_results:
        for scorer_name in fold.scoring_function_metrics:
            if scorer_name not in baseline_scorer_columns:
                baseline_scorer_columns.append(scorer_name)

    aggregate = _aggregate_metric_dicts(
        [fold.validation_metrics for fold in fold_results],
        metric_names,
    )
    aggregate_sf = {}
    if baseline_scorer_columns:
        aggregate_sf = _aggregate_scoring_function_metrics(
            fold_results,
            baseline_scorer_columns,
            metric_names,
        )
    result = CrossValidationResult(
        export_dir=str(export_path.resolve()),
        task=task,
        n_folds=int(cv_config.n_folds),
        effective_folds=len(fold_pairs),
        strategy=strategy,
        epochs=int(cv_config.epochs),
        random_seed=int(cv_config.random_seed),
        objective_metric=objective_metric,
        fold_results=fold_results,
        aggregate_validation_metrics=aggregate,
        scoring_function_columns=list(baseline_scorer_columns),
        aggregate_scoring_function_metrics=aggregate_sf,
        model_config=model_config,
        diagnostics={
            "group_column": cv_config.group_column if task == "dudez_screening" else None,
            "kind_column": cv_config.kind_column if task == "dudez_screening" else None,
            "include_scoring_function_baselines": bool(cv_config.include_scoring_function_baselines),
            "include_descriptor_aggregate_baselines": bool(
                cv_config.include_descriptor_aggregate_baselines
            ),
            "include_sf_consensus_baselines": bool(cv_config.include_sf_consensus_baselines),
        },
    )
    if (
        task == "dudez_screening"
        and cv_config.include_scoring_function_baselines
        and baseline_scorer_columns
    ):
        result.scorer_comparison_summary = build_scorer_comparison_summary(result)

    resolved_output = Path(output_dir) if output_dir is not None else export_path / "cross_validation"
    save_cross_validation_result(result, resolved_output)
    return result


def save_cross_validation_result(result: CrossValidationResult, output_dir: str | Path) -> dict[str, str]:
    '''Write cross-validation JSON and per-fold CSV artifacts.

    Parameters
    ----------
    result : CrossValidationResult
        Cross-validation output.
    output_dir : str | Path
        Destination directory.

    Returns
    -------
    dict[str, str]
        Written artifact paths.
    '''

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "export_dir": result.export_dir,
        "task": result.task,
        "n_folds": result.n_folds,
        "effective_folds": result.effective_folds,
        "strategy": result.strategy,
        "epochs": result.epochs,
        "random_seed": result.random_seed,
        "objective_metric": result.objective_metric,
        "aggregate_validation_metrics": result.aggregate_validation_metrics,
        "scoring_function_columns": result.scoring_function_columns,
        "aggregate_scoring_function_metrics": result.aggregate_scoring_function_metrics,
        "scorer_comparison_summary": result.scorer_comparison_summary,
        "model_config": result.model_config,
        "diagnostics": result.diagnostics,
        "folds": [
            {
                "fold_index": fold.fold_index,
                "n_train": fold.n_train,
                "n_validation": fold.n_validation,
                "train_indices": fold.train_indices,
                "validation_indices": fold.validation_indices,
                "validation_metrics": {
                    OCSCORE_MODEL_SCORER_NAME: fold.validation_metrics,
                    **fold.scoring_function_metrics,
                },
                "scoring_function_metrics": fold.scoring_function_metrics,
                "per_target_metrics": fold.per_target_metrics,
                "diagnostics": fold.diagnostics,
            }
            for fold in result.fold_results
        ],
    }
    json_path = output_path / "cross_validation_results.json"
    json_path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")

    fold_rows: list[dict[str, Any]] = []
    for fold in result.fold_results:
        row = {
            "fold_index": fold.fold_index,
            "n_train": fold.n_train,
            "n_validation": fold.n_validation,
        }
        for metric_name, value in fold.validation_metrics.items():
            if isinstance(value, (int, float, np.floating)):
                row[f"validation_{metric_name}"] = float(value)
        fold_rows.append(row)
    csv_path = output_path / "cross_validation_folds.csv"
    pd.DataFrame(fold_rows).to_csv(csv_path, index=False)

    comparison_rows: list[dict[str, Any]] = []
    comparison_metric_names = (
        DUDEZ_CV_METRICS if result.task == "dudez_screening" else PDBBIND_CV_METRICS
    )
    for fold in result.fold_results:
        model_row = {
            "fold_index": fold.fold_index,
            "scorer": OCSCORE_MODEL_SCORER_NAME,
            "scorer_type": "model",
            "n_validation": fold.n_validation,
        }
        for metric_name in comparison_metric_names:
            if metric_name in fold.validation_metrics:
                model_row[f"validation_{metric_name}"] = float(fold.validation_metrics[metric_name])
        if "ranking_metrics_valid" in fold.validation_metrics:
            model_row["validation_ranking_metrics_valid"] = float(
                fold.validation_metrics["ranking_metrics_valid"]
            )
        comparison_rows.append(model_row)

        for scorer, metrics in sorted(fold.scoring_function_metrics.items()):
            baseline_type = scorer_type_for_baseline_name(scorer)
            scorer_type = baseline_type if baseline_type is not None else "sf"
            sf_row = {
                "fold_index": fold.fold_index,
                "scorer": scorer,
                "scorer_type": scorer_type,
                "n_validation": fold.n_validation,
            }
            for metric_name in comparison_metric_names:
                if metric_name in metrics:
                    sf_row[f"validation_{metric_name}"] = float(metrics[metric_name])
            if "ranking_metrics_valid" in metrics:
                sf_row["validation_ranking_metrics_valid"] = float(metrics["ranking_metrics_valid"])
            comparison_rows.append(sf_row)

    per_target_rows: list[dict[str, Any]] = []
    for fold in result.fold_results:
        per_target_rows.extend(fold.per_target_metrics)

    paths = {
        "output_dir": str(output_path.resolve()),
        "results_json": str(json_path.resolve()),
        "folds_csv": str(csv_path.resolve()),
    }
    if per_target_rows:
        per_target_path = output_path / "cross_validation_per_target_metrics.csv"
        pd.DataFrame(per_target_rows).to_csv(per_target_path, index=False)
        paths["per_target_csv"] = str(per_target_path.resolve())

    if comparison_rows:
        comparison_csv_path = output_path / "cross_validation_fold_comparison.csv"
        pd.DataFrame(comparison_rows).to_csv(comparison_csv_path, index=False)
        paths["fold_comparison_csv"] = str(comparison_csv_path.resolve())

    summary = result.scorer_comparison_summary
    if not summary and result.task == "dudez_screening" and result.scoring_function_columns:
        summary = build_scorer_comparison_summary(result)
    if summary:
        mean_std_path = output_path / "cross_validation_scorer_mean_std.csv"
        pd.DataFrame(summary["mean_std"]).to_csv(mean_std_path, index=False)
        paths["scorer_mean_std_csv"] = str(mean_std_path.resolve())

        wins_path = output_path / "cross_validation_ocscore_wins.csv"
        pd.DataFrame(summary["ocscore_wins"]).to_csv(wins_path, index=False)
        paths["ocscore_wins_csv"] = str(wins_path.resolve())

        rankings_path = output_path / "cross_validation_fold_rankings.csv"
        pd.DataFrame(summary["fold_rankings"]).to_csv(rankings_path, index=False)
        paths["fold_rankings_csv"] = str(rankings_path.resolve())

        summary_path = output_path / "cross_validation_scorer_summary.json"
        summary_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
        paths["scorer_summary_json"] = str(summary_path.resolve())

    return paths


__all__ = [
    "CROSS_VALIDATION_STRATEGIES",
    "CrossValidationConfig",
    "CrossValidationFoldResult",
    "CrossValidationResult",
    "DUDEZ_CV_COMPARISON_METRICS",
    "OCSCORE_MODEL_SCORER_NAME",
    "build_scorer_comparison_summary",
    "diagnose_entity_overlap",
    "evaluate_scoring_function_baselines_on_fold",
    "identify_scoring_function_columns",
    "infer_higher_is_better",
    "iter_receptor_group_kfold_indices",
    "iter_row_kfold_indices",
    "run_cross_validation_from_export",
    "save_cross_validation_result",
    "validate_fold_indices",
    "validate_fold_split",
    "validate_receptor_group_split",
]
