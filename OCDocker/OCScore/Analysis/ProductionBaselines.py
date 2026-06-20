#!/usr/bin/env python3

# Description
###############################################################################
'''Production-grade baseline evaluation on DUDEz screening splits.

Evaluates individual scoring functions and train-only sklearn learners on the
same row indices and ranking metrics used by the staged OCScore protocol.
'''

# Imports
###############################################################################
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_screening_metrics
from OCDocker.OCScore.Optimization.ModelCrossValidation import (
    evaluate_scoring_function_baselines_on_fold,
    identify_scoring_function_columns,
    infer_higher_is_better,
)
from OCDocker.OCScore.Optimization.Protocol import ReplicaResult
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import (
    evaluate_descriptor_aggregate_baselines_on_fold,
    evaluate_sf_consensus_baselines_on_fold,
)

PRODUCTION_BASELINE_RANK_METRICS = (
    "BEDROC",
    "ROC-AUC",
    "PR-AUC",
    "EF1%",
    "EF5%",
    "NDCG@1%",
    "NDCG@5%",
)


# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''


# Classes
###############################################################################

class TrainOnlyFitError(ValueError):
    """Raised when a baseline fit uses non-train row indices."""


@dataclass
class ProductionBaselineConfig:
    """Configuration for production-grade baseline evaluation.

    Parameters
    ----------
    label_column : str, optional
        Binary label column in the DUDEz dataframe, by default ``"label"``.
    group_column : str, optional
        Receptor/group column for grouped screening metrics, by default ``"receptor"``.
    random_seed : int, optional
        Random seed for sklearn learners and shuffle control, by default 42.
    include_xgb : bool, optional
        Attempt XGBoost baseline when importable, by default True.
    include_lgbm : bool, optional
        Attempt LightGBM baseline when importable, by default True.
    include_shuffle_control : bool, optional
        Include shuffled-label logistic regression control, by default True.
    metric_names : Sequence[str], optional
        Ranking metrics retained in outputs, by default production headline metrics.
    bedroc_alpha : float, optional
        BEDROC exponential weighting factor, by default 20.0.
    """

    label_column: str = "label"
    group_column: str = "receptor"
    random_seed: int = 42
    include_xgb: bool = True
    include_lgbm: bool = True
    include_shuffle_control: bool = True
    include_sf_consensus: bool = True
    include_descriptor_aggregates: bool = True
    metric_names: Sequence[str] = field(default_factory=lambda: PRODUCTION_BASELINE_RANK_METRICS)
    bedroc_alpha: float = 20.0


# Functions
###############################################################################
## Public ##

def validate_fit_uses_train_only(
        train_indices: np.ndarray,
        fit_indices: np.ndarray,
    ) -> None:
    '''Ensure baseline fitting uses only train split row indices.

    Parameters
    ----------
    train_indices : np.ndarray
        Row indices in the train split.
    fit_indices : np.ndarray
        Row indices used for fitting the baseline.

    Raises
    ------
    TrainOnlyFitError
        When ``fit_indices`` contains rows outside ``train_indices``.
    '''

    allowed = {int(i) for i in np.asarray(train_indices).reshape(-1)}
    fit_set = {int(i) for i in np.asarray(fit_indices).reshape(-1)}
    leaked = sorted(fit_set - allowed)
    if leaked:
        preview = leaked[:5]
        suffix = "..." if len(leaked) > 5 else ""
        raise TrainOnlyFitError(
            f"Baseline fit indices include non-train rows: {preview}{suffix}"
        )


## Private ##

def _metric_subset(metrics: Mapping[str, float], metric_names: Sequence[str]) -> dict[str, float]:
    '''Extract a subset of metrics by name, filtering out missing or non-finite values.

    Parameters
    ----------
    metrics : Mapping[str, float]
        Full mapping of metric names to values.
    metric_names : Sequence[str]
        Names of metrics to retain in the output.

    Returns
    -------
    dict[str, float]
        Subset of input metrics with valid values.
    '''

    return {
        metric_name: float(metrics[metric_name])
        for metric_name in metric_names
        if metric_name in metrics and np.isfinite(metrics[metric_name])
    }


def _evaluate_split_metrics(
        y_true: np.ndarray,
        scores: np.ndarray,
        groups: Optional[np.ndarray],
        *,
        higher_is_better: bool,
        metric_names: Sequence[str],
        bedroc_alpha: float = 20.0,
    ) -> dict[str, float]:
    '''Evaluate ranking metrics for one split.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels for the split.
    scores : np.ndarray
        Baseline scores for the split.
    groups : np.ndarray or None
        Optional group labels for groupwise metrics.
    higher_is_better : bool
        Whether larger scores indicate stronger predictions.
    metric_names : Sequence[str]
        Metric names to keep.

    Returns
    -------
    dict[str, float]
        Filtered metric values.
    '''

    metrics = evaluate_screening_metrics(
        y_true,
        scores,
        groups=groups,
        higher_is_better=higher_is_better,
        bedroc_alpha=bedroc_alpha,
    )
    return _metric_subset(metrics, metric_names)


def _fit_learned_baseline(
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        train_indices: np.ndarray,
        fit_indices: np.ndarray,
        *,
        estimator: Any,
        random_seed: int,
    ) -> tuple[StandardScaler, Any]:
    '''Fit a train-only learned baseline.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Full feature matrix.
    labels : np.ndarray
        Full label vector.
    train_indices : np.ndarray
        Authorized train split indices.
    fit_indices : np.ndarray
        Indices used for fitting.
    estimator : Any
        Scikit-learn compatible estimator.
    random_seed : int
        Seed assigned to estimators exposing ``random_state``.

    Returns
    -------
    tuple[StandardScaler, Any]
        Fitted scaler and estimator.
    '''

    validate_fit_uses_train_only(train_indices, fit_indices)
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(feature_matrix[fit_indices]).astype(np.float32)
    y_fit = np.asarray(labels, dtype=int).reshape(-1)[fit_indices]
    model = estimator
    if hasattr(model, "random_state"):
        model.random_state = random_seed
    model.fit(x_fit, y_fit)
    return scaler, model


def _predict_learned_scores(
        scaler: StandardScaler,
        model: Any,
        feature_matrix: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
    '''Predict learned-baseline scores for a split.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted feature scaler.
    model : Any
        Fitted estimator.
    feature_matrix : np.ndarray
        Full feature matrix.
    indices : np.ndarray
        Split row indices to score.

    Returns
    -------
    np.ndarray
        Continuous prediction scores.
    '''

    x_split = scaler.transform(feature_matrix[indices]).astype(np.float32)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_split)
        return np.asarray(probabilities[:, 1], dtype=float)
    decision = model.decision_function(x_split)
    return np.asarray(decision, dtype=float)


def _optional_estimator(name: str, random_seed: int) -> tuple[Any | None, str | None]:
    '''Instantiate an optional learned baseline estimator.

    Parameters
    ----------
    name : str
        Optional estimator family name.
    random_seed : int
        Random seed used by the estimator.

    Returns
    -------
    tuple[Any or None, str or None]
        Estimator and skip reason. The estimator is None when unavailable.

    Raises
    ------
    ValueError
        If ``name`` is not a supported optional estimator.
    '''

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None, "xgboost not installed"
        return (
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_seed,
                eval_metric="logloss",
                verbosity=0,
            ),
            None,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return None, "lightgbm not installed"
        return (
            LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_seed,
                verbosity=-1,
            ),
            None,
        )
    raise ValueError(f"Unknown optional estimator: {name}")


def evaluate_learned_sf_baselines(
        dataframe: pd.DataFrame,
        selected_features: Sequence[str],
        split_indices: Mapping[str, Sequence[int]],
        *,
        label_column: str,
        group_column: str,
        config: ProductionBaselineConfig,
        fit_indices: Optional[np.ndarray] = None,
        shuffle_train_labels: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    '''Evaluate train-only sklearn baselines on SF descriptor columns.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, str]]]
        Per-split metric rows and skip notes for optional learners.
    '''

    sf_columns = identify_scoring_function_columns(selected_features)
    if not sf_columns:
        return [], [{"baseline": "learned_sf", "reason": "no scoring-function columns in selected features"}]

    available_columns = [column for column in sf_columns if column in dataframe.columns]
    if not available_columns:
        return [], [{"baseline": "learned_sf", "reason": "scoring-function columns missing from dataframe"}]

    train_idx = np.asarray(split_indices["train"], dtype=int)
    val_idx = np.asarray(split_indices["validation"], dtype=int)
    test_idx = np.asarray(split_indices["test"], dtype=int)
    fit_idx = np.asarray(train_idx if fit_indices is None else fit_indices, dtype=int)

    labels = dataframe[label_column].to_numpy(dtype=int)
    groups = (
        None
        if group_column not in dataframe.columns
        else dataframe[group_column].astype(str).to_numpy()
    )
    feature_matrix = dataframe[available_columns].to_numpy(dtype=float)

    y_train = labels[train_idx].copy()
    if shuffle_train_labels:
        rng = np.random.default_rng(config.random_seed)
        rng.shuffle(y_train)
        labels = labels.copy()
        labels[train_idx] = y_train

    learners: list[tuple[str, Any]] = [
        ("lr_sf", LogisticRegression(max_iter=1000, random_state=config.random_seed)),
        ("rf_sf", RandomForestClassifier(n_estimators=200, random_state=config.random_seed)),
    ]
    skip_notes: list[dict[str, str]] = []
    if config.include_xgb:
        estimator, reason = _optional_estimator("xgboost", config.random_seed)
        if estimator is None:
            skip_notes.append({"baseline": "xgb_sf", "reason": reason or "skipped"})
        else:
            learners.append(("xgb_sf", estimator))
    if config.include_lgbm:
        estimator, reason = _optional_estimator("lightgbm", config.random_seed)
        if estimator is None:
            skip_notes.append({"baseline": "lgbm_sf", "reason": reason or "skipped"})
        else:
            learners.append(("lgbm_sf", estimator))
    if shuffle_train_labels:
        learners = [("shuffled_lr_sf", LogisticRegression(max_iter=1000, random_state=config.random_seed))]

    rows: list[dict[str, Any]] = []
    for baseline_name, estimator in learners:
        scaler, model = _fit_learned_baseline(
            feature_matrix,
            labels,
            train_idx,
            fit_idx,
            estimator=estimator,
            random_seed=config.random_seed,
        )
        for split_name, split_idx in (("validation", val_idx), ("test", test_idx)):
            scores = _predict_learned_scores(scaler, model, feature_matrix, split_idx)
            y_split = labels[split_idx]
            g_split = None if groups is None else groups[split_idx]
            metrics = _evaluate_split_metrics(
                y_split,
                scores,
                g_split,
                higher_is_better=True,
                metric_names=config.metric_names,
                bedroc_alpha=config.bedroc_alpha,
            )
            rows.append(
                {
                    "baseline": baseline_name,
                    "baseline_family": "learned_sf",
                    "split": split_name,
                    **metrics,
                }
            )
    return rows, skip_notes


def evaluate_individual_sf_baselines(
        dataframe: pd.DataFrame,
        selected_features: Sequence[str],
        split_indices: Mapping[str, Sequence[int]],
        *,
        label_column: str,
        group_column: str,
        config: ProductionBaselineConfig,
    ) -> list[dict[str, Any]]:
    '''Evaluate raw scoring-function columns on validation and test splits.'''

    sf_columns = identify_scoring_function_columns(selected_features)
    if not sf_columns:
        return []

    labels = dataframe[label_column].to_numpy(dtype=int)
    groups = (
        None
        if group_column not in dataframe.columns
        else dataframe[group_column].astype(str).to_numpy()
    )
    val_idx = np.asarray(split_indices["validation"], dtype=int)
    test_idx = np.asarray(split_indices["test"], dtype=int)

    rows: list[dict[str, Any]] = []
    for split_name, split_idx in (("validation", val_idx), ("test", test_idx)):
        sf_metrics = evaluate_scoring_function_baselines_on_fold(
            dataframe,
            split_idx,
            labels,
            groups,
            sf_columns,
            metric_names=config.metric_names,
            bedroc_alpha=config.bedroc_alpha,
        )
        for baseline_name, metrics in sf_metrics.items():
            rows.append(
                {
                    "baseline": baseline_name,
                    "baseline_family": "scoring_function",
                    "split": split_name,
                    **_metric_subset(metrics, config.metric_names),
                }
            )
    return rows


def evaluate_sf_consensus_baselines(
        dataframe: pd.DataFrame,
        selected_features: Sequence[str],
        split_indices: Mapping[str, Sequence[int]],
        *,
        label_column: str,
        group_column: str,
        config: ProductionBaselineConfig,
    ) -> list[dict[str, Any]]:
    '''Evaluate SF consensus row aggregates on validation and test splits.'''

    sf_columns = identify_scoring_function_columns(selected_features)
    if not sf_columns:
        return []

    labels = dataframe[label_column].to_numpy(dtype=int)
    groups = (
        None
        if group_column not in dataframe.columns
        else dataframe[group_column].astype(str).to_numpy()
    )
    val_idx = np.asarray(split_indices["validation"], dtype=int)
    test_idx = np.asarray(split_indices["test"], dtype=int)

    rows: list[dict[str, Any]] = []
    for split_name, split_idx in (("validation", val_idx), ("test", test_idx)):
        consensus_metrics = evaluate_sf_consensus_baselines_on_fold(
            dataframe,
            split_idx,
            sf_columns,
            labels,
            groups,
            metric_names=config.metric_names,
            bedroc_alpha=config.bedroc_alpha,
            infer_higher_is_better=infer_higher_is_better,
        )
        for baseline_name, metrics in consensus_metrics.items():
            rows.append(
                {
                    "baseline": baseline_name,
                    "baseline_family": "sf_consensus",
                    "split": split_name,
                    **_metric_subset(metrics, config.metric_names),
                }
            )
    return rows


def evaluate_descriptor_aggregate_baselines(
        dataframe: pd.DataFrame,
        selected_features: Sequence[str],
        split_indices: Mapping[str, Sequence[int]],
        *,
        label_column: str,
        group_column: str,
        config: ProductionBaselineConfig,
    ) -> list[dict[str, Any]]:
    '''Evaluate descriptor row aggregates on validation and test splits.'''

    feature_columns = [column for column in selected_features if column in dataframe.columns]
    if not feature_columns:
        return []

    labels = dataframe[label_column].to_numpy(dtype=int)
    groups = (
        None
        if group_column not in dataframe.columns
        else dataframe[group_column].astype(str).to_numpy()
    )
    feature_matrix = dataframe[feature_columns].to_numpy(dtype=float)
    val_idx = np.asarray(split_indices["validation"], dtype=int)
    test_idx = np.asarray(split_indices["test"], dtype=int)

    rows: list[dict[str, Any]] = []
    for split_name, split_idx in (("validation", val_idx), ("test", test_idx)):
        aggregate_metrics = evaluate_descriptor_aggregate_baselines_on_fold(
            feature_matrix,
            split_idx,
            labels,
            groups,
            metric_names=config.metric_names,
            bedroc_alpha=config.bedroc_alpha,
            infer_higher_is_better=infer_higher_is_better,
        )
        for baseline_name, metrics in aggregate_metrics.items():
            rows.append(
                {
                    "baseline": baseline_name,
                    "baseline_family": "descriptor_aggregate",
                    "split": split_name,
                    **_metric_subset(metrics, config.metric_names),
                }
            )
    return rows


def run_production_baselines_for_replica(
        *,
        replica_name: str,
        dudez_df: pd.DataFrame,
        selected_features: Sequence[str],
        split_indices: Mapping[str, Sequence[int]],
        config: Optional[ProductionBaselineConfig] = None,
    ) -> dict[str, Any]:
    '''Run all configured baselines for one successful replica.'''

    cfg = config or ProductionBaselineConfig()
    rows: list[dict[str, Any]] = []
    skip_notes: list[dict[str, str]] = []

    sf_rows = evaluate_individual_sf_baselines(
        dudez_df,
        selected_features,
        split_indices,
        label_column=cfg.label_column,
        group_column=cfg.group_column,
        config=cfg,
    )
    rows.extend({**row, "replica": replica_name} for row in sf_rows)

    if cfg.include_sf_consensus:
        consensus_rows = evaluate_sf_consensus_baselines(
            dudez_df,
            selected_features,
            split_indices,
            label_column=cfg.label_column,
            group_column=cfg.group_column,
            config=cfg,
        )
        rows.extend({**row, "replica": replica_name} for row in consensus_rows)

    if cfg.include_descriptor_aggregates:
        descriptor_rows = evaluate_descriptor_aggregate_baselines(
            dudez_df,
            selected_features,
            split_indices,
            label_column=cfg.label_column,
            group_column=cfg.group_column,
            config=cfg,
        )
        rows.extend({**row, "replica": replica_name} for row in descriptor_rows)

    learned_rows, learned_skips = evaluate_learned_sf_baselines(
        dudez_df,
        selected_features,
        split_indices,
        label_column=cfg.label_column,
        group_column=cfg.group_column,
        config=cfg,
    )
    rows.extend({**row, "replica": replica_name} for row in learned_rows)
    skip_notes.extend(learned_skips)

    if cfg.include_shuffle_control:
        shuffle_rows, shuffle_skips = evaluate_learned_sf_baselines(
            dudez_df,
            selected_features,
            split_indices,
            label_column=cfg.label_column,
            group_column=cfg.group_column,
            config=cfg,
            shuffle_train_labels=True,
        )
        rows.extend({**row, "replica": replica_name} for row in shuffle_rows)
        skip_notes.extend(shuffle_skips)

    return {
        "replica": replica_name,
        "rows": rows,
        "skip_notes": skip_notes,
    }


def aggregate_baseline_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    '''Aggregate per-replica baseline rows to median metrics per baseline and split.'''

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(list(rows))
    metric_columns = [column for column in frame.columns if column in PRODUCTION_BASELINE_RANK_METRICS]
    group_cols = ["baseline", "baseline_family", "split"]
    grouped = (
        frame.groupby(group_cols, dropna=False)[metric_columns]
        .median(numeric_only=True)
        .reset_index()
    )
    counts = (
        frame.groupby(group_cols, dropna=False)["replica"]
        .nunique()
        .reset_index(name="n_replicas")
    )
    return grouped.merge(counts, on=group_cols, how="left")


def build_baseline_rank_table(summary_df: pd.DataFrame, *, split: str = "test", metric: str = "BEDROC") -> pd.DataFrame:
    '''Rank baselines by aggregated test-split metric (higher is better).'''

    if summary_df.empty:
        return pd.DataFrame(columns=["rank", "baseline", "baseline_family", metric, "n_replicas"])
    subset = summary_df[summary_df["split"] == split].copy()
    if metric not in subset.columns:
        return pd.DataFrame(columns=["rank", "baseline", "baseline_family", metric, "n_replicas"])
    subset = subset.sort_values(metric, ascending=False, kind="mergesort").reset_index(drop=True)
    subset.insert(0, "rank", np.arange(1, len(subset) + 1))
    return subset[["rank", "baseline", "baseline_family", metric, "n_replicas"]]


def write_production_baseline_reports(
        output_dir: str | Path,
        per_replica_results: Sequence[Mapping[str, Any]],
        *,
        skip_notes: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> dict[str, str]:
    '''Write baseline CSV reports under ``output_dir``.'''

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    per_fold_rows: list[dict[str, Any]] = []
    for replica_payload in per_replica_results:
        per_fold_rows.extend(replica_payload.get("rows", []))

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_path = out / "baselines_per_fold.csv"
    per_fold_df.to_csv(per_fold_path, index=False)
    paths["baselines_per_fold_csv"] = str(per_fold_path)

    summary_df = aggregate_baseline_rows(per_fold_rows)
    summary_path = out / "baselines_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    paths["baselines_summary_csv"] = str(summary_path)

    rank_df = build_baseline_rank_table(summary_df)
    rank_path = out / "baselines_rank_table.csv"
    rank_df.to_csv(rank_path, index=False)
    paths["baselines_rank_table_csv"] = str(rank_path)

    notes = list(skip_notes or [])
    for replica_payload in per_replica_results:
        notes.extend(replica_payload.get("skip_notes", []))
    if notes:
        notes_path = out / "baselines_skip_notes.json"
        notes_path.write_text(json.dumps(notes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths["baselines_skip_notes_json"] = str(notes_path)

    return paths


def run_and_write_production_baselines(
        output_dir: str | Path,
        *,
        dudez_df: pd.DataFrame,
        selected_features: Sequence[str],
        replica_results: Sequence[ReplicaResult],
        config: Optional[ProductionBaselineConfig] = None,
) -> dict[str, str]:
    '''Evaluate baselines for all successful replicas and write CSV outputs.'''

    cfg = config or ProductionBaselineConfig()
    per_replica_payloads: list[dict[str, Any]] = []
    for replica in replica_results:
        if not replica.success or replica.context is None:
            continue
        dudez_stage = replica.context.stage_results.get("dudez_optuna") or {}
        split_indices = dudez_stage.get("split_indices")
        if not split_indices:
            continue
        per_replica_payloads.append(
            run_production_baselines_for_replica(
                replica_name=replica.replica_name,
                dudez_df=dudez_df,
                selected_features=selected_features,
                split_indices=split_indices,
                config=cfg,
            )
        )
    return write_production_baseline_reports(output_dir, per_replica_payloads)


__all__ = [
    "PRODUCTION_BASELINE_RANK_METRICS",
    "ProductionBaselineConfig",
    "TrainOnlyFitError",
    "aggregate_baseline_rows",
    "build_baseline_rank_table",
    "evaluate_individual_sf_baselines",
    "evaluate_sf_consensus_baselines",
    "evaluate_descriptor_aggregate_baselines",
    "evaluate_learned_sf_baselines",
    "run_and_write_production_baselines",
    "run_production_baselines_for_replica",
    "validate_fit_uses_train_only",
    "write_production_baseline_reports",
]
