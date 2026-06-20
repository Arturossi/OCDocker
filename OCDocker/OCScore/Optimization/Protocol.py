#!/usr/bin/env python3

# Description
###############################################################################
'''
Minimal staged protocol abstractions for OCScore optimization workflows.

It is imported as:

from OCDocker.OCScore.Optimization.Protocol import ProtocolContext
'''

# Imports
###############################################################################
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import numpy as np
import pandas as pd
import optuna

import OCDocker.OCScore.Optimization.OptunaStorage as ocoptunastorage
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.Reproducibility as ocrepro

from OCDocker.OCScore.Utils.ContentHash import hash_feature_list

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

LOGGER = oclogging.get_logger("ocscore.optimization.protocol")


# Classes
###############################################################################

@dataclass
class ProtocolContext:
    """Shared state passed between staged protocol steps.

    Parameters
    ----------
    pdbbind_df : pd.DataFrame
        Reduced PDBbind dataframe.
    dudez_df : pd.DataFrame
        Reduced DUDEz dataframe.
    selected_features : list[str]
        Descriptor columns selected by the feature-reduction protocol.
    output_dir : str
        Directory where protocol artifacts are written.
    random_seed : int, optional
        Random seed used by protocol stages, by default 42.
    metadata : dict[str, Any], optional
        Optional dataset identifiers, paths, or caller metadata.
    artifacts : dict[str, Any], optional
        Runtime artifacts passed between stages.
    stage_results : dict[str, Any], optional
        JSON-serializable stage summaries.
    protocol_log : dict[str, Any], optional
        Reproducibility log accumulated by the protocol.
    """

    pdbbind_df: pd.DataFrame
    dudez_df: pd.DataFrame
    selected_features: list[str]
    output_dir: str
    random_seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    stage_results: dict[str, Any] = field(default_factory=dict)
    protocol_log: dict[str, Any] = field(default_factory=dict)

    def ensure_output_dir(self) -> Path:
        '''Create and return the output directory.

        Returns
        -------
        pathlib.Path
            Protocol output directory.
        '''

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out


class ProtocolStage(Protocol):
    """Protocol implemented by executable protocol stages.

    Attributes
    ----------
    name : str
        Stable stage name used in logs and result dictionaries.
    """

    name: str

    def run(self, context: ProtocolContext) -> ProtocolContext:
        '''Run the stage and return an updated context.

        Parameters
        ----------
        context : ProtocolContext
            Current protocol context.

        Returns
        -------
        ProtocolContext
            Updated protocol context.
        '''


@dataclass
class StagedProtocol:
    """Sequential protocol runner with explicit stage ownership.

    Parameters
    ----------
    stages : Iterable[ProtocolStage]
        Ordered protocol stages.
    write_protocol_log : bool, optional
        If True, write ``protocol_log.json`` after the run, by default True.
    """

    stages: Iterable[ProtocolStage]
    write_protocol_log: bool = True

    def run(self, context: ProtocolContext) -> ProtocolContext:
        '''Run all stages in order.

        Parameters
        ----------
        context : ProtocolContext
            Initial protocol context.

        Returns
        -------
        ProtocolContext
            Updated context after all stages have completed.
        '''

        out = context.ensure_output_dir()
        context.protocol_log.setdefault("schema_version", 1)
        context.protocol_log.setdefault("random_seed", context.random_seed)
        context.protocol_log.setdefault("metadata", _to_jsonable(_dynamic_protocol_metadata(context.metadata)))
        context.protocol_log.setdefault("stages", [])

        for stage in self.stages:
            LOGGER.debug("Starting protocol stage: %s", stage.name)
            context = stage.run(context)
            context.protocol_log["stages"].append({
                "name": stage.name,
                "result": _to_jsonable(_compact_stage_result_for_protocol_log(context.stage_results.get(stage.name, {}))),
            })
            LOGGER.debug("Finished protocol stage: %s", stage.name)

        if self.write_protocol_log:
            context.protocol_log["reproducibility"] = ocrepro.generate_reproducibility_manifest(
                include_python_packages=False
            )
            path = out / "protocol_log.json"
            path.write_text(json.dumps(_to_jsonable(context.protocol_log), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            context.stage_results["protocol_log_path"] = str(path)


        return context


@dataclass
class ReplicatedProtocolConfig:
    """Configuration for replicated staged protocol execution.

    A replica is one full staged Optuna/modeling execution. Replicas are not
    Optuna trials; each replica contains its own PDBbind Optuna study, transfer
    stage, and DUDEz Optuna study. Feature reduction is expected to run once
    before this protocol and is not repeated per replica.

    Parameters
    ----------
    n_replicas : int, optional
        Number of independent staged protocol executions, by default 1.
    base_seed : int | None, optional
        Base random seed. Replica ``i`` uses ``base_seed + i``. If None, the
        input context random seed is used.
    replica_name_prefix : str, optional
        Prefix used to build replica names, by default ``"replica"``.
    continue_on_replica_failure : bool, optional
        If True, failed replicas are recorded and later replicas continue. If
        False, the first failed replica raises an exception after reports are
        written, by default False.
    write_reports : bool, optional
        If True, write ``replicas_summary.csv``, ``replicas_summary.json``, and
        ``replicas_protocol.json`` in the base output directory, by default True.
    write_protocol_log : bool, optional
        If True, each replica writes its own ``protocol_log.json``, by default
        True.
    replica_jobs : int, optional
        Number of replicas to execute concurrently. Values greater than one use
        concurrent worker threads, by default 1.
    resume_completed : bool, optional
        If True, reuse completed replica directories with a valid protocol log
        instead of rerunning them, by default False.
    """

    n_replicas: int = 1
    base_seed: Optional[int] = None
    replica_name_prefix: str = "replica"
    continue_on_replica_failure: bool = False
    write_reports: bool = True
    write_protocol_log: bool = True
    replica_jobs: int = 1
    resume_completed: bool = False


@dataclass
class ReplicaResult:
    """Result for one replicated staged protocol execution.

    Parameters
    ----------
    replica_index : int
        Zero-based replica index.
    replica_name : str
        Stable replica name, for example ``"replica_000"``.
    seed : int
        Random seed used by this replica.
    output_dir : str
        Replica-specific output directory.
    success : bool
        Whether the replica completed all stages.
    context : ProtocolContext | None, optional
        Final replica context for successful replicas, by default None.
    summary : dict[str, Any], optional
        Per-replica summary row used in aggregate reports.
    error : str | None, optional
        Error message for failed replicas, by default None.
    failed_stage : str | None, optional
        Stage name active when the replica failed, by default None.
    traceback : str | None, optional
        Formatted traceback for failed replicas, by default None.
    """

    replica_index: int
    replica_name: str
    seed: int
    output_dir: str
    success: bool
    context: Optional[ProtocolContext] = None
    summary: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    failed_stage: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class ReplicatedProtocolResult:
    """Aggregated result returned by ``ReplicatedStagedProtocol``.

    Parameters
    ----------
    replica_results : list[ReplicaResult]
        Per-replica result objects.
    summary_df : pd.DataFrame
        One-row-per-replica summary table.
    aggregate_summary : dict[str, Any]
        Mean/std metrics and separate best PDBbind/DUDEz replica selections.
    output_paths : dict[str, str]
        Top-level report paths written by the replicated protocol.
    failed_replicas : list[ReplicaResult]
        Failed replica result objects. Empty if all replicas succeeded.
    """

    replica_results: list[ReplicaResult]
    summary_df: pd.DataFrame
    aggregate_summary: dict[str, Any]
    output_paths: dict[str, str]
    failed_replicas: list[ReplicaResult] = field(default_factory=list)


@dataclass(init=False)
class ReplicatedStagedProtocol:
    """Run a staged Optuna/modeling protocol across independent replicas.

    Replica execution applies to the staged Optuna/modeling protocol only. The
    reduced datasets and selected features are reused from the input context;
    feature reduction is not repeated. Replica ``i`` receives seed
    ``base_seed + i`` and output directory
    ``Path(context.output_dir) / f"{replica_name_prefix}_{i:03d}"``.

    Parameters
    ----------
    stages : Iterable[ProtocolStage]
        Ordered stages for one full modeling protocol execution.
    n_replicas : int, optional
        Number of independent protocol executions, by default 1.
    base_seed : int | None, optional
        Base seed used for deterministic replica seeds. If None, use the input
        context random seed.
    replica_name_prefix : str, optional
        Replica name prefix, by default ``"replica"``.
    continue_on_replica_failure : bool, optional
        Continue after failed replicas and record the failures, by default False.
    write_reports : bool, optional
        Write top-level replica summary reports, by default True.
    write_protocol_log : bool, optional
        Write each replica ``protocol_log.json``, by default True.
    config : ReplicatedProtocolConfig | None, optional
        Optional configuration object. Explicit keyword arguments are used when
        config is None.
    """

    stages: list[ProtocolStage]
    config: ReplicatedProtocolConfig

    def __init__(
            self,
            stages: Iterable[ProtocolStage],
            n_replicas: int = 1,
            base_seed: Optional[int] = None,
            replica_name_prefix: str = "replica",
            continue_on_replica_failure: bool = False,
            write_reports: bool = True,
            write_protocol_log: bool = True,
            replica_jobs: int = 1,
            resume_completed: bool = False,
            config: Optional[ReplicatedProtocolConfig] = None,
        ) -> None:
        '''Initialize a replicated staged protocol runner.

        Parameters
        ----------
        stages : Iterable[ProtocolStage]
            Ordered stages for one full modeling protocol execution.
        n_replicas : int, optional
            Number of independent protocol executions, by default 1.
        base_seed : int | None, optional
            Base seed for deterministic replica seeds, by default None.
        replica_name_prefix : str, optional
            Replica name prefix, by default ``"replica"``.
        continue_on_replica_failure : bool, optional
            Continue after failed replicas, by default False.
        write_reports : bool, optional
            Write top-level replica reports, by default True.
        write_protocol_log : bool, optional
            Write per-replica protocol logs, by default True.
        replica_jobs : int, optional
            Number of replicas to execute concurrently, by default 1.
        resume_completed : bool, optional
            Reuse completed replica outputs instead of rerunning them, by default False.
        config : ReplicatedProtocolConfig | None, optional
            Optional config object, by default None.
        '''

        self.stages = list(stages)
        self.config = copy.deepcopy(config) if config is not None else ReplicatedProtocolConfig(
            n_replicas=n_replicas,
            base_seed=base_seed,
            replica_name_prefix=replica_name_prefix,
            continue_on_replica_failure=continue_on_replica_failure,
            write_reports=write_reports,
            write_protocol_log=write_protocol_log,
            replica_jobs=replica_jobs,
            resume_completed=resume_completed,
        )
        if self.config.n_replicas < 1:
            raise ValueError("n_replicas must be at least 1.")
        if self.config.replica_jobs < 1:
            raise ValueError("replica_jobs must be at least 1.")

    def run(self, context: ProtocolContext) -> ReplicatedProtocolResult:
        '''Run all replicas and aggregate their results.

        Parameters
        ----------
        context : ProtocolContext
            Base context containing reduced datasets, selected features, output
            directory, seed, and feature-reduction metadata.

        Returns
        -------
        ReplicatedProtocolResult
            Per-replica contexts, summary table, aggregate metrics, output
            report paths, and failed replica records.
        '''

        base_output_dir = context.ensure_output_dir()
        base_seed = context.random_seed if self.config.base_seed is None else int(self.config.base_seed)
        n_replicas = int(self.config.n_replicas)
        replica_jobs = min(max(1, int(self.config.replica_jobs)), n_replicas)
        replica_auto_storage = replica_jobs > 1
        replica_results: list[ReplicaResult] = []
        pending_failure: Optional[ReplicaResult] = None

        if replica_jobs == 1:
            for replica_index in range(n_replicas):
                result = None
                if self.config.resume_completed:
                    result = _load_completed_replica_result(
                        stages=self.stages,
                        base_context=context,
                        replica_index=replica_index,
                        base_seed=base_seed,
                        base_output_dir=base_output_dir,
                        replica_name_prefix=self.config.replica_name_prefix,
                    )
                if result is None:
                    result = _execute_replica(
                        stages=self.stages,
                        base_context=context,
                        replica_index=replica_index,
                        base_seed=base_seed,
                        base_output_dir=base_output_dir,
                        replica_name_prefix=self.config.replica_name_prefix,
                        write_protocol_log=self.config.write_protocol_log,
                        replica_auto_storage=replica_auto_storage,
                        clean_incomplete_output=self.config.resume_completed,
                    )
                replica_results.append(result)
                if not result.success:
                    LOGGER.error("Replica %s failed at stage %s: %s", result.replica_name, result.failed_stage, result.error)
                    if not self.config.continue_on_replica_failure:
                        pending_failure = result
                        break
        else:
            LOGGER.info("Running %s replicas with replica_jobs=%s.", n_replicas, replica_jobs)
            indexed_results: dict[int, ReplicaResult] = {}
            pending_indices: list[int] = []
            for replica_index in range(n_replicas):
                result = None
                if self.config.resume_completed:
                    result = _load_completed_replica_result(
                        stages=self.stages,
                        base_context=context,
                        replica_index=replica_index,
                        base_seed=base_seed,
                        base_output_dir=base_output_dir,
                        replica_name_prefix=self.config.replica_name_prefix,
                    )
                if result is None:
                    pending_indices.append(replica_index)
                else:
                    indexed_results[replica_index] = result

            if pending_indices:
                with ThreadPoolExecutor(max_workers=replica_jobs) as executor:
                    futures = {
                        executor.submit(
                            _execute_replica,
                            stages=self.stages,
                            base_context=context,
                            replica_index=replica_index,
                            base_seed=base_seed,
                            base_output_dir=base_output_dir,
                            replica_name_prefix=self.config.replica_name_prefix,
                            write_protocol_log=self.config.write_protocol_log,
                            replica_auto_storage=replica_auto_storage,
                            clean_incomplete_output=self.config.resume_completed,
                        ): replica_index
                        for replica_index in pending_indices
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        indexed_results[result.replica_index] = result
                        if not result.success:
                            LOGGER.error("Replica %s failed at stage %s: %s", result.replica_name, result.failed_stage, result.error)

            replica_results = [indexed_results[index] for index in sorted(indexed_results)]
            if not self.config.continue_on_replica_failure:
                pending_failure = next((result for result in replica_results if not result.success), None)

        summary_rows = [result.summary for result in replica_results]
        summary_df = pd.DataFrame(summary_rows)
        aggregate_summary = _aggregate_replica_summaries(summary_df)
        failed_replicas = [result for result in replica_results if not result.success]
        output_paths: dict[str, str] = {}
        if self.config.write_reports:
            output_paths = _write_replicated_protocol_reports(
                base_output_dir=base_output_dir,
                base_context=context,
                protocol=self,
                replica_results=replica_results,
                summary_df=summary_df,
                aggregate_summary=aggregate_summary,
                base_seed=base_seed,
            )

        replicated_result = ReplicatedProtocolResult(
            replica_results=replica_results,
            summary_df=summary_df,
            aggregate_summary=aggregate_summary,
            output_paths=output_paths,
            failed_replicas=failed_replicas,
        )

        if pending_failure is not None:
            raise RuntimeError(
                f"Replica {pending_failure.replica_name} failed at stage "
                f"{pending_failure.failed_stage}: {pending_failure.error}"
            ) from None

        return replicated_result


# Functions
###############################################################################
## Private ##


def _aggregate_replica_summaries(summary_df: pd.DataFrame) -> dict[str, Any]:
    '''Aggregate successful replica metrics as mean/std by task.

    Parameters
    ----------
    summary_df : pd.DataFrame
        One-row-per-replica summary table.

    Returns
    -------
    dict[str, Any]
        Aggregate mean/std metrics and separate best-replica selections.
    '''

    if summary_df.empty:
        return {"metrics": {}, "best_pdbbind_replica": None, "best_dudez_replica": None}

    successful = summary_df[summary_df.get("status", "success") == "success"].copy()
    metric_columns = {
        "pdbbind_validation_rmse": "pdbbind_best_validation_rmse",
        "pdbbind_test_rmse": "pdbbind_test_rmse",
        "pdbbind_test_mae": "pdbbind_test_mae",
        "pdbbind_test_pearson_r": "pdbbind_test_pearson_r",
        "pdbbind_test_spearman_rho": "pdbbind_test_spearman_rho",
        "pdbbind_test_r2": "pdbbind_test_r2",
        "dudez_validation_primary_metric": "dudez_best_validation_metric",
        "dudez_test_roc_auc": "dudez_test_roc_auc",
        "dudez_test_pr_auc": "dudez_test_pr_auc",
        "dudez_test_bedroc": "dudez_test_bedroc",
        "dudez_test_ef1": "dudez_test_ef1",
        "dudez_test_ef5": "dudez_test_ef5",
        "dudez_test_ndcg_1": "dudez_test_ndcg_1",
        "dudez_test_ndcg_5": "dudez_test_ndcg_5",
    }
    metrics = {
        output_name: _metric_mean_std(successful[column_name])
        for output_name, column_name in metric_columns.items()
        if column_name in successful.columns
    }

    best_pdbbind = None
    if "pdbbind_best_validation_rmse" in successful.columns and not successful.empty:
        pdb_values = pd.to_numeric(successful["pdbbind_best_validation_rmse"], errors="coerce")
        if pdb_values.notna().any():
            row = successful.loc[pdb_values.idxmin()]
            best_pdbbind = {
                "replica_index": _to_jsonable(row.get("replica_index")),
                "replica_name": _to_jsonable(row.get("replica_name")),
                "pdbbind_best_validation_rmse": _to_jsonable(row.get("pdbbind_best_validation_rmse")),
            }

    best_dudez = None
    if "dudez_best_validation_metric" in successful.columns and not successful.empty:
        dudez_values = pd.to_numeric(successful["dudez_best_validation_metric"], errors="coerce")
        if dudez_values.notna().any():
            row = successful.loc[dudez_values.idxmax()]
            best_dudez = {
                "replica_index": _to_jsonable(row.get("replica_index")),
                "replica_name": _to_jsonable(row.get("replica_name")),
                "dudez_primary_metric": _to_jsonable(row.get("dudez_primary_metric")),
                "dudez_best_validation_metric": _to_jsonable(row.get("dudez_best_validation_metric")),
            }

    return {
        "n_replicas": int(len(summary_df)),
        "n_successful_replicas": int(len(successful)),
        "n_failed_replicas": int(len(summary_df) - len(successful)),
        "metrics": metrics,
        "reporting_policy": {
            "headline": "aggregate",
            "export_selection": "best_per_task",
            "primary_claim": "ranking_screening",
        },
        "best_pdbbind_replica": best_pdbbind,
        "best_dudez_replica": best_dudez,
    }


def _load_completed_replica_result(
        *,
        stages: list[ProtocolStage],
        base_context: ProtocolContext,
        replica_index: int,
        base_seed: int,
        base_output_dir: Path,
        replica_name_prefix: str,
    ) -> Optional[ReplicaResult]:
    '''Load a completed replica from disk when its protocol log is complete.'''

    replica_seed = base_seed + replica_index
    replica_name = f"{replica_name_prefix}_{replica_index:03d}"
    replica_output_dir = base_output_dir / replica_name
    protocol_log_path = replica_output_dir / "protocol_log.json"
    if not protocol_log_path.is_file():
        return None

    try:
        payload = json.loads(protocol_log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Ignoring unreadable resume log for %s: %s", replica_name, exc)
        return None

    stage_entries = payload.get("stages")
    if not isinstance(stage_entries, list):
        return None
    expected_names = [str(getattr(stage, "name", stage.__class__.__name__)) for stage in stages]
    stage_names = [str(entry.get("name")) for entry in stage_entries if isinstance(entry, dict)]
    if stage_names != expected_names:
        LOGGER.info(
            "Not resuming %s because recorded stages do not match current protocol.",
            replica_name,
        )
        return None

    stage_results: dict[str, Any] = {}
    for entry in stage_entries:
        if not isinstance(entry, dict):
            return None
        stage_name = str(entry.get("name"))
        result = entry.get("result")
        if not isinstance(result, dict):
            return None
        stage_results[stage_name] = result

    for stage_name in ("pdbbind_optuna", "dudez_optuna"):
        stage_result = stage_results.get(stage_name) or {}
        checkpoint_path = stage_result.get("checkpoint_path")
        if checkpoint_path and not Path(checkpoint_path).is_file():
            LOGGER.info("Not resuming %s because %s checkpoint is missing.", replica_name, stage_name)
            return None
        export_dir = stage_result.get("best_model_export_dir")
        if export_dir and not Path(export_dir).is_dir():
            LOGGER.info("Not resuming %s because %s export directory is missing.", replica_name, stage_name)
            return None

    replica_context = _build_replica_context(
        base_context=base_context,
        replica_index=replica_index,
        replica_name=replica_name,
        replica_seed=replica_seed,
        base_seed=base_seed,
        output_dir=replica_output_dir,
    )
    replica_context.protocol_log = payload
    replica_context.stage_results = stage_results
    replica_context.stage_results["protocol_log_path"] = str(protocol_log_path)
    summary = _build_replica_summary(replica_index, replica_name, replica_seed, replica_context)
    LOGGER.info("Resuming completed replica %s from %s", replica_name, protocol_log_path)
    return ReplicaResult(
        replica_index=replica_index,
        replica_name=replica_name,
        seed=replica_seed,
        output_dir=str(replica_output_dir),
        success=True,
        context=replica_context,
        summary=summary,
    )


def _execute_replica(
        *,
        stages: list[ProtocolStage],
        base_context: ProtocolContext,
        replica_index: int,
        base_seed: int,
        base_output_dir: Path,
        replica_name_prefix: str,
        write_protocol_log: bool,
        replica_auto_storage: bool,
        clean_incomplete_output: bool = False,
    ) -> ReplicaResult:
    '''Execute one replica and return a serializable result.'''

    replica_seed = base_seed + replica_index
    replica_name = f"{replica_name_prefix}_{replica_index:03d}"
    replica_output_dir = base_output_dir / replica_name
    if clean_incomplete_output:
        _clean_incomplete_replica_artifacts(
            stages=stages,
            replica_index=replica_index,
            replica_name=replica_name,
            replica_seed=replica_seed,
            base_output_dir=base_output_dir,
            replica_output_dir=replica_output_dir,
            replica_auto_storage=replica_auto_storage,
        )
    replica_context = _build_replica_context(
        base_context=base_context,
        replica_index=replica_index,
        replica_name=replica_name,
        replica_seed=replica_seed,
        base_seed=base_seed,
        output_dir=replica_output_dir,
    )
    current_stage_name: Optional[str] = None

    try:
        replica_stages = [
            _prepare_stage_for_replica(
                stage,
                replica_index,
                replica_name,
                replica_seed,
                base_output_dir=base_output_dir,
                replica_output_dir=replica_output_dir,
                replica_auto_storage=replica_auto_storage,
            )
            for stage in stages
        ]
        _initialize_protocol_log(replica_context)
        for stage in replica_stages:
            current_stage_name = stage.name
            LOGGER.debug("Starting replica %s stage: %s", replica_name, stage.name)
            replica_context = stage.run(replica_context)
            replica_context.protocol_log["stages"].append({
                "name": stage.name,
                "result": _to_jsonable(_compact_stage_result_for_protocol_log(replica_context.stage_results.get(stage.name, {}))),
            })
            LOGGER.debug("Finished replica %s stage: %s", replica_name, stage.name)

        if write_protocol_log:
            _write_context_protocol_log(replica_context)

        summary = _build_replica_summary(replica_index, replica_name, replica_seed, replica_context)
        replica_context.artifacts = {}
        return ReplicaResult(
            replica_index=replica_index,
            replica_name=replica_name,
            seed=replica_seed,
            output_dir=str(replica_output_dir),
            success=True,
            context=replica_context,
            summary=summary,
        )

    except Exception as exc:
        error_summary = _build_failed_replica_summary(
            replica_index=replica_index,
            replica_name=replica_name,
            replica_seed=replica_seed,
            output_dir=replica_output_dir,
            failed_stage=current_stage_name,
            exc=exc,
        )
        replica_context.artifacts = {}
        return ReplicaResult(
            replica_index=replica_index,
            replica_name=replica_name,
            seed=replica_seed,
            output_dir=str(replica_output_dir),
            success=False,
            context=replica_context,
            summary=error_summary,
            error=str(exc),
            failed_stage=current_stage_name,
            traceback=traceback.format_exc(),
        )


def _clean_incomplete_replica_artifacts(
        *,
        stages: list[ProtocolStage],
        replica_index: int,
        replica_name: str,
        replica_seed: int,
        base_output_dir: Path,
        replica_output_dir: Path,
        replica_auto_storage: bool,
    ) -> None:
    '''Remove stale outputs and Optuna studies for an incomplete replica.

    Resume only reuses replicas with a complete protocol log. When a replica is
    incomplete, its output directory and any replica-specific Optuna studies are
    stale because the rerun should start from the configured trial budget.
    '''

    if replica_output_dir.exists():
        LOGGER.info("Cleaning incomplete replica output before rerun: %s", replica_output_dir)
        shutil.rmtree(replica_output_dir)

    for stage in stages:
        try:
            replica_stage = _prepare_stage_for_replica(
                stage,
                replica_index=replica_index,
                replica_name=replica_name,
                replica_seed=replica_seed,
                base_output_dir=base_output_dir,
                replica_output_dir=replica_output_dir,
                replica_auto_storage=replica_auto_storage,
            )
            config = getattr(replica_stage, "config", None)
            if config is None:
                continue
            study_name = getattr(config, "study_name", None)
            storage = getattr(config, "storage", None)
            if not study_name or not storage:
                continue
            optuna.delete_study(study_name=str(study_name), storage=storage)
            LOGGER.info("Deleted stale Optuna study for incomplete replica: %s", study_name)
        except KeyError:
            continue
        except Exception as exc:
            LOGGER.warning(
                "Could not delete stale Optuna study for incomplete replica %s: %s",
                replica_name,
                exc,
            )


def _build_failed_replica_summary(
        replica_index: int,
        replica_name: str,
        replica_seed: int,
        output_dir: Path,
        failed_stage: Optional[str],
        exc: Exception,
    ) -> dict[str, Any]:
    '''Build a failed-replica summary row.

    Parameters
    ----------
    replica_index : int
        Zero-based replica index.
    replica_name : str
        Replica name.
    replica_seed : int
        Replica seed.
    output_dir : pathlib.Path
        Replica output directory.
    failed_stage : str | None
        Stage active when failure occurred.
    exc : Exception
        Raised exception.

    Returns
    -------
    dict[str, Any]
        Failed-replica summary row.
    '''

    row = _empty_replica_summary(replica_index, replica_name, replica_seed, output_dir)
    row.update({
        "status": "failed",
        "failed_stage": failed_stage,
        "error": str(exc),
    })
    return row


def _build_replica_context(
        base_context: ProtocolContext,
        replica_index: int,
        replica_name: str,
        replica_seed: int,
        base_seed: int,
        output_dir: Path,
    ) -> ProtocolContext:
    '''Create a fresh context for one replica.

    Parameters
    ----------
    base_context : ProtocolContext
        Input context shared across replicas.
    replica_index : int
        Zero-based replica index.
    replica_name : str
        Replica name.
    replica_seed : int
        Replica random seed.
    base_seed : int
        Base seed used to derive replica seeds.
    output_dir : pathlib.Path
        Replica-specific output directory.

    Returns
    -------
    ProtocolContext
        Fresh replica context.
    '''

    metadata = copy.deepcopy(base_context.metadata)
    metadata.update({
        "replica_index": replica_index,
        "replica_name": replica_name,
        "base_seed": base_seed,
        "replica_seed": replica_seed,
    })
    return ProtocolContext(
        pdbbind_df=base_context.pdbbind_df.copy(deep=True),
        dudez_df=base_context.dudez_df.copy(deep=True),
        selected_features=list(base_context.selected_features),
        output_dir=str(output_dir),
        random_seed=replica_seed,
        metadata=metadata,
        artifacts={},
        stage_results={},
        protocol_log={},
    )


def _build_replica_summary(
        replica_index: int,
        replica_name: str,
        replica_seed: int,
        context: ProtocolContext,
    ) -> dict[str, Any]:
    '''Build one successful replica summary row.

    Parameters
    ----------
    replica_index : int
        Zero-based replica index.
    replica_name : str
        Replica name.
    replica_seed : int
        Replica random seed.
    context : ProtocolContext
        Completed replica context.

    Returns
    -------
    dict[str, Any]
        Per-replica summary row.
    '''

    output_dir = Path(context.output_dir)
    row = _empty_replica_summary(replica_index, replica_name, replica_seed, output_dir)
    pdbbind = context.stage_results.get("pdbbind_optuna", {})
    dudez = context.stage_results.get("dudez_optuna", {})
    pdb_val = pdbbind.get("validation_metrics", {}) or {}
    pdb_test = pdbbind.get("test_metrics", {}) or {}
    dudez_test = dudez.get("test_metrics", {}) or {}

    row.update({
        "status": "success",
        "pdbbind_best_trial": pdbbind.get("best_trial"),
        "pdbbind_best_validation_rmse": pdb_val.get("RMSE", pdbbind.get("best_value")),
        "pdbbind_test_rmse": pdb_test.get("RMSE"),
        "pdbbind_test_mae": pdb_test.get("MAE"),
        "pdbbind_test_pearson_r": pdb_test.get("Pearson r"),
        "pdbbind_test_spearman_rho": pdb_test.get("Spearman rho"),
        "pdbbind_test_r2": pdb_test.get("R2"),
        "dudez_best_trial": dudez.get("best_trial"),
        "dudez_primary_metric": dudez.get("objective_metric"),
        "dudez_best_validation_metric": dudez.get("best_value"),
        "dudez_test_roc_auc": dudez_test.get("ROC-AUC"),
        "dudez_test_pr_auc": dudez_test.get("PR-AUC"),
        "dudez_test_bedroc": dudez_test.get("BEDROC"),
        "dudez_test_ef1": dudez_test.get("EF1%"),
        "dudez_test_ef5": dudez_test.get("EF5%"),
        "dudez_test_ndcg_1": dudez_test.get("NDCG@1%"),
        "dudez_test_ndcg_5": dudez_test.get("NDCG@5%"),
        "pdbbind_checkpoint_path": pdbbind.get("checkpoint_path"),
        "dudez_checkpoint_path": dudez.get("checkpoint_path"),
        "protocol_log_path": context.stage_results.get("protocol_log_path", str(output_dir / "protocol_log.json")),
    })
    return row


def _empty_replica_summary(
        replica_index: int,
        replica_name: str,
        replica_seed: int,
        output_dir: Path,
    ) -> dict[str, Any]:
    '''Build an empty summary row with all required columns.

    Parameters
    ----------
    replica_index : int
        Zero-based replica index.
    replica_name : str
        Replica name.
    replica_seed : int
        Replica random seed.
    output_dir : pathlib.Path
        Replica output directory.

    Returns
    -------
    dict[str, Any]
        Empty summary row.
    '''

    return {
        "replica_index": replica_index,
        "replica_name": replica_name,
        "seed": replica_seed,
        "output_dir": str(output_dir),
        "status": None,
        "failed_stage": None,
        "error": None,
        "pdbbind_best_trial": None,
        "pdbbind_best_validation_rmse": None,
        "pdbbind_test_rmse": None,
        "pdbbind_test_mae": None,
        "pdbbind_test_pearson_r": None,
        "pdbbind_test_spearman_rho": None,
        "pdbbind_test_r2": None,
        "dudez_best_trial": None,
        "dudez_primary_metric": None,
        "dudez_best_validation_metric": None,
        "dudez_test_roc_auc": None,
        "dudez_test_pr_auc": None,
        "dudez_test_bedroc": None,
        "dudez_test_ef1": None,
        "dudez_test_ef5": None,
        "dudez_test_ndcg_1": None,
        "dudez_test_ndcg_5": None,
        "pdbbind_checkpoint_path": None,
        "dudez_checkpoint_path": None,
        "protocol_log_path": None,
    }



DYNAMIC_PROTOCOL_METADATA_KEYS = frozenset({
    "base_seed",
    "replica_index",
    "replica_name",
    "replica_seed",
})
STATIC_STAGE_RESULT_KEYS = frozenset({
    "direction",
    "enable_pruning",
    "kind_column",
    "metrics_scope",
    "objective_metric",
    "pruner",
    "report_only_metrics",
    "requested_primary_metric",
    "scaling_metadata",
    "search_phase",
    "search_space",
    "selected_features",
    "split_config",
    "target_column",
})


def _dynamic_protocol_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return only metadata that can vary by replica execution."""

    return {
        key: metadata[key]
        for key in DYNAMIC_PROTOCOL_METADATA_KEYS
        if key in metadata
    }


def _static_context_metadata(metadata: dict[str, Any], selected_features_json: Path) -> dict[str, Any]:
    """Return static metadata without duplicating the full feature list."""

    payload = copy.deepcopy(_to_jsonable(metadata))
    feature_selection = payload.get("feature_selection")
    if isinstance(feature_selection, dict):
        feature_selection.pop("selected_features", None)
        feature_selection["selected_features_json"] = str(selected_features_json)
    return payload


def _compact_stage_result_for_protocol_log(result: dict[str, Any]) -> dict[str, Any]:
    """Remove static study context from per-replica protocol logs."""

    return {
        key: value
        for key, value in dict(result or {}).items()
        if key not in STATIC_STAGE_RESULT_KEYS
    }


def _initialize_protocol_log(context: ProtocolContext) -> None:
    '''Initialize protocol log fields for a context.

    Parameters
    ----------
    context : ProtocolContext
        Context to initialize in-place.
    '''

    context.ensure_output_dir()
    context.protocol_log.setdefault("schema_version", 1)
    context.protocol_log.setdefault("random_seed", context.random_seed)
    context.protocol_log.setdefault("metadata", _to_jsonable(_dynamic_protocol_metadata(context.metadata)))
    context.protocol_log.setdefault("stages", [])


def _metric_mean_std(values: pd.Series) -> dict[str, Any]:
    '''Compute aggregate summary statistics for one metric column.'''

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None, "ci95": None}
    n = int(len(numeric))
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=0))
    summary = {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "ci95": None,
    }
    if n >= 3 and std is not None and std > 0:
        half_width = 1.96 * std / math.sqrt(n)
        summary["ci95"] = {"low": mean - half_width, "high": mean + half_width}
    return summary


def _prepare_stage_for_replica(
        stage: ProtocolStage,
        replica_index: int,
        replica_name: str,
        replica_seed: int,
        base_output_dir: Path,
        replica_output_dir: Optional[Path] = None,
        replica_auto_storage: bool = False,
    ) -> ProtocolStage:
    '''Prepare a stage copy for one replica.

    Parameters
    ----------
    stage : ProtocolStage
        Base stage object.
    replica_index : int
        Zero-based replica index.
    replica_name : str
        Replica name.
    replica_seed : int
        Replica random seed.
    base_output_dir : pathlib.Path
        Top-level protocol output directory used to resolve shared Optuna
        storage when stage config storage is ``"auto"``.
    replica_output_dir : pathlib.Path | None, optional
        Replica output directory used for replica-local auto storage when
        ``replica_auto_storage`` is True.
    replica_auto_storage : bool, optional
        If True, resolve ``storage: auto`` inside the replica output directory
        to avoid SQLite writer contention during parallel replica execution.

    Returns
    -------
    ProtocolStage
        Replica-specific stage copy.
    '''

    replica_stage = copy.deepcopy(stage)
    configure = getattr(replica_stage, "configure_for_replica", None)
    if callable(configure):
        configured = configure(
            replica_index=replica_index,
            replica_name=replica_name,
            replica_seed=replica_seed,
        )
        if configured is not None:
            replica_stage = configured

    config = getattr(replica_stage, "config", None)
    if config is not None:
        if hasattr(config, "study_name"):
            base_name = str(getattr(config, "study_name"))
            if not base_name.endswith(f"_{replica_name}"):
                setattr(config, "study_name", f"{base_name}_{replica_name}")
        for seed_attr in ["random_seed", "sampler_seed"]:
            if hasattr(config, seed_attr):
                setattr(config, seed_attr, replica_seed)
        if hasattr(config, "storage"):
            storage_value = getattr(config, "storage")
            if storage_value == "auto":
                storage_base_dir = replica_output_dir if replica_auto_storage and replica_output_dir is not None else base_output_dir
                storage_value, _ = ocoptunastorage.resolve_optuna_storage(
                    "auto",
                    storage_base_dir,
                )
            setattr(
                config,
                "storage",
                _replica_storage_value(
                    storage_value,
                    replica_index=replica_index,
                    replica_name=replica_name,
                    replica_seed=replica_seed,
                ),
            )
    return replica_stage


def _replica_storage_value(
        storage: Any,
        replica_index: int,
        replica_name: str,
        replica_seed: int,
    ) -> Any:
    '''Build a replica-specific Optuna storage value when possible.

    Parameters
    ----------
    storage : Any
        Stage storage configuration.
    replica_index : int
        Zero-based replica index.
    replica_name : str
        Replica name.
    replica_seed : int
        Replica random seed.

    Returns
    -------
    Any
        Storage value with replica placeholders expanded. Distinct Optuna
        studies are identified by ``study_name``; SQLite paths are not
        automatically suffixed per replica.
    '''

    if not isinstance(storage, str):
        return storage

    try:
        return storage.format(
            replica_index=replica_index,
            replica_name=replica_name,
            replica_seed=replica_seed,
        )
    except (IndexError, KeyError, ValueError):
        return storage


def _stage_config_summary(stage: ProtocolStage) -> dict[str, Any]:
    '''Build a JSON-compatible stage configuration summary.

    Parameters
    ----------
    stage : ProtocolStage
        Stage object.

    Returns
    -------
    dict[str, Any]
        Stage name, class, and config attributes when available.
    '''

    config = getattr(stage, "config", None)
    return {
        "name": getattr(stage, "name", stage.__class__.__name__),
        "class": stage.__class__.__name__,
        "config": _to_jsonable(getattr(config, "__dict__", {})) if config is not None else {},
    }


def _write_context_protocol_log(context: ProtocolContext) -> str:
    '''Write one context protocol log.

    Parameters
    ----------
    context : ProtocolContext
        Completed context.

    Returns
    -------
    str
        Written protocol log path.
    '''

    context.protocol_log["reproducibility"] = ocrepro.generate_reproducibility_manifest(
        include_python_packages=False
    )
    path = Path(context.output_dir) / "protocol_log.json"
    path.write_text(json.dumps(_to_jsonable(context.protocol_log), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    context.stage_results["protocol_log_path"] = str(path)
    return str(path)


def _write_replicated_protocol_reports(
        base_output_dir: Path,
        base_context: ProtocolContext,
        protocol: ReplicatedStagedProtocol,
        replica_results: list[ReplicaResult],
        summary_df: pd.DataFrame,
        aggregate_summary: dict[str, Any],
        base_seed: int,
    ) -> dict[str, str]:
    '''Write top-level replicated protocol reports.

    Parameters
    ----------
    base_output_dir : pathlib.Path
        Base output directory.
    base_context : ProtocolContext
        Input context shared across replicas.
    protocol : ReplicatedStagedProtocol
        Replicated protocol runner.
    replica_results : list[ReplicaResult]
        Per-replica results.
    summary_df : pd.DataFrame
        One-row-per-replica summary table.
    aggregate_summary : dict[str, Any]
        Aggregate metrics.
    base_seed : int
        Base random seed.

    Returns
    -------
    dict[str, str]
        Written report paths.
    '''

    summary_csv = base_output_dir / "replicas_summary.csv"
    summary_json = base_output_dir / "replicas_summary.json"
    protocol_json = base_output_dir / "replicas_protocol.json"
    selected_features_json = base_output_dir / "selected_features.json"
    static_context_json = base_output_dir / "static_context.json"
    selected_features = list(base_context.selected_features)
    selected_features_json.write_text(json.dumps(selected_features, indent=2) + "\n", encoding="utf-8")
    static_context_payload = {
        "schema_version": 1,
        "n_selected_features": len(selected_features),
        "selected_features_hash": hash_feature_list(selected_features),
        "selected_features_json": str(selected_features_json),
        "input_metadata": _static_context_metadata(base_context.metadata, selected_features_json),
    }
    static_context_json.write_text(
        json.dumps(_to_jsonable(static_context_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_df.to_csv(summary_csv, index=False)
    summary_payload = {
        "replicas": summary_df.to_dict(orient="records"),
        "aggregate_summary": aggregate_summary,
        "failed_replicas": [result.summary for result in replica_results if not result.success],
    }
    summary_json.write_text(json.dumps(_to_jsonable(summary_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    protocol_payload = {
        "n_replicas": protocol.config.n_replicas,
        "base_seed": base_seed,
        "replica_name_prefix": protocol.config.replica_name_prefix,
        "continue_on_replica_failure": protocol.config.continue_on_replica_failure,
        "resume_completed": bool(protocol.config.resume_completed),
        "replica_names": [result.replica_name for result in replica_results],
        "replica_seeds": [result.seed for result in replica_results],
        "replica_jobs": int(protocol.config.replica_jobs),
        "stage_list": [_stage_config_summary(stage) for stage in protocol.stages],
        "static_context_json": str(static_context_json),
        "selected_features_json": str(selected_features_json),
        "output_dir": str(base_output_dir),
        "per_replica_output_paths": {
            result.replica_name: result.output_dir for result in replica_results
        },
        "aggregate_summary": aggregate_summary,
        "summary_paths": {
            "replicas_summary_csv": str(summary_csv),
            "replicas_summary_json": str(summary_json),
        },
        "failed_replicas": [result.summary for result in replica_results if not result.success],
    }
    protocol_json.write_text(json.dumps(_to_jsonable(protocol_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "replicas_summary_csv": str(summary_csv),
        "replicas_summary_json": str(summary_json),
        "replicas_protocol_json": str(protocol_json),
        "selected_features_json": str(selected_features_json),
        "static_context_json": str(static_context_json),
    }


def _to_jsonable(value: Any) -> Any:
    '''Convert common scientific Python objects to JSON-compatible values.

    Parameters
    ----------
    value : Any
        Value to convert before JSON serialization.

    Returns
    -------
    Any
        JSON-compatible representation of ``value``.
    '''

    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable({
            key: getattr(value, key)
            for key in value.__dataclass_fields__
            if key != "context"
        })
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    return str(value)


## Public ##

__all__ = [
    "ProtocolContext",
    "ProtocolStage",
    "ReplicaResult",
    "ReplicatedProtocolConfig",
    "ReplicatedProtocolResult",
    "ReplicatedStagedProtocol",
    "StagedProtocol",
]
