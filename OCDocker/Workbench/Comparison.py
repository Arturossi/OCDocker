#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only run comparison helpers for Workbench result manifests.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Any

from OCDocker.Workbench.Decision import parse_pareto_objective
from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.Models import ComparisonDirection
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import MetricSortMode
from OCDocker.Workbench.Models import ParetoObjective
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Models import RunStatus
from OCDocker.Workbench.Models import WorkbenchComparison
from OCDocker.Workbench.Models import WorkbenchComparisonCandidate
from OCDocker.Workbench.Models import WorkbenchComparisonMetric
from OCDocker.Workbench.Registry import discover_result_manifest_paths

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Constants
###############################################################################

DEFAULT_MINIMIZE_HINTS = frozenset(
    {
        "cost",
        "duration",
        "error",
        "latency",
        "loss",
        "mae",
        "mse",
        "rmse",
        "time",
    }
)
VALID_COMPARISON_DIRECTIONS = frozenset(
    {"improved", "regressed", "unchanged", "incomplete"}
)

# Functions
###############################################################################
## Private ##


def _flatten_metrics(metrics: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    '''Flatten nested metric dictionaries into dotted keys.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics dictionary.
    prefix : str
        Dotted prefix used during recursion.

    Returns
    -------
    dict[str, Any]
        Flattened metrics.
    '''

    flattened: dict[str, Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(value, prefix=name))
        else:
            flattened[name] = value
    return flattened


def _coerce_numeric(value: Any) -> float | None:
    '''Return a finite numeric metric value or None.

    Parameters
    ----------
    value : Any
        Metric value.

    Returns
    -------
    float or None
        Numeric value when available.
    '''

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    if numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _artifact_path(manifest_path: Path, artifact: ResultArtifact) -> Path:
    '''Resolve an artifact path relative to a result manifest.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Result manifest path.
    artifact : ResultArtifact
        Artifact declaration.

    Returns
    -------
    pathlib.Path
        Resolved artifact path.
    '''

    if artifact.path.is_absolute():
        return artifact.path
    return manifest_path.parent / artifact.path


def _missing_artifact_count(
    manifest_path: Path, artifacts: tuple[ResultArtifact, ...]
) -> int:
    '''Count declared artifacts that do not exist on disk.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Result manifest path.
    artifacts : tuple[ResultArtifact, ...]
        Declared artifacts.

    Returns
    -------
    int
        Missing artifact count.
    '''

    return sum(
        1
        for artifact in artifacts
        if not _artifact_path(manifest_path, artifact).exists()
    )


def _infer_metric_mode(metric_name: str) -> MetricSortMode:
    '''Infer a conservative comparison mode from a metric name.

    Parameters
    ----------
    metric_name : str
        Metric name.

    Returns
    -------
    MetricSortMode
        ``min`` for loss/error-like metrics and ``max`` otherwise.
    '''

    lowered = metric_name.lower()
    if any(hint in lowered for hint in DEFAULT_MINIMIZE_HINTS):
        return "min"
    return "max"


def _normalize_objectives(
    objectives: tuple[ParetoObjective, ...],
) -> tuple[ParetoObjective, ...]:
    '''Validate metric comparison objectives.

    Parameters
    ----------
    objectives : tuple[ParetoObjective, ...]
        Comparison objectives.

    Returns
    -------
    tuple[ParetoObjective, ...]
        Normalized comparison objectives.
    '''

    cleaned: list[ParetoObjective] = []
    for objective in objectives:
        name = str(objective.metric_name).strip()
        if not name:
            raise ValueError("comparison metric names must not be empty.")
        cleaned.append(ParetoObjective(metric_name=name, mode=objective.mode))
    names = tuple(objective.metric_name for objective in cleaned)
    if len(set(names)) != len(names):
        raise ValueError("comparison metric names must be unique.")
    return tuple(cleaned)


def _load_rows(
    root: Path, *, max_depth: int
) -> tuple[dict[str, dict[str, Any]], tuple[InventoryIssue, ...]]:
    '''Load result manifest rows keyed by run id.

    Parameters
    ----------
    root : pathlib.Path
        Workspace root or result manifest file.
    max_depth : int
        Maximum scan depth.

    Returns
    -------
    tuple
        Row mapping and non-fatal scan issues.
    '''

    rows: dict[str, dict[str, Any]] = {}
    issues: list[InventoryIssue] = []
    for manifest_path in discover_result_manifest_paths(root, max_depth=max_depth):
        try:
            manifest = read_result_manifest(manifest_path)
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))
            continue
        if manifest.run_id in rows:
            issues.append(
                InventoryIssue(
                    path=manifest_path,
                    message=f"Duplicate result manifest run_id: {manifest.run_id}",
                )
            )
            continue
        rows[manifest.run_id] = {
            "manifest_path": manifest_path,
            "run_id": manifest.run_id,
            "status": manifest.status,
            "metrics": _flatten_metrics(manifest.metrics),
            "artifact_count": len(manifest.artifacts),
            "missing_artifact_count": _missing_artifact_count(
                manifest_path,
                manifest.artifacts,
            ),
        }
    return rows, tuple(issues)


def _default_objectives(rows: dict[str, dict[str, Any]]) -> tuple[ParetoObjective, ...]:
    '''Infer comparison objectives from numeric metrics in loaded rows.

    Parameters
    ----------
    rows : dict[str, dict[str, Any]]
        Loaded result rows.

    Returns
    -------
    tuple[ParetoObjective, ...]
        Inferred objectives.
    '''

    metric_names = sorted(
        {
            name
            for row in rows.values()
            for name, value in row["metrics"].items()
            if _coerce_numeric(value) is not None
        }
    )
    return tuple(
        ParetoObjective(metric_name=name, mode=_infer_metric_mode(name))
        for name in metric_names
    )


def _direction(
    baseline_value: float | None,
    candidate_value: float | None,
    mode: MetricSortMode,
) -> ComparisonDirection:
    '''Classify a candidate metric relative to a baseline.

    Parameters
    ----------
    baseline_value : float or None
        Baseline numeric value.
    candidate_value : float or None
        Candidate numeric value.
    mode : MetricSortMode
        Comparison mode.

    Returns
    -------
    ComparisonDirection
        Comparison direction.
    '''

    if baseline_value is None or candidate_value is None:
        return "incomplete"
    if candidate_value == baseline_value:
        return "unchanged"
    if mode == "max":
        return "improved" if candidate_value > baseline_value else "regressed"
    return "improved" if candidate_value < baseline_value else "regressed"


def _percent_delta(delta: float | None, baseline_value: float | None) -> float | None:
    '''Return percent delta relative to baseline magnitude.

    Parameters
    ----------
    delta : float or None
        Candidate minus baseline.
    baseline_value : float or None
        Baseline metric value.

    Returns
    -------
    float or None
        Percent delta when computable.
    '''

    if delta is None or baseline_value is None or baseline_value == 0:
        return None
    return (delta / abs(baseline_value)) * 100.0


def _metric_comparison(
    *,
    objective: ParetoObjective,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> WorkbenchComparisonMetric:
    '''Build one metric comparison row.

    Parameters
    ----------
    objective : ParetoObjective
        Metric objective.
    baseline_metrics : dict[str, Any]
        Baseline flattened metrics.
    candidate_metrics : dict[str, Any]
        Candidate flattened metrics.

    Returns
    -------
    WorkbenchComparisonMetric
        Metric comparison row.
    '''

    name = objective.metric_name
    baseline_missing = name not in baseline_metrics
    candidate_missing = name not in candidate_metrics
    baseline_value = (
        None if baseline_missing else _coerce_numeric(baseline_metrics[name])
    )
    candidate_value = (
        None if candidate_missing else _coerce_numeric(candidate_metrics[name])
    )
    delta = (
        candidate_value - baseline_value
        if baseline_value is not None and candidate_value is not None
        else None
    )
    direction = _direction(baseline_value, candidate_value, objective.mode)
    return WorkbenchComparisonMetric(
        metric_name=name,
        mode=objective.mode,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=delta,
        percent_delta=_percent_delta(delta, baseline_value),
        direction=direction,
        improved=direction == "improved",
        regressed=direction == "regressed",
        baseline_missing=baseline_missing,
        candidate_missing=candidate_missing,
        baseline_non_numeric=(not baseline_missing and baseline_value is None),
        candidate_non_numeric=(not candidate_missing and candidate_value is None),
    )


def _candidate_comparison(
    *,
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
    objectives: tuple[ParetoObjective, ...],
) -> WorkbenchComparisonCandidate:
    '''Build one candidate run comparison.

    Parameters
    ----------
    baseline_row : dict[str, Any]
        Baseline result row.
    candidate_row : dict[str, Any]
        Candidate result row.
    objectives : tuple[ParetoObjective, ...]
        Metric objectives.

    Returns
    -------
    WorkbenchComparisonCandidate
        Candidate comparison payload.
    '''

    metrics = tuple(
        _metric_comparison(
            objective=objective,
            baseline_metrics=baseline_row["metrics"],
            candidate_metrics=candidate_row["metrics"],
        )
        for objective in objectives
    )
    improved_count = sum(1 for metric in metrics if metric.direction == "improved")
    regressed_count = sum(1 for metric in metrics if metric.direction == "regressed")
    unchanged_count = sum(1 for metric in metrics if metric.direction == "unchanged")
    incomplete_count = sum(1 for metric in metrics if metric.direction == "incomplete")
    return WorkbenchComparisonCandidate(
        run_id=candidate_row["run_id"],
        status=candidate_row["status"],
        manifest_path=candidate_row["manifest_path"],
        metrics=metrics,
        improved_count=improved_count,
        regressed_count=regressed_count,
        unchanged_count=unchanged_count,
        incomplete_count=incomplete_count,
        net_score=improved_count - regressed_count,
        artifact_count=candidate_row["artifact_count"],
        missing_artifact_count=candidate_row["missing_artifact_count"],
    )


def _candidate_run_ids(
    rows: dict[str, dict[str, Any]],
    *,
    baseline_run_id: str,
    requested: tuple[str, ...],
    issues: list[InventoryIssue],
    root: Path,
) -> tuple[str, ...]:
    '''Resolve candidate run ids.

    Parameters
    ----------
    rows : dict[str, dict[str, Any]]
        Loaded result rows.
    baseline_run_id : str
        Baseline run id.
    requested : tuple[str, ...]
        Requested candidate run ids.
    issues : list[InventoryIssue]
        Mutable issue collection.
    root : pathlib.Path
        Workspace root used for issue paths.

    Returns
    -------
    tuple[str, ...]
        Candidate run ids.
    '''

    if not requested:
        return tuple(run_id for run_id in sorted(rows) if run_id != baseline_run_id)
    candidates: list[str] = []
    seen: set[str] = set()
    for run_id in requested:
        cleaned = str(run_id).strip()
        if not cleaned:
            continue
        if cleaned == baseline_run_id:
            issues.append(
                InventoryIssue(
                    path=root,
                    message=f"Skipping baseline run_id as candidate: {cleaned}",
                )
            )
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        if cleaned not in rows:
            issues.append(
                InventoryIssue(
                    path=root,
                    message=f"Candidate run_id not found: {cleaned}",
                )
            )
            continue
        candidates.append(cleaned)
    return tuple(candidates)


def _sort_candidates(
    candidates: tuple[WorkbenchComparisonCandidate, ...],
) -> tuple[WorkbenchComparisonCandidate, ...]:
    '''Sort candidate comparisons by decision score.

    Parameters
    ----------
    candidates : tuple[WorkbenchComparisonCandidate, ...]
        Candidate comparisons.

    Returns
    -------
    tuple[WorkbenchComparisonCandidate, ...]
        Sorted candidate comparisons.
    '''

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.net_score,
                -candidate.improved_count,
                candidate.incomplete_count,
                candidate.missing_artifact_count,
                candidate.run_id,
            ),
        )
    )


## Public ##


def parse_comparison_metric(value: str) -> ParetoObjective:
    '''Parse a comparison metric specification.

    Parameters
    ----------
    value : str
        Metric selection in ``metric`` or ``metric:min|max`` form.

    Returns
    -------
    ParetoObjective
        Parsed metric objective.
    '''

    return parse_pareto_objective(value)


def build_run_comparison(
    root: str | Path,
    *,
    baseline_run_id: str,
    candidates: tuple[str, ...] | list[str] | None = None,
    metrics: tuple[ParetoObjective, ...] | list[ParetoObjective] | None = None,
    max_depth: int = 6,
) -> WorkbenchComparison:
    '''Compare result manifests against a selected baseline run.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    baseline_run_id : str
        Run id used as comparison baseline.
    candidates : tuple[str, ...] or list[str] or None
        Optional candidate run ids. If omitted, every non-baseline run is compared.
    metrics : tuple[ParetoObjective, ...] or list[ParetoObjective] or None
        Optional comparison metrics. If omitted, numeric metrics are inferred.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    WorkbenchComparison
        Read-only comparison payload for GUI, CLI, and automation consumers.
    '''

    baseline_id = str(baseline_run_id).strip()
    if not baseline_id:
        raise ValueError("baseline_run_id must not be empty.")
    root_path = Path(root)
    rows, load_issues = _load_rows(root_path, max_depth=max_depth)
    if baseline_id not in rows:
        raise ValueError(f"Baseline run_id not found: {baseline_id}")
    issues = list(load_issues)
    objective_tuple = _normalize_objectives(tuple(metrics or ()))
    if not objective_tuple:
        objective_tuple = _default_objectives(rows)
    if not objective_tuple:
        raise ValueError("No numeric metrics are available for comparison.")
    candidate_ids = _candidate_run_ids(
        rows,
        baseline_run_id=baseline_id,
        requested=tuple(candidates or ()),
        issues=issues,
        root=root_path,
    )
    baseline_row = rows[baseline_id]
    candidate_payloads = _sort_candidates(
        tuple(
            _candidate_comparison(
                baseline_row=baseline_row,
                candidate_row=rows[run_id],
                objectives=objective_tuple,
            )
            for run_id in candidate_ids
        )
    )
    return WorkbenchComparison(
        root=root_path,
        max_depth=max_depth,
        baseline_run_id=baseline_id,
        baseline_manifest_path=baseline_row["manifest_path"],
        baseline_status=baseline_row["status"],
        baseline_artifact_count=baseline_row["artifact_count"],
        baseline_missing_artifact_count=baseline_row["missing_artifact_count"],
        metrics=objective_tuple,
        candidate_count=len(candidate_payloads),
        candidates=candidate_payloads,
        best_candidate=candidate_payloads[0] if candidate_payloads else None,
        result_manifest_count=len(rows),
        issue_count=len(issues),
        issues=tuple(issues),
    )


__all__ = [
    "DEFAULT_MINIMIZE_HINTS",
    "VALID_COMPARISON_DIRECTIONS",
    "build_run_comparison",
    "parse_comparison_metric",
]
