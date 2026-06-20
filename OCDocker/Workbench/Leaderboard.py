#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only metric leaderboard helpers for Workbench result manifests.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Any

from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import MetricLeaderboard
from OCDocker.Workbench.Models import MetricLeaderboardEntry
from OCDocker.Workbench.Models import MetricSortMode
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Registry import discover_result_manifest_paths

# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Constants
###############################################################################

VALID_METRIC_SORT_MODES = frozenset({"min", "max"})

# Functions
###############################################################################
## Private ##


def _resolve_metric(metrics: dict[str, Any], metric_name: str) -> Any:
    '''Resolve a metric value from a metrics dictionary.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics payload from a result manifest.
    metric_name : str
        Metric name or dotted metric path.

    Returns
    -------
    Any
        Resolved metric value.
    '''

    if metric_name in metrics:
        return metrics[metric_name]

    value: Any = metrics
    for part in metric_name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(metric_name)
        value = value[part]
    return value


def _coerce_metric_value(value: Any) -> float:
    '''Coerce a metric value to a finite float.

    Parameters
    ----------
    value : Any
        Metric value.

    Returns
    -------
    float
        Numeric metric value.
    '''

    if isinstance(value, bool):
        raise TypeError("boolean metrics are not rankable")
    metric_value = float(value)
    if metric_value != metric_value:
        raise ValueError("NaN metrics are not rankable")
    if metric_value in (float("inf"), float("-inf")):
        raise ValueError("infinite metrics are not rankable")
    return metric_value


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
        Declared result artifacts.

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


def _entry_from_result_manifest(
    manifest_path: Path, *, metric_name: str
) -> MetricLeaderboardEntry:
    '''Build a leaderboard entry from one result manifest.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Result manifest path.
    metric_name : str
        Metric name or dotted metric path.

    Returns
    -------
    MetricLeaderboardEntry
        Leaderboard entry.
    '''

    manifest = read_result_manifest(manifest_path)
    artifact_count = len(manifest.artifacts)
    missing_count = _missing_artifact_count(manifest_path, manifest.artifacts)
    try:
        raw_value = _resolve_metric(manifest.metrics, metric_name)
    except KeyError:
        return MetricLeaderboardEntry(
            manifest_path=manifest_path,
            run_id=manifest.run_id,
            status=manifest.status,
            metric_name=metric_name,
            metrics=manifest.metrics,
            artifact_count=artifact_count,
            missing_artifact_count=missing_count,
            included=False,
            exclusion_reason=f"Metric not found: {metric_name}",
        )

    try:
        metric_value = _coerce_metric_value(raw_value)
    except (TypeError, ValueError) as exc:
        return MetricLeaderboardEntry(
            manifest_path=manifest_path,
            run_id=manifest.run_id,
            status=manifest.status,
            metric_name=metric_name,
            metrics=manifest.metrics,
            artifact_count=artifact_count,
            missing_artifact_count=missing_count,
            included=False,
            exclusion_reason=f"Metric is not numeric: {exc}",
        )

    return MetricLeaderboardEntry(
        manifest_path=manifest_path,
        run_id=manifest.run_id,
        status=manifest.status,
        metric_name=metric_name,
        metric_value=metric_value,
        metrics=manifest.metrics,
        artifact_count=artifact_count,
        missing_artifact_count=missing_count,
        included=True,
    )


def _rank_entries(
    entries: tuple[MetricLeaderboardEntry, ...], *, mode: MetricSortMode
) -> tuple[MetricLeaderboardEntry, ...]:
    '''Sort and rank included leaderboard entries.

    Parameters
    ----------
    entries : tuple[MetricLeaderboardEntry, ...]
        Included entries.
    mode : MetricSortMode
        Sort mode, either ``min`` or ``max``.

    Returns
    -------
    tuple[MetricLeaderboardEntry, ...]
        Ranked entries.
    '''

    if mode == "max":
        sorted_entries = sorted(
            entries, key=lambda entry: (-float(entry.metric_value), entry.run_id)
        )
    else:
        sorted_entries = sorted(
            entries, key=lambda entry: (float(entry.metric_value), entry.run_id)
        )
    return tuple(
        entry.model_copy(update={"rank": index})
        for index, entry in enumerate(sorted_entries, start=1)
    )


def _validate_metric_name(metric_name: str) -> str:
    '''Validate and normalize a metric name.

    Parameters
    ----------
    metric_name : str
        Metric name.

    Returns
    -------
    str
        Normalized metric name.
    '''

    cleaned = str(metric_name).strip()
    if not cleaned:
        raise ValueError("metric_name must not be empty.")
    return cleaned


## Public ##


def build_metric_leaderboard(
    root: str | Path,
    *,
    metric_name: str,
    mode: MetricSortMode = "max",
    max_depth: int = 6,
) -> MetricLeaderboard:
    '''Build a read-only metric leaderboard from result manifests.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    metric_name : str
        Metric name or dotted metric path to rank.
    mode : MetricSortMode
        Ranking mode, either ``max`` or ``min``.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    MetricLeaderboard
        Read-only leaderboard payload.
    '''

    metric = _validate_metric_name(metric_name)
    if mode not in VALID_METRIC_SORT_MODES:
        raise ValueError("mode must be either 'min' or 'max'.")

    root_path = Path(root)
    included: list[MetricLeaderboardEntry] = []
    skipped: list[MetricLeaderboardEntry] = []
    issues: list[InventoryIssue] = []

    for manifest_path in discover_result_manifest_paths(root_path, max_depth=max_depth):
        try:
            entry = _entry_from_result_manifest(manifest_path, metric_name=metric)
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))
            continue
        if entry.included:
            included.append(entry)
        else:
            skipped.append(entry)

    ranked_entries = _rank_entries(tuple(included), mode=mode)
    return MetricLeaderboard(
        root=root_path,
        metric_name=metric,
        mode=mode,
        max_depth=max_depth,
        ranked_entries=ranked_entries,
        skipped_entries=tuple(skipped),
        best_entry=ranked_entries[0] if ranked_entries else None,
        issue_count=len(issues),
        issues=tuple(issues),
    )


__all__ = ["VALID_METRIC_SORT_MODES", "build_metric_leaderboard"]
