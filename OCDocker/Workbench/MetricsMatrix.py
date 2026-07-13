#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only metric matrix helpers for Workbench result manifests.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Any

from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import MetricMatrix
from OCDocker.Workbench.Models import MetricMatrixRow
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Models import RunStatus
from OCDocker.Workbench.Registry import discover_result_manifest_paths

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Private ##


def _flatten_metrics(metrics: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    '''Flatten nested metrics into dotted keys.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics dictionary.
    prefix : str
        Optional dotted key prefix.

    Returns
    -------
    dict[str, Any]
        Flattened metrics dictionary.
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
    '''Return a finite numeric value or None for non-plot-ready values.

    Parameters
    ----------
    value : Any
        Metric value.

    Returns
    -------
    float or None
        Numeric value when possible.
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


def _normalize_metric_names(
    metric_names: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    '''Normalize optional metric selections.

    Parameters
    ----------
    metric_names : tuple[str, ...] or list[str] or None
        Optional metric names.

    Returns
    -------
    tuple[str, ...]
        Cleaned metric names.
    '''

    if metric_names is None:
        return ()
    names = tuple(str(name).strip() for name in metric_names if str(name).strip())
    if len(set(names)) != len(names):
        raise ValueError("metric names must be unique.")
    return names


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


def _row_from_flattened_metrics(
    *,
    manifest_path: Path,
    run_id: str,
    status: RunStatus,
    flattened_metrics: dict[str, Any],
    metric_names: tuple[str, ...],
    artifact_count: int,
    missing_artifact_count: int,
) -> MetricMatrixRow:
    '''Build one metric matrix row from flattened metrics.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Result manifest path.
    run_id : str
        Run id.
    status : str
        Run status.
    flattened_metrics : dict[str, Any]
        Flattened metric values.
    metric_names : tuple[str, ...]
        Matrix metric columns.
    artifact_count : int
        Declared artifact count.
    missing_artifact_count : int
        Missing artifact count.

    Returns
    -------
    MetricMatrixRow
        Matrix row.
    '''

    values: dict[str, float] = {}
    missing: list[str] = []
    non_numeric: list[str] = []
    for name in metric_names:
        if name not in flattened_metrics:
            missing.append(name)
            continue
        numeric = _coerce_numeric(flattened_metrics[name])
        if numeric is None:
            non_numeric.append(name)
            continue
        values[name] = numeric
    return MetricMatrixRow(
        manifest_path=manifest_path,
        run_id=run_id,
        status=status,
        metric_values=values,
        raw_metrics=flattened_metrics,
        missing_metrics=tuple(missing),
        non_numeric_metrics=tuple(non_numeric),
        artifact_count=artifact_count,
        missing_artifact_count=missing_artifact_count,
    )


## Public ##


def build_metric_matrix(
    root: str | Path,
    *,
    metric_names: tuple[str, ...] | list[str] | None = None,
    max_depth: int = 6,
) -> MetricMatrix:
    '''Build a read-only metric matrix from result manifests.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    metric_names : tuple[str, ...] or list[str] or None
        Optional metric names to include. If omitted, all discovered flattened keys are used.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    MetricMatrix
        Read-only metric matrix payload for GUI tables and plots.
    '''

    root_path = Path(root)
    selected_metrics = _normalize_metric_names(metric_names)
    manifest_payloads: list[tuple[Path, str, RunStatus, dict[str, Any], int, int]] = []
    discovered_metrics: set[str] = set()
    issues: list[InventoryIssue] = []

    for manifest_path in discover_result_manifest_paths(root_path, max_depth=max_depth):
        try:
            manifest = read_result_manifest(manifest_path)
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))
            continue
        flattened = _flatten_metrics(manifest.metrics)
        discovered_metrics.update(flattened)
        manifest_payloads.append(
            (
                manifest_path,
                manifest.run_id,
                manifest.status,
                flattened,
                len(manifest.artifacts),
                _missing_artifact_count(manifest_path, manifest.artifacts),
            )
        )

    matrix_metrics = selected_metrics or tuple(sorted(discovered_metrics))
    rows = tuple(
        _row_from_flattened_metrics(
            manifest_path=manifest_path,
            run_id=run_id,
            status=status,
            flattened_metrics=flattened,
            metric_names=matrix_metrics,
            artifact_count=artifact_count,
            missing_artifact_count=missing_artifact_count,
        )
        for (
            manifest_path,
            run_id,
            status,
            flattened,
            artifact_count,
            missing_artifact_count,
        ) in manifest_payloads
    )
    return MetricMatrix(
        root=root_path,
        max_depth=max_depth,
        metric_names=matrix_metrics,
        rows=rows,
        result_manifest_count=len(manifest_payloads),
        issue_count=len(issues),
        issues=tuple(issues),
    )


__all__ = ["build_metric_matrix"]
