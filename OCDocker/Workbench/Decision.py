#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only decision-analysis helpers for Workbench result manifests.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Any

from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import MetricCatalog
from OCDocker.Workbench.Models import MetricCatalogEntry
from OCDocker.Workbench.Models import MetricSortMode
from OCDocker.Workbench.Models import ParetoEntry
from OCDocker.Workbench.Models import ParetoFront
from OCDocker.Workbench.Models import ParetoObjective
from OCDocker.Workbench.Models import RunStatus
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

VALID_OBJECTIVE_MODES = frozenset({"min", "max"})

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


def _mean(values: tuple[float, ...]) -> float | None:
    '''Return the arithmetic mean for numeric values.

    Parameters
    ----------
    values : tuple[float, ...]
        Numeric values.

    Returns
    -------
    float or None
        Mean value when at least one value is present.
    '''

    if not values:
        return None
    return sum(values) / len(values)


def _load_flattened_result_metrics(
    root: Path, *, max_depth: int
) -> tuple[
    list[tuple[Path, str, RunStatus, dict[str, Any]]], tuple[InventoryIssue, ...]
]:
    '''Load flattened metrics from result manifests below a root.

    Parameters
    ----------
    root : pathlib.Path
        Workspace root or result manifest file.
    max_depth : int
        Maximum scan depth.

    Returns
    -------
    tuple
        Loaded manifest rows and non-fatal issues.
    '''

    payloads: list[tuple[Path, str, RunStatus, dict[str, Any]]] = []
    issues: list[InventoryIssue] = []
    for manifest_path in discover_result_manifest_paths(root, max_depth=max_depth):
        try:
            manifest = read_result_manifest(manifest_path)
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))
            continue
        payloads.append(
            (
                manifest_path,
                manifest.run_id,
                manifest.status,
                _flatten_metrics(manifest.metrics),
            )
        )
    return payloads, tuple(issues)


def _catalog_entry(
    metric_name: str,
    rows: tuple[tuple[Path, str, RunStatus, dict[str, Any]], ...],
) -> MetricCatalogEntry:
    '''Build a metric catalog entry for one metric name.

    Parameters
    ----------
    metric_name : str
        Metric name.
    rows : tuple
        Loaded flattened metric rows.

    Returns
    -------
    MetricCatalogEntry
        Metric catalog entry.
    '''

    numeric_values: list[float] = []
    observed_count = 0
    for _, _, _, metrics in rows:
        if metric_name not in metrics:
            continue
        observed_count += 1
        numeric = _coerce_numeric(metrics[metric_name])
        if numeric is not None:
            numeric_values.append(numeric)

    numeric_tuple = tuple(numeric_values)
    return MetricCatalogEntry(
        metric_name=metric_name,
        observed_count=observed_count,
        numeric_count=len(numeric_tuple),
        non_numeric_count=observed_count - len(numeric_tuple),
        missing_count=len(rows) - observed_count,
        min_value=min(numeric_tuple) if numeric_tuple else None,
        max_value=max(numeric_tuple) if numeric_tuple else None,
        mean_value=_mean(numeric_tuple),
    )


def _normalize_objectives(
    objectives: tuple[ParetoObjective, ...],
) -> tuple[ParetoObjective, ...]:
    '''Validate and normalize Pareto objectives.

    Parameters
    ----------
    objectives : tuple[ParetoObjective, ...]
        Pareto objectives.

    Returns
    -------
    tuple[ParetoObjective, ...]
        Validated objectives.
    '''

    if not objectives:
        raise ValueError("At least one Pareto objective is required.")
    names = tuple(objective.metric_name for objective in objectives)
    if len(set(names)) != len(names):
        raise ValueError("Pareto objective metric names must be unique.")
    return objectives


def _pareto_entry_from_metrics(
    *,
    manifest_path: Path,
    run_id: str,
    status: RunStatus,
    metrics: dict[str, Any],
    objectives: tuple[ParetoObjective, ...],
) -> ParetoEntry:
    '''Build a Pareto entry from flattened metrics.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Result manifest path.
    run_id : str
        Run id.
    status : RunStatus
        Result status.
    metrics : dict[str, Any]
        Flattened metrics.
    objectives : tuple[ParetoObjective, ...]
        Pareto objectives.

    Returns
    -------
    ParetoEntry
        Pareto entry.
    '''

    values: dict[str, float] = {}
    missing: list[str] = []
    non_numeric: list[str] = []
    for objective in objectives:
        name = objective.metric_name
        if name not in metrics:
            missing.append(name)
            continue
        numeric = _coerce_numeric(metrics[name])
        if numeric is None:
            non_numeric.append(name)
            continue
        values[name] = numeric
    return ParetoEntry(
        manifest_path=manifest_path,
        run_id=run_id,
        status=status,
        metric_values=values,
        missing_metrics=tuple(missing),
        non_numeric_metrics=tuple(non_numeric),
        included=not missing and not non_numeric,
    )


def _is_at_least_as_good(a_value: float, b_value: float, mode: MetricSortMode) -> bool:
    '''Return whether one objective value is at least as good as another.

    Parameters
    ----------
    a_value : float
        Candidate value.
    b_value : float
        Comparison value.
    mode : MetricSortMode
        Objective mode.

    Returns
    -------
    bool
        True when ``a_value`` is at least as good as ``b_value``.
    '''

    return a_value >= b_value if mode == "max" else a_value <= b_value


def _is_strictly_better(a_value: float, b_value: float, mode: MetricSortMode) -> bool:
    '''Return whether one objective value is strictly better than another.

    Parameters
    ----------
    a_value : float
        Candidate value.
    b_value : float
        Comparison value.
    mode : MetricSortMode
        Objective mode.

    Returns
    -------
    bool
        True when ``a_value`` is strictly better than ``b_value``.
    '''

    return a_value > b_value if mode == "max" else a_value < b_value


def _dominates(
    candidate: ParetoEntry,
    other: ParetoEntry,
    objectives: tuple[ParetoObjective, ...],
) -> bool:
    '''Return whether one included entry dominates another.

    Parameters
    ----------
    candidate : ParetoEntry
        Candidate dominator.
    other : ParetoEntry
        Candidate dominated entry.
    objectives : tuple[ParetoObjective, ...]
        Pareto objectives.

    Returns
    -------
    bool
        True when ``candidate`` dominates ``other``.
    '''

    at_least = True
    strictly = False
    for objective in objectives:
        name = objective.metric_name
        candidate_value = candidate.metric_values[name]
        other_value = other.metric_values[name]
        at_least = at_least and _is_at_least_as_good(
            candidate_value, other_value, objective.mode
        )
        strictly = strictly or _is_strictly_better(
            candidate_value, other_value, objective.mode
        )
    return at_least and strictly


def _sort_pareto_entries(
    entries: tuple[ParetoEntry, ...], objectives: tuple[ParetoObjective, ...]
) -> tuple[ParetoEntry, ...]:
    '''Sort Pareto entries deterministically by first objective then run id.

    Parameters
    ----------
    entries : tuple[ParetoEntry, ...]
        Pareto entries.
    objectives : tuple[ParetoObjective, ...]
        Pareto objectives.

    Returns
    -------
    tuple[ParetoEntry, ...]
        Sorted entries.
    '''

    first = objectives[0]
    if first.mode == "max":
        return tuple(
            sorted(
                entries,
                key=lambda entry: (
                    -entry.metric_values[first.metric_name],
                    entry.run_id,
                ),
            )
        )
    return tuple(
        sorted(
            entries,
            key=lambda entry: (entry.metric_values[first.metric_name], entry.run_id),
        )
    )


## Public ##


def parse_pareto_objective(value: str) -> ParetoObjective:
    '''Parse a CLI Pareto objective specification.

    Parameters
    ----------
    value : str
        Objective in ``metric`` or ``metric:min|max`` form.

    Returns
    -------
    ParetoObjective
        Parsed objective.
    '''

    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("Pareto objective must not be empty.")
    if ":" not in cleaned:
        return ParetoObjective(metric_name=cleaned, mode="max")
    metric_name, mode = cleaned.rsplit(":", 1)
    metric_name = metric_name.strip()
    mode = mode.strip().lower()
    if not metric_name:
        raise ValueError("Pareto objective metric name must not be empty.")
    if mode not in VALID_OBJECTIVE_MODES:
        raise ValueError("Pareto objective mode must be either 'min' or 'max'.")
    return ParetoObjective(metric_name=metric_name, mode=mode)


def build_metrics_catalog(root: str | Path, *, max_depth: int = 6) -> MetricCatalog:
    '''Build a read-only catalog of metric coverage across result manifests.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    MetricCatalog
        Read-only metric catalog.
    '''

    root_path = Path(root)
    payloads, issues = _load_flattened_result_metrics(root_path, max_depth=max_depth)
    rows = tuple(payloads)
    metric_names = tuple(sorted({name for *_, metrics in rows for name in metrics}))
    return MetricCatalog(
        root=root_path,
        max_depth=max_depth,
        result_manifest_count=len(rows),
        metric_count=len(metric_names),
        metrics=tuple(
            _catalog_entry(metric_name, rows) for metric_name in metric_names
        ),
        issue_count=len(issues),
        issues=issues,
    )


def build_pareto_front(
    root: str | Path,
    *,
    objectives: tuple[ParetoObjective, ...],
    max_depth: int = 6,
) -> ParetoFront:
    '''Build a read-only Pareto front from result manifest metrics.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    objectives : tuple[ParetoObjective, ...]
        Pareto objectives.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    ParetoFront
        Read-only Pareto front payload.
    '''

    normalized_objectives = _normalize_objectives(objectives)
    root_path = Path(root)
    payloads, issues = _load_flattened_result_metrics(root_path, max_depth=max_depth)
    included: list[ParetoEntry] = []
    skipped: list[ParetoEntry] = []
    for manifest_path, run_id, status, metrics in payloads:
        entry = _pareto_entry_from_metrics(
            manifest_path=manifest_path,
            run_id=run_id,
            status=status,
            metrics=metrics,
            objectives=normalized_objectives,
        )
        if entry.included:
            included.append(entry)
        else:
            skipped.append(entry)

    included_entries = tuple(included)
    front: list[ParetoEntry] = []
    dominated: list[ParetoEntry] = []
    for entry in included_entries:
        dominators = tuple(
            candidate.run_id
            for candidate in included_entries
            if candidate.run_id != entry.run_id
            and _dominates(candidate, entry, normalized_objectives)
        )
        updated = entry.model_copy(update={"dominated_by": dominators})
        if dominators:
            dominated.append(updated)
        else:
            front.append(updated)

    return ParetoFront(
        root=root_path,
        max_depth=max_depth,
        objectives=normalized_objectives,
        front_entries=_sort_pareto_entries(tuple(front), normalized_objectives),
        dominated_entries=_sort_pareto_entries(tuple(dominated), normalized_objectives),
        skipped_entries=tuple(sorted(skipped, key=lambda entry: entry.run_id)),
        result_manifest_count=len(payloads),
        issue_count=len(issues),
        issues=issues,
    )


__all__ = [
    "VALID_OBJECTIVE_MODES",
    "build_metrics_catalog",
    "build_pareto_front",
    "parse_pareto_objective",
]
