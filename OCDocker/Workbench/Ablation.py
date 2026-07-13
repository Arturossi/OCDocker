#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only ablation analysis helpers for Workbench result manifests.
'''

# Imports
###############################################################################
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from OCDocker.Workbench.Comparison import build_run_comparison
from OCDocker.Workbench.Comparison import parse_comparison_metric
from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import ParetoObjective
from OCDocker.Workbench.Models import RunStatus
from OCDocker.Workbench.Models import WorkbenchAblationAnalysis
from OCDocker.Workbench.Models import WorkbenchAblationCandidate
from OCDocker.Workbench.Models import WorkbenchComparisonCandidate
from OCDocker.Workbench.Registry import discover_result_manifest_paths

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

ABLATION_CONTAINER_NAME = "ablations"
BASELINE_RUN_ID_HINTS = (
    "train",
    "full_reference",
    "full-reference",
    "reference",
    "baseline",
    "control",
)

# Classes
###############################################################################


@dataclass(frozen=True)
class _AblationRow:
    """Internal result row used during ablation discovery."""

    run_id: str
    status: RunStatus
    manifest_path: Path
    source_path: Path | None
    policy_name: str
    is_ablation: bool
    metric_count: int


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


def _is_numeric(value: Any) -> bool:
    '''Return whether a metric value is finite numeric.

    Parameters
    ----------
    value : Any
        Metric value.

    Returns
    -------
    bool
        True when the value is numeric.
    '''

    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric == numeric and numeric not in (float("inf"), float("-inf"))


def _source_path_from_run_manifest(manifest_path: Path) -> Path | None:
    '''Return the adopted source path declared beside a result manifest.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Result manifest path.

    Returns
    -------
    pathlib.Path or None
        Source path when available.
    '''

    run_manifest_path = manifest_path.parent / "run_manifest.yml"
    if not run_manifest_path.exists():
        return None
    try:
        run_manifest = read_run_manifest(run_manifest_path)
    except Exception:
        return None
    source_path = run_manifest.metadata.get("source_path") if run_manifest.metadata else None
    if source_path:
        return Path(str(source_path))
    return run_manifest.workspace


def _policy_from_path(path: Path | None, fallback: str) -> tuple[str, bool]:
    '''Infer ablation policy name from an adopted source path.

    Parameters
    ----------
    path : pathlib.Path or None
        Adopted source path.
    fallback : str
        Fallback run id.

    Returns
    -------
    tuple[str, bool]
        Policy name and whether the path is under an ablations container.
    '''

    if path is None:
        return fallback, False
    parts = path.parts
    for index, part in enumerate(parts):
        if part == ABLATION_CONTAINER_NAME and index + 1 < len(parts):
            return parts[index + 1], True
    return fallback, False


def _load_ablation_rows(root: Path, *, max_depth: int) -> tuple[dict[str, _AblationRow], tuple[InventoryIssue, ...]]:
    '''Load result rows with ablation policy metadata.

    Parameters
    ----------
    root : pathlib.Path
        Workspace root or result manifest path.
    max_depth : int
        Maximum scan depth.

    Returns
    -------
    tuple
        Rows keyed by run id and non-fatal scan issues.
    '''

    rows: dict[str, _AblationRow] = {}
    issues: list[InventoryIssue] = []
    for manifest_path in discover_result_manifest_paths(root, max_depth=max_depth):
        try:
            manifest = read_result_manifest(manifest_path)
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))
            continue
        if manifest.run_id in rows:
            issues.append(
                InventoryIssue(path=manifest_path, message=f"Duplicate result manifest run_id: {manifest.run_id}")
            )
            continue
        source_path = _source_path_from_run_manifest(manifest_path)
        policy_name, is_ablation = _policy_from_path(source_path, manifest.run_id)
        metrics = _flatten_metrics(manifest.metrics)
        rows[manifest.run_id] = _AblationRow(
            run_id=manifest.run_id,
            status=manifest.status,
            manifest_path=manifest_path,
            source_path=source_path,
            policy_name=policy_name,
            is_ablation=is_ablation,
            metric_count=sum(1 for value in metrics.values() if _is_numeric(value)),
        )
    return rows, tuple(issues)


def _baseline_score(row: _AblationRow) -> tuple[int, str]:
    '''Return auto-baseline priority for one row.

    Parameters
    ----------
    row : _AblationRow
        Candidate row.

    Returns
    -------
    tuple[int, str]
        Sortable baseline priority.
    '''

    run_id = row.run_id.lower()
    policy = row.policy_name.lower()
    source_name = row.source_path.name.lower() if row.source_path is not None else ""
    for index, hint in enumerate(BASELINE_RUN_ID_HINTS):
        if run_id == hint or policy == hint or source_name == hint:
            return (index, row.run_id)
    return (len(BASELINE_RUN_ID_HINTS), row.run_id)


def _choose_baseline(
    rows: dict[str, _AblationRow], baseline_run_id: str | None, root: Path
) -> tuple[_AblationRow, tuple[InventoryIssue, ...]]:
    '''Choose or validate the ablation reference row.

    Parameters
    ----------
    rows : dict[str, _AblationRow]
        Loaded rows keyed by run id.
    baseline_run_id : str or None
        Explicit baseline run id.
    root : pathlib.Path
        Workspace root used for issue paths.

    Returns
    -------
    tuple[_AblationRow, tuple[InventoryIssue, ...]]
        Baseline row and issues.
    '''

    if baseline_run_id:
        baseline = rows.get(str(baseline_run_id).strip())
        if baseline is None:
            raise ValueError(f"Baseline run_id not found: {baseline_run_id}")
        return baseline, ()

    non_ablation_rows = tuple(row for row in rows.values() if not row.is_ablation and row.metric_count > 0)
    if not non_ablation_rows:
        raise ValueError("No non-ablation reference run was found. Pass --baseline explicitly.")
    baseline = sorted(non_ablation_rows, key=_baseline_score)[0]
    issues: list[InventoryIssue] = []
    if len(non_ablation_rows) > 1:
        issues.append(InventoryIssue(path=root, message=f"Auto-selected ablation baseline run_id: {baseline.run_id}"))
    return baseline, tuple(issues)


def _candidate_ids(
    rows: dict[str, _AblationRow],
    *,
    baseline_run_id: str,
    requested: tuple[str, ...],
    issues: list[InventoryIssue],
    root: Path,
) -> tuple[str, ...]:
    '''Resolve ablation candidate run ids.

    Parameters
    ----------
    rows : dict[str, _AblationRow]
        Loaded rows.
    baseline_run_id : str
        Baseline run id.
    requested : tuple[str, ...]
        Requested run ids or policy names.
    issues : list[InventoryIssue]
        Mutable issues collection.
    root : pathlib.Path
        Workspace root used for issues.

    Returns
    -------
    tuple[str, ...]
        Candidate run ids.
    '''

    if not requested:
        return tuple(sorted(row.run_id for row in rows.values() if row.is_ablation and row.run_id != baseline_run_id))

    by_policy: dict[str, list[str]] = {}
    for row in rows.values():
        by_policy.setdefault(row.policy_name, []).append(row.run_id)

    resolved: list[str] = []
    seen: set[str] = set()
    for value in requested:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        matches: tuple[str, ...]
        if cleaned in rows:
            matches = (cleaned,)
        else:
            matches = tuple(sorted(by_policy.get(cleaned, ())))
        if not matches:
            issues.append(InventoryIssue(path=root, message=f"Ablation candidate not found: {cleaned}"))
            continue
        for run_id in matches:
            if run_id == baseline_run_id:
                issues.append(
                    InventoryIssue(path=root, message=f"Skipping baseline run_id as ablation candidate: {run_id}")
                )
                continue
            if run_id in seen:
                continue
            seen.add(run_id)
            resolved.append(run_id)
    return tuple(resolved)


def _ablation_candidate(comparison: WorkbenchComparisonCandidate, row: _AblationRow) -> WorkbenchAblationCandidate:
    '''Build one ablation candidate payload.

    Parameters
    ----------
    comparison : WorkbenchComparisonCandidate
        Generic comparison payload.
    row : _AblationRow
        Ablation metadata row.

    Returns
    -------
    WorkbenchAblationCandidate
        Ablation candidate payload.
    '''

    return WorkbenchAblationCandidate(
        policy_name=row.policy_name,
        run_id=comparison.run_id,
        status=comparison.status,
        manifest_path=comparison.manifest_path,
        source_path=row.source_path,
        metrics=comparison.metrics,
        improved_count=comparison.improved_count,
        regressed_count=comparison.regressed_count,
        unchanged_count=comparison.unchanged_count,
        incomplete_count=comparison.incomplete_count,
        net_score=comparison.net_score,
        artifact_count=comparison.artifact_count,
        missing_artifact_count=comparison.missing_artifact_count,
    )


## Public ##


def parse_ablation_metric(value: str) -> ParetoObjective:
    '''Parse an ablation metric specification.

    Parameters
    ----------
    value : str
        Metric selection in ``metric`` or ``metric:min|max`` form.

    Returns
    -------
    ParetoObjective
        Parsed metric objective.
    '''

    return parse_comparison_metric(value)


def build_ablation_analysis(
    root: str | Path,
    *,
    baseline_run_id: str | None = None,
    candidates: tuple[str, ...] | list[str] | None = None,
    metrics: tuple[ParetoObjective, ...] | list[ParetoObjective] | None = None,
    max_depth: int = 6,
) -> WorkbenchAblationAnalysis:
    '''Compare adopted OCScore ablation runs against a reference run.

    Parameters
    ----------
    root : str or pathlib.Path
        Workbench root or result manifest file to scan.
    baseline_run_id : str or None
        Optional explicit reference run id. If omitted, a non-ablation run such
        as ``train`` is selected when available.
    candidates : tuple[str, ...] or list[str] or None
        Optional ablation run ids or policy names. If omitted, every detected
        ablation policy is compared.
    metrics : tuple[ParetoObjective, ...] or list[ParetoObjective] or None
        Optional comparison metrics. If omitted, numeric metrics are inferred.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    WorkbenchAblationAnalysis
        Read-only ablation analysis payload.
    '''

    root_path = Path(root)
    rows, load_issues = _load_ablation_rows(root_path, max_depth=max_depth)
    if not rows:
        raise ValueError("No result manifests were found for ablation analysis.")

    baseline, baseline_issues = _choose_baseline(rows, baseline_run_id, root_path)
    issues = list(load_issues) + list(baseline_issues)
    candidate_run_ids = _candidate_ids(
        rows,
        baseline_run_id=baseline.run_id,
        requested=tuple(candidates or ()),
        issues=issues,
        root=root_path,
    )
    if not candidate_run_ids:
        issues.append(InventoryIssue(path=root_path, message="No ablation candidates were found."))
        return WorkbenchAblationAnalysis(
            root=root_path,
            max_depth=max_depth,
            baseline_run_id=baseline.run_id,
            baseline_policy_name=baseline.policy_name,
            baseline_manifest_path=baseline.manifest_path,
            baseline_source_path=baseline.source_path,
            metrics=tuple(metrics or ()),
            result_manifest_count=len(rows),
            detected_ablation_count=sum(1 for row in rows.values() if row.is_ablation),
            candidate_count=0,
            candidates=(),
            best_candidate=None,
            issue_count=len(issues),
            issues=tuple(issues),
        )

    comparison = build_run_comparison(
        root_path,
        baseline_run_id=baseline.run_id,
        candidates=candidate_run_ids,
        metrics=tuple(metrics or ()),
        max_depth=max_depth,
    )
    by_run_id = {candidate.run_id: candidate for candidate in comparison.candidates}
    ablation_candidates = tuple(
        _ablation_candidate(by_run_id[run_id], rows[run_id]) for run_id in candidate_run_ids if run_id in by_run_id
    )
    sorted_candidates = tuple(
        sorted(
            ablation_candidates,
            key=lambda candidate: (
                -candidate.net_score,
                -candidate.improved_count,
                candidate.incomplete_count,
                candidate.missing_artifact_count,
                candidate.policy_name,
                candidate.run_id,
            ),
        )
    )
    all_issues = tuple(issues) + comparison.issues
    return WorkbenchAblationAnalysis(
        root=root_path,
        max_depth=max_depth,
        baseline_run_id=baseline.run_id,
        baseline_policy_name=baseline.policy_name,
        baseline_manifest_path=baseline.manifest_path,
        baseline_source_path=baseline.source_path,
        metrics=comparison.metrics,
        result_manifest_count=len(rows),
        detected_ablation_count=sum(1 for row in rows.values() if row.is_ablation),
        candidate_count=len(sorted_candidates),
        candidates=sorted_candidates,
        best_candidate=sorted_candidates[0] if sorted_candidates else None,
        issue_count=len(all_issues),
        issues=all_issues,
    )


__all__ = [
    "ABLATION_CONTAINER_NAME",
    "BASELINE_RUN_ID_HINTS",
    "build_ablation_analysis",
    "parse_ablation_metric",
]
