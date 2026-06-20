#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only adoption helpers for existing OCDocker output directories.
'''

# Imports
###############################################################################
from __future__ import annotations

import csv
import json
import re

from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from OCDocker.Workbench.IO import write_model
from OCDocker.Workbench.Models import ArtifactKind
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Models import ResultManifest
from OCDocker.Workbench.Models import RunManifest
from OCDocker.Workbench.Models import RunStatus
from OCDocker.Workbench.Models import WorkbenchAdoptedRun
from OCDocker.Workbench.Models import WorkbenchAdoptionCandidate
from OCDocker.Workbench.Models import WorkbenchAdoptionPlan
from OCDocker.Workbench.Models import WorkbenchAdoptionResult
from OCDocker.Workbench.Models import WorkbenchSpecType

# License
###############################################################################
"""OCDocker
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
"""

# Constants
###############################################################################

DEFAULT_ADOPTION_MAX_DEPTH = 3
DEFAULT_MAX_METRIC_FILE_BYTES = 1_048_576
METRIC_NAME_HINTS = (
    "metric",
    "metrics",
    "result",
    "results",
    "score",
    "scores",
    "summary",
    "evaluation",
    "eval",
)
METRIC_SUFFIXES = frozenset({".csv", ".json", ".yaml", ".yml"})
LOG_SUFFIXES = frozenset({".log", ".out", ".err"})
LOG_NAME_HINTS = ("log", "stdout", "stderr")
ARTIFACT_KIND_BY_SUFFIX: dict[str, ArtifactKind] = {
    ".csv": "csv",
    ".db": "database",
    ".htm": "html",
    ".html": "html",
    ".jpeg": "image",
    ".jpg": "image",
    ".json": "json",
    ".log": "log",
    ".md": "markdown",
    ".out": "log",
    ".err": "log",
    ".pdf": "pdf",
    ".png": "image",
    ".sqlite": "database",
    ".svg": "image",
    ".yaml": "json",
    ".yml": "json",
}
ADOPTABLE_SUFFIXES = frozenset(ARTIFACT_KIND_BY_SUFFIX) | frozenset(
    {".txt", ".tsv", ".parquet", ".pkl", ".pickle", ".pt", ".pth"}
)
RUN_MANIFEST_FILENAME = "run_manifest.yml"
RESULT_MANIFEST_FILENAME = "result_manifest.yml"

# Functions
###############################################################################
## Private ##


def _issue(path: Path, message: str) -> InventoryIssue:
    '''Build a non-fatal adoption issue.

    Parameters
    ----------
    path : pathlib.Path
        Path related to the issue.
    message : str
        Issue message.

    Returns
    -------
    InventoryIssue
        Issue payload.
    '''

    return InventoryIssue(path=path, message=message)


def _validate_scan_arguments(source_root: Path, max_depth: int, max_metric_file_bytes: int) -> None:
    '''Validate adoption scan arguments.

    Parameters
    ----------
    source_root : pathlib.Path
        Existing output root to scan.
    max_depth : int
        Maximum directory depth below ``source_root``.
    max_metric_file_bytes : int
        Maximum metric file size to parse.
    '''

    if not source_root.exists():
        raise FileNotFoundError(f"Adoption source does not exist: {source_root}")
    if not source_root.is_dir():
        raise ValueError(f"Adoption source must be a directory: {source_root}")
    if max_depth < 0:
        raise ValueError("max_depth must be greater than or equal to zero.")
    if max_metric_file_bytes < 1:
        raise ValueError("max_metric_file_bytes must be greater than or equal to one.")


def _is_hidden(path: Path) -> bool:
    '''Return whether a path should be hidden from adoption scans.

    Parameters
    ----------
    path : pathlib.Path
        Path to inspect.

    Returns
    -------
    bool
        True when the path is hidden.
    '''

    return path.name.startswith(".") or path.name == "__pycache__"


def _iter_directories(root: Path, *, max_depth: int) -> tuple[Path, ...]:
    '''Return directories below a source root up to a maximum depth.

    Parameters
    ----------
    root : pathlib.Path
        Source root.
    max_depth : int
        Maximum directory depth.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Directories ordered by path.
    '''

    directories: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        directory, depth = queue.pop(0)
        directories.append(directory)
        if depth >= max_depth:
            continue
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            if child.is_dir() and not _is_hidden(child):
                queue.append((child, depth + 1))
    return tuple(directories)


def _iter_adoption_directories(root: Path, *, max_depth: int) -> tuple[Path, ...]:
    '''Return directories that can hold adopted Workbench run evidence.

    Parameters
    ----------
    root : pathlib.Path
        Source root.
    max_depth : int
        Maximum directory depth for regular directory discovery.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Directories ordered for adoption scanning.
    '''

    directories = list(_iter_directories(root, max_depth=max_depth))
    seen = set(directories)
    for directory in tuple(directories):
        ablation_containers: list[Path] = []
        if directory.name == "ablations":
            ablation_containers.append(directory)
        direct_container = directory / "ablations"
        if direct_container.is_dir() and not _is_hidden(direct_container):
            ablation_containers.append(direct_container)

        for container in ablation_containers:
            if container not in seen:
                directories.append(container)
                seen.add(container)
            for child in sorted(container.iterdir(), key=lambda path: path.name):
                if child.is_dir() and not _is_hidden(child) and child not in seen:
                    directories.append(child)
                    seen.add(child)

    return tuple(directories)


def _direct_files(directory: Path) -> tuple[Path, ...]:
    '''Return visible direct files in a directory.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to inspect.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Direct files ordered by name.
    '''

    return tuple(
        sorted(
            (path for path in directory.iterdir() if path.is_file() and not _is_hidden(path)),
            key=lambda path: path.name,
        )
    )


def _has_name_hint(path: Path, hints: tuple[str, ...]) -> bool:
    '''Return whether a filename contains any configured hint.

    Parameters
    ----------
    path : pathlib.Path
        Path to inspect.
    hints : tuple[str, ...]
        Lower-case hints.

    Returns
    -------
    bool
        True when a hint is present.
    '''

    lowered = path.stem.lower()
    return any(hint in lowered for hint in hints)


def _is_metric_file(path: Path) -> bool:
    '''Return whether a file should be parsed for metrics.

    Parameters
    ----------
    path : pathlib.Path
        File path.

    Returns
    -------
    bool
        True when the file is a supported metric candidate.
    '''

    if path.name.lower() in {RUN_MANIFEST_FILENAME, RESULT_MANIFEST_FILENAME}:
        return False
    return path.suffix.lower() in METRIC_SUFFIXES and _has_name_hint(path, METRIC_NAME_HINTS)


def _is_log_file(path: Path) -> bool:
    '''Return whether a file should be declared as a run log.

    Parameters
    ----------
    path : pathlib.Path
        File path.

    Returns
    -------
    bool
        True when the file is log-like.
    '''

    suffix = path.suffix.lower()
    return suffix in LOG_SUFFIXES or (suffix == ".txt" and _has_name_hint(path, LOG_NAME_HINTS))


def _artifact_kind(path: Path) -> ArtifactKind:
    '''Infer a Workbench artifact kind from a path suffix.

    Parameters
    ----------
    path : pathlib.Path
        Artifact path.

    Returns
    -------
    ArtifactKind
        Inferred artifact kind.
    '''

    return ARTIFACT_KIND_BY_SUFFIX.get(path.suffix.lower(), "other")


def _artifact_role(path: Path) -> str:
    '''Infer a simple artifact role from a filename.

    Parameters
    ----------
    path : pathlib.Path
        Artifact path.

    Returns
    -------
    str
        Artifact role.
    '''

    if _is_metric_file(path):
        return "metrics"
    if _is_log_file(path):
        return "log"
    if _has_name_hint(path, ("plot", "figure", "chart")):
        return "plot"
    if _has_name_hint(path, ("report", "summary")):
        return "report"
    return ""


def _is_adoptable_artifact(path: Path) -> bool:
    '''Return whether a file should be declared as an adopted artifact.

    Parameters
    ----------
    path : pathlib.Path
        File path.

    Returns
    -------
    bool
        True when the file has a supported artifact suffix.
    '''

    if path.name.lower() in {RUN_MANIFEST_FILENAME, RESULT_MANIFEST_FILENAME}:
        return False
    return path.suffix.lower() in ADOPTABLE_SUFFIXES


def _flatten_numeric_metrics(value: Any, *, prefix: str = "") -> dict[str, float]:
    '''Flatten numeric metrics from a nested mapping.

    Parameters
    ----------
    value : Any
        Input value.
    prefix : str
        Dotted prefix used during recursion.

    Returns
    -------
    dict[str, float]
        Flattened numeric metrics.
    '''

    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric_metrics(item, prefix=name))
        return flattened
    if isinstance(value, bool):
        return flattened
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return flattened
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return flattened
    if prefix:
        flattened[prefix] = numeric
    return flattened


def _read_limited_text(path: Path, *, max_bytes: int) -> str:
    '''Read a bounded metric file as text.

    Parameters
    ----------
    path : pathlib.Path
        File path.
    max_bytes : int
        Maximum accepted file size.

    Returns
    -------
    str
        File text.
    '''

    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Metric file is too large to parse safely ({size} bytes > {max_bytes} bytes).")
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_mapping_metrics(path: Path, *, max_bytes: int) -> dict[str, float]:
    '''Parse numeric metrics from a JSON or YAML mapping.

    Parameters
    ----------
    path : pathlib.Path
        Metric file path.
    max_bytes : int
        Maximum accepted file size.

    Returns
    -------
    dict[str, float]
        Parsed metrics.
    '''

    text = _read_limited_text(path, max_bytes=max_bytes)
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        return {}
    payload = loaded.get("metrics", loaded)
    return _flatten_numeric_metrics(payload)


def _parse_csv_metrics(path: Path, *, max_bytes: int) -> dict[str, float]:
    '''Parse numeric metrics from a CSV file.

    Parameters
    ----------
    path : pathlib.Path
        CSV path.
    max_bytes : int
        Maximum accepted file size.

    Returns
    -------
    dict[str, float]
        Parsed metrics.
    '''

    text = _read_limited_text(path, max_bytes=max_bytes)
    reader = csv.DictReader(StringIO(text))
    fieldnames = tuple(reader.fieldnames or ())
    lowered = {name.lower(): name for name in fieldnames}
    metrics: dict[str, float] = {}
    if "metric" in lowered and "value" in lowered:
        metric_key = lowered["metric"]
        value_key = lowered["value"]
        for row in reader:
            name = str(row.get(metric_key, "")).strip()
            if not name:
                continue
            metrics.update(_flatten_numeric_metrics(row.get(value_key), prefix=name))
        return metrics

    rows = tuple(reader)
    if len(rows) == 1:
        for key, value in rows[0].items():
            metrics.update(_flatten_numeric_metrics(value, prefix=key))
    return metrics


def _parse_metric_file(path: Path, *, max_bytes: int) -> dict[str, float]:
    '''Parse numeric metrics from one supported metric file.

    Parameters
    ----------
    path : pathlib.Path
        Metric file path.
    max_bytes : int
        Maximum accepted file size.

    Returns
    -------
    dict[str, float]
        Parsed metrics.
    '''

    if path.suffix.lower() == ".csv":
        return _parse_csv_metrics(path, max_bytes=max_bytes)
    return _parse_mapping_metrics(path, max_bytes=max_bytes)


def _slugify(value: str) -> str:
    '''Build a stable run id from a directory name.

    Parameters
    ----------
    value : str
        Raw value.

    Returns
    -------
    str
        Slugified run id.
    '''

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "adopted-run"


def _unique_run_id(base: str, seen: set[str]) -> str:
    '''Return a run id that is unique within an adoption plan.

    Parameters
    ----------
    base : str
        Base run id.
    seen : set[str]
        Already used run ids.

    Returns
    -------
    str
        Unique run id.
    '''

    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}-{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _candidate_status(metrics: dict[str, float], override: RunStatus | None) -> RunStatus:
    '''Infer or return an adoption run status.

    Parameters
    ----------
    metrics : dict[str, float]
        Parsed metrics.
    override : RunStatus or None
        Optional explicit status.

    Returns
    -------
    RunStatus
        Candidate status.
    '''

    if override is not None:
        return override
    if metrics:
        return "completed"
    return "defined"


def _artifact_from_path(path: Path) -> ResultArtifact:
    '''Build an adopted artifact declaration from an existing file.

    Parameters
    ----------
    path : pathlib.Path
        Existing artifact path.

    Returns
    -------
    ResultArtifact
        Artifact declaration using the original absolute path.
    '''

    return ResultArtifact(
        name=path.stem or path.name,
        path=path.resolve(strict=False),
        kind=_artifact_kind(path),
        role=_artifact_role(path),
    )


def _build_candidate(
    directory: Path,
    *,
    run_id: str,
    spec_type: WorkbenchSpecType,
    status: RunStatus | None,
    max_metric_file_bytes: int,
) -> WorkbenchAdoptionCandidate | None:
    '''Build one adoption candidate from a source directory.

    Parameters
    ----------
    directory : pathlib.Path
        Existing output directory.
    run_id : str
        Unique run id.
    spec_type : WorkbenchSpecType
        Adopted Workbench spec type.
    status : RunStatus or None
        Optional explicit run status.
    max_metric_file_bytes : int
        Maximum accepted metric file size.

    Returns
    -------
    WorkbenchAdoptionCandidate or None
        Candidate when the directory contains adoptable evidence.
    '''

    files = _direct_files(directory)
    metric_files = tuple(path for path in files if _is_metric_file(path))
    log_files = tuple(path.resolve(strict=False) for path in files if _is_log_file(path))
    artifact_files = tuple(path for path in files if _is_adoptable_artifact(path))
    if not metric_files and not log_files and not artifact_files:
        return None

    metrics: dict[str, float] = {}
    issues: list[InventoryIssue] = []
    for metric_file in metric_files:
        try:
            metrics.update(_parse_metric_file(metric_file, max_bytes=max_metric_file_bytes))
        except Exception as exc:
            issues.append(_issue(metric_file, f"Could not parse metrics: {exc}"))

    artifacts = tuple(_artifact_from_path(path) for path in artifact_files)
    return WorkbenchAdoptionCandidate(
        source_path=directory,
        run_id=run_id,
        spec_type=spec_type,
        name=directory.name or run_id,
        status=_candidate_status(metrics, status),
        workspace=directory.resolve(strict=False),
        metric_files=tuple(path.resolve(strict=False) for path in metric_files),
        log_files=log_files,
        artifacts=artifacts,
        metrics=metrics,
        issue_count=len(issues),
        issues=tuple(issues),
    )


def _run_metadata(candidate: WorkbenchAdoptionCandidate) -> dict[str, Any]:
    '''Build metadata stored in adopted run manifests.

    Parameters
    ----------
    candidate : WorkbenchAdoptionCandidate
        Adoption candidate.

    Returns
    -------
    dict[str, Any]
        Run manifest metadata.
    '''

    return {
        "adopted": True,
        "source_path": str(candidate.source_path),
        "metric_files": [str(path) for path in candidate.metric_files],
    }


def _ensure_destination_run_dir(
    destination_root: Path,
    run_id: str,
    *,
    overwrite: bool,
) -> Path:
    '''Create or validate a destination run directory.

    Parameters
    ----------
    destination_root : pathlib.Path
        Workbench destination root.
    run_id : str
        Run id.
    overwrite : bool
        Whether existing manifests may be overwritten.

    Returns
    -------
    pathlib.Path
        Destination run directory.
    '''

    run_dir = destination_root / run_id
    manifest_paths = (run_dir / RUN_MANIFEST_FILENAME, run_dir / RESULT_MANIFEST_FILENAME)
    if not overwrite and any(path.exists() for path in manifest_paths):
        raise FileExistsError(f"Adopted Workbench manifests already exist for run {run_id!r}: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


## Public ##


def build_adoption_plan(
    source_root: str | Path,
    *,
    max_depth: int = DEFAULT_ADOPTION_MAX_DEPTH,
    spec_type: WorkbenchSpecType = "ocscore_ablation",
    status: RunStatus | None = None,
    run_id_prefix: str = "",
    max_metric_file_bytes: int = DEFAULT_MAX_METRIC_FILE_BYTES,
    require_metrics: bool = False,
) -> WorkbenchAdoptionPlan:
    '''Build a dry-run plan for adopting existing output directories.

    Parameters
    ----------
    source_root : str or pathlib.Path
        Existing output root to scan. It is never modified.
    max_depth : int
        Maximum directory depth below ``source_root`` to inspect. OCScore
        ablation policy directories under an ``ablations`` container are
        included when the parent run directory or container is reached.
    spec_type : WorkbenchSpecType
        Workbench spec type assigned to adopted run manifests.
    status : RunStatus or None
        Optional status override. If omitted, completed is used when metrics are found.
    run_id_prefix : str
        Optional prefix applied to generated run ids.
    max_metric_file_bytes : int
        Maximum metric file size parsed during adoption planning.
    require_metrics : bool
        If True, only directories with at least one parsed metric are included.

    Returns
    -------
    WorkbenchAdoptionPlan
        Dry-run adoption plan.
    '''

    root = Path(source_root)
    _validate_scan_arguments(root, max_depth, max_metric_file_bytes)
    seen: set[str] = set()
    candidates: list[WorkbenchAdoptionCandidate] = []
    issues: list[InventoryIssue] = []

    for directory in _iter_adoption_directories(root, max_depth=max_depth):
        base_id = _slugify(f"{run_id_prefix}{directory.name or 'root'}")
        try:
            candidate = _build_candidate(
                directory,
                run_id=base_id,
                spec_type=spec_type,
                status=status,
                max_metric_file_bytes=max_metric_file_bytes,
            )
        except OSError as exc:
            issues.append(_issue(directory, f"Could not inspect directory: {exc}"))
            continue
        if candidate is None or (require_metrics and not candidate.metrics):
            continue
        run_id = _unique_run_id(base_id, seen)
        candidates.append(candidate.model_copy(update={"run_id": run_id}))

    if not candidates:
        issues.append(_issue(root, "No adoptable output directories were found."))

    candidate_issues = tuple(issue for candidate in candidates for issue in candidate.issues)
    all_issues = tuple(issues) + candidate_issues
    return WorkbenchAdoptionPlan(
        source_root=root,
        max_depth=max_depth,
        spec_type=spec_type,
        status=status,
        run_id_prefix=run_id_prefix,
        require_metrics=require_metrics,
        candidate_count=len(candidates),
        candidates=tuple(candidates),
        issue_count=len(all_issues),
        issues=all_issues,
    )


def write_adoption_workspace(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    max_depth: int = DEFAULT_ADOPTION_MAX_DEPTH,
    spec_type: WorkbenchSpecType = "ocscore_ablation",
    status: RunStatus | None = None,
    run_id_prefix: str = "",
    max_metric_file_bytes: int = DEFAULT_MAX_METRIC_FILE_BYTES,
    require_metrics: bool = False,
    overwrite: bool = False,
) -> WorkbenchAdoptionResult:
    '''Write Workbench manifests for existing output directories.

    Parameters
    ----------
    source_root : str or pathlib.Path
        Existing output root to scan. It is never modified.
    destination_root : str or pathlib.Path
        Workbench root where new manifest directories are written.
    max_depth : int
        Maximum directory depth below ``source_root`` to inspect. OCScore
        ablation policy directories under an ``ablations`` container are
        included when the parent run directory or container is reached.
    spec_type : WorkbenchSpecType
        Workbench spec type assigned to adopted run manifests.
    status : RunStatus or None
        Optional status override.
    run_id_prefix : str
        Optional prefix applied to generated run ids.
    max_metric_file_bytes : int
        Maximum metric file size parsed during adoption planning.
    require_metrics : bool
        If True, only directories with at least one parsed metric are included.
    overwrite : bool
        Whether existing destination manifests may be overwritten.

    Returns
    -------
    WorkbenchAdoptionResult
        Summary of written manifests.
    '''

    plan = build_adoption_plan(
        source_root,
        max_depth=max_depth,
        spec_type=spec_type,
        status=status,
        run_id_prefix=run_id_prefix,
        max_metric_file_bytes=max_metric_file_bytes,
        require_metrics=require_metrics,
    )
    destination = Path(destination_root)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Adoption destination must be a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    adopted_runs: list[WorkbenchAdoptedRun] = []
    for candidate in plan.candidates:
        run_dir = _ensure_destination_run_dir(destination, candidate.run_id, overwrite=overwrite)
        run_manifest_path = run_dir / RUN_MANIFEST_FILENAME
        result_manifest_path = run_dir / RESULT_MANIFEST_FILENAME
        run_manifest = RunManifest(
            run_id=candidate.run_id,
            spec_type=candidate.spec_type,
            name=candidate.name,
            status=candidate.status,
            workspace=candidate.workspace,
            log_files=candidate.log_files,
            artifacts=candidate.artifacts,
            metadata=_run_metadata(candidate),
        )
        result_manifest = ResultManifest(
            run_id=candidate.run_id,
            status=candidate.status,
            artifacts=candidate.artifacts,
            metrics=candidate.metrics,
        )
        write_model(run_manifest_path, run_manifest)
        write_model(result_manifest_path, result_manifest)
        adopted_runs.append(
            WorkbenchAdoptedRun(
                source_path=candidate.source_path,
                run_id=candidate.run_id,
                workspace=candidate.workspace,
                run_manifest_path=run_manifest_path,
                result_manifest_path=result_manifest_path,
                metric_count=len(candidate.metrics),
                artifact_count=len(candidate.artifacts),
                log_count=len(candidate.log_files),
            )
        )

    return WorkbenchAdoptionResult(
        source_root=plan.source_root,
        destination_root=destination,
        run_count=len(adopted_runs),
        runs=tuple(adopted_runs),
        issue_count=plan.issue_count,
        issues=plan.issues,
    )


__all__ = [
    "DEFAULT_ADOPTION_MAX_DEPTH",
    "DEFAULT_MAX_METRIC_FILE_BYTES",
    "build_adoption_plan",
    "write_adoption_workspace",
]
