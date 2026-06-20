#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only discovery helpers for Workbench run inventories.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Models import RunInventoryItem
from OCDocker.Workbench.Models import WorkspaceInventory

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

RUN_MANIFEST_FILENAMES = frozenset(
    {"run_manifest.json", "run_manifest.yml", "run_manifest.yaml"}
)
RESULT_MANIFEST_FILENAMES = frozenset(
    {"result_manifest.json", "result_manifest.yml", "result_manifest.yaml"}
)


# Functions
###############################################################################
## Private ##


def _sorted_children(path: Path) -> tuple[Path, ...]:
    '''Return directory children in deterministic order.

    Parameters
    ----------
    path : pathlib.Path
        Directory to inspect.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Sorted children.
    '''

    return tuple(sorted(path.iterdir(), key=lambda child: child.name))


def _iter_limited_files(root: Path, max_depth: int) -> Iterable[Path]:
    '''Yield files below a root without descending beyond a depth limit.

    Parameters
    ----------
    root : pathlib.Path
        Root file or directory.
    max_depth : int
        Maximum directory depth below root to descend.

    Yields
    ------
    pathlib.Path
        Discovered file paths.
    '''

    if max_depth < 0:
        raise ValueError("max_depth must be greater than or equal to zero.")
    if not root.exists():
        raise FileNotFoundError(f"Workbench inventory root does not exist: {root}")
    if root.is_file():
        yield root
        return

    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        for child in reversed(_sorted_children(current)):
            if child.is_file():
                yield child
            elif depth < max_depth and child.is_dir() and not child.is_symlink():
                stack.append((child, depth + 1))


def _manifest_paths(
    root: str | Path, filenames: frozenset[str], *, max_depth: int
) -> tuple[Path, ...]:
    '''Return matching manifest paths below a root.

    Parameters
    ----------
    root : str or pathlib.Path
        Root file or directory.
    filenames : frozenset[str]
        Accepted manifest file names.
    max_depth : int
        Maximum directory depth below root to descend.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Matching manifest paths.
    '''

    root_path = Path(root)
    return tuple(
        path
        for path in _iter_limited_files(root_path, max_depth)
        if path.name.lower() in filenames
    )


def _artifact_check_path(manifest_path: Path, artifact: ResultArtifact) -> Path:
    '''Return the path used to check whether an artifact exists.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Run manifest path containing the artifact declaration.
    artifact : ResultArtifact
        Artifact declaration.

    Returns
    -------
    pathlib.Path
        Absolute artifact path or path relative to the manifest directory.
    '''

    if artifact.path.is_absolute():
        return artifact.path
    return manifest_path.parent / artifact.path


## Public ##


def discover_run_manifest_paths(
    root: str | Path, *, max_depth: int = 6
) -> tuple[Path, ...]:
    '''Discover Workbench run manifest paths below a root.

    Parameters
    ----------
    root : str or pathlib.Path
        Root file or directory to scan.
    max_depth : int
        Maximum directory depth below root to descend.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Discovered run manifest paths.
    '''

    return _manifest_paths(root, RUN_MANIFEST_FILENAMES, max_depth=max_depth)


def discover_result_manifest_paths(
    root: str | Path, *, max_depth: int = 6
) -> tuple[Path, ...]:
    '''Discover Workbench result manifest paths below a root.

    Parameters
    ----------
    root : str or pathlib.Path
        Root file or directory to scan.
    max_depth : int
        Maximum directory depth below root to descend.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Discovered result manifest paths.
    '''

    return _manifest_paths(root, RESULT_MANIFEST_FILENAMES, max_depth=max_depth)


def summarize_run_manifest(path: str | Path) -> RunInventoryItem:
    '''Read one run manifest and return its compact inventory summary.

    Parameters
    ----------
    path : str or pathlib.Path
        Run manifest path.

    Returns
    -------
    RunInventoryItem
        Compact run summary.
    '''

    manifest_path = Path(path)
    manifest = read_run_manifest(manifest_path)
    missing_artifacts = tuple(
        _artifact_check_path(manifest_path, artifact)
        for artifact in manifest.artifacts
        if not _artifact_check_path(manifest_path, artifact).exists()
    )
    return RunInventoryItem(
        manifest_path=manifest_path,
        run_id=manifest.run_id,
        spec_type=manifest.spec_type,
        name=manifest.name,
        status=manifest.status,
        workspace=manifest.workspace,
        updated_at=manifest.updated_at,
        artifact_count=len(manifest.artifacts),
        missing_artifacts=missing_artifacts,
    )


def scan_workspace(root: str | Path, *, max_depth: int = 6) -> WorkspaceInventory:
    '''Build a read-only inventory of Workbench manifests below a root.

    Parameters
    ----------
    root : str or pathlib.Path
        Root file or directory to scan.
    max_depth : int
        Maximum directory depth below root to descend.

    Returns
    -------
    WorkspaceInventory
        Read-only inventory payload for CLI or GUI consumers.
    '''

    root_path = Path(root)
    runs: list[RunInventoryItem] = []
    issues: list[InventoryIssue] = []

    for manifest_path in discover_run_manifest_paths(root_path, max_depth=max_depth):
        try:
            runs.append(summarize_run_manifest(manifest_path))
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))

    result_manifest_paths = discover_result_manifest_paths(
        root_path, max_depth=max_depth
    )
    for result_path in result_manifest_paths:
        try:
            read_result_manifest(result_path)
        except Exception as exc:
            issues.append(InventoryIssue(path=result_path, message=str(exc)))

    return WorkspaceInventory(
        root=root_path,
        max_depth=max_depth,
        runs=tuple(runs),
        result_manifests=result_manifest_paths,
        issues=tuple(issues),
    )


__all__ = [
    "RESULT_MANIFEST_FILENAMES",
    "RUN_MANIFEST_FILENAMES",
    "discover_result_manifest_paths",
    "discover_run_manifest_paths",
    "scan_workspace",
    "summarize_run_manifest",
]
