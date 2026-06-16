#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only status inspection helpers for Workbench run bundles.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path

from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Models import RunPathStatus
from OCDocker.Workbench.Models import RunStatusReport
from OCDocker.Workbench.Registry import RESULT_MANIFEST_FILENAMES
from OCDocker.Workbench.Registry import RUN_MANIFEST_FILENAMES

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

# Functions
###############################################################################
## Private ##


def _direct_manifest_paths(
    directory: Path, filenames: frozenset[str]
) -> tuple[Path, ...]:
    '''Return direct child manifest paths that match accepted file names.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to inspect.
    filenames : frozenset[str]
        Accepted manifest file names.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Matching direct child manifest paths.
    '''

    return tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.name.lower() in filenames
            ),
            key=lambda path: path.name,
        )
    )


def _resolve_run_manifest_path(target: str | Path) -> Path:
    '''Resolve a run manifest path from a manifest file or bundle directory.

    Parameters
    ----------
    target : str or pathlib.Path
        Run manifest file or bundle directory.

    Returns
    -------
    pathlib.Path
        Resolved run manifest path.
    '''

    target_path = Path(target)
    if target_path.is_file():
        return target_path
    if not target_path.exists():
        raise FileNotFoundError(
            f"Workbench status target does not exist: {target_path}"
        )
    if not target_path.is_dir():
        raise ValueError(
            f"Workbench status target is not a file or directory: {target_path}"
        )

    candidates = _direct_manifest_paths(target_path, RUN_MANIFEST_FILENAMES)
    if not candidates:
        raise FileNotFoundError(
            f"No Workbench run manifest found directly in directory: {target_path}"
        )
    return candidates[0]


def _resolve_path(base_path: Path, path: Path) -> Path:
    '''Resolve a manifest-relative path without touching the filesystem.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative paths.
    path : pathlib.Path
        Path to resolve.

    Returns
    -------
    pathlib.Path
        Absolute or manifest-relative resolved path.
    '''

    if path.is_absolute():
        return path
    return base_path / path


def _path_status(
    base_path: Path, path: Path, *, name: str = "", role: str = ""
) -> RunPathStatus:
    '''Build filesystem status for a path referenced by a run manifest.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative paths.
    path : pathlib.Path
        Path to inspect.
    name : str
        Display name for the path.
    role : str
        Semantic role for the path.

    Returns
    -------
    RunPathStatus
        Read-only path status.
    '''

    resolved = _resolve_path(base_path, path)
    return RunPathStatus(
        path=resolved,
        exists=resolved.exists(),
        is_file=resolved.is_file(),
        is_dir=resolved.is_dir(),
        name=name,
        role=role,
    )


def _artifact_status(base_path: Path, artifact: ResultArtifact) -> RunPathStatus:
    '''Build filesystem status for a declared result artifact.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative artifact paths.
    artifact : ResultArtifact
        Artifact declaration.

    Returns
    -------
    RunPathStatus
        Read-only artifact path status.
    '''

    return _path_status(
        base_path,
        artifact.path,
        name=artifact.name,
        role=artifact.role or artifact.kind,
    )


def _pid_alive(pid: int | None) -> bool | None:
    '''Return whether a process id appears alive on Linux-like systems.

    Parameters
    ----------
    pid : int or None
        Process id recorded in a run manifest.

    Returns
    -------
    bool or None
        True or False when ``/proc`` is available, otherwise None.
    '''

    if pid is None:
        return None
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    try:
        return (proc_root / str(pid)).exists()
    except OSError:
        return None


def _result_manifest_path(run_dir: Path) -> Path | None:
    '''Return the first direct result manifest found beside a run manifest.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory containing a run manifest.

    Returns
    -------
    pathlib.Path or None
        Result manifest path when present.
    '''

    candidates = _direct_manifest_paths(run_dir, RESULT_MANIFEST_FILENAMES)
    if not candidates:
        return None
    return candidates[0]


## Public ##


def inspect_run_status(target: str | Path) -> RunStatusReport:
    '''Inspect one Workbench run manifest without executing or controlling it.

    Parameters
    ----------
    target : str or pathlib.Path
        Run manifest file or prepared bundle directory.

    Returns
    -------
    RunStatusReport
        Read-only status report for GUI, CLI, or automation consumers.
    '''

    manifest_path = _resolve_run_manifest_path(target)
    manifest = read_run_manifest(manifest_path)
    base_path = manifest_path.parent
    result_path = _result_manifest_path(base_path)
    workspace_status = _path_status(
        base_path,
        manifest.workspace,
        name="workspace",
        role="workspace",
    )

    return RunStatusReport(
        manifest_path=manifest_path,
        run_id=manifest.run_id,
        spec_type=manifest.spec_type,
        name=manifest.name,
        status=manifest.status,
        workspace=workspace_status.path,
        workspace_status=workspace_status,
        updated_at=manifest.updated_at,
        command=manifest.command,
        pid=manifest.pid,
        pid_alive=_pid_alive(manifest.pid),
        result_manifest_path=result_path,
        result_manifest_exists=result_path is not None,
        log_files=tuple(
            _path_status(base_path, path, name=path.name, role="log")
            for path in manifest.log_files
        ),
        artifacts=tuple(
            _artifact_status(base_path, artifact) for artifact in manifest.artifacts
        ),
    )


__all__ = ["inspect_run_status"]
