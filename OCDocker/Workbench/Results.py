#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only result summarization helpers for Workbench manifests.
'''

# Imports
###############################################################################
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Models import ResultArtifactStatus
from OCDocker.Workbench.Models import ResultSummary
from OCDocker.Workbench.Models import RunStatus

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

# Type aliases
###############################################################################

ManifestSourceType = Literal["run_manifest", "result_manifest"]


# Functions
###############################################################################
## Private ##


def _load_manifest_fields(
    path: str | Path,
) -> tuple[
    ManifestSourceType,
    str,
    RunStatus,
    tuple[ResultArtifact, ...],
    dict[str, Any],
    datetime | None,
]:
    '''Load a result or run manifest and return result-summary fields.

    Parameters
    ----------
    path : str or pathlib.Path
        Manifest path to read.

    Returns
    -------
    tuple
        Source type, run id, status, artifacts, metrics, and generated timestamp.
    '''

    manifest_path = Path(path)
    try:
        result_manifest = read_result_manifest(manifest_path)
    except Exception:
        run_manifest = read_run_manifest(manifest_path)
        return (
            "run_manifest",
            run_manifest.run_id,
            run_manifest.status,
            run_manifest.artifacts,
            {},
            None,
        )
    return (
        "result_manifest",
        result_manifest.run_id,
        result_manifest.status,
        result_manifest.artifacts,
        result_manifest.metrics,
        result_manifest.generated_at,
    )


def _artifact_path(manifest_path: Path, artifact: ResultArtifact) -> Path:
    '''Resolve an artifact path relative to its manifest directory.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Manifest path containing the artifact declaration.
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


def _artifact_status(
    manifest_path: Path, artifact: ResultArtifact
) -> ResultArtifactStatus:
    '''Build read-only filesystem status for a declared result artifact.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Manifest path containing the artifact declaration.
    artifact : ResultArtifact
        Artifact declaration.

    Returns
    -------
    ResultArtifactStatus
        Artifact status payload.
    '''

    path = _artifact_path(manifest_path, artifact)
    return ResultArtifactStatus(
        path=path,
        exists=path.exists(),
        is_file=path.is_file(),
        is_dir=path.is_dir(),
        name=artifact.name,
        role=artifact.role,
        kind=artifact.kind,
        description=artifact.description,
    )


## Public ##


def summarize_results(manifest_path: str | Path) -> ResultSummary:
    '''Summarize declared artifacts and metrics without exporting or copying.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Run or result manifest path.

    Returns
    -------
    ResultSummary
        Read-only result summary for GUI, CLI, or automation consumers.
    '''

    source_manifest_path = Path(manifest_path)
    source_type, run_id, status, artifacts, metrics, generated_at = (
        _load_manifest_fields(source_manifest_path)
    )
    artifact_statuses = tuple(
        _artifact_status(source_manifest_path, artifact) for artifact in artifacts
    )
    existing_count = sum(1 for artifact in artifact_statuses if artifact.exists)
    return ResultSummary(
        source_manifest_path=source_manifest_path,
        source_type=source_type,
        run_id=run_id,
        status=status,
        generated_at=generated_at,
        metrics=metrics,
        artifacts=artifact_statuses,
        artifact_count=len(artifact_statuses),
        existing_artifact_count=existing_count,
        missing_artifact_count=len(artifact_statuses) - existing_count,
    )


__all__ = ["ManifestSourceType", "summarize_results"]
