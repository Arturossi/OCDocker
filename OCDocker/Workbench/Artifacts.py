#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only artifact indexing helpers for Workbench result browsing.
'''

# Imports
###############################################################################
from __future__ import annotations

from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Literal

from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.Models import ArtifactKind
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Models import RunStatus
from OCDocker.Workbench.Models import WorkbenchArtifactEntry
from OCDocker.Workbench.Models import WorkbenchArtifactIndex
from OCDocker.Workbench.Registry import discover_result_manifest_paths
from OCDocker.Workbench.Registry import discover_run_manifest_paths

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

# Type aliases
###############################################################################

ArtifactManifestSource = Literal["run_manifest", "result_manifest"]

# Functions
###############################################################################
## Private ##


def _clean_filter_values(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    '''Normalize optional string filter values.

    Parameters
    ----------
    values : tuple[str, ...] or list[str] or None
        Optional filter values.

    Returns
    -------
    tuple[str, ...]
        Cleaned filter values.
    '''

    if values is None:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


def _artifact_path(manifest_path: Path, artifact: ResultArtifact) -> Path:
    '''Resolve an artifact path relative to its declaring manifest.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Manifest path containing the artifact declaration.
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


def _modified_at(path: Path) -> datetime | None:
    '''Return a file modification timestamp when available.

    Parameters
    ----------
    path : pathlib.Path
        Artifact path.

    Returns
    -------
    datetime or None
        UTC modification timestamp when the path exists.
    '''

    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _size_bytes(path: Path) -> int | None:
    '''Return a file size when available.

    Parameters
    ----------
    path : pathlib.Path
        Artifact path.

    Returns
    -------
    int or None
        File size in bytes when available.
    '''

    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return None
    return None


def _entry_from_artifact(
    *,
    source_type: ArtifactManifestSource,
    source_manifest_path: Path,
    run_id: str,
    status: RunStatus,
    artifact: ResultArtifact,
) -> WorkbenchArtifactEntry:
    '''Build one artifact index entry.

    Parameters
    ----------
    source_type : ArtifactManifestSource
        Source manifest type.
    source_manifest_path : pathlib.Path
        Source manifest path.
    run_id : str
        Run id.
    status : RunStatus
        Run status.
    artifact : ResultArtifact
        Artifact declaration.

    Returns
    -------
    WorkbenchArtifactEntry
        Artifact index entry.
    '''

    path = _artifact_path(source_manifest_path, artifact)
    exists = path.exists()
    is_file = path.is_file()
    is_dir = path.is_dir()
    return WorkbenchArtifactEntry(
        source_type=source_type,
        source_manifest_path=source_manifest_path,
        run_id=run_id,
        status=status,
        name=artifact.name,
        path=path,
        kind=artifact.kind,
        role=artifact.role,
        description=artifact.description,
        exists=exists,
        is_file=is_file,
        is_dir=is_dir,
        suffix=path.suffix.lower(),
        size_bytes=_size_bytes(path),
        modified_at=_modified_at(path) if exists else None,
    )


def _matches_filters(
    entry: WorkbenchArtifactEntry,
    *,
    kinds: tuple[str, ...],
    roles: tuple[str, ...],
    require_existing: bool,
) -> bool:
    '''Return whether an artifact entry matches requested filters.

    Parameters
    ----------
    entry : WorkbenchArtifactEntry
        Artifact entry.
    kinds : tuple[str, ...]
        Accepted artifact kinds.
    roles : tuple[str, ...]
        Accepted artifact roles.
    require_existing : bool
        If True, include only artifacts that exist on disk.

    Returns
    -------
    bool
        True when the entry should be included.
    '''

    if kinds and entry.kind not in kinds:
        return False
    if roles and entry.role not in roles:
        return False
    if require_existing and not entry.exists:
        return False
    return True


def _count_by(
    entries: tuple[WorkbenchArtifactEntry, ...], field_name: str
) -> dict[str, int]:
    '''Count artifact entries by a string field.

    Parameters
    ----------
    entries : tuple[WorkbenchArtifactEntry, ...]
        Artifact entries.
    field_name : str
        Field name to count.

    Returns
    -------
    dict[str, int]
        Field value counts.
    '''

    counts: dict[str, int] = {}
    for entry in entries:
        value = str(getattr(entry, field_name) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


## Public ##


def build_artifact_index(
    root: str | Path,
    *,
    kinds: tuple[str, ...] | list[str] | None = None,
    roles: tuple[str, ...] | list[str] | None = None,
    require_existing: bool = False,
    max_depth: int = 6,
) -> WorkbenchArtifactIndex:
    '''Build a read-only cross-run artifact index from Workbench manifests.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or manifest file to scan.
    kinds : tuple[str, ...] or list[str] or None
        Optional artifact kinds to include.
    roles : tuple[str, ...] or list[str] or None
        Optional artifact roles to include.
    require_existing : bool
        If True, include only artifacts that currently exist on disk.
    max_depth : int
        Maximum directory depth below root to scan.

    Returns
    -------
    WorkbenchArtifactIndex
        Artifact browser payload for GUI, CLI, and automation consumers.
    '''

    root_path = Path(root)
    kind_filters = _clean_filter_values(kinds)
    role_filters = _clean_filter_values(roles)
    entries: list[WorkbenchArtifactEntry] = []
    issues: list[InventoryIssue] = []
    run_manifest_count = 0
    result_manifest_count = 0

    for manifest_path in discover_run_manifest_paths(root_path, max_depth=max_depth):
        try:
            manifest = read_run_manifest(manifest_path)
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))
            continue
        run_manifest_count += 1
        for artifact in manifest.artifacts:
            entry = _entry_from_artifact(
                source_type="run_manifest",
                source_manifest_path=manifest_path,
                run_id=manifest.run_id,
                status=manifest.status,
                artifact=artifact,
            )
            if _matches_filters(
                entry,
                kinds=kind_filters,
                roles=role_filters,
                require_existing=require_existing,
            ):
                entries.append(entry)

    for manifest_path in discover_result_manifest_paths(root_path, max_depth=max_depth):
        try:
            manifest = read_result_manifest(manifest_path)
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))
            continue
        result_manifest_count += 1
        for artifact in manifest.artifacts:
            entry = _entry_from_artifact(
                source_type="result_manifest",
                source_manifest_path=manifest_path,
                run_id=manifest.run_id,
                status=manifest.status,
                artifact=artifact,
            )
            if _matches_filters(
                entry,
                kinds=kind_filters,
                roles=role_filters,
                require_existing=require_existing,
            ):
                entries.append(entry)

    artifact_entries = tuple(
        sorted(
            entries,
            key=lambda entry: (
                entry.run_id,
                entry.source_type,
                entry.role,
                entry.kind,
                entry.name,
                str(entry.path),
            ),
        )
    )
    existing_count = sum(1 for entry in artifact_entries if entry.exists)
    return WorkbenchArtifactIndex(
        root=root_path,
        max_depth=max_depth,
        filters={
            "kinds": kind_filters,
            "roles": role_filters,
            "require_existing": require_existing,
        },
        run_manifest_count=run_manifest_count,
        result_manifest_count=result_manifest_count,
        artifact_count=len(artifact_entries),
        existing_artifact_count=existing_count,
        missing_artifact_count=len(artifact_entries) - existing_count,
        kind_counts=_count_by(artifact_entries, "kind"),
        role_counts=_count_by(artifact_entries, "role"),
        entries=artifact_entries,
        issue_count=len(issues),
        issues=tuple(issues),
    )


__all__ = ["ArtifactManifestSource", "build_artifact_index"]
