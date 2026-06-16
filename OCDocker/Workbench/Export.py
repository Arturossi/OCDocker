#!/usr/bin/env python3

# Description
###############################################################################
'''
Publishable export helpers for Workbench manifests.
'''

# Imports
###############################################################################
from __future__ import annotations

import json
import re
import shutil

from pathlib import Path
from typing import Any

from OCDocker.Workbench.IO import model_to_data
from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.Models import ExportedArtifact
from OCDocker.Workbench.Models import PublicationExport
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Models import RunStatus

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

PUBLICATION_MANIFEST_FILENAME = "publication_manifest.json"
PUBLICATION_README_FILENAME = "README.md"
PUBLICATION_ARTIFACT_DIRNAME = "artifacts"


# Functions
###############################################################################
## Private ##


def _load_manifest(
    path: str | Path,
) -> tuple[str, RunStatus, tuple[ResultArtifact, ...], dict[str, Any]]:
    '''Load a result or run manifest and return export-relevant fields.

    Parameters
    ----------
    path : str or pathlib.Path
        Manifest path.

    Returns
    -------
    tuple[str, RunStatus, tuple[ResultArtifact, ...], dict[str, Any]]
        Run id, status, artifacts, and metrics.
    '''

    manifest_path = Path(path)
    try:
        result_manifest = read_result_manifest(manifest_path)
    except Exception:
        run_manifest = read_run_manifest(manifest_path)
        return run_manifest.run_id, run_manifest.status, run_manifest.artifacts, {}
    return (
        result_manifest.run_id,
        result_manifest.status,
        result_manifest.artifacts,
        result_manifest.metrics,
    )


def _artifact_source_path(manifest_path: Path, artifact: ResultArtifact) -> Path:
    '''Resolve an artifact path relative to its manifest.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Source manifest path.
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


def _artifact_export_name(
    index: int, artifact: ResultArtifact, source_path: Path
) -> str:
    '''Build a stable export filename for an artifact.

    Parameters
    ----------
    index : int
        One-based artifact index.
    artifact : ResultArtifact
        Artifact declaration.
    source_path : pathlib.Path
        Resolved artifact source path.

    Returns
    -------
    str
        Safe export filename.
    '''

    base = source_path.name or artifact.name
    if not base:
        base = f"artifact-{index}"
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not safe_base:
        safe_base = f"artifact-{index}"
    return f"{index:03d}-{safe_base}"


def _copy_artifact(
    source_path: Path, destination_path: Path, *, overwrite: bool
) -> None:
    '''Copy one artifact file or directory into the export directory.

    Parameters
    ----------
    source_path : pathlib.Path
        Existing source artifact.
    destination_path : pathlib.Path
        Destination artifact path.
    overwrite : bool
        If True, overwrite an existing destination.
    '''

    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(f"Export artifact already exists: {destination_path}")
        if destination_path.is_dir():
            shutil.rmtree(destination_path)
        else:
            destination_path.unlink()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        shutil.copytree(source_path, destination_path)
    else:
        shutil.copy2(source_path, destination_path)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    '''Write a JSON payload to disk.

    Parameters
    ----------
    path : pathlib.Path
        Output path.
    payload : dict[str, Any]
        JSON-compatible payload.

    Returns
    -------
    pathlib.Path
        Written path.
    '''

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _readme_text(export: PublicationExport) -> str:
    '''Build a compact README for a publication export.

    Parameters
    ----------
    export : PublicationExport
        Export summary.

    Returns
    -------
    str
        Markdown README text.
    '''

    lines = [
        f"# OCDocker Workbench Export: {export.run_id}",
        "",
        f"- Run ID: `{export.run_id}`",
        f"- Status: `{export.status}`",
        f"- Source manifest: `{export.source_manifest_path}`",
        "",
        "## Artifacts",
        "",
    ]
    if not export.artifacts:
        lines.append("No artifacts were declared in the source manifest.")
    else:
        lines.append("| Name | Kind | Role | Exists | Copied | Path |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for artifact in export.artifacts:
            path = (
                artifact.export_path
                if artifact.export_path is not None
                else artifact.source_path
            )
            lines.append(
                "| "
                f"{artifact.name} | {artifact.kind} | {artifact.role} | "
                f"{artifact.exists} | {artifact.copied} | `{path}` |"
            )
    if export.metrics:
        lines.extend(["", "## Metrics", "", "```json"])
        lines.append(json.dumps(export.metrics, indent=2, sort_keys=True))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


## Public ##


def build_publication_export(
    manifest_path: str | Path,
    export_dir: str | Path,
    *,
    copy_artifacts: bool = False,
    overwrite: bool = False,
) -> PublicationExport:
    '''Build a publishable export scaffold from a run or result manifest.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Source run or result manifest path.
    export_dir : str or pathlib.Path
        Export directory to create or update.
    copy_artifacts : bool
        If True, copy declared existing artifacts into ``artifacts/``.
    overwrite : bool
        If True, overwrite existing export files and copied artifacts.

    Returns
    -------
    PublicationExport
        Export summary.
    '''

    source_manifest_path = Path(manifest_path)
    root = Path(export_dir)
    readme_path = root / PUBLICATION_README_FILENAME
    publication_manifest_path = root / PUBLICATION_MANIFEST_FILENAME
    protected_outputs = (readme_path, publication_manifest_path)
    existing = tuple(path for path in protected_outputs if path.exists())
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing export file(s): {joined}"
        )

    run_id, status, artifacts, metrics = _load_manifest(source_manifest_path)
    root.mkdir(parents=True, exist_ok=True)
    exported_artifacts: list[ExportedArtifact] = []
    artifact_root = root / PUBLICATION_ARTIFACT_DIRNAME
    for index, artifact in enumerate(artifacts, start=1):
        source_path = _artifact_source_path(source_manifest_path, artifact)
        exists = source_path.exists()
        export_path: Path | None = None
        copied = False
        if copy_artifacts and exists:
            export_path = artifact_root / _artifact_export_name(
                index, artifact, source_path
            )
            _copy_artifact(source_path, export_path, overwrite=overwrite)
            copied = True
        exported_artifacts.append(
            ExportedArtifact(
                name=artifact.name,
                source_path=source_path,
                export_path=export_path,
                kind=artifact.kind,
                role=artifact.role,
                description=artifact.description,
                exists=exists,
                copied=copied,
            )
        )

    export = PublicationExport(
        root=root,
        source_manifest_path=source_manifest_path,
        run_id=run_id,
        status=status,
        readme_path=readme_path,
        publication_manifest_path=publication_manifest_path,
        artifacts=tuple(exported_artifacts),
        metrics=metrics,
    )
    readme_path.write_text(_readme_text(export), encoding="utf-8")
    _write_json(publication_manifest_path, model_to_data(export))
    return export


__all__ = [
    "PUBLICATION_ARTIFACT_DIRNAME",
    "PUBLICATION_MANIFEST_FILENAME",
    "PUBLICATION_README_FILENAME",
    "build_publication_export",
]
