#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench publication exports.
'''

# Imports
###############################################################################
from __future__ import annotations

import json

import pytest

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import build_publication_export
from OCDocker.Workbench import write_model

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

# Functions
###############################################################################
## Private ##


def _write_result_manifest(tmp_path) -> tuple:
    '''Write a result manifest with existing and missing artifacts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    tuple
        Manifest path and existing artifact path.
    '''

    artifact_path = tmp_path / "metrics.csv"
    artifact_path.write_text("metric,value\nauc,0.9\n", encoding="utf-8")
    manifest = ResultManifest(
        run_id="run-export",
        status="completed",
        artifacts=(
            ResultArtifact(
                name="metrics", path="metrics.csv", kind="csv", role="table"
            ),
            ResultArtifact(
                name="missing", path="missing.html", kind="html", role="report"
            ),
        ),
        metrics={"auc": 0.9},
    )
    manifest_path = write_model(tmp_path / "result_manifest.yml", manifest)
    return manifest_path, artifact_path


## Public ##


def test_build_publication_export_manifest_only(tmp_path) -> None:
    '''Publication exports create metadata and README without copying by default.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manifest_path, _ = _write_result_manifest(tmp_path)
    export_dir = tmp_path / "export"

    export = build_publication_export(manifest_path, export_dir)

    assert export.publication_manifest_path.is_file()
    assert export.readme_path.is_file()
    assert export.artifacts[0].exists is True
    assert export.artifacts[0].copied is False
    assert export.artifacts[0].export_path is None
    assert export.artifacts[1].exists is False
    payload = json.loads(export.publication_manifest_path.read_text(encoding="utf-8"))
    assert payload["metrics"] == {"auc": 0.9}
    assert "run-export" in export.readme_path.read_text(encoding="utf-8")


def test_build_publication_export_can_copy_declared_artifacts(tmp_path) -> None:
    '''Publication exports copy existing declared artifacts on request.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manifest_path, artifact_path = _write_result_manifest(tmp_path)
    export_dir = tmp_path / "export"

    export = build_publication_export(manifest_path, export_dir, copy_artifacts=True)

    assert export.artifacts[0].copied is True
    assert export.artifacts[0].export_path is not None
    assert export.artifacts[0].export_path.read_text(
        encoding="utf-8"
    ) == artifact_path.read_text(encoding="utf-8")
    assert export.artifacts[1].copied is False

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_publication_export(manifest_path, export_dir)

    overwritten = build_publication_export(manifest_path, export_dir, overwrite=True)
    assert overwritten.publication_manifest_path.is_file()
