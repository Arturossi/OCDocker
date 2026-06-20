#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench artifact index payloads.
'''

# Imports
###############################################################################
from __future__ import annotations

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import build_artifact_index
from OCDocker.Workbench import write_model

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Public ##


def test_build_artifact_index_scans_run_and_result_manifests(tmp_path) -> None:
    '''Artifact indexes combine run and result manifest declarations.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    existing = run_dir / "metrics.csv"
    existing.write_text("metric,value\nauc,0.9\n", encoding="utf-8")
    write_model(
        run_dir / "run_manifest.yml",
        RunManifest(
            run_id="run-a",
            spec_type="ocscore_study",
            name="run-a",
            status="completed",
            workspace=run_dir,
            artifacts=(
                ResultArtifact(name="log", path="missing.log", kind="log", role="log"),
            ),
        ),
    )
    write_model(
        run_dir / "result_manifest.yml",
        ResultManifest(
            run_id="run-a",
            status="completed",
            artifacts=(
                ResultArtifact(
                    name="metrics",
                    path="metrics.csv",
                    kind="csv",
                    role="metrics",
                    description="Metric table",
                ),
            ),
        ),
    )

    index = build_artifact_index(tmp_path, max_depth=2)

    assert index.run_manifest_count == 1
    assert index.result_manifest_count == 1
    assert index.artifact_count == 2
    assert index.existing_artifact_count == 1
    assert index.missing_artifact_count == 1
    assert index.kind_counts == {"csv": 1, "log": 1}
    assert index.role_counts == {"log": 1, "metrics": 1}
    entries = {entry.name: entry for entry in index.entries}
    assert entries["metrics"].exists is True
    assert entries["metrics"].is_file is True
    assert entries["metrics"].size_bytes == existing.stat().st_size
    assert entries["metrics"].suffix == ".csv"
    assert entries["metrics"].modified_at is not None
    assert entries["log"].exists is False
    assert entries["log"].source_type == "run_manifest"


def test_build_artifact_index_filters_kind_role_and_existing(tmp_path) -> None:
    '''Artifact indexes support GUI-side kind, role, and existence filters.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "figure.png").write_text("png", encoding="utf-8")
    write_model(
        run_dir / "result_manifest.yml",
        ResultManifest(
            run_id="run-a",
            status="completed",
            artifacts=(
                ResultArtifact(
                    name="figure", path="figure.png", kind="image", role="plot"
                ),
                ResultArtifact(
                    name="table", path="table.csv", kind="csv", role="metrics"
                ),
            ),
        ),
    )

    index = build_artifact_index(
        tmp_path,
        kinds=("image",),
        roles=("plot",),
        require_existing=True,
        max_depth=2,
    )

    assert index.artifact_count == 1
    assert index.entries[0].name == "figure"
    assert index.filters == {
        "kinds": ("image",),
        "roles": ("plot",),
        "require_existing": True,
    }
