#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench read-only result summaries.
'''

# Imports
###############################################################################
from __future__ import annotations

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import summarize_results
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


def test_summarize_results_reports_metrics_and_artifact_existence(tmp_path) -> None:
    '''Result summaries report metrics and existing or missing artifacts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    (tmp_path / "metrics.csv").write_text("metric,value\nauc,0.9\n", encoding="utf-8")
    manifest_path = write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(
            run_id="run-results",
            status="completed",
            artifacts=(
                ResultArtifact(
                    name="metrics", path="metrics.csv", kind="csv", role="table"
                ),
                ResultArtifact(name="report", path="missing.html", kind="html"),
            ),
            metrics={"auc": 0.9},
        ),
    )

    summary = summarize_results(manifest_path)

    assert summary.source_type == "result_manifest"
    assert summary.run_id == "run-results"
    assert summary.metrics == {"auc": 0.9}
    assert summary.artifact_count == 2
    assert summary.existing_artifact_count == 1
    assert summary.missing_artifact_count == 1
    assert summary.artifacts[0].exists is True
    assert summary.artifacts[0].kind == "csv"
    assert summary.artifacts[0].role == "table"
    assert summary.artifacts[1].exists is False


def test_summarize_results_accepts_run_manifest(tmp_path) -> None:
    '''Result summaries can inspect artifacts declared by run manifests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    (tmp_path / "log.txt").write_text("done\n", encoding="utf-8")
    manifest_path = write_model(
        tmp_path / "run_manifest.yml",
        RunManifest(
            run_id="run-fallback",
            spec_type="ocscore_study",
            name="fallback",
            status="completed",
            workspace=tmp_path,
            artifacts=(ResultArtifact(name="log", path="log.txt", kind="log"),),
        ),
    )

    summary = summarize_results(manifest_path)

    assert summary.source_type == "run_manifest"
    assert summary.generated_at is None
    assert summary.metrics == {}
    assert summary.artifact_count == 1
    assert summary.existing_artifact_count == 1
