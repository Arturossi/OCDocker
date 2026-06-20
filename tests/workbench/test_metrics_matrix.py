#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench metric matrix payloads.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import build_metric_matrix
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
## Public ##


def test_build_metric_matrix_discovers_flattened_numeric_metrics(tmp_path) -> None:
    '''Metric matrices flatten metrics and preserve row diagnostics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "metrics.csv").write_text("metric,value\nauc,0.8\n", encoding="utf-8")
    write_model(
        first_dir / "result_manifest.yml",
        ResultManifest(
            run_id="run-first",
            status="completed",
            metrics={"auc": 0.8, "validation": {"loss": 0.3}, "label": "ok"},
            artifacts=(ResultArtifact(name="metrics", path="metrics.csv", kind="csv"),),
        ),
    )
    write_model(
        second_dir / "result_manifest.yml",
        ResultManifest(
            run_id="run-second",
            status="completed",
            metrics={"auc": 0.9, "validation": {"loss": 0.2}},
            artifacts=(ResultArtifact(name="missing", path="missing.csv", kind="csv"),),
        ),
    )

    matrix = build_metric_matrix(tmp_path, max_depth=2)

    assert matrix.metric_names == ("auc", "label", "validation.loss")
    assert matrix.result_manifest_count == 2
    assert [row.run_id for row in matrix.rows] == ["run-first", "run-second"]
    assert matrix.rows[0].metric_values == {"auc": 0.8, "validation.loss": 0.3}
    assert matrix.rows[0].non_numeric_metrics == ("label",)
    assert matrix.rows[1].missing_metrics == ("label",)
    assert matrix.rows[1].missing_artifact_count == 1
    assert matrix.issue_count == 0


def test_build_metric_matrix_respects_selected_metrics(tmp_path) -> None:
    '''Metric matrices can restrict columns to requested metrics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(
            run_id="run-selected",
            status="completed",
            metrics={"auc": 0.85, "validation": {"loss": 0.25}},
        ),
    )

    matrix = build_metric_matrix(
        tmp_path,
        metric_names=("validation.loss", "missing"),
        max_depth=1,
    )

    assert matrix.metric_names == ("validation.loss", "missing")
    assert matrix.rows[0].metric_values == {"validation.loss": 0.25}
    assert matrix.rows[0].missing_metrics == ("missing",)


def test_build_metric_matrix_reports_invalid_manifest_issues(tmp_path) -> None:
    '''Invalid result manifests are preserved as non-fatal issues.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    valid_dir = tmp_path / "valid"
    bad_dir = tmp_path / "bad"
    valid_dir.mkdir()
    bad_dir.mkdir()
    write_model(
        valid_dir / "result_manifest.yml",
        ResultManifest(run_id="run-valid", status="completed", metrics={"auc": 0.7}),
    )
    (bad_dir / "result_manifest.yml").write_text("run_id: broken\n", encoding="utf-8")

    matrix = build_metric_matrix(tmp_path, max_depth=2)

    assert [row.run_id for row in matrix.rows] == ["run-valid"]
    assert matrix.issue_count == 1
    assert matrix.issues[0].path == bad_dir / "result_manifest.yml"


def test_build_metric_matrix_rejects_duplicate_metric_selection(tmp_path) -> None:
    '''Duplicate selected metric names are rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(ValueError, match="unique"):
        build_metric_matrix(tmp_path, metric_names=("auc", "auc"))
