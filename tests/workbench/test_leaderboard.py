#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench metric leaderboards.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import build_metric_leaderboard
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


def test_build_metric_leaderboard_ranks_numeric_metrics(tmp_path) -> None:
    '''Metric leaderboards rank numeric result metrics and keep skipped entries.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    first_dir = tmp_path / "runs" / "first"
    second_dir = tmp_path / "runs" / "second"
    skipped_dir = tmp_path / "runs" / "skipped"
    for path in (first_dir, second_dir, skipped_dir):
        path.mkdir(parents=True)
    (first_dir / "metrics.csv").write_text("auc,0.8\n", encoding="utf-8")

    write_model(
        first_dir / "result_manifest.yml",
        ResultManifest(
            run_id="run-first",
            status="completed",
            metrics={"auc": 0.8, "loss": 0.3},
            artifacts=(ResultArtifact(name="metrics", path="metrics.csv", kind="csv"),),
        ),
    )
    write_model(
        second_dir / "result_manifest.yml",
        ResultManifest(
            run_id="run-second",
            status="completed",
            metrics={"auc": 0.9, "loss": 0.4},
            artifacts=(ResultArtifact(name="missing", path="missing.csv", kind="csv"),),
        ),
    )
    write_model(
        skipped_dir / "result_manifest.yml",
        ResultManifest(run_id="run-skipped", status="failed", metrics={"loss": 1.0}),
    )

    leaderboard = build_metric_leaderboard(tmp_path, metric_name="auc", max_depth=3)

    assert [entry.run_id for entry in leaderboard.ranked_entries] == [
        "run-second",
        "run-first",
    ]
    assert [entry.rank for entry in leaderboard.ranked_entries] == [1, 2]
    assert leaderboard.best_entry.run_id == "run-second"
    assert leaderboard.ranked_entries[0].metric_value == 0.9
    assert leaderboard.ranked_entries[0].missing_artifact_count == 1
    assert [entry.run_id for entry in leaderboard.skipped_entries] == ["run-skipped"]
    assert "Metric not found" in leaderboard.skipped_entries[0].exclusion_reason
    assert leaderboard.issue_count == 0


def test_build_metric_leaderboard_supports_min_mode_and_dotted_metrics(
    tmp_path,
) -> None:
    '''Leaderboards can rank lower-is-better and dotted metrics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    write_model(
        first_dir / "result_manifest.yml",
        ResultManifest(
            run_id="run-first",
            status="completed",
            metrics={"validation": {"loss": 0.2}},
        ),
    )
    write_model(
        second_dir / "result_manifest.yml",
        ResultManifest(
            run_id="run-second",
            status="completed",
            metrics={"validation": {"loss": 0.1}},
        ),
    )

    leaderboard = build_metric_leaderboard(
        tmp_path,
        metric_name="validation.loss",
        mode="min",
        max_depth=2,
    )

    assert [entry.run_id for entry in leaderboard.ranked_entries] == [
        "run-second",
        "run-first",
    ]
    assert leaderboard.best_entry.metric_value == 0.1


def test_build_metric_leaderboard_preserves_invalid_manifest_issues(tmp_path) -> None:
    '''Invalid manifests are reported as issues without aborting the scan.

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

    leaderboard = build_metric_leaderboard(tmp_path, metric_name="auc", max_depth=2)

    assert [entry.run_id for entry in leaderboard.ranked_entries] == ["run-valid"]
    assert leaderboard.issue_count == 1
    assert leaderboard.issues[0].path == bad_dir / "result_manifest.yml"


def test_build_metric_leaderboard_rejects_invalid_inputs(tmp_path) -> None:
    '''Invalid leaderboard options fail before scanning.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(ValueError, match="metric_name"):
        build_metric_leaderboard(tmp_path, metric_name="")
    with pytest.raises(ValueError, match="mode"):
        build_metric_leaderboard(tmp_path, metric_name="auc", mode="median")
