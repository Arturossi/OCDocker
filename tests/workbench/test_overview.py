#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench workspace overview payloads.
'''

# Imports
###############################################################################
from __future__ import annotations

from datetime import datetime
from datetime import timezone

import pytest

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import build_workspace_overview
from OCDocker.Workbench import write_model

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
## Public ##


def test_build_workspace_overview_counts_runs_and_recent_items(tmp_path) -> None:
    '''Workspace overviews include counts, issues, and recent run summaries.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_a = tmp_path / "runs" / "a"
    run_b = tmp_path / "runs" / "b"
    bad = tmp_path / "runs" / "bad"
    for path in (run_a, run_b, bad):
        path.mkdir(parents=True)

    write_model(
        run_a / "run_manifest.yml",
        RunManifest(
            run_id="run-a",
            spec_type="ocscore_study",
            name="first",
            status="completed",
            workspace=run_a,
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            artifacts=(ResultArtifact(name="missing", path="missing.csv", kind="csv"),),
        ),
    )
    write_model(
        run_b / "run_manifest.yml",
        RunManifest(
            run_id="run-b",
            spec_type="vs_campaign",
            name="second",
            status="running",
            workspace=run_b,
            updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
    )
    write_model(
        run_b / "result_manifest.yml",
        ResultManifest(run_id="run-b", status="running"),
    )
    (bad / "run_manifest.yml").write_text("run_id: broken\n", encoding="utf-8")

    overview = build_workspace_overview(tmp_path, max_depth=3, recent_limit=1)

    assert overview.run_count == 2
    assert overview.result_manifest_count == 1
    assert overview.issue_count == 1
    assert overview.missing_artifact_count == 1
    assert overview.status_counts["completed"] == 1
    assert overview.status_counts["running"] == 1
    assert overview.status_counts["failed"] == 0
    assert overview.spec_type_counts["ocscore_study"] == 1
    assert overview.spec_type_counts["vs_campaign"] == 1
    assert [run.run_id for run in overview.recent_runs] == ["run-b"]
    assert overview.issues[0].path == bad / "run_manifest.yml"


def test_build_workspace_overview_rejects_empty_recent_limit(tmp_path) -> None:
    '''Recent-run limits must be positive.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(ValueError, match="recent_limit"):
        build_workspace_overview(tmp_path, recent_limit=0)
