#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench single-run status inspection.
'''

# Imports
###############################################################################
from __future__ import annotations

import os

import pytest

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import inspect_run_status
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


def test_inspect_run_status_reports_paths_and_result_manifest(tmp_path) -> None:
    '''Run status inspection reports manifest paths without executing anything.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "run.log").write_text("started\n", encoding="utf-8")
    (tmp_path / "metrics.csv").write_text("metric,value\nauc,0.9\n", encoding="utf-8")
    write_model(
        tmp_path / "run_manifest.yml",
        RunManifest(
            run_id="run-status",
            spec_type="ocscore_study",
            name="status-study",
            status="running",
            workspace=".",
            command=("ocdocker", "ocscore", "train"),
            pid=os.getpid(),
            log_files=("logs/run.log", "logs/missing.log"),
            artifacts=(
                ResultArtifact(name="metrics", path="metrics.csv", kind="csv"),
                ResultArtifact(name="report", path="missing.html", kind="html"),
            ),
        ),
    )
    write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(run_id="run-status", status="completed"),
    )

    report = inspect_run_status(tmp_path)

    assert report.run_id == "run-status"
    assert report.status == "running"
    assert report.workspace_status.exists is True
    assert report.pid_alive is True
    assert report.result_manifest_exists is True
    assert report.result_manifest_path == tmp_path / "result_manifest.yml"
    assert [item.exists for item in report.log_files] == [True, False]
    assert [item.exists for item in report.artifacts] == [True, False]
    assert report.artifacts[0].role == "csv"


def test_inspect_run_status_requires_manifest_in_directory(tmp_path) -> None:
    '''Status inspection reports missing direct run manifests clearly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(FileNotFoundError, match="No Workbench run manifest"):
        inspect_run_status(tmp_path)
