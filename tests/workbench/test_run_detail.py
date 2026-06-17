#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench aggregate run drill-down helpers.
'''

# Imports
###############################################################################
from __future__ import annotations

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import build_run_detail
from OCDocker.Workbench import write_model

# License
###############################################################################
"""
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
"""

# Functions
###############################################################################
## Private ##


def _write_detail_run(tmp_path, *, with_result: bool = True):
    '''Write a synthetic Workbench run bundle for detail tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    with_result : bool
        Whether to write a result manifest.

    Returns
    -------
    pathlib.Path
        Run directory path.
    '''

    run_dir = tmp_path / "run-detail"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("started\nfinished\n", encoding="utf-8")
    (run_dir / "metrics.csv").write_text("metric,value\nauc,0.92\n", encoding="utf-8")
    write_model(
        run_dir / "run_manifest.yml",
        RunManifest(
            run_id="run-detail",
            spec_type="ocscore_study",
            name="detail-study",
            status="completed",
            workspace=".",
            command=("ocdocker", "ocscore", "train"),
            log_files=("run.log",),
            artifacts=(
                ResultArtifact(
                    name="metrics",
                    path="metrics.csv",
                    kind="csv",
                    role="metrics",
                ),
            ),
        ),
    )
    if with_result:
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(
                run_id="run-detail",
                status="completed",
                artifacts=(
                    ResultArtifact(
                        name="metrics",
                        path="metrics.csv",
                        kind="csv",
                        role="metrics",
                    ),
                ),
                metrics={"auc": 0.92, "loss": 0.13},
            ),
        )
    return run_dir


## Public ##


def test_build_run_detail_combines_status_logs_and_results(tmp_path) -> None:
    '''Run details combine existing read-only helper payloads.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_dir = _write_detail_run(tmp_path)

    detail = build_run_detail(run_dir, lines=1, max_bytes=1024)

    assert detail.run_id == "run-detail"
    assert detail.status == "completed"
    assert detail.status_report.command == ("ocdocker", "ocscore", "train")
    assert detail.log_preview is not None
    assert detail.log_preview.logs[0].text == "finished"
    assert detail.result_summary is not None
    assert detail.result_summary.source_type == "result_manifest"
    assert detail.result_summary.metrics == {"auc": 0.92, "loss": 0.13}
    assert detail.result_summary.existing_artifact_count == 1
    assert detail.issue_count == 0


def test_build_run_detail_falls_back_to_run_manifest_summary(tmp_path) -> None:
    '''Run details summarize run-manifest artifacts before results exist.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_dir = _write_detail_run(tmp_path, with_result=False)

    detail = build_run_detail(run_dir)

    assert detail.status_report.result_manifest_exists is False
    assert detail.result_summary is not None
    assert detail.result_summary.source_type == "run_manifest"
    assert detail.result_summary.metrics == {}
    assert detail.result_summary.artifact_count == 1
    assert detail.issue_count == 0
