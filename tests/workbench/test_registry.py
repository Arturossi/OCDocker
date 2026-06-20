#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench run inventory discovery.
'''

# Imports
###############################################################################
from __future__ import annotations

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import discover_run_manifest_paths
from OCDocker.Workbench import scan_workspace
from OCDocker.Workbench import summarize_run_manifest
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


def test_discover_run_manifest_paths_respects_depth(tmp_path) -> None:
    '''Manifest discovery obeys the configured directory depth.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    shallow = tmp_path / "run_manifest.yml"
    deep = tmp_path / "one" / "two" / "run_manifest.yml"
    shallow.write_text("run_id: invalid\n", encoding="utf-8")
    deep.parent.mkdir(parents=True)
    deep.write_text("run_id: invalid\n", encoding="utf-8")

    assert discover_run_manifest_paths(tmp_path, max_depth=0) == (shallow,)
    assert discover_run_manifest_paths(tmp_path, max_depth=2) == (shallow, deep)


def test_summarize_run_manifest_reports_missing_artifacts(tmp_path) -> None:
    '''Run summaries include artifact counts and missing artifact paths.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    existing = tmp_path / "metrics.csv"
    existing.write_text("metric,value\nauc,0.9\n", encoding="utf-8")
    manifest = RunManifest(
        run_id="run-001",
        spec_type="ocscore_study",
        name="inventory-study",
        status="completed",
        workspace=tmp_path,
        artifacts=(
            ResultArtifact(name="metrics", path="metrics.csv", kind="csv"),
            ResultArtifact(name="report", path="missing.html", kind="html"),
        ),
    )
    manifest_path = write_model(tmp_path / "run_manifest.yml", manifest)

    summary = summarize_run_manifest(manifest_path)

    assert summary.run_id == "run-001"
    assert summary.artifact_count == 2
    assert summary.missing_artifacts == (tmp_path / "missing.html",)


def test_scan_workspace_collects_runs_results_and_issues(tmp_path) -> None:
    '''Workspace scans summarize valid manifests and preserve invalid issues.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_dir = tmp_path / "runs" / "study"
    run_dir.mkdir(parents=True)
    manifest = RunManifest(
        run_id="run-002",
        spec_type="ocscore_study",
        name="scan-study",
        workspace=run_dir,
        command=("ocdocker", "ocscore", "train"),
    )
    write_model(run_dir / "run_manifest.yml", manifest)
    write_model(
        run_dir / "result_manifest.yml",
        ResultManifest(run_id="run-002", status="completed"),
    )
    bad_dir = tmp_path / "runs" / "bad"
    bad_dir.mkdir()
    (bad_dir / "run_manifest.yml").write_text("run_id: broken\n", encoding="utf-8")

    inventory = scan_workspace(tmp_path, max_depth=3)

    assert [run.run_id for run in inventory.runs] == ["run-002"]
    assert inventory.result_manifests == (run_dir / "result_manifest.yml",)
    assert len(inventory.issues) == 1
    assert inventory.issues[0].path == bad_dir / "run_manifest.yml"
