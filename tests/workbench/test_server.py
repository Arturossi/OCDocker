#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for the read-only Workbench HTTP API payload layer.
'''

# Imports
###############################################################################
from __future__ import annotations

import http.client
import threading

import pytest

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import WorkbenchAPIError
from OCDocker.Workbench import build_workbench_api_handler
from OCDocker.Workbench import build_workbench_api_payload
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


def _write_api_workspace(tmp_path) -> None:
    '''Write result manifests for API tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    for run_id, metrics in {
        "baseline": {"auc": 0.85, "loss": 0.20},
        "candidate": {"auc": 0.90, "loss": 0.18},
    }.items():
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(run_id=run_id, status="completed", metrics=metrics),
        )


def _write_api_run_bundle(tmp_path):
    '''Write a synthetic run bundle for API tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    pathlib.Path
        Run directory path.
    '''

    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("queued\ncompleted\n", encoding="utf-8")
    (run_dir / "metrics.csv").write_text("metric,value\nauc,0.93\n", encoding="utf-8")
    write_model(
        run_dir / "run_manifest.yml",
        RunManifest(
            run_id="run-001",
            spec_type="ocscore_study",
            name="api-detail",
            status="completed",
            workspace=".",
            log_files=("run.log",),
            artifacts=(ResultArtifact(name="metrics", path="metrics.csv", kind="csv"),),
        ),
    )
    write_model(
        run_dir / "result_manifest.yml",
        ResultManifest(
            run_id="run-001",
            status="completed",
            metrics={"auc": 0.93},
            artifacts=(ResultArtifact(name="metrics", path="metrics.csv", kind="csv"),),
        ),
    )
    return run_dir


## Public ##


def test_build_workbench_api_payload_serves_health_and_index(tmp_path) -> None:
    '''Workbench API payloads expose service metadata.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    index = build_workbench_api_payload(tmp_path, "/", default_max_depth=2)
    health = build_workbench_api_payload(tmp_path, "/health", default_max_depth=2)

    assert index["service"] == "ocdocker-workbench"
    assert index["read_only"] is True
    assert "/api/overview" in index["endpoints"]
    assert "/api/run-detail" in index["endpoints"]
    assert health["ok"] is True
    assert health["root"] == str(tmp_path)


def test_build_workbench_api_payload_serves_decision_endpoints(tmp_path) -> None:
    '''Workbench API payloads wrap existing read-only decision helpers.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_api_workspace(tmp_path)

    overview = build_workbench_api_payload(
        tmp_path,
        "/api/overview",
        {"max_depth": ["2"]},
    )
    leaderboard = build_workbench_api_payload(
        tmp_path,
        "/api/leaderboard",
        {"metric": ["auc"], "max_depth": ["2"]},
    )
    comparison = build_workbench_api_payload(
        tmp_path,
        "/api/compare",
        {
            "baseline": ["baseline"],
            "metric": ["auc:max", "loss:min"],
            "max_depth": ["2"],
        },
    )

    assert overview["result_manifest_count"] == 2
    assert leaderboard["best_entry"]["run_id"] == "candidate"
    assert comparison["best_candidate"]["run_id"] == "candidate"
    assert comparison["best_candidate"]["net_score"] == 2


def test_build_workbench_api_payload_serves_plot_endpoint(tmp_path) -> None:
    '''Workbench API payloads can produce Plotly-compatible plot payloads.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_api_workspace(tmp_path)

    payload = build_workbench_api_payload(
        tmp_path,
        "/api/plot",
        {
            "kind": ["scatter"],
            "x_metric": ["auc"],
            "y_metric": ["loss"],
            "max_depth": ["2"],
        },
    )

    assert payload["plot_kind"] == "metric_scatter"
    assert payload["included_count"] == 2
    assert payload["data"][0]["type"] == "scatter"


def test_build_workbench_api_payload_serves_run_detail_endpoint(tmp_path) -> None:
    '''Workbench API payloads expose aggregate run details.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_api_run_bundle(tmp_path)

    payload = build_workbench_api_payload(
        tmp_path,
        "/api/run-detail",
        {"target": ["run-001"], "lines": ["1"]},
        default_max_depth=2,
    )

    assert payload["run_id"] == "run-001"
    assert payload["status_report"]["result_manifest_exists"] is True
    assert payload["log_preview"]["logs"][0]["text"] == "completed"
    assert payload["result_summary"]["metrics"] == {"auc": 0.93}
    assert payload["issue_count"] == 0


def test_build_workbench_api_payload_rejects_unknown_endpoint(tmp_path) -> None:
    '''Workbench API payloads reject unknown endpoints.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(WorkbenchAPIError, match="Unknown Workbench API endpoint"):
        build_workbench_api_payload(tmp_path, "/api/unknown")


def test_build_workbench_api_handler_binds_root_and_depth(tmp_path) -> None:
    '''Workbench API handlers preserve root and default depth.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    handler = build_workbench_api_handler(tmp_path, default_max_depth=3)

    assert handler.workbench_root == tmp_path
    assert handler.workbench_default_max_depth == 3


def test_workbench_api_handler_serves_embedded_browser_assets(tmp_path) -> None:
    '''Workbench API handlers serve embedded browser assets.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    from http.server import ThreadingHTTPServer

    handler = build_workbench_api_handler(tmp_path, default_max_depth=2)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/app")
        response = connection.getresponse()
        body = response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 200
    assert response.getheader("Content-Type") == "text/html; charset=utf-8"
    assert b"Decision Console" in body
