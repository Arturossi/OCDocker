#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for the strict OCScore Workbench HTTP API payload layer.
"""

# Imports
###############################################################################
from __future__ import annotations

import http.client
import threading

import pytest

from OCDocker.Workbench import WorkbenchAPIError
from OCDocker.Workbench import build_workbench_api_handler
from OCDocker.Workbench import build_workbench_api_payload

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


def _write_api_ocscore_root(root) -> None:
    '''Write a synthetic OCScore root for API tests.

    Parameters
    ----------
    root : pathlib.Path
        Temporary root.
    '''

    replica = root / "replica_1"
    replica.mkdir()
    (replica / "metrics.csv").write_text("metric,value\nBEDROC,0.77\n", encoding="utf-8")
    ablation = root / "ablation" / "no_ligand" / "replica_1"
    ablation.mkdir(parents=True)
    (ablation / "metrics.csv").write_text("metric,value\nBEDROC,0.52\n", encoding="utf-8")


## Public ##


def test_build_workbench_api_payload_indexes_strict_endpoints(tmp_path) -> None:
    '''Workbench API index exposes only the strict dashboard endpoint set.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    payload = build_workbench_api_payload(tmp_path, "/api")

    assert payload["dashboard_model"] == "strict_ocscore_layout"
    assert "/api/ocscore-workspace" in payload["endpoints"]
    assert "/api/figure-asset?path=..." in payload["endpoints"]
    assert "/api/optuna-dashboard" in payload["endpoints"]
    assert payload["optuna_dashboard"]["auto_ports"] is True
    assert payload["optuna_dashboard"]["slot_count"] == 1
    assert payload["optuna_dashboard"]["slot_count_source"] == "replica_count"
    assert payload["optuna_dashboard"]["min_slot_count"] == 1
    assert payload["optuna_dashboard"]["max_slot_count"] == 50
    assert "/api/evidence" not in payload["endpoints"]
    assert "/api/plot" not in payload["endpoints"]


def test_build_workbench_api_payload_serves_ocscore_workspace(tmp_path) -> None:
    '''Workbench API exposes the strict OCScore workspace payload.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_api_ocscore_root(tmp_path)

    payload = build_workbench_api_payload(
        tmp_path,
        "/api/ocscore-workspace",
    )

    assert payload["study_count"] == 2
    assert payload["baseline_study"]["completed_count"] == 1
    assert payload["baseline_study"]["expected_replica_count"] == 1
    assert payload["baseline_study"]["missing_count"] == 0
    assert payload["baseline_study"]["metric_summary"]["test_bedroc"]["mean"] == 0.77
    assert payload["ablation_studies"][0]["study_name"] == "no_ligand"


def test_build_workbench_api_payload_rejects_legacy_endpoint(tmp_path) -> None:
    '''Workbench API rejects legacy generic dashboard endpoints.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(WorkbenchAPIError, match="Unknown Workbench API endpoint"):
        build_workbench_api_payload(tmp_path, "/api/evidence")


def test_build_workbench_api_handler_serves_figure_assets(tmp_path) -> None:
    '''Workbench API handlers serve allowed OCScore figure assets.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    from http.server import ThreadingHTTPServer
    from urllib.parse import quote

    image = tmp_path / "export" / "dudez" / "shap" / "shap_beeswarm_plot.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")

    handler = build_workbench_api_handler(tmp_path, max_depth=2)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
        connection.request("GET", f"/api/figure-asset?path={quote(str(image))}")
        response = connection.getresponse()
        body = response.read()
        connection.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert response.status == 200
    assert response.getheader("Content-Type") == "image/png"
    assert body == b"png"


def test_build_workbench_api_handler_serves_packaged_browser_assets(tmp_path) -> None:
    '''Workbench API handlers serve packaged browser assets.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    from http.server import ThreadingHTTPServer

    handler = build_workbench_api_handler(tmp_path, max_depth=2)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
        connection.request("GET", "/app")
        response = connection.getresponse()
        body = response.read()
        connection.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert response.status == 200
    assert response.getheader("Content-Type") == "text/html; charset=utf-8"
    assert b"OCScore Control Dashboard" in body
