#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for the strict OCScore Workbench HTTP API payload layer.
"""

# Imports
###############################################################################
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from OCDocker.Workbench import WorkbenchAPIError
from OCDocker.Workbench import build_workbench_api_app
from OCDocker.Workbench import build_workbench_api_payload

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Functions
###############################################################################
## Private ##


def _mark_replica_stages_complete(replica_dir) -> None:
    pdbbind = replica_dir / "pdbbind"
    pdbbind.mkdir(exist_ok=True)
    (pdbbind / "pdbbind_best.pt").write_bytes(b"pt")
    dudez = replica_dir / "dudez"
    dudez.mkdir(exist_ok=True)
    (dudez / "dudez_best.pt").write_bytes(b"pt")


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
    _mark_replica_stages_complete(replica)
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
    assert "/api/ablation-design" in payload["endpoints"]
    assert "/api/ablation-design/preview" in payload["endpoints"]
    assert "/api/ablation-design/features" in payload["endpoints"]
    assert "/api/ablation-design/write" in payload["endpoints"]
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


def test_build_workbench_api_app_serves_figure_assets(tmp_path) -> None:
    '''Workbench API app serves allowed OCScore figure assets.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    image = tmp_path / "export" / "dudez" / "shap" / "shap_beeswarm_plot.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")

    app = build_workbench_api_app(tmp_path, max_depth=2)
    client = TestClient(app)
    response = client.get("/api/figure-asset", params={"path": str(image)})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"png"


def test_build_workbench_api_app_serves_packaged_browser_assets(tmp_path) -> None:
    '''Workbench API app serves packaged browser assets.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    app = build_workbench_api_app(tmp_path, max_depth=2)
    client = TestClient(app)
    response = client.get("/app")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert b"OCScore Control Dashboard" in response.content


def test_build_workbench_api_app_health_and_unknown_endpoint(tmp_path) -> None:
    '''Workbench API app answers health checks and rejects unknown routes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    app = build_workbench_api_app(tmp_path, max_depth=2)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    missing = client.get("/api/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["ok"] is False
