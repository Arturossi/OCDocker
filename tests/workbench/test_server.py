#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for the strict OCScore Workbench HTTP API payload layer.
"""

# Imports
###############################################################################
from __future__ import annotations

import time

from pathlib import Path

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


def _write_vs_design_workspace(root) -> dict:
    '''Write a synthetic receptor/ligand/box workspace for VS design API tests.

    Parameters
    ----------
    root : pathlib.Path
        Temporary root.

    Returns
    -------
    dict
        Absolute paths for the written receptor, ligand, and box files.
    '''

    receptor = root / "receptor.pdb"
    receptor.write_text("ATOM", encoding="utf-8")
    ligand_dir = root / "compounds" / "ligands" / "ligand"
    ligand_dir.mkdir(parents=True)
    ligand = ligand_dir / "ligand.smi"
    ligand.write_text("CCO", encoding="utf-8")
    box = root / "box.pdb"
    box.write_text("REMARK", encoding="utf-8")
    return {"receptor": str(receptor), "ligand": str(ligand), "box": str(box)}


def test_vs_design_endpoints_are_listed_in_index(tmp_path) -> None:
    '''The endpoint index advertises the VS design routes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)

    payload = client.get("/api/").json()
    assert "/api/vs-design" in payload["endpoints"]
    assert "/api/vs-design/preview" in payload["endpoints"]
    assert "/api/vs-design/plan" in payload["endpoints"]


def test_get_vs_design_discovers_candidates(tmp_path) -> None:
    '''GET /api/vs-design discovers receptor/ligand/box candidates.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_vs_design_workspace(tmp_path)
    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)

    response = client.get("/api/vs-design")

    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert len(candidates["receptors"]) == 1
    assert len(candidates["ligands"]) == 1
    assert len(candidates["boxes"]) == 1


def test_vs_design_preview_and_plan_round_trip(tmp_path) -> None:
    '''POST preview then plan for a valid draft returns a launchable command.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    paths = _write_vs_design_workspace(tmp_path)
    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)
    draft = {"kind": "vs", "engine": "smina", **paths}

    preview = client.post("/api/vs-design/preview", json=draft)
    assert preview.status_code == 200
    assert preview.json()["valid"] is True

    plan = client.post("/api/vs-design/plan", json=draft)
    assert plan.status_code == 200
    assert plan.json()["kind"] == "vs"
    assert "--engine" in plan.json()["args"]


def test_vs_design_plan_on_invalid_draft_returns_400(tmp_path) -> None:
    '''POST plan on an invalid draft returns a structured 400, not a 500.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)

    response = client.post("/api/vs-design/plan", json={"kind": "vs", "engine": "bogus"})

    assert response.status_code == 400
    assert response.json()["ok"] is False


def _write_vs_campaign_samples(root) -> Path:
    '''Write an ``input/{sample}/...`` layout for VS campaign API tests.

    Parameters
    ----------
    root : pathlib.Path
        Temporary root.

    Returns
    -------
    pathlib.Path
        The created ``input/`` directory.
    '''

    input_dir = root / "input"
    for sample in ("sample_001", "sample_002"):
        sample_dir = input_dir / sample
        sample_dir.mkdir(parents=True)
        (sample_dir / "receptor.pdbqt").write_text("ATOM", encoding="utf-8")
        (sample_dir / "ligand.pdbqt").write_text("MOL", encoding="utf-8")
        (sample_dir / "box.txt").write_text("REMARK", encoding="utf-8")
    return input_dir


def test_vs_campaign_endpoints_are_listed_in_index(tmp_path) -> None:
    '''The endpoint index advertises the VS campaign routes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)

    payload = client.get("/api/").json()
    assert "/api/vs-campaign" in payload["endpoints"]
    assert "/api/vs-campaign/preview" in payload["endpoints"]
    assert "/api/vs-campaign/plan" in payload["endpoints"]


def test_vs_campaign_discover_preview_plan_round_trip(tmp_path) -> None:
    '''GET discover, then POST preview and plan, for a real multi-sample layout.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    input_dir = _write_vs_campaign_samples(tmp_path)
    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)

    discover = client.get("/api/vs-campaign", params={"input_dir": str(input_dir)})
    assert discover.status_code == 200
    manifest = discover.json()["manifest"]
    assert len(manifest) == 2

    preview = client.post("/api/vs-campaign/preview", json={"manifest": manifest})
    assert preview.status_code == 200
    assert preview.json()["valid"] is True

    plan = client.post("/api/vs-campaign/plan", json={"manifest": manifest})
    assert plan.status_code == 200
    assert plan.json()["kind"] == "vs_campaign"
    assert len(plan.json()["manifest"]) == 2


def test_vs_campaign_plan_on_empty_manifest_returns_400(tmp_path) -> None:
    '''POST plan with an empty manifest returns a structured 400, not a 500.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)

    response = client.post("/api/vs-campaign/plan", json={"manifest": []})

    assert response.status_code == 400
    assert response.json()["ok"] is False


def test_post_jobs_launches_vs_campaign_with_manifest(tmp_path, monkeypatch) -> None:
    '''POST /api/jobs launches a vs_campaign job when a manifest is supplied.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    monkeypatch : pytest.MonkeyPatch
        Used to pin the job bearer token for this test.
    '''

    monkeypatch.setenv("OCDOCKER_WORKBENCH_TOKEN", "test-token")
    input_dir = _write_vs_campaign_samples(tmp_path)
    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)

    manifest = client.get("/api/vs-campaign", params={"input_dir": str(input_dir)}).json()["manifest"]

    response = client.post(
        "/api/jobs",
        json={"kind": "vs_campaign", "args": [], "manifest": manifest},
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 201
    assert response.json()["kind"] == "vs_campaign"
    assert response.json()["command"][:2] == ["/bin/sh", "-c"]


def test_vs_campaign_plan_snakemake_engine_returns_real_argv(tmp_path) -> None:
    '''POST /api/vs-campaign/plan with engine="snakemake" returns a real snakemake command.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    pytest.importorskip("snakemake")

    input_dir = _write_vs_campaign_samples(tmp_path)
    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)
    manifest = client.get("/api/vs-campaign", params={"input_dir": str(input_dir)}).json()["manifest"]

    response = client.post(
        "/api/vs-campaign/plan",
        json={"manifest": manifest, "engine": "snakemake", "cores": 5, "outdir": "runs"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "snakemake"
    assert payload["cores"] == 5
    assert payload["results_dir"] == "runs"
    assert "snakemake" in payload["shell_command"]


def test_post_jobs_launches_vs_campaign_with_snakemake_engine(tmp_path, monkeypatch) -> None:
    '''POST /api/jobs threads engine/cores/results_dir through to the launched job.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    monkeypatch : pytest.MonkeyPatch
        Used to pin the job bearer token for this test.
    '''

    pytest.importorskip("snakemake")

    monkeypatch.setenv("OCDOCKER_WORKBENCH_TOKEN", "test-token")
    input_dir = _write_vs_campaign_samples(tmp_path)
    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)
    manifest = client.get("/api/vs-campaign", params={"input_dir": str(input_dir)}).json()["manifest"]

    response = client.post(
        "/api/jobs",
        json={"kind": "vs_campaign", "args": [], "manifest": manifest, "engine": "snakemake", "cores": 2, "results_dir": "runs"},
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 201
    command = response.json()["command"]
    assert "snakemake" in command
    assert "results_dir=runs" in command


def test_campaign_progress_endpoint_reports_structured_status(tmp_path, monkeypatch) -> None:
    '''GET /api/jobs/{job_id}/campaign-progress returns structured per-sample status.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    monkeypatch : pytest.MonkeyPatch
        Used to pin the job bearer token for this test.
    '''

    pytest.importorskip("snakemake")

    monkeypatch.setenv("OCDOCKER_WORKBENCH_TOKEN", "test-token")
    input_dir = _write_vs_campaign_samples(tmp_path)
    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)
    manifest = client.get("/api/vs-campaign", params={"input_dir": str(input_dir)}).json()["manifest"]

    launch = client.post(
        "/api/jobs",
        json={"kind": "vs_campaign", "args": [], "manifest": manifest, "engine": "snakemake", "cores": 2},
        headers={"Authorization": "Bearer test-token"},
    )
    job_id = launch.json()["job_id"]

    for _ in range(100):
        record = client.get(f"/api/jobs/{job_id}").json()
        if record["status"] != "running":
            break
        time.sleep(0.1)

    response = client.get(f"/api/jobs/{job_id}/campaign-progress")

    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "snakemake"
    assert set(payload["samples"]) == {"sample_001", "sample_002"}


def test_campaign_progress_endpoint_degrades_gracefully_for_non_campaign_jobs(tmp_path, monkeypatch) -> None:
    '''The endpoint returns engine "unknown" for a non-vs_campaign job, not an error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    monkeypatch : pytest.MonkeyPatch
        Used to pin the job bearer token for this test.
    '''

    monkeypatch.setenv("OCDOCKER_WORKBENCH_TOKEN", "test-token")
    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)

    launch = client.post("/api/jobs", json={"kind": "vs", "args": ["--help"]}, headers={"Authorization": "Bearer test-token"})
    job_id = launch.json()["job_id"]
    for _ in range(50):
        record = client.get(f"/api/jobs/{job_id}").json()
        if record["status"] != "running":
            break
        time.sleep(0.05)

    response = client.get(f"/api/jobs/{job_id}/campaign-progress")

    assert response.status_code == 200
    assert response.json()["engine"] == "unknown"


def test_campaign_progress_endpoint_unknown_job_returns_404(tmp_path) -> None:
    '''The endpoint returns 404 for an unknown job id, matching /api/jobs/{job_id}.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    app = build_workbench_api_app(tmp_path, max_depth=6)
    client = TestClient(app)

    response = client.get("/api/jobs/does-not-exist/campaign-progress")

    assert response.status_code == 404
