#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for the OCDocker Workbench MCP server tools.
"""

# Imports
###############################################################################
from __future__ import annotations

import asyncio
import threading

import pytest
import uvicorn

from mcp.server.fastmcp.exceptions import ToolError

from OCDocker.MCP import build_ocdocker_mcp_server
from OCDocker.Workbench import build_workbench_api_app

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Functions
###############################################################################
## Private ##


@pytest.fixture
def live_workbench_api(tmp_path):
    '''Serve a real Workbench API on a background thread for the test's duration.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary served root.

    Yields
    ------
    str
        Base URL of the running Workbench API.
    '''

    replica = tmp_path / "replica_1"
    replica.mkdir()
    (replica / "metrics.csv").write_text("metric,value\nBEDROC,0.77\n", encoding="utf-8")

    app = build_workbench_api_app(tmp_path, max_depth=2)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="critical")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        pass
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


## Public ##


def test_build_ocdocker_mcp_server_registers_curated_tools() -> None:
    '''The MCP server exposes exactly the curated read/plan/execute tool set.'''

    server = build_ocdocker_mcp_server(base_url="http://127.0.0.1:0")
    tools = {tool.name for tool in asyncio.run(server.list_tools())}

    assert tools == {
        "get_health",
        "get_workspace",
        "get_ablation_design_context",
        "preview_ablation_design",
        "plan_ablation_design",
        "get_protocol_similarity",
        "list_jobs",
        "get_job",
        "get_job_logs",
        "plan_job",
        "run_job",
        "cancel_job",
    }


def test_get_health_returns_workbench_payload(live_workbench_api) -> None:
    '''get_health proxies the Workbench API health payload.

    Parameters
    ----------
    live_workbench_api : str
        Base URL of a running Workbench API (fixture).
    '''

    server = build_ocdocker_mcp_server(base_url=live_workbench_api)
    _content, payload = asyncio.run(server.call_tool("get_health", {}))
    assert payload["ok"] is True
    assert payload["service"] == "ocdocker-workbench"


def test_get_job_raises_for_unknown_job(live_workbench_api) -> None:
    '''get_job raises a structured error for an unknown job id.

    Parameters
    ----------
    live_workbench_api : str
        Base URL of a running Workbench API (fixture).
    '''

    server = build_ocdocker_mcp_server(base_url=live_workbench_api)
    with pytest.raises(ToolError, match="Unknown job id"):
        asyncio.run(server.call_tool("get_job", {"job_id": "does-not-exist"}))


def test_plan_job_previews_command_without_launching(live_workbench_api) -> None:
    '''plan_job returns the computed command without creating a job.

    Parameters
    ----------
    live_workbench_api : str
        Base URL of a running Workbench API (fixture).
    '''

    async def _run():
        server = build_ocdocker_mcp_server(base_url=live_workbench_api)
        _content, plan = await server.call_tool("plan_job", {"kind": "vs", "args": ["--help"]})
        _content, jobs = await server.call_tool("list_jobs", {})
        return plan, jobs

    plan, jobs = asyncio.run(_run())
    assert plan["kind"] == "vs"
    assert plan["command"][1:] == ["vs", "--help"]
    assert jobs["jobs"] == []


def test_run_job_without_confirm_does_not_launch(live_workbench_api) -> None:
    '''run_job with confirm=False returns a plan and launches nothing.

    Parameters
    ----------
    live_workbench_api : str
        Base URL of a running Workbench API (fixture).
    '''

    async def _run():
        server = build_ocdocker_mcp_server(base_url=live_workbench_api)
        _content, result = await server.call_tool("run_job", {"kind": "vs", "args": ["--help"], "confirm": False})
        _content, jobs = await server.call_tool("list_jobs", {})
        return result, jobs

    result, jobs = asyncio.run(_run())
    assert result["launched"] is False
    assert "plan" in result
    assert jobs["jobs"] == []


def test_run_job_without_confirm_does_not_require_a_token(live_workbench_api, monkeypatch) -> None:
    '''run_job with confirm=False never needs the job bearer token.

    Parameters
    ----------
    live_workbench_api : str
        Base URL of a running Workbench API (fixture).
    monkeypatch : pytest.MonkeyPatch
        Standard pytest fixture.
    '''

    def _fail(*_args, **_kwargs):
        raise AssertionError("resolve_workbench_job_token should not be called when confirm=False")

    monkeypatch.setattr("OCDocker.MCP.Server.resolve_workbench_job_token", _fail)

    server = build_ocdocker_mcp_server(base_url=live_workbench_api)
    _content, payload = asyncio.run(server.call_tool("run_job", {"kind": "vs", "args": ["--help"], "confirm": False}))
    assert payload["launched"] is False


def test_run_job_with_confirm_launches_and_cancel_without_confirm_previews(live_workbench_api) -> None:
    '''run_job(confirm=True) launches; cancel_job(confirm=False) only previews.

    Parameters
    ----------
    live_workbench_api : str
        Base URL of a running Workbench API (fixture).
    '''

    async def _run():
        server = build_ocdocker_mcp_server(base_url=live_workbench_api)
        _content, launched = await server.call_tool("run_job", {"kind": "vs", "args": ["--help"], "confirm": True})
        job_id = launched["job"]["job_id"]
        _content, preview = await server.call_tool("cancel_job", {"job_id": job_id, "confirm": False})
        return launched, job_id, preview

    launched, job_id, preview = asyncio.run(_run())
    assert launched["launched"] is True
    assert preview["cancelled"] is False
    assert preview["job"]["job_id"] == job_id
