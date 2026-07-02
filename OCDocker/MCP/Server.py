#!/usr/bin/env python3

# Description
###############################################################################
"""
MCP server exposing the OCDocker Workbench API as LLM-callable tools.

Read/plan/preview tools are always available. Job-execute tools
(``run_job``, ``cancel_job``) require ``confirm=True`` and the Workbench job
bearer token; see :mod:`OCDocker.Workbench.Auth`.
"""

# Imports
###############################################################################
from __future__ import annotations

from typing import Any
from typing import Literal

import httpx

from mcp.server.fastmcp import FastMCP

from OCDocker.Workbench.Auth import resolve_workbench_job_token
from OCDocker.Workbench.Models import WorkbenchJobKind
from OCDocker.Workbench.Server import DEFAULT_WORKBENCH_API_HOST
from OCDocker.Workbench.Server import DEFAULT_WORKBENCH_API_PORT

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

DEFAULT_WORKBENCH_API_URL = f"http://{DEFAULT_WORKBENCH_API_HOST}:{DEFAULT_WORKBENCH_API_PORT}"
MCP_SERVER_NAME = "ocdocker-workbench"
MCP_SERVER_INSTRUCTIONS = (
    "Tools for inspecting an OCDocker Workbench workspace (OCScore studies, ablations, jobs) and for "
    "designing, launching, and monitoring vs/pipeline/ocscore_train/ocscore_reduce/vs_campaign jobs. "
    "For a single-target docking run, use get_vs_design_context to discover receptor/ligand/box "
    "candidates, preview_vs_design to validate a draft, and plan_vs_design to get the exact command "
    "- then pass its kind/args/cwd to run_job. For a multi-sample batch, use get_vs_campaign_context "
    "to discover an input/{sample}/... layout (or hand-author a manifest), preview_vs_campaign to "
    "validate it, and plan_vs_campaign to get the vs_campaign job payload - then pass its "
    "kind/manifest/args/cwd to run_job to launch every row as one tracked job. Pass engine=\"snakemake\" "
    "to plan_vs_campaign/run_job for real DAG orchestration (parallel via cores, resumable) instead of "
    "the default sequential shell loop - both continue past a failing row. Use get_campaign_progress(job_id) "
    "to check structured per-sample status while a campaign runs; engine=\"snakemake\" gives reliable "
    "per-sample pending/running/done/failed state, engine=\"shell\" only reliable aggregate counts. "
    "Read/plan/preview tools are always safe to call. run_job and cancel_job execute real, "
    "possibly long-running work: call plan_job first, show the plan to the user, and only call "
    "run_job with confirm=True after they agree. Never set confirm=True without an explicit "
    "go-ahead from the user for that specific job."
)

# Classes
###############################################################################


class OCDockerMCPError(Exception):
    """Raised when a Workbench API call made from an MCP tool fails."""


# Functions
###############################################################################
## Private ##


async def _request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    authorized: bool = False,
) -> dict[str, Any]:
    '''Issue one Workbench API request and raise on failure.

    Parameters
    ----------
    client : httpx.AsyncClient
        Client bound to the Workbench API base URL.
    method : str
        HTTP method.
    path : str
        Request path.
    json : dict[str, Any] or None
        JSON request body.
    params : dict[str, Any] or None
        Query parameters.
    authorized : bool
        Attach the Workbench job bearer token when True.

    Returns
    -------
    dict[str, Any]
        Parsed JSON response body.

    Raises
    ------
    OCDockerMCPError
        If the request fails or the Workbench API returns an error status.
    '''

    headers = {"Authorization": f"Bearer {resolve_workbench_job_token()}"} if authorized else {}
    try:
        response = await client.request(method, path, json=json, params=params, headers=headers)
    except httpx.HTTPError as exc:
        raise OCDockerMCPError(f"Could not reach Workbench API at {client.base_url}{path}: {exc}") from exc
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": response.text}
    if response.status_code >= 400:
        detail = payload.get("error", payload) if isinstance(payload, dict) else payload
        raise OCDockerMCPError(f"{method} {path} failed ({response.status_code}): {detail}")
    return payload


## Public ##


def build_ocdocker_mcp_server(*, base_url: str = DEFAULT_WORKBENCH_API_URL) -> FastMCP:
    '''Build the OCDocker Workbench MCP server.

    Parameters
    ----------
    base_url : str
        Base URL of a running ``ocdocker workbench serve`` API.

    Returns
    -------
    mcp.server.fastmcp.FastMCP
        Configured MCP server, not yet running.
    '''

    server = FastMCP(MCP_SERVER_NAME, instructions=MCP_SERVER_INSTRUCTIONS)
    client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    @server.tool()
    async def get_health() -> dict[str, Any]:
        '''Check whether the Workbench API is reachable and read-only status.'''

        return await _request(client, "GET", "/health")

    @server.tool()
    async def get_workspace() -> dict[str, Any]:
        '''Return the served OCScore workspace summary: studies, replicas, metrics, ablations.'''

        return await _request(client, "GET", "/api/ocscore-workspace")

    @server.tool()
    async def get_ablation_design_context() -> dict[str, Any]:
        '''Return bundled/workspace ablation policies and defaults for designing a new ablation.'''

        return await _request(client, "GET", "/api/ablation-design")

    @server.tool()
    async def preview_ablation_design(policy: dict[str, Any]) -> dict[str, Any]:
        '''Preview which features a draft ablation feature-policy would keep or exclude.

        ``policy`` matches the Workbench Design tab body: an object with a ``policy``
        block (``include_features``, ``include_patterns``, ``exclude_features``,
        ``exclude_patterns``, ...) plus optional identity/source fields.
        '''

        return await _request(client, "POST", "/api/ablation-design/preview", json=policy)

    @server.tool()
    async def plan_ablation_design(policy: dict[str, Any]) -> dict[str, Any]:
        '''Generate the policy YAML and ``ocdocker ocscore train`` command for a draft ablation.

        Same ``policy`` body shape as ``preview_ablation_design``. This only plans —
        it does not write files or launch anything.
        '''

        return await _request(client, "POST", "/api/ablation-design/plan", json=policy)

    @server.tool()
    async def get_vs_design_context(input_dir: str | None = None) -> dict[str, Any]:
        '''Discover receptor/ligand/box candidates for designing a `vs`/`pipeline` run.

        Best-effort scan by filename/extension heuristics, not a fixed layout —
        there is no single canonical input location in OCDocker. Results are
        candidates to choose from, not a requirement; an empty scan is not an
        error. ``input_dir`` optionally narrows the scan to one subdirectory of
        the served root instead of scanning the whole workspace.
        '''

        params: dict[str, Any] = {}
        if input_dir is not None:
            params["input_dir"] = input_dir
        return await _request(client, "GET", "/api/vs-design", params=params)

    @server.tool()
    async def preview_vs_design(draft: dict[str, Any]) -> dict[str, Any]:
        '''Validate one draft single-target VS design without running anything.

        ``draft`` is an object with ``kind`` ("vs" or "pipeline"), ``receptor``,
        ``ligand``, ``box`` (paths — absolute, or relative to the served root),
        plus kind-specific fields: ``engine`` for "vs", or ``engines`` /
        ``rescoring_engines`` (lists) for "pipeline". Only single-target design
        is supported — one receptor, one ligand, one box per draft; there is no
        multi-compound library/batch screening in OCDocker today.
        '''

        return await _request(client, "POST", "/api/vs-design/preview", json=draft)

    @server.tool()
    async def plan_vs_design(draft: dict[str, Any]) -> dict[str, Any]:
        '''Build the exact `ocdocker vs`/`pipeline` command for a valid draft design.

        Same ``draft`` body shape as ``preview_vs_design`` — call that first and
        show the user any errors/warnings. On success, returns
        ``{"kind", "args", "cwd", "shell_command"}`` ready to hand directly to
        ``run_job`` (as ``kind``/``args``/``cwd``) to actually launch it, subject
        to the same ``confirm=True`` gate as every other execute tool.
        '''

        return await _request(client, "POST", "/api/vs-design/plan", json=draft)

    @server.tool()
    async def get_vs_campaign_context(input_dir: str | None = None) -> dict[str, Any]:
        '''Discover a draft multi-sample manifest from an `input/{sample}/...` layout.

        Matches the convention used by the bundled example Snakefiles: one
        subdirectory per sample directly under the scan root, each containing a
        receptor/ligand/box file. Best-effort — a workspace not organized this
        way returns an empty manifest with an explanatory issue, not an error;
        a manifest can also be hand-authored directly for ``preview_vs_campaign``.
        '''

        params: dict[str, Any] = {}
        if input_dir is not None:
            params["input_dir"] = input_dir
        return await _request(client, "GET", "/api/vs-campaign", params=params)

    @server.tool()
    async def preview_vs_campaign(manifest: list[dict[str, Any]]) -> dict[str, Any]:
        '''Validate a draft multi-sample VS campaign manifest without running anything.

        ``manifest`` is a non-empty list of rows, each an object with ``sample``,
        ``row_kind`` ("vs" or "pipeline"), ``receptor``, ``ligand``, ``box``
        (paths), ``engines`` (list), and optional ``rescoring_engines`` (list,
        "pipeline" rows only). Rows may mix "vs" and "pipeline". There is no
        limit on manifest size beyond what the served workspace can validate.
        '''

        return await _request(client, "POST", "/api/vs-campaign/preview", json={"manifest": manifest})

    @server.tool()
    async def plan_vs_campaign(manifest: list[dict[str, Any]], outdir: str | None = None) -> dict[str, Any]:
        '''Build the `vs_campaign` job payload for a valid draft manifest.

        Same ``manifest`` shape as ``preview_vs_campaign`` — call that first and
        show the user any errors/warnings, especially for large manifests. On
        success, returns ``{"kind": "vs_campaign", "manifest", "args", "cwd",
        "shell_command"}`` ready to hand directly to ``run_job`` (as
        ``kind``/``manifest``/``args``/``cwd``) to launch the whole batch as one
        tracked job that runs every row and continues past individual failures.
        '''

        body: dict[str, Any] = {"manifest": manifest}
        if outdir is not None:
            body["outdir"] = outdir
        return await _request(client, "POST", "/api/vs-campaign/plan", json=body)

    @server.tool()
    async def get_protocol_similarity(metric: str | None = None, reference: str | None = None) -> dict[str, Any]:
        '''Return pairwise feature-similarity and outcome comparison across ablation protocols.

        Parameters
        ----------
        metric : str or None
            Metric key to overlay on similarity clusters.
        reference : str or None
            Reference protocol name for the added/removed feature diff.
        '''

        params: dict[str, Any] = {}
        if metric is not None:
            params["metric"] = metric
        if reference is not None:
            params["reference"] = reference
        return await _request(client, "GET", "/api/ablation-protocol-similarity", params=params)

    @server.tool()
    async def list_jobs() -> dict[str, Any]:
        '''List every tracked Workbench job (launched vs/pipeline/ocscore_train/ocscore_reduce runs).'''

        return await _request(client, "GET", "/api/jobs")

    @server.tool()
    async def get_job(job_id: str) -> dict[str, Any]:
        '''Return one tracked Workbench job's status, command, and exit code.'''

        return await _request(client, "GET", f"/api/jobs/{job_id}")

    @server.tool()
    async def get_job_logs(job_id: str, lines: int = 80) -> dict[str, Any]:
        '''Return a bounded stdout/stderr tail for one tracked Workbench job.'''

        return await _request(client, "GET", f"/api/jobs/{job_id}/logs", params={"lines": lines})

    @server.tool()
    async def plan_job(
        kind: WorkbenchJobKind,
        args: list[str] | None = None,
        cwd: str | None = None,
        manifest: list[dict[str, Any]] | None = None,
        engine: str = "shell",
        cores: int = 4,
        results_dir: str | None = None,
    ) -> dict[str, Any]:
        '''Preview the exact command a job would run, without launching it.

        Always call this before ``run_job`` and show the resulting command to the
        user. ``kind`` selects the ``ocdocker`` subcommand: "vs", "pipeline",
        "ocscore_train", "ocscore_reduce", or "vs_campaign". ``args`` are extra
        CLI flags appended after the subcommand, e.g. ["--protocol", "production.yml",
        "--output-dir", "runs/run-001"] — for "vs_campaign" they are common flags
        applied to every row instead. ``manifest`` is required for "vs_campaign"
        (build one with ``plan_vs_campaign`` first) and ignored otherwise.
        ``engine`` ("shell" or "snakemake"), ``cores`` (Snakemake ``--cores``),
        and ``results_dir`` (shared base output directory, per-sample-nested)
        are only meaningful for "vs_campaign".
        '''

        return await _request(
            client,
            "POST",
            "/api/jobs/plan",
            json={
                "kind": kind, "args": args or [], "cwd": cwd, "manifest": manifest,
                "engine": engine, "cores": cores, "results_dir": results_dir,
            },
        )

    @server.tool()
    async def run_job(
        kind: WorkbenchJobKind,
        args: list[str] | None = None,
        cwd: str | None = None,
        manifest: list[dict[str, Any]] | None = None,
        engine: str = "shell",
        cores: int = 4,
        results_dir: str | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        '''Launch a tracked job as a background subprocess. Requires explicit confirmation.

        Call ``plan_job`` first and show the command to the user. Only pass
        ``confirm=True`` after the user has explicitly agreed to run that specific
        job — without it, this returns the plan instead of launching anything.
        ``manifest`` is required for ``kind="vs_campaign"`` (a multi-sample batch
        run as one tracked job) and ignored for every other kind. ``engine``,
        ``cores``, and ``results_dir`` are only meaningful for "vs_campaign" —
        see ``plan_job``.
        '''

        body = {
            "kind": kind, "args": args or [], "cwd": cwd, "manifest": manifest,
            "engine": engine, "cores": cores, "results_dir": results_dir,
        }
        if not confirm:
            plan = await _request(client, "POST", "/api/jobs/plan", json=body)
            return {
                "launched": False,
                "message": "Not launched: call run_job again with confirm=True after the user agrees to this command.",
                "plan": plan,
            }
        record = await _request(client, "POST", "/api/jobs", json=body, authorized=True)
        return {"launched": True, "job": record}

    @server.tool()
    async def get_campaign_progress(job_id: str) -> dict[str, Any]:
        '''Return structured per-sample progress for one tracked vs_campaign job.

        Works for any job kind, degrading to ``{"engine": "unknown", ...}`` for
        non-campaign jobs or unrecognized log formats — never an error. For
        ``engine="snakemake"`` campaigns this reports real per-sample
        pending/running/done/failed status; for ``engine="shell"`` campaigns
        only aggregate success/failure counts are reliable (see
        :func:`OCDocker.Workbench.CampaignProgress.parse_campaign_progress`).
        '''

        return await _request(client, "GET", f"/api/jobs/{job_id}/campaign-progress")

    @server.tool()
    async def cancel_job(job_id: str, confirm: bool = False) -> dict[str, Any]:
        '''Cancel a running tracked job. Requires explicit confirmation.

        Only pass ``confirm=True`` after the user has explicitly agreed to cancel
        this specific job.
        '''

        if not confirm:
            current = await _request(client, "GET", f"/api/jobs/{job_id}")
            return {
                "cancelled": False,
                "message": "Not cancelled: call cancel_job again with confirm=True after the user agrees.",
                "job": current,
            }
        record = await _request(client, "POST", f"/api/jobs/{job_id}/cancel", authorized=True)
        return {"cancelled": True, "job": record}

    return server


def serve_ocdocker_mcp(*, base_url: str = DEFAULT_WORKBENCH_API_URL) -> None:
    '''Serve the OCDocker Workbench MCP server over stdio until interrupted.

    Parameters
    ----------
    base_url : str
        Base URL of a running ``ocdocker workbench serve`` API.
    '''

    server = build_ocdocker_mcp_server(base_url=base_url)
    server.run(transport="stdio")


__all__ = [
    "DEFAULT_WORKBENCH_API_URL",
    "MCP_SERVER_NAME",
    "OCDockerMCPError",
    "build_ocdocker_mcp_server",
    "serve_ocdocker_mcp",
]
