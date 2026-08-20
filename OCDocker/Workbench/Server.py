#!/usr/bin/env python3

# Description
###############################################################################
"""
FastAPI-backed local HTTP API for the OCDocker Workbench dashboard.
"""

# Imports
###############################################################################
from __future__ import annotations

import json
import re
import secrets

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from typing import AsyncIterator

from fastapi import Depends
from fastapi import FastAPI
from fastapi import Header
from fastapi import Query
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from OCDocker.Workbench.Auth import resolve_workbench_job_token
from OCDocker.Workbench.Auth import workbench_job_token_path
from OCDocker.Workbench.IO import model_to_data
from OCDocker.Workbench.CampaignProgress import CAMPAIGN_PROGRESS_LOG_BYTE_LIMIT
from OCDocker.Workbench.CampaignProgress import CAMPAIGN_PROGRESS_LOG_LINE_LIMIT
from OCDocker.Workbench.CampaignProgress import parse_campaign_progress
from OCDocker.Workbench.Jobs import DEFAULT_CAMPAIGN_CORES
from OCDocker.Workbench.Jobs import DEFAULT_CAMPAIGN_ENGINE
from OCDocker.Workbench.Jobs import JOB_KIND_COMMAND_PREFIX
from OCDocker.Workbench.Jobs import JobError
from OCDocker.Workbench.Jobs import JobManager
from OCDocker.Workbench.Logs import DEFAULT_LOG_BYTE_LIMIT
from OCDocker.Workbench.Logs import DEFAULT_LOG_LINE_LIMIT
from OCDocker.Workbench.Models import WorkbenchJobKind
from OCDocker.Workbench.OCScoreLayout import DEFAULT_OCSCORE_MAX_METRIC_FILE_BYTES
from OCDocker.Workbench.OCScoreLayout import DEFAULT_OCSCORE_SCAN_DEPTH
from OCDocker.Workbench.OCScoreLayout import build_ocscore_workspace
from OCDocker.Workbench.OCScoreLayout import MAX_OPTUNA_DASHBOARD_SLOT_COUNT
from OCDocker.Workbench.OCScoreLayout import MIN_OPTUNA_DASHBOARD_SLOT_COUNT
from OCDocker.Workbench.OCScoreLayout import resolve_optuna_dashboard_slot_count
from OCDocker.Workbench.OptunaDashboard import DEFAULT_OPTUNA_DASHBOARD_HOST
from OCDocker.Workbench.OptunaDashboard import OptunaDashboardError
from OCDocker.Workbench.OptunaDashboard import OptunaDashboardManager
from OCDocker.Workbench.Schema import build_schema_catalog
from OCDocker.Workbench.AblationDesign import build_ablation_design_context
from OCDocker.Workbench.AblationDesign import handle_ablation_design_post
from OCDocker.Workbench.AblationProtocolSimilarity import build_ablation_protocol_similarity_analysis
from OCDocker.Workbench.Templates import build_template_payload
from OCDocker.Workbench.VSDesign import discover_vs_campaign_candidates
from OCDocker.Workbench.VSDesign import discover_vs_design_candidates
from OCDocker.Workbench.VSDesign import plan_vs_campaign
from OCDocker.Workbench.VSDesign import plan_vs_design
from OCDocker.Workbench.VSDesign import preview_vs_campaign
from OCDocker.Workbench.VSDesign import preview_vs_design
from OCDocker.Workbench.Web import build_workbench_web_asset
from OCDocker.Workbench.Web import is_workbench_web_asset_path
from OCDocker.Workbench.Web import WORKBENCH_WEB_ROUTES

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Type aliases
###############################################################################

QueryMap = dict[str, list[str]]

# Constants
###############################################################################

DEFAULT_WORKBENCH_API_HOST = "127.0.0.1"
DEFAULT_WORKBENCH_API_PORT = 8765
WORKBENCH_API_VERSION = 1
FIGURE_ASSET_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml; charset=utf-8",
    ".pdf": "application/pdf",
}
# Classes
###############################################################################


class WorkbenchAPIError(Exception):
    """HTTP-aware error raised by Workbench API payload builders."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        '''Create a Workbench API error.

        Parameters
        ----------
        message : str
            Error message.
        status_code : int
            HTTP status code returned by the local API handler.
        '''

        super().__init__(message)
        self.status_code = status_code


class JobCreateRequest(BaseModel):
    """Request body for ``POST /api/jobs``."""

    kind: WorkbenchJobKind
    args: list[str] = []
    cwd: str | None = None
    manifest: list[dict[str, Any]] | None = None
    """Required for ``kind="vs_campaign"``; ignored for every other kind."""
    engine: str = DEFAULT_CAMPAIGN_ENGINE
    """``"shell"`` or ``"snakemake"``; only meaningful for ``kind="vs_campaign"``."""
    cores: int = DEFAULT_CAMPAIGN_CORES
    """``--cores`` passed to Snakemake; only meaningful for ``engine="snakemake"``."""
    results_dir: str | None = None
    """Shared base output directory (each row still writes to its own
    ``<results_dir>/<sample>``); only meaningful for ``kind="vs_campaign"``."""


# Functions
###############################################################################
## Private ##


def _endpoint_index(root: Path) -> dict[str, Any]:
    '''Build the endpoint index payload.

    Parameters
    ----------
    root : pathlib.Path
        Served OCScore root.

    Returns
    -------
    dict[str, Any]
        Endpoint index payload.
    '''

    return {
        "service": "ocdocker-workbench",
        "api_version": WORKBENCH_API_VERSION,
        "root": str(root),
        "read_only": False,
        "web_app": "/app",
        "dashboard_model": "strict_ocscore_layout",
        "endpoints": [
            "/health",
            "/api/ocscore-workspace",
            "/api/figure-asset?path=...",
            "/api/optuna-dashboard",
            "/api/ablation-design",
            "/api/ablation-design/features",
            "/api/ablation-design/preview",
            "/api/ablation-design/plan",
            "/api/ablation-design/write",
            "/api/ablation-protocol-similarity",
            "/api/vs-design",
            "/api/vs-design/preview",
            "/api/vs-design/plan",
            "/api/vs-campaign",
            "/api/vs-campaign/preview",
            "/api/vs-campaign/plan",
            "/api/schema",
            "/api/template",
            "/api/jobs",
            "/api/jobs/plan",
            "/api/jobs/{job_id}",
            "/api/jobs/{job_id}/logs",
            "/api/jobs/{job_id}/campaign-progress",
            "/api/jobs/{job_id}/cancel",
        ],
        "optuna_dashboard": {
            "available": OptunaDashboardManager.is_available(),
            "host": DEFAULT_OPTUNA_DASHBOARD_HOST,
            "auto_ports": True,
            "slot_count": resolve_optuna_dashboard_slot_count(root),
            "slot_count_source": "replica_count",
            "min_slot_count": MIN_OPTUNA_DASHBOARD_SLOT_COUNT,
            "max_slot_count": MAX_OPTUNA_DASHBOARD_SLOT_COUNT,
            "scan_start_offset": 1,
        },
        "job_execution": {
            "enabled": True,
            "kinds": sorted(JOB_KIND_COMMAND_PREFIX),
            "auth": "bearer_token",
        },
    }


def _values(query: QueryMap, name: str) -> tuple[str, ...]:
    '''Return repeated query values for one key.

    Parameters
    ----------
    query : QueryMap
        Parsed query string.
    name : str
        Query key.

    Returns
    -------
    tuple[str, ...]
        Query values.
    '''

    return tuple(value for value in query.get(name, ()) if str(value).strip())


def _first(query: QueryMap, name: str, default: str | None = None) -> str | None:
    '''Return the first query value for one key.

    Parameters
    ----------
    query : QueryMap
        Parsed query string.
    name : str
        Query key.
    default : str or None
        Default value when the key is absent.

    Returns
    -------
    str or None
        Query value or default.
    '''

    values = _values(query, name)
    return values[0] if values else default


def _required(query: QueryMap, name: str) -> str:
    '''Return a required query value.

    Parameters
    ----------
    query : QueryMap
        Parsed query string.
    name : str
        Query key.

    Returns
    -------
    str
        Required value.
    '''

    value = _first(query, name)
    if value is None:
        raise WorkbenchAPIError(f"Missing required query parameter: {name}")
    return value


def _optional_int_query(query: QueryMap, name: str) -> int | None:
    '''Parse an optional integer query value.

    Parameters
    ----------
    query : QueryMap
        Parsed query string.
    name : str
        Query key.

    Returns
    -------
    int or None
        Parsed value when present.
    '''

    raw = _first(query, name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise WorkbenchAPIError(f"Query parameter {name} must be an integer.") from exc


def _int_query(query: QueryMap, name: str, default: int) -> int:
    '''Parse an integer query value.

    Parameters
    ----------
    query : QueryMap
        Parsed query string.
    name : str
        Query key.
    default : int
        Default value.

    Returns
    -------
    int
        Parsed value.
    '''

    raw = _first(query, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise WorkbenchAPIError(f"Query parameter {name} must be an integer.") from exc


def _allowed_asset_roots(root: Path) -> tuple[Path, ...]:
    '''Return filesystem roots that may serve dashboard figure assets.

    Parameters
    ----------
    root : pathlib.Path
        Served OCScore root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Resolved allowed roots.
    '''

    roots = [root]
    if root.name == "train":
        roots.append(root.parent)
    resolved_roots: list[Path] = []
    for item in roots:
        try:
            resolved_roots.append(item.resolve())
        except OSError:
            continue
    return tuple(resolved_roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    '''Return whether a path is below a root path.

    Parameters
    ----------
    path : pathlib.Path
        Candidate path.
    root : pathlib.Path
        Root path.

    Returns
    -------
    bool
        True when the path is below the root.
    '''

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _figure_asset(root: Path, query: QueryMap) -> tuple[bytes, str]:
    '''Read one allowed dashboard figure asset.

    Parameters
    ----------
    root : pathlib.Path
        Served OCScore root.
    query : QueryMap
        Parsed request query.

    Returns
    -------
    tuple[bytes, str]
        Asset bytes and content type.
    '''

    raw_path = _required(query, "path")
    candidate = Path(raw_path)
    suffix = candidate.suffix.lower()
    content_type = FIGURE_ASSET_CONTENT_TYPES.get(suffix)
    if content_type is None:
        raise WorkbenchAPIError("Unsupported figure asset type.", status_code=415)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise WorkbenchAPIError("Figure asset does not exist.", status_code=404) from exc
    if not resolved.is_file():
        raise WorkbenchAPIError("Figure asset is not a file.", status_code=404)
    if not any(_is_relative_to(resolved, allowed_root) for allowed_root in _allowed_asset_roots(root)):
        raise WorkbenchAPIError("Figure asset is outside the served OCScore root.", status_code=403)
    return resolved.read_bytes(), content_type


def _campaign_expected_samples(record: Any) -> list[str] | None:
    '''Best-effort recovery of a vs_campaign job's sample names from its stored command.

    Parameters
    ----------
    record : Any
        Tracked ``WorkbenchJobRecord``.

    Returns
    -------
    list[str] or None
        Sample names, or None if they could not be recovered (never raises).
    '''

    if record.kind != "vs_campaign":
        return None
    command = record.command
    for part in command:
        if part.startswith("samples="):
            try:
                return list(json.loads(part[len("samples="):]))
            except (ValueError, TypeError):
                return None
    if command[:2] == ("/bin/sh", "-c") and len(command) >= 3:
        return re.findall(r"^echo '\[sample \d+/\d+\] (.+?)'$", command[2], re.MULTILINE)
    return None


def _model_payload(model: Any) -> dict[str, Any]:
    '''Convert a pydantic model to an API payload.

    Parameters
    ----------
    model : Any
        Model instance.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload.
    '''

    data = model_to_data(model)
    if not isinstance(data, dict):
        raise WorkbenchAPIError("Workbench API payload must be an object.", status_code=500)
    return data


## Public ##


def build_workbench_api_payload(
    root: str | Path,
    endpoint: str,
    query: QueryMap | None = None,
    *,
    max_depth: int = DEFAULT_OCSCORE_SCAN_DEPTH,
) -> dict[str, Any]:
    '''Build one strict Workbench API payload without starting a server.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore root.
    endpoint : str
        API endpoint path.
    query : QueryMap or None
        Parsed query parameters.
    max_depth : int
        Maximum recursive depth inside each replica.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload.
    '''

    root_path = Path(root)
    request_query = {} if query is None else query
    path = endpoint.rstrip("/") or "/"

    if path in {"/", "/api"}:
        return _endpoint_index(root_path)
    if path in {"/health", "/api/health"}:
        return {
            "ok": True,
            "service": "ocdocker-workbench",
            "api_version": WORKBENCH_API_VERSION,
            "root": str(root_path),
            "read_only": False,
            "dashboard_model": "strict_ocscore_layout",
        }
    if path == "/api/ocscore-workspace":
        return _model_payload(
            build_ocscore_workspace(
                root_path,
                expected_replica_count=_optional_int_query(request_query, "replicas"),
                max_depth=max_depth,
                max_metric_file_bytes=_int_query(
                    request_query,
                    "max_metric_file_bytes",
                    DEFAULT_OCSCORE_MAX_METRIC_FILE_BYTES,
                ),
                metric_names=_values(request_query, "metric"),
            )
        )
    if path == "/api/ablation-design":
        return build_ablation_design_context(root_path)
    if path == "/api/ablation-protocol-similarity":
        metric_values = _values(request_query, "metric")
        reference_values = _values(request_query, "reference")
        catalog_values = _values(request_query, "include_catalog_only")
        include_catalog_only = catalog_values[0].strip().lower() in {"1", "true", "yes"} if catalog_values else False
        return _model_payload(
            build_ablation_protocol_similarity_analysis(
                root_path,
                reference_policy=reference_values[0] if reference_values else None,
                metric=metric_values[0] if metric_values else None,
                include_catalog_only=include_catalog_only,
                max_depth=max_depth,
            )
        )
    if path == "/api/ablation-design/features":
        raise WorkbenchAPIError(
            "Use POST /api/ablation-design/features with input paths in the JSON body.",
            status_code=405,
        )
    if path == "/api/schema":
        names = _values(request_query, "name")
        return build_schema_catalog(names or None)
    if path == "/api/template":
        return build_template_payload(_required(request_query, "name"))
    raise WorkbenchAPIError(f"Unknown Workbench API endpoint: {endpoint}", status_code=404)


def _optuna_dashboard_payload(
    manager: OptunaDashboardManager,
    endpoint: str,
    query: QueryMap,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    '''Build one Optuna dashboard control payload.

    Parameters
    ----------
    manager : OptunaDashboardManager
        Dashboard manager bound to the served root.
    endpoint : str
        Request path.
    query : QueryMap
        Parsed query string.
    body : dict[str, Any] or None
        Parsed JSON body for POST requests.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload.
    '''

    replica_path = _first(query, "replica_path")
    if replica_path is None and body is not None:
        replica_path = body.get("replica_path")
    if endpoint.endswith("/status") or endpoint == "/api/optuna-dashboard/status":
        if replica_path is None:
            return manager.status()
        return manager.status(replica_path)
    if endpoint == "/api/optuna-dashboard" and body is not None:
        if replica_path is None:
            raise WorkbenchAPIError("Missing required field: replica_path")
        try:
            return manager.start(replica_path)
        except OptunaDashboardError as exc:
            raise WorkbenchAPIError(str(exc), status_code=exc.status_code) from exc
    if replica_path is None:
        raise WorkbenchAPIError("Missing required query parameter: replica_path")
    try:
        return manager.stop(replica_path)
    except OptunaDashboardError as exc:
        raise WorkbenchAPIError(str(exc), status_code=exc.status_code) from exc


def _require_job_token(authorization: str | None = Header(default=None)) -> None:
    '''Require a valid bearer token on execute-capable job endpoints.

    Parameters
    ----------
    authorization : str or None
        Raw ``Authorization`` request header.

    Raises
    ------
    WorkbenchAPIError
        If the header is missing or does not match the configured job token.
    '''

    provided = ""
    if authorization is not None and authorization.lower().startswith("bearer "):
        provided = authorization[len("bearer "):].strip()
    expected = resolve_workbench_job_token()
    if not provided or not secrets.compare_digest(provided, expected):
        raise WorkbenchAPIError("Missing or invalid bearer token.", status_code=401)


class _NoStoreMiddleware(BaseHTTPMiddleware):
    """Attach the ``Cache-Control: no-store`` header used by the stdlib server."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        '''Add the shared no-store cache header to every response.

        Parameters
        ----------
        request : starlette.requests.Request
            Incoming request.
        call_next : Any
            Next handler in the middleware chain.

        Returns
        -------
        starlette.responses.Response
            Response with the shared header applied.
        '''

        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return cast(Response, response)


def build_workbench_api_app(
    root: str | Path,
    *,
    max_depth: int = DEFAULT_OCSCORE_SCAN_DEPTH,
    server_port: int = DEFAULT_WORKBENCH_API_PORT,
    optuna_dashboard_host: str = DEFAULT_OPTUNA_DASHBOARD_HOST,
    optuna_dashboard_port_start: int | None = None,
    optuna_dashboard_port_end: int | None = None,
    optuna_dashboard_slots: int | None = None,
) -> FastAPI:
    '''Build a FastAPI application bound to one OCScore root.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore root.
    max_depth : int
        Maximum recursive depth inside each replica.
    server_port : int
        TCP port the app will be served on (used to auto-select Optuna dashboard ports).
    optuna_dashboard_host : str
        Bind host used for local Optuna dashboard subprocesses.
    optuna_dashboard_port_start : int or None
        Explicit first Optuna dashboard port.
    optuna_dashboard_port_end : int or None
        Explicit last Optuna dashboard port.
    optuna_dashboard_slots : int or None
        Override Optuna dashboard slot count.

    Returns
    -------
    fastapi.FastAPI
        Configured application.
    '''

    root_path = Path(root)
    resolved_optuna_slots = resolve_optuna_dashboard_slot_count(root_path, override=optuna_dashboard_slots)
    optuna_manager = OptunaDashboardManager(
        root_path,
        host=optuna_dashboard_host,
        server_port=server_port,
        port_start=optuna_dashboard_port_start,
        port_end=optuna_dashboard_port_end,
        slot_count=resolved_optuna_slots,
    )
    job_manager = JobManager(root_path)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        '''Stop tracked Optuna dashboard subprocesses on application shutdown.'''

        try:
            yield
        finally:
            optuna_manager.stop_all()

    app = FastAPI(
        title="OCDocker Workbench API",
        version=str(WORKBENCH_API_VERSION),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=_lifespan,
    )
    app.state.workbench_root = root_path
    app.state.workbench_max_depth = max_depth
    app.state.workbench_optuna_manager = optuna_manager
    app.state.workbench_job_manager = job_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(_NoStoreMiddleware)

    @app.exception_handler(WorkbenchAPIError)
    async def _workbench_error_handler(_request: Request, exc: WorkbenchAPIError) -> JSONResponse:
        '''Translate a WorkbenchAPIError into the shared error response shape.'''

        return JSONResponse({"ok": False, "error": str(exc)}, status_code=exc.status_code)

    @app.exception_handler(JobError)
    async def _job_error_handler(_request: Request, exc: JobError) -> JSONResponse:
        '''Translate a JobError into the shared error response shape.'''

        return JSONResponse({"ok": False, "error": str(exc)}, status_code=exc.status_code)

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        '''Translate an unvalidated ValueError into a 400 error response.'''

        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.exception_handler(FileExistsError)
    async def _file_exists_error_handler(_request: Request, exc: FileExistsError) -> JSONResponse:
        '''Translate a FileExistsError into a 409 error response.'''

        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        '''Translate routing/HTTP errors into the shared error response shape.'''

        if exc.status_code == 404:
            return JSONResponse(
                {"ok": False, "error": f"Unknown Workbench API endpoint: {request.url.path}"},
                status_code=404,
            )
        return JSONResponse({"ok": False, "error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        '''Translate any unhandled exception into a 500 error response.'''

        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    @app.get("/")
    @app.get("/api")
    async def get_index() -> dict[str, Any]:
        '''Return the Workbench API endpoint index.'''

        return build_workbench_api_payload(root_path, "/api", max_depth=max_depth)

    @app.get("/health")
    @app.get("/api/health")
    async def get_health() -> dict[str, Any]:
        '''Return the Workbench API health payload.'''

        return build_workbench_api_payload(root_path, "/health", max_depth=max_depth)

    @app.get("/api/ocscore-workspace")
    async def get_ocscore_workspace(
        replicas: int | None = Query(None),
        max_metric_file_bytes: int | None = Query(None),
        metric: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        '''Return the strict OCScore workspace summary for the served root.'''

        query: QueryMap = {"metric": metric}
        if replicas is not None:
            query["replicas"] = [str(replicas)]
        if max_metric_file_bytes is not None:
            query["max_metric_file_bytes"] = [str(max_metric_file_bytes)]
        return build_workbench_api_payload(root_path, "/api/ocscore-workspace", query, max_depth=max_depth)

    @app.get("/api/ablation-design")
    async def get_ablation_design() -> dict[str, Any]:
        '''Return the catalog and workspace defaults for the ablation designer UI.'''

        return build_workbench_api_payload(root_path, "/api/ablation-design", max_depth=max_depth)

    @app.get("/api/ablation-design/features")
    async def get_ablation_design_features() -> dict[str, Any]:
        '''Reject GET on the features endpoint, which requires a POST body.'''

        return build_workbench_api_payload(root_path, "/api/ablation-design/features", max_depth=max_depth)

    @app.post("/api/ablation-design/preview")
    @app.post("/api/ablation-design/plan")
    @app.post("/api/ablation-design/features")
    @app.post("/api/ablation-design/write")
    async def post_ablation_design(request: Request) -> dict[str, Any]:
        '''Dispatch one ablation-design POST endpoint (preview, plan, features, write).'''

        body = await _read_json_body(request)
        return handle_ablation_design_post(root_path, request.url.path, body)

    @app.get("/api/ablation-protocol-similarity")
    async def get_ablation_protocol_similarity(
        metric: str | None = Query(None),
        reference: str | None = Query(None),
        include_catalog_only: str | None = Query(None),
    ) -> dict[str, Any]:
        '''Return the ablation protocol similarity analysis for the served root.'''

        query: QueryMap = {}
        if metric is not None:
            query["metric"] = [metric]
        if reference is not None:
            query["reference"] = [reference]
        if include_catalog_only is not None:
            query["include_catalog_only"] = [include_catalog_only]
        return build_workbench_api_payload(root_path, "/api/ablation-protocol-similarity", query, max_depth=max_depth)

    @app.get("/api/vs-design")
    async def get_vs_design(input_dir: str | None = Query(None)) -> dict[str, Any]:
        '''Discover receptor/ligand/box candidates for a VS design under the served root.'''

        return discover_vs_design_candidates(root_path, input_dir=input_dir, max_depth=max_depth)

    @app.post("/api/vs-design/preview")
    async def post_vs_design_preview(request: Request) -> dict[str, Any]:
        '''Validate one draft VS design (receptor/ligand/box/engine) without running anything.'''

        body = await _read_json_body(request)
        return preview_vs_design(root_path, body)

    @app.post("/api/vs-design/plan")
    async def post_vs_design_plan(request: Request) -> dict[str, Any]:
        '''Build the exact `ocdocker vs`/`pipeline` argv for a valid draft VS design.'''

        body = await _read_json_body(request)
        return plan_vs_design(root_path, body)

    @app.get("/api/vs-campaign")
    async def get_vs_campaign(input_dir: str | None = Query(None)) -> dict[str, Any]:
        '''Discover a draft multi-sample manifest from an `input/{sample}/...` layout.'''

        return discover_vs_campaign_candidates(root_path, input_dir=input_dir)

    @app.post("/api/vs-campaign/preview")
    async def post_vs_campaign_preview(request: Request) -> dict[str, Any]:
        '''Validate a draft multi-sample VS campaign manifest without running anything.'''

        body = await _read_json_body(request)
        return preview_vs_campaign(root_path, body)

    @app.post("/api/vs-campaign/plan")
    async def post_vs_campaign_plan(request: Request) -> dict[str, Any]:
        '''Build the `vs_campaign` job payload for a valid draft manifest.'''

        body = await _read_json_body(request)
        return plan_vs_campaign(root_path, body)

    @app.get("/api/schema")
    async def get_schema(name: list[str] = Query(default=[])) -> dict[str, Any]:
        '''Return the JSON Schema catalog for Workbench models.'''

        query: QueryMap = {"name": name}
        return build_workbench_api_payload(root_path, "/api/schema", query, max_depth=max_depth)

    @app.get("/api/template")
    async def get_template(name: str = Query(...)) -> dict[str, Any]:
        '''Return one bundled starter spec template.'''

        query: QueryMap = {"name": [name]}
        return build_workbench_api_payload(root_path, "/api/template", query, max_depth=max_depth)

    @app.get("/api/figure-asset")
    async def get_figure_asset(path: str = Query(...)) -> Response:
        '''Return one allowed dashboard figure asset from the served root.'''

        body, content_type = _figure_asset(root_path, {"path": [path]})
        return Response(content=body, media_type=content_type)

    @app.get("/api/optuna-dashboard")
    @app.get("/api/optuna-dashboard/status")
    async def get_optuna_dashboard(request: Request, replica_path: str | None = Query(None)) -> dict[str, Any]:
        '''Return local Optuna dashboard status for one or all replicas.'''

        query: QueryMap = {"replica_path": [replica_path]} if replica_path else {}
        return _optuna_dashboard_payload(optuna_manager, request.url.path, query)

    @app.post("/api/optuna-dashboard")
    async def post_optuna_dashboard(request: Request) -> dict[str, Any]:
        '''Start a local Optuna dashboard subprocess for one replica.'''

        body = await _read_json_body(request)
        return _optuna_dashboard_payload(optuna_manager, request.url.path, {}, body)

    @app.delete("/api/optuna-dashboard")
    async def delete_optuna_dashboard(replica_path: str | None = Query(None)) -> dict[str, Any]:
        '''Stop a local Optuna dashboard subprocess for one replica.'''

        query: QueryMap = {"replica_path": [replica_path]} if replica_path else {}
        return _optuna_dashboard_payload(optuna_manager, "/api/optuna-dashboard", query)

    @app.post("/api/jobs", status_code=201, dependencies=[Depends(_require_job_token)])
    async def post_job(payload: JobCreateRequest) -> dict[str, Any]:
        '''Launch a new tracked Workbench job (requires a bearer token).'''

        record = job_manager.launch(
            payload.kind, payload.args, cwd=payload.cwd, manifest=payload.manifest,
            engine=payload.engine, cores=payload.cores, results_dir=payload.results_dir,
        )
        return _model_payload(record)

    @app.post("/api/jobs/plan")
    async def post_job_plan(payload: JobCreateRequest) -> dict[str, Any]:
        '''Preview the command a job would run, without launching it.'''

        return job_manager.plan(
            payload.kind, payload.args, cwd=payload.cwd, manifest=payload.manifest,
            engine=payload.engine, cores=payload.cores, results_dir=payload.results_dir,
        )

    @app.get("/api/jobs")
    async def get_jobs() -> dict[str, Any]:
        '''List every tracked Workbench job.'''

        return {"jobs": [_model_payload(record) for record in job_manager.list()]}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        '''Return one tracked Workbench job.'''

        return _model_payload(job_manager.get(job_id))

    @app.get("/api/jobs/{job_id}/logs")
    async def get_job_logs(
        job_id: str,
        lines: int = Query(DEFAULT_LOG_LINE_LIMIT, ge=1),
        max_bytes: int = Query(DEFAULT_LOG_BYTE_LIMIT, ge=1),
    ) -> dict[str, Any]:
        '''Return a bounded stdout/stderr tail for one tracked Workbench job.'''

        stdout_preview, stderr_preview = job_manager.logs(job_id, lines=lines, max_bytes=max_bytes)
        return {"stdout": _model_payload(stdout_preview), "stderr": _model_payload(stderr_preview)}

    @app.get("/api/jobs/{job_id}/campaign-progress")
    async def get_job_campaign_progress(job_id: str) -> dict[str, Any]:
        '''Return structured per-sample progress for one tracked vs_campaign job.

        Works for any job kind, degrading to ``engine: "unknown"`` when the
        log text does not match either execution engine's format.
        '''

        record = job_manager.get(job_id)
        stdout_preview, stderr_preview = job_manager.logs(
            job_id, lines=CAMPAIGN_PROGRESS_LOG_LINE_LIMIT, max_bytes=CAMPAIGN_PROGRESS_LOG_BYTE_LIMIT,
        )
        log_text = f"{stdout_preview.text}\n{stderr_preview.text}"
        return parse_campaign_progress(log_text, expected_samples=_campaign_expected_samples(record))

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(_require_job_token)])
    async def post_job_cancel(job_id: str) -> dict[str, Any]:
        '''Cancel one running tracked Workbench job (requires a bearer token).'''

        return _model_payload(job_manager.cancel(job_id))

    async def get_web_asset(request: Request) -> Response:
        '''Serve one packaged Workbench browser asset.'''

        path = request.url.path
        if not is_workbench_web_asset_path(path):
            raise WorkbenchAPIError(f"Unknown Workbench web asset: {path}", status_code=404)
        try:
            content_type, body = build_workbench_web_asset(path)
        except (KeyError, FileNotFoundError) as exc:
            raise WorkbenchAPIError(str(exc), status_code=404) from exc
        return Response(content=body, media_type=content_type)

    for web_asset_route in WORKBENCH_WEB_ROUTES:
        app.add_api_route(web_asset_route, get_web_asset, methods=["GET"])

    return app


async def _read_json_body(request: Request) -> dict[str, Any]:
    '''Parse a JSON request body from one FastAPI request.

    Parameters
    ----------
    request : fastapi.Request
        Active request.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.
    '''

    raw = await request.body()
    if not raw:
        raise WorkbenchAPIError("Expected a JSON request body.", status_code=400)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkbenchAPIError("Request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise WorkbenchAPIError("Request body must be a JSON object.", status_code=400)
    return payload


def serve_workbench_api(
    root: str | Path,
    *,
    host: str = DEFAULT_WORKBENCH_API_HOST,
    port: int = DEFAULT_WORKBENCH_API_PORT,
    max_depth: int = DEFAULT_OCSCORE_SCAN_DEPTH,
    optuna_dashboard_host: str = DEFAULT_OPTUNA_DASHBOARD_HOST,
    optuna_dashboard_port_start: int | None = None,
    optuna_dashboard_port_end: int | None = None,
    optuna_dashboard_slots: int | None = None,
    verbose: bool = False,
) -> None:
    '''Serve the OCDocker Workbench API until interrupted.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore root.
    host : str
        Bind host.
    port : int
        Bind port.
    max_depth : int
        Maximum recursive depth inside each replica.
    '''

    import uvicorn

    app = build_workbench_api_app(
        root,
        max_depth=max_depth,
        server_port=port,
        optuna_dashboard_host=optuna_dashboard_host,
        optuna_dashboard_port_start=optuna_dashboard_port_start,
        optuna_dashboard_port_end=optuna_dashboard_port_end,
        optuna_dashboard_slots=optuna_dashboard_slots,
    )
    optuna_manager = app.state.workbench_optuna_manager
    try:
        print(f"Workbench API serving {root} at http://{host}:{port}.")
        print(f"Workbench browser dashboard: http://{host}:{port}/app")
        resolve_workbench_job_token()
        print(f"Job-execute endpoints (/api/jobs*) require a bearer token: {workbench_job_token_path()}")
        if OptunaDashboardManager.is_available():
            pool = optuna_manager.port_pool
            pool_label = ", ".join(str(item) for item in pool)
            if optuna_manager.auto_ports:
                print(
                    "Optuna dashboards: "
                    f"ports {pool_label} "
                    f"({optuna_manager.max_sessions} slots from replica count, auto from server port {port})."
                )
            else:
                print(
                    "Optuna dashboards: "
                    f"ports {pool[0]}-{pool[-1]} "
                    "(explicit range, launched from the UI)."
                )
        else:
            print('Optuna dashboards unavailable: install with pip install "ocdocker[ml]"')
        uvicorn.run(app, host=host, port=port, log_level="info" if verbose else "warning")
    except KeyboardInterrupt:
        print("\nWorkbench API stopped.")
    finally:
        optuna_manager.stop_all()


__all__ = [
    "DEFAULT_WORKBENCH_API_HOST",
    "DEFAULT_WORKBENCH_API_PORT",
    "WORKBENCH_API_VERSION",
    "WorkbenchAPIError",
    "build_workbench_api_app",
    "build_workbench_api_payload",
    "serve_workbench_api",
]
