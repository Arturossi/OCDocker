#!/usr/bin/env python3

# Description
###############################################################################
"""
Local HTTP API for the strict OCScore Workbench dashboard.
"""

# Imports
###############################################################################
from __future__ import annotations

import json

from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse

from OCDocker.Workbench.IO import model_to_data
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
from OCDocker.Workbench.Templates import build_template_payload
from OCDocker.Workbench.Web import build_workbench_web_asset
from OCDocker.Workbench.Web import is_workbench_web_asset_path

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
        "read_only": True,
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
            "/api/schema",
            "/api/template",
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
            "read_only": True,
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


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    '''Parse a JSON request body from one HTTP handler.

    Parameters
    ----------
    handler : http.server.BaseHTTPRequestHandler
        Active request handler.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.
    '''

    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        raise WorkbenchAPIError("Expected a JSON request body.", status_code=400)
    raw = handler.rfile.read(length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkbenchAPIError("Request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise WorkbenchAPIError("Request body must be a JSON object.", status_code=400)
    return payload


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


def build_workbench_api_handler(
    root: str | Path,
    *,
    max_depth: int = DEFAULT_OCSCORE_SCAN_DEPTH,
    server_port: int = DEFAULT_WORKBENCH_API_PORT,
    optuna_dashboard_host: str = DEFAULT_OPTUNA_DASHBOARD_HOST,
    optuna_dashboard_port_start: int | None = None,
    optuna_dashboard_port_end: int | None = None,
    optuna_dashboard_slots: int | None = None,
    verbose: bool = False,
) -> type[BaseHTTPRequestHandler]:
    '''Build an HTTP handler bound to one OCScore root.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore root.
    max_depth : int
        Maximum recursive depth inside each replica.

    Returns
    -------
    type[http.server.BaseHTTPRequestHandler]
        Configured request handler class.
    '''

    root_path = Path(root)
    resolved_optuna_slots = resolve_optuna_dashboard_slot_count(
        root_path,
        override=optuna_dashboard_slots,
    )
    optuna_manager = OptunaDashboardManager(
        root_path,
        host=optuna_dashboard_host,
        server_port=server_port,
        port_start=optuna_dashboard_port_start,
        port_end=optuna_dashboard_port_end,
        slot_count=resolved_optuna_slots,
    )

    class WorkbenchRequestHandler(BaseHTTPRequestHandler):
        """Request handler for one strict OCScore Workbench root."""

        workbench_root = root_path
        workbench_default_max_depth = max_depth
        workbench_optuna_manager = optuna_manager
        workbench_verbose = verbose

        def do_OPTIONS(self) -> None:
            '''Return local CORS headers for browser-based GUI development.'''

            self.send_response(204)
            self._send_common_headers()
            self.end_headers()

        def do_GET(self) -> None:
            '''Handle one read-only API or web asset GET request.'''

            parsed = urlparse(self.path)
            if is_workbench_web_asset_path(parsed.path):
                content_type, body = build_workbench_web_asset(parsed.path)
                self._send_bytes(body, content_type=content_type, status_code=200)
                return
            query = parse_qs(parsed.query, keep_blank_values=False)
            try:
                if parsed.path == "/api/figure-asset":
                    body, content_type = _figure_asset(self.workbench_root, query)
                    self._send_bytes(body, content_type=content_type, status_code=200)
                    return
                if parsed.path in {"/api/optuna-dashboard", "/api/optuna-dashboard/status"}:
                    payload = _optuna_dashboard_payload(self.workbench_optuna_manager, parsed.path, query)
                    self._send_json(payload, status_code=200)
                    return
                payload = build_workbench_api_payload(
                    self.workbench_root,
                    parsed.path,
                    query,
                    max_depth=self.workbench_default_max_depth,
                )
                self._send_json(payload, status_code=200)
            except WorkbenchAPIError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)

        def do_POST(self) -> None:
            '''Handle local Optuna dashboard and ablation-design requests.'''

            parsed = urlparse(self.path)
            try:
                if parsed.path in {
                    "/api/ablation-design/preview",
                    "/api/ablation-design/plan",
                    "/api/ablation-design/features",
                    "/api/ablation-design/write",
                }:
                    body = _read_json_body(self)
                    payload = handle_ablation_design_post(self.workbench_root, parsed.path, body)
                    self._send_json(payload, status_code=200)
                    return
                if parsed.path != "/api/optuna-dashboard":
                    self._send_json({"ok": False, "error": "Unknown Workbench API endpoint."}, status_code=404)
                    return
                body = _read_json_body(self)
                payload = _optuna_dashboard_payload(self.workbench_optuna_manager, parsed.path, {}, body)
                self._send_json(payload, status_code=200)
            except WorkbenchAPIError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=400)
            except FileExistsError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=409)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)

        def do_DELETE(self) -> None:
            '''Handle local Optuna dashboard stop requests.'''

            parsed = urlparse(self.path)
            if parsed.path != "/api/optuna-dashboard":
                self._send_json({"ok": False, "error": "Unknown Workbench API endpoint."}, status_code=404)
                return
            query = parse_qs(parsed.query, keep_blank_values=False)
            try:
                payload = _optuna_dashboard_payload(self.workbench_optuna_manager, parsed.path, query)
                self._send_json(payload, status_code=200)
            except WorkbenchAPIError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status_code=500)

        def log_message(self, format: str, *args: Any) -> None:
            '''Log HTTP requests when verbose mode is enabled.'''

            if self.workbench_verbose:
                super().log_message(format, *args)

        def _send_common_headers(self) -> None:
            '''Send headers shared by all API responses.'''

            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")

        def _send_json(self, payload: dict[str, Any], *, status_code: int) -> None:
            '''Send a JSON response.

            Parameters
            ----------
            payload : dict[str, Any]
                Response payload.
            status_code : int
                HTTP status code.
            '''

            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self._send_bytes(body, content_type="application/json; charset=utf-8", status_code=status_code)

        def _send_bytes(self, body: bytes, *, content_type: str, status_code: int) -> None:
            '''Send a byte response.

            Parameters
            ----------
            body : bytes
                Response body.
            content_type : str
                Content type header.
            status_code : int
                HTTP status code.
            '''

            self.send_response(status_code)
            self._send_common_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return WorkbenchRequestHandler


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
    '''Serve the strict OCScore Workbench API until interrupted.

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

    handler = build_workbench_api_handler(
        root,
        max_depth=max_depth,
        server_port=port,
        optuna_dashboard_host=optuna_dashboard_host,
        optuna_dashboard_port_start=optuna_dashboard_port_start,
        optuna_dashboard_port_end=optuna_dashboard_port_end,
        optuna_dashboard_slots=optuna_dashboard_slots,
        verbose=verbose,
    )
    server = ThreadingHTTPServer((host, port), handler)
    optuna_manager = handler.workbench_optuna_manager
    try:
        print(f"Workbench API serving {root} at http://{host}:{port} (read-only).")
        print(f"Workbench browser dashboard: http://{host}:{port}/app")
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
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWorkbench API stopped.")
    finally:
        optuna_manager.stop_all()
        server.server_close()


__all__ = [
    "DEFAULT_WORKBENCH_API_HOST",
    "DEFAULT_WORKBENCH_API_PORT",
    "WORKBENCH_API_VERSION",
    "WorkbenchAPIError",
    "build_workbench_api_handler",
    "build_workbench_api_payload",
    "serve_workbench_api",
]
