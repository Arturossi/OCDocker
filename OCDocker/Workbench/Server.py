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
from OCDocker.Workbench.Schema import build_schema_catalog
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
            "/api/schema",
            "/api/template",
        ],
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
    if path == "/api/schema":
        names = _values(request_query, "name")
        return build_schema_catalog(names or None)
    if path == "/api/template":
        return build_template_payload(_required(request_query, "name"))
    raise WorkbenchAPIError(f"Unknown Workbench API endpoint: {endpoint}", status_code=404)


def build_workbench_api_handler(
    root: str | Path,
    *,
    max_depth: int = DEFAULT_OCSCORE_SCAN_DEPTH,
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

    class WorkbenchRequestHandler(BaseHTTPRequestHandler):
        """Request handler for one strict OCScore Workbench root."""

        workbench_root = root_path
        workbench_default_max_depth = max_depth

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

        def log_message(self, format: str, *args: Any) -> None:
            '''Suppress default per-request stderr logging.'''

            return

        def _send_common_headers(self) -> None:
            '''Send headers shared by all API responses.'''

            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
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

    handler = build_workbench_api_handler(root, max_depth=max_depth)
    server = ThreadingHTTPServer((host, port), handler)
    try:
        print(f"Workbench API serving {root} at http://{host}:{port} (read-only).")
        print(f"Workbench browser dashboard: http://{host}:{port}/app")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWorkbench API stopped.")
    finally:
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
