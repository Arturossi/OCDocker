#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only local HTTP API for Workbench GUI integrations.
'''

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

from OCDocker.Workbench.Ablation import build_ablation_analysis
from OCDocker.Workbench.Ablation import parse_ablation_metric
from OCDocker.Workbench.Artifacts import build_artifact_index
from OCDocker.Workbench.Comparison import build_run_comparison
from OCDocker.Workbench.Comparison import parse_comparison_metric
from OCDocker.Workbench.Decision import build_metrics_catalog
from OCDocker.Workbench.Decision import build_pareto_front
from OCDocker.Workbench.Decision import parse_pareto_objective
from OCDocker.Workbench.Evidence import build_evidence_index
from OCDocker.Workbench.Evidence import resolve_evidence_asset
from OCDocker.Workbench.IO import model_to_data
from OCDocker.Workbench.Leaderboard import build_metric_leaderboard
from OCDocker.Workbench.Logs import preview_run_logs
from OCDocker.Workbench.MetricsMatrix import build_metric_matrix
from OCDocker.Workbench.Overview import build_workspace_overview
from OCDocker.Workbench.Plots import build_leaderboard_plot
from OCDocker.Workbench.Plots import build_metric_scatter_plot
from OCDocker.Workbench.Plots import build_parallel_coordinates_plot
from OCDocker.Workbench.Plots import build_pareto_scatter_plot
from OCDocker.Workbench.Registry import scan_workspace
from OCDocker.Workbench.Report import build_analysis_report
from OCDocker.Workbench.Results import summarize_results
from OCDocker.Workbench.RunDetail import build_run_detail
from OCDocker.Workbench.Schema import build_schema_catalog
from OCDocker.Workbench.Status import inspect_run_status
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


def _endpoint_index(root: Path, *, default_max_depth: int) -> dict[str, Any]:
    '''Build the endpoint index payload.

    Parameters
    ----------
    root : pathlib.Path
        Workbench root served by the API.
    default_max_depth : int
        Default scan depth used by endpoints.

    Returns
    -------
    dict[str, Any]
        Endpoint index payload.
    '''

    return {
        "service": "ocdocker-workbench",
        "api_version": WORKBENCH_API_VERSION,
        "root": str(root),
        "default_max_depth": default_max_depth,
        "read_only": True,
        "web_app": "/app",
        "endpoints": [
            "/health",
            "/api/overview",
            "/api/inventory",
            "/api/artifacts",
            "/api/ablations",
            "/api/evidence",
            "/api/evidence-asset",
            "/api/metrics-catalog",
            "/api/metrics-matrix",
            "/api/leaderboard",
            "/api/pareto",
            "/api/plot",
            "/api/report",
            "/api/run-detail",
            "/api/compare",
            "/api/status",
            "/api/logs",
            "/api/results",
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
        First query value or default.
    '''

    values = _values(query, name)
    if not values:
        return default
    return values[0]


def _required(query: QueryMap, name: str) -> str:
    '''Return a required query value or raise an API error.

    Parameters
    ----------
    query : QueryMap
        Parsed query string.
    name : str
        Query key.

    Returns
    -------
    str
        Required query value.
    '''

    value = _first(query, name)
    if value is None:
        raise WorkbenchAPIError(f"Missing required query parameter: {name}")
    return value


def _int_query(query: QueryMap, name: str, default: int) -> int:
    '''Parse an integer query parameter.

    Parameters
    ----------
    query : QueryMap
        Parsed query string.
    name : str
        Query key.
    default : int
        Default value when the key is absent.

    Returns
    -------
    int
        Parsed integer value.
    '''

    raw = _first(query, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise WorkbenchAPIError(f"Query parameter {name} must be an integer.") from exc


def _bool_query(query: QueryMap, name: str, default: bool = False) -> bool:
    '''Parse a boolean query parameter.

    Parameters
    ----------
    query : QueryMap
        Parsed query string.
    name : str
        Query key.
    default : bool
        Default value when the key is absent.

    Returns
    -------
    bool
        Parsed boolean value.
    '''

    raw = _first(query, name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise WorkbenchAPIError(f"Query parameter {name} must be boolean.")


def _max_depth(query: QueryMap, default_max_depth: int) -> int:
    '''Return scan depth from query or API defaults.

    Parameters
    ----------
    query : QueryMap
        Parsed query string.
    default_max_depth : int
        Default scan depth.

    Returns
    -------
    int
        Scan depth.
    '''

    return _int_query(query, "max_depth", default_max_depth)


def _safe_path(root: Path, value: str | None, *, field_name: str) -> Path:
    '''Resolve a request path under the served Workbench root.

    Parameters
    ----------
    root : pathlib.Path
        Served root.
    value : str or None
        Requested path. Relative paths are resolved below ``root``.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    pathlib.Path
        Resolved path.
    '''

    if value is None:
        return root
    raw_path = Path(value)
    path = raw_path if raw_path.is_absolute() else root / raw_path
    root_base = root if root.is_dir() else root.parent
    resolved_root = root_base.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise WorkbenchAPIError(
            f"Query parameter {field_name} must resolve under the served root.",
            status_code=403,
        )
    return path


def _model_payload(model: Any) -> dict[str, Any]:
    '''Return JSON-compatible data for a Workbench model.

    Parameters
    ----------
    model : Any
        Model or mapping.

    Returns
    -------
    dict[str, Any]
        JSON-compatible payload.
    '''

    if hasattr(model, "model_dump"):
        return model_to_data(model)
    if isinstance(model, dict):
        return model
    raise TypeError(f"Unsupported API payload type: {type(model)!r}")


def _plot_payload(root: Path, query: QueryMap, *, default_max_depth: int) -> dict[str, Any]:
    '''Build a plot endpoint payload.

    Parameters
    ----------
    root : pathlib.Path
        Served root.
    query : QueryMap
        Parsed query string.
    default_max_depth : int
        Default scan depth.

    Returns
    -------
    dict[str, Any]
        Plot payload.
    '''

    kind = _required(query, "kind")
    max_depth = _max_depth(query, default_max_depth)
    if kind == "leaderboard":
        metric = _required(query, "metric")
        plot = build_leaderboard_plot(
            root,
            metric_name=metric,
            mode=_first(query, "mode", "max"),
            max_depth=max_depth,
            top_n=_int_query(query, "top_n", 20),
        )
    elif kind == "scatter":
        plot = build_metric_scatter_plot(
            root,
            x_metric=_required(query, "x_metric"),
            y_metric=_required(query, "y_metric"),
            color_metric=_first(query, "color_metric"),
            max_depth=max_depth,
        )
    elif kind == "parallel":
        plot = build_parallel_coordinates_plot(
            root,
            metric_names=_values(query, "metric"),
            max_depth=max_depth,
        )
    elif kind == "pareto":
        objectives = tuple(parse_pareto_objective(value) for value in _values(query, "objective"))
        plot = build_pareto_scatter_plot(root, objectives=objectives, max_depth=max_depth)
    else:
        raise WorkbenchAPIError(f"Unsupported plot kind: {kind}")
    return _model_payload(plot)


def _evidence_asset_response(root: Path, query: QueryMap, *, default_max_depth: int) -> tuple[str, bytes]:
    '''Build a constrained binary evidence asset response.

    Parameters
    ----------
    root : pathlib.Path
        Served Workbench root.
    query : QueryMap
        Parsed query string.
    default_max_depth : int
        Default manifest scan depth.

    Returns
    -------
    tuple[str, bytes]
        HTTP content type and asset bytes.
    '''

    try:
        asset_path, content_type = resolve_evidence_asset(
            root,
            _required(query, "path"),
            max_depth=_max_depth(query, default_max_depth),
            source_depth=_int_query(query, "source_depth", 6),
        )
    except FileNotFoundError as exc:
        raise WorkbenchAPIError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise WorkbenchAPIError(str(exc), status_code=403) from exc
    try:
        return content_type, asset_path.read_bytes()
    except OSError as exc:
        raise WorkbenchAPIError(f"Could not read evidence asset: {asset_path}", status_code=404) from exc


def _json_bytes(payload: dict[str, Any]) -> bytes:
    '''Encode an API payload as JSON bytes.

    Parameters
    ----------
    payload : dict[str, Any]
        API payload.

    Returns
    -------
    bytes
        Encoded JSON payload.
    '''

    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


## Public ##


def build_workbench_api_payload(
    root: str | Path,
    endpoint: str,
    query: QueryMap | None = None,
    *,
    default_max_depth: int = 6,
) -> dict[str, Any]:
    '''Build a read-only Workbench API payload for one endpoint.

    Parameters
    ----------
    root : str or pathlib.Path
        Served Workbench root.
    endpoint : str
        Request path without query string.
    query : QueryMap or None
        Parsed query string values.
    default_max_depth : int
        Default scan depth used when an endpoint omits ``max_depth``.

    Returns
    -------
    dict[str, Any]
        JSON-compatible endpoint payload.
    '''

    root_path = Path(root)
    request_query = {} if query is None else query
    path = endpoint.rstrip("/") or "/"
    max_depth = _max_depth(request_query, default_max_depth)

    if path in {"/", "/api"}:
        return _endpoint_index(root_path, default_max_depth=default_max_depth)
    if path in {"/health", "/api/health"}:
        return {
            "ok": True,
            "service": "ocdocker-workbench",
            "api_version": WORKBENCH_API_VERSION,
            "root": str(root_path),
            "read_only": True,
        }
    if path == "/api/overview":
        return _model_payload(
            build_workspace_overview(
                root_path,
                max_depth=max_depth,
                recent_limit=_int_query(request_query, "recent_limit", 20),
            )
        )
    if path == "/api/inventory":
        return _model_payload(scan_workspace(root_path, max_depth=max_depth))
    if path == "/api/artifacts":
        return _model_payload(
            build_artifact_index(
                root_path,
                kinds=_values(request_query, "kind"),
                roles=_values(request_query, "role"),
                require_existing=_bool_query(request_query, "require_existing", False),
                max_depth=max_depth,
            )
        )
    if path == "/api/evidence":
        return _model_payload(
            build_evidence_index(
                root_path,
                max_depth=max_depth,
                source_depth=_int_query(request_query, "source_depth", 6),
                max_entries=_int_query(request_query, "max_entries", 400),
                max_csv_rows=_int_query(request_query, "max_csv_rows", 1000),
                max_series=_int_query(request_query, "max_series", 8),
                max_shap_features=_int_query(request_query, "max_shap_features", 30),
            )
        )
    if path == "/api/metrics-catalog":
        return _model_payload(build_metrics_catalog(root_path, max_depth=max_depth))
    if path == "/api/metrics-matrix":
        return _model_payload(
            build_metric_matrix(
                root_path,
                metric_names=_values(request_query, "metric"),
                max_depth=max_depth,
            )
        )
    if path == "/api/leaderboard":
        return _model_payload(
            build_metric_leaderboard(
                root_path,
                metric_name=_required(request_query, "metric"),
                mode=_first(request_query, "mode", "max"),
                max_depth=max_depth,
            )
        )
    if path == "/api/pareto":
        objectives = tuple(parse_pareto_objective(value) for value in _values(request_query, "objective"))
        return _model_payload(build_pareto_front(root_path, objectives=objectives, max_depth=max_depth))
    if path == "/api/plot":
        return _plot_payload(root_path, request_query, default_max_depth=default_max_depth)
    if path == "/api/report":
        leaderboards = tuple(parse_pareto_objective(value) for value in _values(request_query, "leaderboard"))
        objectives = tuple(parse_pareto_objective(value) for value in _values(request_query, "objective"))
        return _model_payload(
            build_analysis_report(
                root_path,
                leaderboards=leaderboards,
                pareto_objectives=objectives,
                max_depth=max_depth,
                recent_limit=_int_query(request_query, "recent_limit", 20),
                top_n=_int_query(request_query, "top_n", 5),
            )
        )

    if path == "/api/ablations":
        metrics = tuple(parse_ablation_metric(value) for value in _values(request_query, "metric"))
        return _model_payload(
            build_ablation_analysis(
                root_path,
                baseline_run_id=_first(request_query, "baseline"),
                candidates=_values(request_query, "candidate"),
                metrics=metrics,
                max_depth=max_depth,
            )
        )
    if path == "/api/compare":
        metrics = tuple(parse_comparison_metric(value) for value in _values(request_query, "metric"))
        return _model_payload(
            build_run_comparison(
                root_path,
                baseline_run_id=_required(request_query, "baseline"),
                candidates=_values(request_query, "candidate"),
                metrics=metrics,
                max_depth=max_depth,
            )
        )
    if path == "/api/run-detail":
        target = _safe_path(root_path, _required(request_query, "target"), field_name="target")
        return _model_payload(
            build_run_detail(
                target,
                lines=_int_query(request_query, "lines", 80),
                max_bytes=_int_query(request_query, "max_bytes", 65536),
                encoding=_first(request_query, "encoding", "utf-8"),
            )
        )
    if path == "/api/status":
        target = _safe_path(root_path, _first(request_query, "target"), field_name="target")
        return _model_payload(inspect_run_status(target))
    if path == "/api/logs":
        target = _safe_path(root_path, _first(request_query, "target"), field_name="target")
        return _model_payload(
            preview_run_logs(
                target,
                lines=_int_query(request_query, "lines", 80),
                max_bytes=_int_query(request_query, "max_bytes", 65536),
                encoding=_first(request_query, "encoding", "utf-8"),
            )
        )
    if path == "/api/results":
        manifest = _safe_path(
            root_path,
            _first(request_query, "manifest"),
            field_name="manifest",
        )
        return _model_payload(summarize_results(manifest))
    if path == "/api/schema":
        names = _values(request_query, "name")
        return build_schema_catalog(names or None)
    if path == "/api/template":
        return build_template_payload(_required(request_query, "name"))
    raise WorkbenchAPIError(f"Unknown Workbench API endpoint: {endpoint}", status_code=404)


def build_workbench_api_handler(
    root: str | Path,
    *,
    default_max_depth: int = 6,
) -> type[BaseHTTPRequestHandler]:
    '''Build an HTTP request handler bound to one Workbench root.

    Parameters
    ----------
    root : str or pathlib.Path
        Served Workbench root.
    default_max_depth : int
        Default scan depth used by API endpoints.

    Returns
    -------
    type[http.server.BaseHTTPRequestHandler]
        Configured request handler class.
    '''

    root_path = Path(root)

    class WorkbenchRequestHandler(BaseHTTPRequestHandler):
        """Request handler for one read-only Workbench API root."""

        workbench_root = root_path
        workbench_default_max_depth = default_max_depth

        def do_OPTIONS(self) -> None:
            '''Return local CORS headers for browser-based GUI development.'''

            self.send_response(204)
            self._send_common_headers()
            self.end_headers()

        def do_GET(self) -> None:
            '''Handle one read-only Workbench API or web asset GET request.'''

            parsed = urlparse(self.path)
            if is_workbench_web_asset_path(parsed.path):
                content_type, body = build_workbench_web_asset(parsed.path)
                self._send_bytes(body, content_type=content_type, status_code=200)
                return

            query = parse_qs(parsed.query, keep_blank_values=False)
            if parsed.path == "/api/evidence-asset":
                try:
                    content_type, body = _evidence_asset_response(
                        self.workbench_root,
                        query,
                        default_max_depth=self.workbench_default_max_depth,
                    )
                    self._send_bytes(body, content_type=content_type, status_code=200)
                except WorkbenchAPIError as exc:
                    self._send_json(
                        {"ok": False, "error": str(exc)},
                        status_code=exc.status_code,
                    )
                except Exception as exc:
                    self._send_json(
                        {"ok": False, "error": str(exc)},
                        status_code=500,
                    )
                return
            try:
                payload = build_workbench_api_payload(
                    self.workbench_root,
                    parsed.path,
                    query,
                    default_max_depth=self.workbench_default_max_depth,
                )
                self._send_json(payload, status_code=200)
            except WorkbenchAPIError as exc:
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status_code=exc.status_code,
                )
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status_code=500,
                )

        def log_message(self, format: str, *args: Any) -> None:
            '''Suppress default per-request stderr logging.'''

            return

        def _send_common_headers(self) -> None:
            '''Send headers shared by all API responses.'''

            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")

        def _send_bytes(self, body: bytes, *, content_type: str, status_code: int) -> None:
            '''Send one byte response.'''

            self.send_response(status_code)
            self._send_common_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict[str, Any], *, status_code: int) -> None:
            '''Send one JSON response.'''

            self._send_bytes(
                _json_bytes(payload),
                content_type="application/json; charset=utf-8",
                status_code=status_code,
            )

    return WorkbenchRequestHandler


def serve_workbench_api(
    root: str | Path,
    *,
    host: str = DEFAULT_WORKBENCH_API_HOST,
    port: int = DEFAULT_WORKBENCH_API_PORT,
    max_depth: int = 6,
) -> None:
    '''Serve the read-only Workbench API until interrupted.

    Parameters
    ----------
    root : str or pathlib.Path
        Served Workbench root.
    host : str
        Host interface to bind. Use ``127.0.0.1`` for SSH port forwarding.
    port : int
        TCP port to bind.
    max_depth : int
        Default scan depth used by endpoints.
    '''

    handler = build_workbench_api_handler(root, default_max_depth=max_depth)
    server = ThreadingHTTPServer((host, port), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = [
    "DEFAULT_WORKBENCH_API_HOST",
    "DEFAULT_WORKBENCH_API_PORT",
    "WORKBENCH_API_VERSION",
    "QueryMap",
    "WorkbenchAPIError",
    "build_workbench_api_handler",
    "build_workbench_api_payload",
    "serve_workbench_api",
]
