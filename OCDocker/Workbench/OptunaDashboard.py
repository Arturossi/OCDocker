#!/usr/bin/env python3

# Description
###############################################################################
"""
Local Optuna dashboard launcher for the OCScore Workbench.
"""

# Imports
###############################################################################
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from OCDocker.OCScore.Optimization.OptunaStorage import DEFAULT_OPTUNA_DB_FILENAME

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

# Constants
###############################################################################

DEFAULT_OPTUNA_DASHBOARD_HOST = "127.0.0.1"
MIN_OPTUNA_DASHBOARD_SLOT_COUNT = 1
_OPTUNA_DASHBOARD_STARTUP_SECONDS = 0.35
_OPTUNA_PORT_SCAN_LIMIT = 256


# Classes
###############################################################################


@dataclass(frozen=True)
class OptunaDashboardSession:
    """One locally launched Optuna dashboard process."""

    replica_path: str
    storage_path: str
    host: str
    port: int
    pid: int
    url: str


class OptunaDashboardError(Exception):
    """Raised when Optuna dashboard launch or lookup fails."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class OptunaDashboardManager:
    """Launch and track local Optuna dashboard processes for replica databases."""

    def __init__(
        self,
        root: str | Path,
        *,
        host: str = DEFAULT_OPTUNA_DASHBOARD_HOST,
        server_port: int | None = None,
        port_start: int | None = None,
        port_end: int | None = None,
        slot_count: int,
    ) -> None:
        if slot_count < MIN_OPTUNA_DASHBOARD_SLOT_COUNT:
            raise ValueError("optuna dashboard slot_count must be >= 1")
        self.root = Path(root).resolve()
        self.host = host
        self.auto_ports = port_start is None and port_end is None
        if self.auto_ports:
            if server_port is None:
                raise ValueError("server_port is required when optuna dashboard ports are not explicitly configured")
            self.port_scan_start = server_port + 1
            self.port_scan_end = None
            self.max_sessions = slot_count
            self.port_pool = discover_free_ports(host, self.port_scan_start, slot_count)
        else:
            if port_start is None or port_end is None:
                raise ValueError("optuna dashboard port_start and port_end must both be set for an explicit range")
            if port_end < port_start:
                raise ValueError("optuna dashboard port_end must be >= port_start")
            self.port_scan_start = port_start
            self.port_scan_end = port_end
            self.max_sessions = port_end - port_start + 1
            self.port_pool = list(range(port_start, port_end + 1))
        self._sessions: dict[str, OptunaDashboardSession] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}

    @staticmethod
    def is_available() -> bool:
        '''Return whether an Optuna dashboard launcher is available locally.'''

        if shutil.which("optuna-dashboard"):
            return True
        try:
            import optuna_dashboard  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def resolve_replica_storage(replica_path: Path) -> Path:
        '''Return the Optuna SQLite database for one replica directory.

        Parameters
        ----------
        replica_path : pathlib.Path
            Replica output directory.

        Returns
        -------
        pathlib.Path
            Resolved Optuna database path.
        '''

        candidate = replica_path / DEFAULT_OPTUNA_DB_FILENAME
        if candidate.is_file():
            return candidate.resolve()
        raise OptunaDashboardError(
            f"No Optuna database found under {replica_path}. Expected {DEFAULT_OPTUNA_DB_FILENAME}.",
            status_code=404,
        )

    def allowed_roots(self) -> tuple[Path, ...]:
        '''Return filesystem roots that may host replica Optuna databases.'''

        roots = [self.root]
        if self.root.name == "train":
            roots.append(self.root.parent)
        resolved: list[Path] = []
        for item in roots:
            try:
                resolved.append(item.resolve())
            except OSError:
                continue
        return tuple(resolved)

    def validate_replica_path(self, replica_path: Path) -> Path:
        '''Resolve and authorize one replica directory path.

        Parameters
        ----------
        replica_path : pathlib.Path
            Requested replica directory.

        Returns
        -------
        pathlib.Path
            Resolved replica directory.
        '''

        try:
            resolved = replica_path.resolve(strict=True)
        except OSError as exc:
            raise OptunaDashboardError("Replica path does not exist.", status_code=404) from exc
        if not resolved.is_dir():
            raise OptunaDashboardError("Replica path is not a directory.", status_code=400)
        if not any(_is_relative_to(resolved, allowed_root) for allowed_root in self.allowed_roots()):
            raise OptunaDashboardError("Replica path is outside the served OCScore root.", status_code=403)
        return resolved

    def start(self, replica_path: str | Path) -> dict[str, Any]:
        '''Launch or reuse an Optuna dashboard for one replica.

        Parameters
        ----------
        replica_path : str or pathlib.Path
            Replica directory containing ``optuna.db``.

        Returns
        -------
        dict[str, Any]
            JSON-safe session payload.
        '''

        if not self.is_available():
            raise OptunaDashboardError(
                'Optuna dashboard is not installed. Install with: pip install "ocdocker[ml]"',
                status_code=503,
            )
        self.cleanup_dead()
        resolved_replica = self.validate_replica_path(Path(replica_path))
        storage_path = self.resolve_replica_storage(resolved_replica)
        key = str(storage_path)
        existing = self._sessions.get(key)
        if existing is not None and self._process_alive(key):
            return self._session_payload(existing, reused=True)
        port = self._allocate_port()
        process = self._launch(storage_path, port)
        session = OptunaDashboardSession(
            replica_path=str(resolved_replica),
            storage_path=key,
            host=self.host,
            port=port,
            pid=process.pid,
            url=f"http://{self.host}:{port}/",
        )
        self._sessions[key] = session
        self._processes[key] = process
        return self._session_payload(session, reused=False)

    def status(self, replica_path: str | Path | None = None) -> dict[str, Any]:
        '''Return dashboard session status for one replica or all sessions.

        Parameters
        ----------
        replica_path : str, pathlib.Path, or None
            Optional replica directory filter.

        Returns
        -------
        dict[str, Any]
            JSON-safe status payload.
        '''

        self.cleanup_dead()
        if replica_path is None:
            return {
                "ok": True,
                "available": self.is_available(),
                "host": self.host,
                "auto_ports": self.auto_ports,
                "port_pool": self.port_pool,
                "max_sessions": self.max_sessions,
                "sessions": [self._session_payload(session, reused=True) for session in self._sessions.values()],
            }
        resolved_replica = self.validate_replica_path(Path(replica_path))
        storage_path = str(self.resolve_replica_storage(resolved_replica))
        session = self._sessions.get(storage_path)
        if session is None or not self._process_alive(storage_path):
            return {
                "ok": True,
                "running": False,
                "replica_path": str(resolved_replica),
                "storage_path": storage_path,
            }
        return {"ok": True, "running": True, **self._session_payload(session, reused=True)}

    def stop(self, replica_path: str | Path) -> dict[str, Any]:
        '''Stop a running Optuna dashboard for one replica.

        Parameters
        ----------
        replica_path : str or pathlib.Path
            Replica directory.

        Returns
        -------
        dict[str, Any]
            JSON-safe stop result.
        '''

        resolved_replica = self.validate_replica_path(Path(replica_path))
        storage_path = str(self.resolve_replica_storage(resolved_replica))
        stopped = self._terminate(storage_path)
        return {
            "ok": True,
            "stopped": stopped,
            "replica_path": str(resolved_replica),
            "storage_path": storage_path,
        }

    def stop_all(self) -> None:
        '''Stop every tracked Optuna dashboard process.'''

        for key in list(self._processes):
            self._terminate(key)

    def cleanup_dead(self) -> None:
        '''Drop sessions whose child processes have already exited.'''

        for key in list(self._processes):
            if not self._process_alive(key):
                self._terminate(key, ignore_missing=True)

    def _session_payload(self, session: OptunaDashboardSession, *, reused: bool) -> dict[str, Any]:
        return {
            "replica_path": session.replica_path,
            "storage_path": session.storage_path,
            "host": session.host,
            "port": session.port,
            "pid": session.pid,
            "url": session.url,
            "reused": reused,
            "running": True,
        }

    def _allocate_port(self) -> int:
        used = {session.port for session in self._sessions.values() if self._process_alive(session.storage_path)}
        if len(used) >= self.max_sessions:
            raise OptunaDashboardError(
                f"All {self.max_sessions} Optuna dashboard slots are in use. "
                "Stop an existing dashboard before launching another.",
                status_code=503,
            )
        port = self.port_scan_start
        scanned = 0
        while scanned < _OPTUNA_PORT_SCAN_LIMIT:
            if self.port_scan_end is not None and port > self.port_scan_end:
                break
            if port not in used and _port_is_free(self.host, port):
                return port
            port += 1
            scanned += 1
        if self.auto_ports:
            detail = (
                f"No free Optuna dashboard port found starting at {self.port_scan_start} "
                f"within {_OPTUNA_PORT_SCAN_LIMIT} candidates."
            )
        else:
            detail = (
                f"No free Optuna dashboard ports in range {self.port_scan_start}-{self.port_scan_end}. "
                "Stop an existing dashboard or widen the configured port range."
            )
        raise OptunaDashboardError(detail, status_code=503)

    def _launch(self, storage_path: Path, port: int) -> subprocess.Popen[Any]:
        storage_url = f"sqlite:///{storage_path}"
        command = _dashboard_command(storage_url, host=self.host, port=port)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise OptunaDashboardError(f"Could not launch Optuna dashboard: {exc}") from exc
        time.sleep(_OPTUNA_DASHBOARD_STARTUP_SECONDS)
        if process.poll() is not None:
            stderr = ""
            if process.stderr is not None:
                stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            detail = f": {stderr}" if stderr else ""
            raise OptunaDashboardError(f"Optuna dashboard exited immediately{detail}", status_code=500)
        return process

    def _process_alive(self, storage_key: str) -> bool:
        process = self._processes.get(storage_key)
        return process is not None and process.poll() is None

    def _terminate(self, storage_key: str, *, ignore_missing: bool = False) -> bool:
        process = self._processes.pop(storage_key, None)
        self._sessions.pop(storage_key, None)
        if process is None:
            return False if not ignore_missing else False
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        return True


# Functions
###############################################################################
## Private ##


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def discover_free_ports(
    host: str,
    start: int,
    count: int,
    *,
    max_probe: int = _OPTUNA_PORT_SCAN_LIMIT,
) -> list[int]:
    '''Return the first ``count`` free TCP ports at or above ``start``.

    Parameters
    ----------
    host : str
        Bind host to probe.
    start : int
        First candidate port.
    count : int
        Number of free ports to discover.
    max_probe : int
        Maximum number of candidate ports to inspect.

    Returns
    -------
    list[int]
        Discovered free ports in ascending order.
    '''

    ports: list[int] = []
    port = start
    limit = start + max_probe
    while len(ports) < count and port < limit:
        if _port_is_free(host, port):
            ports.append(port)
        port += 1
    if len(ports) < count:
        raise ValueError(
            f"Could only find {len(ports)} free Optuna dashboard ports starting at {start}.",
        )
    return ports


def _dashboard_command(storage_url: str, *, host: str, port: int) -> list[str]:
    if shutil.which("optuna-dashboard"):
        return ["optuna-dashboard", storage_url, "--host", host, "--port", str(port)]
    return [sys.executable, "-m", "optuna_dashboard", storage_url, "--host", host, "--port", str(port)]


__all__ = [
    "DEFAULT_OPTUNA_DASHBOARD_HOST",
    "MIN_OPTUNA_DASHBOARD_SLOT_COUNT",
    "OptunaDashboardError",
    "OptunaDashboardManager",
    "OptunaDashboardSession",
    "discover_free_ports",
]
