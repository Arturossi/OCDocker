#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for local Optuna dashboard launch helpers.
"""

# Imports
###############################################################################
from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from OCDocker.Workbench.OptunaDashboard import OptunaDashboardError
from OCDocker.Workbench.OptunaDashboard import OptunaDashboardManager
from OCDocker.Workbench.OptunaDashboard import discover_free_ports

# License
###############################################################################
"""OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
"""

# Functions
###############################################################################
## Public ##


def test_discover_free_ports_skips_occupied_ports() -> None:
    '''Auto port discovery skips occupied ports and continues scanning.'''

    def port_free(host: str, port: int) -> bool:
        return port not in {13, 15}

    with patch("OCDocker.Workbench.OptunaDashboard._port_is_free", side_effect=port_free):
        ports = discover_free_ports("127.0.0.1", 11, 5)

    assert ports == [11, 12, 14, 16, 17]


def test_optuna_dashboard_manager_auto_ports_from_server_port(tmp_path) -> None:
    '''When ports are unset, the manager reserves slots after the server port.'''

    with patch("OCDocker.Workbench.OptunaDashboard._port_is_free", return_value=True):
        manager = OptunaDashboardManager(tmp_path, server_port=10, slot_count=5)

    assert manager.auto_ports is True
    assert manager.port_pool == [11, 12, 13, 14, 15]
    assert manager.max_sessions == 5


def test_optuna_dashboard_manager_reuses_running_session(tmp_path) -> None:
    '''Starting the same replica twice returns the existing dashboard session.'''

    replica = tmp_path / "replica_001"
    replica.mkdir()
    (replica / "optuna.db").write_bytes(b"sqlite")
    manager = OptunaDashboardManager(tmp_path, port_start=8790, port_end=8790, slot_count=1)
    process = MagicMock()
    process.pid = 12345
    process.poll.return_value = None
    process.stderr = None

    with patch.object(OptunaDashboardManager, "is_available", return_value=True), patch(
        "OCDocker.Workbench.OptunaDashboard._port_is_free",
        return_value=True,
    ), patch("OCDocker.Workbench.OptunaDashboard.subprocess.Popen", return_value=process):
        first = manager.start(replica)
        second = manager.start(replica)

    assert first["port"] == 8790
    assert second["reused"] is True
    assert second["url"] == first["url"]


def test_optuna_dashboard_manager_rejects_missing_database(tmp_path) -> None:
    '''Replica directories without optuna.db cannot launch a dashboard.'''

    replica = tmp_path / "replica_001"
    replica.mkdir()
    manager = OptunaDashboardManager(tmp_path, server_port=8765, slot_count=5)

    with patch.object(OptunaDashboardManager, "is_available", return_value=True), pytest.raises(
        OptunaDashboardError,
        match="No Optuna database found",
    ):
        manager.start(replica)


def test_optuna_dashboard_manager_rejects_outside_root(tmp_path) -> None:
    '''Replica paths outside the served root are rejected.'''

    outside = tmp_path.parent / "outside_replica"
    outside.mkdir(exist_ok=True)
    (outside / "optuna.db").write_bytes(b"sqlite")
    manager = OptunaDashboardManager(tmp_path / "train", server_port=8765, slot_count=1)

    with patch.object(OptunaDashboardManager, "is_available", return_value=True), pytest.raises(
        OptunaDashboardError,
        match="outside the served OCScore root",
    ):
        manager.start(outside)
