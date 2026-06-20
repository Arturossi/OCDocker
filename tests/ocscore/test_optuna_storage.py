#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for shared Optuna SQLite storage helpers."""

# Imports
###############################################################################
import sqlite3

from pathlib import Path

import pytest

from OCDocker.OCScore.Optimization.OptunaStorage import configure_sqlite_wal
from OCDocker.OCScore.Optimization.OptunaStorage import resolve_optuna_storage

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""


# Functions
###############################################################################
## Public ##

@pytest.mark.order(494)
def test_configure_sqlite_wal_enables_wal_mode(tmp_path):
    db_path = tmp_path / "optuna.db"
    configure_sqlite_wal(db_path)

    connection = sqlite3.connect(str(db_path))
    try:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        connection.close()

    assert journal_mode.lower() == "wal"


@pytest.mark.order(494)
def test_resolve_optuna_storage_auto_uses_single_output_db(tmp_path):
    storage_url, storage_path = resolve_optuna_storage("auto", tmp_path)

    assert storage_url == f"sqlite:///{tmp_path / 'optuna.db'}"
    assert storage_path == str(tmp_path / "optuna.db")
    assert (tmp_path / "optuna.db").is_file()
