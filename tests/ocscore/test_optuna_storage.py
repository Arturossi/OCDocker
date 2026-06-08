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
