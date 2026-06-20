#!/usr/bin/env python3

# Description
###############################################################################
'''
SQLite Optuna storage helpers for OCScore staged optimization.

Usage:

from OCDocker.OCScore.Optimization.OptunaStorage import resolve_optuna_storage
'''

# Imports
###############################################################################
from __future__ import annotations

import sqlite3

from pathlib import Path
from typing import Optional

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

DEFAULT_OPTUNA_DB_FILENAME = "optuna.db"


# Functions
###############################################################################
## Public ##

def configure_sqlite_wal(db_path: Path) -> None:
    '''Enable WAL journal mode on a SQLite database file.

    Creates the parent directory and database file when they do not exist.

    Parameters
    ----------
    db_path : pathlib.Path
        SQLite database path.
    '''

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.commit()
    finally:
        connection.close()


def resolve_optuna_storage(
        storage: Optional[str],
        output_dir: Path,
        filename: str = DEFAULT_OPTUNA_DB_FILENAME,
    ) -> tuple[Optional[str], Optional[str]]:
    '''Resolve an Optuna storage URL and optional local SQLite path.

    Parameters
    ----------
    storage : str | None
        Optuna storage URL. Use ``"auto"`` for ``{output_dir}/{filename}``.
    output_dir : pathlib.Path
        Protocol output directory used when ``storage`` is ``"auto"``.
    filename : str, optional
        SQLite filename for ``"auto"`` storage, by default ``"optuna.db"``.

    Returns
    -------
    tuple[str | None, str | None]
        Optuna storage URL and local SQLite path when applicable.
    '''

    if storage is None:
        return None, None

    if storage == "auto":
        db_path = Path(output_dir) / filename
        configure_sqlite_wal(db_path)
        return f"sqlite:///{db_path}", str(db_path)

    if storage.startswith("sqlite:///"):
        db_path = Path(storage[len("sqlite:///"):])
        configure_sqlite_wal(db_path)
        return storage, str(db_path)

    return storage, storage
