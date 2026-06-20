#!/usr/bin/env python3

# Description
###############################################################################
"""
Focused tests for explicit database initialization and DB helper hardening.
"""

# Imports
###############################################################################
import importlib
import sys

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import scoped_session, sessionmaker

import OCDocker.DB.DB as ocdb
import OCDocker.DB.DBMinimal as dbmin
import OCDocker.DB.Models.Base as base_mod
from OCDocker.DB.DBMinimal import (
    cleanup_engine,
    cleanup_session,
    create_engine,
    create_session,
)
from OCDocker.DB.Models.Base import Base
from OCDocker.DB.Models.Complexes import Complexes
from OCDocker.DB.Models.Ligands import Ligands
from OCDocker.DB.Models.Receptors import Receptors

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Functions
###############################################################################
## Private ##


def _sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = scoped_session(sessionmaker(bind=engine))
    return engine, Session


## Public ##


def test_db_module_imports_do_not_bootstrap_or_print(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OC_BUILD_DOCS", raising=False)

    for module_name in [
        "OCDocker.DB",
        "OCDocker.DB.DB",
        "OCDocker.DB.DBMinimal",
        "OCDocker.DB.baseDB",
    ]:
        importlib.import_module(module_name)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert not list(tmp_path.glob("*.db"))

    import OCDocker.Initialise as ocinit

    assert ocinit.bootstrapped is False


def test_invalid_backend_fails_clearly():
    with pytest.raises(
        dbmin.DatabaseConfigurationError, match="Unsupported database backend"
    ):
        dbmin.validate_database_config("oracle")


def test_invalid_port_fails_clearly():
    with pytest.raises(dbmin.DatabaseConfigurationError, match="must be an integer"):
        dbmin.validate_database_config("postgresql", port="not-a-port")


def test_missing_sqlite_path_handling_is_clear():
    with pytest.raises(
        dbmin.DatabaseConfigurationError, match="SQLite backend requires"
    ):
        dbmin.validate_database_config("sqlite", require_sqlite_path=True)


def test_missing_sqlite_parent_directory_fails_clearly(tmp_path):
    missing_parent = tmp_path / "missing" / "ocdocker.db"
    with pytest.raises(
        dbmin.DatabaseConfigurationError, match="directory does not exist"
    ):
        dbmin.validate_database_config("sqlite", sqlite_path=str(missing_parent))


def test_missing_sqlalchemy_utils_dependency_has_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "sqlalchemy_utils", None)
    with pytest.raises(dbmin.MissingDatabaseDependencyError) as excinfo:
        dbmin.database_exists(URL.create(drivername="postgresql", database="missing"))

    message = str(excinfo.value)
    assert "pip install" in message
    assert "ocdocker[db]" in message


def test_remote_database_creation_requires_explicit_opt_in(monkeypatch):
    url = URL.create(drivername="postgresql", database="ocdocker")
    created = []

    monkeypatch.setattr(dbmin, "database_exists", lambda _url: False)
    monkeypatch.setattr(dbmin, "create_database", lambda _url: created.append(_url))

    with pytest.raises(
        dbmin.DatabaseCreationNotAllowedError, match="create_if_missing=True"
    ):
        dbmin.create_database_if_not_exists(url)
    assert created == []

    assert dbmin.create_database_if_not_exists(url, create_if_missing=True) is True
    assert created == [url]


def test_sqlite_temporary_file_setup_and_table_creation(tmp_path):
    db_file = tmp_path / "ocdocker.db"
    engine = create_engine(URL.create(drivername="sqlite", database=str(db_file)))
    try:
        ocdb.create_tables(engine)
        assert db_file.exists()
    finally:
        cleanup_engine(engine)


def test_db_operation_accepts_injected_session():
    engine, Session = _sqlite_session()
    try:
        assert Ligands.insert({"name": "injected"}, session=Session) is True
        found = Ligands.find_first("injected", session=Session)
        assert getattr(found, "name", None) == "injected"
    finally:
        cleanup_session(Session)
        cleanup_engine(engine)


def test_default_session_fallback_after_explicit_initialization(monkeypatch):
    engine, Session = _sqlite_session()
    try:
        import OCDocker.Initialise as ocinit

        monkeypatch.setattr(base_mod, "session", None)
        monkeypatch.setattr(ocinit, "session", Session, raising=False)
        monkeypatch.setattr(ocinit, "engine", engine, raising=False)
        monkeypatch.setattr(ocinit, "db_url", engine.url, raising=False)

        assert Ligands.insert({"name": "fallback"}) is True
        found = Ligands.find_first("fallback")
        assert getattr(found, "name", None) == "fallback"
    finally:
        cleanup_session(Session)
        cleanup_engine(engine)


def test_export_helper_runs_against_small_sqlite_fixture():
    engine = create_engine("sqlite:///:memory:")
    try:
        ocdb.create_tables(engine)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            ligand = Ligands(name="lig_1")
            receptor = Receptors(name="rec_1")
            session.add_all([ligand, receptor])
            session.flush()
            session.add(
                Complexes(
                    name="complex_1",
                    ligand_id=ligand.id,
                    receptor_id=receptor.id,
                    OCSCORE=1.25,
                )
            )
            session.commit()

            df = ocdb.export_db_to_csv(
                session, output_format="dataframe", drop_na=False
            )

        assert isinstance(df, pd.DataFrame)
        assert df.loc[0, "name"] == "complex_1"
        assert df.loc[0, "OCSCORE"] == 1.25
    finally:
        cleanup_engine(engine)


def test_setup_database_and_cleanup_on_sqlite_memory():
    engine = ocdb.setup_database("sqlite:///:memory:")
    session = create_session(engine)
    try:
        assert session is not None
        with session() as db_session:
            df = ocdb.export_db_to_csv(db_session, output_format="dataframe")
            assert isinstance(df, pd.DataFrame)
    finally:
        cleanup_session(session)
        cleanup_engine(engine)


@pytest.mark.database
@pytest.mark.external
def test_optional_postgresql_backend_is_skipped_unless_configured():
    pytest.skip(
        "Set a dedicated external PostgreSQL test URL before enabling this test."
    )


@pytest.mark.database
@pytest.mark.external
def test_optional_mysql_backend_is_skipped_unless_configured():
    pytest.skip("Set OCDOCKER_TEST_MYSQL_URL before enabling this test.")
