#!/usr/bin/env python3

# Description
###############################################################################
'''
Exercise CRUD helpers in DB.Models.Base using an in-memory SQLite session and
the concrete Ligands model.
'''

# Imports
###############################################################################
import pytest

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import scoped_session, sessionmaker

import OCDocker.DB.Models.Base as base_mod
from OCDocker.DB.DBMinimal import create_engine
from OCDocker.DB.Models.Base import Base
from OCDocker.DB.Models.Ligands import Ligands

# License
###############################################################################
'''
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
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

def _make_sqlite_session():
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    Base.metadata.create_all(engine)
    Session = scoped_session(sessionmaker(bind=engine))
    return Session


## Public ##

@pytest.mark.order(49)
def test_base_crud_on_ligands_sqlite_memory():
    # Prepare transient engine + session and patch into Base module
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    Base.metadata.create_all(engine)
    Session = scoped_session(sessionmaker(bind=engine))
    base_mod.session = Session  # patch module-level session used by Base methods

    # Insert
    assert Ligands.insert({"name": "L1"}) is True

    # Insert-or-update (update path)
    assert Ligands.insert_or_update({"name": "L1"}) is True

    # Update by name
    assert Ligands.update("L1", {"name": "L1"}) is True

    # Delete by name
    assert Ligands.delete("L1") is True

    # Column helpers
    assert Ligands.determine_column_type("countAtoms").__class__.__name__ in ("Integer", "INTEGER")
    assert Ligands.determine_column_type("someFloat").__class__.__name__ in ("Float", "FLOAT")


@pytest.mark.order(251)
def test_base_methods_return_defaults_when_session_not_created(monkeypatch):
    calls = []
    monkeypatch.setattr(base_mod, "session", None)
    monkeypatch.setattr(base_mod.ocerror.Error, "session_not_created", lambda msg: calls.append(msg))

    assert Ligands.delete("missing") is False
    assert Ligands.find("missing") == []
    assert Ligands.find_all() == []
    assert Ligands.find_all_names() == []
    assert Ligands.find_attribute("name", "missing") == []
    assert Ligands.find_first("missing") is None
    assert Ligands.insert({"name": "X"}) is False
    assert Ligands.insert_or_update({"name": "X"}) is False
    assert Ligands.update("missing", {"name": "X"}) is False
    assert len(calls) == 9


@pytest.mark.order(252)
def test_base_validation_and_lookup_branches(monkeypatch):
    Session = _make_sqlite_session()
    monkeypatch.setattr(base_mod, "session", Session)

    malformed = []
    monkeypatch.setattr(base_mod.ocerror.Error, "malformed_payload", lambda msg: malformed.append(msg))

    assert Ligands.insert({"bad": "payload"}) is False
    assert Ligands.find_attribute("name", "L", operator="<=>") == []
    assert Ligands.find_attribute("unknown_column", "L", operator="==") == []
    assert len(malformed) == 3


@pytest.mark.order(253)
def test_base_duplicate_and_not_found_paths(monkeypatch):
    Session = _make_sqlite_session()
    monkeypatch.setattr(base_mod, "session", Session)

    duplicates = []
    missing = []
    monkeypatch.setattr(base_mod.ocerror.Error, "data_already_exists", lambda msg: duplicates.append(msg))
    monkeypatch.setattr(base_mod.ocerror.Error, "data_not_found", lambda msg: missing.append(msg))

    assert Ligands.insert({"name": "dup"}) is True
    assert Ligands.insert({"name": "dup"}, ignorePresence=True) is True
    assert Ligands.insert({"name": "dup"}) is False
    assert Ligands.delete("does_not_exist") is False
    assert Ligands.update("does_not_exist", {"name": "other"}) is False
    assert duplicates
    assert len(missing) == 2

    assert "dup" in Ligands.find_all_names()
    assert len(Ligands.find_attribute("name", "dup", operator="==")) == 1
    assert len(Ligands.find_attribute("name", ["dup", "x"], operator="in")) == 1
    assert Ligands.to_dict()["name"] is not None


@pytest.mark.order(254)
def test_base_handles_sqlalchemy_commit_failures(monkeypatch):
    Session = _make_sqlite_session()
    monkeypatch.setattr(base_mod, "session", Session)
    assert Ligands.insert({"name": "persisted"}) is True

    session_impl = type(Session())

    def _raise_commit(self):
        _ = self
        raise SQLAlchemyError("forced commit failure")

    monkeypatch.setattr(session_impl, "commit", _raise_commit, raising=True)

    assert Ligands.delete("persisted") is False
    assert Ligands.insert({"name": "new_entry"}) is False
    assert Ligands.update("persisted", {"name": "renamed"}) is False
    assert Ligands.insert_or_update({"name": "persisted"}) is False
