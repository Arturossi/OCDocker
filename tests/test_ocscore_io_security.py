#!/usr/bin/env python3

# Description
###############################################################################
'''
Security tests for OCScore serialized-object loading.
'''

# Imports
###############################################################################
import pickle

import pytest

import OCDocker.OCScore.Utils.IO as ocscoreio
import OCDocker.Toolbox.Security as ocsec

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

## Public ##

@pytest.mark.order(68)
def test_load_object_pickle_blocked_without_trust_opt_in(tmp_path, monkeypatch):
    file_path = tmp_path / "obj.pkl"
    payload = {"a": 1}
    with open(file_path, "wb") as f:
        pickle.dump(payload, f)

    monkeypatch.delenv("OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION", raising=False)

    with pytest.raises(PermissionError):
        _ = ocscoreio.load_object(str(file_path), serialization_method="pickle")


@pytest.mark.order(69)
def test_load_object_pickle_allows_explicit_trust(tmp_path, monkeypatch):
    file_path = tmp_path / "obj.pkl"
    payload = {"a": 1}
    with open(file_path, "wb") as f:
        pickle.dump(payload, f)

    monkeypatch.delenv("OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION", raising=False)

    loaded = ocscoreio.load_object(str(file_path), serialization_method="pickle", trusted=True)
    assert loaded == payload


@pytest.mark.order(70)
def test_load_object_pickle_allows_env_opt_in(tmp_path, monkeypatch):
    file_path = tmp_path / "obj.pkl"
    payload = {"a": 1}
    with open(file_path, "wb") as f:
        pickle.dump(payload, f)

    monkeypatch.setenv("OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION", "1")

    loaded = ocscoreio.load_object(str(file_path), serialization_method="pickle")
    assert loaded == payload


@pytest.mark.order(71)
def test_allow_unsafe_runtime_helper_enables_deserialization(tmp_path, monkeypatch):
    file_path = tmp_path / "obj.pkl"
    payload = {"a": 1}
    with open(file_path, "wb") as f:
        pickle.dump(payload, f)

    monkeypatch.delenv("OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION", raising=False)
    monkeypatch.delenv("OCDOCKER_ALLOW_SCRIPT_EXEC", raising=False)

    ocsec.allow_unsafe_runtime(deserialization=True, script_exec=False)

    loaded = ocscoreio.load_object(str(file_path), serialization_method="pickle")
    assert loaded == payload
