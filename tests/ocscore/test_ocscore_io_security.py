#!/usr/bin/env python3

# Description
###############################################################################
'''
Security tests for OCScore serialized-object loading.
'''

# Imports
###############################################################################
import joblib
import pickle

import pytest

import OCDocker.OCScore.Utils.IO as ocscoreio
import OCDocker.Toolbox.Security as ocsec

# License
###############################################################################
'''OCDocker
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

    # Prevent env leakage to subsequent tests
    monkeypatch.delenv("OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION", raising=False)
    monkeypatch.delenv("OCDOCKER_ALLOW_SCRIPT_EXEC", raising=False)


@pytest.mark.order(82)
def test_load_object_joblib_with_trust(tmp_path, monkeypatch):
    file_path = tmp_path / "obj.joblib"
    payload = {"a": 1, "b": [1, 2, 3]}
    joblib.dump(payload, str(file_path))

    monkeypatch.delenv("OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION", raising=False)

    loaded = ocscoreio.load_object(str(file_path), serialization_method="joblib", trusted=True)
    assert loaded == payload


@pytest.mark.order(83)
def test_load_object_invalid_serialization_method_raises_value_error(tmp_path):
    file_path = tmp_path / "obj.any"
    file_path.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError):
        _ = ocscoreio.load_object(str(file_path), serialization_method="not-a-method", trusted=True)
