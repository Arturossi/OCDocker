#!/usr/bin/env python3

# Description
###############################################################################
'''
Additional branch coverage tests for OCScore.Utils.Workers.
'''

# Imports
###############################################################################
import importlib.util as util
import numpy as np
import sys
import types

from pathlib import Path

import pytest

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


class _StubOptimizer:
    def __init__(self, *args, **kwargs):
        _ = (args, kwargs)

    def optimize(self, **kwargs):
        _ = kwargs
        return "study"

    def ablate(self, **kwargs):
        _ = kwargs
        return None


# Functions
###############################################################################
## Private ##

def _install_workers_stubs(monkeypatch):
    optuna_mod = types.ModuleType("optuna")
    optuna_mod.study = types.SimpleNamespace(Study=object)
    samplers_mod = types.ModuleType("optuna.samplers")

    class _TPESampler:
        pass

    samplers_mod.TPESampler = _TPESampler  # type: ignore[attr-defined]
    optuna_mod.samplers = samplers_mod  # type: ignore[attr-defined]

    printing_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    printing_mod.printv = lambda *_a, **_k: None  # type: ignore[attr-defined]

    ae_mod = types.ModuleType("OCDocker.OCScore.Dimensionality.AutoencoderOptimizer")
    ae_mod.AutoencoderOptimizer = _StubOptimizer  # type: ignore[attr-defined]

    ga_mod = types.ModuleType("OCDocker.OCScore.Dimensionality.GeneticAlgorithm")
    ga_mod.GeneticAlgorithm = _StubOptimizer  # type: ignore[attr-defined]

    dnn_mod = types.ModuleType("OCDocker.OCScore.DNN.DNNOptimizer")
    dnn_mod.DNNOptimizer = _StubOptimizer  # type: ignore[attr-defined]

    trans_mod = types.ModuleType("OCDocker.OCScore.Transformer.TransOptimizer")
    trans_mod.TransOptimizer = _StubOptimizer  # type: ignore[attr-defined]

    xgb_mod = types.ModuleType("OCDocker.OCScore.XGBoost.XGBoostOptimizer")
    xgb_mod.XGBoostOptimizer = _StubOptimizer  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "optuna", optuna_mod)
    monkeypatch.setitem(sys.modules, "optuna.samplers", samplers_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", printing_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Dimensionality.AutoencoderOptimizer", ae_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Dimensionality.GeneticAlgorithm", ga_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.DNN.DNNOptimizer", dnn_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Transformer.TransOptimizer", trans_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.XGBoost.XGBoostOptimizer", xgb_mod)


def _import_workers(monkeypatch):
    _install_workers_stubs(monkeypatch)
    path = Path(__file__).resolve().parents[2] / "OCDocker" / "OCScore" / "Utils" / "Workers.py"
    spec = util.spec_from_file_location("ocscore_workers_branch_module", path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


## Public ##

@pytest.fixture
def ocworkers(monkeypatch):
    return _import_workers(monkeypatch)


@pytest.mark.order(341)
def test_ae_worker_covers_non_verbose_branches(monkeypatch, ocworkers):
    print_calls = {"count": 0}

    class _AE:
        def __init__(self, *_a, **_k):
            pass

        def optimize(self, **_k):
            return "ae-study"

    monkeypatch.setattr(ocworkers, "AutoencoderOptimizer", _AE)
    monkeypatch.setattr(ocworkers.ocprint, "printv", lambda *_a, **_k: print_calls.__setitem__("count", print_calls["count"] + 1))
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: None)

    X = np.zeros((3, 2))
    study = ocworkers.AEworker(1, 2, X, X, X, (2,), "sqlite:///ae.db", "/tmp/models", verbose=False, n_trials=1)
    assert study == "ae-study"
    assert print_calls["count"] == 0


@pytest.mark.order(342)
def test_ga_worker_non_none_best_params_and_non_verbose(monkeypatch, ocworkers):
    calls = {"params": None, "prints": 0}

    class _GA:
        def __init__(self, *_a, **kwargs):
            calls["params"] = kwargs["xgboost_params"]

        def optimize(self, **_k):
            return "study", {"f1": 1}, 0.1

    monkeypatch.setattr(ocworkers, "GeneticAlgorithm", _GA)
    monkeypatch.setattr(ocworkers.ocprint, "printv", lambda *_a, **_k: calls.__setitem__("prints", calls["prints"] + 1))
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: None)

    X = np.zeros((3, 2))
    y = np.zeros(3)
    out = ocworkers.GAWorker(
        pid=2,
        id=8,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        best_params={"eta": 0.2},
        verbose=False,
    )
    assert out[0] == "study"
    assert calls["params"] == {"eta": 0.2}
    assert calls["prints"] == 0


@pytest.mark.order(343)
def test_nn_ablation_worker_list_and_array_branches_with_verbose(monkeypatch, ocworkers):
    calls = {"ablate": 0, "printv": 0}

    class _DNN:
        def __init__(self, *_a, **_k):
            pass

        def ablate(self, **_k):
            calls["ablate"] = calls["ablate"] + 1

    monkeypatch.setattr(ocworkers, "DNNOptimizer", _DNN)
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(ocworkers.ocprint, "printv", lambda *_a, **_k: calls.__setitem__("printv", calls["printv"] + 1))

    X = np.zeros((2, 2))
    y = np.zeros(2)
    masks = [np.array([1, 0]), np.array([0, 1])]
    out1 = ocworkers.NNAblationworker(1, 1, X, y, X, y, X, y, masks, "sqlite:///x.db", {"layers": [4]}, verbose=True)
    out2 = ocworkers.NNAblationworker(1, 2, X, y, X, y, X, y, np.array([1, 1]), "sqlite:///x.db", {"layers": [4]}, verbose=True)

    assert out1 is None
    assert out2 is None
    assert calls["ablate"] == 3
    assert calls["printv"] == 4


@pytest.mark.order(344)
def test_nn_seed_ablation_worker_verbose_and_invalid_seed_type(monkeypatch, ocworkers):
    calls = {"ablate": 0, "printv": 0}

    class _DNN:
        def __init__(self, *_a, **_k):
            pass

        def ablate(self, **_k):
            calls["ablate"] = calls["ablate"] + 1

    monkeypatch.setattr(ocworkers, "DNNOptimizer", _DNN)
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(ocworkers.ocprint, "printv", lambda *_a, **_k: calls.__setitem__("printv", calls["printv"] + 1))

    X = np.zeros((2, 2))
    y = np.zeros(2)
    mask = np.array([1, 0])
    params = {"layers": [8, 4]}

    out1 = ocworkers.NNSeedAblationworker(1, 1, X, y, X, y, X, y, mask, "sqlite:///x.db", params, [1, 2], verbose=True)
    out2 = ocworkers.NNSeedAblationworker(1, 2, X, y, X, y, X, y, mask, "sqlite:///x.db", params, 3, verbose=True)
    assert out1 is None
    assert out2 is None
    assert calls["ablate"] == 3
    assert calls["printv"] == 4

    with pytest.raises(ValueError, match="Seeds must be a list of ints or an int"):
        ocworkers.NNSeedAblationworker(1, 3, X, y, X, y, X, y, mask, "sqlite:///x.db", params, "bad-seed")  # type: ignore[arg-type]


@pytest.mark.order(345)
def test_nn_worker_runs_optimize_and_covers_verbose_true_and_false(monkeypatch, ocworkers):
    calls = {"optimize": 0, "printv": 0}

    class _DNN:
        def __init__(self, *_a, **_k):
            pass

        def optimize(self, **kwargs):
            _ = kwargs
            calls["optimize"] = calls["optimize"] + 1

    monkeypatch.setattr(ocworkers, "DNNOptimizer", _DNN)
    monkeypatch.setattr(ocworkers.ocprint, "printv", lambda *_a, **_k: calls.__setitem__("printv", calls["printv"] + 1))

    X = np.zeros((2, 2))
    y = np.zeros(2)

    out1 = ocworkers.NNworker(1, 1, X, y, X, y, X, y, "sqlite:///x.db", verbose=True, n_trials=1)
    out2 = ocworkers.NNworker(1, 2, X, y, X, y, X, y, "sqlite:///x.db", verbose=False, n_trials=1)

    assert out1 is None
    assert out2 is None
    assert calls["optimize"] == 2
    assert calls["printv"] == 3


@pytest.mark.order(346)
def test_trans_worker_non_verbose_path(monkeypatch, ocworkers):
    calls = {"optimize": 0, "printv": 0}

    class _Trans:
        def __init__(self, *_a, **_k):
            pass

        def optimize(self, **kwargs):
            _ = kwargs
            calls["optimize"] = calls["optimize"] + 1

    monkeypatch.setattr(ocworkers, "TransOptimizer", _Trans)
    monkeypatch.setattr(ocworkers.ocprint, "printv", lambda *_a, **_k: calls.__setitem__("printv", calls["printv"] + 1))

    X = np.zeros((2, 2))
    y = np.zeros(2)
    out = ocworkers.Transworker(1, 9, X, y, X, y, X, y, "sqlite:///tr.db", verbose=False, n_trials=1)
    assert out is None
    assert calls["optimize"] == 1
    assert calls["printv"] == 0


@pytest.mark.order(347)
def test_xgb_worker_verbose_prints_and_returns_study(monkeypatch, ocworkers):
    calls = {"printv": 0}

    class _XGB:
        def __init__(self, *_a, **_k):
            pass

        def optimize(self, **_k):
            return "xgb-study"

    monkeypatch.setattr(ocworkers, "XGBoostOptimizer", _XGB)
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(ocworkers.ocprint, "printv", lambda *_a, **_k: calls.__setitem__("printv", calls["printv"] + 1))

    X = np.zeros((3, 2))
    y = np.zeros(3)
    study = ocworkers.XGBworker(1, 4, X, X, X, y, y, y, "sqlite:///xgb.db", verbose=True, n_trials=1)

    assert study == "xgb-study"
    assert calls["printv"] == 2


@pytest.mark.order(348)
def test_ae_worker_retries_transient_optuna_storage_errors(monkeypatch, ocworkers):
    calls = {"optimize": 0, "sleep": 0}

    class _AE:
        def __init__(self, *_a, **_k):
            pass

        def optimize(self, **_k):
            calls["optimize"] = calls["optimize"] + 1
            if calls["optimize"] == 1:
                raise RuntimeError(
                    "sqlite3.OperationalError: table alembic_version already exists"
                )
            return "ae-study"

    monkeypatch.setattr(ocworkers, "AutoencoderOptimizer", _AE)
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: calls.__setitem__("sleep", calls["sleep"] + 1))
    monkeypatch.setattr(ocworkers.ocprint, "print_warning", lambda *_a, **_k: None, raising=False)

    X = np.zeros((3, 2))
    study = ocworkers.AEworker(1, 2, X, X, X, (2,), "sqlite:///ae.db", "/tmp/models", verbose=True, n_trials=1)
    assert study == "ae-study"
    assert calls["optimize"] == 2
    assert calls["sleep"] == 1


@pytest.mark.order(349)
def test_transient_storage_error_detector_covers_mysql_and_postgres(ocworkers):
    transient_messages = [
        'psycopg2.errors.DeadlockDetected: deadlock detected',
        'psycopg2.errors.SerializationFailure: could not serialize access due to concurrent update',
        'psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint "alembic_version_pkc"',
        'psycopg2.errors.DuplicateTable: relation "study_system_attributes" already exists',
        "(pymysql.err.OperationalError) (1213, 'Deadlock found when trying to get lock; try restarting transaction')",
        "(pymysql.err.OperationalError) (1205, 'Lock wait timeout exceeded; try restarting transaction')",
        "(pymysql.err.InternalError) (1050, \"Table 'study_directions' already exists\")",
        "(pymysql.err.IntegrityError) (1062, \"Duplicate entry 'v3.2.0.a' for key 'alembic_version.PRIMARY'\")",
    ]

    for msg in transient_messages:
        assert ocworkers._is_transient_optuna_storage_error(RuntimeError(msg))

    assert not ocworkers._is_transient_optuna_storage_error(RuntimeError("model architecture mismatch"))
