#!/usr/bin/env python3

# Description
###############################################################################
'''
Smoke tests for OCScore.Utils.Workers with dependency stubs.
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
    spec = util.spec_from_file_location("ocscore_workers_stubbed_module", path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


## Public ##

@pytest.fixture
def ocworkers(monkeypatch):
    return _import_workers(monkeypatch)


@pytest.mark.order(178)
def test_ga_worker_uses_empty_best_params_when_none(monkeypatch, ocworkers):
    calls = {"params": None, "study_name": None}

    class _GA:
        def __init__(self, *_args, **kwargs):
            calls["params"] = kwargs["xgboost_params"]

        def optimize(self, **kwargs):
            calls["study_name"] = kwargs["study_name"]
            return "study", {"f1": 1}, 0.25

    monkeypatch.setattr(ocworkers, "GeneticAlgorithm", _GA)
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: None)

    X = np.zeros((3, 2))
    y = np.zeros(3)
    study, features, score = ocworkers.GAWorker(
        pid=1,
        id=7,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        storage="sqlite:///x.db",
        best_params=None,
        n_trials=2,
        verbose=True,
    )
    assert study == "study"
    assert features == {"f1": 1}
    assert score == 0.25
    assert calls["params"] == {}
    assert calls["study_name"] == "GA_Feature_Selection_7"


@pytest.mark.order(179)
def test_nn_ablation_worker_rejects_invalid_mask(ocworkers):
    X = np.zeros((2, 2))
    y = np.zeros(2)
    with pytest.raises(ValueError):
        _ = ocworkers.NNAblationworker(
            pid=1,
            id=1,
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            X_val=X,
            y_val=y,
            mask="invalid",  # type: ignore[arg-type]
            storage="sqlite:///x.db",
            network_params={},
        )


@pytest.mark.order(180)
def test_nn_seed_ablation_worker_handles_list_and_int_seeds(monkeypatch, ocworkers):
    created = {"count": 0}
    ablated = {"count": 0}

    class _DNN:
        def __init__(self, *_args, **_kwargs):
            created["count"] = created["count"] + 1

        def ablate(self, **_kwargs):
            ablated["count"] = ablated["count"] + 1

    monkeypatch.setattr(ocworkers, "DNNOptimizer", _DNN)
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: None)

    X = np.zeros((2, 2))
    y = np.zeros(2)
    mask = np.array([1, 0], dtype=int)
    params = {"layers": [16, 8]}

    _ = ocworkers.NNSeedAblationworker(1, 1, X, y, X, y, X, y, mask, "sqlite:///x.db", params, [11, 22], use_gpu=False)
    _ = ocworkers.NNSeedAblationworker(1, 2, X, y, X, y, X, y, mask, "sqlite:///x.db", params, 33, use_gpu=False)

    assert created["count"] == 3
    assert ablated["count"] == 3


@pytest.mark.order(181)
def test_xgb_worker_direction_switch_and_default_params(monkeypatch, ocworkers):
    calls = {"params": [], "direction": []}

    class _XGB:
        def __init__(self, *_args, **kwargs):
            calls["params"].append(kwargs["params"])

        def optimize(self, **kwargs):
            calls["direction"].append(kwargs["direction"])
            return "xgb-study"

    monkeypatch.setattr(ocworkers, "XGBoostOptimizer", _XGB)
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: None)

    X = np.zeros((3, 2))
    y = np.zeros(3)

    out1 = ocworkers.XGBworker(0, 1, X, X, None, y, y, y, "sqlite:///x.db", params=None)  # type: ignore[arg-type]
    out2 = ocworkers.XGBworker(0, 2, X, X, X, y, y, y, "sqlite:///x.db", params={"eta": 0.1})

    assert out1 == "xgb-study"
    assert out2 == "xgb-study"
    assert calls["params"][0] == {}
    assert calls["params"][1] == {"eta": 0.1}
    assert calls["direction"] == ["maximize", "minimize"]


@pytest.mark.order(182)
def test_ae_and_transformer_workers_smoke(monkeypatch, ocworkers):
    ae_calls = {"study_name": None}
    trans_calls = {"study_name": None}

    class _AE:
        def __init__(self, *_args, **_kwargs):
            pass

        def optimize(self, **kwargs):
            ae_calls["study_name"] = kwargs["study_name"]
            return "ae-study"

    class _Trans:
        def __init__(self, *_args, **_kwargs):
            pass

        def optimize(self, **kwargs):
            trans_calls["study_name"] = kwargs["study_name"]
            return None

    monkeypatch.setattr(ocworkers, "AutoencoderOptimizer", _AE)
    monkeypatch.setattr(ocworkers, "TransOptimizer", _Trans)
    monkeypatch.setattr(ocworkers.time, "sleep", lambda *_a, **_k: None)

    X = np.zeros((2, 2))
    y = np.zeros(2)

    study = ocworkers.AEworker(1, 5, X, X, X, (2,), "sqlite:///ae.db", "/tmp/models", verbose=True, n_trials=1)
    _ = ocworkers.Transworker(1, 9, X, y, X, y, X, y, "sqlite:///tr.db", verbose=True, n_trials=1)

    assert study == "ae-study"
    assert ae_calls["study_name"] == "Autoencoder_Optimization_5"
    assert trans_calls["study_name"] == "Trans_Optimization_9"
