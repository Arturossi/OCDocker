#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.XGBoost.XGBoostOptimizer.
'''

# Imports
###############################################################################
import importlib.util
import numpy as np
import pytest
import sys
import types

from pathlib import Path

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


class _FakeTrial:
    def __init__(self):
        self.user_attrs = {}

    def suggest_int(self, _name, low, _high):
        return low

    def suggest_float(self, _name, low, _high):
        return low

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _FakeModel:
    def predict(self, x):
        if hasattr(x, "get"):
            x = x.get()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            return x_arr
        return x_arr[:, 0]


class _FakeGPUArray:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def get(self):
        return self._arr


# Functions
###############################################################################
## Private ##

def _ensure_package(name: str, path: Path) -> None:
    pkg = sys.modules.get(name, types.ModuleType(name))
    pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _load_xgb_optimizer_module(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "OCDocker"

    _ensure_package("OCDocker", pkg_root)
    _ensure_package("OCDocker.OCScore", pkg_root / "OCScore")
    _ensure_package("OCDocker.OCScore.XGBoost", pkg_root / "OCScore" / "XGBoost")
    _ensure_package("OCDocker.Toolbox", pkg_root / "Toolbox")

    import optuna

    integration_mod = types.ModuleType("optuna.integration")

    class _PruningCallback:
        def __init__(self, trial, monitor):
            self.trial = trial
            self.monitor = monitor

    integration_mod.XGBoostPruningCallback = _PruningCallback  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "optuna.integration", integration_mod)

    printing_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    printing_mod.printv = lambda *_a, **_k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", printing_mod)

    xgb_run_mod = types.ModuleType("OCDocker.OCScore.XGBoost.OCxgboost")
    xgb_run_mod.run_xgboost = lambda *_a, **_k: (_FakeModel(), 0.5)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.XGBoost.OCxgboost", xgb_run_mod)

    path = pkg_root / "OCScore" / "XGBoost" / "XGBoostOptimizer.py"
    spec = importlib.util.spec_from_file_location("ocscore_xgboost_optimizer_module", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


## Public ##

@pytest.fixture
def ocxgboptimizer(monkeypatch):
    return _load_xgb_optimizer_module(monkeypatch)


@pytest.mark.order(390)
def test_objective_without_validation_populates_trial_params(monkeypatch, ocxgboptimizer):
    calls = {}

    def _run_xgboost(*_args, **kwargs):
        calls["params"] = kwargs["params"]
        return _FakeModel(), 0.42

    monkeypatch.setattr(ocxgboptimizer.OCxgboost, "run_xgboost", _run_xgboost)

    optimizer = ocxgboptimizer.XGBoostOptimizer(
        X_train=np.array([[0.1, 0.2], [0.3, 0.4]]),
        y_train=np.array([0.0, 1.0]),
        X_test=np.array([[0.5, 0.6], [0.7, 0.8]]),
        y_test=np.array([0.0, 1.0]),
        params=None,
        use_gpu=False,
    )

    trial = _FakeTrial()
    metric = optimizer.objective(trial)

    assert metric == pytest.approx(0.42)
    assert calls["params"]["eval_metric"] == "auc"
    assert "callbacks" in calls["params"]
    assert calls["params"]["early_stopping_rounds"] == 20
    assert trial.user_attrs == {}


@pytest.mark.order(391)
def test_objective_with_validation_and_gpu_sets_auc_user_attr(monkeypatch, ocxgboptimizer):
    cupy_mod = types.ModuleType("cupy")
    cupy_mod.asarray = lambda arr: _FakeGPUArray(arr)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cupy", cupy_mod)

    calls = {}

    def _run_xgboost(*_args, **kwargs):
        calls["params"] = kwargs["params"]
        return _FakeModel(), 0.33

    monkeypatch.setattr(ocxgboptimizer.OCxgboost, "run_xgboost", _run_xgboost)

    optimizer = ocxgboptimizer.XGBoostOptimizer(
        X_train=np.array([[0.1], [0.9]]),
        y_train=np.array([0.0, 1.0]),
        X_test=np.array([[0.2], [0.8]]),
        y_test=np.array([0.0, 1.0]),
        X_validation=np.array([[0.2], [0.8]]),
        y_validation=np.array([0.0, 1.0]),
        use_gpu=True,
    )

    trial = _FakeTrial()
    metric = optimizer.objective(trial)

    assert metric == pytest.approx(0.33)
    assert optimizer.params["device"] == "cuda"
    assert calls["params"]["eval_metric"] == "rmse"
    assert "AUC" in trial.user_attrs
    assert 0.0 <= trial.user_attrs["AUC"] <= 1.0


@pytest.mark.order(392)
def test_optimize_creates_study_and_logs_when_verbose(monkeypatch, ocxgboptimizer):
    captured = {"create_kwargs": None, "optimize_kwargs": None, "prints": []}

    class _FakeStudy:
        def __init__(self):
            self.best_params = {"max_depth": 5}
            self.best_value = 0.12

        def optimize(self, objective, n_trials, n_jobs):
            captured["optimize_kwargs"] = (objective, n_trials, n_jobs)

    fake_study = _FakeStudy()

    def _create_study(**kwargs):
        captured["create_kwargs"] = kwargs
        return fake_study

    monkeypatch.setattr(ocxgboptimizer.optuna, "create_study", _create_study)
    monkeypatch.setattr(ocxgboptimizer.ocprint, "printv", lambda msg: captured["prints"].append(msg))

    optimizer = ocxgboptimizer.XGBoostOptimizer(
        X_train=np.array([[0.1], [0.9]]),
        y_train=np.array([0.0, 1.0]),
        X_test=np.array([[0.2], [0.8]]),
        y_test=np.array([0.0, 1.0]),
        storage="sqlite:///xgb_test.db",
        verbose=True,
    )

    study = optimizer.optimize(
        direction="maximize",
        n_trials=3,
        n_jobs=2,
        study_name="xgb-unit-study",
        load_if_exists=False,
    )

    assert study is fake_study
    assert captured["create_kwargs"]["direction"] == "maximize"
    assert captured["create_kwargs"]["study_name"] == "xgb-unit-study"
    assert captured["create_kwargs"]["storage"] == "sqlite:///xgb_test.db"
    assert captured["create_kwargs"]["load_if_exists"] is False
    assert captured["optimize_kwargs"][1:] == (3, 2)
    assert len(captured["prints"]) == 2
