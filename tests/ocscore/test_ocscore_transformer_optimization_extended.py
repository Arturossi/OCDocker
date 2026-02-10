#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Optimization.Transformer runtime branches.
'''

# Imports
###############################################################################
import importlib.util
import numpy as np
import sys
import types

import pytest

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


# Functions
###############################################################################
## Private ##

def _ensure_package(name: str, path: Path) -> None:
    pkg = sys.modules.get(name, types.ModuleType(name))
    pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _minimal_trans_data() -> dict:
    return {
        "study_name": "unit_transformer_study",
        "X_train": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "X_test": np.array([[0.5, 0.6]], dtype=float),
        "X_val": np.array([[0.7, 0.8]], dtype=float),
        "y_train": np.array([0.0, 1.0], dtype=float),
        "y_test": np.array([0.5], dtype=float),
        "y_val": np.array([0.25], dtype=float),
    }


## Public ##

@pytest.fixture()
def octrans(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    pkg_root = root / "OCDocker"

    _ensure_package("OCDocker", pkg_root)
    _ensure_package("OCDocker.OCScore", pkg_root / "OCScore")
    _ensure_package("OCDocker.OCScore.Optimization", pkg_root / "OCScore" / "Optimization")
    _ensure_package("OCDocker.OCScore.Utils", pkg_root / "OCScore" / "Utils")
    _ensure_package("OCDocker.Toolbox", pkg_root / "Toolbox")

    error_mod = types.ModuleType("OCDocker.Error")

    class _Error:
        @staticmethod
        def value_error(_msg):
            return None

    error_mod.Error = _Error  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Error", error_mod)

    data_mod = types.ModuleType("OCDocker.OCScore.Utils.Data")
    data_mod.load_data = lambda **kwargs: _minimal_trans_data()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Utils.Data", data_mod)

    io_mod = types.ModuleType("OCDocker.OCScore.Utils.IO")
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Utils.IO", io_mod)

    workers_mod = types.ModuleType("OCDocker.OCScore.Utils.Workers")
    workers_mod.Transworker = lambda *_a, **_k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Utils.Workers", workers_mod)

    print_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    print_mod.print_warning = lambda _msg: None  # type: ignore[attr-defined]
    print_mod.printv = lambda _msg: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", print_mod)

    return _load_module(
        "OCDocker.OCScore.Optimization.Transformer",
        pkg_root / "OCScore" / "Optimization" / "Transformer.py",
    )


@pytest.mark.order(438)
def test_optimize_transformer_joblib_and_load_data(monkeypatch, octrans):
    calls = {"load_data": 0, "workers": [], "warnings": [], "verbose": []}

    def _load_data(**kwargs):
        _ = kwargs
        calls["load_data"] += 1
        return _minimal_trans_data()

    def _trans_worker(*args):
        calls["workers"].append(args)
        return "ok"

    def _fake_delayed(fn):
        def _builder(*args, **kwargs):
            _ = kwargs
            return lambda: fn(*args)
        return _builder

    class _FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, jobs):
            return [job() for job in jobs]

    monkeypatch.setattr(octrans.ocscoredata, "load_data", _load_data)
    monkeypatch.setattr(octrans.ocscoreworkers, "Transworker", _trans_worker)
    monkeypatch.setattr(octrans, "delayed", _fake_delayed)
    monkeypatch.setattr(octrans, "Parallel", _FakeParallel)
    monkeypatch.setattr(octrans.ocprint, "print_warning", lambda msg: calls["warnings"].append(msg))
    monkeypatch.setattr(octrans.ocprint, "printv", lambda msg: calls["verbose"].append(msg))

    out = octrans.optimize_Transformer(
        df_path="dummy.csv.gz",
        storage_id=8,
        base_models_folder=".",
        data=None,
        run_Trans_optimization=True,
        num_processes_Trans=2,
        total_trials_Trans=5,
        parallel_backend="joblib",
        use_gpu=False,
        verbose=True,
    )

    assert out is None
    assert calls["load_data"] == 1
    assert len(calls["workers"]) == 2
    assert calls["workers"][0][0] == 0
    assert calls["workers"][0][14] == 2
    assert calls["warnings"]
    assert calls["verbose"]


@pytest.mark.order(439)
def test_optimize_transformer_multiprocessing_and_skip(monkeypatch, octrans):
    calls = {"workers": []}

    def _trans_worker(*args):
        calls["workers"].append(args)
        return "ok"

    class _FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def starmap(self, fn, args_list):
            return [fn(*args) for args in args_list]

    monkeypatch.setattr(octrans.ocscoreworkers, "Transworker", _trans_worker)
    monkeypatch.setattr(octrans, "Pool", _FakePool)
    monkeypatch.setattr(octrans.ocprint, "print_warning", lambda *_a, **_k: None)
    monkeypatch.setattr(octrans.ocprint, "printv", lambda *_a, **_k: None)

    out_mp = octrans.optimize_Transformer(
        df_path="unused.csv",
        storage_id=9,
        base_models_folder=".",
        data=_minimal_trans_data(),
        run_Trans_optimization=True,
        num_processes_Trans=1,
        total_trials_Trans=1,
        parallel_backend="multiprocessing",
        use_gpu=False,
        verbose=False,
    )
    assert out_mp is None
    assert len(calls["workers"]) == 1

    out_skip = octrans.optimize_Transformer(
        df_path="unused.csv",
        storage_id=10,
        base_models_folder=".",
        data=_minimal_trans_data(),
        run_Trans_optimization=False,
        parallel_backend="joblib",
    )
    assert out_skip is None
    assert octrans.optimize is octrans.optimize_Transformer
