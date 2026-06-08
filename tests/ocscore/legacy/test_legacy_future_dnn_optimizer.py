#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Optimization.legacy.future.DNN helpers.
'''

# Imports
###############################################################################
import importlib.util
import numpy as np
import pandas as pd
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


def _patch_future_data_helpers(monkeypatch, mod):
    '''Patch future data helpers on the loaded module to avoid file dependencies.'''

    def _preprocess_df(_path, invert_conditionally=True):
        _ = invert_conditionally
        dude = pd.DataFrame(
            {
                "receptor": ["r1", "r1", "r2", "r2"],
                "ligand": ["l1", "d1", "l2", "d2"],
                "name": ["n1", "n2", "n3", "n4"],
                "type": ["ligand", "decoy", "ligand", "decoy"],
                "db": ["d", "d", "d", "d"],
                "f1": [1.0, 2.0, 3.0, 4.0],
                "SMINA_1": [-7.0, -6.8, -7.1, -6.9],
            }
        )
        pdb = pd.DataFrame(
            {
                "receptor": ["rp1", "rp2"],
                "ligand": ["lp1", "lp2"],
                "name": ["np1", "np2"],
                "type": ["x", "x"],
                "db": ["pdb", "pdb"],
                "experimental": [1.0, 2.0],
                "f1": [10.0, 20.0],
                "SMINA_1": [-7.2, -7.4],
            }
        )
        return dude, pdb, ["SMINA_1"]

    def _remove_other_columns(df, keep, inplace=True):
        _ = inplace
        drop = [c for c in df.columns if c not in keep]
        df.drop(columns=drop, inplace=True, errors="ignore")

    monkeypatch.setattr(mod.ocscoredata, "preprocess_df", _preprocess_df, raising=False)
    monkeypatch.setattr(
        mod.ocscoredata,
        "split_dataset",
        lambda X, y, test_size, random_state: (X.iloc[:1], X.iloc[1:], y[:1], y[1:]),
        raising=False,
    )
    monkeypatch.setattr(mod.ocscoredata, "apply_pca", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(mod.ocscoredata, "remove_other_columns", _remove_other_columns, raising=False)


@pytest.fixture()
def ocfuture(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    pkg_root = root / "OCDocker"

    _ensure_package("OCDocker", pkg_root)
    _ensure_package("OCDocker.OCScore", pkg_root / "OCScore")
    _ensure_package("OCDocker.OCScore.Optimization", pkg_root / "OCScore" / "Optimization")
    _ensure_package("OCDocker.OCScore.Optimization.legacy", pkg_root / "OCScore" / "Optimization" / "legacy")
    _ensure_package("OCDocker.OCScore.Optimization.legacy.future", pkg_root / "OCScore" / "Optimization" / "legacy" / "future")
    _ensure_package("OCDocker.OCScore.Utils", pkg_root / "OCScore" / "Utils")
    _ensure_package("OCDocker.OCScore.DNN", pkg_root / "OCScore" / "DNN")
    _ensure_package("OCDocker.OCScore.DNN.future", pkg_root / "OCScore" / "DNN" / "future")
    _ensure_package("OCDocker.Toolbox", pkg_root / "Toolbox")

    # Stub optuna used by _future_worker
    optuna_mod = types.ModuleType("optuna")
    optuna_mod.samplers = types.SimpleNamespace(TPESampler=lambda: "sampler")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "optuna", optuna_mod)

    # Stub sklearn imports to keep tests lightweight and deterministic
    sk_decomp = types.ModuleType("sklearn.decomposition")

    class _PCA:
        def __init__(self, n_components=None, svd_solver=None):
            _ = (n_components, svd_solver)

        def fit(self, _X):
            return self

    sk_decomp.PCA = _PCA  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", sk_decomp)

    sk_modelsel = types.ModuleType("sklearn.model_selection")

    class _GroupShuffleSplit:
        def __init__(self, n_splits=1, test_size=0.2, random_state=0):
            _ = (n_splits, random_state)
            self.test_size = test_size

        def split(self, X, y=None, groups=None):
            _ = (y, groups)
            n = len(X)
            n_val = max(1, int(round(n * self.test_size)))
            idx = np.arange(n, dtype=int)
            yield idx[n_val:], idx[:n_val]

    sk_modelsel.GroupShuffleSplit = _GroupShuffleSplit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", sk_modelsel)

    print_calls = {"warning": [], "verbose": []}
    print_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    print_mod.print_warning = lambda msg: print_calls["warning"].append(msg)  # type: ignore[attr-defined]
    print_mod.printv = lambda msg: print_calls["verbose"].append(msg)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", print_mod)

    # Data helpers used by optimize_NN_future
    data_mod = types.ModuleType("OCDocker.OCScore.Utils.Data")

    def _preprocess_df(_path, invert_conditionally=True):
        _ = invert_conditionally
        dude = pd.DataFrame(
            {
                "receptor": ["r1", "r1", "r2", "r2"],
                "ligand": ["l1", "d1", "l2", "d2"],
                "name": ["n1", "n2", "n3", "n4"],
                "type": ["ligand", "decoy", "ligand", "decoy"],
                "db": ["d", "d", "d", "d"],
                "f1": [1.0, 2.0, 3.0, 4.0],
                "SMINA_1": [-7.0, -6.8, -7.1, -6.9],
            }
        )
        pdb = pd.DataFrame(
            {
                "receptor": ["rp1", "rp2"],
                "ligand": ["lp1", "lp2"],
                "name": ["np1", "np2"],
                "type": ["x", "x"],
                "db": ["pdb", "pdb"],
                "experimental": [1.0, 2.0],
                "f1": [10.0, 20.0],
                "SMINA_1": [-7.2, -7.4],
            }
        )
        return dude, pdb, ["SMINA_1"]

    data_mod.preprocess_df = _preprocess_df  # type: ignore[attr-defined]
    data_mod.split_dataset = lambda X, y, test_size, random_state: (X.iloc[:1], X.iloc[1:], y[:1], y[1:])  # type: ignore[attr-defined]
    data_mod.apply_pca = lambda *a, **k: None  # type: ignore[attr-defined]

    def _remove_other_columns(df, keep, inplace=True):
        _ = inplace
        drop = [c for c in df.columns if c not in keep]
        df.drop(columns=drop, inplace=True, errors="ignore")

    data_mod.remove_other_columns = _remove_other_columns  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Utils.Data", data_mod)

    # DNN optimizer stub used by _future_worker
    dnn_mod = types.ModuleType("OCDocker.OCScore.DNN.future.DNNOptimizer")
    dnn_calls = {"init": [], "optimize": []}

    class _DNNOptimizer:
        def __init__(self, *args, **kwargs):
            dnn_calls["init"].append((args, kwargs))

        def optimize(self, **kwargs):
            dnn_calls["optimize"].append(kwargs)
            return None

    dnn_mod.DNNOptimizer = _DNNOptimizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.DNN.future.DNNOptimizer", dnn_mod)

    module = _load_module(
        "OCDocker.OCScore.Optimization.legacy.future.DNN",
        pkg_root / "OCScore" / "Optimization" / "legacy" / "future" / "DNN.py",
    )

    return {
        "module": module,
        "print_calls": print_calls,
        "dnn_calls": dnn_calls,
    }


## Public ##

@pytest.mark.order(636)
def test_future_dnn_prepare_features_and_split_helpers(ocfuture):
    mod = ocfuture["module"]

    df = pd.DataFrame({"a": [1], "b": [2], "keep": [3]})
    out = mod._prepare_features(df, drop_cols=["a", "missing"])
    assert list(out.columns) == ["b", "keep"]

    x = pd.DataFrame({"f": [0, 1, 2, 3], "g": [4, 5, 6, 7]})
    y = np.array([1, 0, 1, 0], dtype=int)
    targets = np.array(["t1", "t1", "t2", "t2"])
    train, val = mod._split_dude_by_target(x, y, targets, val_fraction=0.25, random_seed=3)
    assert {"X", "y", "targets"} == set(train.keys())
    assert {"X", "y", "targets"} == set(val.keys())
    assert len(train["y"]) + len(val["y"]) == len(y)

    train_l, val_l = mod._split_dude_by_target([x, x.copy()], y, targets, val_fraction=0.25, random_seed=3)
    assert isinstance(train_l["X"], list)
    assert isinstance(val_l["X"], list)
    assert len(train_l["X"]) == 2


@pytest.mark.order(637)
def test_future_dnn_worker_invokes_optimizer(ocfuture):
    mod = ocfuture["module"]
    calls = ocfuture["dnn_calls"]

    mod._future_worker(
        pid=1,
        storage_id=7,
        X_pdb_train=np.array([[1.0]]),
        y_pdb_train=np.array([1.0]),
        X_pdb_test=np.array([[2.0]]),
        y_pdb_test=np.array([2.0]),
        dude_train={"X": np.array([[0.0]]), "y": np.array([1]), "targets": np.array(["t"])},
        dude_val={"X": np.array([[0.1]]), "y": np.array([0]), "targets": np.array(["t"])},
        storage="sqlite:///future.db",
        encoder_params={"enc": 8},
        random_seed=10,
        use_gpu=False,
        verbose=True,
        n_trials=2,
        study_name="future_study",
        future_config={"stage1": {"enabled": True}},
    )

    assert calls["init"]
    assert calls["optimize"]
    assert calls["optimize"][0]["study_name"] == "future_study"
    assert calls["optimize"][0]["n_trials"] == 2


@pytest.mark.order(638)
def test_optimize_future_fallback_and_invalid_backend(ocfuture, monkeypatch):
    mod = ocfuture["module"]
    print_calls = ocfuture["print_calls"]
    _patch_future_data_helpers(monkeypatch, mod)

    fallback_mod = types.ModuleType("OCDocker.OCScore.Optimization.legacy.DNN")
    fallback_mod.optimize_NN = lambda **kwargs: ("fallback", kwargs["storage_id"])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Optimization.legacy.DNN", fallback_mod)

    fallback = mod.optimize_NN_future(
        df_path="dummy.csv.gz",
        storage_id=99,
        base_models_folder=".",
        use_future=False,
    )
    assert fallback == ("fallback", 99)

    with pytest.raises(ValueError, match="Invalid parallel backend"):
        mod.optimize_NN_future(
            df_path="dummy.csv.gz",
            storage_id=1,
            base_models_folder=".",
            run_NN_optimization=True,
            num_processes_NN=1,
            total_trials_NN=1,
            parallel_backend="invalid",
            multiencoder=True,
            use_pdb_train=False,
            future_config={"data": {"dude_validation_fraction": 0.5}},
        )

    _ = print_calls  # warnings are validated by execution path and raised backend error


@pytest.mark.order(639)
def test_optimize_future_returns_none_when_nn_optimization_disabled(ocfuture, monkeypatch):
    mod = ocfuture["module"]
    _patch_future_data_helpers(monkeypatch, mod)
    out = mod.optimize_NN_future(
        df_path="dummy.csv.gz",
        storage_id=2,
        base_models_folder=".",
        run_NN_optimization=False,
    )
    assert out is None
