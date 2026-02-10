#!/usr/bin/env python3

# Description
###############################################################################
'''
Runtime coverage tests for OCScore optimization wrappers and lightweight paths.
'''

# Imports
###############################################################################
import importlib
import importlib.util
import numpy as np
import pandas as pd
import pytest
import sys
import types

from pathlib import Path
from types import SimpleNamespace

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


class _FakeXGBRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls = []

    def fit(self, X_train, y_train, eval_set=None, verbose=False):
        self.fit_calls.append((np.asarray(X_train).shape, np.asarray(y_train).shape, eval_set, verbose))

    def evals_result(self):
        return {"validation_0": {"rmse": [1.0, 0.25]}}


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


def _minimal_data_dict() -> dict:
    x_train = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    x_test = pd.DataFrame({"f1": [0.5, 0.6], "f2": [0.7, 0.8]})
    x_val = pd.DataFrame({"f1": [0.9], "f2": [1.0]})
    y_train = np.array([0.0, 1.0], dtype=float)
    y_test = np.array([0.0, 1.0], dtype=float)
    y_val = np.array([0.5], dtype=float)
    return {
        "models_folder": ".",
        "study_name": "unit_study",
        "X_train": x_train,
        "X_test": x_test,
        "X_val": x_val,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
    }


@pytest.fixture()
def ocmods(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    pkg_root = root / "OCDocker"

    _ensure_package("OCDocker", pkg_root)
    _ensure_package("OCDocker.OCScore", pkg_root / "OCScore")
    _ensure_package("OCDocker.OCScore.Optimization", pkg_root / "OCScore" / "Optimization")
    _ensure_package("OCDocker.OCScore.Optimization.future", pkg_root / "OCScore" / "Optimization" / "future")
    _ensure_package("OCDocker.OCScore.Utils", pkg_root / "OCScore" / "Utils")
    _ensure_package("OCDocker.OCScore.XGBoost", pkg_root / "OCScore" / "XGBoost")
    _ensure_package("OCDocker.OCScore.Transformer", pkg_root / "OCScore" / "Transformer")
    _ensure_package("OCDocker.Toolbox", pkg_root / "Toolbox")

    error_mod = types.ModuleType("OCDocker.Error")

    class _Error:
        @staticmethod
        def value_error(_msg):
            return None

    error_mod.Error = _Error  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Error", error_mod)

    data_mod = types.ModuleType("OCDocker.OCScore.Utils.Data")
    data_mod.load_data = lambda **_k: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Utils.Data", data_mod)

    workers_mod = types.ModuleType("OCDocker.OCScore.Utils.Workers")
    workers_mod.XGBworker = lambda *_a, **_k: None  # type: ignore[attr-defined]
    workers_mod.GAWorker = lambda *_a, **_k: None  # type: ignore[attr-defined]
    workers_mod.AEworker = lambda *_a, **_k: None  # type: ignore[attr-defined]
    workers_mod.NNworker = lambda *_a, **_k: None  # type: ignore[attr-defined]
    workers_mod.Transworker = lambda *_a, **_k: None  # type: ignore[attr-defined]
    workers_mod.NNAblationworker = lambda *_a, **_k: None  # type: ignore[attr-defined]
    workers_mod.NNSeedAblationworker = lambda *_a, **_k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Utils.Workers", workers_mod)

    io_mod = types.ModuleType("OCDocker.OCScore.Utils.IO")
    io_mod.store_object = lambda *_a, **_k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Utils.IO", io_mod)

    print_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    print_mod.print_warning = lambda _msg: None  # type: ignore[attr-defined]
    print_mod.printv = lambda _msg: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", print_mod)

    xgb_mod = types.ModuleType("xgboost")
    xgb_mod.XGBRegressor = _FakeXGBRegressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xgboost", xgb_mod)

    modules = {
        "pkg_opt": _load_module("OCDocker.OCScore.Optimization", pkg_root / "OCScore" / "Optimization" / "__init__.py"),
        "pkg_opt_future": _load_module("OCDocker.OCScore.Optimization.future", pkg_root / "OCScore" / "Optimization" / "future" / "__init__.py"),
        "pkg_trans": _load_module("OCDocker.OCScore.Transformer", pkg_root / "OCScore" / "Transformer" / "__init__.py"),
        "pkg_xgb": _load_module("OCDocker.OCScore.XGBoost", pkg_root / "OCScore" / "XGBoost" / "__init__.py"),
        "dnn_opt": _load_module("OCDocker.OCScore.Optimization.DNN", pkg_root / "OCScore" / "Optimization" / "DNN.py"),
        "trans_opt": _load_module("OCDocker.OCScore.Optimization.Transformer", pkg_root / "OCScore" / "Optimization" / "Transformer.py"),
        "xgb_opt": _load_module("OCDocker.OCScore.Optimization.XGBoost", pkg_root / "OCScore" / "Optimization" / "XGBoost.py"),
        "xgb_run": _load_module("OCDocker.OCScore.XGBoost.OCxgboost", pkg_root / "OCScore" / "XGBoost" / "OCxgboost.py"),
    }
    return modules


## Public ##


@pytest.mark.order(357)
def test_ocscore_package_inits_expose_empty_all(ocmods):
    assert ocmods["pkg_opt"].__all__ == []
    assert ocmods["pkg_opt_future"].__all__ == []
    assert ocmods["pkg_trans"].__all__ == []
    assert ocmods["pkg_xgb"].__all__ == []


@pytest.mark.order(358)
def test_run_xgboost_trains_and_reads_metric(ocmods):
    ocxgbrun = ocmods["xgb_run"]
    x_train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    y_train = np.array([0.0, 1.0], dtype=float)
    x_test = np.array([[0.5, 0.5]], dtype=float)
    y_test = np.array([1.0], dtype=float)

    model, metric = ocxgbrun.run_xgboost(
        x_train,
        y_train,
        x_test,
        y_test,
        params={"eval_metric": "RMSE"},
        verbose=True,
    )

    assert isinstance(model, _FakeXGBRegressor)
    assert model.fit_calls
    assert metric == pytest.approx(0.25)


@pytest.mark.order(359)
def test_optimize_transformer_invalid_backend_raises(monkeypatch, ocmods):
    warnings = []
    octransopt = ocmods["trans_opt"]
    monkeypatch.setattr(octransopt.ocprint, "print_warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(octransopt.ocprint, "printv", lambda _msg: None)
    monkeypatch.setattr(octransopt.ocerror.Error, "value_error", lambda _msg: None)

    with pytest.raises(ValueError, match="Invalid parallel backend"):
        octransopt.optimize_Transformer(
            df_path="unused.csv",
            storage_id=7,
            base_models_folder=".",
            data=_minimal_data_dict(),
            run_Trans_optimization=True,
            num_processes_Trans=2,
            total_trials_Trans=3,
            parallel_backend="invalid",
            verbose=True,
        )

    assert any("not divisible" in msg for msg in warnings)


@pytest.mark.order(360)
def test_optimize_xgb_invalid_backend_raises(monkeypatch, ocmods):
    warnings = []
    ocxgbopt = ocmods["xgb_opt"]
    monkeypatch.setattr(ocxgbopt.ocprint, "print_warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(ocxgbopt.ocprint, "printv", lambda _msg: None)
    monkeypatch.setattr(ocxgbopt.ocerror.Error, "value_error", lambda _msg: None)

    with pytest.raises(ValueError, match="Invalid parallel backend"):
        ocxgbopt.optimize_XGB(
            df_path="unused.csv",
            storage_id=11,
            base_models_folder=".",
            data=_minimal_data_dict(),
            run_pre_XGB_optimization=True,
            total_trials_pre_XGB=3,
            num_processes_pre_XGB=2,
            run_GA_optimization=False,
            run_XGB_optimization=False,
            parallel_backend="invalid",
            verbose=True,
        )

    assert any("total_trials_pre_XGB is not divisible" in msg for msg in warnings)


@pytest.mark.order(361)
def test_optimize_xgb_skip_all_paths_returns_none(ocmods):
    ocxgbopt = ocmods["xgb_opt"]
    result = ocxgbopt.optimize_XGB(
        df_path="unused.csv",
        storage_id=12,
        base_models_folder=".",
        data=_minimal_data_dict(),
        run_pre_XGB_optimization=False,
        run_GA_optimization=False,
        run_XGB_optimization=False,
        parallel_backend="invalid",
    )
    assert result is None


@pytest.mark.order(362)
def test_optimize_nn_invalid_backend_raises(ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    with pytest.raises(ValueError, match="Invalid parallel backend"):
        ocdnnopt.optimize_NN(
            df_path="unused.csv",
            storage_id=13,
            base_models_folder=".",
            data=_minimal_data_dict(),
            autoencoder=False,
            run_NN_optimization=True,
            num_processes_NN=2,
            total_trials_NN=3,
            parallel_backend="invalid",
        )


@pytest.mark.order(363)
def test_perform_ablation_study_nn_invalid_backend_raises(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    monkeypatch.setattr(ocdnnopt.ocerror.Error, "value_error", lambda _msg: None)
    x_train = pd.DataFrame({"VINA_1": [0.2, 0.3], "f1": [1.0, 2.0]})
    y_train = pd.Series([0.0, 1.0])

    with pytest.raises(ValueError, match="Invalid parallel backend"):
        ocdnnopt.perform_ablation_study_NN(
            X_train=x_train,
            y_train=y_train,
            X_test=x_train,
            y_test=y_train,
            X_val=x_train,
            y_val=y_train,
            id=21,
            num_processes=2,
            encoder_params={},
            best_params={},
            random_seed=42,
            use_gpu=False,
            verbose=False,
            load_if_exists=False,
            study_name="ablation_test",
            storage="sqlite://",
            masks=[[1, 1]],
            parallel_backend="invalid",
        )


@pytest.mark.order(364)
def test_perform_seed_ablation_study_nn_invalid_backend_raises(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    monkeypatch.setattr(ocdnnopt.ocerror.Error, "value_error", lambda _msg: None)
    x_train = np.array([[0.2, 1.0], [0.3, 2.0]], dtype=float)
    y_train = np.array([0.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="Invalid parallel backend"):
        ocdnnopt.perform_seed_ablation_study_NN(
            X_train=x_train,
            y_train=y_train,
            X_test=x_train,
            y_test=y_train,
            X_val=x_train,
            y_val=y_train,
            id=22,
            num_processes=2,
            encoder_params={},
            best_params={},
            use_gpu=False,
            verbose=False,
            load_if_exists=False,
            study_name="seed_ablation_test",
            storage="sqlite://",
            mask=np.array([1, 1], dtype=int),
            seeds=[10],
            parallel_backend="invalid",
        )


@pytest.mark.order(384)
def test_optimize_xgb_joblib_full_pipeline_uses_loaded_data(monkeypatch, ocmods):
    ocxgbopt = ocmods["xgb_opt"]
    base_data = _minimal_data_dict()
    calls = {"load_data": 0, "xgb": [], "ga": []}

    def _load_data(**_kwargs):
        calls["load_data"] = calls["load_data"] + 1
        return base_data

    def _fake_delayed(fn):
        def _builder(*args, **kwargs):
            return lambda: fn(*args, **kwargs)
        return _builder

    class _FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, jobs):
            return [job() for job in jobs]

    class _FakeStudy:
        def __init__(self, kind):
            self.kind = kind
            trial = SimpleNamespace(params={"eta": 0.15})
            self.trials = {0: trial, 0.0: trial}

        def trials_dataframe(self):
            if self.kind == "pre":
                return pd.DataFrame(
                    [{"number": 0, "value": 0.25, "user_attrs_AUC": 0.80}]
                )
            return pd.DataFrame(
                [{"value": 0.20, "user_attrs_best_AUC": 0.90, "user_attrs_best_individual": [1, 0]}]
            )

    def _load_study(study_name, storage):
        _ = storage
        if study_name.startswith("Pre_XGB_Optimization_"):
            return _FakeStudy("pre")
        if study_name.startswith("feature_selection_"):
            return _FakeStudy("ga")
        raise AssertionError(f"Unexpected study name: {study_name}")

    def _xgb_worker(*args, **kwargs):
        calls["xgb"].append((args, kwargs))
        return "xgb-study"

    def _ga_worker(*args, **kwargs):
        calls["ga"].append((args, kwargs))
        return ("ga-study", {"f1": 1}, 0.1)

    monkeypatch.setattr(ocxgbopt.ocscoredata, "load_data", _load_data)
    monkeypatch.setattr(ocxgbopt, "delayed", _fake_delayed)
    monkeypatch.setattr(ocxgbopt, "Parallel", _FakeParallel)
    monkeypatch.setattr(ocxgbopt.optuna, "load_study", _load_study)
    monkeypatch.setattr(ocxgbopt.ocscoreworkers, "XGBworker", _xgb_worker)
    monkeypatch.setattr(ocxgbopt.ocscoreworkers, "GAWorker", _ga_worker)
    monkeypatch.setattr(ocxgbopt.ocprint, "printv", lambda *_a, **_k: None)
    monkeypatch.setattr(ocxgbopt.ocprint, "print_warning", lambda *_a, **_k: None)

    result = ocxgbopt.optimize_XGB(
        df_path="dummy.csv",
        storage_id=31,
        base_models_folder="/tmp/models",
        data=None,
        storage="sqlite:///xgb_runtime.db",
        run_pre_XGB_optimization=True,
        num_processes_pre_XGB=1,
        total_trials_pre_XGB=1,
        run_GA_optimization=True,
        num_processes_GA=1,
        total_trials_GA=1,
        run_XGB_optimization=True,
        num_processes_XGB=1,
        total_trials_XGB=1,
        parallel_backend="joblib",
        use_gpu=False,
        verbose=True,
    )

    assert result is None
    assert calls["load_data"] == 1
    assert len(calls["xgb"]) == 2
    assert len(calls["ga"]) == 1
    assert calls["ga"][0][0][9] == {"eta": 0.15}
    assert calls["xgb"][1][0][2].shape[1] == 1
    assert calls["xgb"][1][0][4].shape[1] == 1


@pytest.mark.order(385)
def test_optimize_xgb_multiprocessing_pipeline_with_none_validation(monkeypatch, ocmods):
    ocxgbopt = ocmods["xgb_opt"]
    data = _minimal_data_dict()
    data["X_val"] = None
    data["y_val"] = None
    calls = {"xgb": [], "ga": []}

    class _FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def starmap(self, fn, arg_list):
            out = []
            for args in arg_list:
                out.append(fn(*args))
            return out

    class _FakeStudy:
        def __init__(self, kind):
            self.kind = kind
            trial = SimpleNamespace(params={"eta": 0.09})
            self.trials = {0: trial, 0.0: trial}

        def trials_dataframe(self):
            if self.kind == "pre":
                return pd.DataFrame(
                    [{"number": 0, "value": 0.30, "user_attrs_AUC": 0.70}]
                )
            return pd.DataFrame(
                [{"value": 0.25, "user_attrs_best_AUC": 0.95, "user_attrs_best_individual": [0, 1]}]
            )

    def _load_study(study_name, storage):
        _ = storage
        if study_name.startswith("Pre_XGB_Optimization_"):
            return _FakeStudy("pre")
        if study_name.startswith("feature_selection_"):
            return _FakeStudy("ga")
        raise AssertionError(f"Unexpected study name: {study_name}")

    def _xgb_worker(*args, **kwargs):
        calls["xgb"].append((args, kwargs))
        return "xgb-study"

    def _ga_worker(*args, **kwargs):
        calls["ga"].append((args, kwargs))
        return ("ga-study", {"f2": 1}, 0.2)

    monkeypatch.setattr(ocxgbopt, "Pool", _FakePool)
    monkeypatch.setattr(ocxgbopt.optuna, "load_study", _load_study)
    monkeypatch.setattr(ocxgbopt.ocscoreworkers, "XGBworker", _xgb_worker)
    monkeypatch.setattr(ocxgbopt.ocscoreworkers, "GAWorker", _ga_worker)
    monkeypatch.setattr(ocxgbopt.ocprint, "printv", lambda *_a, **_k: None)
    monkeypatch.setattr(ocxgbopt.ocprint, "print_warning", lambda *_a, **_k: None)

    result = ocxgbopt.optimize_XGB(
        df_path="unused.csv",
        storage_id=32,
        base_models_folder="/tmp/models",
        data=data,
        storage="sqlite:///xgb_runtime_mp.db",
        run_pre_XGB_optimization=True,
        num_processes_pre_XGB=1,
        total_trials_pre_XGB=1,
        run_GA_optimization=True,
        num_processes_GA=1,
        total_trials_GA=1,
        run_XGB_optimization=True,
        num_processes_XGB=1,
        total_trials_XGB=1,
        parallel_backend="multiprocessing",
        use_gpu=False,
        verbose=True,
    )

    assert result is None
    assert len(calls["xgb"]) == 2
    assert len(calls["ga"]) == 1
    assert calls["xgb"][1][0][2].shape[1] == 1
    assert calls["xgb"][1][0][4] is None


@pytest.mark.order(386)
def test_run_xgboost_without_eval_metric_raises_key_error(ocmods):
    ocxgbrun = ocmods["xgb_run"]
    x_train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    y_train = np.array([0.0, 1.0], dtype=float)
    x_test = np.array([[0.5, 0.5]], dtype=float)
    y_test = np.array([1.0], dtype=float)

    with pytest.raises(KeyError, match="eval_metric"):
        ocxgbrun.run_xgboost(
            x_train,
            y_train,
            x_test,
            y_test,
            params=None,
            verbose=False,
        )


@pytest.mark.order(394)
def test_optimize_nn_joblib_without_autoencoder_runs_nn_worker(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    calls = {"load_data": 0, "nn": []}

    def _load_data(**_kwargs):
        calls["load_data"] = calls["load_data"] + 1
        return _minimal_data_dict()

    def _fake_delayed(fn):
        def _builder(*args, **kwargs):
            return lambda: fn(*args, **kwargs)
        return _builder

    class _FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, jobs):
            return [job() for job in jobs]

    def _nn_worker(*args, **kwargs):
        calls["nn"].append((args, kwargs))
        return None

    monkeypatch.setattr(ocdnnopt.ocscoredata, "load_data", _load_data)
    monkeypatch.setattr(ocdnnopt, "delayed", _fake_delayed)
    monkeypatch.setattr(ocdnnopt, "Parallel", _FakeParallel)
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "NNworker", _nn_worker)
    monkeypatch.setattr(ocdnnopt.ocprint, "printv", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdnnopt.ocprint, "print_warning", lambda *_a, **_k: None)

    result = ocdnnopt.optimize_NN(
        df_path="dummy.csv",
        storage_id=40,
        base_models_folder="/tmp/models",
        data=None,
        autoencoder=False,
        run_NN_optimization=True,
        num_processes_NN=1,
        total_trials_NN=1,
        parallel_backend="joblib",
        use_gpu=False,
        verbose=True,
    )

    assert result is None
    assert calls["load_data"] == 1
    assert len(calls["nn"]) == 1
    assert calls["nn"][0][0][9] is None


@pytest.mark.order(395)
def test_optimize_nn_joblib_autoencoder_single_encoder_uses_best_params(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    calls = {"ae": [], "nn": []}

    def _fake_delayed(fn):
        def _builder(*args, **kwargs):
            return lambda: fn(*args, **kwargs)
        return _builder

    class _FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, jobs):
            return [job() for job in jobs]

    class _FakeStudy:
        def __init__(self):
            trial = SimpleNamespace(params={"latent_dim": 8})
            self.trials = {0: trial, 0.0: trial}

        def trials_dataframe(self):
            return pd.DataFrame(
                [{"number": 0, "state": "COMPLETE", "value": 0.2, "user_attrs_val_rmse": 0.1}]
            )

    def _ae_worker(*args, **kwargs):
        calls["ae"].append((args, kwargs))
        return "ae-study"

    def _nn_worker(*args, **kwargs):
        calls["nn"].append((args, kwargs))
        return None

    monkeypatch.setattr(ocdnnopt, "delayed", _fake_delayed)
    monkeypatch.setattr(ocdnnopt, "Parallel", _FakeParallel)
    monkeypatch.setattr(ocdnnopt.optuna, "load_study", lambda *_a, **_k: _FakeStudy())
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "AEworker", _ae_worker)
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "NNworker", _nn_worker)
    monkeypatch.setattr(ocdnnopt.ocprint, "printv", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdnnopt.ocprint, "print_warning", lambda *_a, **_k: None)

    result = ocdnnopt.optimize_NN(
        df_path="unused.csv",
        storage_id=41,
        base_models_folder="/tmp/models",
        data=_minimal_data_dict(),
        autoencoder=True,
        multiencoder=False,
        run_autoencoder_optimization=True,
        num_processes_autoencoder=1,
        total_trials_autoencoder=1,
        run_NN_optimization=True,
        num_processes_NN=1,
        total_trials_NN=1,
        parallel_backend="joblib",
        use_gpu=False,
    )

    assert result is None
    assert len(calls["ae"]) == 1
    assert len(calls["nn"]) == 1
    assert calls["nn"][0][0][9] == {"latent_dim": 8}


@pytest.mark.order(396)
def test_optimize_nn_joblib_multiencoder_path_runs_ae_for_lig_and_rec(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    calls = {"ae": [], "nn": [], "studies": []}

    x_train = pd.DataFrame(
        {
            "VINA_1": [0.1, 0.2, 0.3],
            "SMINA_1": [0.2, 0.3, 0.4],
            "ODDT_1": [0.3, 0.4, 0.5],
            "PLANTS_1": [0.4, 0.5, 0.6],
            "f1": [1.0, 1.1, 1.2],
        }
    )
    x_test = x_train.copy()
    x_val = x_train.copy()
    data = {
        "models_folder": ".",
        "study_name": "nn_study",
        "X_train": x_train,
        "X_test": x_test,
        "X_val": x_val,
        "y_train": np.array([0.0, 1.0, 0.0], dtype=float),
        "y_test": np.array([0.0, 1.0, 0.0], dtype=float),
        "y_val": np.array([0.0, 1.0, 0.0], dtype=float),
    }

    original_getitem = pd.DataFrame.__getitem__

    def _safe_getitem(self, key):
        if isinstance(key, list):
            return self.reindex(columns=key, fill_value=0.0)
        return original_getitem(self, key)

    def _fake_delayed(fn):
        def _builder(*args, **kwargs):
            return lambda: fn(*args, **kwargs)
        return _builder

    class _FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, jobs):
            return [job() for job in jobs]

    class _FakeStudy:
        def __init__(self, name):
            trial = SimpleNamespace(params={"source_study": name})
            self.trials = {0: trial, 0.0: trial}

        def trials_dataframe(self):
            return pd.DataFrame(
                [{"number": 0, "value": 0.3, "user_attrs_val_rmse": 0.2}]
            )

    def _load_study(study_name, storage):
        _ = storage
        calls["studies"].append(study_name)
        return _FakeStudy(study_name)

    def _ae_worker(*args, **kwargs):
        calls["ae"].append((args, kwargs))
        return "ae-study"

    def _nn_worker(*args, **kwargs):
        calls["nn"].append((args, kwargs))
        return None

    monkeypatch.setattr(pd.DataFrame, "__getitem__", _safe_getitem, raising=False)
    monkeypatch.setattr(ocdnnopt.np.linalg, "svd", lambda *_a, **_k: np.array([4.0, 2.0, 1.0], dtype=float))
    monkeypatch.setattr(ocdnnopt, "delayed", _fake_delayed)
    monkeypatch.setattr(ocdnnopt, "Parallel", _FakeParallel)
    monkeypatch.setattr(ocdnnopt.optuna, "load_study", _load_study)
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "AEworker", _ae_worker)
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "NNworker", _nn_worker)
    monkeypatch.setattr(ocdnnopt.ocprint, "printv", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdnnopt.ocprint, "print_warning", lambda *_a, **_k: None)

    result = ocdnnopt.optimize_NN(
        df_path="unused.csv",
        storage_id=42,
        base_models_folder="/tmp/models",
        data=data,
        autoencoder=True,
        multiencoder=True,
        run_autoencoder_optimization=True,
        num_processes_autoencoder=1,
        total_trials_autoencoder=1,
        run_NN_optimization=True,
        num_processes_NN=1,
        total_trials_NN=1,
        parallel_backend="joblib",
        use_gpu=False,
        verbose=True,
    )

    assert result is None
    assert len(calls["ae"]) == 2
    assert len(calls["nn"]) == 1
    assert "AO_Optimization_LIG_42_TPE" in calls["studies"]
    assert "AO_Optimization_REC_42_TPE" in calls["studies"]
    assert isinstance(calls["nn"][0][0][9], list)
    assert len(calls["nn"][0][0][9]) == 3
    assert calls["nn"][0][0][9][0]["n_layers_encoder"] == 1


@pytest.mark.order(397)
def test_optimize_nn_multiprocessing_without_autoencoder_runs_nn_worker(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    calls = {"nn": []}

    class _FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def starmap(self, fn, arg_list):
            out = []
            for args in arg_list:
                out.append(fn(*args))
            return out

    def _nn_worker(*args, **kwargs):
        calls["nn"].append((args, kwargs))
        return None

    monkeypatch.setattr(ocdnnopt, "Pool", _FakePool)
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "NNworker", _nn_worker)
    monkeypatch.setattr(ocdnnopt.ocprint, "print_warning", lambda *_a, **_k: None)

    result = ocdnnopt.optimize_NN(
        df_path="unused.csv",
        storage_id=43,
        base_models_folder="/tmp/models",
        data=_minimal_data_dict(),
        autoencoder=False,
        run_NN_optimization=True,
        num_processes_NN=1,
        total_trials_NN=1,
        parallel_backend="multiprocessing",
        use_gpu=False,
    )

    assert result is None
    assert len(calls["nn"]) == 1
    assert calls["nn"][0][0][9] is None


@pytest.mark.order(398)
def test_perform_ablation_study_nn_joblib_generates_masks_and_skips_evaluated(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    calls = {"ablation": []}

    x_train = pd.DataFrame(
        {"VINA_1": [0.1, 0.2], "SMINA_1": [0.3, 0.4], "f1": [1.0, 1.1]}
    )
    y_train = pd.Series([0.0, 1.0])

    class _FakeStudy:
        def trials_dataframe(self):
            return pd.DataFrame(
                [{"state": "COMPLETE", "user_attrs_Feature_Mask": "111"}]
            )

    def _fake_delayed(fn):
        def _builder(*args, **kwargs):
            return lambda: fn(*args, **kwargs)
        return _builder

    class _FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, jobs):
            return [job() for job in jobs]

    def _ablation_worker(*args, **kwargs):
        calls["ablation"].append((args, kwargs))
        return None

    monkeypatch.setattr(ocdnnopt.optuna, "load_study", lambda *_a, **_k: _FakeStudy())
    monkeypatch.setattr(ocdnnopt, "delayed", _fake_delayed)
    monkeypatch.setattr(ocdnnopt, "Parallel", _FakeParallel)
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "NNAblationworker", _ablation_worker)

    result = ocdnnopt.perform_ablation_study_NN(
        X_train=x_train,
        y_train=y_train,
        X_test=x_train,
        y_test=y_train,
        X_val=x_train,
        y_val=y_train,
        id=44,
        num_processes=4,
        encoder_params={},
        best_params={},
        random_seed=42,
        use_gpu=False,
        verbose=False,
        load_if_exists=False,
        study_name="ablation_runtime",
        storage="sqlite://",
        masks=None,
        parallel_backend="joblib",
    )

    assert result is None
    assert len(calls["ablation"]) == 3
    assert all(isinstance(call[0][8], list) for call in calls["ablation"])


@pytest.mark.order(399)
def test_perform_ablation_study_nn_multiprocessing_respects_inner_process_count(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    calls = {"ablation": []}

    x_train = pd.DataFrame({"VINA_1": [0.1, 0.2], "f1": [1.0, 1.1]})
    y_train = pd.Series([0.0, 1.0])

    class _FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def starmap(self, fn, arg_list):
            out = []
            for args in arg_list:
                out.append(fn(*args))
            return out

    def _ablation_worker(*args, **kwargs):
        calls["ablation"].append((args, kwargs))
        return None

    monkeypatch.setattr(ocdnnopt, "Pool", _FakePool)
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "NNAblationworker", _ablation_worker)

    result = ocdnnopt.perform_ablation_study_NN(
        X_train=x_train,
        y_train=y_train,
        X_test=x_train,
        y_test=y_train,
        X_val=x_train,
        y_val=y_train,
        id=45,
        num_processes=4,
        encoder_params={},
        best_params={},
        random_seed=42,
        use_gpu=False,
        verbose=False,
        load_if_exists=False,
        study_name="ablation_runtime_mp",
        storage="sqlite://",
        masks=[[1, 1], [0, 1]],
        parallel_backend="multiprocessing",
    )

    assert result is None
    assert len(calls["ablation"]) == 2


@pytest.mark.order(400)
def test_perform_seed_ablation_study_nn_joblib_uses_default_seed_range(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    calls = {"seed": []}

    def _fake_delayed(fn):
        def _builder(*args, **kwargs):
            return lambda: fn(*args, **kwargs)
        return _builder

    class _FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, jobs):
            return [job() for job in jobs]

    def _seed_worker(*args, **kwargs):
        calls["seed"].append((args, kwargs))
        return None

    monkeypatch.setattr(ocdnnopt, "delayed", _fake_delayed)
    monkeypatch.setattr(ocdnnopt, "Parallel", _FakeParallel)
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "NNSeedAblationworker", _seed_worker)

    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    result = ocdnnopt.perform_seed_ablation_study_NN(
        X_train=x,
        y_train=y,
        X_test=x,
        y_test=y,
        X_val=x,
        y_val=y,
        id=46,
        num_processes=4,
        encoder_params={},
        best_params={},
        use_gpu=False,
        verbose=False,
        load_if_exists=False,
        study_name="seed_runtime",
        storage="sqlite://",
        mask=np.array([1, 1], dtype=int),
        seeds=None,
        parallel_backend="joblib",
    )

    assert result is None
    assert len(calls["seed"]) == 4
    assert sum(len(call[0][11]) for call in calls["seed"]) == 1000


@pytest.mark.order(401)
def test_perform_seed_ablation_study_nn_multiprocessing_with_custom_seeds(monkeypatch, ocmods):
    ocdnnopt = ocmods["dnn_opt"]
    calls = {"seed": []}

    class _FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def starmap(self, fn, arg_list):
            out = []
            for args in arg_list:
                out.append(fn(*args))
            return out

    def _seed_worker(*args, **kwargs):
        calls["seed"].append((args, kwargs))
        return None

    monkeypatch.setattr(ocdnnopt, "Pool", _FakePool)
    monkeypatch.setattr(ocdnnopt.ocscoreworkers, "NNSeedAblationworker", _seed_worker)

    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    result = ocdnnopt.perform_seed_ablation_study_NN(
        X_train=x,
        y_train=y,
        X_test=x,
        y_test=y,
        X_val=x,
        y_val=y,
        id=47,
        num_processes=3,
        encoder_params={},
        best_params={},
        use_gpu=False,
        verbose=False,
        load_if_exists=False,
        study_name="seed_runtime_mp",
        storage="sqlite://",
        mask=np.array([1, 1], dtype=int),
        seeds=[7],
        parallel_backend="multiprocessing",
    )

    assert result is None
    assert len(calls["seed"]) == 1
    assert calls["seed"][0][0][11] == [7]
