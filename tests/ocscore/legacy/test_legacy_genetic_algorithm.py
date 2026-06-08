#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Dimensionality.GeneticAlgorithm helpers.
'''

# Imports
###############################################################################
import importlib.util
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


@pytest.fixture()
def ocga(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    pkg_root = root / "OCDocker"

    _ensure_package("OCDocker", pkg_root)
    _ensure_package("OCDocker.OCScore", pkg_root / "OCScore")
    _ensure_package("OCDocker.OCScore.Optimization.legacy.models.dimensionality", pkg_root / "OCScore" / "Optimization" / "legacy" / "models" / "dimensionality")
    _ensure_package("OCDocker.OCScore.Optimization.legacy.models.xgboost", pkg_root / "OCScore" / "Optimization" / "legacy" / "models" / "xgboost")
    _ensure_package("OCDocker.Toolbox", pkg_root / "Toolbox")

    cp_mod = types.ModuleType("cupy")
    cp_mod.asarray = lambda arr: np.asarray(arr)  # type: ignore[attr-defined]
    cp_mod.asnumpy = lambda arr: np.asarray(arr)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cupy", cp_mod)

    print_calls = []
    print_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    print_mod.printv = lambda msg: print_calls.append(msg)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", print_mod)

    xgb_calls = {"run": 0}
    xgb_mod = types.ModuleType("OCDocker.OCScore.Optimization.legacy.models.xgboost.OCxgboost")

    class _Model:
        def __init__(self, n_features):
            self.n_features_in_ = n_features

        def predict(self, X):
            Xn = np.asarray(X)
            return np.zeros(Xn.shape[0], dtype=float)

    def _run_xgboost(X_train, y_train, X_test, y_test, params, verbose=False):
        _ = (y_train, y_test, params, verbose)
        xgb_calls["run"] += 1
        n_features = np.asarray(X_train).shape[1]
        return _Model(n_features), float(np.asarray(X_test).shape[1])

    xgb_mod.run_xgboost = _run_xgboost  # type: ignore[attr-defined]
    xgb_mod.XGBRegressor = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Optimization.legacy.models.xgboost.OCxgboost", xgb_mod)

    module = _load_module(
        "OCDocker.OCScore.Optimization.legacy.models.dimensionality.GeneticAlgorithm",
        pkg_root / "OCScore" / "Optimization" / "legacy" / "models" / "dimensionality" / "GeneticAlgorithm.py",
    )

    return {
        "module": module,
        "print_calls": print_calls,
        "xgb_calls": xgb_calls,
    }


class _TrialStub:
    def __init__(self):
        self.attrs = {}

    def suggest_int(self, _name, low, _high):
        return low

    def suggest_float(self, _name, low, _high):
        return low

    def set_user_attr(self, key, value):
        self.attrs[key] = value

    def report(self, _value, _step):
        return None

    def should_prune(self):
        return False


## Public ##

@pytest.mark.order(640)
def test_ga_population_mutation_and_tournament_helpers(ocga):
    mod = ocga["module"]
    ga = mod.GeneticAlgorithm(
        X_train=np.zeros((6, 4)),
        y_train=np.array([0, 1, 0, 1, 0, 1]),
        X_test=np.zeros((6, 4)),
        y_test=np.array([0, 1, 0, 1, 0, 1]),
        xgboost_params={},
        use_gpu=False,
        fixed_features_index=[1],
        verbose=False,
    )

    pop = ga.initialize_population(number_of_features=4, population_size=6)
    assert pop.shape == (6, 4)
    assert np.all(pop[:, 1])
    assert np.all(np.any(pop, axis=1))

    ind = np.array([True, False, False, True], dtype=bool)
    ga.fixed_features_index = [0]
    mutated = ga.mutation(ind.copy(), mutation_rate=1.0)
    assert mutated[0] is True or mutated[0] == True

    ga.rng = types.SimpleNamespace(choice=lambda *a, **k: np.array([0, 1, 2]))  # type: ignore[assignment]
    population = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    fitness = np.array([0.1, 0.9, 0.5], dtype=float)

    ga.direction = "maximize"
    winner_max = ga.tournament_selection(population, fitness, tournament_size=3)
    assert np.array_equal(winner_max, population[1])

    ga.direction = "minimize"
    winner_min = ga.tournament_selection(population, fitness, tournament_size=3)
    assert np.array_equal(winner_min, population[0])


@pytest.mark.order(641)
def test_ga_objective_and_optimize_paths(ocga, monkeypatch):
    mod = ocga["module"]

    ga = mod.GeneticAlgorithm(
        X_train=np.zeros((4, 4)),
        y_train=np.array([0, 1, 0, 1]),
        X_test=np.zeros((4, 4)),
        y_test=np.array([0, 1, 0, 1]),
        xgboost_params={"eval_metric": "rmse"},
        use_gpu=True,
        verbose=True,
    )
    assert ga.xgboost_params["device"] == "cuda"

    monkeypatch.setattr(
        ga,
        "genetic_algorithm",
        lambda trial_params, trial: (np.array([1, 0, 1, 0], dtype=bool), object(), 0.33, None),
    )
    trial = _TrialStub()
    score = ga.objective(trial)
    assert score == pytest.approx(0.33)
    assert trial.attrs["best_individual"] == "1010"

    class _Study:
        def __init__(self):
            self.best_params = {"population_size": 8}
            self.best_value = 0.11
            self.best_trial = types.SimpleNamespace(user_attrs={})
            self.optimize_calls = []

        def optimize(self, objective, n_trials, n_jobs):
            _ = objective
            self.optimize_calls.append((n_trials, n_jobs))

    fake_study = _Study()
    monkeypatch.setattr(mod.optuna, "create_study", lambda **kwargs: fake_study)

    study, best_params, best_score = ga.optimize(
        direction="maximize",
        n_trials=3,
        n_jobs=2,
        study_name="ga_unit",
        load_if_exists=False,
        verbose=True,
    )

    assert study is fake_study
    assert best_params == {"population_size": 8}
    assert best_score == pytest.approx(0.11)
    assert fake_study.optimize_calls == [(3, 2)]
