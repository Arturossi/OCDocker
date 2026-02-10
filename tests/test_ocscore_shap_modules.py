#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.SHAP modules.
'''

# Imports
###############################################################################
import importlib
import json
import sys
import types

import numpy as np
import pandas as pd

from types import SimpleNamespace

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

def _load_shap_modules(monkeypatch):
    fake_shap = types.ModuleType("shap")
    fake_shap_calls = {"summary": 0}

    class _FakeDeepExplainer:
        def __init__(self, model, background):
            self.model = model
            self.background = background

        def shap_values(self, x):
            arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            return arr[:, :, None]

    class _FakeKernelExplainer:
        def __init__(self, fn, background):
            self.fn = fn
            self.background = background

        def shap_values(self, x):
            _ = self.fn(np.asarray(x))
            return np.asarray(x)

    def _summary_plot(*_a, **_k):
        fake_shap_calls["summary"] += 1

    fake_shap.DeepExplainer = _FakeDeepExplainer
    fake_shap.KernelExplainer = _FakeKernelExplainer
    fake_shap.summary_plot = _summary_plot
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    shap_pkg = importlib.import_module("OCDocker.OCScore.Analysis.SHAP")
    cli_mod = importlib.import_module("OCDocker.OCScore.Analysis.SHAP.Cli")
    data_mod = importlib.import_module("OCDocker.OCScore.Analysis.SHAP.Data")
    explain_mod = importlib.import_module("OCDocker.OCScore.Analysis.SHAP.Explain")
    model_mod = importlib.import_module("OCDocker.OCScore.Analysis.SHAP.Model")
    plots_mod = importlib.import_module("OCDocker.OCScore.Analysis.SHAP.Plots")
    runner_mod = importlib.import_module("OCDocker.OCScore.Analysis.SHAP.Runner")
    studies_mod = importlib.import_module("OCDocker.OCScore.Analysis.SHAP.Studies")

    plots_mod = importlib.reload(plots_mod)
    data_mod = importlib.reload(data_mod)
    explain_mod = importlib.reload(explain_mod)
    model_mod = importlib.reload(model_mod)
    studies_mod = importlib.reload(studies_mod)
    runner_mod = importlib.reload(runner_mod)
    cli_mod = importlib.reload(cli_mod)
    shap_pkg = importlib.reload(shap_pkg)

    return {
        "pkg": shap_pkg,
        "cli": cli_mod,
        "data": data_mod,
        "explain": explain_mod,
        "model": model_mod,
        "plots": plots_mod,
        "runner": runner_mod,
        "studies": studies_mod,
        "calls": fake_shap_calls,
    }


## Public ##

@pytest.mark.order(289)
def test_shap_module_init_exports(monkeypatch):
    mods = _load_shap_modules(monkeypatch)
    assert "run_shap_analysis" in mods["pkg"].__all__
    assert "plots" in mods["pkg"].__all__


@pytest.mark.order(290)
def test_shap_cli_build_parser_and_main(monkeypatch, capsys):
    mods = _load_shap_modules(monkeypatch)
    cli_mod = mods["cli"]

    parser = cli_mod.build_argparser()
    args = parser.parse_args(
        [
            "--storage", "sqlite://",
            "--ao_study", "ao",
            "--nn_study", "nn",
            "--seed_study", "seed",
            "--mask_study", "mask",
            "--df_path", "df.csv.gz",
            "--base_models", "models",
            "--study_number", "12",
            "--out_dir", "out",
            "--explainer", "kernel",
            "--seed", "7",
            "--no_csv",
        ]
    )
    assert args.explainer == "kernel"
    assert args.no_csv is True

    monkeypatch.setattr(
        cli_mod,
        "run_shap_analysis",
        lambda **_k: SimpleNamespace(out_dir="out", shap_values_npy="out/shap_values.npy"),
    )

    rc = cli_mod.main(
        [
            "--storage", "sqlite://",
            "--ao_study", "ao",
            "--nn_study", "nn",
            "--seed_study", "seed",
            "--mask_study", "mask",
            "--df_path", "df.csv.gz",
            "--base_models", "models",
            "--study_number", "1",
            "--out_dir", "out",
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert json.loads(printed)["out_dir"] == "out"


@pytest.mark.order(291)
def test_shap_data_load_prepare_and_xtrain_none_error(monkeypatch):
    mods = _load_shap_modules(monkeypatch)
    data_mod = mods["data"]

    monkeypatch.setattr(data_mod.ocscoredata, "preprocess_df", lambda _p: (pd.DataFrame(), pd.DataFrame(), ["s1"]))
    monkeypatch.setattr(
        data_mod.ocscoredata,
        "load_data",
        lambda **_k: {
            "X_train": pd.DataFrame({"f1": [1.0, 2.0]}),
            "X_test": pd.DataFrame({"f1": [3.0]}),
            "X_val": pd.DataFrame({"f1": [4.0]}),
            "y_val": pd.Series([0.5]),
        },
    )
    monkeypatch.setattr(data_mod.ocscoredata, "invert_values_conditionally", lambda x: x)

    out = data_mod.load_and_prepare_data("df.csv.gz", "models", 1)
    assert out.X_train.shape == (2, 1)
    assert out.feature_names == ["f1"]
    assert out.y_val.shape[0] == 1

    monkeypatch.setattr(
        data_mod.ocscoredata,
        "load_data",
        lambda **_k: {
            "X_train": None,
            "X_test": pd.DataFrame({"f1": [3.0]}),
            "X_val": pd.DataFrame({"f1": [4.0]}),
            "y_val": pd.Series([0.5]),
        },
    )
    monkeypatch.setattr(data_mod.ocscoredata, "invert_values_conditionally", lambda x: x)

    with pytest.raises(ValueError, match="X_train"):
        data_mod.load_and_prepare_data("df.csv.gz", "models", 1)


@pytest.mark.order(292)
def test_shap_explain_helpers(monkeypatch):
    mods = _load_shap_modules(monkeypatch)
    explain_mod = mods["explain"]

    assert str(explain_mod._cuda_device()) in {"cpu", "cuda"}

    arr1 = explain_mod._squeeze_shap([np.array([[1.0, 2.0]])])
    assert arr1.shape == (1, 2)

    arr2 = explain_mod._squeeze_shap([np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])])
    assert np.allclose(arr2, np.array([[4.0, 6.0]]))

    arr3 = explain_mod._squeeze_shap(np.array([[[1.0], [2.0]]]))
    assert arr3.shape == (1, 2)

    with pytest.raises(ValueError):
        explain_mod._squeeze_shap(np.array([1.0, 2.0, 3.0]))

    df = pd.DataFrame({"g": [0, 0, 1, 1], "x": [1, 2, 3, 4]})
    idx_plain = explain_mod._stratified_indices(df, n=3, by=None, seed=1)
    idx_group = explain_mod._stratified_indices(df, n=3, by=["g"], seed=1)
    assert len(idx_plain) == 3
    assert len(idx_group) <= 3
    assert sorted(idx_group.tolist()) == idx_group.tolist()


@pytest.mark.order(293)
def test_shap_compute_values_deep_kernel_and_invalid(monkeypatch):
    mods = _load_shap_modules(monkeypatch)
    explain_mod = mods["explain"]

    class _NN:
        def __call__(self, x):
            return x.sum(dim=1, keepdim=True)

    neural = SimpleNamespace(NN=_NN())
    X_background = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.2, 0.3]})
    X_eval = pd.DataFrame({"f1": [1.5, 2.5], "f2": [0.4, 0.5]})

    out_deep = explain_mod.compute_shap_values(
        neural=neural,
        X_background=X_background,
        X_eval=X_eval,
        explainer="deep",
        background_size=2,
        eval_size=2,
        rng_seed=3,
    )
    out_kernel = explain_mod.compute_shap_values(
        neural=neural,
        X_background=X_background,
        X_eval=X_eval,
        explainer="kernel",
        background_size=2,
        eval_size=2,
        rng_seed=4,
    )

    assert out_deep.shape == (2, 2)
    assert out_kernel.shape == (2, 2)

    calls = []
    monkeypatch.setattr(explain_mod.ocerror.Error, "value_error", lambda msg: calls.append(msg))
    with pytest.raises(ValueError, match="explainer must be"):
        explain_mod.compute_shap_values(neural, X_background, X_eval, explainer="bad")
    assert calls


@pytest.mark.order(294)
def test_shap_model_build_neural_net(monkeypatch):
    mods = _load_shap_modules(monkeypatch)
    model_mod = mods["model"]

    class _FakeNNInner:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True

    class _FakeNeural:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.NN = _FakeNNInner()

    monkeypatch.setattr(model_mod, "NeuralNet", _FakeNeural)
    monkeypatch.setattr(model_mod.torch.cuda, "is_available", lambda: False)

    out = model_mod.build_neural_net(
        input_dim=5,
        autoencoder_params={"a": 1},
        nn_params={"b": 2},
        seed=11,
        mask=[1, 0, 1],
        use_gpu=None,
        verbose=True,
    )
    assert out.kwargs["use_gpu"] is False
    assert out.NN.eval_called is True


@pytest.mark.order(295)
def test_shap_studies_select_best_from_studies(monkeypatch):
    mods = _load_shap_modules(monkeypatch)
    studies_mod = mods["studies"]

    class _Trial:
        def __init__(self, params=None, user_attrs=None):
            self.params = params or {}
            self.user_attrs = user_attrs or {}

    class _Study:
        def __init__(self, df, trials):
            self._df = df
            self.trials = trials

        def trials_dataframe(self):
            return self._df

    ao_df = pd.DataFrame({"number": [0], "state": ["COMPLETE"], "value": [0.1], "user_attrs_val_rmse": [0.2]})
    nn_df = pd.DataFrame({"number": [0], "state": ["COMPLETE"], "value": [0.8], "user_attrs_AUC": [0.3]})
    seed_df = pd.DataFrame({"number": [0], "state": ["COMPLETE"], "value": [0.2], "user_attrs_AUC": [0.9]})
    mask_df = pd.DataFrame({"number": [0], "state": ["COMPLETE"], "value": [0.2], "user_attrs_AUC": [0.8]})

    ao_study = _Study(ao_df, [_Trial(params={"enc": 64})])
    nn_study = _Study(nn_df, [_Trial(params={"layers": 3})])
    seed_study = _Study(seed_df, [_Trial(user_attrs={"random_seed": 123})])
    mask_study = _Study(mask_df, [_Trial(user_attrs={"Feature_Mask": [1, 0, 1, 1]})])
    mapping = {"ao": ao_study, "nn": nn_study, "seed": seed_study, "mask": mask_study}

    monkeypatch.setattr(studies_mod.optuna, "load_study", lambda study_name, storage: mapping[study_name])
    handles = studies_mod.StudyHandles("ao", "nn", "seed", "mask", "sqlite://")
    best = studies_mod.select_best_from_studies(handles)

    assert best.autoencoder_params["enc"] == 64
    assert best.nn_params["layers"] == 3
    assert best.seed == 123
    assert np.array_equal(best.mask, np.array([1, 0, 1, 1]))


@pytest.mark.order(296)
def test_shap_runner_happy_path_with_and_without_csv(monkeypatch, tmp_path):
    mods = _load_shap_modules(monkeypatch)
    runner_mod = mods["runner"]

    best = SimpleNamespace(
        autoencoder_params={"enc": 16},
        nn_params={"layers": 2},
        seed=42,
        mask=np.array([1, 1, 0]),
    )
    data = SimpleNamespace(
        X_train=pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.1, 0.2]}),
        X_test=pd.DataFrame({"f1": [1.5, 2.5], "f2": [0.15, 0.25]}),
        feature_names=["f1", "f2"],
    )
    monkeypatch.setattr(runner_mod, "select_best_from_studies", lambda _s: best)
    monkeypatch.setattr(runner_mod, "load_and_prepare_data", lambda **_k: data)
    monkeypatch.setattr(runner_mod, "build_neural_net", lambda **_k: SimpleNamespace(NN=object()))
    monkeypatch.setattr(runner_mod, "compute_shap_values", lambda **_k: np.array([[0.2, -0.2], [0.1, -0.1]]))
    calls = {"barh": 0, "beeswarm": 0}
    monkeypatch.setattr(runner_mod.plots, "feature_importance_barh", lambda *_a, **_k: calls.__setitem__("barh", calls["barh"] + 1))
    monkeypatch.setattr(runner_mod.plots, "beeswarm", lambda *_a, **_k: calls.__setitem__("beeswarm", calls["beeswarm"] + 1))

    studies = runner_mod.StudyHandles("ao", "nn", "seed", "mask", "sqlite://")
    out_with_csv = runner_mod.run_shap_analysis(
        studies=studies,
        df_path="df.csv.gz",
        base_models_folder="models",
        study_number=1,
        out_dir=str(tmp_path / "with_csv"),
        save_csv=True,
    )
    out_no_csv = runner_mod.run_shap_analysis(
        studies=studies,
        df_path="df.csv.gz",
        base_models_folder="models",
        study_number=1,
        out_dir=str(tmp_path / "no_csv"),
        save_csv=False,
    )

    assert (tmp_path / "with_csv" / "shap_values.npy").exists()
    assert (tmp_path / "with_csv" / "shap_values.csv").exists()
    assert (tmp_path / "no_csv" / "shap_values.npy").exists()
    assert out_with_csv.shap_values_csv is not None
    assert out_no_csv.shap_values_csv is None
    assert calls["barh"] == 2
    assert calls["beeswarm"] == 2


@pytest.mark.order(297)
def test_shap_plots_helpers(monkeypatch, tmp_path):
    mods = _load_shap_modules(monkeypatch)
    plots_mod = mods["plots"]
    calls = mods["calls"]

    saved = []
    monkeypatch.setattr(plots_mod.plt, "savefig", lambda path, **_k: saved.append(path))

    zero_rel = plots_mod._relative_importance(np.zeros((3, 2), dtype=float))
    assert np.allclose(zero_rel, np.array([0.0, 0.0]))

    shap_2d = np.array([[0.2, -0.1], [0.4, -0.2]], dtype=float)
    X_eval = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})

    out_bee = plots_mod.beeswarm(shap_2d, X_eval, out_png=str(tmp_path / "bee.png"))
    out_bar = plots_mod.feature_importance_barh(shap_2d, ["f1", "f2"], out_png=str(tmp_path / "bar.png"), top_k=1)
    out_hm_np = plots_mod.shap_correlation_heatmap(shap_2d, out_png=str(tmp_path / "hm_np.png"), feature_names=["f1", "f2"])
    out_hm_df = plots_mod.shap_correlation_heatmap(X_eval, out_png=str(tmp_path / "hm_df.png"))

    assert out_bee.endswith("bee.png")
    assert out_bar.endswith("bar.png")
    assert out_hm_np.endswith("hm_np.png")
    assert out_hm_df.endswith("hm_df.png")
    assert calls["summary"] >= 1
    assert len(saved) >= 4


@pytest.mark.order(298)
def test_shap_explain_stratified_indices_with_group_columns(monkeypatch):
    mods = _load_shap_modules(monkeypatch)
    explain_mod = mods["explain"]

    df = pd.DataFrame(
        {
            "grp1": [0, 0, 1, 1, 1],
            "grp2": [1, 1, 1, 0, 0],
            "x": [10, 20, 30, 40, 50],
        }
    )
    idx = explain_mod._stratified_indices(df, n=4, by=["grp1", "grp2"], seed=9)
    assert idx.ndim == 1
    assert len(idx) <= 4


@pytest.mark.order(414)
def test_shap_cli_main_passes_optional_arguments_and_saves_csv_by_default(monkeypatch, capsys):
    mods = _load_shap_modules(monkeypatch)
    cli_mod = mods["cli"]

    captured = {}

    def _fake_run_shap_analysis(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(out_dir="out", shap_values_npy="out/shap_values.npy")

    monkeypatch.setattr(cli_mod, "run_shap_analysis", _fake_run_shap_analysis)

    rc = cli_mod.main(
        [
            "--storage", "sqlite://",
            "--ao_study", "ao",
            "--nn_study", "nn",
            "--seed_study", "seed",
            "--mask_study", "mask",
            "--df_path", "df.csv.gz",
            "--base_models", "models",
            "--study_number", "3",
            "--out_dir", "out",
            "--explainer", "deep",
            "--background_size", "15",
            "--eval_size", "5",
            "--stratify_by", "target", "active",
            "--seed", "77",
        ]
    )

    assert rc == 0
    assert captured["background_size"] == 15
    assert captured["eval_size"] == 5
    assert captured["stratify_by"] == ["target", "active"]
    assert captured["seed"] == 77
    assert captured["save_csv"] is True
    printed = capsys.readouterr().out
    assert json.loads(printed)["shap_values_npy"] == "out/shap_values.npy"


@pytest.mark.order(415)
def test_shap_data_accepts_numpy_y_val(monkeypatch):
    mods = _load_shap_modules(monkeypatch)
    data_mod = mods["data"]

    monkeypatch.setattr(data_mod.ocscoredata, "preprocess_df", lambda _p: (pd.DataFrame(), pd.DataFrame(), ["s1"]))
    monkeypatch.setattr(
        data_mod.ocscoredata,
        "load_data",
        lambda **_k: {
            "X_train": pd.DataFrame({"f1": [1.0, 2.0]}),
            "X_test": pd.DataFrame({"f1": [3.0]}),
            "X_val": pd.DataFrame({"f1": [4.0]}),
            "y_val": np.array([0.25, 0.75], dtype=float),
        },
    )
    monkeypatch.setattr(data_mod.ocscoredata, "invert_values_conditionally", lambda x: x)

    out = data_mod.load_and_prepare_data("df.csv.gz", "models", 1)
    assert out.y_val.shape == (2,)
    assert np.allclose(out.y_val, np.array([0.25, 0.75], dtype=float))


@pytest.mark.order(416)
def test_shap_runner_passes_mask_as_list_to_model_builder(monkeypatch, tmp_path):
    mods = _load_shap_modules(monkeypatch)
    runner_mod = mods["runner"]

    best = SimpleNamespace(
        autoencoder_params={"enc": 16},
        nn_params={"layers": 2},
        seed=7,
        mask=np.array([1, 0, 1]),
    )
    data = SimpleNamespace(
        X_train=pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.1, 0.2]}),
        X_test=pd.DataFrame({"f1": [1.5, 2.5], "f2": [0.15, 0.25]}),
        feature_names=["f1", "f2"],
    )
    calls = {"mask": None}

    monkeypatch.setattr(runner_mod, "select_best_from_studies", lambda _s: best)
    monkeypatch.setattr(runner_mod, "load_and_prepare_data", lambda **_k: data)
    monkeypatch.setattr(
        runner_mod,
        "build_neural_net",
        lambda **kwargs: calls.__setitem__("mask", kwargs.get("mask")) or SimpleNamespace(NN=object()),
    )
    monkeypatch.setattr(runner_mod, "compute_shap_values", lambda **_k: np.array([[0.2, -0.2], [0.1, -0.1]]))
    monkeypatch.setattr(runner_mod.plots, "feature_importance_barh", lambda *_a, **_k: None)
    monkeypatch.setattr(runner_mod.plots, "beeswarm", lambda *_a, **_k: None)

    studies = runner_mod.StudyHandles("ao", "nn", "seed", "mask", "sqlite://")
    out = runner_mod.run_shap_analysis(
        studies=studies,
        df_path="df.csv.gz",
        base_models_folder="models",
        study_number=1,
        out_dir=str(tmp_path / "mask_list"),
        save_csv=False,
    )

    assert isinstance(calls["mask"], list)
    assert calls["mask"] == [1, 0, 1]
    assert out.shap_values_csv is None


@pytest.mark.order(417)
def test_shap_studies_accepts_string_feature_mask(monkeypatch):
    mods = _load_shap_modules(monkeypatch)
    studies_mod = mods["studies"]

    class _Trial:
        def __init__(self, params=None, user_attrs=None):
            self.params = params or {}
            self.user_attrs = user_attrs or {}

    class _Study:
        def __init__(self, df, trials):
            self._df = df
            self.trials = trials

        def trials_dataframe(self):
            return self._df

    ao_df = pd.DataFrame({"number": [0], "state": ["COMPLETE"], "value": [0.1], "user_attrs_val_rmse": [0.2]})
    nn_df = pd.DataFrame({"number": [0], "state": ["COMPLETE"], "value": [0.8], "user_attrs_AUC": [0.3]})
    seed_df = pd.DataFrame({"number": [0], "state": ["COMPLETE"], "value": [0.2], "user_attrs_AUC": [0.9]})
    mask_df = pd.DataFrame({"number": [0], "state": ["COMPLETE"], "value": [0.2], "user_attrs_AUC": [0.8]})

    mapping = {
        "ao": _Study(ao_df, [_Trial(params={"enc": 64})]),
        "nn": _Study(nn_df, [_Trial(params={"layers": 3})]),
        "seed": _Study(seed_df, [_Trial(user_attrs={"random_seed": 123})]),
        "mask": _Study(mask_df, [_Trial(user_attrs={"Feature_Mask": "10110"})]),
    }
    monkeypatch.setattr(studies_mod.optuna, "load_study", lambda study_name, storage: mapping[study_name])

    handles = studies_mod.StudyHandles("ao", "nn", "seed", "mask", "sqlite://")
    best = studies_mod.select_best_from_studies(handles)
    assert np.array_equal(best.mask, np.array([1, 0, 1, 1, 0]))
