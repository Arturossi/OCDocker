#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore analysis core utilities.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd

from types import SimpleNamespace

import pytest

import OCDocker.OCScore.Analysis.Correlation as occorrelation
import OCDocker.OCScore.Analysis.FeatureImportance as ocfeatimp
import OCDocker.OCScore.Analysis.Metrics.Bootstrap as ocbootstrap

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

class _DummyExplainer:
    def __init__(self, payload, expected_value):
        self._payload = payload
        self.expected_value = expected_value

    def shap_values(self, X, nsamples="auto"):
        _ = (X, nsamples)
        return self._payload


class _TypeErrorExplainer:
    def __init__(self, payload, expected_value):
        self._payload = payload
        self.expected_value = expected_value

    def shap_values(self, X, nsamples="auto"):
        _ = nsamples
        if X is not None:
            raise TypeError("nsamples not supported")
        return self._payload


def _accuracy_metric(y_true: np.ndarray, y_score: np.ndarray) -> float:
    preds = (y_score >= 0.5).astype(int)
    return float(np.mean(preds == y_true))


## Public ##

@pytest.mark.order(232)
def test_bootstrap_ci_without_bootstrapping_returns_nan_bounds():
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_score = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
    est, low, high = ocbootstrap.bootstrap_ci(y_true, y_score, _accuracy_metric, n_boot=0)
    assert est == pytest.approx(1.0)
    assert np.isnan(low)
    assert np.isnan(high)


@pytest.mark.order(233)
def test_bootstrap_ci_with_and_without_strata():
    y_true = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    y_score = np.array([0.2, 0.8, 0.9, 0.1, 0.7, 0.4], dtype=float)

    est1, low1, high1 = ocbootstrap.bootstrap_ci(y_true, y_score, _accuracy_metric, n_boot=20, random_state=1)
    est2, low2, high2 = ocbootstrap.bootstrap_ci(
        y_true,
        y_score,
        _accuracy_metric,
        n_boot=20,
        random_state=1,
        strata=["a", "a", "b", "b", "a", "b"],
    )

    assert 0.0 <= low1 <= high1 <= 1.0
    assert 0.0 <= low2 <= high2 <= 1.0
    assert 0.0 <= est1 <= 1.0
    assert 0.0 <= est2 <= 1.0


@pytest.mark.order(234)
def test_correlation_analysis_generates_barplot(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(occorrelation.sns, "barplot", lambda **_k: None)
    monkeypatch.setattr(occorrelation.plt, "savefig", lambda path, dpi=300: saved.append((path, dpi)))

    results_df = pd.DataFrame(
        {
            "study_name": ["s1", "s2", "s3", "s4"],
            "study_type": ["MethodA", "MethodA", "MethodB", "MethodB"],
            "best_combined_value": [0.9, 1.1, 1.2, 1.0],
            "best_combined_auc": [0.7, 0.72, 0.8, 0.78],
            "best_combined_metric": [0.1, 0.12, 0.2, 0.18],
        }
    )
    final_metrics = pd.DataFrame(
        {
            "study_name": ["raw1", "raw2", "consensus1", "consensus2"],
            "Methodology": ["Raw", "Raw", "Consensus", "Consensus"],
            "RMSE": [0.8, 0.82, 1.0, 0.98],
            "AUC": [0.75, 0.76, 0.78, 0.79],
            "combined_metric": [0.11, 0.115, 0.12, 0.118],
        }
    )

    occorrelation.correlation_analysis(
        results_df=results_df,
        final_metrics=final_metrics,
        n_trials=5,
        error_threshold=2.0,
        save_path=str(tmp_path),
        colour_mapping={"MethodA": "blue"},
    )

    assert saved
    assert saved[0][0].endswith("Experiments_Correlation_barplot_5.png")


@pytest.mark.order(235)
def test_feature_importance_background_and_shape_helpers():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    meta = pd.DataFrame({"target": [0, 0, 1, 1], "active": [1, 0, 1, 0]})

    bg = ocfeatimp.build_stratified_background(X, meta, ["target", "active"], per_stratum=1, seed=7)
    assert bg.ndim == 2
    assert bg.shape[1] == 2

    with pytest.raises(ValueError):
        ocfeatimp.build_stratified_background(X[:2], meta, ["target"], per_stratum=1, seed=0)

    one_dim = ocfeatimp._ensure_2d(np.array([1.0, 2.0, 3.0]))
    assert one_dim.shape == (3, 1)

    from_df = ocfeatimp._ensure_2d(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert from_df.shape == (2, 2)


@pytest.mark.order(236)
def test_compute_shap_values_handles_list_outputs_and_base_values(monkeypatch):
    monkeypatch.setattr(ocfeatimp, "shap", SimpleNamespace())

    payload = [
        np.array([[0.1, -0.1], [0.2, -0.2]], dtype=float),
        np.array([[0.3, -0.3], [0.4, -0.4]], dtype=float),
    ]
    explainer = _DummyExplainer(payload=payload, expected_value=[0.0, 0.5])
    out = ocfeatimp.compute_shap_values(explainer, np.array([[1.0, 2.0], [3.0, 4.0]]), class_index=1)

    assert out["shap_values"].shape == (2, 2)
    assert np.allclose(out["base_values"], np.array([0.5, 0.5]))


@pytest.mark.order(237)
def test_compute_shap_values_typeerror_fallback_and_scalar_base(monkeypatch):
    monkeypatch.setattr(ocfeatimp, "shap", SimpleNamespace())

    class _NoNsamplesExplainer:
        expected_value = 0.25

        def shap_values(self, X):
            _ = X
            return np.array([[1.0, -1.0]], dtype=float)

    out = ocfeatimp.compute_shap_values(_NoNsamplesExplainer(), np.array([[1.0, 2.0]]))
    assert out["shap_values"].shape == (1, 2)
    assert np.allclose(out["base_values"], np.array([0.25]))


@pytest.mark.order(238)
def test_make_explainer_and_importance_table_branches(monkeypatch):
    calls = []

    class _FakeShap:
        @staticmethod
        def TreeExplainer(model, data=None):
            calls.append(("tree", type(model).__name__, np.asarray(data).shape))
            return "tree_explainer"

        @staticmethod
        def DeepExplainer(model, bg):
            calls.append(("deep", type(model).__name__, np.asarray(bg).shape))
            return "deep_explainer"

        @staticmethod
        def KernelExplainer(fn, bg, link=None):
            calls.append(("kernel", fn.__name__ if hasattr(fn, "__name__") else "lambda", np.asarray(bg).shape, link))
            return "kernel_explainer"

    monkeypatch.setattr(ocfeatimp, "shap", _FakeShap())

    class _TreeModel:
        feature_importances_ = np.array([0.1, 0.2])

        def predict_proba(self, X):
            return np.column_stack([1 - np.asarray(X)[:, 0], np.asarray(X)[:, 0]])

    class _DeepModel:
        pass

    _DeepModel.__module__ = "torch.nn.modules"

    class _KernelModel:
        def predict(self, X):
            return np.asarray(X)[:, 0]

    bg = np.array([[0.1, 0.2], [0.3, 0.4]])

    expl_tree, idx_tree = ocfeatimp.make_explainer(_TreeModel(), bg, method="auto")
    expl_deep, idx_deep = ocfeatimp.make_explainer(_DeepModel(), bg, method="auto")
    expl_kernel, idx_kernel = ocfeatimp.make_explainer(_KernelModel(), bg, method="kernel", link="logit")
    assert (expl_tree, idx_tree) == ("tree_explainer", 1)
    assert (expl_deep, idx_deep) == ("deep_explainer", 1)
    assert (expl_kernel, idx_kernel) == ("kernel_explainer", 1)

    with pytest.raises(ValueError):
        ocfeatimp.make_explainer(_KernelModel(), bg, method="invalid")

    assert any(call[0] == "tree" for call in calls)
    assert any(call[0] == "deep" for call in calls)
    assert any(call[0] == "kernel" for call in calls)

    shap_values = np.array([[0.2, -0.1, 0.4], [0.1, -0.5, 0.2]])
    table_full = ocfeatimp.shap_importance_table(shap_values, feature_names=["f0", "f1", "f2"])
    table_top2 = ocfeatimp.shap_importance_table(shap_values, k=2)

    assert list(table_full.columns) == ["feature", "mean_abs_shap", "rank"]
    assert table_full.iloc[0]["mean_abs_shap"] >= table_full.iloc[1]["mean_abs_shap"]
    assert table_top2.shape[0] == 2


@pytest.mark.order(239)
def test_require_shap_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(ocfeatimp, "shap", None)
    with pytest.raises(ImportError):
        ocfeatimp._require_shap()


@pytest.mark.order(352)
def test_feature_importance_additional_background_and_base_value_paths(monkeypatch):
    monkeypatch.setattr(ocfeatimp, "shap", SimpleNamespace())

    X = np.arange(24, dtype=float).reshape(12, 2)
    meta = pd.DataFrame({"target": [0] * 6 + [1] * 6, "active": [0, 1, 0, 1, 0, 1] * 2})
    bg = ocfeatimp.build_stratified_background(X, meta, ["target"], per_stratum=2, seed=1)
    assert bg.shape == (4, 2)

    class _BaseShortExplainer:
        expected_value = [0.33]

        def shap_values(self, X, nsamples="auto"):
            _ = nsamples
            return np.asarray(X, dtype=float)

    short_base_out = ocfeatimp.compute_shap_values(
        _BaseShortExplainer(),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        class_index=1,
    )
    assert np.allclose(short_base_out["base_values"], np.array([0.33, 0.33]))

    class _BaseArrayExplainer:
        expected_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)

        def shap_values(self, X, nsamples="auto"):
            _ = nsamples
            return np.asarray(X, dtype=float)

    array_base_out = ocfeatimp.compute_shap_values(
        _BaseArrayExplainer(),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
        class_index=1,
    )
    assert np.allclose(array_base_out["base_values"], np.array([0.3, 0.4]))


@pytest.mark.order(353)
def test_feature_importance_make_explainer_extra_paths(monkeypatch):
    calls = []

    class _FakeShap:
        @staticmethod
        def TreeExplainer(model, data=None):
            calls.append(("tree", model.__class__.__name__, np.asarray(data).shape))
            return "tree_explicit"

        @staticmethod
        def DeepExplainer(model, bg):
            calls.append(("deep", model.__class__.__name__, np.asarray(bg).shape))
            return "deep_explicit"

        @staticmethod
        def KernelExplainer(fn, bg, link=None):
            calls.append(("kernel", fn(np.array([[0.2, 0.8]], dtype=float)), np.asarray(bg).shape, link))
            return "kernel_any"

    monkeypatch.setattr(ocfeatimp, "shap", _FakeShap())

    class _KernelModel:
        def predict(self, X):
            return np.asarray(X)[:, 0]

    class _TreeModel:
        def apply(self, X):
            return np.asarray(X)

    class _DeepModel:
        pass

    _DeepModel.__module__ = "tensorflow.keras"

    bg = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)

    # method='auto' + non-tree/non-deep model should fall back to kernel and sanitize unknown link
    kernel_auto, kernel_idx = ocfeatimp.make_explainer(_KernelModel(), bg, method="auto", link="unsupported")
    assert (kernel_auto, kernel_idx) == ("kernel_any", 1)

    # Explicit tree/deep branches
    tree_explicit, tree_idx = ocfeatimp.make_explainer(_TreeModel(), bg, method="tree")
    deep_explicit, deep_idx = ocfeatimp.make_explainer(_DeepModel(), bg, method="deep")
    assert (tree_explicit, tree_idx) == ("tree_explicit", 1)
    assert (deep_explicit, deep_idx) == ("deep_explicit", 1)

    # Explicit predict_fn branch
    pred_calls = {"count": 0}

    def _custom_predict(X):
        pred_calls["count"] += 1
        return np.asarray(X)[:, 1]

    kernel_custom, custom_idx = ocfeatimp.make_explainer(
        _KernelModel(),
        bg,
        method="kernel",
        link="identity",
        predict_fn=_custom_predict,
    )
    assert (kernel_custom, custom_idx) == ("kernel_any", 1)
    assert pred_calls["count"] >= 1
    assert any(call[0] == "tree" for call in calls)
    assert any(call[0] == "deep" for call in calls)
    assert any(call[0] == "kernel" and call[-1] is None for call in calls)
    assert any(call[0] == "kernel" and call[-1] == "identity" for call in calls)
