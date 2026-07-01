#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for export-bundle SHAP (staged pipeline)."""

# Imports
###############################################################################
from types import SimpleNamespace

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import OCDocker.OCScore.Analysis.SHAP.ExportRunner as ocexpshap
import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""


# Functions
###############################################################################
## Private ##

def _tiny_dudez_dataframe(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "receptor": [f"r{i // 3}" for i in range(n)],
            "kind": ["ligand" if i % 2 == 0 else "decoy" for i in range(n)],
        }
    )


def _fake_bundle(model, selected_features, split_indices, task="dudez_screening"):
    return {
        "model": model,
        "scaler": None,
        "selected_features": selected_features,
        "split_indices": split_indices,
        "retrain_config": {"task": task},
        "feature_metadata": {
            "feature_policy": {
                "feature_policy_name": "shape_only",
                "selected_features_after_train_only_reduction": selected_features,
            }
        },
        "device": torch.device("cpu"),
    }


## Public ##


@pytest.mark.order(430)
def test_export_shap_happy_path_with_mocked_explainer(monkeypatch, tmp_path):
    '''Export SHAP should write values, report, and plot artifacts.'''

    features = ["f0", "f1"]
    params = {
        "encoder_architecture_index": 0,
        "encoder_hidden_sizes": [4],
        "encoder_latent_dim": 2,
        "encoder_depth": 1,
        "encoder_is_monotonic": False,
        "projection_dim": 0,
        "encoder_activation": "GELU",
        "encoder_dropout": 0.0,
        "classifier_hidden_size": 4,
        "classifier_dropout": 0.0,
        "classifier_activation": "GELU",
        "dudez_use_transfer": False,
    }
    model = ocstaged.build_dudez_model(input_size=2, params=params)
    split_indices = {
        "validation_indices": np.array([0, 1, 2]),
        "test_indices": np.array([3, 4, 5]),
    }
    bundle = _fake_bundle(model, features, split_indices)
    monkeypatch.setattr(ocexpshap.ocexport, "load_exported_model", lambda *_a, **_k: bundle)
    seen = {}

    def fake_compute_shap_values(**kwargs):
        seen["eval_f0"] = kwargs["X_eval"]["f0"].tolist()
        return np.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]])

    monkeypatch.setattr(
        ocexpshap,
        "compute_shap_values",
        fake_compute_shap_values,
    )
    monkeypatch.setattr(
        ocexpshap.shap_plots,
        "save_shap_plot_suite",
        lambda *_a, **_k: seen.update(
            {
                "metadata_receptor": _k["sample_metadata"]["receptor"].tolist(),
                "label_kind": _k["labels"]["kind"].tolist(),
            }
        )
        or {
            "feature_importance_png": str(tmp_path / "shap_out" / "policy_shap_feature_importance.png"),
            "beeswarm_png": str(tmp_path / "shap_out" / "policy_shap_beeswarm.png"),
        },
    )

    out = ocexpshap.run_export_shap_analysis(
        tmp_path / "export",
        _tiny_dudez_dataframe(),
        tmp_path / "shap_out",
        target_column="receptor",
        label_column="kind",
    )

    assert (tmp_path / "shap_out" / "shap_values.npy").exists()
    assert (tmp_path / "shap_out" / "shap_values.csv").exists()
    report = json.loads((tmp_path / "shap_out" / "shap_report.json").read_text(encoding="utf-8"))
    assert report["feature_policy"]["feature_policy_name"] == "shape_only"
    assert report["eval_split"] == "validation"
    assert report["selected_features"] == features
    assert seen["eval_f0"] == pytest.approx(_tiny_dudez_dataframe()["f0"].iloc[[0, 1, 2]].tolist())
    assert seen["metadata_receptor"] == _tiny_dudez_dataframe()["receptor"].iloc[[0, 1, 2]].tolist()
    assert seen["label_kind"] == _tiny_dudez_dataframe()["kind"].iloc[[0, 1, 2]].tolist()
    assert out.shap_values_csv is not None


@pytest.mark.order(431)
def test_export_shap_missing_split_indices_raises(monkeypatch, tmp_path):
    '''Export SHAP should require validation and test split indices.'''

    features = ["f0", "f1"]
    model = SimpleNamespace()
    bundle = _fake_bundle(model, features, split_indices={})
    monkeypatch.setattr(ocexpshap.ocexport, "load_exported_model", lambda *_a, **_k: bundle)

    with pytest.raises(ValueError, match="split_indices"):
        ocexpshap.run_export_shap_analysis(
            tmp_path / "export",
            _tiny_dudez_dataframe(),
            tmp_path / "shap_out",
        )


@pytest.mark.order(432)
def test_export_shap_missing_feature_columns_raises(monkeypatch, tmp_path):
    '''Export SHAP should reject dataframes missing selected features.'''

    features = ["f0", "f1", "missing_feature"]
    model = SimpleNamespace()
    split_indices = {
        "validation_indices": np.array([0, 1]),
        "test_indices": np.array([2, 3]),
    }
    bundle = _fake_bundle(model, features, split_indices)
    monkeypatch.setattr(ocexpshap.ocexport, "load_exported_model", lambda *_a, **_k: bundle)

    with pytest.raises(ValueError, match="missing selected export features"):
        ocexpshap.run_export_shap_analysis(
            tmp_path / "export",
            _tiny_dudez_dataframe(),
            tmp_path / "shap_out",
        )


@pytest.mark.order(433)
def test_export_shap_wrapper_exposes_scalar_nn_output():
    '''SHAP wrapper should expose one scalar model output per row.'''

    params = {
        "encoder_architecture_index": 0,
        "encoder_hidden_sizes": [4],
        "encoder_latent_dim": 2,
        "encoder_depth": 1,
        "encoder_is_monotonic": False,
        "projection_dim": 0,
        "encoder_activation": "GELU",
        "encoder_dropout": 0.0,
        "classifier_hidden_size": 4,
        "classifier_dropout": 0.0,
        "classifier_activation": "GELU",
        "dudez_use_transfer": False,
    }
    model = ocstaged.build_dudez_model(input_size=2, params=params)
    wrapper = ocexpshap._ShapNeuralWrapper(model, "dudez_screening")
    x = torch.randn(3, 2)
    out = wrapper.NN(x)
    assert out.shape == (3, 1)
