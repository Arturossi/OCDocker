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
"""
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
    monkeypatch.setattr(
        ocexpshap,
        "compute_shap_values",
        lambda **_k: np.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]]),
    )
    monkeypatch.setattr(ocexpshap.shap_plots, "feature_importance_barh", lambda *_a, **_k: None)
    monkeypatch.setattr(ocexpshap.shap_plots, "beeswarm", lambda *_a, **_k: None)

    out = ocexpshap.run_export_shap_analysis(
        tmp_path / "export",
        _tiny_dudez_dataframe(),
        tmp_path / "shap_out",
    )

    assert (tmp_path / "shap_out" / "shap_values.npy").exists()
    assert (tmp_path / "shap_out" / "shap_values.csv").exists()
    report = json.loads((tmp_path / "shap_out" / "shap_report.json").read_text(encoding="utf-8"))
    assert report["feature_policy"]["feature_policy_name"] == "shape_only"
    assert report["selected_features"] == features
    assert out.shap_values_csv is not None


@pytest.mark.order(431)
def test_export_shap_missing_split_indices_raises(monkeypatch, tmp_path):
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
