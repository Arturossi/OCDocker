#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for exported-model inference helpers."""

# Imports
###############################################################################
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import OCDocker.OCScore.Optimization.ModelExport as ocexport
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

def _build_pdbbind_export(tmp_path, features: list[str], *, hidden_sizes: list[int] | None = None, latent_dim: int = 3):
    input_size = len(features)
    encoder_hidden = hidden_sizes or [8, 4]
    params = {
        "encoder_architecture_index": 0,
        "encoder_hidden_sizes": encoder_hidden,
        "encoder_latent_dim": latent_dim,
        "encoder_depth": 2,
        "encoder_is_monotonic": True,
        "projection_dim": 0,
        "encoder_activation": "GELU",
        "encoder_dropout": 0.0,
        "optimizer_learning_rate": 1e-3,
        "optimizer_weight_decay": 1e-4,
        "optimizer_batch_size": 4,
        "decoder_hidden_sizes": [],
        "decoder_depth": 0,
        "decoder_lambda_rec": 0.0,
        "dae_noise_type": "none",
        "dae_mask_prob": 0.0,
        "dae_gaussian_std": 0.0,
        "pdbbind_regression_loss": "mse",
        "pdbbind_huber_delta": 1.0,
    }
    model = ocstaged.build_pdbbind_model(input_size=input_size, params=params)
    rng = np.random.default_rng(0)
    splits = {
        "X_train": rng.normal(size=(8, input_size)).astype(np.float32),
        "y_train": rng.normal(size=8).astype(np.float32),
        "X_val": rng.normal(size=(4, input_size)).astype(np.float32),
        "y_val": rng.normal(size=4).astype(np.float32),
        "X_test": rng.normal(size=(4, input_size)).astype(np.float32),
        "y_test": rng.normal(size=4).astype(np.float32),
        "train_indices": np.arange(8),
        "validation_indices": np.arange(8, 12),
        "test_indices": np.arange(12, 16),
        "split_config": {"target_column": "experimental", "validation_size": 0.2, "test_size": 0.2},
        "split_diagnostics": {},
    }
    splits["scaler"] = __import__(
        "sklearn.preprocessing",
        fromlist=["StandardScaler"],
    ).StandardScaler().fit(splits["X_train"])

    export_dir = tmp_path / "pdbbind_best_model"
    ocexport.export_best_model_bundle(
        export_dir=export_dir,
        task="pdbbind_regression",
        model=model,
        model_config=params,
        selected_features=features,
        best_trial_number=1,
        best_objective_value=1.0,
        validation_metrics={"RMSE": 1.0},
        test_metrics={"RMSE": 1.1},
        stage_config={"epochs": 1},
        splits=splits,
        objective_metric="RMSE",
        direction="minimize",
        best_params={"optimizer_batch_size": 4},
        random_seed=0,
    )
    return export_dir


def _feature_dataframe(features: list[str], rows: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {feature: rng.normal(size=rows) for feature in features}
    data.update({"receptor": [f"r{i}" for i in range(rows)], "name": [f"l{i}" for i in range(rows)]})
    return pd.DataFrame(data)


## Public ##

@pytest.mark.order(290)
def test_validate_export_features_missing_column():
    with pytest.raises(ValueError, match="missing selected export features"):
        ocexport.validate_export_features(pd.DataFrame({"f0": [1.0]}), ["f0", "f1"])


@pytest.mark.order(291)
def test_predict_from_export_pdbbind_applies_scaler(tmp_path, monkeypatch):
    features = [f"f{i}" for i in range(4)]
    export_dir = _build_pdbbind_export(tmp_path, features)
    dataframe = _feature_dataframe(features)

    bundle = ocexport.load_exported_model(export_dir)
    scaler = bundle["scaler"]
    scaler.transform = MagicMock(side_effect=scaler.transform)
    monkeypatch.setattr(
        ocexport,
        "load_exported_model",
        lambda *_args, **_kwargs: {**bundle, "scaler": scaler},
    )

    output = ocexport.predict_from_export(export_dir, dataframe)
    assert len(output) == len(dataframe)
    assert "ocscore_prediction" in output.columns
    assert "receptor" in output.columns
    scaler.transform.assert_called_once()


@pytest.mark.order(292)
def test_predict_from_export_permuted_columns(tmp_path):
    features = [f"f{i}" for i in range(4)]
    export_dir = _build_pdbbind_export(tmp_path, features)
    dataframe = _feature_dataframe(features)
    shuffled = dataframe[list(dataframe.columns[::-1])]

    baseline = ocexport.predict_from_export(export_dir, dataframe)
    permuted = ocexport.predict_from_export(export_dir, shuffled)
    np.testing.assert_allclose(
        baseline["ocscore_prediction"].to_numpy(),
        permuted["ocscore_prediction"].to_numpy(),
    )


@pytest.mark.order(293)
def test_predict_from_export_drops_nan_and_inf_rows(tmp_path):
    features = [f"f{i}" for i in range(3)]
    export_dir = _build_pdbbind_export(tmp_path, features)
    dataframe = _feature_dataframe(features, rows=4)
    dataframe.loc[1, "f0"] = np.nan
    dataframe.loc[2, "f1"] = np.inf

    output = ocexport.predict_from_export(export_dir, dataframe)
    assert len(output) == 2


@pytest.mark.order(294)
def test_predict_from_export_all_non_finite_raises(tmp_path):
    features = [f"f{i}" for i in range(2)]
    export_dir = _build_pdbbind_export(tmp_path, features)
    dataframe = pd.DataFrame({"f0": [np.nan, np.nan], "f1": [np.inf, np.nan]})

    with pytest.raises(ValueError, match="No rows remain"):
        ocexport.predict_from_export(export_dir, dataframe)


@pytest.mark.order(295)
def test_predict_from_export_dudez_transfer_override(tmp_path):
    input_size = 3
    features = [f"f{i}" for i in range(input_size)]
    extractor = ocstaged.FeatureExtractor(
        input_size=input_size,
        hidden_sizes=[4, 3],
        latent_dim=2,
        activation="GELU",
        dropout=0.0,
        projection_dim=0,
    )
    params = {
        "dudez_use_transfer": True,
        "dudez_fine_tuning_mode": "partial",
        "dudez_num_unfrozen_layers": 1,
        "dudez_classifier_hidden_size": 4,
        "dudez_classifier_dropout": 0.0,
        "dudez_classifier_activation": "GELU",
        "dudez_use_class_weighting": True,
        "optimizer_learning_rate": 1e-3,
        "optimizer_weight_decay": 1e-4,
        "optimizer_batch_size": 8,
    }
    model = ocstaged.build_dudez_model(
        input_size=input_size,
        params=params,
        transferred_extractor=extractor,
    )
    dudez_dir = tmp_path / "dudez_best_model"
    ocexport.export_best_model_bundle(
        export_dir=dudez_dir,
        task="dudez_screening",
        model=model,
        model_config=params,
        selected_features=features,
        best_trial_number=1,
        best_objective_value=0.5,
        validation_metrics={"BEDROC": 0.5},
        test_metrics={"BEDROC": 0.4},
        stage_config={"epochs": 1},
        splits={
            "train_indices": np.array([0, 1]),
            "validation_indices": np.array([2]),
            "test_indices": np.array([3]),
            "split_config": {},
            "split_diagnostics": {},
        },
        objective_metric="BEDROC",
        direction="maximize",
        best_params={"optimizer_batch_size": 8},
        random_seed=0,
        validate=False,
        extra={"pdbbind_best_model_export_dir": str(tmp_path / "missing_pdbbind")},
    )

    pdbbind_dir = _build_pdbbind_export(
        tmp_path / "linked",
        features,
        hidden_sizes=[4, 3],
        latent_dim=2,
    )
    dataframe = _feature_dataframe(features, rows=2)
    dataframe["kind"] = ["ligands", "decoys"]

    output = ocexport.predict_from_export(
        dudez_dir,
        dataframe,
        pdbbind_export_dir=pdbbind_dir,
    )
    assert "ocscore_prediction" in output.columns
    assert "ocscore_probability" in output.columns
    assert len(output) == 2


@pytest.mark.order(296)
def test_predict_from_export_dudez_missing_pdbbind_path_raises(tmp_path):
    input_size = 2
    features = [f"f{i}" for i in range(input_size)]
    extractor = ocstaged.FeatureExtractor(
        input_size=input_size,
        hidden_sizes=[4, 3],
        latent_dim=2,
        activation="GELU",
        dropout=0.0,
        projection_dim=0,
    )
    params = {
        "dudez_use_transfer": True,
        "dudez_fine_tuning_mode": "partial",
        "dudez_num_unfrozen_layers": 1,
        "dudez_classifier_hidden_size": 4,
        "dudez_classifier_dropout": 0.0,
        "dudez_classifier_activation": "GELU",
        "dudez_use_class_weighting": True,
        "optimizer_learning_rate": 1e-3,
        "optimizer_weight_decay": 1e-4,
        "optimizer_batch_size": 8,
    }
    model = ocstaged.build_dudez_model(
        input_size=input_size,
        params=params,
        transferred_extractor=extractor,
    )
    dudez_dir = tmp_path / "dudez_only"
    ocexport.export_best_model_bundle(
        export_dir=dudez_dir,
        task="dudez_screening",
        model=model,
        model_config=params,
        selected_features=features,
        best_trial_number=1,
        best_objective_value=0.5,
        validation_metrics={"BEDROC": 0.5},
        test_metrics={"BEDROC": 0.4},
        stage_config={"epochs": 1},
        splits={
            "train_indices": np.array([0]),
            "validation_indices": np.array([1]),
            "test_indices": np.array([2]),
            "split_config": {},
            "split_diagnostics": {},
        },
        objective_metric="BEDROC",
        direction="maximize",
        best_params={"optimizer_batch_size": 8},
        random_seed=0,
        validate=False,
        extra={"pdbbind_best_model_export_dir": str(tmp_path / "missing")},
    )

    with pytest.raises(ValueError, match="linked PDBbind export directory"):
        ocexport.predict_from_export(dudez_dir, _feature_dataframe(features, rows=1))
