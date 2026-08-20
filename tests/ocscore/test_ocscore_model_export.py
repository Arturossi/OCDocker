#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for exported best-model bundles."""

# Imports
###############################################################################
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import OCDocker.OCScore.Optimization.ModelExport as ocexport
import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""


# Functions
###############################################################################
## Public ##

@pytest.mark.order(270)
def test_export_and_reload_pdbbind_bundle(tmp_path):
    input_size = 6
    params = {
        "encoder_architecture_index": 0,
        "encoder_hidden_sizes": [8, 4],
        "encoder_latent_dim": 3,
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
    features = [f"f{i}" for i in range(input_size)]
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
    splits["scaler"] = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler().fit(splits["X_train"])

    export_dir = tmp_path / "best_model"
    feature_policy = {
        "feature_policy_name": "no_pmi",
        "excluded_features_found": ["ligand_PMI1", "ligand_PMI2", "ligand_PMI3"],
        "selected_features_after_train_only_reduction": features,
    }
    paths = ocexport.export_best_model_bundle(
        export_dir=export_dir,
        task="pdbbind_regression",
        model=model,
        model_config=params,
        selected_features=features,
        best_trial_number=3,
        best_objective_value=1.23,
        validation_metrics={"RMSE": 1.23},
        test_metrics={"RMSE": 1.45},
        stage_config={"epochs": 2, "n_trials": 3},
        splits=splits,
        objective_metric="RMSE",
        direction="minimize",
        best_params={"optimizer_batch_size": 4},
        random_seed=42,
        extra={"feature_policy": feature_policy},
    )

    assert (export_dir / "best_model.pt").exists()
    assert (export_dir / "architecture.json").exists()
    assert (export_dir / "retrain_config.json").exists()
    assert (export_dir / "best_trial_summary.json").exists()
    assert (export_dir / "feature_metadata.json").exists()
    assert (export_dir / "scaler.joblib").exists()
    assert paths["best_model_path"].endswith("best_model.pt")

    validation = ocexport.validate_export_bundle(export_dir)
    assert validation["validated"] is True

    bundle = ocexport.load_exported_model(export_dir)
    assert bundle["selected_features"] == features
    assert bundle["feature_metadata"]["feature_policy"]["feature_policy_name"] == "no_pmi"
    retrain_config = json.loads((export_dir / "retrain_config.json").read_text(encoding="utf-8"))
    assert retrain_config["extra"]["feature_policy"]["excluded_features_found"] == ["ligand_PMI1", "ligand_PMI2", "ligand_PMI3"]
    summary = json.loads((export_dir / "best_trial_summary.json").read_text(encoding="utf-8"))
    assert summary["feature_policy"]["feature_policy_name"] == "no_pmi"
    assert isinstance(bundle["model"], ocstaged.PDBbindRegressionModel)

    architecture = json.loads((export_dir / "architecture.json").read_text(encoding="utf-8"))
    assert architecture["task"] == "pdbbind_regression"
    assert architecture["encoder"]["hidden_sizes"] == [8, 4]


@pytest.mark.order(271)
def test_export_and_reload_dudez_bundle(tmp_path):
    input_size = 5
    extractor = ocstaged.FeatureExtractor(
        input_size=input_size,
        hidden_sizes=[6, 4],
        latent_dim=3,
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
    features = [f"f{i}" for i in range(input_size)]
    splits = {
        "train_indices": np.array([0, 1, 2, 3]),
        "validation_indices": np.array([4, 5]),
        "test_indices": np.array([6, 7]),
        "split_config": {"validation_size": 0.2, "test_size": 0.2},
        "split_diagnostics": {},
    }
    export_dir = tmp_path / "dudez_best_model"
    ocexport.export_best_model_bundle(
        export_dir=export_dir,
        task="dudez_screening",
        model=model,
        model_config=params,
        selected_features=features,
        best_trial_number=1,
        best_objective_value=0.75,
        validation_metrics={"BEDROC": 0.75},
        test_metrics={"BEDROC": 0.7},
        stage_config={"epochs": 2, "primary_metric": "BEDROC"},
        splits=splits,
        objective_metric="BEDROC",
        direction="maximize",
        best_params={"optimizer_batch_size": 8},
        random_seed=7,
        validate=False,
    )

    bundle = ocexport.load_exported_model(
        export_dir,
        transferred_extractor=extractor,
    )
    assert isinstance(bundle["model"], ocstaged.DUDEzScreeningModel)
    assert bundle["scaler"] is None


@pytest.mark.order(272)
def test_dudez_bundle_falls_back_to_pdbbind_scaler(tmp_path):
    # DUDEz bundles never persist their own scaler.joblib; scoring must fall
    # back to the linked PDBbind bundle's scaler instead of using raw features.
    import sklearn.preprocessing

    input_size = 5

    pdbbind_params = {
        "encoder_architecture_index": 0,
        "encoder_hidden_sizes": [6, 4],
        "encoder_latent_dim": 3,
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
    pdbbind_model = ocstaged.build_pdbbind_model(input_size=input_size, params=pdbbind_params)
    features = [f"f{i}" for i in range(input_size)]
    rng = np.random.default_rng(1)
    x_train = rng.normal(loc=50.0, scale=10.0, size=(8, input_size)).astype(np.float32)
    pdbbind_splits = {
        "X_train": x_train,
        "y_train": rng.normal(size=8).astype(np.float32),
        "X_val": rng.normal(loc=50.0, scale=10.0, size=(4, input_size)).astype(np.float32),
        "y_val": rng.normal(size=4).astype(np.float32),
        "X_test": rng.normal(loc=50.0, scale=10.0, size=(4, input_size)).astype(np.float32),
        "y_test": rng.normal(size=4).astype(np.float32),
        "train_indices": np.arange(8),
        "validation_indices": np.arange(8, 12),
        "test_indices": np.arange(12, 16),
        "split_config": {"target_column": "experimental", "validation_size": 0.2, "test_size": 0.2},
        "split_diagnostics": {},
    }
    fitted_scaler = sklearn.preprocessing.StandardScaler().fit(x_train)
    pdbbind_splits["scaler"] = fitted_scaler

    pdbbind_dir = tmp_path / "pdbbind_best_model"
    ocexport.export_best_model_bundle(
        export_dir=pdbbind_dir,
        task="pdbbind_regression",
        model=pdbbind_model,
        model_config=pdbbind_params,
        selected_features=features,
        best_trial_number=1,
        best_objective_value=1.0,
        validation_metrics={"RMSE": 1.0},
        test_metrics={"RMSE": 1.0},
        stage_config={"epochs": 2, "n_trials": 1},
        splits=pdbbind_splits,
        objective_metric="RMSE",
        direction="minimize",
        best_params={"optimizer_batch_size": 4},
        random_seed=1,
    )
    assert (pdbbind_dir / "scaler.joblib").exists()

    # DUDEz bundle exported without its own scaler (splits has no "scaler" key)
    dudez_params = {
        "dudez_use_transfer": False,
        "dudez_classifier_hidden_size": 4,
        "dudez_classifier_dropout": 0.0,
        "dudez_classifier_activation": "GELU",
        "dudez_use_class_weighting": True,
        "optimizer_learning_rate": 1e-3,
        "optimizer_weight_decay": 1e-4,
        "optimizer_batch_size": 8,
    }
    dudez_model = ocstaged.build_dudez_model(input_size=input_size, params=dudez_params, transferred_extractor=None)
    dudez_splits = {
        "train_indices": np.array([0, 1, 2, 3]),
        "validation_indices": np.array([4, 5]),
        "test_indices": np.array([6, 7]),
        "split_config": {"validation_size": 0.2, "test_size": 0.2},
        "split_diagnostics": {},
    }
    dudez_dir = tmp_path / "dudez_best_model"
    ocexport.export_best_model_bundle(
        export_dir=dudez_dir,
        task="dudez_screening",
        model=dudez_model,
        model_config=dudez_params,
        selected_features=features,
        best_trial_number=1,
        best_objective_value=0.75,
        validation_metrics={"BEDROC": 0.75},
        test_metrics={"BEDROC": 0.7},
        stage_config={"epochs": 2, "primary_metric": "BEDROC"},
        splits=dudez_splits,
        objective_metric="BEDROC",
        direction="maximize",
        best_params={"optimizer_batch_size": 8},
        random_seed=1,
        validate=False,
    )
    assert not (dudez_dir / "scaler.joblib").exists()

    # Without pdbbind_export_dir: no regression, scaler stays None.
    bundle_no_fallback = ocexport.load_exported_model(dudez_dir)
    assert bundle_no_fallback["scaler"] is None

    # With pdbbind_export_dir: falls back to the PDBbind bundle's fitted scaler.
    bundle_with_fallback = ocexport.load_exported_model(dudez_dir, pdbbind_export_dir=pdbbind_dir)
    assert bundle_with_fallback["scaler"] is not None
    assert isinstance(bundle_with_fallback["scaler"], sklearn.preprocessing.StandardScaler)
    np.testing.assert_allclose(bundle_with_fallback["scaler"].mean_, fitted_scaler.mean_)

    # predict_from_export must forward pdbbind_export_dir into the scaler
    # fallback too, not just into transferred-extractor resolution.
    import pandas as pd
    df = pd.DataFrame(rng.normal(loc=50.0, scale=10.0, size=(3, input_size)), columns=features)
    predictions = ocexport.predict_from_export(dudez_dir, df, pdbbind_export_dir=pdbbind_dir)
    assert "ocscore_prediction" in predictions.columns
    assert len(predictions) == 3
