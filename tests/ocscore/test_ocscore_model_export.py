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
"""OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
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
