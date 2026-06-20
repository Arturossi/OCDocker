#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for external blind evaluation mode.'''

# Imports
###############################################################################
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import OCDocker.OCScore.Optimization.ModelExport as ocexport
import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged
from OCDocker.OCScore.Utils.ContentHash import hash_file
from OCDocker.OCScore.Utils.ExternalBlindEvaluation import EXTERNAL_BLIND_EVAL_JSON
from OCDocker.OCScore.Utils.ExternalBlindEvaluation import EXTERNAL_BLIND_PREDICTIONS_CSV
from OCDocker.OCScore.Utils.ExternalBlindEvaluation import ExternalBlindConfig
from OCDocker.OCScore.Utils.ExternalBlindEvaluation import run_external_blind_evaluation

# License
###############################################################################
'''OCDocker
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
'''


def _build_pdbbind_export(tmp_path, features: list[str]):
    input_size = len(features)
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

    export_dir = tmp_path / "best_model"
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


def _blind_csv(tmp_path, features: list[str], *, include_label: bool = True) -> Path:
    rows = {
        "receptor": ["r1", "r2"],
        "name": ["l1", "l2"],
        **{feature: np.linspace(0.1, 0.4, 2) for feature in features},
        "extra_column": [1.0, 2.0],
    }
    if include_label:
        rows["experimental"] = [1.0, 2.0]
    blind_path = tmp_path / "blind.csv"
    pd.DataFrame(rows).to_csv(blind_path, index=False)
    return blind_path


@pytest.mark.order(420)
def test_external_blind_writes_json_and_predictions(tmp_path):
    features = [f"f{i}" for i in range(4)]
    export_dir = _build_pdbbind_export(tmp_path, features)
    blind_path = _blind_csv(tmp_path, features)
    output_dir = tmp_path / "blind_eval"

    report = run_external_blind_evaluation(
        ExternalBlindConfig(
            export_dir=export_dir,
            blind_csv=blind_path,
            output_dir=output_dir,
        )
    )

    eval_path = output_dir / EXTERNAL_BLIND_EVAL_JSON
    predictions_path = output_dir / EXTERNAL_BLIND_PREDICTIONS_CSV
    assert eval_path.is_file()
    assert predictions_path.is_file()
    assert report["validation_mode"] == "external-blind"
    assert report["n_rows"] == 2
    assert report["metrics"] is not None
    assert "RMSE" in report["metrics"]
    assert report["blind_dataset_hash"] == hash_file(blind_path)
    assert "extra_column" in report["extra_features_ignored"]


@pytest.mark.order(421)
def test_external_blind_missing_selected_feature_fails(tmp_path):
    features = [f"f{i}" for i in range(4)]
    export_dir = _build_pdbbind_export(tmp_path, features)
    blind_path = _blind_csv(tmp_path, features[:-1])
    with pytest.raises(ValueError, match="missing required selected features"):
        run_external_blind_evaluation(
            ExternalBlindConfig(
                export_dir=export_dir,
                blind_csv=blind_path,
                output_dir=tmp_path / "out",
            )
        )


@pytest.mark.order(422)
def test_external_blind_labels_missing_predictions_only(tmp_path):
    features = [f"f{i}" for i in range(4)]
    export_dir = _build_pdbbind_export(tmp_path, features)
    blind_path = _blind_csv(tmp_path, features, include_label=False)
    output_dir = tmp_path / "blind_eval_no_labels"

    report = run_external_blind_evaluation(
        ExternalBlindConfig(
            export_dir=export_dir,
            blind_csv=blind_path,
            output_dir=output_dir,
        )
    )
    assert report["metrics"] is None
    assert report["metric_scope"] == "labels_unavailable"


@pytest.mark.order(423)
def test_external_blind_does_not_call_feature_reduction(tmp_path):
    features = [f"f{i}" for i in range(4)]
    export_dir = _build_pdbbind_export(tmp_path, features)
    blind_path = _blind_csv(tmp_path, features)
    with patch("OCDocker.OCScore.Utils.FeatureReduction.run_feature_reduction_protocol") as mock_reduce:
        run_external_blind_evaluation(
            ExternalBlindConfig(
                export_dir=export_dir,
                blind_csv=blind_path,
                output_dir=tmp_path / "out",
            )
        )
    mock_reduce.assert_not_called()


@pytest.mark.order(424)
def test_external_blind_forbidden_dataset_hash_fails(tmp_path):
    features = [f"f{i}" for i in range(4)]
    export_dir = _build_pdbbind_export(tmp_path, features)
    blind_path = _blind_csv(tmp_path, features)
    with pytest.raises(ValueError, match="forbidden training/validation/test"):
        run_external_blind_evaluation(
            ExternalBlindConfig(
                export_dir=export_dir,
                blind_csv=blind_path,
                output_dir=tmp_path / "out",
                forbidden_dataset_hashes=[hash_file(blind_path)],
            )
        )
