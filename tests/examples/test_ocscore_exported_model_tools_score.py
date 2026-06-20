#!/usr/bin/env python3

# Description
###############################################################################
"""Smoke tests for example 16 score subcommand."""

# Imports
###############################################################################
import argparse
import json

from pathlib import Path

import numpy as np
import pandas as pd
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
## Private ##

def _load_example_module():
    from OCDocker.OCScore.CLI import export_tools

    return export_tools


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


## Public ##

@pytest.mark.order(300)
def test_score_subcommand_writes_predictions_csv(tmp_path, capsys):
    example = _load_example_module()
    features = [f"f{i}" for i in range(4)]
    export_dir = _build_pdbbind_export(tmp_path, features)

    archive_dir = tmp_path / "raw_archive"
    archive_dir.mkdir()
    rows = pd.DataFrame(
        {
            "receptor": ["r1", "r2"],
            "name": ["l1", "l2"],
            "experimental": [1.0, 2.0],
            **{feature: np.linspace(0.1, 0.4, 2) for feature in features},
        }
    )
    rows.to_csv(archive_dir / "pipeline_results.csv", index=False)

    output_csv = tmp_path / "predictions.csv"
    args = argparse.Namespace(
        export_dir=str(export_dir),
        retrain_from=None,
        device="cpu",
        raw_archive=str(archive_dir),
        output_csv=str(output_csv),
        pdbbind_export_dir=None,
        archive_member=None,
    )
    example._cmd_score(args)

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["n_predictions"] == 2
    assert output_csv.is_file()

    written = pd.read_csv(output_csv)
    assert "ocscore_prediction" in written.columns
    assert "receptor" in written.columns


@pytest.mark.order(301)
def test_score_cmd_requires_export_dir():
    example = _load_example_module()
    args = argparse.Namespace(
        export_dir=None,
        retrain_from=None,
        device="cpu",
        raw_archive="archive",
        output_csv="out.csv",
        pdbbind_export_dir=None,
        archive_member=None,
    )
    with pytest.raises(ValueError, match="Provide --export-dir"):
        example._cmd_score(args)
