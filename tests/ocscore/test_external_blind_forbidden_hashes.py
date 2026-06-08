#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for automatic forbidden-hash protection in external blind evaluation.'''

# Imports
###############################################################################
import numpy as np
import pandas as pd
import pytest

import OCDocker.OCScore.Optimization.ModelExport as ocexport
import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged
from OCDocker.OCScore.Utils.ContentHash import hash_dataframe_partition
from OCDocker.OCScore.Utils.ExternalBlindEvaluation import ExternalBlindConfig
from OCDocker.OCScore.Utils.ExternalBlindEvaluation import run_external_blind_evaluation

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


def _build_export_with_forbidden_hashes(tmp_path, features: list[str]):
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
    source_df = pd.DataFrame(
        {
            "name": [f"c{i}" for i in range(8)],
            "receptor": ["r1"] * 8,
            "experimental": np.linspace(1.0, 2.0, 8),
            **{feature: np.linspace(0.1, 0.8, 8) for feature in features},
        }
    )
    splits = {
        "X_train": source_df.iloc[:4][features].to_numpy(dtype=np.float32),
        "y_train": source_df.iloc[:4]["experimental"].to_numpy(dtype=np.float32),
        "X_val": source_df.iloc[4:6][features].to_numpy(dtype=np.float32),
        "y_val": source_df.iloc[4:6]["experimental"].to_numpy(dtype=np.float32),
        "X_test": source_df.iloc[6:8][features].to_numpy(dtype=np.float32),
        "y_test": source_df.iloc[6:8]["experimental"].to_numpy(dtype=np.float32),
        "train_indices": np.arange(4),
        "validation_indices": np.arange(4, 6),
        "test_indices": np.arange(6, 8),
        "split_config": {"target_column": "experimental"},
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
        source_dataframe=source_df,
    )
    train_hash = hash_dataframe_partition(source_df, np.arange(4))
    return export_dir, source_df, train_hash


@pytest.mark.order(440)
def test_export_bundle_stores_forbidden_hashes(tmp_path):
    features = [f"f{i}" for i in range(3)]
    export_dir, _, train_hash = _build_export_with_forbidden_hashes(tmp_path, features)
    bundle = ocexport.load_exported_model(export_dir)
    forbidden = bundle["feature_metadata"].get("forbidden_evaluation_dataset_hashes") or []
    assert train_hash in forbidden


@pytest.mark.order(441)
def test_external_blind_rejects_matching_training_partition_hash(tmp_path):
    features = [f"f{i}" for i in range(3)]
    export_dir, source_df, train_hash = _build_export_with_forbidden_hashes(tmp_path, features)
    blind_path = tmp_path / "blind.csv"
    source_df.iloc[:4].to_csv(blind_path, index=False)
    blind_hash = hash_dataframe_partition(source_df, np.arange(4))
    assert blind_hash == train_hash
    with pytest.raises(ValueError, match="forbidden training/validation/test"):
        run_external_blind_evaluation(
            ExternalBlindConfig(
                export_dir=export_dir,
                blind_csv=blind_path,
                output_dir=tmp_path / "blind_eval",
            )
        )
