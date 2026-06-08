#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for future Autoencoder models and optimizers.
'''

# Imports
###############################################################################
import pytest

import numpy as np

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
def _import_future_autoencoder():
    pytest.importorskip("torch")
    from OCDocker.OCScore.Dimensionality.legacy.Autoencoder import Autoencoder
    from OCDocker.OCScore.Dimensionality.legacy.AutoencoderOptimizer import AutoencoderOptimizer

    return Autoencoder, AutoencoderOptimizer


## Public ##

def test_future_autoencoder_optimizer_minimal():
    Autoencoder, AutoencoderOptimizer = _import_future_autoencoder()
    rng = np.random.default_rng(0)

    X_train = rng.normal(size=(32, 12)).astype(np.float32)
    X_test = rng.normal(size=(16, 12)).astype(np.float32)
    X_val = rng.normal(size=(16, 12)).astype(np.float32)

    y_train = rng.normal(size=(32,)).astype(np.float32)
    y_test = rng.normal(size=(16,)).astype(np.float32)
    y_val = rng.normal(size=(16,)).astype(np.float32)

    future_config = {
        "model": {"encoder_hidden_sizes": [32], "latent_dim": 8, "energy_head_sizes": [16]},
        "stage1": {
            "epochs": 1,
            "batch_size": 8,
            "noise_type": "none",
            "lambda_recon": 1.0,
            "lambda_energy": 0.5
        },
        "stage2": {"enabled": False},
        "checkpoint": {"save_best": False}
    }

    trainer = AutoencoderOptimizer(
        X_train,
        X_test,
        X_val,
        encoding_dims=(4, 16),
        storage="sqlite:///:memory:",
        models_folder="./models/Autoencoder/",
        random_seed=42,
        use_gpu=False,
        verbose=False,
        y_train=y_train,
        y_test=y_test,
        y_validation=y_val,
        future_config=future_config
    )

    study = trainer.optimize(n_trials=1, n_jobs=1)
    assert study is not None
    assert "recon_loss_train" in study.best_trial.user_attrs


def test_future_autoencoder_sanity():
    Autoencoder, _ = _import_future_autoencoder()
    model = Autoencoder(
        input_size=16,
        encoder_hidden_sizes=[32, 16],
        latent_dim=8,
        energy_head_sizes=[16]
    )

    info = model.sanity_check(batch_size=4)
    assert info["input_shape"] == (4, 16)
    assert info["latent_shape"] == (4, 8)
    assert info["reconstruction_shape"] == (4, 16)
