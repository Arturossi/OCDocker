#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for future OCScore DNN embeddings flow.

Usage:

pytest tests/test_ocscore_dnn_future_embeddings.py
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
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

def test_future_dnn_from_embeddings_minimal():
    torch = pytest.importorskip("torch")
    from OCDocker.OCScore.Dimensionality.future.Autoencoder import Autoencoder
    from OCDocker.OCScore.DNN.future.DNNOptimizer import DNNOptimizer

    rng = np.random.default_rng(0)

    X = rng.normal(size=(32, 20)).astype(np.float32)
    y = rng.normal(size=(32,)).astype(np.float32)

    ae = Autoencoder(
        input_size=20,
        encoder_hidden_sizes=[16],
        latent_dim=8,
        energy_head_sizes=None
    )

    with torch.no_grad():
        Z = ae.encode(torch.tensor(X, dtype=torch.float32)).cpu().numpy()

    X_train, X_test = Z[:24], Z[24:]
    y_train, y_test = y[:24], y[24:]

    future_config = {
        "model": {
            "shared_sizes": [16],
            "head_sizes": [8],
            "embedding_dim": 4,
            "dropout": 0.0,
            "batch_norm": False
        },
        "stage1": {
            "epochs": 1,
            "batch_size": 8,
            "noise_type": "none",
            "lambda_recon": 0.0
        },
        "stage2": {"enabled": False}
    }

    trainer = DNNOptimizer.from_embeddings(
        X_train,
        y_train,
        X_test,
        y_test,
        storage="sqlite:///:memory:",
        use_gpu=False,
        verbose=False,
        future_config=future_config
    )

    study = trainer.optimize(n_trials=1, n_jobs=1)
    assert study is not None
