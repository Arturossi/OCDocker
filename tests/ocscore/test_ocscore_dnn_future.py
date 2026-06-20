#!/usr/bin/env python3

# Description
###############################################################################
'''
Sanity checks for the future DNN optimization pipeline.
'''

# Imports
###############################################################################
import pytest

import numpy as np

from OCDocker.OCScore.DNN.future.DNNOptimizer import DNNOptimizer

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    pytest.skip("torch not available", allow_module_level=True)


def test_future_dnn_sanity():
    rng = np.random.default_rng(42)

    X_pdb_train = rng.normal(size=(32, 12)).astype(np.float32)
    y_pdb_train = rng.normal(size=(32,)).astype(np.float32)
    X_pdb_test = rng.normal(size=(16, 12)).astype(np.float32)
    y_pdb_test = rng.normal(size=(16,)).astype(np.float32)

    X_dude = rng.normal(size=(40, 12)).astype(np.float32)
    y_dude = np.array([1] * 10 + [0] * 30, dtype=int)
    targets = np.array(["T1"] * 20 + ["T2"] * 20)

    future_config = {
        "stage1": {"epochs": 1, "batch_size": 16, "lambda_recon": 0.1, "lambda_energy": 1.0},
        "stage2": {"epochs": 1, "batch_size_per_target": None},
        "data": {"dude_validation_fraction": 0.5, "dude_split_by_target": True},
        "optimization": {"multi_objective": False, "objective_metric": "AUC"},
        "separation_targets": targets
    }

    trainer = DNNOptimizer(
        X_pdb_train,
        y_pdb_train,
        X_pdb_test,
        y_pdb_test,
        X_validation=X_dude,
        y_validation=y_dude,
        storage="sqlite:///:memory:",
        random_seed=42,
        use_gpu=False,
        verbose=False,
        future_config=future_config
    )

    study = trainer.optimize(n_trials=1, n_jobs=1)
    assert study is not None
