#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Transformer.TransOptimizer helpers.
'''

# Imports
###############################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pytest

from torch.utils.data import DataLoader

import OCDocker.OCScore.Optimization.legacy.models.transformer.TransOptimizer as octrans

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


class _PruneTrial:
    def report(self, _value, _step):
        return None

    def should_prune(self):
        return True

    def set_user_attr(self, _key, _value):
        return None


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(711)
def test_trans_custom_dataset_and_transformer_model_paths():
    feats = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    target = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    ds = octrans.CustomDataset(feats, target)
    assert len(ds) == 2
    x0, y0 = ds[0]
    assert torch.equal(x0, feats[0])
    assert torch.equal(y0, target[0])

    with pytest.raises(ValueError, match="Unknown initialization function"):
        _ = octrans.TransformerModel(
            input_dim=2,
            d_model=4,
            output_dim=1,
            nhead=1,
            num_encoder_layers=1,
            dim_feedforward=8,
            dropout=0.1,
            init_type="bad_init",
            init_params={},
            random_seed=0,
            device=torch.device("cpu"),
        )

    model = octrans.TransformerModel(
        input_dim=2,
        d_model=4,
        output_dim=1,
        nhead=1,
        num_encoder_layers=1,
        dim_feedforward=8,
        dropout=0.1,
        init_type="zeros",
        init_params={},
        random_seed=0,
        device=torch.device("cpu"),
    )
    out = model(torch.ones((2, 1, 2), dtype=torch.float32))
    assert out.shape == (2, 1, 1)


@pytest.mark.order(712)
def test_trans_optimizer_optimize_and_train_test_prune(monkeypatch):
    x_train = np.zeros((8, 1, 2), dtype=np.float32)
    y_train = np.zeros((8, 1, 1), dtype=np.float32)
    x_test = np.zeros((8, 1, 2), dtype=np.float32)
    y_test = np.zeros((8, 1, 1), dtype=np.float32)

    opt = octrans.TransOptimizer(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        storage="sqlite:///trans_unit.db",
        use_gpu=False,
        verbose=True,
    )

    class _Study:
        def __init__(self):
            self.best_params = {"d_model": 64}
            self.optimize_calls = []

        def optimize(self, objective, n_trials, n_jobs):
            _ = objective
            self.optimize_calls.append((n_trials, n_jobs))

    fake_study = _Study()
    print_calls = []
    monkeypatch.setattr(octrans.LOGGER, "info", lambda msg, *args: print_calls.append(msg % args if args else msg))
    monkeypatch.setattr(octrans.optuna, "create_study", lambda **kwargs: fake_study)

    best = opt.optimize(
        direction="minimize",
        n_trials=3,
        study_name="trans_unit",
        load_if_exists=False,
        n_jobs=1,
    )
    assert best == {"d_model": 64}
    assert fake_study.optimize_calls == [(3, 1)]
    assert print_calls

    model = nn.Linear(2, 1)
    train_ds = octrans.CustomDataset(torch.zeros((4, 2), dtype=torch.float32), torch.zeros((4, 1), dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=1)
    test_loader = DataLoader(train_ds, batch_size=1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    with pytest.raises(octrans.optuna.exceptions.TrialPruned):
        _ = opt.train_test_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip_grad=0.1,
            trial=_PruneTrial(),
            batch_size=1,
            epochs=1,
        )

