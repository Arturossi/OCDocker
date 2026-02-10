#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Dimensionality.AutoencoderOptimizer helpers.
'''

# Imports
###############################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pytest

from torch.utils.data import DataLoader

import OCDocker.OCScore.Dimensionality.AutoencoderOptimizer as ocae

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


class _SuggestTrial:
    def __init__(self):
        self.attrs = {}
        self.number = 0

    def suggest_float(self, _name, low, _high):
        return low

    def suggest_int(self, _name, low, _high):
        return low

    def suggest_categorical(self, _name, choices):
        return choices[0]

    def report(self, _value, _step):
        return None

    def should_prune(self):
        return False

    def set_user_attr(self, key, value):
        self.attrs[key] = value


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(424)
def test_autoencoder_dataset_and_module_accessors():
    feats = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    ds = ocae.AutoencoderDataset(feats)
    assert len(ds) == 2
    x0, y0 = ds[0]
    assert torch.equal(x0, feats[0])
    assert torch.equal(y0, feats[0])

    model = ocae.Autoencoder(
        input_size=2,
        encoding_dim=2,
        encoder_activation_fn=nn.ReLU(),
        decoder_activation_fn=nn.Identity(),
        decoding_dim=[2],
        device=torch.device("cpu"),
    )
    model.eval()
    out = model(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert out.shape == (1, 2)
    assert model.get_decoder_topology() == ["Linear"]
    assert model.get_encoder_topology() == ["Linear", "BatchNorm1d"]


@pytest.mark.order(425)
def test_autoencoder_rejects_non_list_encoding_dim_when_encoder_config_is_list():
    with pytest.raises(ValueError, match="encoding_dim should be a list"):
        _ = ocae.Autoencoder(
            input_size=2,
            encoding_dim=2,  # type: ignore[arg-type]
            encoder_activation_fn=[(nn.ReLU, {})],
            decoder_activation_fn=[(nn.ReLU, {})],
            decoding_dim=[2],
            device=torch.device("cpu"),
        )


@pytest.mark.order(426)
def test_autoencoder_optimizer_evaluate_and_prune_branches():
    x_train = np.zeros((6, 2), dtype=np.float32)
    x_test = np.zeros((6, 2), dtype=np.float32)
    opt = ocae.AutoencoderOptimizer(x_train, x_test, use_gpu=False, verbose=False)
    criterion = nn.MSELoss()

    with pytest.raises(ValueError, match="No DataLoader available"):
        _ = opt.evaluate_autoencoder(nn.Identity(), criterion)

    test_ds = ocae.AutoencoderDataset(torch.zeros((6, 2), dtype=torch.float32))
    opt.test_loader = DataLoader(test_ds, batch_size=2)
    rmse = opt.evaluate_autoencoder(nn.Identity(), criterion)
    assert rmse == pytest.approx(0.0)

    train_ds = ocae.AutoencoderDataset(torch.zeros((6, 2), dtype=torch.float32))
    opt.train_loader = DataLoader(train_ds, batch_size=2)
    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ocae.optuna.exceptions.TrialPruned):
        _ = opt.train_autoencoder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            clip_grad=0.5,
            epochs=1,
            trial=_PruneTrial(),
        )


@pytest.mark.order(427)
def test_autoencoder_optimizer_objective_dataset_guard_and_optimize(monkeypatch):
    x_train = np.zeros((8, 2), dtype=np.float32)
    x_test = np.zeros((8, 2), dtype=np.float32)
    opt = ocae.AutoencoderOptimizer(x_train, x_test, use_gpu=False, verbose=True)
    opt.train_dataset = None
    opt.test_dataset = None

    with pytest.raises(ValueError, match="train_dataset and test_dataset"):
        _ = opt.objective(_SuggestTrial())

    class _Study:
        def __init__(self):
            self.best_trial = type("BestTrial", (), {"value": 0.12, "params": {"lr": 1e-4}})()
            self.optimize_calls = []

        def optimize(self, objective, n_trials, n_jobs):
            _ = objective
            self.optimize_calls.append((n_trials, n_jobs))

    fake_study = _Study()
    print_calls = []
    monkeypatch.setattr(ocae.ocprint, "printv", lambda msg: print_calls.append(msg))
    monkeypatch.setattr(ocae.optuna, "create_study", lambda **kwargs: fake_study)

    study = opt.optimize(
        direction="minimize",
        n_trials=2,
        study_name="ae_unit",
        load_if_exists=False,
        n_jobs=1,
    )
    assert study is fake_study
    assert fake_study.optimize_calls == [(2, 1)]
    assert print_calls

