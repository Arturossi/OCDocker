#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore future dimensionality utilities and losses.
'''

# Imports
###############################################################################
import numpy as np

import pytest

torch = pytest.importorskip("torch")

import OCDocker.OCScore.Dimensionality.future.losses as ocfutureloss
import OCDocker.OCScore.Dimensionality.future.utils as ocfutureutils

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

## Public ##

@pytest.mark.order(430)
def test_future_dimensionality_losses_paths():
    pred = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    target = torch.tensor([[0.0], [2.5]], dtype=torch.float32)

    mse = ocfutureloss.reconstruction_loss(pred, target, loss_type="mse")
    mae = ocfutureloss.reconstruction_loss(pred, target, loss_type="mae")
    huber = ocfutureloss.reconstruction_loss(pred, target, loss_type="huber", huber_delta=0.5)
    rmse = ocfutureloss.reconstruction_loss(pred, target, loss_type="rmse")

    assert mse.item() >= 0.0
    assert mae.item() >= 0.0
    assert huber.item() >= 0.0
    assert rmse.item() >= 0.0

    energy = ocfutureloss.energy_loss(pred, target, loss_type="mae")
    assert torch.isclose(energy, mae)

    mu = torch.zeros((2, 3), dtype=torch.float32)
    logvar = torch.zeros((2, 3), dtype=torch.float32)
    kld = ocfutureloss.kl_divergence(mu, logvar)
    assert kld.item() == pytest.approx(0.0, abs=1e-7)

    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    embeddings = inputs[:, :1] * 2.0
    penalty = ocfutureloss.contractive_penalty(embeddings, inputs)
    assert penalty.item() > 0.0

    none_penalty = ocfutureloss.contractive_penalty(None, inputs)
    assert none_penalty.item() == pytest.approx(0.0)


@pytest.mark.order(431)
def test_future_dimensionality_utils_noise_and_stats(monkeypatch):
    x = torch.ones((3, 2), dtype=torch.float32)

    out_none = ocfutureutils.apply_noise(x, noise_type="none")
    assert torch.allclose(out_none, x)

    torch.manual_seed(0)
    out_gaussian = ocfutureutils.apply_noise(x, noise_type="gaussian", gaussian_std=0.25)
    assert not torch.allclose(out_gaussian, x)

    out_mask = ocfutureutils.apply_noise(x, noise_type="mask", mask_prob=1.0)
    assert torch.count_nonzero(out_mask).item() == 0

    single = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    out_single_swap = ocfutureutils.apply_noise(single, noise_type="swap", swap_prob=1.0)
    assert torch.allclose(out_single_swap, single)

    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.tensor([1, 2, 0], device=device))
    out_swap = ocfutureutils.apply_noise(
        torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
        noise_type="swap",
        swap_prob=1.0,
    )
    assert torch.equal(out_swap, torch.tensor([[2.0], [3.0], [1.0]]))

    empty_stats = ocfutureutils.embedding_stats(np.zeros((0, 0), dtype=float))
    assert empty_stats["variance"] == []
    assert empty_stats["collapse_rate"] == 0.0
    assert empty_stats["mean_norm"] == 0.0

    emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float)
    stats = ocfutureutils.embedding_stats(emb)
    assert stats["collapse_rate"] == pytest.approx(1.0)
    assert stats["mean_norm"] > 0.0


@pytest.mark.order(432)
def test_future_dimensionality_utils_ramp_and_spearman():
    assert ocfutureutils.ramp_weight(2.0, epoch=0, ramp_epochs=0, ramp_type="linear") == pytest.approx(2.0)
    assert ocfutureutils.ramp_weight(2.0, epoch=0, ramp_epochs=4, ramp_type="linear") == pytest.approx(0.5)
    assert ocfutureutils.ramp_weight(2.0, epoch=10, ramp_epochs=4, ramp_type="linear") == pytest.approx(2.0)

    sig = ocfutureutils.ramp_weight(3.0, epoch=1, ramp_epochs=4, ramp_type="sigmoid")
    assert 0.0 < sig < 3.0

    assert ocfutureutils.spearman_corr(np.array([]), np.array([])) == 0.0
    assert ocfutureutils.spearman_corr(np.array([1.0, 2.0]), np.array([1.0])) == 0.0

    inc = ocfutureutils.spearman_corr(np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0]))
    dec = ocfutureutils.spearman_corr(np.array([1.0, 2.0, 3.0]), np.array([5.0, 4.0, 3.0]))
    assert inc == pytest.approx(1.0, abs=1e-6)
    assert dec == pytest.approx(-1.0, abs=1e-6)
