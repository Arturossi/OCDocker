#!/usr/bin/env python3

# Description
###############################################################################
'''Loss utilities for the future Autoencoder pipeline.'''

# Imports
###############################################################################

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Methods
###############################################################################
def reconstruction_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: Literal["mse", "rmse", "mae", "huber"] = "mse",
        huber_delta: float = 1.0
    ) -> torch.Tensor:
    '''Compute reconstruction loss between prediction and target.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted reconstruction tensor.
    target : torch.Tensor
        Target reconstruction tensor.
    loss_type : Literal["mse", "rmse", "mae", "huber"], optional
        Reconstruction loss type, by default "mse".
    huber_delta : float, optional
        Delta parameter for Huber loss, by default 1.0.

    Returns
    -------
    torch.Tensor
        Scalar reconstruction loss.
    '''

    if loss_type == "mae":
        return nn.L1Loss()(pred, target)
    if loss_type == "huber":
        # Huber is robust to outliers while remaining differentiable.
        return nn.SmoothL1Loss(beta=huber_delta)(pred, target)
    if loss_type == "rmse":
        # RMSE preserves original unit scale.
        mse = nn.MSELoss()(pred, target)
        return torch.sqrt(mse + 1e-8)
    return nn.MSELoss()(pred, target)


def energy_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: Literal["mse", "rmse", "mae", "huber"] = "huber",
        huber_delta: float = 1.0
    ) -> torch.Tensor:
    '''Compute energy regression loss.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted energies.
    target : torch.Tensor
        Target energies.
    loss_type : Literal["mse", "rmse", "mae", "huber"], optional
        Energy loss type, by default "huber".
    huber_delta : float, optional
        Delta parameter for Huber loss, by default 1.0.

    Returns
    -------
    torch.Tensor
        Scalar energy loss.
    '''

    return reconstruction_loss(pred, target, loss_type=loss_type, huber_delta=huber_delta)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    '''Compute KL divergence for VAE (mean over batch).

    Parameters
    ----------
    mu : torch.Tensor
        Latent mean tensor.
    logvar : torch.Tensor
        Latent log-variance tensor.

    Returns
    -------
    torch.Tensor
        Scalar KL divergence loss.
    '''

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # Mean over batch to keep scale stable across batch sizes.
    return kld.mean()


def contractive_penalty(embeddings: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    '''Compute contractive penalty (Jacobian norm).

    Parameters
    ----------
    embeddings : torch.Tensor
        Latent embeddings (N, D).
    inputs : torch.Tensor
        Inputs with requires_grad=True (N, F).

    Returns
    -------
    torch.Tensor
        Scalar contractive penalty.
    '''

    if embeddings is None or inputs is None:
        return torch.tensor(0.0, device=inputs.device if inputs is not None else None)

    penalty = torch.tensor(0.0, device=inputs.device)
    for i in range(embeddings.shape[1]):
        # Accumulate per-latent Jacobian norm (contractive penalty).
        grad = torch.autograd.grad(
            embeddings[:, i].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        penalty = penalty + (grad.pow(2).sum(dim=1)).mean()

    return penalty
