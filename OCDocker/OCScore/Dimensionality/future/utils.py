#!/usr/bin/env python3

# Description
###############################################################################
'''Utilities for the future Autoencoder pipeline.'''

# Imports
###############################################################################

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

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

# Methods
###############################################################################
def apply_noise(
        inputs: torch.Tensor,
        noise_type: str = "none",
        mask_prob: float = 0.0,
        gaussian_std: float = 0.0,
        swap_prob: float = 0.0
    ) -> torch.Tensor:
    '''Apply input noise for denoising autoencoder training.

    Parameters
    ----------
    inputs : torch.Tensor
        Input batch tensor.
    noise_type : str, optional
        Noise strategy string (e.g., "mask", "gaussian", "swap", combos), by default "none".
    mask_prob : float, optional
        Feature masking probability, by default 0.0.
    gaussian_std : float, optional
        Gaussian noise standard deviation, by default 0.0.
    swap_prob : float, optional
        Swap noise probability, by default 0.0.

    Returns
    -------
    torch.Tensor
        Noised input tensor.
    '''

    if noise_type == "none":
        return inputs

    x = inputs

    if "gaussian" in noise_type and gaussian_std > 0.0:
        # Additive noise perturbs continuous inputs.
        x = x + torch.randn_like(x) * gaussian_std

    if "mask" in noise_type and mask_prob > 0.0:
        # Feature masking simulates missing data and encourages robustness.
        mask = torch.rand_like(x) < mask_prob
        x = x.masked_fill(mask, 0.0)

    if "swap" in noise_type and swap_prob > 0.0:
        batch_size = x.shape[0]
        if batch_size > 1:
            perm = torch.randperm(batch_size, device=x.device)
            swap_mask = torch.rand_like(x) < swap_prob
            swapped = x[perm]
            # Swap features across samples to encourage robustness.
            x = torch.where(swap_mask, swapped, x)

    return x


def ramp_weight(
        target: float,
        epoch: int,
        ramp_epochs: int,
        ramp_type: str = "linear"
    ) -> float:
    '''Compute ramped weight value for a given epoch.

    Parameters
    ----------
    target : float
        Final target weight.
    epoch : int
        Current epoch index (0-based).
    ramp_epochs : int
        Number of epochs to ramp over.
    ramp_type : str, optional
        Ramp schedule type ("linear" or "sigmoid"), by default "linear".

    Returns
    -------
    float
        Ramped weight for the given epoch.
    '''

    if ramp_epochs <= 0:
        return float(target)

    progress = min(1.0, max(0.0, float(epoch + 1) / float(ramp_epochs)))

    if ramp_type == "sigmoid":
        # Sigmoid ramp grows slowly then saturates.
        return float(target) * (1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5))))

    return float(target) * progress


def _rankdata(values: np.ndarray) -> np.ndarray:
    '''Compute rank data for Spearman correlation.

    Parameters
    ----------
    values : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Rank-transformed array.
    '''

    order = np.argsort(values)
    # Ranks are zero-based; ties are not specially handled (consistent ordering).
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    '''Compute Spearman correlation (rank-based Pearson).

    Parameters
    ----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.

    Returns
    -------
    float
        Spearman correlation coefficient.
    '''

    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size != y.size:
        return 0.0

    rx = _rankdata(x)
    ry = _rankdata(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)) + 1e-8
    return float(np.sum(rx * ry) / denom)


def embedding_stats(embeddings: np.ndarray, collapse_threshold: float = 1e-6) -> Dict[str, object]:
    '''Compute basic embedding statistics.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix (N, D).
    collapse_threshold : float, optional
        Variance threshold to define collapsed dimensions, by default 1e-6.

    Returns
    -------
    Dict[str, object]
        Dictionary with variance, collapse rate, and mean norm.
    '''

    if embeddings.size == 0:
        return {
            "variance": [],
            "collapse_rate": 0.0,
            "mean_norm": 0.0
        }

    var = np.var(embeddings, axis=0)
    # Collapse rate captures fraction of near-zero variance dimensions.
    collapse_rate = float(np.mean(var < collapse_threshold))
    mean_norm = float(np.mean(np.linalg.norm(embeddings, axis=1)))

    return {
        "variance": var.tolist(),
        "collapse_rate": collapse_rate,
        "mean_norm": mean_norm
    }

