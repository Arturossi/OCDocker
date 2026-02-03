#!/usr/bin/env python3

# Description
###############################################################################
'''Loss functions for the future DNN pipeline (ranking, contrastive, weighting).'''

# Imports
###############################################################################
from __future__ import annotations

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Iterable, Tuple

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
class UncertaintyWeighting(nn.Module):
    """Uncertainty-based loss balancing for multi-task objectives.

    Parameters
    ----------
    task_names : Iterable[str]
        Names of tasks to balance. The order defines parameter indexing.
    init_log_var : float, optional
        Initial value for log-variance parameters, by default 0.0.
    
    Examples
    --------
    >>> import torch
    >>> from OCDocker.OCScore.DNN.future.losses import UncertaintyWeighting
    >>> task_names = ["regression", "classification"]
    >>> model = UncertaintyWeighting(task_names)
    >>> losses = {
    ...     "regression": torch.tensor(2.0),
    ...     "classification": torch.tensor(1.0)
    ... }
    >>> total_loss, weights = model(losses)
    >>> print(total_loss)
    tensor(3.6931, grad_fn=<AddBackward0>)
    >>> print(weights)
    {'regression': 0.6065306663513184, 'classification': 1.0}
    """

    def __init__(self, task_names: Iterable[str], init_log_var: float = 0.0) -> None:
        '''Initialize uncertainty weighting module.

        Parameters
        ----------
        task_names : Iterable[str]
            Task names to balance.
        init_log_var : float, optional
            Initial log-variance value, by default 0.0.
        '''

        super(UncertaintyWeighting, self).__init__()

        self.task_names = list(task_names)

        # One log-variance per task
        init = torch.full((len(self.task_names),), float(init_log_var))
        self.log_vars = nn.Parameter(init)


    def forward(self, losses: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        '''Combine losses using learned uncertainty weights.

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Dictionary of task losses.

        Returns
        -------
        tuple[torch.Tensor, Dict[str, float]]
            Combined loss and current weights per task.
        '''

        total = torch.tensor(0.0, device=self.log_vars.device)
        weights = {}

        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            log_var = self.log_vars[i]
            # Precision = exp(-log_var), consistent with Kendall et al. uncertainty weighting.
            precision = torch.exp(-log_var)
            total = total + precision * losses[name] + log_var
            weights[name] = float(precision.detach().cpu().item())

        return total, weights



# Functions
###############################################################################
## Private ##

def _lambda_rank_loss_single_k(scores: torch.Tensor, labels: torch.Tensor, k: float | int) -> torch.Tensor:
    '''LambdaRank loss for a single k (fraction or integer).

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores (N,).
    labels : torch.Tensor
        Binary labels (N,).
    k : float | int
        Fraction or integer defining top-k cutoff.

    Returns
    -------
    torch.Tensor
        LambdaRank loss value for this k.
    '''

    n = labels.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=scores.device)

    k_int = _safe_k(k, n)
    if k_int <= 0:
        return torch.tensor(0.0, device=scores.device)

    # Gains
    gains = torch.pow(2.0, labels) - 1.0

    # Ideal DCG
    sorted_labels, _ = torch.sort(labels, descending=True)
    discounts = 1.0 / torch.log2(torch.arange(2, k_int + 2, device=scores.device, dtype=torch.float32))
    idcg = torch.sum(sorted_labels[:k_int] * discounts)

    if idcg.item() <= 0:
        return torch.tensor(0.0, device=scores.device)

    # Predicted ranks based on model scores (higher is better).
    _, order = torch.sort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, n + 1, device=scores.device, dtype=torch.float32)

    discounts_all = 1.0 / torch.log2(ranks + 1.0)
    discounts_all = discounts_all * (ranks <= k_int).float()

    pos_mask = labels > 0.0
    neg_mask = labels <= 0.0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    s_pos = scores[pos_mask]
    s_neg = scores[neg_mask]

    d_pos = discounts_all[pos_mask]
    d_neg = discounts_all[neg_mask]

    # Delta DCG for swapping pos/neg
    delta = torch.abs(d_pos[:, None] - d_neg[None, :])

    # Pairwise logistic loss
    diff = s_pos[:, None] - s_neg[None, :]
    pair_loss = torch.log1p(torch.exp(-diff))

    loss = (pair_loss * delta).mean() / (idcg + 1e-8)

    return loss


def _safe_k(k: float | int, n: int) -> int:
    '''Convert k fraction or integer into a safe integer within [1, n].

    Parameters
    ----------
    k : float | int
        Fraction (0-1) or absolute integer k.
    n : int
        Total number of items.

    Returns
    -------
    int
        Safe integer k within bounds.
    '''

    if n <= 0:
        return 0
    if isinstance(k, float):
        k = max(1, int(round(k * n)))
    else:
        k = int(k)
    return max(1, min(k, n))


## Public ##

def focal_binary_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ) -> torch.Tensor:
    '''Binary focal loss with logits.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model logits.
    targets : torch.Tensor
        Binary targets (0/1).
    alpha : float, optional
        Balancing factor, by default 0.25.
    gamma : float, optional
        Focusing parameter, by default 2.0.
    reduction : str, optional
        Reduction mode: 'mean', 'sum', or 'none'. Default is 'mean'.

    Returns
    -------
    torch.Tensor
        Focal loss value.
    '''

    targets = targets.float()
    # Sigmoid over logits keeps computation stable for probabilities.
    probs = torch.sigmoid(logits)
    # pt is probability of the true class; focal term downweights easy examples.
    pt = torch.where(targets == 1.0, probs, 1.0 - probs)
    alpha_t = torch.where(targets == 1.0, alpha, 1.0 - alpha)
    loss = -alpha_t * (1.0 - pt).pow(gamma) * torch.log(pt + 1e-8)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()

    return loss


def lambda_rank_ndcg_loss(
        scores: torch.Tensor,
        labels: torch.Tensor,
        k_fractions: Tuple[float, float] = (0.01, 0.05),
        weights: Tuple[float, float] = (0.5, 0.5)
    ) -> torch.Tensor:
    '''LambdaRank-style loss with NDCG@k weighting (top-heavy).

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores (N,).
    labels : torch.Tensor
        Binary labels (N,).
    k_fractions : tuple[float, float], optional
        Fractions of the ranked list to emphasize (e.g., 0.01, 0.05).
    weights : tuple[float, float], optional
        Weights for each k in k_fractions.

    Returns
    -------
    torch.Tensor
        LambdaRank NDCG loss.
    '''

    if scores.ndim > 1:
        scores = scores.view(-1)
    labels = labels.view(-1).float()

    if scores.numel() <= 1:
        return torch.tensor(0.0, device=scores.device)

    total_loss = torch.tensor(0.0, device=scores.device)

    # Aggregate multiple top-k cutoffs to emphasize early recognition.
    for frac, weight in zip(k_fractions, weights):
        total_loss = total_loss + weight * _lambda_rank_loss_single_k(scores, labels, frac)

    return total_loss


def supervised_contrastive_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
    '''Supervised contrastive loss (SupCon) with L2-normalized embeddings.

    Parameters
    ----------
    embeddings : torch.Tensor
        Embeddings of shape (N, D).
    labels : torch.Tensor
        Binary or multiclass labels of shape (N,).
    temperature : float, optional
        Softmax temperature, by default 0.1.

    Returns
    -------
    torch.Tensor
        SupCon loss value.
    '''

    if embeddings is None:
        return torch.tensor(0.0, device=labels.device)

    if embeddings.shape[0] <= 1:
        return torch.tensor(0.0, device=labels.device)

    labels = labels.view(-1)
    device = embeddings.device

    embeddings = F.normalize(embeddings, dim=1)

    # Similarity matrix
    # Temperature controls softness of similarity distribution.
    logits = torch.matmul(embeddings, embeddings.T) / max(temperature, 1e-8)

    # Mask self-contrast cases
    mask = torch.eye(logits.shape[0], device=device, dtype=torch.bool)
    logits = logits.masked_fill(mask, float('-inf'))

    # Positive pairs mask
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_equal = labels_equal & ~mask

    # If no positive pairs, return zero
    if labels_equal.sum() == 0:
        return torch.tensor(0.0, device=device)

    # Log-softmax over rows to form normalized similarities.
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # Mean log-prob of positives
    mean_log_prob_pos = (labels_equal.float() * log_prob).sum(dim=1) / (labels_equal.float().sum(dim=1) + 1e-8)

    loss = -mean_log_prob_pos.mean()

    return loss
