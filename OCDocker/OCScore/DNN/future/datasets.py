#!/usr/bin/env python3

# Description
###############################################################################
'''Datasets and samplers for the future DNN pipeline.'''

# Imports
###############################################################################
from __future__ import annotations

import math
import random
import torch

import numpy as np

from torch.utils.data import Dataset, Sampler
from typing import Dict, List, Sequence, Union

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
class EnergyDataset(Dataset):
    """Dataset for regression targets (energy labels).

    Parameters
    ----------
    features : np.ndarray
        Input features.
    energies : np.ndarray
        Regression targets (e.g., energies).
    mask : np.ndarray | None, optional
        Feature mask for single-branch inputs.

    Notes
    -----
    Returns (features, energy) where energy has shape (1,).

    Examples
    --------
    >>> import numpy as np
    >>> from OCDocker.OCScore.DNN.future.datasets import EnergyDataset
    >>> features = np.random.rand(100, 20)  # 100 samples, 20 features each
    >>> energies = np.random.rand(100)      # 100 energy labels
    >>> mask = np.random.randint(0, 2, size=(100, 20))  # Random binary mask
    >>> dataset = EnergyDataset(features, energies, mask)
    >>> sample_features, sample_energy = dataset[0]
    >>> print(sample_features.shape)  # torch.Size([20])
    >>> print(sample_energy.shape)    # torch.Size([1])
    """

    def __getitem__(self, idx: int) -> tuple:
        '''Return a dataset sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            Features and energy target tensors.
        '''

        return self.features[idx], self.energies[idx]

    def __init__(
            self,
            features: np.ndarray,
            energies: np.ndarray,
            mask: Union[np.ndarray, None] = None
        ) -> None:
        '''Initialize energy dataset.

        Parameters
        ----------
        features : np.ndarray
            Input features.
        energies : np.ndarray
            Energy targets.
        mask : np.ndarray | None, optional
            Feature mask, by default None.
        '''

        feat = np.asarray(features)
        if mask is not None:
            # Apply feature mask once to avoid per-batch overhead.
            feat = feat * mask
        self.features = torch.tensor(feat, dtype=torch.float32)

        # Ensure regression targets have shape [N, 1] for consistent batching.
        self.energies = torch.tensor(np.asarray(energies), dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        '''Return dataset length.

        Returns
        -------
        int
            Number of samples.
        '''

        return int(self.features.shape[0])


class TargetRankingDataset(Dataset):
    """Dataset for ranking with per-target grouping.

    Parameters
    ----------
    features : np.ndarray
        Input features.
    labels : np.ndarray
        Binary labels (1 for active, 0 for decoy).
    target_ids : Sequence[str]
        Target identifiers per sample (used for grouping).
    mask : np.ndarray | None, optional
        Feature mask for single-branch inputs.

    Notes
    -----
    Returns (features, label, target_id) where target_id is an integer index.
    Target ids are stable based on first appearance order in target_ids.
    """

    def __getitem__(self, idx: int) -> tuple:
        '''Return a dataset sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            Features, label, and target id.
        '''

        # Return raw tensors; model heads apply any normalization.
        return self.features[idx], self.labels[idx], self.target_ids[idx]

    def __init__(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            target_ids: Sequence[str],
            mask: Union[np.ndarray, None] = None
        ) -> None:
        '''Initialize target ranking dataset.

        Parameters
        ----------
        features : np.ndarray
            Input features.
        labels : np.ndarray
            Binary labels.
        target_ids : Sequence[str]
            Target identifiers.
        mask : np.ndarray | None, optional
            Feature mask, by default None.
        '''

        if len(target_ids) != len(labels):
            raise ValueError("target_ids must have the same length as labels")

        feat = np.asarray(features)
        if mask is not None:
            # Mask is applied once to keep dataset sampling fast.
            feat = feat * mask
        self.features = torch.tensor(feat, dtype=torch.float32)

        self.labels = torch.tensor(np.asarray(labels), dtype=torch.float32)

        # Map target identifiers to integers for efficient grouping
        # dict.fromkeys preserves order of first appearance for stable target ids.
        unique_targets = list(dict.fromkeys(target_ids))
        self.target_to_index = {t: i for i, t in enumerate(unique_targets)}
        self.target_ids = np.array([self.target_to_index[t] for t in target_ids], dtype=int)

        # Precompute target -> indices mapping
        self.target_to_indices: Dict[int, List[int]] = {}
        for idx, t in enumerate(self.target_ids):
            # Cache indices per target for fast grouped sampling.
            self.target_to_indices.setdefault(int(t), []).append(idx)

    def __len__(self) -> int:
        '''Return dataset length.

        Returns
        -------
        int
            Number of samples.
        '''

        return int(self.features.shape[0])


class TargetBatchSampler(Sampler[List[int]]):
    """Sampler that yields batches grouped by target.

    Parameters
    ----------
    target_to_indices : dict[int, list[int]]
        Mapping from target id to list of indices.
    batch_size : int | None, optional
        If provided, limits batch size per target. If None, uses full target.
    shuffle : bool, optional
        Shuffle target order each epoch. Default True.
    split_target_batches : bool, optional
        If True, split each target into multiple batches of size batch_size.
        If False, sample a single batch per target. Default False.

    Notes
    -----
    This sampler groups indices by target id to preserve per-target ranking
    structure during training.
    
    Examples
    --------
    >>> from OCDocker.OCScore.DNN.future.datasets import TargetBatchSampler
    >>> target_to_indices = {
    ...     0: [0, 1, 2, 3],
    ...     1: [4, 5, 6],
    ...     2: [7, 8]
    ... }
    >>> sampler = TargetBatchSampler(target_to_indices, batch_size=2, shuffle=False,
    ...                              split_target_batches=True)
    >>> for batch in sampler:
    ...     print(batch)
    [0, 1]
    [2, 3]
    [4, 5]
    [6]
    [7, 8]
    """

    def __init__(
            self,
            target_to_indices: Dict[int, List[int]],
            batch_size: Union[int, None] = None,
            shuffle: bool = True,
            split_target_batches: bool = False
        ) -> None:
        '''Initialize target batch sampler.

        Parameters
        ----------
        target_to_indices : dict[int, list[int]]
            Mapping from target id to indices.
        batch_size : int | None, optional
            Maximum batch size per target, by default None.
        shuffle : bool, optional
            Shuffle targets each epoch, by default True.
        split_target_batches : bool, optional
            Split targets into multiple batches, by default False.
        '''

        self.target_to_indices = target_to_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_target_batches = split_target_batches

        self.target_ids = list(self.target_to_indices.keys())

    def __iter__(self):
        '''Yield batches grouped by target.

        Yields
        ------
        list[int]
            Indices for a batch.
        '''

        target_ids = self.target_ids[:]
        if self.shuffle:
            # Shuffle targets to avoid ordering bias across epochs.
            random.shuffle(target_ids)

        for target_id in target_ids:
            indices = self.target_to_indices[target_id]

            if self.batch_size is None or self.batch_size >= len(indices):
                yield indices
                continue

            if not self.split_target_batches:
                # Sample a single subset for this target to keep batches compact.
                yield random.sample(indices, self.batch_size)
            else:
                # Split into multiple batches for long target lists.
                shuffled = indices[:]
                random.shuffle(shuffled)
                for i in range(0, len(shuffled), self.batch_size):
                    yield shuffled[i:i + self.batch_size]

    def __len__(self) -> int:
        '''Return number of batches.

        Returns
        -------
        int
            Number of batches per epoch.
        '''

        if self.batch_size is None or not self.split_target_batches:
            return len(self.target_ids)

        # Approximate number of batches when splitting targets
        total = 0
        for idxs in self.target_to_indices.values():
            total += int(math.ceil(len(idxs) / float(self.batch_size)))
        return total


# Functions
###############################################################################
## Private ##

## Public ##
