#!/usr/bin/env python3

# Description
###############################################################################
'''Datasets and samplers for the future DNN pipeline.'''

# Imports
###############################################################################

from __future__ import annotations

import random
import math
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

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

# Classes
###############################################################################


class EnergyDataset(Dataset):
    '''Dataset for regression targets (energy labels).

    Parameters
    ----------
    features : np.ndarray
        Input features.
    energies : np.ndarray
        Regression targets (e.g., energies).
    mask : np.ndarray | None, optional
        Feature mask for single-branch inputs.
    '''

    def __init__(
            self,
            features: np.ndarray,
            energies: np.ndarray,
            mask: Union[np.ndarray, None] = None
        ) -> None:
        feat = np.asarray(features)
        if mask is not None:
            feat = feat * mask
        self.features = torch.tensor(feat, dtype=torch.float32)

        self.energies = torch.tensor(np.asarray(energies), dtype=torch.float32).view(-1, 1)


    def __len__(self) -> int:
        return self.features.shape[0]


    def __getitem__(self, idx: int) -> tuple:
        return self.features[idx], self.energies[idx]


class TargetRankingDataset(Dataset):
    '''Dataset for ranking with per-target grouping.

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
    '''

    def __init__(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            target_ids: Sequence[str],
            mask: Union[np.ndarray, None] = None
        ) -> None:

        if len(target_ids) != len(labels):
            raise ValueError("target_ids must have the same length as labels")

        feat = np.asarray(features)
        if mask is not None:
            feat = feat * mask
        self.features = torch.tensor(feat, dtype=torch.float32)

        self.labels = torch.tensor(np.asarray(labels), dtype=torch.float32)

        # Map target identifiers to integers for efficient grouping
        unique_targets = list(dict.fromkeys(target_ids))
        self.target_to_index = {t: i for i, t in enumerate(unique_targets)}
        self.target_ids = np.array([self.target_to_index[t] for t in target_ids], dtype=int)

        # Precompute target -> indices mapping
        self.target_to_indices: Dict[int, List[int]] = {}
        for idx, t in enumerate(self.target_ids):
            self.target_to_indices.setdefault(int(t), []).append(idx)


    def __len__(self) -> int:
        return self.features.shape[0]


    def __getitem__(self, idx: int) -> tuple:
        return self.features[idx], self.labels[idx], self.target_ids[idx]


class TargetBatchSampler(Sampler[List[int]]):
    '''Sampler that yields batches grouped by target.

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
    '''

    def __init__(
            self,
            target_to_indices: Dict[int, List[int]],
            batch_size: Union[int, None] = None,
            shuffle: bool = True,
            split_target_batches: bool = False
        ) -> None:

        self.target_to_indices = target_to_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_target_batches = split_target_batches

        self.target_ids = list(self.target_to_indices.keys())


    def __iter__(self):
        target_ids = self.target_ids[:]
        if self.shuffle:
            random.shuffle(target_ids)

        for target_id in target_ids:
            indices = self.target_to_indices[target_id]

            if self.batch_size is None or self.batch_size >= len(indices):
                yield indices
                continue

            if not self.split_target_batches:
                # Sample a single subset for this target
                yield random.sample(indices, self.batch_size)
            else:
                # Split into multiple batches
                shuffled = indices[:]
                random.shuffle(shuffled)
                for i in range(0, len(shuffled), self.batch_size):
                    yield shuffled[i:i + self.batch_size]


    def __len__(self) -> int:
        if self.batch_size is None or not self.split_target_batches:
            return len(self.target_ids)

        # Approximate number of batches when splitting targets
        total = 0
        for idxs in self.target_to_indices.values():
            total += int(math.ceil(len(idxs) / float(self.batch_size)))
        return total
