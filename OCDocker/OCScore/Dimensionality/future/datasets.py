#!/usr/bin/env python3

# Description
###############################################################################
'''Datasets for the future Autoencoder pipeline.'''

# Imports
###############################################################################

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

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


class AutoencoderDataset(Dataset):
    """Dataset for autoencoder training with optional energy targets.

    Parameters
    ----------
    features : np.ndarray
        Input feature matrix.
    energies : np.ndarray | None, optional
        Energy labels for supervised head. If None, all samples are unlabeled.
    feature_mask : np.ndarray | None, optional
        Feature mask (element-wise) applied to inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from OCDocker.OCScore.Dimensionality.future.datasets import AutoencoderDataset
    >>> features = np.random.rand(100, 20)  # 100 samples, 20 features each
    >>> energies = np.random.rand(100)      # 100 energy labels
    >>> feature_mask = np.random.randint(0, 2, size=(100, 20))  # Random binary mask
    >>> dataset = AutoencoderDataset(features, energies, feature_mask)
    >>> sample_features, sample_energy, sample_mask = dataset[0]
    >>> print(sample_features.shape)  # torch.Size([20])
    >>> print(sample_energy.shape)    # torch.Size([1])
    >>> print(sample_mask)            # tensor(True) or tensor(False)
    """

    def __init__(
            self,
            features: np.ndarray,
            energies: Optional[np.ndarray] = None,
            feature_mask: Optional[np.ndarray] = None
        ) -> None:
        '''Initialize dataset.

        Parameters
        ----------
        features : np.ndarray
            Input feature matrix.
        energies : np.ndarray | None, optional
            Energy targets, by default None.
        feature_mask : np.ndarray | None, optional
            Feature mask, by default None.
        '''

        feat = np.asarray(features, dtype=np.float32)
        if feature_mask is not None:
            # Apply mask once to keep dataset sampling fast.
            feat = feat * np.asarray(feature_mask, dtype=np.float32)
        self.features = torch.tensor(feat, dtype=torch.float32)

        if energies is None:
            # Use NaNs to represent missing labels; mask marks unlabeled samples.
            self.energies = torch.full((self.features.shape[0], 1), float('nan'))
            self.energy_mask = torch.zeros(self.features.shape[0], dtype=torch.bool)
        else:
            energies_arr = np.asarray(energies, dtype=np.float32).reshape(-1, 1)
            # Treat NaN energies as missing labels for mixed datasets.
            mask = ~np.isnan(energies_arr)
            energies_arr = np.where(mask, energies_arr, 0.0)
            self.energies = torch.tensor(energies_arr, dtype=torch.float32)
            # Mask marks samples with valid energy labels.
            self.energy_mask = torch.tensor(mask.reshape(-1), dtype=torch.bool)


    def __len__(self) -> int:
        '''Return dataset length.

        Returns
        -------
        int
            Number of samples.
        '''

        return self.features.shape[0]


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Return a dataset sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Features, energies, and energy mask tensors.
        '''

        return self.features[idx], self.energies[idx], self.energy_mask[idx]
