#!/usr/bin/env python3

# Description
###############################################################################
'''
Frozen feature contract and inference architecture for the Gen3B OCScore models.

A generation bundle carries one feature contract -- the manifest naming the 365 input columns in
order, the four blocks they fall into, and the context standardisation fitted during training --
shared by every model family shipped beside it. Scores and ligand descriptors are standardised
within receptor, so a whole library is ranked against one target at a time and a single compound
on its own cannot be scored.

Each model declares the feature blocks it takes, and is fed only the columns of those blocks.

Usage:

import OCDocker.OCScore.Release.Bundle as ocrelbundle
'''

from __future__ import annotations

# Imports
###############################################################################
import json

from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
BLOCK_LETTERS = {"A_scores": "A", "B_ligand": "B", "C_receptor": "C", "D_pocket": "D"}

SCORE_RENAMES = {
    "SMINA_VINA": "smina_vina",
    "SMINA_VINARDO": "smina_vinardo",
    "SMINA_SCORING_AD4": "smina_ad4_scoring",
    "SMINA_SCORING_DKOES": "smina_dkoes_scoring",
    "SMINA_OLD_SCORING_DKOES": "smina_dkoes_scoring_old",
    "SMINA_FAST_DKOES": "smina_dkoes_fast",
    "VINA_VINA": "vina_vina",
    "VINA_VINARDO": "vina_vinardo",
    "PLANTS_CHEMPLP": "plants_chemplp",
    "PLANTS_PLP": "plants_plp",
    "PLANTS_PLP95": "plants_plp95",
    "GNINA_SCORING_AD4": "gnina_ad4_scoring",
    "GNINA_VINA": "gnina_vina",
    "GNINA_VINARDO": "gnina_vinardo",
    "GNINA_SCORING_DKOES": "gnina_dkoes_scoring",
    "GNINA_OLD_SCORING_DKOES": "gnina_dkoes_scoring_old",
    "GNINA_FAST_DKOES": "gnina_dkoes_fast",
}

# Classes
###############################################################################


class TunedNet(nn.Module):
    '''Shared trunk followed by a per-task branch, as frozen for the M1.1 release.

    Parameters
    ----------
    d_in : int
        Number of input features.
    cfg : dict
        Architecture configuration stored in the checkpoint: depth, width, latent, share,
        dropout and use_cons.
    '''

    def __init__(self, d_in: int, cfg: dict[str, Any]) -> None:
        super().__init__()

        depth, width, lat = cfg["depth"], cfg["width"], cfg["latent"]
        n_shared = max(1, int(round(depth * cfg["share"]))) if cfg["share"] > 0 else 0
        n_branch = depth - n_shared

        def stack(d0: int, n: int, d1: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = d0
            for _ in range(n):
                layers += [nn.Linear(prev, width), nn.GELU(), nn.Dropout(cfg["dropout"])]
                prev = width
            layers.append(nn.Linear(prev, d1))
            return nn.Sequential(*layers)

        self.trunk = stack(d_in, max(0, n_shared - 1), width) if n_shared else None
        d_mid = width if n_shared else d_in
        self.br_aff = stack(d_mid, n_branch, lat)
        self.br_scr = stack(d_mid, n_branch, lat)
        h = lat + (1 if cfg["use_cons"] else 0)
        self.aff = nn.Sequential(
            nn.Linear(h, width), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(width, 1)
        )
        self.scr = nn.Sequential(
            nn.Linear(h, width), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(width, 1)
        )
        self.use_cons = cfg["use_cons"]

    def forward(self, x: torch.Tensor, which: str, cons: torch.Tensor) -> torch.Tensor:
        '''Run the screening or affinity head.

        Parameters
        ----------
        x : torch.Tensor
            Preprocessed feature matrix.
        which : str
            ``"scr"`` for the screening head, ``"aff"`` for the affinity head.
        cons : torch.Tensor
            Consensus column, used only when the checkpoint sets ``use_cons``.

        Returns
        -------
        torch.Tensor
            One logit per row.
        '''

        h = self.trunk(x) if self.trunk is not None else x
        z = (self.br_aff if which == "aff" else self.br_scr)(h)
        if self.use_cons:
            z = torch.cat([z, cons], 1)
        return cast(torch.Tensor, (self.aff if which == "aff" else self.scr)(z).squeeze(-1))


class FeatureContract:
    '''Manifest and fitted preprocessing shared by every family in a generation bundle.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory holding ``feature_manifest.json`` and ``preprocessing.npz``.
    '''

    def __init__(self, path: Union[str, Path]) -> None:
        self.dir = Path(path)
        with open(self.dir / "feature_manifest.json", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        pre = np.load(self.dir / "preprocessing.npz", allow_pickle=True)
        self.ctx_columns = [str(c) for c in pre["ctx_columns"]]
        self.ctx_mean = pre["ctx_mean"]
        self.ctx_std = pre["ctx_std"]
        self.feature_order = [str(c) for c in pre["feature_order"]]

        if self.feature_order != list(self.manifest["feature_order"]):
            raise ValueError(
                "feature_manifest.json and preprocessing.npz disagree on the feature order."
            )

        block_of = {c: g for g in self.manifest["groups"] for c in self.manifest["groups"][g]}
        self.block_letter = np.array([BLOCK_LETTERS[block_of[c]] for c in self.feature_order])

    @property
    def groups(self) -> dict[str, list[str]]:
        '''Feature names per block.

        Returns
        -------
        dict of str to list of str
            Mapping from block name to the columns it contains.
        '''

        return {k: list(v) for k, v in self.manifest["groups"].items()}

    def mask(self, live_blocks: str) -> np.ndarray:
        '''Mark the contract columns that belong to the given blocks.

        Parameters
        ----------
        live_blocks : str
            Block letters, for example ``"ABCD"`` or ``"AC"``.

        Returns
        -------
        numpy.ndarray
            Float vector of ones on the columns of those blocks and zeros elsewhere.
        '''

        return np.isin(self.block_letter, list(live_blocks)).astype(np.float32)

    def columns(self, live_blocks: str) -> list[str]:
        '''Name the contract columns that belong to the given blocks, in contract order.

        Parameters
        ----------
        live_blocks : str
            Block letters, for example ``"ABCD"`` or ``"AC"``.

        Returns
        -------
        list of str
            Column names of those blocks.
        '''

        keep = self.mask(live_blocks).astype(bool)
        return [c for c, k in zip(self.feature_order, keep) if k]

    def featurize(self, dataframe: pd.DataFrame, groups: Any,
                  blocks: str = "ABCD") -> np.ndarray:
        '''Apply the training-time transforms recorded in the bundle.

        Scores are negated so that higher is better, then scores and ligand descriptors are
        standardised within receptor. Context columns use the standardisation fitted on PDBbind
        during training. Only the requested blocks are computed; the columns of the others are
        left at zero and must not be read by any model.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Table carrying every column of the requested blocks.
        groups : array-like
            Receptor identifier per row, defining the within-receptor standardisation.
        blocks : str, optional
            Block letters to compute; all four by default.

        Returns
        -------
        numpy.ndarray
            Float32 matrix with one column per manifest feature, in manifest order.
        '''

        wanted = set(self.columns(blocks))
        missing = [c for c in self.feature_order if c in wanted and c not in dataframe.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} required feature columns are absent, first few: {missing[:8]}"
            )

        g = np.asarray(groups)
        cols: dict[str, np.ndarray] = {}
        for c in self.manifest["groups"]["A_scores"]:
            if c in wanted:
                cols[c] = self._z_within(-dataframe[c].astype(float).to_numpy(), g)
        for c in self.manifest["groups"]["B_ligand"]:
            if c in wanted:
                cols[c] = self._z_within(dataframe[c].astype(float).to_numpy(), g)

        cmean = dict(zip(self.ctx_columns, self.ctx_mean))
        cstd = dict(zip(self.ctx_columns, self.ctx_std))
        for c in self.manifest["groups"]["C_receptor"] + self.manifest["groups"]["D_pocket"]:
            if c in wanted:
                cols[c] = (dataframe[c].astype(float).to_numpy() - cmean[c]) / (cstd[c] + 1e-12)

        empty = np.zeros(len(dataframe))
        return np.stack([cols.get(c, empty) for c in self.feature_order],
                        axis=1).astype(np.float32)

    @staticmethod
    def _z_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
        '''Standardise within receptor using the sample standard deviation.

        Parameters
        ----------
        values : numpy.ndarray
            Column to standardise.
        groups : numpy.ndarray
            Receptor identifier per row.

        Returns
        -------
        numpy.ndarray
            Standardised column, with non-finite results replaced by zero.
        '''

        out = np.zeros(len(values))
        for t in np.unique(groups):
            m = groups == t
            out[m] = (values[m] - values[m].mean()) / (values[m].std(ddof=1) + 1e-12)
        return np.nan_to_num(out)


# Functions
###############################################################################
## Public ##


def load_network(checkpoint: Union[str, Path], device: str = "cpu") -> tuple[TunedNet, dict]:
    '''Load one frozen checkpoint into an evaluation-mode network.

    Parameters
    ----------
    checkpoint : str or pathlib.Path
        Path to a ``.pt`` checkpoint written by the release or ablation training scripts.
    device : str, optional
        Torch device for the loaded weights.

    Returns
    -------
    tuple of (TunedNet, dict)
        The network in evaluation mode and the checkpoint metadata.
    '''

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    net = TunedNet(payload["d_in"], payload["cfg"]).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, payload


def rank_ensemble(predictions: list[np.ndarray], groups: Any) -> np.ndarray:
    '''Combine seed predictions as the mean of within-receptor percentile ranks.

    Parameters
    ----------
    predictions : list of numpy.ndarray
        One score vector per seed, all aligned to the same rows.
    groups : array-like
        Receptor identifier per row.

    Returns
    -------
    numpy.ndarray
        Ensemble score per row, in the unit interval within each receptor.
    '''

    from scipy.stats import rankdata

    g = np.asarray(groups)
    out = np.zeros(len(g))
    for t in np.unique(g):
        m = g == t
        out[m] = np.mean([rankdata(p[m]) / m.sum() for p in predictions], axis=0)
    return out


__all__ = [
    "BLOCK_LETTERS",
    "SCORE_RENAMES",
    "FeatureContract",
    "TunedNet",
    "load_network",
    "rank_ensemble",
]
