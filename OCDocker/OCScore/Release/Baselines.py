#!/usr/bin/env python3

# Description
###############################################################################
'''
Reference comparators shipped beside the OCScore networks.

A screening number means nothing on its own, so the bundle carries the comparators a result has
to be read against. Two are untrained rules computed from the score block alone, mean and median
consensus. Two are simple learned models fitted on the same training rows as the networks, ridge
and elastic-net logistic regression, at three feature arms each. One is a memorisation baseline,
the maximum ECFP4 similarity to the training actives, which asks whether a ranking could have
been obtained by recognising familiar chemistry. The last is a random ranking.

None of these are scorers to report in place of the release model; they are the floor it has to
clear, and the memorisation and random rows are controls in the same sense as the
randomised-label network.

Usage:

import OCDocker.OCScore.Release.Baselines as ocrelbaselines
'''

from __future__ import annotations

# Imports
###############################################################################
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import pandas as pd

import OCDocker.OCScore.Release.Bundle as ocrelbundle

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
BASELINES_SUBDIR = "baselines"
LINEAR_COEFFICIENTS = "linear_coefficients.npz"
TRAIN_ACTIVE_FINGERPRINTS = "train_active_fingerprints.npz"
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
RANDOM_SEED = 0

# Functions
###############################################################################
## Private ##


def _popcount_table() -> np.ndarray:
    '''Build the byte population-count lookup used by the Tanimoto kernel.

    Returns
    -------
    numpy.ndarray
        Number of set bits for every byte value.
    '''

    # Annotated so the return is typed with both the NumPy 2.4 and 2.5 stubs
    table: np.ndarray = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)
    return table


def _max_tanimoto(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    '''Maximum Tanimoto similarity of each query fingerprint against a reference set.

    Parameters
    ----------
    query : numpy.ndarray
        Packed query fingerprints, one row per compound.
    reference : numpy.ndarray
        Packed reference fingerprints.

    Returns
    -------
    numpy.ndarray
        Maximum similarity per query row.
    '''

    counts = _popcount_table()
    n_query, n_reference = counts[query].sum(1), counts[reference].sum(1)
    out = np.zeros(len(query))
    for start in range(0, len(query), 256):
        stop = start + 256
        inter = counts[query[start:stop, None, :] & reference[None, :, :]].sum(2)
        union = n_query[start:stop, None] + n_reference[None, :] - inter
        out[start:stop] = np.max(np.where(union > 0, inter / np.maximum(union, 1), 0.0), axis=1)
    return out


## Public ##


def baselines_dir(root: Union[str, Path]) -> Path:
    '''Locate the baseline artifacts inside a generation bundle.

    Parameters
    ----------
    root : str or pathlib.Path
        Generation bundle directory.

    Returns
    -------
    pathlib.Path
        Directory holding the baseline artifacts.
    '''

    return Path(root) / BASELINES_SUBDIR


def score_consensus(contract: ocrelbundle.FeatureContract, features: np.ndarray) -> dict[str, np.ndarray]:
    '''Untrained consensus of the score block.

    The contract already negates scores so that higher is better and standardises them within
    receptor, so the consensus is the row-wise mean or median of those columns.

    Parameters
    ----------
    contract : OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded feature contract.
    features : numpy.ndarray
        Preprocessed feature matrix.

    Returns
    -------
    dict of str to numpy.ndarray
        Mean and median consensus scores.
    '''

    block = contract.mask("A").astype(bool)
    scores = features[:, block]
    n = int(block.sum())
    return {
        f"consensus_mean_{n}|rule": scores.mean(axis=1),
        f"consensus_median_{n}|rule": np.median(scores, axis=1),
    }


def score_linear(root: Union[str, Path], features: np.ndarray) -> dict[str, np.ndarray]:
    '''Score the shipped ridge and elastic-net models.

    Their coefficient vectors span the full contract and are zero outside the blocks each arm was
    fitted on, so no additional masking is needed.

    Parameters
    ----------
    root : str or pathlib.Path
        Generation bundle directory.
    features : numpy.ndarray
        Preprocessed feature matrix.

    Returns
    -------
    dict of str to numpy.ndarray
        One decision function per shipped linear model.
    '''

    path = baselines_dir(root) / LINEAR_COEFFICIENTS
    if not path.is_file():
        return {}

    store = np.load(path)
    out: dict[str, np.ndarray] = {}
    for key in (k for k in store.files if k.endswith("|coef")):
        name = key.split("|")[0]
        coefficients = store[key]
        if coefficients.shape[0] != features.shape[1]:
            raise ValueError(
                f"Baseline {name!r} has {coefficients.shape[0]} coefficients but the contract "
                f"has {features.shape[1]} features."
            )
        intercept = float(store[f"{name}|intercept"][0])
        out[f"{name.replace(' ', '_')}|linear"] = features @ coefficients + intercept
    return out


def score_ecfp4(root: Union[str, Path], smiles: pd.Series) -> dict[str, np.ndarray]:
    '''Maximum ECFP4 similarity of each compound to the training actives.

    Parameters
    ----------
    root : str or pathlib.Path
        Generation bundle directory.
    smiles : pandas.Series
        SMILES string per compound.

    Returns
    -------
    dict of str to numpy.ndarray
        Similarity against all training actives and against each benchmark separately.
    '''

    path = baselines_dir(root) / TRAIN_ACTIVE_FINGERPRINTS
    if not path.is_file():
        return {}

    try:
        from rdkit import Chem, DataStructs, RDLogger as _RDLogger
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        return {}

    # The rdkit stubs do not declare DisableLog
    RDLogger: Any = _RDLogger
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
    )
    bits = np.zeros((len(smiles), FINGERPRINT_BITS), dtype=bool)
    for i, text in enumerate(smiles.astype(str)):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            continue
        holder = np.zeros(FINGERPRINT_BITS, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), holder)
        bits[i] = holder.astype(bool)

    packed = np.packbits(bits, axis=1)
    store = np.load(path)
    reference = {
        "ecfp4_all_training_actives": np.vstack([store[k] for k in store.files]),
        **{f"ecfp4_{k}_actives": store[k] for k in store.files},
    }
    return {f"{name}|similarity": _max_tanimoto(packed, ref) for name, ref in reference.items()}


def score_random(n_rows: int, seed: int = RANDOM_SEED) -> dict[str, np.ndarray]:
    '''Random ranking, the floor every method must clear.

    Parameters
    ----------
    n_rows : int
        Number of rows to score.
    seed : int, optional
        Seed for the permutation.

    Returns
    -------
    dict of str to numpy.ndarray
        A single random score vector.
    '''

    rng = np.random.default_rng(seed)
    return {"random_ranking|control": rng.permutation(n_rows).astype(float)}


def score_baselines(contract: ocrelbundle.FeatureContract,
                    root: Union[str, Path],
                    features: np.ndarray,
                    smiles: Optional[pd.Series] = None,
                    seed: int = RANDOM_SEED) -> dict[str, np.ndarray]:
    '''Score every available baseline.

    Parameters
    ----------
    contract : OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded feature contract.
    root : str or pathlib.Path
        Generation bundle directory.
    features : numpy.ndarray
        Preprocessed feature matrix.
    smiles : pandas.Series, optional
        SMILES per compound; the ECFP4 baseline is skipped when absent.
    seed : int, optional
        Seed for the random ranking.

    Returns
    -------
    dict of str to numpy.ndarray
        Score vector per baseline, keyed ``name|variant``.
    '''

    out: dict[str, np.ndarray] = {}
    out.update(score_consensus(contract, features))
    out.update(score_linear(root, features))
    if smiles is not None:
        out.update(score_ecfp4(root, smiles))
    out.update(score_random(len(features), seed))
    return out


__all__ = [
    "BASELINES_SUBDIR",
    "LINEAR_COEFFICIENTS",
    "TRAIN_ACTIVE_FINGERPRINTS",
    "baselines_dir",
    "score_baselines",
    "score_consensus",
    "score_ecfp4",
    "score_linear",
    "score_random",
]
