#!/usr/bin/env python3

# Description
###############################################################################
'''
Discovery of the OCScore model families shipped with a generation bundle.

The Gen3B generation ships three families. ``release`` is the frozen M1.1 model, the one a
screening result should normally be reported from. ``ablation`` is the eight-arm block factorial,
each arm trained on the same data with some input blocks zeroed, which exists to attribute
performance to the blocks rather than to be reported as an alternative scorer. ``control`` is the
randomised-training-label model: it never saw a true label, so on a sound evaluation set it must
land at chance, and anything else is a warning about the set rather than a result.

Usage:

import OCDocker.OCScore.Release.Catalog as ocrelcatalog
'''

from __future__ import annotations

# Imports
###############################################################################
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
GENERATION = "gen3b"
MODELS_DIR_ENV = "OCSCORE_MODELS_DIR"
CONTRACT_SUBDIR = "release_m1_1"
BLOCK_OF_TOKEN = {"S": "A", "L": "B", "R": "C", "P": "D"}
FAMILIES = ("release", "ablation", "control")

# Classes
###############################################################################


@dataclass(frozen=True)
class ModelSpec:
    '''One shipped model: a family, its input mask, and its seed checkpoints.'''

    name: str
    family: str
    live_blocks: str
    directory: Path
    checkpoints: tuple[Path, ...]
    seeds: tuple[int, ...]

    @property
    def is_full_input(self) -> bool:
        '''Whether the model sees every input block.

        Returns
        -------
        bool
            True when no block is masked away.
        '''

        return set(self.live_blocks) == {"A", "B", "C", "D"}


# Functions
###############################################################################
## Private ##


def _seed_of(path: Path) -> int:
    '''Extract the training seed from a checkpoint filename.

    Parameters
    ----------
    path : pathlib.Path
        Checkpoint path ending in ``seed<N>.pt``.

    Returns
    -------
    int
        The seed encoded in the filename.
    '''

    return int(path.stem.rsplit("seed", 1)[-1])


def _blocks_of_arm(arm: str) -> str:
    '''Translate an ablation arm directory name into its live block letters.

    Parameters
    ----------
    arm : str
        Arm directory name such as ``"S_L_R_P"``.

    Returns
    -------
    str
        Sorted block letters, for example ``"ABCD"``.
    '''

    try:
        return "".join(sorted(BLOCK_OF_TOKEN[t] for t in arm.split("_")))
    except KeyError as exc:
        raise ValueError(f"Unrecognised ablation arm directory {arm!r}.") from exc


def _collect(directory: Path) -> tuple[tuple[Path, ...], tuple[int, ...]]:
    '''List the seed checkpoints in a model directory.

    Parameters
    ----------
    directory : pathlib.Path
        Directory holding ``*seed<N>.pt`` files.

    Returns
    -------
    tuple
        Checkpoint paths and their seeds, both ordered by seed.
    '''

    found = sorted(directory.glob("*seed*.pt"), key=_seed_of)
    return tuple(found), tuple(_seed_of(p) for p in found)


## Public ##


def default_root() -> Path:
    '''Resolve the shipped generation bundle directory.

    The environment variable named by ``MODELS_DIR_ENV`` wins when set, otherwise the bundle is
    taken from ``OCScore_models/<generation>`` beside the installed package.

    Returns
    -------
    pathlib.Path
        Directory expected to hold the generation bundle.
    '''

    override = os.environ.get(MODELS_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve() / GENERATION
    return Path(__file__).resolve().parents[3] / "OCScore_models" / GENERATION


def contract_dir(root: Optional[Union[str, Path]] = None) -> Path:
    '''Locate the feature contract shared by every family in the bundle.

    Parameters
    ----------
    root : str or pathlib.Path, optional
        Generation bundle directory; the shipped bundle is used when omitted.

    Returns
    -------
    pathlib.Path
        Directory holding ``feature_manifest.json`` and ``preprocessing.npz``.
    '''

    return (Path(root) if root is not None else default_root()) / CONTRACT_SUBDIR


def discover(root: Optional[Union[str, Path]] = None,
             families: Optional[tuple[str, ...]] = None) -> list[ModelSpec]:
    '''List every model shipped in a generation bundle.

    Parameters
    ----------
    root : str or pathlib.Path, optional
        Generation bundle directory; the shipped bundle is used when omitted.
    families : tuple of str, optional
        Restrict the result to these families; all of them when omitted.

    Returns
    -------
    list of ModelSpec
        Release model first, then ablation arms, then controls.
    '''

    base = Path(root) if root is not None else default_root()
    if not base.is_dir():
        raise FileNotFoundError(
            f"No OCScore generation bundle at {base}. Set {MODELS_DIR_ENV} or pass an explicit "
            "root."
        )

    wanted = set(families or FAMILIES)
    specs: list[ModelSpec] = []

    release = base / CONTRACT_SUBDIR
    if "release" in wanted and release.is_dir():
        paths, seeds = _collect(release)
        if paths:
            specs.append(ModelSpec("release_m1_1", "release", "ABCD", release, paths, seeds))

    if "ablation" in wanted and (base / "ablation").is_dir():
        for arm in sorted((base / "ablation").iterdir()):
            if not arm.is_dir():
                continue
            paths, seeds = _collect(arm)
            if paths:
                specs.append(ModelSpec(f"ablation_{arm.name}", "ablation",
                                       _blocks_of_arm(arm.name), arm, paths, seeds))

    if "control" in wanted and (base / "control").is_dir():
        for ctl in sorted((base / "control").iterdir()):
            if not ctl.is_dir():
                continue
            paths, seeds = _collect(ctl)
            if paths:
                specs.append(ModelSpec(f"control_{ctl.name}", "control", "ABCD",
                                       ctl, paths, seeds))

    return specs


__all__ = [
    "CONTRACT_SUBDIR",
    "FAMILIES",
    "GENERATION",
    "MODELS_DIR_ENV",
    "ModelSpec",
    "contract_dir",
    "default_root",
    "discover",
]
