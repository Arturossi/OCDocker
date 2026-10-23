#!/usr/bin/env python3

# Description
###############################################################################
'''
Receptor and pocket descriptors for a target the models have never seen.

The 55 context features describe the target rather than any compound, so they are computed once
per receptor and written onto every row of that receptor's table. A merged table produced by a
full pipeline run already carries them; a table assembled any other way usually does not, and
this is how they are produced from a structure.

Receptor and pocket descriptors are not interchangeable. The receptor set describes the whole
protein and the pocket set only the residues lining the binding site, selected by a cutoff around
a reference ligand. The two sets share 24 of their names and none of the shared values agree, so
a table carrying one of them is still missing the other.

Usage:

import OCDocker.OCScore.Release.Context as ocrelcontext
'''

from __future__ import annotations

# Imports
###############################################################################
import json
import tempfile

from pathlib import Path
from typing import Any, Optional, Union, cast

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
DEFAULT_CUTOFF = 8.0
DEFAULT_PH = 7.4
DEFAULT_GRAVY_SCALE = "KyteDoolitle"

# Functions
###############################################################################
## Private ##


def _extract_resname(pdb_path: Union[str, Path], resname: str) -> str:
    '''Write one residue's HETATM records to a temporary PDB.

    Parameters
    ----------
    pdb_path : str or pathlib.Path
        Structure containing the reference ligand.
    resname : str
        HETATM residue name to extract.

    Returns
    -------
    str
        Path of the temporary single-residue PDB.
    '''

    with open(pdb_path, encoding="utf-8") as handle:
        lines = [line for line in handle
                 if line.startswith("HETATM") and line[17:20].strip() == resname]
    if not lines:
        raise ValueError(f"No HETATM records for residue {resname!r} in {pdb_path}.")

    holder = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    holder.writelines(lines)
    holder.write("END\n")
    holder.close()
    return holder.name


## Public ##


def compute_context(receptor: Union[str, Path],
                    reference_ligand: Union[str, Path],
                    name: str = "receptor",
                    ligand_resname: Optional[str] = None,
                    cutoff: float = DEFAULT_CUTOFF,
                    ph: float = DEFAULT_PH,
                    gravy_scale: str = DEFAULT_GRAVY_SCALE) -> dict[str, dict[str, Any]]:
    '''Compute the whole-protein and binding-site descriptor sets for one target.

    Parameters
    ----------
    receptor : str or pathlib.Path
        Receptor structure, protein only.
    reference_ligand : str or pathlib.Path
        Ligand defining the binding site: its own file, or a structure to pull a residue from
        when ``ligand_resname`` is given.
    name : str, optional
        Name recorded on the receptor and pocket objects.
    ligand_resname : str, optional
        HETATM residue name to extract from ``reference_ligand``.
    cutoff : float, optional
        Angstrom cutoff around the reference ligand defining the pocket.
    ph : float, optional
        pH used for the pocket protonation.
    gravy_scale : str, optional
        Hydropathy scale used for GRAVY.

    Returns
    -------
    dict
        Mapping with ``receptor`` and ``pocket`` sub-mappings, keys already prefixed.
    '''

    import OCDocker.Pocket as ocpocket
    import OCDocker.Receptor as ocreceptor

    ligand_path = (_extract_resname(reference_ligand, ligand_resname)
                   if ligand_resname else str(reference_ligand))

    receptor_object = ocreceptor.Receptor(str(receptor), name=name, gravy_scale=gravy_scale)
    pocket_object = ocpocket.Pocket(str(receptor), reference_ligand=ligand_path, name=name,
                                    cutoff=cutoff, ph=ph, gravy_scale=gravy_scale)
    return {
        "receptor": {f"receptor_{k}": v for k, v in receptor_object.get_descriptors().items()},
        "pocket": {f"pocket_{k}": v for k, v in pocket_object.get_descriptors().items()},
    }


def load_context(path: Union[str, Path]) -> dict[str, Any]:
    '''Read a descriptor JSON, for one target or for several.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON holding either one target's ``receptor`` and ``pocket`` mappings, or a mapping from
        receptor identifier to those.

    Returns
    -------
    dict
        The parsed mapping, unchanged.
    '''

    with open(path, encoding="utf-8") as handle:
        return cast(dict[str, Any], json.load(handle))


def is_multi_target(descriptors: dict[str, Any]) -> bool:
    '''Decide whether a descriptor mapping covers several receptors.

    Parameters
    ----------
    descriptors : dict
        Parsed descriptor mapping.

    Returns
    -------
    bool
        True when the top level is keyed by receptor identifier.
    '''

    return not ({"receptor", "pocket"} & set(descriptors))


def write_context(descriptors: dict[str, Any], path: Union[str, Path]) -> None:
    '''Write a descriptor mapping to JSON.

    Parameters
    ----------
    descriptors : dict
        Mapping to write.
    path : str or pathlib.Path
        Destination file.
    '''

    Path(path).write_text(json.dumps(descriptors, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


__all__ = [
    "DEFAULT_CUTOFF",
    "DEFAULT_GRAVY_SCALE",
    "DEFAULT_PH",
    "compute_context",
    "is_multi_target",
    "load_context",
    "write_context",
]
