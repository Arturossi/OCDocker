#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to process the PDBbind dataset.

Usage:

import OCDocker.DB.PDBbind as ocpdbbind
'''

# Imports
###############################################################################
import math
import os
import re

from glob import glob
from typing import Dict, Union, Optional

import OCDocker.Error as ocerror

from OCDocker.Config import get_config
from OCDocker.Toolbox.Constants import order
import OCDocker.Toolbox.Constants as occ


# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

_base_db_module = None


class _LazyBaseDB:
    """Load baseDB only when docking/preparation helpers need it."""

    def prepare(self, *args, **kwargs):
        return _get_base_db().prepare(*args, **kwargs)

    def run_docking(self, *args, **kwargs):
        return _get_base_db().run_docking(*args, **kwargs)


def _get_base_db():
    global _base_db_module
    if _base_db_module is None:
        import OCDocker.DB.baseDB as base_db
        _base_db_module = base_db
    return _base_db_module


ocbdb = _LazyBaseDB()

_AFFINITY_RE = re.compile(
    r"^(?P<kind>[A-Za-z0-9]+)"
    r"(?P<relation><=|>=|=|<|>|~)"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<unit>[A-Za-z]+)$"
)


def _normalise_molar_order(order_prefix: str) -> str:
    '''Normalise historical molar order names used by PDBbind parsing.'''

    return "un" if order_prefix == "M" else order_prefix


def _format_molar_order(order_prefix: str) -> str:
    '''Format an order prefix as a molar concentration unit.'''

    return "M" if order_prefix == "un" else f"{order_prefix}M"


def _unit_to_order_prefix(unit: str) -> str:
    '''Convert a concentration unit string to the Constants.order prefix.'''

    if unit == "M":
        return "un"
    if unit.endswith("M") and len(unit) > 1:
        return unit[:-1]
    raise ValueError(f"Unsupported PDBbind affinity unit: {unit}")


def _parse_affinity_token(token: str) -> tuple[str, str, float, str, str]:
    '''Parse a PDBbind affinity token such as Kd=49uM or IC50<=3.5nM.'''

    match = _AFFINITY_RE.match(token)
    if match is None:
        raise ValueError(f"Invalid PDBbind affinity token: {token}")

    kind = match.group("kind")
    relation = match.group("relation")
    value = float(match.group("value"))
    unit = match.group("unit")
    unit_prefix = _unit_to_order_prefix(unit)
    return kind, relation, value, unit, unit_prefix


def _format_neg_log_kdki(kdki_molar: float, relation: str) -> str:
    '''Format -log10(Kd/Ki) and flip inequality direction for log space.'''

    if kdki_molar <= 0:
        return ""

    neg_log = -math.log10(kdki_molar)
    value = f"{neg_log:.6g}"
    relation_map = {
        "=": "",
        "~": "~",
        "<": ">",
        "<=": ">=",
        ">": "<",
        ">=": "<=",
    }
    return f"{relation_map.get(relation, '')}{value}"


def _parse_index_annotation(annotation: str) -> tuple[str, str, str]:
    '''Parse the reference, ligand name, and trailing note after //.'''

    annotation = annotation.strip()
    if not annotation:
        return "", "", ""

    parts = annotation.split(maxsplit=1)
    reference = parts[0]
    rest = parts[1].strip() if len(parts) > 1 else ""
    ligand_name = ""
    comment = rest

    ligand_match = re.search(r"\(([^)]*)\)", rest)
    if ligand_match is not None:
        ligand_name = ligand_match.group(1).strip()
        before = rest[:ligand_match.start()].strip()
        after = rest[ligand_match.end():].strip()
        comment = " ".join(part for part in (before, after) if part)

    return reference, ligand_name, comment


def _parse_index_line(
    line: str,
    unit_scale: Dict[str, float],
    molar_scale: Dict[str, float],
    protein_data_order: str,
) -> Optional[Dict[str, Union[str, float]]]:
    '''Parse one non-comment PDBbind index line.'''

    data_part, _, annotation = line.partition("//")
    split_line = data_part.split()
    if len(split_line) < 4:
        return None

    affinity_idx: Optional[int] = None
    for idx, token in enumerate(split_line[3:], start=3):
        if _AFFINITY_RE.match(token) is not None:
            affinity_idx = idx
            break

    if affinity_idx is None:
        raise ValueError(f"No PDBbind affinity token found in line: {line.rstrip()}")

    tp, relation, raw_value, raw_unit, unit_prefix = _parse_affinity_token(split_line[affinity_idx])
    if unit_prefix not in unit_scale or unit_prefix not in molar_scale:
        raise ValueError(f"Unsupported PDBbind affinity unit prefix: {unit_prefix}")

    kdki = raw_value * unit_scale[unit_prefix]
    kdki_molar = raw_value * molar_scale[unit_prefix]
    neg_log_kdki = (
        split_line[3]
        if affinity_idx > 3
        else _format_neg_log_kdki(kdki_molar, relation)
    )
    dG = float(occ.convert_Ki_Kd_to_dG(kdki_molar))
    reference, ligand_name, comment = _parse_index_annotation(annotation)

    return {
        "Protein": split_line[0],
        "resolution": split_line[1],
        "release_year": split_line[2],
        "-logKd/Ki": neg_log_kdki,
        "Ki/Kd": tp,
        "Ki/Kd_relation": relation,
        "Ki/Kd_value": kdki,
        "Ki/Kd_order": protein_data_order,
        "Ki/Kd_raw_value": raw_value,
        "Ki/Kd_raw_unit": raw_unit,
        "dG": dG,
        "dG_kcal_mol": dG / 1000,
        "reference": reference,
        "ligand_name": ligand_name,
        "index_comment": comment,
    }

## Public ##

def prepare(overwrite: bool = False) -> None:
    '''Prepares the PDBbind database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the preparation if the results already exist, by default False.

    Returns
    -------
    None
    '''

    return ocbdb.prepare("pdbbind", overwrite = overwrite)


def read_index(index_file: Optional[str] = None) -> Optional[Dict[str, Dict[str, Union[str, float]]]]:
    '''Read the index file from pdbbind database and returns a list of dictionaries with the data.

    Parameters
    ----------
    index_file : str, optional
        Direct path to a PDBbind INDEX_refined_data file. If not provided, the
        first matching file under ``config.pdbbind_archive/index`` is used.

    Returns
    -------
    Dict[str, Dict[str, str | float]] | None
        A dict of dictionaries where each dictionary represents the data for a single protein.
        If the file does not exist, it will return None.
    '''

    config = get_config()
    if index_file is None:
        indexFiles = sorted(glob(config.pdbbind_archive + '/index/INDEX_refined_data.*'))

        # Check if any index file was found
        if not indexFiles:
            # File does not exist, raise an error and return None
            _ = ocerror.Error.file_not_exist(f"The index file does not exist in '{config.pdbbind_archive}/index/'. Please check if the PDBbind database is correctly installed.", level=ocerror.ReportLevel.WARNING)
            return None

        indexFile = indexFiles[0]
    else:
        indexFile = index_file

    # If the file exists
    if os.path.isfile(indexFile):
        # List to hold the protein data
        kdki_order = _normalise_molar_order(config.paths.pdbbind_kdki_order)
        proteinDataOrder = _format_molar_order(kdki_order)
        unit_scale = order[kdki_order]
        molar_scale = order["un"]
        proteinDataDict = {}  # Dict of dictionaries to hold data for each protein

        # Open the file in read mode
        with open(indexFile, 'r') as f:
            # Loop through the file line by line
            for line in f:
                # If the line starts with a #, skip it (no useful info)
                if line.startswith("#") or not line.strip():
                    continue

                protein_entry = _parse_index_line(
                    line,
                    unit_scale = unit_scale,
                    molar_scale = molar_scale,
                    protein_data_order = proteinDataOrder,
                )
                if protein_entry is None:
                    continue

                # Add the dictionary to the dict setting the protein name as the key
                proteinDataDict[str(protein_entry["Protein"])] = protein_entry

        # Return the list of dictionaries
        return proteinDataDict
    else:
        # File does not exist, raise an error and return None
        _ = ocerror.Error.file_not_exist(f"The file {indexFile} does not exist. Please check if the PDBbind database is correctly installed.", level=ocerror.ReportLevel.WARNING)
        return None


def run_gnina(overwrite: bool = False) -> int:
    '''Runs gnina in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated, by default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raise
    -----
    None
    '''

    return ocbdb.run_docking("pdbbind", "gnina", overwrite = overwrite)


def run_plants(overwrite: bool = False) -> int:
    '''Runs PLANTS in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the PLANTS if the results already exist, by default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    return ocbdb.run_docking("pdbbind", "plants", overwrite = overwrite)


def run_smina(overwrite: bool = False) -> int:
    '''Runs smina in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the smina if the results already exist, by default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    return ocbdb.run_docking("pdbbind", "smina", overwrite = overwrite)


def run_vina(overwrite: bool = False) -> int:
    '''Runs vina in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the vina if the results already exist, by default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    return ocbdb.run_docking("pdbbind", "vina", overwrite = overwrite)
