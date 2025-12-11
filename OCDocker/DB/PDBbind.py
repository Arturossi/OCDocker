#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to process the PDBbind dataset.

They are imported as:

import OCDocker.DB.PDBbind as ocpdbbind
'''

# Imports
###############################################################################
import os

from glob import glob
from typing import Dict, Union, Optional

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

from OCDocker.Toolbox.Constants import order

import OCDocker.DB.baseDB as ocbdb

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

# Functions
###############################################################################
## Private ##

## Public ##
def read_index() -> Optional[Dict[str, Dict[str, Union[str, float]]]]:
    '''Read the index file from pdbbind database and returns a list of dictionaries with the data.

    Returns
    -------
    Dict[str, Dict[str, str | float]] | None
        A dict of dictionaries where each dictionary represents the data for a single protein.
        If the file does not exist, it will return None.
    '''

    from OCDocker.Config import get_config
    config = get_config()
    indexFiles = glob(config.pdbbind_archive + '/index/INDEX_refined_data.*')
    
    # Check if any index file was found
    if not indexFiles:
        # File does not exist, raise an error and return None
        _ = ocerror.Error.file_not_exist(f"The index file does not exist in '{config.pdbbind_archive}/index/'. Please check if the PDBbind database is correctly installed.", level=ocerror.ReportLevel.WARNING)  # type: ignore
        return None
    
    indexFile = indexFiles[0]

    # If the file exists
    if os.path.isfile(indexFile):
        # List to hold the protein data
        proteinDataOrder = f"{config.paths.pdbbind_kdki_order}M"
        proteinDataDict = {}  # Dict of dictionaries to hold data for each protein

        # Open the file in read mode
        with open(indexFile, 'r') as f:
            # Loop through the file line by line
            for line in f:
                # If the line starts with a #, skip it (no useful info)
                if line.startswith("#"):
                    continue
                
                # Split the line by spaces
                splitedLine = line.split()

                # Extract Kd/Ki and type (Kd/Ki)
                tp, kdki = splitedLine[4].split("=")
                
                # Normalize the Kd/Ki values to a consistent unit (mol/L)
                config = get_config()
                if "mM" in kdki:
                    kdki = float(kdki.replace("mM", "")) * order[config.paths.pdbbind_kdki_order]["m"]
                elif "uM" in kdki:
                    kdki = float(kdki.replace("uM", "")) * order[config.paths.pdbbind_kdki_order]["u"]
                elif "nM" in kdki:
                    kdki = float(kdki.replace("nM", "")) * order[config.paths.pdbbind_kdki_order]["n"]
                elif "pM" in kdki:
                    kdki = float(kdki.replace("pM", "")) * order[config.paths.pdbbind_kdki_order]["p"]
                elif "fM" in kdki:
                    kdki = float(kdki.replace("fM", "")) * order[config.paths.pdbbind_kdki_order]["f"]
                elif "cM" in kdki:
                    kdki = float(kdki.replace("cM", "")) * order[config.paths.pdbbind_kdki_order]["c"]
                else:  # Assume M if not otherwise specified
                    kdki = float(kdki.replace("M", "")) * order[config.paths.pdbbind_kdki_order]["M"]

                # Create a dictionary for this protein and its data
                protein_entry = {
                    "Protein": splitedLine[0],
                    "resolution": splitedLine[1],
                    "release_year": splitedLine[2],
                    "-logKd/Ki": splitedLine[3],
                    "Ki/Kd": tp,
                    "Ki/Kd_value": kdki,
                    "Ki/Kd_order": proteinDataOrder,
                    "dG": occ.convert_Ki_Kd_to_dG(kdki)
                }

                # Add the dictionary to the dict setting the protein name as the key
                proteinDataDict[splitedLine[0]] = protein_entry

        # Return the list of dictionaries
        return proteinDataDict
    else:
        # File does not exist, raise an error and return None
        _ = ocerror.Error.file_not_exist(f"The file {indexFile} does not exist. Please check if the PDBbind database is correctly installed.", level=ocerror.ReportLevel.WARNING)  # type: ignore
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
