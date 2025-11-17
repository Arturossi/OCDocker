#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used as base for all databases. It
contains functions that are common to all databases.

They are imported as:

import OCDocker.DB.baseDB as ocbdb
'''

# Imports
###############################################################################
import os

from glob import glob

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

import OCDocker.Processing.Dock as ocdock
import OCDocker.Processing.Preprocessing.Prepare as ocprepare

import OCDocker.Toolbox.Printing as ocprint


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
def prepare(archive: str, overwrite: bool = False, spacing: float = 0.33, sanitize: bool = True) -> None:
    '''Prepares the database.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind].
    overwrite : bool, optional
        If True overwrites the files, if False does not overwrite the files. The default is False.
    spacing : float, optional
        The spacing to be used in the grid. The default is 0.33.
    sanitize : bool, optional
        If True sanitizes the ligands, if False does not sanitize the ligands. The default is True.

    Returns
    -------
    None
    '''

    # Find which kind of archive it will be
    config = get_config()
    if archive.lower() == "dudez":
        chosenArchive = config.dudez_archive
    elif archive.lower() == "pdbbind":
        chosenArchive = config.pdbbind_archive
    else:
        ocprint.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")
        return None

    # Get all paths in the database
    paths = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index']]

    # Generate boxes for all receptors
    ocprint.printv("Generating information regarding possible ligand site.")

    # Prepare it
    ocprepare.prepare(paths, overwrite, archive, sanitize, spacing)

    return None


def run_docking(archive: str, dockingAlgorithm: str, digestFormat: str = "json", overwrite: bool = False) -> int:
    '''Run docking.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind].
    dockingAlgorithm : str
        The docking algorithm to be used. The options are [vina, smina, plants].
    digestFormat : str, optional
        The format of the digest file. The options are [json]. The default is "json".
    overwrite : bool, optional
        If True overwrites the files, if False does not overwrite the files. The default is False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Make archive lowercase
    archive = os.path.basename(archive).lower()

    # TODO: add support to custom databases
    # Find which kind of archive it will be
    if archive == "dudez":
        config = get_config()
        chosenArchive = config.dudez_archive
    elif archive == "pdbbind":
        config = get_config()
        chosenArchive = config.pdbbind_archive
    else:
        return ocerror.Error.not_supported_archive(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.") # type: ignore

    # TODO: add support to more docking algorithms
    # Check if the docking algorithm is valid
    if dockingAlgorithm not in ["gnina", "vina", "smina", "plants"]:
        return ocerror.Error.not_supported_docking_algorithm(f"Docking software not recognized. Expected ('gnina', 'vina', 'smina', 'plants') and got '{dockingAlgorithm}'.") # type: ignore

    # Get all dirs paths in the database
    ptnDirs = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index']]

    # Create the complex list
    complexList = []
    
    # For each dir in dirs, let's grab all ligands
    for ptnDir in ptnDirs:
        # Parameterize paths
        ligands = f"{ptnDir}/compounds/ligands"
        decoys = f"{ptnDir}/compounds/decoys"
        candidates = f"{ptnDir}/compounds/candidates"

        # Append to the complex list the merged ligandAlternative list with the list with ligands, decoys and candidates. This is made because each receptor must have its own list of ligands, decoys and candidates, otherwise the docking could be done with the same ligands, decoys and candidates for all receptors making everything out of control.
        complexList.append((ptnDir, glob(f"{ligands}/*") + glob(f"{decoys}/*") + glob(f"{candidates}/*")))
    
    # Run docking
    return ocdock.run_docking(complexList, archive, dockingAlgorithm, overwrite, digestFormat)
