#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to process the DUDE-Z dataset.

Usage:

import OCDocker.DB.DUDEz as ocdudez
'''

# Imports
###############################################################################
import pandas as pd

from typing import Dict, Union

import OCDocker.DB.baseDB as ocbdb

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

## Public ##
def prepare(overwrite: bool = False, spacing: float = 0.33, sanitize: bool = True) -> None:
    '''Prepares the DUDEz database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated, by default False.
    spacing : float, optional
        The spacing between the grid points, by default 0.33.
    sanitize : bool, optional
        If True, sanitizes the ligands, by default True.

    Raise
    -----
    None
    '''

    # Prepare the rest of the database
    ocbdb.prepare("dudez", overwrite = overwrite, spacing = spacing, sanitize = sanitize)
    # Verify its integrity
    #verify_integrity()
    
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

    return ocbdb.run_docking("dudez", "gnina", overwrite = overwrite)


def run_plants(overwrite: bool = False) -> int:
    '''Runs PLANTS in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated, by default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table)

    Raise
    -----
    None
    '''

    return ocbdb.run_docking("dudez", "plants", overwrite = overwrite)


def run_smina(overwrite: bool = False) -> int:
    '''Runs smina in the whole database.

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

    return ocbdb.run_docking("dudez", "smina", overwrite = overwrite)


def run_vina(overwrite: bool = False) -> int:
    '''Runs vina in the whole database.

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

    return ocbdb.run_docking("dudez", "vina", overwrite = overwrite)
