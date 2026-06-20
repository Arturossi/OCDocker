#!/usr/bin/env python3

# Description
###############################################################################
"""
Shared database import and workflow adapter utilities.

``baseDB.py`` contains shared helpers used to prepare/populate OCDocker
reference databases and coordinate common dataset/database access patterns.
It intentionally remains the shared database base/adapter layer in this cleanup;
its functionality is not being renamed or relocated.

Usage:

import OCDocker.DB.baseDB as ocbdb
"""

# Imports
###############################################################################
import os

from glob import glob

import OCDocker.Error as ocerror

import OCDocker.Toolbox.Printing as ocprint

from OCDocker.Config import get_config

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Classes
###############################################################################


class _LazyPrepareModule:
    def prepare(self, *args, **kwargs):
        import OCDocker.Processing.Preprocessing.Prepare as module

        return module.prepare(*args, **kwargs)


class _LazyDockModule:
    def run_dock(self, *args, **kwargs):
        import OCDocker.Processing.Dock as module

        return module.run_dock(*args, **kwargs)


ocprepare = _LazyPrepareModule()
ocdock = _LazyDockModule()

# Functions
###############################################################################
## Private ##


## Public ##
def prepare(
    archive: str,
    overwrite: bool = False,
    spacing: float = 0.33,
    sanitize: bool = True,
    all_boxes: bool = False,
) -> None:
    """Prepares the database.

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
    """

    # Find which kind of archive it will be
    config = get_config()
    if archive.lower() == "dudez":
        chosenArchive = config.dudez_archive
    elif archive.lower() == "pdbbind":
        chosenArchive = config.pdbbind_archive
    else:
        ocprint.print_error(
            f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}."
        )
        return None

    # Get all paths in the database
    paths = [
        d
        for d in glob(f"{chosenArchive}/*")
        if os.path.basename(d.split(os.path.sep)[-1]) not in ["index"]
    ]

    # Generate boxes for all receptors
    ocprint.printv("Generating information regarding possible ligand site.")

    # Prepare it
    ocprepare.prepare(paths, overwrite, archive, sanitize, spacing, all_boxes=all_boxes)

    return None


def run_docking(
    archive: str,
    dockingAlgorithm: str,
    digestFormat: str = "json",
    overwrite: bool = False,
    all_boxes: bool = False,
) -> int:
    """Run docking.

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
    """

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
        return ocerror.Error.not_supported_archive(
            f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}."
        )

    # TODO: add support to more docking algorithms
    # Check if the docking algorithm is valid
    if dockingAlgorithm not in ["gnina", "vina", "smina", "plants"]:
        return ocerror.Error.not_supported_docking_algorithm(
            f"Docking software not recognized. Expected ('gnina', 'vina', 'smina', 'plants') and got '{dockingAlgorithm}'."
        )

    # Get all dirs paths in the database
    ptnDirs = [
        d
        for d in glob(f"{chosenArchive}/*")
        if os.path.basename(d.split(os.path.sep)[-1]) not in ["index"]
    ]

    # Create the complex list
    complexList = []

    # For each dir in dirs, let's grab all ligands
    for ptnDir in ptnDirs:
        # Parameterize paths
        ligands = f"{ptnDir}/compounds/ligands"
        decoys = f"{ptnDir}/compounds/decoys"
        candidates = f"{ptnDir}/compounds/candidates"

        # Append to the complex list the merged ligandAlternative list with the list with ligands, decoys and candidates. This is made because each receptor must have its own list of ligands, decoys and candidates, otherwise the docking could be done with the same ligands, decoys and candidates for all receptors making everything out of control.
        complexList.append(
            (
                ptnDir,
                glob(f"{ligands}/*") + glob(f"{decoys}/*") + glob(f"{candidates}/*"),
            )
        )

    # Run docking
    return ocdock.run_dock(
        complexList,
        archive,
        dockingAlgorithm,
        overwrite,
        digestFormat,
        all_boxes=all_boxes,
    )
