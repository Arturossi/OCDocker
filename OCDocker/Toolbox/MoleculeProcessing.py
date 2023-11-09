#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to extract and process information
of any kind of molecule.

They are imported as:

import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
'''

# Imports
###############################################################################
import os

from spyrmsd import io, rmsd
from threading import Lock
from typing import Dict, List, Tuple, Union

from OCDocker.Initialise import *
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##
def split_poses(ligand: str, ligandName: str, outPath: str, suffix: str = "", logFile: str = "") -> Union[int, Tuple[int, str]]:
    '''Split the input ligand into its poses.

    Parameters
    ----------
    ligand : str
        The path to the input ligand.
    ligandName : str
        The name of the input ligand.
    outPath : str
        The path to the output folder.
    suffix : str, optional
        The suffix to be added to the output files, by default "".
    logFile : str, optional
        The path to the log file, by default "".
    
    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Split the input ligand
    cmd = [vina_split, "--input", ligand, "--flex", "''", "--ligand", f"{outPath}/{ligandName}{suffix}"]

    # Print verbosity
    ocprint.printv(f"Spliting the ligand '{ligand}'.")

    # Run the command
    return ocrun.run(cmd, logFile = logFile)

def get_rmsd_matrix(molecules: List[str]) -> Dict[str, Dict[str, float]]:
    '''Get the rmsd matrix between a list of molecules.

    Parameters
    ----------
    molecules : List[str]
        The list of molecules.

    Returns
    -------
    Dict[str, Dict[str, float]]
        The rmsd matrix.
    '''

    # Initialise the rmsd matrix
    rmsdMatrix = {}

    # For each molecule in molecules
    for molecule in molecules:
        # Initialise the row of the rmsd matrix
        rmsdRow = {}

        # For each molecule in molecules
        for otherMolecule in molecules:
            # If the molecule is the same as the otherMolecule
            if molecule == otherMolecule:
                # Append 0 to the row
                rmsdRow[otherMolecule] = 0
            else:
                # Get the rmsd between the molecule and the other molecule
                tmpMolecule = get_rmsd(molecule, otherMolecule)
                # Get the rmsd between the molecule and the other molecule
                rmsdRow[otherMolecule] = tmpMolecule if isinstance(tmpMolecule, float) else tmpMolecule[0] # type: ignore

        # Append the row to the rmsd matrix
        rmsdMatrix[molecule] = rmsdRow 

    # Return the rmsd matrix
    return rmsdMatrix

def get_rmsd(reference: str, molecule: str) -> Union[List[float], float]:
    '''Get the rmsd between a reference and a molecule file (it supports more than one molecule in this second file).

    Parameters
    ----------
    reference : str
        The reference file.
    molecule : str
        The molecule file.

    Returns
    -------
    List[float] | float
        The rmsd between the reference and the molecule file.
    '''

    # Load reference
    ref = io.loadmol(reference)
    # Remove its hydrogens
    ref.strip()
    # Load all molecules (if only one, a list with a single element will be generated)
    mols = io.loadallmols(molecule)
    # For each molecule in molecules
    for mol in mols:
        # Remove its hydrogens
        mol.strip() # type: ignore

    # Get the reference and molecules coordinates
    refCoordinates = ref.coordinates
    molCoordinates = [mol.coordinates for mol in mols] # type: ignore

    # Get the reference and molecules atomicnums
    refAtmNum = ref.atomicnums
    molAtmNum = mols[0].atomicnums # type: ignore

    # Get the reference and molecules adjacency_matrix
    refAdjMat = ref.adjacency_matrix
    molAdjMat = mols[0].adjacency_matrix # type: ignore

    # Return the symmetric rmsd (account for symmetry because it is important)
    return rmsd.symmrmsd(refCoordinates, molCoordinates, refAtmNum, molAtmNum, refAdjMat, molAdjMat)

def make_only_ATOM_and_CRYST_pdb(structurePath: str) -> int:
    '''Make a pdb file with only ATOM and CRYST1 records.

    Parameters
    ----------
    structurePath : str
        The path to the structure file.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Initialise hasCryst1 flag
    hasCryst1 = False
    
    # List of lines and dssp lines
    lines = []

    # Check if structurePath is a valid file
    if os.path.isfile(structurePath):
        # Open it (for cleaning)
        with open(structurePath, 'r') as pdbFile:
            # For each line in pdbFile
            for line in pdbFile:
                if not line.startswith("CRYST1") and not hasCryst1:
                    # Set the hasCryst1 flag to True
                    hasCryst1 = True
                    # Add the line to the list
                    lines.append("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")

                # Check if the line starts with ATOM
                if line.startswith("ATOM"):
                    # Check if there is a chain in the line (all the lines should have a chain)
                    if line[21] == " ":
                        # Assume that the protein has only one chain and call it A
                        line = f"{line[:21]}A{line[22:]}"
                    # Add the line to the list
                    lines.append(line)
        # Create a lock for multithreading
        lock = Lock()
        # Start the lock with statement
        with lock:
            # Write the lines to the file
            with open(structurePath, 'w') as pdbFile:
                # Write the lines list to the file
                pdbFile.writelines(lines)
        
        return ocerror.Error.ok()
    else:
        return ocerror.Error.file_not_exist(message = f"The file '{structurePath}' does not exist!", level = ocerror.ReportLevel.ERROR)
