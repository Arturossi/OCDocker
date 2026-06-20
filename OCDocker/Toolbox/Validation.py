#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to validate data.

Usage:

import OCDocker.Toolbox.Validation as ocvalidation
'''

# Imports
###############################################################################
import os
import time

from Bio.PDB import MMCIFParser, PDBParser
from typing import Union

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Printing as ocprint

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##
def _safe_print_warning(message: str) -> None:
    '''Print warning safely even when a stubbed Printing module is incomplete.'''

    warn = getattr(ocprint, "print_warning", None)
    if callable(warn):
        warn(message)
        return

    # Fallback to print_error for test stubs that only expose error logging.
    err = getattr(ocprint, "print_error", None)
    if callable(err):
        err(f"WARNING: {message}")
        return

    # Last resort to keep behavior observable without crashing.
    print(f"WARNING: {message}")

def _safe_print_error(message: str) -> None:
    '''Print error safely even when a stubbed Printing module is incomplete.'''

    err = getattr(ocprint, "print_error", None)
    if callable(err):
        err(message)
        return

    print(f"ERROR: {message}")


## Public ##


def is_algorithm_allowed(path: str) -> bool:
    '''Finds if the given dir is a folder from an allowed algorithm.

    Parameters
    ----------
    path : str
        Path to the dir which will be tested.
        The algorithm list and their shortcodes:
        - AffinityPropagation: ap
        - AgglomerativeClustering: ac
        - Birch: bi
        - DBSCAN: db
        - KMeans:  km
        - MeanShift: ms
        - MiniBatchKMeans: mb
        - NoCluster: na
        - OPTICS: op
        - SpectralClustering: sc

    Returns
    -------
    bool
        True if the dir is an allowed algorithm, False otherwise.
    '''

    # Allowed algorithms
    allowed = ["ap", "ac", "bi", "db", "km", "ms", "mb", "na", "op", "sc"]
    return path.split(os.path.sep).pop() in allowed


def is_molecule_valid(molecule: str) -> bool:
    '''Check if a molecule is valid (protein or ligand).

    Parameters
    ----------
    molecule : str
        The molecule to be checked.

    Returns
    -------
    bool
        True if the molecule is valid, False otherwise.
    '''

    # Check if file exists
    if os.path.isfile(molecule):
        # Check which is its extension to use the correct function
        extension = os.path.splitext(molecule)[1].lower()
        # Test if the molecule should be loaded with biopython or rdkit
        if molecule.lower().endswith((".cif", ".mmcif", ".pdb")):
            try:
                # Now we know that it is a file path, check which is its extension to use the correct function
                extension = os.path.splitext(molecule)[1].lower()
                # Choose the parser based on extension
                parser: PDBParser | MMCIFParser
                if extension == ".pdb":
                    parser = PDBParser()
                elif extension in [".cif", ".mmcif"]:
                    parser = MMCIFParser()
                else:
                    # Not suitable extension, so... say False!!!
                    return False
                # Parse it
                _ = parser.get_structure("Please, be ok", molecule)
                # If no problems occur, the molecule should be fine
                return True
            except (OSError, IOError, ValueError, AttributeError, ImportError):
                # Uh oh, some problem has been found
                return False
        elif type(validate_obabel_extension(molecule)) == str:
            try:
                # Import RDKit lazily to avoid hard dependency at import time
                from rdkit import Chem
                # Check if the extension is within the supported ones, if yes, parse it
                if extension == ".mol2":
                    parsed_mol = Chem.rdmolfiles.MolFromMol2File(molecule, sanitize = True)
                    if parsed_mol is None:
                        return False
                elif extension == ".sdf":
                    supplier = Chem.rdmolfiles.SDMolSupplier(molecule, sanitize = True)
                    if supplier is None:
                        return False
                    if not any(mol is not None for mol in supplier):
                        return False
                elif extension == ".mol":
                    parsed_mol = Chem.rdmolfiles.MolFromMolFile(molecule, sanitize = True)
                    if parsed_mol is None:
                        return False
                elif extension == ".pdbqt":
                    # RDKit's PDB parser can misread PDBQT atom types (e.g., "A") as elements.
                    # Use OpenBabel to validate PDBQT files instead.
                    try:
                        from openbabel import openbabel
                        ob_conversion = openbabel.OBConversion()
                        ob_conversion.SetInFormat("pdbqt")
                        ob_mol = openbabel.OBMol()
                        if not ob_conversion.ReadFile(ob_mol, molecule):
                            return False
                        if ob_mol.NumAtoms() <= 0:
                            return False
                    except Exception:
                        return False
                elif extension in [".smi", ".smiles"]:
                    # Read SMILES string from file and parse
                    try:
                        with open(molecule, "r") as f:
                            smi = f.read().strip().split()[0]
                    except (OSError, IOError, FileNotFoundError, IndexError):
                        return False
                    parsed_mol = Chem.rdmolfiles.MolFromSmiles(smi, sanitize = True)
                    if parsed_mol is None:
                        return False
                else:
                    # Not suitable extension, so... say False!!!!
                    return False
                # If no problems occur, the molecule should be fine
                return True
            except (OSError, IOError, ValueError, AttributeError, ImportError):
                # Uh oh, some problem has been found
                return False
    # No file, so it is False
    return False


def is_molecule_valid_with_retry(molecule: str, retries: int = 5, delay: float = 1.0) -> bool:
    '''Check if a molecule is valid, retrying when the file is empty or mid-write.

    Parameters
    ----------
    molecule : str
        The molecule to be checked.
    retries : int, optional
        Number of read attempts before giving up. Default is 5.
    delay : float, optional
        Delay in seconds between attempts. Default is 1.0.

    Returns
    -------
    bool
        True if the molecule is valid within the retry window, False otherwise.
    '''

    attempts = max(1, retries)
    for attempt in range(attempts):
        size_ok = False
        if os.path.isfile(molecule):
            try:
                size_ok = os.path.getsize(molecule) > 0
            except OSError:
                size_ok = False
        if size_ok:
            if is_molecule_valid(molecule):
                return True
        if attempt < attempts - 1 and delay > 0:
            time.sleep(delay)
    return False


def validate_digest_extension(digestPath: str, digestFormat: str) -> bool:
    """Validates the digest extension.

    Parameters
    ----------
    digestPath : str
        The digest file path.
    digestFormat : str
        The format of the digest file. The options are: [ json (default), hdf5 (not implemented) ]

    Returns
    -------
    bool
        If the extension is supported or not.
    """

    # Supported extensions for digest file
    supportedExtensions = ["json"]

    # Check if the format options is valid
    if not digestFormat.lower() in supportedExtensions:
        _safe_print_warning(
            f"The format '{digestFormat}' is not supported. Trying to determine its extension from the file '{digestPath}'."
        )
        # Get the extension from the file
        digestFormat = digestPath.split(".")[-1]
        # Check if the extension is valid
        if not digestFormat.lower() in supportedExtensions:
            _safe_print_error(
                f"The format '{digestFormat}' is not supported. The supported formats are: {supportedExtensions}"
            )
            return False
        return True
    return True


def validate_obabel_extension(path: str) -> Union[str, int]:
    '''Validate the input file extension to ensure the compability with obabel lib.

    Parameters
    ----------
    path : str
        Path to the input file.

    Returns
    -------
    str | int
        The exit code of the command (based on the Error.py code table) if fails or the extension otherwise.
    '''

    supportedExtensions = [
                            'acesin', 'adf', 'alc', 'ascii', 'bgf', 'box', 'bs', 'c3d1', 'c3d2', 'cac',
                            'caccrt', 'cache', 'cacint', 'can', 'cdjson', 'cdxml', 'cht', 'cif', 'ck', 'cml',
                            'cmlr', 'cof', 'com', 'confabreport', 'CONFIG', 'CONTCAR', 'CONTFF', 'copy', 'crk2d', 'crk3d',
                            'csr', 'cssr', 'ct', 'cub', 'cube', 'dalmol', 'dmol', 'dx', 'ent', 'exyz',
                            'fa', 'fasta', 'feat', 'fh', 'fhiaims', 'fix', 'fps', 'fpt', 'fract', 'fs',
                            'fsa', 'gamin', 'gau', 'gjc', 'gjf', 'gpr', 'gr96', 'gro', 'gukin', 'gukout',
                            'gzmat', 'hin', 'inchi', 'inchikey', 'inp', 'jin', 'k', 'lmpdat', 'lpmd', 'mcdl',
                            'mcif', 'MDFF', 'mdl', 'ml2', 'mmcif', 'mmd', 'mmod', 'mna', 'mol', 'mol2',
                            'mold', 'molden', 'molf', 'molreport', 'mop', 'mopcrt', 'mopin', 'mp', 'mpc',
                            'mpd', 'mpqcin', 'mrv', 'msms', 'nul', 'nw', 'orcainp', 'outmol', 'paint',
                            'pcjson', 'pcm', 'pdb', 'pdbqt', 'png', 'pointcloud', 'POSCAR', 'POSFF', 'pov',
                            'pqr', 'pqs', 'qcin', 'report', 'rinchi', 'rsmi', 'rxn', 'sd', 'sdf',
                            'smi', 'smiles', 'stl', 'svg', 'sy2', 'tdd', 'text', 'therm', 'tmol',
                            'txt', 'txyz', 'unixyz', 'VASP', 'vmol', 'xed', 'xyz', 'yob', 'zin'
                          ]
    extension = os.path.splitext(path)[1][1:]

    if extension in supportedExtensions:
        return extension
    return ocerror.Error.unsupported_extension(message=f"Unsupported extension for input molecule file! Supported extensions are '{' '.join(supportedExtensions)}' and got '{extension}'.")
