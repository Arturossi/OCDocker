#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to validate data.

They are imported as:

import OCDocker.Toolbox.Validation as ocvalidation
'''

# Imports
###############################################################################
import os
import rdkit

from Bio.PDB import MMCIFParser, PDBParser
from typing import Union

import OCDocker.Toolbox.Printing as ocprint

from OCDocker.Initialise import *

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
        extension = os.path.splitext(molecule)[1]
        # Test if the molecule should be loaded with biopython or rdkit
        if molecule.endswith((".cif", ".pdb")):
            try:
                # Now we know that it is a file path, check which is its extension to use the correct function
                extension = os.path.splitext(molecule)[1]
                # Choose the parser based on extension
                if extension == ".pdb":
                    parser = PDBParser()
                elif extension == ".cif":
                    parser = MMCIFParser()
                else:
                    # Not suitable extension, so... say False!!!
                    return False
                # Parse it
                _ = parser.get_structure("Please, be ok", molecule)
                # If no problems occur, the molecule should be fine
                return True
            except:
                # Uh oh, some problem has been found
                return False
        elif type(validate_obabel_extension(molecule)) == str:
            try:
                # Check if the extension is within the supported ones, if yes, parse it
                if extension == ".mol2":
                    _ = rdkit.Chem.rdmolfiles.MolFromMol2File(molecule, sanitize = True) # type: ignore
                elif extension == ".sdf":
                    _ = rdkit.Chem.rdmolfiles.SDMolSupplier(molecule, sanitize = True) # type: ignore
                elif extension == ".mol":
                    _ = rdkit.Chem.rdmolfiles.MolFromMolFile(molecule, sanitize = True) # type: ignore
                elif extension == ".pdbqt":
                    _ = rdkit.Chem.rdmolfiles.MolFromMolFile(molecule, sanitize = True) # type: ignore
                elif extension in [".smi", ".smiles"]:
                    _ = rdkit.Chem.rdmolfiles.MolFromSmiles(molecule, sanitize = True) # type: ignore
                else:
                    # Not suitable extension, so... say False!!!!
                    return False
                # If no problems occur, the molecule should be fine
                return True
            except:
                # Uh oh, some problem has been found
                return False
    # No file, so it is False
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
        ocprint.print_warning(f"The format '{digestFormat}' is not supported. Trying to determine its extension from the file '{digestPath}'.")
        # Get the extension from the file
        digestFormat = digestPath.split(".")[-1]
        # Check if the extension is valid
        if not digestFormat.lower() in supportedExtensions:
            ocprint.print_error(f"The format '{digestFormat}' is not supported. The supported formats are: {supportedExtensions}")
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
