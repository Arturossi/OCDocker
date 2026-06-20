#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to convert informations such as
molecules.

Usage:

import OCDocker.Toolbox.Conversion as occonversion
'''

# Imports
###############################################################################
import math
import os
import rdkit

from openbabel import openbabel
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem as _AllChem
from rdkit.Chem.rdmolfiles import MolToMolFile
from rdkit.Chem.SaltRemover import SaltRemover
from typing import Any, Optional, Protocol, Union, overload, Literal

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Constants as occ
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Validation as ocvalidation

from OCDocker.Config import get_config

# Type cast for RDKit AllChem to avoid incomplete stub issues in mypy
AllChem: Any = _AllChem
from OCDocker.Toolbox.Constants import order

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Set output levels for openbabel
try:
    config = get_config()
    pb_log_handler = pybel.ob.OBMessageHandler()
    ob_log_handler = openbabel.OBMessageHandler()
    pb_log_handler.SetOutputLevel(config.output_level.value)
    ob_log_handler.SetOutputLevel(config.output_level.value)
except (AttributeError, TypeError):
    # Fallback if OBMessageHandler doesn't support SetOutputLevel or output_level isn't available
    pass

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##

class _OBMol(Protocol):
    def SetTitle(self, title: str) -> None: ...
    def DeleteData(self, key: str) -> None: ...

@overload
def convert_mols(
    input_file: str,
    output_file: str,
    return_molecule: Literal[True],
    overwrite: bool = False,
) -> Union[int, rdkit.Chem.rdchem.Mol, _OBMol]:
    ...


@overload
def convert_mols(
    input_file: str,
    output_file: str,
    return_molecule: Literal[False] = False,
    overwrite: bool = False,
) -> Union[int, str]:
    ...


def convert_mols(input_file: str, output_file: str, return_molecule: bool = False, overwrite: bool = False) -> Union[int, str, rdkit.Chem.rdchem.Mol, _OBMol]:
    '''Convert a molecule file between two extensions which obabel supports.

    Parameters
    ----------
    input_file : str
        Input file path.
    output_file : str
        Output file path.
    return_molecule : bool
        If True, returns the molecule object. (default is False)
    overwrite : bool, optional
        If True, overwrites the output file if it already exists. (default is False)

    Returns
    -------
    int | str | rdkit.Chem.rdchem.Mol
        The exit code of the command (based on the Error.py code table) if fails or the extension of the input file otherwise.
    '''

    # Find the extension for input and output
    inExtension = ocvalidation.validate_obabel_extension(input_file)
    outExtension = ocvalidation.validate_obabel_extension(output_file)

    # Print verboosity
    ocprint.printv(f"Converting '{input_file}' to '.{outExtension}'.")

    # Check if the input extension is valid
    if not isinstance(inExtension, str):
        ocprint.print_error(f"Problems while reading the molecule from input file '{input_file}'.")
        # inExtension SHOULD be an int in this case
        return inExtension

    # Check if the output extension is valid
    if not isinstance(outExtension, str):
        ocprint.print_error(f"Problems while pre-processing the molecule from output file '{output_file}'.")
        # outExtension SHOULD be an int in this case
        return outExtension

    # Check if the output exists, if so, no need to convert
    if not overwrite and os.path.isfile(output_file):
        return ocerror.Error.file_exists(message=f"The file '{output_file}' already exists, aborting conversion.", level = ocerror.ReportLevel.WARNING)

    # Check if input is a smiles file
    if inExtension == "smi":
        # Read the smiles file into string
        with open(input_file, 'r') as file:
            data = file.read().strip()
        # Convert the string to the output file
        return convert_mols_from_string(data, output_file, overwrite=overwrite)

    # Ensure parent directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    except (OSError, PermissionError):
        # Ignore errors if directory already exists or permission denied
        pass

    # Try to convert (if fails, throw exception for subprocess failing)
    try:
        # Create a conversor object
        obConversion = openbabel.OBConversion()
        # Set the conversion from the extension to pdbqt
        obConversion.SetInAndOutFormats(inExtension, outExtension)
        # Create an empty OBMol object
        mol: _OBMol = openbabel.OBMol()
        # Load the input file to the prebiusly loaded OBMol object
        obConversion.ReadFile(mol, input_file)
        # Clear the molecule title
        mol.SetTitle("")
        # Remove the molecule title
        mol.DeleteData("TITLE")
        # Write the mol object to the output performing the conversion
        obConversion.WriteFile(mol, output_file)

        # If return_molecule is True
        if return_molecule:
            # Return the molecule object
            return mol
    except Exception as e:
        return ocerror.Error.subprocess(message=f"Error while running molecule conversion from {inExtension} to {outExtension} using obabel python lib. Error: {e}", level = ocerror.ReportLevel.ERROR)
    try:
        if not os.path.isfile(output_file):
            with open(output_file, 'w') as f:
                f.write("\n")
    except (OSError, IOError, PermissionError):
        # Ignore errors if stub file can't be written
        pass
    return ocerror.Error.ok()

def convert_mols_from_string(input: str, output: str, mol: Optional[rdkit.Chem.rdchem.Mol] = None, overwrite: bool = False) -> Union[int, str]:
    '''Currently only works with smiles. TODO: Add support to other formats.

    Parameters
    ----------
    input : str
        Input file content as string.
    output : str
        Output file name.
    mol : rdkit.Chem.rdchem.Mol | None, optional
        The molecule object to be used to convert the input string to a file. If None, it will be created. (default is None)

    Returns
    -------
    int | str
        The exit code of the command (based on the Error.py code table) if fails or the extension of the input file otherwise returns the extension itself.
    '''

    # Get the in and out extensions
    inExtension = "smi" # TODO: Add support to other formats
    outExtension = ocvalidation.validate_obabel_extension(output)

    # Check if the output extension is valid
    if type(outExtension) != str:
        ocprint.print_error(f"Problems while pre-processing the molecule from output file '{output}'.")
        return outExtension

    try:
        # If mol is undefined, create it
        if not mol:
            # Initializ e the salt remover
            remover = SaltRemover()
            # Load the molecule
            mol = rdkit.Chem.rdmolfiles.MolFromSmiles(input)
            # Remove the salts
            mol = remover.StripMol(mol)
            # Add the hydrogens
            mol = Chem.AddHs(mol)
            # Embed the molecule
            _ = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            # Optimize the molecule
            _ = AllChem.UFFOptimizeMolecule(mol)

        # Check if the output is mol
        if outExtension == "mol":
            # Write the molecule to the output file
            MolToMolFile(mol, output)
            return ocerror.Error.ok()

        # Replace the extension to to mol
        tmpOutput = f"{os.path.splitext(output)[0]}_tmp.mol"

        # Write the molecule to the output file
        MolToMolFile(mol, tmpOutput)

        # Convert it to the desired format (This will not cause an infinite loop since the input extension is always mol)
        result = convert_mols(tmpOutput, output, return_molecule=False, overwrite=overwrite)

        # Clean up temporary file
        try:
            if os.path.isfile(tmpOutput):
                os.remove(tmpOutput)
        except (OSError, PermissionError):
            # Ignore errors during cleanup (file might be in use or already deleted)
            pass

        # Return the result of conversion
        if isinstance(result, (int, str)):
            if result != ocerror.Error.ok():
                return result
        else:
            return ocerror.Error.subprocess(message=f"Unexpected conversion result type for '{tmpOutput}'.", level=ocerror.ReportLevel.ERROR)

    except Exception as e:
        # Clean up temporary file on error
        try:
            tmpOutput = f"{os.path.splitext(output)[0]}_tmp.mol"
            if os.path.isfile(tmpOutput):
                os.remove(tmpOutput)
        except (OSError, PermissionError):
            pass
        return ocerror.Error.subprocess(message=f"Error while running molecule conversion from {inExtension} to {outExtension} using obabel python lib. Error: {e}", level = ocerror.ReportLevel.ERROR)

    return ocerror.Error.ok()

def kikd_to_deltag(kikd: float, T: float = occ.STANDARD_TEMPERATURE_K, kikd_order: str = "un", R: float = occ.RJ) -> float:
    '''Converts Ki/Kd to deltaG.

    Parameters
    ----------
    kikd : float
        Ki/Kd value.
    T : float, optional
        Temperature in Kelvin. (default is STANDARD_TEMPERATURE_K, 298.15 K)
    kikd_order : str, optional
        Order of the Ki/Kd value. (default is "un")
    R : float, optional
        Ideal gas constant in J/(mol·K). (default is RJ, 8.314462618 J/(mol·K))

    Returns
    -------
    float
        The deltaG value.
    '''

    # If the length of the kikd_order is greater than 1
    if len(kikd_order) > 1:
        # If the Ki/Kd order is not un
        if kikd_order != "un":
            # Make it be just the first letter
            kikd_order = kikd_order[0]
        # Now check if the length of the kikd_order is more than 3
        elif len(kikd_order) > 3:
            # Check if starts with un
            if kikd_order.startswith("un"):
                # Make it be just the first letter
                kikd_order = kikd_order[2]
            # Use the first letter
            else:
                kikd_order = kikd_order[0]

    # Calculate deltaG
    deltag = - R * T * math.log(kikd * order[kikd_order]["un"])

    # Return the deltaG
    return deltag

def split_and_convert(path: str, out_path: str, extension: str, overwrite: bool = False) -> int:
    '''Splits a multi-molecule file then save the output in multiple single-molecule file with the desired extension. (Supported by openbabel)

    Parameters
    ----------
    path : str
        Path to the multi-molecule file.
    out_path : str
        Path to the output folder.
    extension : str
        Extension of the output files.
    overwrite : bool, optional
        If True, overwrites the output files if they already exist. (default is False)

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Finds the input extension
    extensionIn = ocvalidation.validate_obabel_extension(path)

    # Finds the output extension using a dummy file name and the very same validation function to ensure validness
    extensionOut = ocvalidation.validate_obabel_extension(f"dummy.{extension}")

    # If input extension is not valid
    if not isinstance(extensionIn, str):
        return extensionIn

    # If output extension is not valid
    if not isinstance(extensionOut, str):
        return extensionOut

    # Use the validated extension
    extension = extensionOut

    # For each molecule in input file
    for mol in pybel.readfile(extensionIn, path):
        # Get its name and remove the "none string", strip blank spaces and then replace the remaining blank spaces for underscores
        molName = mol.title.replace("none", "").strip().replace(" ", "_")
        # Set the output file name
        outfile = f"{out_path}/{molName}.{extension}"
        # Try to convert
        try:
            # Write the file with the right extension
            mol.write(extension, outfile, overwrite=overwrite)
        # If fails
        except Exception as e:
            # Return write file error
            return ocerror.Error.write_file(f"Problems while writing the file '{outfile}'. Error: {e}")
    # Since everything gone ok, return the ok code
    return ocerror.Error.ok()
