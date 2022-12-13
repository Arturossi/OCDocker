#!/usr/lib/python3

# Imports
###############################################################################
import contextlib
import datetime
import inspect
import os
import mmap
import pickle
import rdkit
import shutil
import subprocess
import tarfile
import urllib.request

from Bio.PDB import * 
from glob import glob
from tqdm import tqdm
from openbabel import openbabel
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToMolFile
from rdkit.Chem.SaltRemover import SaltRemover
from spyrmsd import io, rmsd
from threading import Lock
from typing import Any, Dict, Generator, List, Tuple, Union

from OCDocker.Initialise import *


# Set output levels for openbabel
pb_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler = openbabel.OBMessageHandler()
pb_log_handler.SetOutputLevel(args.output_level)
ob_log_handler.SetOutputLevel(args.output_level)

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

# Description
###############################################################################
'''
Sets of classes and functions that are used by the OCDocker in many sources.

They are imported as:

import OCDocker.Toolbox as octools
'''

# Classes
###############################################################################
class DownloadProgressBar(tqdm):
    """Deal with the progress bar to track download. Extends the tqdm class."""
    
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = 0) -> None:
        '''Update the progress bar.

        Parameters
        ----------
        b : int, optional
            Number of blocks transferred so far [1]
        bsize : int, optional
            Size of each block (in tqdm units) [1]
        tsize : int, optional
            Total size (in tqdm units). If [None] remains unchanged.

        Returns
        -------
        None

        Raises
        ------
        None
        '''

        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# Functions
###############################################################################
## Private ##

## Public ##

### Print functions

def printv(message: str) -> None:
    '''Function to print if verbosity mode is set.

    Parameters
    ----------
    message : str
        Message to be printed.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    if args.output_level >= 5:
        today = datetime.datetime.now()
        print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {message}")
    return

def print_info(message: str, force = False) -> None:
    '''Function to print info.

    Parameters
    ----------
    message : str
        Message to be printed.
    force : bool, optional
        Forces the system to print the message, even if output_level is turning it off (USE WITH CAUTION!!!).

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    if args.output_level >= 2 or force:
        today = datetime.datetime.now()
        if args.output_level >= 4:
            print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['c']}INFO{clrs['n']}: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.") # type: ignore
        else:
            print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['c']}INFO{clrs['n']}: {message}")
    return

def print_success(message: str, force: bool = False) -> None:
    '''Print success. [DEPRECATED]

    Parameters
    ----------
    message : str
        Message to be printed.
    force : bool, optional
        Forces the system to print the message, even if output_level is turning it off (USE WITH CAUTION!!!).

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    if args.output_level >= 3 or force:
        today = datetime.datetime.now()
        if args.output_level >= 4:
            print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['g']}SUCCESS{clrs['n']}: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.") # type: ignore
        else:
            print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['g']}SUCCESS{clrs['n']}: {message}")
    return

def print_warning(message: str, force: bool = False) -> None:
    '''Function to print warning. [DEPRECATED]

    Parameters
    ----------
    message : str
        Message to be printed.
    force : bool, optional
        Forces the system to print the message, even if output_level is turning it off (USE WITH CAUTION!!!).
        
    Returns
    -------
    None

    Raises
    ------
    None
    '''

    if args.output_level >= 1 or force:
        today = datetime.datetime.now()
        if args.output_level == 4:
            print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['y']}WARNING{clrs['n']}: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.") # type: ignore
        else:
            print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['y']}WARNING{clrs['n']}: {message}")
    return

def print_error(message: str, force: bool = False) -> None:
    '''Print error. [DEPRECATED]

    Parameters
    ----------
    message : str
        Message to be printed.
    force : bool, optional
        Forces the system to print the message, even if output_level is turning it off (USE WITH CAUTION!!!).

    Returns
    -------
    None
    
    Raises
    ------
    None
    '''

    if args.output_level > 0 or force:
        today = datetime.datetime.now()
        if args.output_level == 4:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] {clrs['r']}ERROR{clrs['n']}: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.") # type: ignore
        else:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] {clrs['r']}ERROR{clrs['n']}: {message}")
    return

def print_info_log(message: str, logfile:str, mode: str = "a") -> None:
    '''Function to print info into log.

    Parameters
    ----------
    message : str
        Message to be printed.
    logfile : str
        Log file to be used.
    mode : str, optional
        Mode to open the file. Default is "a" (append).

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] INFO: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n") # type: ignore
    return

def print_success_log(message: str, logfile: str, mode: str = "a") -> None:
    '''Function to print success into log.

    Parameters
    ----------
    message : str
        Message to be printed.
    logfile : str
        Log file to be used.
    mode : str, optional
        Mode to open the file. Default is "a" (append).

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] SUCCESS: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n") # type: ignore
    return

def print_warning_log(message: str, logfile: str, mode: str = "a") -> None:
    '''Function to print warning into log.

    Parameters
    ----------
    message : str
        Message to be printed.
    logfile : str
        Log file to be used.
    mode : str, optional
        Mode to open the file. Default is "a" (append).

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] WARNING: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n") # type: ignore
    return

def print_error_log(message: str, logfile: str, mode: str = "a") -> None:
    '''Function to print error into log.

    Parameters
    ----------
    message : str
        Message to be printed.
    logfile : str
        Log file to be used.
    mode : str, optional
        Mode to open the file. Default is "a" (append).

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] ERROR: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n") # type: ignore
    return

def print_section(n: int, name: str, logName = "OCDocker_Progress.out") -> None:
    '''Print the section header and write progress to the progress file.

    Parameters
    ----------
    n : int
        Section number.
    name : str
        Section name (empty string for no log).
    logName : str, optional
        Log file name. Default is "OCDocker_Progress.out".

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Print a nice section header
    print(f"\n{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
          f"{clrs['r']}| " +
          f"{clrs['y']}S{clrs['r']}|" +
          f"{clrs['y']}E{clrs['r']}|" +
          f"{clrs['y']}C{clrs['r']}|" +
          f"{clrs['y']}T{clrs['r']}|" +
          f"{clrs['y']}I{clrs['r']}|" +
          f"{clrs['y']}O{clrs['r']}|" +
          f"{clrs['y']}N{clrs['r']}|" +
          f"{clrs['c']} {str(n)}{clrs['r']} | " +
          f"{clrs['c']}{str(name)}\n" +
          f"{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
          clrs['n'])
    # Check if the section should be logged
    if name:
        # Check if is the Runtime Arguments section
        if name == "Runtime Arguments":
            with open(logName, "w") as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')}: Starting new OCDocker run\n") # type: ignore
        else:
            with open(logName, "a") as f:
                f.write(f"\n{datetime.now().strftime('%H:%M:%S')}: {str(name)}...\n") # type: ignore
    return

def section(n: int, name: str) -> str:
    '''Return the section header.

    Parameters
    ----------
    n : int
        Section number.
    name : str
        Section name.

    Returns
    -------
    str
        Section header.

    Raises
    ------
    None
    '''

    # Create a nice section header to return
    section_string = str(f"\n{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
                         f"{clrs['r']}| "+
                         f"{clrs['y']}S{clrs['r']}|" +
                         f"{clrs['y']}E{clrs['r']}|" +
                         f"{clrs['y']}C{clrs['r']}|" +
                         f"{clrs['y']}T{clrs['r']}|" +
                         f"{clrs['y']}I{clrs['r']}|" +
                         f"{clrs['y']}O{clrs['r']}|" +
                         f"{clrs['y']}N{clrs['r']}|" +
                         f"{clrs['c']} {str(n)}{clrs['r']} | " +
                         f"{clrs['c']}{str(name)}\n" +
                         f"{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
                         clrs['n'])

    return section_string

def print_subsection(n: int, name: str, logName: str = "OCDocker_Progess.out") -> None:
    '''Print the subsection header in progress file.

    Parameters
    ----------
    n : int
        Subsection number.
    name : str
        Subsection name.
    logName : str
        Log file name. Default is "OCDocker_Progress.out".

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Print a nice subsection header
    print(f"\n{clrs['r']}|" +
          f"{clrs['y']}S" +
          f"{clrs['y']}u" +
          f"{clrs['y']}b" +
          f"{clrs['y']}s" +
          f"{clrs['y']}e" +
          f"{clrs['y']}c" +
          f"{clrs['y']}t" +
          f"{clrs['y']}o" +
          f"{clrs['y']}i" +
          f"{clrs['y']}n" +
          f"{clrs['c']} {str(n)}{clrs['r']}| " +
          f"{clrs['c']}{str(name)}\n" +
          f"{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
          clrs['n'])

    if name:
        with open("OCDocker_Progress.out", "a") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')}: {str(name)}...\n") # type: ignore
    return

def subsection(n: int, name: str) -> str:
    '''Return the subsection header.

    Parameters
    ----------
    n : int
        Subsection number.
    name : str
        Subsection name.

    Returns
    -------
    str
        Subsection header.

    Raises
    ------
    None
    '''

    # Create a nice subsection header to return
    subsection_string = str(f"\n{clrs['r']}|" +
                            f"{clrs['y']}S" +
                            f"{clrs['y']}u" +
                            f"{clrs['y']}b" +
                            f"{clrs['y']}s" +
                            f"{clrs['y']}e" +
                            f"{clrs['y']}c" +
                            f"{clrs['y']}t" +
                            f"{clrs['y']}i" +
                            f"{clrs['y']}o" +
                            f"{clrs['y']}n" +
                            f"{clrs['c']} {str(n)}{clrs['r']}| " +
                            f"{clrs['c']}{str(name)}\n" +
                            f"{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
                            clrs['n'])

    return subsection_string

def print_sorry()-> None:
    '''Function to print sorry message.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Print a nice looking sorry message :/
    print(f"**We are {clrs['y']}t{clrs['r']}e"+
          f"{clrs['y']}r{clrs['r']}r{clrs['y']}i"+
          f"{clrs['r']}b{clrs['y']}l{clrs['r']}y"+
          f"{clrs['n']} sorry... =(\n")
    return None

### Conversion functions

def convertMolsFromString(input: str, output: str, mol: Union[rdkit.Chem.rdchem.Mol, None] = None) -> Union[int, str]: # type: ignore
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

    Raises
    ------
    None
    '''

    # Get the in and out extensions 
    inExtension = "smi" # TODO: Add support to other formats
    outExtension = validate_obabel_extension(output)

    # Check if the output extension is valid
    if type(outExtension) != str:
        print_error(f"Problems while pre-processing the molecule from output file '{output}'.")
        return outExtension

    try:
        # If mol is undefined, create it
        if not mol:
            # Initializ e the salt remover
            remover = SaltRemover()
            # Load the molecule
            mol = rdkit.Chem.rdmolfiles.MolFromSmiles(input) # type: ignore
            # Remove the salts
            mol = remover.StripMol(mol)
            # Add the hydrogens
            mol = Chem.AddHs(mol) # type: ignore
            # Embed the molecule
            _ = AllChem.EmbedMolecule(mol, AllChem.ETKDG()) # type: ignore
            # Optimize the molecule
            _ = AllChem.UFFOptimizeMolecule(mol) # type: ignore
        
        # Check if the output is mol
        if outExtension == "mol":
            # Write the molecule to the output file
            MolToMolFile(mol, output)
            return errors.ok()
        
        # Replace the extension to to mol
        tmpOutput = f"{os.path.splitext(output)[0]}_tmp.mol"
        
        # Write the molecule to the output file
        MolToMolFile(mol, tmpOutput)

        # Convert it to the desired format (This will not cause an infinite loop since the input extension is always mol)
        convertMols(tmpOutput, output)
        
    except Exception as e:
        return errors.subprocess(message=f"Error while running molecule conversion from {inExtension} to {outExtension} using obabel python lib. Error: {e}", level="error")

    return errors.ok()

def convertMols(input: str, output: str) -> Union[int, str]:
    '''Convert a molecule file between two extensions which obabel supports.

    Parameters
    ----------
    input : str
        Input file name.
    output : str
        Output file name.

    Returns
    -------
    int | str
        The exit code of the command (based on the Error.py code table) if fails or the extension of the input file otherwise.
        
    Raises
    ------
    None
    '''

    # Find the extension for input and output
    inExtension = validate_obabel_extension(input)
    outExtension = validate_obabel_extension(output)

    # Print verboosity
    printv(f"Converting '{input}' to '.{outExtension}'.")

    # Check if the input extension is valid
    if type(inExtension) != str:
        print_error(f"Problems while reading the molecule from input file '{input}'.")
        # inExtension SHOULD be an int in this case
        return inExtension

    # Check if the output extension is valid
    if type(outExtension) != str:
        print_error(f"Problems while pre-processing the molecule from output file '{output}'.")
        # outExtension SHOULD be an int in this case
        return outExtension

    # Check if the output exists, if so, no need to convert
    if os.path.isfile(output):
        return errors.file_exists(message=f"The file '{output}' already exists, aborting conversion.", level="warn")

    # Check if input is a smiles file
    if inExtension == "smi":
        # Read the smiles file into string
        with open(input, 'r') as file:
            data = file.read().strip()
        # Convert the string to the output file
        return convertMolsFromString(data, output)

    # Try to convert (if fails, throw exception for subprocess failing)
    try:
        # Create a conversor object
        obConversion = openbabel.OBConversion()
        # Set the conversion from the extension to pdbqt
        obConversion.SetInAndOutFormats(inExtension, outExtension)
        # Create an empty OBMol object
        mol = openbabel.OBMol()
        # Load the input file to the prebiusly loaded OBMol object
        obConversion.ReadFile(mol, input)
        # Write the mol object to the output performing the conversion
        obConversion.WriteFile(mol, output)
    except Exception as e:
        return errors.subprocess(message=f"Error while running molecule conversion from {inExtension} to {outExtension} using obabel python lib. Error: {e}", level="error")
    return errors.ok()

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

    Raises
    ------
    None
    '''

    # Finds the input extension
    extensionIn = validate_obabel_extension(path)

    # If input extension is not valid
    if type(extension) != str:
        # Return the unsupported_extension
        return errors.unsupported_extension(f"Unsupported extension provided while spliting '{path}' file. Supported extensions are the one supported by OpenBabel.", "error")

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
            return errors.write_file(f"Problems while writing the file '{outfile}'. Error: {e}")
    # Since everything gone ok, return the ok code
    return errors.ok()

### Pickle functions

def to_pickle(filePath: str, data: Any) -> int:
    '''Pickle a dict in a given path.

    Parameters
    ----------
    filePath : str
        Path to the pickle file.
    data : Any
        Data to be pickled.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    try:
        with open(filePath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        return errors.write_file(f"Problems while pickling the file '{filePath}'. Error: {e}")
    return errors.ok()

def from_pickle(filePath: str) -> Union[int, Any]:
    '''Unpickle a pickle file into a dict.

    Parameters
    ----------
    filePath : str
        Path to the pickle file.

    Returns
    -------
    int | Any
        The exit code of the command (based on the Error.py code table) if fails or the unpickled data otherwise.

    Raises
    ------
    None
    '''

    data = None
    try:
        with open(filePath, 'rb') as handle:
            data = pickle.load(handle)
    except Exception as e:
        return errors.read_file(f"Problems while unpickling the file '{filePath}'. Error: {e}")
    return data

### Log functions

def clear_past_logs() -> None:
    '''Clear past logs entries.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None
    '''
    
    # For each dir in the log dir
    for pastLog in [d for d in glob(f"{logdir}/*") if os.path.isdir(d)]:
        # Extra check for avoid wrong deletions
        if pastLog.endswith("past"):
            # Remove all the folder
            shutil.rmtree(pastLog)
    return None

### Validation functions

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

    Raises
    ------
    None
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
    return errors.unsupported_extension(message=f"Unsupported extension for input molecule file! Supported extensions are '{' '.join(supportedExtensions)}' and got '{extension}'.")

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

    Raises
    ------
    None
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

    Raises
    ------
    None
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

### File manipulation

def lazyread_mmap(file_name: str, decode: str = "utf-8") -> Generator[str, None, None]:
    '''Read a file in sequential order using mmap.

    Parameters
    ----------
    file_name : str
        The file to be read.
    decode : str, optional
        The decode to be used, by default "utf-8"

    Returns
    -------
    Generator[str, None, None]
        A generator with the lines of the file in sequential order.

    Raises
    ------
    None
    '''

    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        with mmap.mmap(read_obj.fileno(), 0, access = mmap.ACCESS_READ) as mmap_obj:
            # Read line by line
            for line in iter(mmap_obj.readline, b''):
                yield line.decode(decode)

def lazyread_reverse_order_mmap(file_name: str, decode: str = "utf-8") -> Generator[str, None, None]:
    '''Read a file in reverse order using mmap.

    Parameters
    ----------
    file_name : str
        The file to be read.
    decode : str, optional
        The decode to be used, by default "utf-8"

    Returns
    -------
    Generator[str, None, None]
        A generator with the lines of the file in reverse order.

    Raises
    ------
    None
    '''

    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        with mmap.mmap(read_obj.fileno(), 0, access = mmap.ACCESS_READ) as mmap_obj:
            # Move the cursor to the end of the file
            mmap_obj.seek(0, os.SEEK_END)
            # Get the current position of pointer i.e eof
            pointer_location = mmap_obj.tell()
            # Create a buffer to keep the last read line
            buffer = bytearray()
            # Loop till pointer reaches the top of the file
            while pointer_location >= 0:
                # Move the file pointer to the location pointed by pointer_location
                mmap_obj.seek(pointer_location)
                # Shift pointer location by -1
                pointer_location = pointer_location - 1
                # read that byte / character
                new_byte = mmap_obj.read(1)
                # If the read byte is new line character then it means one line is read
                if new_byte == b'\n':
                    # Fetch the line from buffer and yield it
                    yield buffer.decode(decode)[::-1]
                    # Reinitialize the byte array to save next line
                    buffer = bytearray()
                else:
                    # If last read character is not eol then add it in buffer
                    buffer.extend(new_byte)
            # As file is read completely, if there is still data in buffer, then its the first line.
            if len(buffer) > 0:
                # Yield the first line too
                yield buffer.decode(decode)[::-1]

def lazyread(file_name: str, decode: str = "utf-8") -> Generator[str, None, None]:
    '''Read a file in sequential order.

    Parameters
    ----------
    file_name : str
        The file to be read.
    decode : str, optional
        The decode to be used, by default "utf-8"

    Returns
    -------
    Generator[str, None, None]
        A generator with the lines of the file in sequential order.

    Raises
    ------
    None
    '''

    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        # Read line by line
        for line in iter(read_obj.readline, b''):
            yield line.decode(decode)

def lazyread_reverse_order(file_name: str, decode: str = "utf-8") -> Generator[str, None, None]:
    '''Read a file in reverse order.

    Parameters
    ----------
    file_name : str
        The file to be read.
    decode : str, optional
        The decode to be used, by default "utf-8"

    Returns
    -------
    Generator[str, None, None]
        A generator with the lines of the file in reverse order.

    Raises
    ------
    None
    '''

    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location - 1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte == b'\n':
                # Fetch the line from buffer and yield it
                yield buffer.decode(decode)[::-1]
                # Reinitialize the byte array to save next line
                buffer = bytearray()
            else:
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)
        # As file is read completely, if there is still data in buffer, then its the first line.
        if len(buffer) > 0:
            # Yield the first line too
            yield buffer.decode(decode)[::-1]

### Other functions

def untar(fname: str, out_path: str = ".", delete: bool = False) -> int:
    '''Untar a file.

    Parameters
    ----------
    fname : str
        The file to be untarred.
    out_path : str, optional
        The path where the file will be untarred.
        Default is the current directory.
    delete : bool, optional
        If True, the tar file will be deleted after the untar process.
        Default is False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Print verboosity
    printv(f"Untarring file '{fname}' to the output '{out_path}'")
    # Check if the file has the right extensions
    if fname.endswith("tar.gz") or fname.endswith(".tgz") or fname.endswith(".gz"):
        try:
            printv("Preparing to untar the file...")
            # open your tar.gz file
            with tarfile.open(name=fname) as tar:
                # Redirect output to tqdm.write
                with redirect_to_tqdm():
                    # Go over each member
                    for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                        # Extract member
                        tar.extract(member=member, path=out_path)
            # Report success on untarring the file
            _ = errors.ok(f"The file {fname} has been {clrs['g']}successfully{clrs['n']} untarred to the dir {out_path}!")
            # If delete flag is set, delete file
            if delete:
                #shutil.rmtree(fname) # remove the files
                os.remove(fname) # remove the files
                return errors.ok(f"The file {fname} has been {clrs['y']}deleted!{clrs['n']}") # Report success on deleting the file
            return errors.ok()
        except Exception as e:
            return errors.untar_file(message=f"{clrs['r']}Failed{clrs['n']} to untar the file {fname}.\n\n{clrs['r']}Error{clrs['n']}: {e}", level="error")
    else:
        # No supported extension has been provided
        return errors.unsupported_extension(message=f"The file {fname} is not a tar.gz file. {clrs['y']}Aborting execution{clrs['n']}", level="error")

def safe_create_dir(dirname: str) -> int:
    '''Create a dir if not exists.

    Parameters
    ----------
    dirname : str
        The dir to be created.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Try to create
    try:
        # If file does not exists
        if not os.path.isdir(dirname):
            # Create it
            os.mkdir(dirname)
            # Print verbosity
            if args.output_level >= 3:
                return errors.ok(f"Successfully created the directory {dirname}")
            return errors.ok()
        else:
            # It exists
            return errors.dir_exists(message=f"The dir '{dirname}' already exists!", level="warn")
    except Exception as e:
        # Some error has occurred
        return errors.create_dir(message=f"Problem found while creating the dir {dirname}: {e}", level="error")
    # This should never appear since all the other paths ends in some kind of return
    return errors.unknown(message=f"What are you expecting for? This message should NEVER appear!!!!!!! Btw problems while creating a dir safetly.", level="error")

def download_url(url: str , out_path: str) -> None:
    '''Download a file from given url.

    Parameters
    ----------
    url : str
        The url to download the file from.
    out_path : str
        The path where the file will be downloaded.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Print verboosity
    printv(f"Downloading a file from '{url}' and saving to {out_path}.")
    # Create the progress bar object
    with DownloadProgressBar(unit="B",
                             unit_scale=True,
                             miniters=1,
                             desc=url.split(os.path.sep)[-1]) as t:
        urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)
    return None

def run(cmd: List[str], logFile: str = "", cwd : str = "") -> Union[int, Tuple[int, str]]:
    '''Run the given command (generic).

    Parameters
    ----------
    cmd : List[str]
        The command to be run.
    logFile : str, optional
        The file where the output will be saved.
        Default is "".
    cwd : str, optional
        The current working directory.
        Default is "".

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
    
    Raises
    ------
    None
    '''

    if not cmd:
        return errors.not_set(message = f"The variable cmd is not set or is an empty list!", level = "error")

    if type(cmd) != list:
        return errors.wrong_type(message = f"The argument cmd has to be a list! Found '{type(cmd)}' instead...", level = "error")

    # Print verboosity
    printv(f"Running the command '{' '.join(cmd)}'.")

    if logFile == "":
        printv(f"No log will be made")
        logFile = os.devnull
    else:
        printv(f"Logging into '{logFile}'")

    try:
        if cwd == "":
            with open(logFile, "w") as outfile:
                proc = subprocess.run(cmd, stdout = outfile, stderr = subprocess.PIPE)
        else:
            with open(logFile, "w") as outfile:
                proc = subprocess.run(cmd, stdout = outfile, cwd=cwd, stderr = subprocess.PIPE)
    except Exception as e:
        return errors.subprocess(message = f"Found a problem while executing the command '{' '.join(cmd)}': {e}", level="error")

    # If the command has not been executed successfully
    if proc.returncode != 0:
        return errors.subprocess(message = f"The command '{' '.join(cmd)}' has not been executed successfully!", level = "error"), proc.stderr.decode("utf-8")
    return errors.ok()

def get_rmsd(reference: str, molecule: str) -> Union[List, float]:
    '''Get the rmsd between a reference and a molecule file (it supports more than one molecule in this second file).

    Parameters
    ----------
    reference : str
        The reference file.
    molecule : str
        The molecule file.

    Returns
    -------
    List | float
        The rmsd between the reference and the molecule file.

    Raises
    ------
    None
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

    Raises
    ------
    None
    '''

    # Initialise hasCryst1 flag
    hasCryst1 = False
    
    # List of lines and dssp lines
    lines = []

    # Check if structurePath is a valid file
    if os.path.isfile(structurePath):
        # Open it (for cleaning)
        with open(structurePath, "r") as pdbFile:
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
            with open(structurePath, "w") as pdbFile:
                # Write the lines list to the file
                pdbFile.writelines(lines)
        
        return errors.ok()
    else:
        return errors.file_do_not_exist(message = f"The file '{structurePath}' does not exist!", level = "error")

### Special functions

@contextlib.contextmanager
def redirect_to_tqdm():
    '''Redirects the stdout to tqdm.write()

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Store builtin print
    old_print = print
    def new_print(*args, **kwargs):
        # If tqdm.write raises error, use builtin print
        try:
            tqdm.write(*args, **kwargs)
        except:
            old_print(*args, ** kwargs)
    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print # type: ignore
        yield
    finally:
        inspect.builtins.print = old_print # type: ignore
