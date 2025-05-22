#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to manipulate and create files and
folders.

They are imported as:

import OCDocker.Toolbox.FilesFolders as ocff
'''

# Imports
###############################################################################
import json
import os
import h5py
import pickle
import tarfile

import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Union

import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Validation as ocvalidation

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
def empty_docking_digest(digestPath: str, overwrite: bool = False, digestFormat : str = "") -> Union[Dict[str, List[float]], int]:
    """Create an empty digest file.

    Parameters
    ----------
    digestPath : str
        Where to store the digest file.
    overwrite : bool, optional
        If True, overwrites the output files if they already exist. (default is False)
    digestFormat : str, optional
        The format of the digest file. The options are: [ '' (no output, default), json, hdf5 (not implemented) ]

    Returns
    -------
    Dict[str, List[float]] | int
        The empty digest object or the error code.
    """

    # Create the empty digest variable
    digest = {
        "gnina_pose": [np.NaN],
        "gnina_affinity": [np.NaN],
        "smina_pose": [np.NaN],
        "smina_affinity": [np.NaN], 
        "PLANTS_TOTAL_SCORE": [np.NaN],
        "PLANTS_SCORE_RB_PEN": [np.NaN],
        "PLANTS_SCORE_NORM_HEVATOMS": [np.NaN],
        "PLANTS_SCORE_NORM_CRT_HEVATOMS": [np.NaN], 
        "PLANTS_SCORE_NORM_WEIGHT": [np.NaN],
        "PLANTS_SCORE_NORM_CRT_WEIGHT": [np.NaN],
        "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": [np.NaN],
        "vina_pose": [np.NaN],
        "vina_affinity": [np.NaN]
        }

    # Check if the digest format is not empty
    if digestFormat != "":
        # Check if the file does not exists or if the overwrite flag is true
        if not os.path.isdir(digestPath) or overwrite:
            # Check if the digest extension is supported
            if ocvalidation.validate_digest_extension(digestPath, digestFormat):
                # Write the digest file
                if digestFormat == "json":
                    # Write the json file
                    try:
                        # Open the json file in write mode
                        with open(digestPath, 'w') as f:
                            # Dump the data
                            json.dump(digest, f)
                    except Exception as e:
                        return ocerror.Error.write_file(f"Could not write the digest file '{digestPath}'.", level = ocerror.ReportLevel.ERROR) # type: ignore
    return digest

def to_hdf5(filePath: str, data: Any) -> int:
    '''Save a data in a hdf5 file. If the data is a dict, use its keys as hdf5 key files. If is a list or tuple, use its indexes as keys. Otherwise, use the key 'data'.

    Parameters
    ----------
    filePath : str
        Path to the hdf5 file.
    data : Any
        Data to be saved.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    try:
        with h5py.File(filePath, 'w') as hf:
            # Check if the data is a dict
            if isinstance(data, dict):
                # Iterate over the keys
                for key in data:
                    # Save the data
                    hf.create_dataset(key, data = data[key])
            elif isinstance(data, (list, tuple)):
                # Iterate over the indexes
                for i in range(len(data)):
                    # Save the data
                    hf.create_dataset(str(i), data = data[i])
            else:
                # Save the data
                hf.create_dataset("data", data = data)
    except Exception as e:
        return ocerror.Error.write_file(f"Problems while saving the hdf5 file '{filePath}'. Error: {e}") # type: ignore
    return ocerror.Error.ok() # type: ignore
    
def from_hdf5(filePath: str) -> Union[None, Any]:
    '''Read a data from a hdf5 file.

    Parameters
    ----------
    filePath : str
        Path to the hdf5 file.

    Returns
    -------
    None | Any
        The data read from the file or None if there was an error.
    '''

    # Check if the file does not exists
    if not os.path.isfile(filePath):
        # Verbose its inexistence
        ocprint.printv(f"The hdf5 file '{filePath}' does not exists.")
        # Return an empty dict
        return {}

    data = None
    try:
        with h5py.File(filePath, 'r') as hf:
            # Read the hdf5 file with h5py
            data = {key: hf[key][:] for key in hf.keys()} # type: ignore
    except Exception as e:
        _ = ocerror.Error.read_file(f"Problems while reading the hdf5 file '{filePath}'. Error: {e}") # type: ignore
    return data

def to_pickle(filePath: str, data: Any) -> int:
    '''Pickle an object in a given path.

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
    '''

    try:
        with open(filePath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        return ocerror.Error.write_file(f"Problems while pickling the file '{filePath}'. Error: {e}") # type: ignore
    return ocerror.Error.ok() # type: ignore

def from_pickle(filePath: str) -> Union[None, Any]:
    '''Unpickle a pickle file into an object.

    Parameters
    ----------
    filePath : str
        Path to the pickle file.

    Returns
    -------
    None | Any
        The object if success or None otherwise.
    '''

    data = None
    try:
        with open(filePath, 'rb') as handle:
            data = pickle.load(handle)
    except Exception as e:
        _ = ocerror.Error.read_file(f"Problems while unpickling the file '{filePath}'. Error: {e}") # type: ignore
    return data

def safe_create_dir(dirname: Union[str, Path]) -> int:
    '''Create a dir if not exists.

    Parameters
    ----------
    dirname : str, Path
        The dir to be created. It can be a string or a Path object.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Try to create
    try:
        # Check if the dirname is a string
        if isinstance(dirname, str):
            # If file does not exists
            if not os.path.isdir(dirname):
                # Create it
                os.mkdir(dirname)
                # Print verbosity
                if ocerror.Error.output_level >= ocerror.ReportLevel.SUCCESS:
                    return ocerror.Error.ok(f"Successfully created the directory '{dirname}'") # type: ignore
                return ocerror.Error.ok() # type: ignore
            else:
                # It exists
                return ocerror.Error.dir_exists(message=f"The dir '{dirname}' already exists!", level = ocerror.ReportLevel.WARNING) # type: ignore
        # Check if the dirname is a Path object
        elif isinstance(dirname, Path):
            # If file does not exists
            if not dirname.is_dir():
                # Create it
                dirname.mkdir(parents=True, exist_ok=True)
                # Print verbosity
                if ocerror.Error.output_level >= ocerror.ReportLevel.SUCCESS:
                    return ocerror.Error.ok(f"Successfully created the directory '{dirname}'") # type: ignore
            else:
                # It is a file
                return ocerror.Error.file_exists(message=f"The dir '{dirname}' is a file!", level = ocerror.ReportLevel.WARNING) # type: ignore
    except Exception as e:
        # Some error has occurred
        return ocerror.Error.create_dir(message=f"Problem found while creating the dir '{dirname}': {e}", level = ocerror.ReportLevel.ERROR) # type: ignore
    # This should never appear since all the other paths ends in some kind of return
    return ocerror.Error.unknown(message=f"What are you expecting for? This message should NEVER appear!!!!!!! Btw problems while creating a dir safetly.", level = ocerror.ReportLevel.ERROR) # type: ignore

def safe_remove_dir(dirname: str) -> int:
    ''' Remove a dir if exists.

    Parameters
    ----------
    dirname : str
        The dir to be removed.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Try to create
    try:
        # If file does exists
        if os.path.isdir(dirname):
            # Remove it
            shutil.rmtree(dirname)
            # Print verbosity
            if ocerror.Error.output_level >= ocerror.ReportLevel.SUCCESS:
                return ocerror.Error.ok(f"Successfully removed the directory '{dirname}'") # type: ignore
            return ocerror.Error.ok() # type: ignore
        else:
            # It exists
            return ocerror.Error.dir_exists(message=f"The dir '{dirname}' does not exist!", level = ocerror.ReportLevel.WARNING) # type: ignore
    except Exception as e:
        # Some error has occurred
        return ocerror.Error.remove_dir(message=f"Problem found while removing the dir '{dirname}': {e}", level = ocerror.ReportLevel.ERROR) # type: ignore
    # This should never appear since all the other paths ends in some kind of return
    return ocerror.Error.unknown(message=f"What are you expecting for? This message should NEVER appear!!!!!!! Btw problems while creating a dir safetly.", level = ocerror.ReportLevel.ERROR) # type: ignore

def safe_remove_file(filePath: str) -> int:
    '''Remove a file if exists.

    Parameters
    ----------
    filePath : str
        The file to be removed.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Check if the file exists
    if os.path.isfile(filePath):
        # Try to remove
        try:
            # Remove it
            os.remove(filePath)
            # Print verbosity
            if ocerror.Error.output_level >= ocerror.ReportLevel.SUCCESS:
                return ocerror.Error.ok(f"Successfully removed the file '{filePath}'") # type: ignore
            return ocerror.Error.ok() # type: ignore
        except Exception as e:
            # Some error has occurred
            return ocerror.Error.unknown(message=f"Problem found while removing the file '{filePath}': {e}", level = ocerror.ReportLevel.ERROR) # type: ignore
    else:
        # Return file not found error
        return ocerror.Error.file_not_exist(message=f"The file '{filePath}' does not exists!", level = ocerror.ReportLevel.WARNING) # type: ignore
    # This should never appear since all the other paths ends in some kind of return
    return ocerror.Error.unknown(message=f"What are you expecting for? This message should NEVER appear!!!!!!! Btw problems while removing a file safetly.", level = ocerror.ReportLevel.ERROR) # type: ignore

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
    '''

    # Print verboosity
    ocprint.printv(f"Untarring file '{fname}' to the output '{out_path}'")
    # Check if the file has the right extensions
    if fname.endswith("tar.gz") or fname.endswith(".tgz") or fname.endswith(".gz"):
        try:
            ocprint.printv("Preparing to untar the file...")
            # open your tar.gz file
            with tarfile.open(name=fname) as tar:
                # Redirect output to tqdm.write
                with ocbasetools.redirect_to_tqdm():
                    # Go over each member
                    for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                        # Extract member
                        tar.extract(member=member, path=out_path)
            # Report success on untarring the file
            _ = ocerror.Error.ok(f"The file {fname} has been {clrs['g']}successfully{clrs['n']} untarred to the dir {out_path}!", level = ocerror.ReportLevel.SUCCESS) # type: ignore
            # If delete flag is set, delete file
            if delete:
                #shutil.rmtree(fname) # remove the files
                os.remove(fname) # remove the files
                return ocerror.Error.ok(f"The file {fname} has been {clrs['y']}deleted!{clrs['n']}") # Report success on deleting the file # type: ignore
            return ocerror.Error.ok() # type: ignore
        except Exception as e:
            return ocerror.Error.untar_file(message=f"{clrs['r']}Failed{clrs['n']} to untar the file {fname}.\n\n{clrs['r']}Error{clrs['n']}: {e}", level = ocerror.ReportLevel.ERROR) # type: ignore
    else:
        # No supported extension has been provided
        return ocerror.Error.unsupported_extension(message=f"The file {fname} is not a tar.gz file. {clrs['y']}Aborting execution{clrs['n']}", level = ocerror.ReportLevel.ERROR) # type: ignore
