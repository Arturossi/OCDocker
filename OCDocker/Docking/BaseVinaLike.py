#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of generic functions to read and write Vina-like docking logs, generate docking digests, and retrieve docked poses.

They are imported as:

import OCDocker.Docking.BaseVinaLike as ocbasevina
'''

# Imports
###############################################################################
import errno
import json
import os
from glob import glob
from typing import Dict, List, Callable

import numpy as np

from OCDocker.Config import get_config
import OCDocker.Error as ocerror
import OCDocker.Toolbox.IO as ocio
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Validation as ocvalidation
import OCDocker.Toolbox.FilesFolders as ocff

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

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _read_log_generic(path: str, scoring_key: str, engine: str, error_log: str, onlyBest: bool = False) -> Dict[int, Dict[int, float]]:
    '''Read the vinalike log path, returning the data from complexes.

    Parameters
    ----------
    path : str
        The path to the vinalike log file.
    scoring_key : str
        The key to be used for the scoring value in the returned dictionary.
    engine : str
        The name of the engine used to generate the log file (e.g., "vina", "smina").
    error_log : str
        The name of the error log file to be used for logging errors.
    onlyBest : bool, optional
        If True, only the best pose will be returned. By default False.

    Returns
    -------
    Dict[int, Dict[int, float]]
        A dictionary with the data from the vina log file.
    '''

    # Create a dictionary to store the info
    data = {}

    # Check if file exists
    if os.path.isfile(path):
        # Catch any error that might occur
        try:
            # Check if file is empty
            if os.stat(path).st_size == 0:
                # Print the error
                _ = ocerror.Error.empty_file( # type: ignore
                    f"The {engine} log file '{path}' is empty.",
                    ocerror.ReportLevel.ERROR,
                )

                # Return the dictionary with invalid default data
                return data
        
            # Try except to avoid broken pipe ocerror.Error
            try:
                # Lazy read the file reversely 
                for line in ocio.lazyread_reverse_order_mmap(path):
                    # While the line does not start with "-----+"
                    if line.startswith("-----+"):
                        break

                    # Split the last line
                    splitLine = line.split()

                    # Check if there are 4 elements in the splitLine
                    if len(splitLine) == 4:
                        # Assign the data in the dictionary with the pose as key and the affinity as value
                        data[int(splitLine[0])] = {scoring_key: splitLine[1]}
                
                # If onlyBest is True
                if onlyBest:
                    # Return only the best pose (-1 since the data is reversed)
                    return {list(data.keys())[-1]: list(data.values())[-1]}
            
                # Otherwise return the data
                return data
            except IOError as e:
                if e.errno == errno.EPIPE:
                    ocprint.print_error(f"Problems while reading file '{path}'. Error: {e}")
                    config = get_config()
                    ocprint.print_error_log(
                        f"Problems while reading file '{path}'. Error: {e}",
                        f"{config.logdir}/{error_log}",
                    )

            # Return the df reversing the order and reseting the index
            return data
        except Exception as e:
            _ = ocerror.Error.read_docking_log_error( # type: ignore
                f"Problems while reading the {engine} log file '{path}'. Error: {e}",
                ocerror.ReportLevel.ERROR,
            )
            return data
        
    # Throw an error
    _ = ocerror.Error.file_not_exist( # type: ignore
        f"The file '{path}' does not exists. Please ensure its existance before calling this function.",
    )

    # Return a dict with a NaN value
    return data


def _read_rescoring_log_generic(path: str, start_string: str, engine: str, error_log: str) -> float:
    '''Read the vina rescoring log path, returning the computed affinity.

    Parameters
    ----------
    path : str
        The path to the vina rescoring log file.
    start_string : str
        The string that indicates the start of the affinity line in the log file.
    engine : str
        The name of the engine used to generate the log file (e.g., "vina", "smina").
    error_log : str
        The name of the error log file to be used for logging errors.

    Returns
    -------
    float
        The affinity of the ligand.
    '''

    # Check if file exists
    if os.path.isfile(path):
        # Catch any error that might occur
        try:
            # Check if file is empty
            if os.stat(path).st_size == 0:
                # Print the error
                _ = ocerror.Error.empty_file( # type: ignore
                    f"The {engine} rescoring log file '{path}' is empty.",
                    ocerror.ReportLevel.ERROR,
                )
                # Return NaN
                return np.nan

            # Try except to avoid broken pipe ocerror.Error
            try:
                # Read the file reversely
                for line in ocio.lazyread_reverse_order_mmap(path):
                    # If the line starts with "Estimated Free Energy of Binding" means that its the correct line
                    if line.startswith(start_string):
                        # Parse the value from the line
                        value = (
                            line.split(start_string)[1]
                            .split("(kcal/mol)")[0]
                            .strip()
                            .split(" ")[-1]
                        )
                        # Convert the value to float then return it
                        return float(value)
            except IOError as e:
                if e.errno == errno.EPIPE:
                    ocprint.print_error(f"Problems while reading file '{path}'. Error: {e}")
                    config = get_config()
                    ocprint.print_error_log(
                        f"Problems while reading file '{path}'. Error: {e}",
                        f"{config.logdir}/{error_log}",
                    )
            return np.nan
        except Exception as e:
            _ = ocerror.Error.read_docking_log_error( # type: ignore
                f"Problems while reading the {engine} log file '{path}'. Error: {e}",
                ocerror.ReportLevel.ERROR,
            )
            return np.nan
    
    # Throw an error
    _ = ocerror.Error.file_not_exist( # type: ignore
        f"The file '{path}' does not exists. Please ensure its existance before calling this function.",
    )

    # Return NaN if the file does not exist
    return np.nan


def _generate_digest_generic(
    digestPath: str,
    logPath: str,
    read_log_func: Callable[[str], Dict[int, Dict[int, float]]],
    overwrite: bool = False,
    digestFormat: str = "json",
) -> int:
    '''Generate the docking digest.
    
    Parameters
    ----------
    digestPath : str
        Where to store the digest file.
    logPath : str
        The log path.
    overwrite : bool, optional
        If True, overwrites the output files if they already exist. (default is False)
    digestFormat : str, optional
        The format of the digest file. The options are: [ json (default), hdf5 (not implemented) ]

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Check if the file does not exists or if the overwrite flag is true
    if not os.path.isdir(digestPath) or overwrite:
        # Check if the digest extension is supported
        if ocvalidation.validate_digest_extension(digestPath, digestFormat):

            # Create the digest variable
            digest = None

            # Check if the file exists
            if os.path.isfile(digestPath):
                # If is a json file
                if digestFormat == "json":
                    # Read the json file
                    try:
                        # Open the json file in read mode
                        with open(digestPath, "r") as f:
                            # Load the data
                            digest = json.load(f)

                            # Check if the digest variable is fine
                            if not isinstance(digest, dict):
                                return ocerror.Error.wrong_type( # type: ignore
                                    f"The digest file '{digestPath}' is not valid.",
                                    ocerror.ReportLevel.ERROR,
                                )
                    except (OSError, IOError, FileNotFoundError, json.JSONDecodeError):
                        return ocerror.Error.file_not_exist( # type: ignore
                            f"Could not read the digest file '{digestPath}'.",
                            ocerror.ReportLevel.ERROR,
                        )
            else:
                # Since it does not exists, create it
                digest = ocff.empty_docking_digest(digestPath, overwrite)
            
            # Read the docking object log to generate the docking digest
            dockingDigest = read_log_func(logPath)

            # Check if the digest variable is fine
            if not isinstance(digest, dict):
                return ocerror.Error.wrong_type( # type: ignore
                    f"The docking digest file '{digestPath}' is not valid.",
                    ocerror.ReportLevel.ERROR,
                )
        
            # Merge the digest and the docking digest
            digest = {**digest, **dockingDigest} # type: ignore

            # If the format is json, write the digest file
            if digestFormat == "json":
                # Write the json file
                try:
                    # Open the json file in write mode
                    with open(digestPath, "w") as f:
                        # Dump the data
                        json.dump(digest, f)
                except (OSError, IOError, PermissionError):
                    return ocerror.Error.write_file( # type: ignore
                        f"Could not write the digest file '{digestPath}'.",
                        ocerror.ReportLevel.ERROR,
                    )
            return ocerror.Error.ok()  # type: ignore
        
        return ocerror.Error.unsupported_extension( # type: ignore
            f"The provided extension '{digestFormat}' is not supported.",
            ocerror.ReportLevel.ERROR,
        )
    
    return ocerror.Error.file_exists( # type: ignore
        f"The file '{digestPath}' already exists. If you want to overwrite it yse the overwrite flag.",
        level=ocerror.ReportLevel.WARNING,


    )


def _get_docked_poses_generic(posesPath: str, error_method: Callable) -> List[str]:
    '''Get the docked poses from the poses path.

    Parameters
    ----------
    posesPath : str
        The path to the poses folder.
    error_method : Callable
        The error method to be used in case the poses path does not exist.

    Returns
    -------
    List[str]
        A list with the paths to the docked poses.
    '''

    # Check if the posesPath exists
    if os.path.isdir(posesPath):
        return [d for d in glob(f"{posesPath}/*_split_*.pdbqt") if os.path.isfile(d)]
    
    # Print an error message
    error_method(
        message=f"The poses path '{posesPath}' does not exist.",
        level=ocerror.ReportLevel.ERROR,
    )

    return []





## Public ##

# ---------------------------------------------------------------------------
# Wrappers for Vina
# ---------------------------------------------------------------------------


def read_vina_log(path: str, onlyBest: bool = False) -> Dict[int, Dict[int, float]]:
    '''Wrapper for reading the Vina log file.
    
    Parameters
    ----------
    path : str
        The path to the Vina log file.
    onlyBest : bool, optional
        If True, only the best pose will be returned. By default False.
    
    Returns
    -------
    Dict[int, Dict[int, float]]
        A dictionary with the data from the Vina log file.
    '''

    # Call the generic read log function with the Vina scoring key
    config = get_config()
    return _read_log_generic(path, config.vina.scoring, "vina", "vina_read_log_ERROR.log", onlyBest)


def read_vina_rescoring_log(path: str) -> float:
    '''Wrapper for reading the Vina rescoring log file.

    Parameters
    ----------
    path : str
        The path to the Vina rescoring log file.
    
    Returns
    -------
    float
        The estimated free energy of binding from the Vina rescoring log file.
    '''

    return _read_rescoring_log_generic(path, "Estimated Free Energy of Binding", "vina", "vina_read_log_ERROR.log")


def generate_vina_digest(digestPath: str, logPath: str, overwrite: bool = False, digestFormat: str = "json") -> int:
    '''Wrapper for generating the Vina digest.
    
    Parameters
    ----------
    digestPath : str
        Where to store the digest file.
    logPath : str
        The path to the Vina log file.
    overwrite : bool, optional
        If True, overwrites the output files if they already exist. (default is False)
    digestFormat : str, optional
        The format of the digest file. The options are: [ json (default), hdf5 (not implemented) ]

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Call the generic generate digest function with the Vina read log function
    return _generate_digest_generic(digestPath, logPath, read_vina_log, overwrite, digestFormat)


def get_vina_docked_poses(posesPath: str) -> List[str]:
    '''Get the paths for the docked poses from Vina output directory.
    
    Parameters
    ----------
    posesPath : str
        The path to the directory containing the docked poses.
    
    Returns
    -------
    List[str]
        A list with the paths for the docked poses. Returns an empty list if the directory does not exist.
    '''
    
    return _get_docked_poses_generic(posesPath, ocerror.Error.dir_not_exist) # type: ignore





# ---------------------------------------------------------------------------
# Wrappers for Smina
# ---------------------------------------------------------------------------


def read_smina_log(path: str, onlyBest: bool = False) -> Dict[int, Dict[int, float]]:
    '''Wrapper for reading the Smina log file.

    Parameters
    ----------
    path : str
        The path to the Smina log file.
    onlyBest : bool, optional
        If True, only the best pose will be returned. By default False.
    
    Returns
    -------
    Dict[int, Dict[int, float]]
        A dictionary with the data from the Smina log file.
    '''

    # Call the generic read log function with the Smina scoring key
    config = get_config()
    return _read_log_generic(path, config.smina.scoring, "smina", "smina_read_log_ERROR.log", onlyBest)


def read_smina_rescoring_log(path: str) -> float:
    '''Wrapper for reading the Smina rescoring log file.

    Parameters
    ----------
    path : str
        The path to the Smina rescoring log file.
    
    Returns
    -------
    float
        The affinity of the ligand from the Smina rescoring log file.
    '''

    return _read_rescoring_log_generic(path, "Affinity", "smina", "smina_read_log_ERROR.log")


def generate_smina_digest(digestPath: str, logPath: str, overwrite: bool = False, digestFormat: str = "json") -> int:
    '''Wrapper for generating the Smina digest.

    Parameters
    ----------
    digestPath : str
        Where to store the digest file.
    logPath : str
        The path to the Smina log file.
    overwrite : bool, optional
        If True, overwrites the output files if they already exist. (default is False)
    digestFormat : str, optional
        The format of the digest file. The options are: [ json (default), hdf5 (not implemented) ]
    
    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Call the generic generate digest function with the Smina read log function
    return _generate_digest_generic(digestPath, logPath, read_smina_log, overwrite, digestFormat)


def get_smina_docked_poses(posesPath: str) -> List[str]:
    '''Wrapper for getting the Smina docked poses.

    Parameters
    ----------
    posesPath : str
        The path to the Smina docked poses folder.

    Returns
    -------
    List[str]
        A list with the paths to the Smina docked poses.
    '''

    # Use the Error class to get the error method for directory not existing
    err = getattr(ocerror.Error, "dir_does_not_exist", ocerror.Error.dir_not_exist) # type: ignore

    # Call the generic get docked poses function
    return _get_docked_poses_generic(posesPath, err)
