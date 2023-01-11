#!/usr/lib/python3

# Imports
###############################################################################
import gc
import os
import time
import vaex

import numpy as np
import vaex.dataframe as vdf

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.Toolbox as octools
import OCDocker.Docking.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
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
This module is responsible for digest processing.

It is imported as:

import OCDocker.Processing.Postprocessing.Readlogs as ocreadlogs
'''

# Classes
###############################################################################


# Functions
###############################################################################
def __core_read_log(processDirData: Tuple[str, str]) -> Dict[str, vdf.DataFrameLocal]:
    '''Reads Vina, Smina and PLANTS logs and then return a dict of dataframes.

    Parameters
    ----------
    processDirData : Tuple[str, str]
        A tuple with the directory and the ligand name.

    Returns
    -------
     Dict[str, vdf.DataFrameLocal]
        A dictionary containing the dataframes of the logs.
        A dict of dicts of dataframes. Each element of the first dict is a complex protein-ligand, and each element of the second dict is a docking algorithm results.

    Raises
    ------
    None
    '''

    # Unpack the tuple
    processDir, tp = processDirData

    # Get protein and ligand names
    processSplited = processDir.split(os.path.sep)
    ptn = processSplited[-4]
    lgd = processSplited[-1]

    # Create docking dicts
    vinaDict = { "vina_pose": [], "vina_affinity": [] }
    gninaDict = { "gnina_pose": [], "gnina_affinity": [] }
    sminaDict = { "smina_pose": [], "smina_affinity": [] }
    plantsDict = { "PLANTS_TOTAL_SCORE": [], "PLANTS_SCORE_RB_PEN": [], "PLANTS_SCORE_NORM_HEVATOMS": [], "PLANTS_SCORE_NORM_CRT_HEVATOMS": [], "PLANTS_SCORE_NORM_WEIGHT": [], "PLANTS_SCORE_NORM_CRT_WEIGHT": [], "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": [] }

    # Get run number
    runNumber = 0 # TODO: Add support to multiple runs

    # Dict to hold the protein data
    proteinData = { f"{ptn}-{lgd}": None }

    # VINA
    vinaDir = f"{processDir}/vinaFiles"
    # Parameterize the log path
    logPath = f"{vinaDir}/vina_{runNumber}.log"

    # Check if exists
    if os.path.isfile(logPath):
        # Read the log into dict
        gendict = ocvina.read_log(logPath)

        # For each key, value in vinaDict
        for key, value in gendict.items():
            # Append the value to the vina dict
            vinaDict[key].append(value[0])
    else:
        _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its vina output.")
        # Set the elements in vinaDict as np.NaN
        vinaDict = { "vina_pose": [np.NaN], "vina_affinity": [np.NaN] }

    # SMINA
    sminaDir = f"{processDir}/sminaFiles"
    # Parameterize the log path
    logPath = f"{sminaDir}/smina.log"

    # Check if smina log exists
    if os.path.isfile(logPath): # TODO: Add support to multiple runs
        # Read the log into dict
        gendict = ocsmina.read_log(logPath)

        # For each key, value in sminaDict
        for key, value in gendict.items():
            # Append the value to the smina dict
            sminaDict[key].append(value[0])
    else:
        _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its SMINA output.")
        # Set the elements in sminaDict as np.NaN
        sminaDict = { "smina_pose": [np.NaN], "smina_affinity": [np.NaN] }

    # GNINA
    gninaDir = f"{processDir}/gninaFiles"
    # Parameterize the log path
    logPath = f"{gninaDir}/gnina_{runNumber}.log"

    # Check if exists
    if os.path.isfile(logPath):
        # Read the log into dict
        gendict = ocgnina.read_log(logPath)

        # For each key, value in gninaDict
        for key, value in gendict.items():
            # Append the value to the gnina dict
            gninaDict[key].append(value[0])
    else:
        _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its gnina output.")
        # Set the elements in gninaDict as np.NaN
        gninaDict = { "gnina_pose": [np.NaN], "gnina_affinity": [np.NaN] }

    # PLANTS
    plantsDir = f"{processDir}/plantsFiles"
    # Parameterize the log path
    logPath = f"{plantsDir}/run/bestranking.csv"

    # Check if exists
    if os.path.isfile(logPath):
        # Read the log into dict
        gendict = ocplants.read_log(logPath)

        # For each key, value in plantsDict
        for key, value in gendict.items():
            # Append the value to the plants dict
            plantsDict[key].append(value[0])
    else:
        _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its PLANTS output.")
        # Set the elements in plantsDict as np.NaN
        plantsDict = { "PLANTS_TOTAL_SCORE": [np.NaN], "PLANTS_SCORE_RB_PEN": [np.NaN], "PLANTS_SCORE_NORM_HEVATOMS": [np.NaN], "PLANTS_SCORE_NORM_CRT_HEVATOMS": [np.NaN], "PLANTS_SCORE_NORM_WEIGHT": [np.NaN], "PLANTS_SCORE_NORM_CRT_WEIGHT": [np.NaN], "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": [np.NaN] }

    # Create the maxLenList
    maxLenList = []

    # Add each score to the list its len greater than 0 (never should be negative)
    if len(vinaDict["vina_pose"]) > 0:
        maxLenList.append(len(vinaDict["vina_pose"]))
    if len(sminaDict["smina_pose"]) > 0:
        maxLenList.append(len(sminaDict["smina_pose"]))
    if len(gninaDict["gnina_pose"]) > 0:
        maxLenList.append(len(gninaDict["gnina_pose"]))
    if len(plantsDict["PLANTS_TOTAL_SCORE"]) > 0:
        maxLenList.append(len(plantsDict["PLANTS_TOTAL_SCORE"]))
    
    # Check if the list is empty
    if len(maxLenList) == 0:
        # Set the maxLen to 1
        maxLen = 1
    else:
        # Get the number of elements of the dict with the largest number of elements
        maxLen = max(maxLenList)

    # Add the concatenated the dicts. The single elements are repeated to match the largest dict to the proteinData dict using ptn as the key
    proteinData[f"{ptn}-{lgd}"] = vaex.from_dict(
        {
            **{
                "Protein": [ptn for _ in range(maxLen)],
                "Ligand": [lgd for _ in range(maxLen)],
                "type": [tp for _ in range(maxLen)]
            },
            **vinaDict,
            **sminaDict,
            **gninaDict,
            **plantsDict
        }
    )

    # Clean the memory
    del vinaDict, sminaDict, plantsDict#, gninaDict

    # Return the proteinData dict
    return proteinData # type: ignore

def __thread_read_log_parallel(arguments: Tuple[Tuple[str, str]]) -> Dict[str, vdf.DataFrameLocal]:
    '''Thread aid function to call __core_read_log.

    Parameters
    ----------
    arguments : Tuple[Tuple[str, str]]
        A tuple with the directory and the ligand type (ligand, decoy, candidate).

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary containing the dataframes of the logs.

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_read_log(arguments[0])

def read_log_parallel(paths: List[Tuple[str, str]], desc: str) -> Dict[str, vdf.DataFrameLocal]:
    '''Read the logs of the ligands in parallel.

    Parameters
    ----------
    paths : List[Tuple[str, str]]
        A list of tuples with the directory and the ligand type (ligand, decoy, candidate).
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary containing the dataframes of the logs.

    Raises
    ------
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []

    # For each file in the glob
    for path in paths:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((path, None))

    # If logfile exists, backup it for vina, smina and plants (for error and warnings)
    if os.path.isfile(f"{logdir}/read_log_ERROR.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            octools.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/read_log_ERROR.log", f"{logdir}/read_log_past/read_log_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Dict to store the read data
    data = {}

    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for innerData in tqdm(p.imap_unordered(__thread_read_log_parallel, arguments), total = len(arguments), desc = desc):
                # Get the key from innerData
                key = list(innerData.keys())[0]
                # Set the value of the key in data to the value of the key in innerData
                data[key] = innerData[key]
                # Clean the memory
                del innerData
                gc.collect()
                
    except IOError as e:
        octools.print_error_log(f"Problem while reading logs in parallel. Exception: {e}", f"{logdir}/read_log_ERROR_report.log")
        octools.print_error(f"Problem while reading logs in parallel. Exception: {e}")

    return data

def read_log_no_parallel(paths: List[Tuple[str, str]], desc: str) -> Dict[str, vdf.DataFrameLocal]:
    '''Read the logs of the docking results for the ligands in serial.

    Parameters
    ----------
    ptnDirs : List[Tuple[str, str]]
        A list of tuples with the directory and the ligand type (ligand, decoy, candidate).
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary with the protein name as the key and a dictionary with the vina, smina and plants dataframes as the value.
    
    Raises
    ------
    None
    '''

    # Dict to store the read data
    data = {}

    # If logfile exists, backup it for vina, smina and plants (for error and warnings)
    if os.path.isfile(f"{logdir}/vina_read_log_ERROR.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            octools.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/vina_read_log_ERROR.log", f"{logdir}/read_log_past/vina_read_log_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")
    if os.path.isfile(f"{logdir}/smina_read_log_ERROR.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            octools.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/smina_read_log_ERROR.log", f"{logdir}/read_log_past/smina_read_log_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")
    if os.path.isfile(f"{logdir}/plants_read_log_ERROR.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            octools.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/plants_read_log_ERROR.log", f"{logdir}/read_log_past/plants_read_log_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for path, tp in tqdm(iterable = paths, total = len(paths), desc = desc):
            # Call the core read log function (shared between parallel and not parallel) and store the data into the data dict
            data.update(__core_read_log((path, tp)))
            # Clear the memory
            gc.collect()

    return data

def read_log_single(path: Tuple[str, str]) -> Dict[str, vdf.DataFrameLocal]:
    '''Warper to prepare the jobs, recieves a directory, and pass it to the __core_prepare function.

    TODO: Add the support to custom databases.

    Parameters
    ----------
    path : Tuple[str, str]
        A tuple with the directory and the ligand type (ligand, decoy, candidate).

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary with the protein name as the key and a dictionary with the vina, smina and plants dataframes as the value.

    Raises
    ------
    None
    '''

    # Read the log
    return __core_read_log(path)

def read_logs(paths: Union[List[Tuple[str, str]], List[Tuple[str, str]]], archive: str) -> None:
    '''Read the logs of the docking results for the ligands.

    Parameters
    ----------
    paths : Tuple[str] | str
        The list of directories or the directory to be processed.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    '''

    # If the path is a list
    if isinstance(paths, list):
        label = f"Processing {archive}"
        # Check if multiprocessing is enabled
        if args.multiprocess:
            # Prepare the pdbbind
            read_log_parallel(paths, label)
        else:
            # Prepare the database
            read_log_no_parallel(paths, label)
    else:
        read_log_single(paths)
