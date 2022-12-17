#!/usr/lib/python3

# Imports
###############################################################################
import os

import pandas as pd
import vaex.dataframe as vdf

from typing import Dict, Union

from OCDocker.Initialise import *

import OCDocker.baseDB as ocbdb
import OCDocker.Toolbox as octools


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
Sets of classes and functions that are used to process the DUDE-Z dataset.

They are imported as:

import OCDocker.DUDEz as ocdudez
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##
def verify_integrity() -> None:
    '''Verifies the integrity of the DUDEz database.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raise
    -----
    None
    '''

    return None

def convert_debug_to_production(chosenAlgorithm: str = "ac", strict: bool = False, removeDebug: bool = False) -> None:
    '''Converts debug folders to production mode. It is required to choose an algorithm which will be used furtherly in the pipeline.

    Parameters
    ----------
    chosenAlgorithm : str, optional
        The chosen algorithm, by default "ac". The short code for the chosen algorithm. The options are:
        - ap: AffinityPropagation
        - ac: AgglomerativeClustering
        - bc: Birch
        - db: DBSCAN
        - km: KMeans
        - ms: MeanShift
        - mb: MiniBatchKMeans
        - na: No algorithm
        - op: OPTICS
        - sc: SpectralClustering
    strict : bool, optional
        If True, it will only convert the debug folders that have the chosen algorithm, by default False.
    removeDebug : bool, optional
        If True, it will remove the debug folder, by default False.

    Returns
    -------
    None

    Raise
    -----
    None
    '''

    ocbdb.convert_debug_to_production(dudez_archive, chosenAlgorithm = chosenAlgorithm, strict = strict, removeDebug = removeDebug)

    return None

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

    Returns
    -------
    None

    Raise
    -----
    None
    '''

    # Prepare the rest of the database
    ocbdb.prepare("dudez", overwrite = overwrite, spacing = spacing, sanitize = sanitize)
    # Verify its integrity
    #verify_integrity()
    
    return None

def run_p2rank(overwrite: bool = False) -> None:
    '''Runs P2Rank in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated, by default False.

    Returns
    -------
    None

    Raise
    -----
    None
    '''

    return ocbdb.run_p2rank("dudez", overwrite = overwrite)

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

    return ocbdb.run_dock("dudez", "gnina", overwrite = overwrite)

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

    return ocbdb.run_dock("dudez", "vina", overwrite = overwrite)

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

    return ocbdb.run_dock("dudez", "smina", overwrite = overwrite)

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

    return ocbdb.run_dock("dudez", "plants", overwrite = overwrite)

def read_logs(picklePath = "") -> Union[Dict[str, vdf.DataFrameLocal], None]:
    '''Parse the database into multiple serializable objects.

    Parameters
    ----------
    picklePath : str, optional
        Path to the pickle file, by default "", which means that the pickle file will not be saved.

    Returns
    -------
    Dict[str, Dict[str, vdf.DataFrameLocal]] | None
        A dictionary with the keys being the protein-ligand names and the values being the dataframes.

    Raises
    ------
    None
    '''

    return ocbdb.read_logs("dudez", picklePath = picklePath)

def generate_dock_result_csv(csv_path: str = "", log_dumps: Union[Dict[str, pd.DataFrame], None] = None) -> None:
    '''Uses the structure from read_logs to generate an output for all docking softwares.

    Parameters
    ----------
    csv_path : str, optional
        The path to the csv file to be generated. If empty, it will be generated in the current directory, by default "{parsed_archive}/DUDEz.csv".
    log_dumps : Dict[str, pd.DataFrame] | None, optional
        The structure from read_logs. If None, it will be generated, by default None.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Check if the csv_path is empty
    if csv_path == "":
        # Set the csv_path to the default
        csv_path = f"{parsed_archive}/dudez.csv"

    return ocbdb.generate_dock_result_csv("dudez", csv_path, log_dumps = log_dumps) # type: ignore

def merge_descriptors_in_dataframe(readMode:str = "hdf5", saveMode: str = "hdf5", picklenize: bool = False, returnDf: bool = False, skipMergePicklePath: str = "") -> Union[vdf.DataFrameLocal, None]:
    '''Reads all the descriptors jsons and return a vdf.DataFrameLocal.

    Parameters
    ----------
    readMode : str, optional
        The read mode for the descriptors. Can be "hdf5" or "csv", by default "hdf5".
    saveMode : str, optional
        The save mode for the descriptors. Can be "hdf5", "csv" or "", by default "hdf5". If empty, the dataframe will not be saved.
    picklenize : bool, optional
        If True, will save the dataframe as a pickle file in different steps during the execution. The default is False.
    returnDf : bool, optional
        If True, will return the dataframe. The default is False.
    skipMergePicklePath : str, optional
        The path to the pickle file with the dataframe. If empty, the dataframe will not be loaded from a pickle file. The default is "".

    Returns
    -------
    vdf.DataFrameLocal | None
        A dataframe with all the descriptors and affinity results or None if any error occur while reading the input file or if returnDf is set to false.

    Raises
    ------
    None
    '''

    # Get the dataframe with descriptors and docking scores
    return ocbdb.merge_descriptors_in_dataframe("dudez", readMode = readMode, saveMode = saveMode, picklenize = picklenize, returnDf = returnDf, skipMergePicklePath = skipMergePicklePath)
    