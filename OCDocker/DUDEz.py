#!/usr/lib/python3

# Imports
###############################################################################
import os

import pandas as pd
import vaex.dataframe as vdf

from multiprocessing import Pool
from typing import Dict, Union

from OCDocker.Initialise import *

import OCDocker.Ligand as ocl
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

def read_logs_legacy(picklePath = "") -> Union[Dict[str, Dict[str, pd.DataFrame]], None]:
    '''Parse the database into multiple serializable objects.

    Parameters
    ----------
    picklePath : str, optional
        Path to the pickle file, by default "", which means that the pickle file will not be saved.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, pd.DataFrame]]] | None
        A dictionary with the following structure or None if the routine fails:
        {
            "protein": {
                "ligand": {
                    "algorithm": pd.DataFrame
                }
            }
        }

    Raises
    ------
    None
    '''

    return ocbdb.read_logs_legacy("dudez", picklePath = picklePath)

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

def generate_dock_result_csv(csv_path: str = "", log_dumps: Union[Dict[str, pd.DataFrame], None] = None, chunksize: int = 500) -> None:
    '''Uses the structure from read_logs to generate an output for all docking softwares.

    Parameters
    ----------
    csv_path : str, optional
        The path to the csv file to be generated. If empty, it will be generated in the current directory, by default "{parsed_archive}/DUDEz.csv".
    log_dumps : Dict[str, pd.DataFrame] | None, optional
        The structure from read_logs. If None, it will be generated, by default None.
    chunksize : int, optional
        The chunksize to be used in the pandas dataframe, by default 500.

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

    return ocbdb.generate_dock_result_csv("dudez", csv_path, log_dumps = log_dumps, chunksize = chunksize) # type: ignore

def merge_descriptors_in_dataframe_legacy(saveCsv = True) -> Union[pd.DataFrame, None]:
    '''Reads all the descriptors jsons and return a pd.DataFrame.

    Parameters
    ----------
    saveCsv : bool, optional
        If True, saves the dataframe as a csv file, by default True

    Returns
    -------
    pd.DataFrame | None
        A dataframe with all the descriptors or None if any error occur while reading the csv.

    Raises
    ------
    None
    '''

    # Get the dataframe with descriptors and docking scores
    dudezdf = ocbdb.merge_descriptors_in_dataframe_legacy("dudez", saveCsv = saveCsv)

    # If the save csv flag is set
    if saveCsv:
        # Parameterize the csvs paths
        csv_path_out = f"{parsed_archive}/dudez_complete.csv"
        if os.path.isfile(csv_path_out):
            octools.print_warning(f"The file {csv_path_out} already exists, it will be OVERWRITTEN!!")
        # Check if the dudezdf is not None
        if dudezdf is not None:
            # Write the data to a new csv file
            dudezdf.to_csv(csv_path_out, index = False)
        else:
            octools.print_warning(f"The dataframe is None, no csv will be generated")

    return dudezdf

def merge_descriptors_in_dataframe(saveCsv = True) -> Union[vdf.DataFrameLocal, None]:
    '''Reads all the descriptors jsons and return a vdf.DataFrameLocal.

    Parameters
    ----------
    saveCsv : bool, optional
        If True, saves the dataframe as a csv file, by default True

    Returns
    -------
    vdf.DataFrameLocal | None
        A dataframe with all the descriptors or None if any error occur while reading the csv.

    Raises
    ------
    None
    '''

    # Get the dataframe with descriptors and docking scores
    dudezdf = ocbdb.merge_descriptors_in_dataframe("dudez", saveCsv = saveCsv)
    
    # If the save csv flag is set
    if saveCsv:
        # Parameterize the csvs paths
        csv_path_out = f"{parsed_archive}/DUDEz_complete.csv"

        if os.path.isfile(csv_path_out):
            octools.print_warning(f"The file {csv_path_out} already exists, it will be OVERWRITTEN!!")

        # Check if the dudezdf is not None
        if dudezdf is not None:
            # Write the data to a new csv file
            dudezdf.to_csv(csv_path_out, index = False)
        else:
            octools.print_warning(f"The dataframe is None, no csv will be generated")

    return dudezdf
