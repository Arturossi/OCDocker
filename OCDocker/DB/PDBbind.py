#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to process the PDBbind dataset.

They are imported as:

import OCDocker.DB.PDBbind as ocpdbbind
'''

# Imports
###############################################################################
import os

from glob import glob
from typing import Dict, List, Union

from OCDocker.Initialise import *

import OCDocker.DB.baseDB as ocbdb
import OCDocker.Processing.Preprocessing.p2rank as ocp2rank

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
def verify_integrity() -> None:
    '''Verifies the integrity of the PDBbind database

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    #ocbdb.verify_integrity(pdbbind_archive)
    return None

def convert_debug_to_production(chosenAlgorithm: str = "ac", strict: bool = False, removeDebug: bool = False) -> None:
    '''Converts debug folders to production mode. It is required to choose an algorithm which will be used furtherly in the pipeline.

    Parameters
    ----------
    chosenAlgorithm : str
        The algorithm that will be used in the pipeline. It can be either "ac" or "p2rank".
            AffinityPropagation: ap
            AgglomerativeClustering: ac
            Birch: bi
            DBSCAN: db
            KMeans:  km
            MeanShift: ms
            MiniBatchKMeans: mb
            NoCluster: na
            OPTICS: op
            SpectralClustering: sc
            Ward: wa
    strict : bool
        If True, it will only convert the folders that have the chosen algorithm. If False, it will convert all folders.
    removeDebug : bool
        If True, it will remove the debug folder after the conversion.

    Returns
    -------
    None
    '''

    ocp2rank.convert_debug_to_production(pdbbind_archive, chosenAlgorithm = chosenAlgorithm, strict = strict, removeDebug = removeDebug)
    return None

def read_index() -> Union[Dict[str, Dict[str, Union[str, float]]], None]:
    '''Read the index file from pdbbind database and returns a list of dictionaries with the data.

    Returns
    -------
    Dict[str, Dict[str, str | float]] | None
        A dict of dictionaries where each dictionary represents the data for a single protein.
        If the file does not exist, it will return None.
    '''

    indexFile = glob(pdbbind_archive + '/index/INDEX_refined_data.*')[0]

    # If the file exists
    if os.path.isfile(indexFile):
        # List to hold the protein data
        proteinDataOrder = f"{pdbbind_KdKi_order}M"
        proteinDataDict = {}  # Dict of dictionaries to hold data for each protein

        # Open the file in read mode
        with open(indexFile, 'r') as f:
            # Loop through the file line by line
            for line in f:
                # If the line starts with a #, skip it (no useful info)
                if line.startswith("#"):
                    continue
                
                # Split the line by spaces
                splitedLine = line.split()

                # Extract Kd/Ki and type (Kd/Ki)
                tp, kdki = splitedLine[4].split("=")
                
                # Normalize the Kd/Ki values to a consistent unit (mol/L)
                if "mM" in kdki:
                    kdki = float(kdki.replace("mM", "")) * order[pdbbind_KdKi_order]["m"]
                elif "uM" in kdki:
                    kdki = float(kdki.replace("uM", "")) * order[pdbbind_KdKi_order]["u"]
                elif "nM" in kdki:
                    kdki = float(kdki.replace("nM", "")) * order[pdbbind_KdKi_order]["n"]
                elif "pM" in kdki:
                    kdki = float(kdki.replace("pM", "")) * order[pdbbind_KdKi_order]["p"]
                elif "fM" in kdki:
                    kdki = float(kdki.replace("fM", "")) * order[pdbbind_KdKi_order]["f"]
                elif "cM" in kdki:
                    kdki = float(kdki.replace("cM", "")) * order[pdbbind_KdKi_order]["c"]
                else:  # Assume M if not otherwise specified
                    kdki = float(kdki.replace("M", "")) * order[pdbbind_KdKi_order]["M"]

                # Create a dictionary for this protein and its data
                protein_entry = {
                    "Protein": splitedLine[0],
                    "resolution": splitedLine[1],
                    "release_year": splitedLine[2],
                    "-logKd/Ki": splitedLine[3],
                    "Ki/Kd": tp,
                    "Ki/Kd_value": kdki,
                    "Ki/Kd_order": proteinDataOrder,
                    "dG": occ.convert_Ki_Kd_to_dG(kdki)
                }

                # Add the dictionary to the dict setting the protein name as the key
                proteinDataDict[splitedLine[0]] = protein_entry

        # Return the list of dictionaries
        return proteinDataDict
    else:
        # File does not exist, raise an error and return None
        _ = ocerror.Error.file_not_exist(f"The file {indexFile} does not exist. Please check if the PDBbind database is correctly installed.", level=ocerror.ReportLevel.WARNING)  # type: ignore
        return None

def run_p2rank(overwrite: bool = False) -> None:
    '''Runs P2Rank in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the P2Rank if the results already exist, by default False.

    Returns
    -------
    None
    '''

    return ocbdb.run_p2rank("pdbbind", overwrite = overwrite)

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

    return ocbdb.run_docking("pdbbind", "gnina", overwrite = overwrite)

def run_vina(overwrite: bool = False) -> int:
    '''Runs vina in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the vina if the results already exist, by default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    return ocbdb.run_docking("pdbbind", "vina", overwrite = overwrite)

def run_smina(overwrite: bool = False) -> int:
    '''Runs smina in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the smina if the results already exist, by default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    return ocbdb.run_docking("pdbbind", "smina", overwrite = overwrite)

def run_plants(overwrite: bool = False) -> int:
    '''Runs PLANTS in the whole database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the PLANTS if the results already exist, by default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    return ocbdb.run_docking("pdbbind", "plants", overwrite = overwrite)

def prepare(overwrite: bool = False) -> None:
    '''Prepares the PDBbind database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the preparation if the results already exist, by default False.

    Returns
    -------
    None
    '''

    return ocbdb.prepare("pdbbind", overwrite = overwrite)

"""
TODO: remove this function
def read_logs(saveChunk: int = 100, overwrite: bool = False) -> Union[dict, None]:
    '''Parse the database into multiple serializable objects.

    Parameters
    ----------
    saveChunk : int, optional
        The number of lines to save in each chunk, by default 100.
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the preparation if the results already exist, by default False.

    Returns
    -------
    dict | None
        The parsed data.
    '''

    return ocbdb.read_logs("pdbbind", saveChunk = saveChunk, overwrite = overwrite)
"""

def generate_dock_result_csv(csv_path: str = "", log_dumps: Union[dict, None] = None) -> None:
    '''Uses the structure from read_logs to generate an output for all docking softwares.

    Parameters
    ----------
    csv_path : str, optional
        The path to the output csv file. If not specified, it will use the default path, by default f"{parsed_archive}/PDBbind.csv".
    log_dumps : dict, optional
        The parsed data.

    Returns
    -------
    None
    '''

    # Check if csv_path is empty
    if csv_path == "":
        # It is empty, use the default path
        csv_path = f"{parsed_archive}/pdbbind.csv"

    return ocbdb.generate_dock_result_csv("pdbbind", csv_path, log_dumps = log_dumps)

"""
TODO: remove this function
def merge_descriptors_in_dataframe(readMode:str = "hdf5", saveMode: str = "hdf5", picklenize: bool = False, returnDf: bool = False, skipMergePicklePath: str = "", verboseOperations: bool = False) -> Union[dict, None]:
    '''Reads all the descriptors jsons and return a dict.

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
    dict | None
        A dataframe with all the descriptors and affinity results or None if any error occur while reading the input file or if returnDf is set to false.
    '''
    
    # Get the dataframe with descriptors and docking scores
    return ocbdb.merge_descriptors_in_dataframe("pdbbind", readMode = readMode, saveMode = saveMode, picklenize = picklenize, returnDf = returnDf, skipMergePicklePath = skipMergePicklePath, verboseOperations = verboseOperations)
"""
