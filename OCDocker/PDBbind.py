#!/usr/lib/python3

# Imports
###############################################################################
import os

import pandas as pd
import vaex.dataframe as vdf

from glob import glob
from typing import Dict, List, Union

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
Sets of classes and functions that are used to process the PDBbind dataset.

They are imported as:

import OCDocker.PDBbind as ocpdbbind
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

    Raises
    ------
    None
    '''

    ocbdb.verify_integrity(pdbbind_archive)
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

    Raises
    ------
    None
    '''

    ocbdb.convert_debug_to_production(pdbbind_archive, chosenAlgorithm = chosenAlgorithm, strict = strict, removeDebug = removeDebug)
    return None

def read_index() -> Union[Dict[str, List[str]], None]:
    '''Read the index file from pdbbind database and returns a list of the data (dict).

    Parameters
    ----------
    None

    Returns
    -------
    Dict[str, List[str]] | None
        A list of dictionaries with the data from the index file. If the file does not exist, it will return None.

    Raises
    ------
    None
    '''

    indexFile = glob(pdbbind_archive + '/index/INDEX_refined_data.*')[0]
    # If the file exists
    if os.path.isfile(indexFile):
        # List to hold the protein data
        proteinDataOrder = f"{pdbbind_KdKi_order}M"
        proteinData = {
                    "Protein": [],
                    "resolution": [],
                    "release_year": [],
                    "-logKd/Ki": [],
                    "Ki/Kd": [],
                    "Ki/Kd_value": [],
                    "Ki/Kd_order": []
                    }
        # Open the file in read mode
        with open(indexFile, "r") as f:
            # This will loop the entire file (better than load the whole file in memory... imagine a huge file being loaded...)
            while True:
                # Read one line
                line = f.readline()
                # Check if there is a line
                if not line:
                    # If line is none, break the loop
                    break
                # If the line starts with a #
                if line.startswith("#"):
                    # Skip it (no useful info)
                    continue
                # Split the line in spaces
                splitedLine = line.split()
                # The columns are listed below (with a sample)
                # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
                # 2r58  2.00  2007   2.00  Kd=10mM       // 2r58.pdf (MLY)
                # Separate the type (Kd/Ki) from the value
                tp, kdki = splitedLine[4].split("=")
                # Convert all units to the same order (see the variable pdbbind_KdKi_order in initialise.py file for the precise order)
                if "mM" in kdki: # If mili (10e-3)
                    kdki = float(kdki.replace("mM", "")) * order[pdbbind_KdKi_order]["m"]
                elif "uM" in kdki: # If micro (10e-6)
                    kdki = float(kdki.replace("uM", "")) * order[pdbbind_KdKi_order]["u"]
                elif "nM" in kdki: # If nano (10e-9)
                    kdki = float(kdki.replace("nM", "")) * order[pdbbind_KdKi_order]["n"]
                elif "pM" in kdki: # If pico (10e-12)
                    kdki = float(kdki.replace("pM", "")) * order[pdbbind_KdKi_order]["p"]
                elif "fM" in kdki: # If femto (10e-15) not expected to show
                    kdki = float(kdki.replace("fM", "")) * order[pdbbind_KdKi_order]["f"]
                elif "cM" in kdki: # If centi (10e-2) not expected to show
                    kdki = float(kdki.replace("cM", "")) * order[pdbbind_KdKi_order]["c"]
                else: # Will consider just molar, but this is not expected to show
                    kdki = float(kdki.replace("M", "")) * order[pdbbind_KdKi_order]["M"]

                # Add to the dict
                proteinData["Protein"].append(splitedLine[0])
                proteinData["resolution"].append(splitedLine[1])
                proteinData["release_year"].append(splitedLine[2])
                proteinData["-logKd/Ki"].append(splitedLine[3])
                proteinData["Ki/Kd"].append(tp)
                proteinData["Ki/Kd_value"].append(kdki)
                proteinData["Ki/Kd_order"].append(proteinDataOrder)

        # Return the data
        return proteinData
    else:
        # There is no file, throw an error
        _ = errors.file_do_not_exist(f"The file {indexFile} does not exist. Please check if the PDBbind database is correctly installed.", level = "error")
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

    Raises
    ------
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

    return ocbdb.run_dock("pdbbind", "gnina", overwrite = overwrite)

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

    Raises
    ------
    None
    '''

    return ocbdb.run_dock("pdbbind", "vina", overwrite = overwrite)

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

    Raises
    ------
    None
    '''

    return ocbdb.run_dock("pdbbind", "smina", overwrite = overwrite)

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

    Raises
    ------
    None
    '''

    return ocbdb.run_dock("pdbbind", "plants", overwrite = overwrite)

def prepare(overwrite: bool = False) -> None:
    '''Prepares the PDBbind database.

    Parameters
    ----------
    overwrite : bool, optional
        If True, it will overwrite the results. If False, it will not run the preparation if the results already exist, by default False.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    return ocbdb.prepare("pdbbind", overwrite = overwrite)

def read_logs(picklePath: str = "") -> Union[Dict[str, vdf.DataFrameLocal], None]:
    '''Parse the database into multiple serializable objects.

    Parameters
    ----------
    picklePath : str, optional
        The path to the pickle file, by default "".

    Returns
    -------
    Dict[str, Dict[str, vdf.DataFrameLocal]]
        The parsed data.

    Raises
    ------
    None
    '''

    return ocbdb.read_logs("pdbbind", picklePath = picklePath)

def generate_dock_result_csv(csv_path: str = "", log_dumps: Union[Dict[str, vdf.DataFrameLocal], None] = None) -> None:
    '''Uses the structure from read_logs to generate an output for all docking softwares.

    Parameters
    ----------
    csv_path : str, optional
        The path to the output csv file. If not specified, it will use the default path, by default f"{parsed_archive}/PDBbind.csv".
    log_dumps : Dict[str, vdf.DataFrameLocal], optional
        The parsed data.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Check if csv_path is empty
    if csv_path == "":
        # It is empty, use the default path
        csv_path = f"{parsed_archive}/pdbbind.csv"

    return ocbdb.generate_dock_result_csv("pdbbind", csv_path, log_dumps = log_dumps)

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
    return ocbdb.merge_descriptors_in_dataframe("pdbbind", readMode = readMode, saveMode = saveMode, picklenize = picklenize, returnDf = returnDf, skipMergePicklePath = skipMergePicklePath)
