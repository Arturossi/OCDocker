#!/usr/lib/python3

# Imports
###############################################################################
import os
import pandas as pd

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

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
def __thread_validation(arguments):
    '''
    Function to
    Input:
     arguments [tuple(string, string)] - Tuple containing, in this order:
        - [string] The molecule path
        - [string] The database dir
    Return:
      [int]
        1 - If a problem has been found
        0 - If no problem has been found
    '''
    return 0

def __inner_validate_database_molecules(database, subset):
    '''
    Validates all the molecules in the DUDEz database.
    Input:
      database [string] - The database dir
      subset   [string] - The database subset (DUDE_Z_ligands, DUDE_Z_decoys, extrema_decoys, goldilocks_decoys)
    Return:
      -
    '''
    return

def __validate_database_molecules():
    '''
    Validates all the molecules in the DUDEz database.
    Input:
      -
    Return:
      -
    '''
    return

def __paralel_check_repeated_ligands(arguments):
    '''
    Runs the ligand simmilarity check in parallel.
    Input:
      arguments [tuple(Ligand,list(Ligand))] - A tuple with 2 positions, the first is the reference ligand and the list of ligands to be compared.
    Return:
      -
    '''
    return None

def __check_for_repeated_ligands():
    '''
    Checks if there is any repeated ligand in the DUDEz database.
    Input:
      -
    Return:
      -
    '''
    return

## Public ##
def get_all_ligands():
    '''
    Gets all the ligands in the DUDEz database.
    Input:
      molecule [string] - Path of the molecule.
    Return:
      list(ocl.Ligand) - A list of all ligands in the DUDEz database
    '''
    return None

def get_ligands_from_molecule(molecule):
    '''
    Gets all the ligands in the DUDEz database.
    Input:
      molecule [string] - Path of the molecule.
    Return:
      list(ocl.Ligand) - A list of all ligands in the DUDEz database
    '''
    return None

def verify_integrity():
    '''
    Verifies the integrity of the DUDEz database
    Input:
      -
    Return:
      -
    '''
    return

def convert_debug_to_production(chosenAlgorithm = "ac", strict = False, removeDebug = False):
    '''
    Converts debug folders to production mode. It is required to choose an algorithm which will be used furtherly in the pipeline.
    Input:
     chosenAlgorithm [string] DEFAULT: ac  - The short code for the chosen algorithm. The choices are:
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
     strict          [bool] DEFAULT: False - If True does not convert the data even if there is only one dir, if False will convert the data if the protein has only one dir (this is good when you ran with only one algorithm, some proteins may have been run with "na")
     removeDebug     [bool] DEFAULT: False - If True removes debug folders (NO TURNING BACK), if False leave the dirs
    Return:
      -
    '''
    ocbdb.convert_debug_to_production(dudez_archive, chosenAlgorithm = chosenAlgorithm, strict = strict, removeDebug = removeDebug)

def prepare(overwrite = False, sanitize = True):
    '''
    Prepares the DUDEz database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with the output already generated.
     sanitize  [bool] DEFAULT: True  - Flag to denote if the molecule should be sanitized
    Return:
      -
    '''
    # Prepare the rest of the database
    ocbdb.prepare("dudez", overwrite = overwrite, sanitize = sanitize)
    # Verify its integrity
    #verify_integrity()

def run_p2rank(overwrite = False):
    '''
    Runs P2Rank in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_p2rank("dudez", overwrite = overwrite)

def run_vina(overwrite = False):
    '''
    Runs vina in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_dock("dudez", "vina", overwrite = overwrite)

def run_smina(overwrite = False):
    '''
    Runs smina in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_dock("dudez", "smina", overwrite = overwrite)

def run_plants(overwrite = False):
    '''
    Runs PLANTS in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_dock("dudez", "plants", overwrite = overwrite)

def read_logs(picklePath = ""):
    '''
    Parse the database into multiple serializable objects.
    Input:
     picklePath [string] DEFAULT: "" - The path where to store the pickle file. If empty no pickle file will be generated
    Return:
     -
    '''
    return ocbdb.read_logs("dudez", picklePath = picklePath)

def generate_dock_result_csv(log_dumps, csv_path, chunksize=500):
    '''
    Uses the structure from read_logs to generate an output for all docking softwares.
    Input:
     archive   [string]                                     - Which archive will be processed [dudez, pdbbind, astex]
     log_dumps [dict of dicts of pd.DataFrame]              - The dump generated from the read_logs function
     csv_path  [string]                                     - Path to the csv file
     chunksize [int]                           DEFAULT: 500 - Chunk size to write the csv
    Return:
     -
    '''
    return ocbdb.generate_dock_result_csv("dudez", log_dumps, csv_path, chunksize=chunksize)

def merge_descriptors_in_dataframe(saveCsv=True):
    '''
    Reads all the descriptors jsons and return a pd.DataFrame.
    Input:
     saveCsv [bool] DEFAULT: True - If True will save to the Prepared folder in the database
    Return:
     [pd.DataFrame]
    '''
    # Get the dataframe with descriptors and docking scores
    dudezdf = ocbdb.merge_descriptors_in_dataframe("dudez", saveCsv=False)

    if saveCsv:
        # Parameterize the csvs paths
        csv_path_out = f"{parsed_archive}/DUDEz_complete.csv"
        if os.path.isfile(csv_path_out):
            octools.print_warning(f"The file {csv_path_out} already exists, it will be OVERWRITTEN!!")
        # Write the data to a new csv file
        dudezdf.to_csv(csv_path_out, index=False)

    return dudezdf
