#!/usr/lib/python3

# Imports
###############################################################################
import os

from glob import glob

from OCDocker.Initialise import *
import OCDocker.baseDB as ocbdb

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
def verify_integrity():
    '''
    Verifies the integrity of the PDBbind database
    Input:
      -
    Return:
      -
    '''
    ocbdb.verify_integrity(pdbbind_archive)

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
    ocbdb.convert_debug_to_production(pdbbind_archive, chosenAlgorithm = chosenAlgorithm, strict = strict, removeDebug = removeDebug)

def get_database_single_file():
    '''
    Parse the database into a SINGLE serializable object. (Avoid this, the database is big, so it will bug everything)
    Input:
     archive   [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      [dict of tuples]
    '''
    return ocbdb.get_database_single("pdbbind")

def get_database_multiple_files(sliceSize = 100):
    '''
    Parse the database into a multiple serializable objects. (Avoid this, the database is big, so it will bug everything)
    Input:
     sliceSize [int] - DEFAULT: 100 - Number of elements in each chunk. (Please, always use the same value)
    Return:
      [dict of tuples]
    '''
    return ocbdb.get_database("pdbbind", sliceSize = sliceSize)

def read_index():
    '''
    Read the index file from pdbbind database and return a list of data.
    Input:
     -
    Return:
     [dict]
    '''
    indexFile = glob(pdbbind_archive + '/index/INDEX_refined_data.*')[0]
    # If the file exists
    if os.path.isfile(indexFile):
        # Dict to hold the protein data
        proteinData = {"valOrder": f"{pdbbind_KiKd_order}M"}
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
                tp, val = splitedLine[4].split("=")
                # Convert all units to the same order (see the variable pdbbind_KiKd_order in initialise.py file for the precise order)
                if "mM" in val: # If mili (10e-3)
                    val = float(val.replace("mM", "")) * order[pdbbind_KiKd_order]["m"]
                elif "uM" in val: # If micro (10e-6)
                    val = float(val.replace("uM", "")) * order[pdbbind_KiKd_order]["u"]
                elif "nM" in val: # If nano (10e-9)
                    val = float(val.replace("nM", "")) * order[pdbbind_KiKd_order]["n"]
                elif "pM" in val: # If pico (10e-12)
                    val = float(val.replace("pM", "")) * order[pdbbind_KiKd_order]["p"]
                elif "fM" in val: # If femto (10e-15) not expected to show
                    val = float(val.replace("fM", "")) * order[pdbbind_KiKd_order]["f"]
                elif "cM" in val: # If centi (10e-2) not expected to show
                    val = float(val.replace("cM", "")) * order[pdbbind_KiKd_order]["c"]
                else: # Will consider just molar, but this is not expected to show
                    val = float(val.replace("M", "")) * order[pdbbind_KiKd_order]["M"]
                # Add to the dict having as a key the pdb code
                proteinData[splitedLine[0]] = {
                    "resolution": splitedLine[1],
                    "release_year": splitedLine[2],
                    "-logKd/Ki": splitedLine[3],
                    "type": tp,
                    "val": val
                    }
        # Return the data
        return proteinData
    else:
        # There is no file, throw an error
        _ = errors.file_do_not_exist(f"The file {indexFile} does not exist. Please check if the PDBbind database is correctly installed.", level = "error")
        return None
    # This return should never exist, but here it is
    return None

def run_vina(overwrite = False):
    '''
    Runs vina in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_dock("pdbbind", "vina", overwrite = overwrite)

def run_smina(overwrite = False):
    '''
    Runs smina in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_dock("pdbbind", "smina", overwrite = overwrite)

def run_plants(overwrite = False):
    '''
    Runs PLANTS in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_dock("pdbbind", "plants", overwrite = overwrite)

def prepare(overwrite = False):
    '''
    Prepares the PDBbind database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.prepare("pdbbind", overwrite = overwrite)

def read_logs(picklePath = ""):
    '''
    Parse the database into multiple serializable objects.
    Input:
     archive    [string]             - Which archive will be processed. [dudez, pdbbind, astex]
     picklePath [string] DEFAULT: "" - The path where to store the pickle file. If empty no pickle file will be generated.
    Return:
     -
    '''
    return ocbdb.read_logs("pdbbind", picklePath = picklePath)

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
    return ocbdb.generate_dock_result_csv("pdbbind", log_dumps, csv_path, chunksize=chunksize)
