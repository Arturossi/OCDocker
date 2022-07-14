#!/usr/lib/python3

# Imports
###############################################################################
import os

from glob import glob
import pandas as pd

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
    Read the index file from pdbbind database and returns a list of the data (dict).
    Input:
     -
    Return:
     [list of dicts]
    '''
    indexFile = glob(pdbbind_archive + '/index/INDEX_refined_data.*')[0]
    # If the file exists
    if os.path.isfile(indexFile):
        # List to hold the protein data
        proteinDataOrder = f"{pdbbind_KdKi_order}M"
        proteinData = []
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
                # Add to the list having as a key the pdb code
                proteinData.append({
                    "Protein": splitedLine[0],
                    "resolution": splitedLine[1],
                    "release_year": splitedLine[2],
                    "-logKd/Ki": splitedLine[3],
                    "Ki/Kd": tp,
                    "Ki/Kd_value": kdki,
                    "Ki/Kd_order": proteinDataOrder
                    })
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

def merge_descriptors_in_dataframe(saveCsv=True):
    '''
    Reads all the descriptors jsons and return a pd.DataFrame.
    Input:
     saveCsv [bool]   DEFAULT: True - If True will save to the Prepared folder in the database
    Return:
     [pd.DataFrame]
    '''
    # Get the dataframe with descriptors and docking scores
    pdbbinddf = ocbdb.merge_descriptors_in_dataframe("pdbbind", saveCsv=False)

    # Merge the pdbbinddf DataFrame with the metadata from the PDBbind database using the Protein column as a comparer
    pdbbinddf = pd.merge(pdbbinddf, pd.DataFrame(read_index()), on="Protein", how="left")

    if saveCsv:
        # Parameterize the csvs paths
        csv_path_out = f"{parsed_archive}/PDBbind_complete.csv"
        if os.path.isfile(csv_path_out):
            octools.print_warning(f"The file {csv_path_out} already exists, it will be OVERWRITTEN!!")
        # Write the data to a new csv file
        pdbbinddf.to_csv(csv_path_out, index=False)

    return pdbbinddf
