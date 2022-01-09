#!/usr/lib/python3

# Imports
###############################################################################
import os
from glob import glob
from tqdm import tqdm

from OCDocker.Initialise import *
import OCDocker.Vina as ocvina
import OCDocker.Toolbox as octools
import OCDocker.ExternalTools.runprank as runprank

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
Sets of classes and functions that are used to process the Astex Diverse
dataset.

They are imported as:

import OCDocker.Astex as ocastex
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##
def __run_p2rank(dir):
    '''
    Runs p2rank for a given directory.
    Input:
      dir [string] - Directory of the protein to run p2rank.
    Return:
      -
    '''
    # Set the input file name path
    fin = f"{dir}/protein.pdb"

    # Set the output path
    fout = f"{dir}/p2rank"

    # Algorithms to be analyzed (Only Agglomerative Clustering)
    algorithms = {
        "AffinityPropagation": False,
        "AgglomerativeClustering": True,
        "Birch": False,
        "DBSCAN": False,
        "KMeans": False,
        "MeanShift": False,
        "MiniBatchKMeans": False,
        "NoCluster": False,
        "OPTICS": False,
        "SpectralClustering": False
    }

    try:
        # Run p2rank
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = args.debug, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = args.verbosity)
    except Exception as e:
        octools.print_warn(f"The protein '{dir}' had a problem while running p2rank. Retrying to run p2rank. Exception: {e}  ")
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = args.debug, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = args.verbosity)

    return

def __run_create_vina_conf_from_box(dir):
    '''
    Creates vina conf file from box
    Input:
      dir [string] - Directory of the protein to run p2rank.
    Return:
      -
    '''
    # Set the input file name path
    fin = f"{dir}/protein.pdb"

    # Run vina
    ocvina.generate_vina_files_database(dir, fin)

    return

def verify_integrity():
    '''
    Verifies the integrity of the Astex database
    Input:
      -
    Return:
      -
    '''
    # Verify the integrity of the database
    octools.printv("Verifiying the integrity of the Astex database")

    # Get all dirs paths in the DUDEZ database
    dirs = glob(f"{astex_archive}/*")

    # For each directory in the database folder
    for dir in tqdm(iterable=dirs, total=len(dirs)):
        # Parameterizing paths
        p2rankDir = f"{dir}/p2rank"
        vinaDir = f"{dir}/vinaFiles"

        octools.printv(f"Checking directories for the protein '{dir}'.")

        # If has no p2rank dir
        if not os.path.isdir(p2rankDir):
            octools.print_warning(f"The protein '{dir}' has no p2rank folder. Trying to fix...")

            # Create the p2rank output dir
            errorCode = octools.safe_create_dir(p2rankDir)

            if os.path.isdir(p2rankDir):
                octools.print_success(f"The p2rank dir has been generated for '{dir}'.")
            else:
                octools.print_error(f"Unable to generate the p2rank dir for '{dir}'... Error code {errorCode}.")
                octools.print_error_log(f"Unable to generate the p2rank dir for '{dir}'... Error code {errorCode}.", f"{logdir}/Astex_integrity_report.log")
                continue

        # If has no vinaFiles dir
        if not os.path.isdir(vinaDir):
            octools.print_warning(f"The protein '{dir}' has no vinaFiles folder. Trying to fix...")

            # Create the p2rank output dir
            errorCode = octools.safe_create_dir(vinaDir)

            if os.path.isdir(vinaDir):
                octools.print_success(f"The vinaFiles dir has been generated for '{dir}'.")
            else:
                octools.print_error(f"Unable to generate the vinaFiles dir for '{dir}'... Error code {errorCode}.")
                octools.print_error_log(f"Unable to generate the vinaFiles dir for '{dir}'... Error code {errorCode}.", f"{logdir}/Astex_integrity_report.log")
                continue

        octools.printv(f"Checking files for the protein '{dir}'")

        # Check how many boxes are in the p2rankDir
        boxCount = len(glob(f"{p2rankDir}/box*.pdb"))

        # If there is no box in the p2rank output, p2rank will run
        if boxCount == 0:
            octools.print_warning(f"The protein '{dir}' has no box file. Trying to fix...")

            # Run p2rank
            __run_p2rank(dir)

            # Check how many boxes are in the p2rankDir (again)
            boxCount = len(glob(f"{p2rankDir}/box*.pdb"))

            if boxCount > 0:
                octools.print_success(f"Box files generated for '{dir}'.")
            else:
                octools.print_error(f"The protein '{dir}' still has no box file.")
                octools.print_error_log(f"The protein '{dir}' still has no box file.", f"{logdir}/Astex_integrity_report.log")
                continue

        # If there is not the same amount of box files as folders in vinaFiles folder
        if len(glob(f"{dir}/vinaFiles/*")) < boxCount:
            octools.print_warning(f"The protein '{dir}' has not the same amount of vina conf files as the amount of box files. Trying to fix...")

            # Run the vina conf creation from box
            __run_create_vina_conf_from_box(dir)

            # If there is not the same amount of box files as folders in vinaFiles folder (again)
            if len(glob(f"{dir}/vinaFiles/*")) < boxCount:
                octools.print_success(f"Conf files generated for '{dir}'.")
            else:
                octools.print_error(f"Unable to generate the conf files for '{dir}'...")
                octools.print_error_log(f"Unable to generate the conf files dir for '{dir}'...", f"{logdir}/Astex_integrity_report.log")
                continue

    octools.printv("Integrity check of the Astex database accomplished.")

    return

def prepare(overwrite = False):
    '''
    Prepares the Astex database.
    Input:
      overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    # Generate boxes for all receptors
    octools.printv("Generating information regarding possible ligand site.")

    # Get all dirs paths in the DUDEZ database
    dirs = glob(f"{astex_archive}/*")

    # For each directory in the database folder
    for dir in tqdm(iterable=dirs, total=len(dirs)):
        # Set the input file name path
        fin = f"{dir}/protein"

        # Set the ligand input file name path
        lfin = f"{dir}/ligand"

        # Find the protein name
        ptn = dir.split("/")[-1]

        # Set the output path
        fout = f"{dir}/p2rank"

        # Create the p2rank output dir
        _ = octools.safe_create_dir(fout)

        # Convert the protein file from mol2 to pdb
        _ = octools.convertMols(f"{fin}.mol2", f"{fin}.pdb")

        # Convert the ligand file from mol to mol2
        _ = octools.convertMols(f"{lfin}.mol", f"{lfin}.mol2")

        # Reset the input file variable
        fin = f"{fin}.pdb"

        # Parameterizing box count
        boxCount = len(glob(f"{fout}/box*.pdb"))

        # If overwrite mode is on or there is no box in the p2rank output, p2rank will run
        if boxCount == 0 or overwrite:
            # Run p2rank
            __run_p2rank(dir)

        else:
            octools.print_info(f"The protein '{dir}' already has its p2rank output generated, skipping its execution.")

        # If overwrite mode is on or there is not the same amount of box files as folders in vinaFiles folder
        if len(glob(f"{dir}/vinaFiles/*")) == boxCount or overwrite:
            # Create the vina inputs from the boxes
            ocvina.generate_vina_files_database(dir, fin)
        else:
            octools.print_info(f"The protein '{dir}' already has its vina file generated, skipping its execution.")

    return
