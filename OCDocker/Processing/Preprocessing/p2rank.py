#!/usr/lib/python3

# Description
###############################################################################
'''
This module is responsible for digest processing.

It is imported as:

import OCDocker.Processing.Preprocessing.p2rank as ocp2rank
'''

# Imports
###############################################################################
import gc
import os
import shutil

from glob import glob
from multiprocessing import Pool
from threading import Lock
from tqdm import tqdm
from typing import List, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.ExternalTools.runprank as runprank
import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Validation as ocvalidation

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
def __run_p2rank(dir: str, fin: str, overwrite: bool = False) -> None:
    '''Runs p2rank for a given directory.

    Parameters
    ----------
    dir : str
        Path where the data is.
    fin : str
        PDB file as input.
    overwrite : bool, optional
        Flag for demanding file overwrite, by default False.

    Returns
    -------
    None
    '''

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

    # Create a lock for multithreading
    lock = Lock()
    # Start the lock with statement
    with lock:
        try:
            # Run p2rank
            runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = True if args.output_level >= 3 else False, overwrite = overwrite)
        except Exception as e:
            ocprint.print_warning(f"The protein '{dir}' had a problem while running p2rank. Retrying to run p2rank. Exception: {e}")
            runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = True if args.output_level >= 3 else False, overwrite = overwrite)

    return None

def __core_p2rank(dir: str, overwrite: bool) -> None:
    '''Core function to run p2rank.

    Parameters
    ----------
    dir : str
        Path where the data is.
    overwrite : bool
        Flag for demanding file overwrite.

    Returns
    -------
    None
    '''

    # Set the output path
    fout = f"{dir}/p2rank"

    # Create the p2rank output dir
    _ = ocff.safe_create_dir(fout)

    # Parameterizing box count
    boxCount = len(glob(f"{fout}/box*.pdb"))

    # If overwrite mode is on or there is no box in the p2rank output, p2rank will run
    if boxCount == 0 or overwrite:
        # Run p2rank
        __run_p2rank(dir, f"{dir}/receptor.pdb", overwrite=overwrite) 
    else:
        ocprint.print_info(f"The protein '{dir}' already has its p2rank output generated, skipping its execution.")

    return None

def __thread_p2rank(arguments: Tuple[str, bool, str]) -> None:
    '''Thread aid function to call __core_p2rank.

    Parameters
    ----------
    arguments : Tuple[str, bool, str]
        Tuple with the arguments to be passed to __core_p2rank. The arguments are: (dir, overwrite, archive). Where dir is the path where the data is, overwrite is a flag for demanding file overwrite and archive is which archive will be processed [dudez, pdbbind].

    Returns
    -------
    None
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Call core prepare function (shared between thread and no thread)
        return __core_p2rank(arguments[0], arguments[1])

def __p2rank_parallel(paths: List[str], overwrite: bool, desc: str) -> None:
    '''Runs p2rank in parallel.

    Parameters
    ----------
    paths: List[str]
        List of directories to be processed.
    overwrite: bool
        Flag for demanding file overwrite.
    desc: str
        Description to be used in the progress bar.

    Returns
    -------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for path in paths:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((path, overwrite))
    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for _ in tqdm(p.imap_unordered(__thread_p2rank, arguments), total = len(arguments), desc = desc):
                # Clear the memory
                gc.collect()
    except IOError as e:
        errMsg = f"Problem while running p2rank in parallel. Exception: {e}"
        ocprint.print_error_log(errMsg, f"{logdir}/p2rank_report.log")
        ocprint.print_error(errMsg)

    # Return
    return None

def __p2rank_no_parallel(paths: List[str], overwrite: bool, desc: str) -> None:
    '''Runs p2rank in serial.

    Parameters
    ----------
    paths: List[str]
        List of directories to be processed.
    overwrite: bool
        Flag for demanding file overwrite.
    desc: str
        Description to be used in the progress bar.

    Returns
    -------
    None
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        for path in tqdm(iterable=paths, total=len(paths), desc=desc):
            # Call the core p2rank function
            __core_p2rank(path, overwrite)
            # Clear the memory
            gc.collect()
    return None

def __p2rank_single(path: str, overwrite: bool) -> None:
    '''Runs p2rank in a single directory.

    TODO: Add the support to custom databases.

    Parameters
    ----------
    path: str
        Directory to be processed.
    overwrite: bool
        Flag for demanding file overwrite.
    desc: str
        Description to be used in the progress bar.

    Returns
    -------
    None
    '''

    # Call the core p2rank function
    __core_p2rank(path, overwrite)

    # Clear the memory
    gc.collect()

    return None

## Public ##

def run_p2rank(paths: Union[List[str], str], overwrite: bool) -> None:
    '''Runs p2rank.

    Parameters
    ----------
    paths : List[str] | str
        The list of directories or the directory to be processed.
    overwrite : bool
        If True, the function will overwrite the files if they already exists.
    '''

    # If the path is a list
    if isinstance(paths, list):
        # If logfile exists, backup it
        oclogging.backup_log(f"p2rank_report")

        # Set the description
        label = f"Running p2rank"
        
        # Check if multiprocessing is enabled
        if args.multiprocess:
            # Prepare the pdbbind
            __p2rank_parallel(paths, overwrite, label)
        else:
            # Prepare the database
            __p2rank_no_parallel(paths, overwrite, label)
    else:
        __p2rank_single(paths, overwrite)

def convert_debug_to_production(chosenArchive: str, chosenAlgorithm: str = "ac", strict: bool = False, removeDebug: bool = False) -> None:
    '''Converts debug folders to production mode. It is required to choose an algorithm which will be used furtherly in the pipeline.

    Parameters
    ----------
    chosenArchive : str
        The archive to be converted. The options are [dudez, pdbbind].
    chosenAlgorithm : str, optional
        The algorithm to be used in the pipeline. The default is "ac". The short code for the algorithms are
            AffinityPropagation: ap, 
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
    strict : bool, optional
        If True does not convert the data even if there is only one dir, if False will convert the data if the protein has only one dir (this is good when you ran with only one algorithm, some proteins may have been run with "na"). The default is False.
    removeDebug : bool, optional
        If True removes the debug folder after the conversion, if False keeps the debug folder. The default is False.
    '''

    # Generate boxes for all receptors
    ocprint.printv("Converting p2rank debug to production file tree.")

    # Get all dirs paths in the DUDEz database
    dirs = glob(f"{chosenArchive}/*")

    # Redirect output to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # For each directory in the database folder
        for dir in tqdm(iterable=dirs, total=len(dirs)):
            # Print text
            ocprint.printv(f"Processing '{dir}'.")

            # Parameterize the p2rank dir
            p2rankDir = f"{dir}/p2rank"

            # Flag to check if the algorithm folder has been found
            hasDir = False

            # Get all the dirs which are in the allowed values
            p2rankFiles = [d for d in glob(f"{p2rankDir}/*") if ocvalidation.is_algorithm_allowed(d) and os.path.isdir(d)]

            # Parameterize the amount of dirs
            p2rankFilesLen = len(p2rankFiles)

            # If there is any dir
            if p2rankFilesLen > 0:
                # If there is only one file
                if p2rankFilesLen == 1 and not strict:
                    ocprint.print_info(f"There is only one file.")
                    # Set the hasDir as true
                    hasDir = True
                    # Get the boxes
                    boxes = glob(f"{p2rankFiles[0]}/*")
                    # If no box is found (folders WILL NOT BE REMOVED)
                    if len(boxes) < 1:
                        ocprint.print_error(f"The protein '{dir}' has no box!!!!!")
                        ocprint.print_error_log(f"The protein '{dir}' has no box!!!!!", f"{logdir}/{chosenArchive}_conversion_report.log")
                        continue
                    # Get the algorithm name
                    algorithm = p2rankFiles[0].split(os.path.sep)[-1]
                    # For each box found
                    for box in boxes:
                        # Create the destination box name
                        boxDest = os.path.basename(box).replace(f"_{algorithm}","")
                        # Copy the box to the parent directory
                        shutil.copyfile(box, f"{p2rankDir}/{boxDest}")
                else:
                    for p2rankFile in p2rankFiles:
                        # Get the algorithm name
                        algorithm = p2rankFile.split(os.path.sep)[-1]
                        if algorithm == chosenAlgorithm:
                            # Set the hasDir as true
                            hasDir = True
                            # Get the boxes
                            boxes = glob(f"{p2rankFile}/*")
                            # If no box is found (folders WILL NOT BE REMOVED)
                            if len(boxes) < 1:
                                ocprint.print_error(f"The protein '{dir}' has no box!!!!!")
                                ocprint.print_error_log(f"The protein '{dir}' has no box!!!!!", f"{logdir}/{chosenArchive}_conversion_report.log")
                                continue
                            # Get the algorithm name
                            algorithm = p2rankFile.split(os.path.sep)[-1]
                # If the algorithm folder has been found
                if hasDir:
                    # Check if remove is set
                    if removeDebug:
                        # Print to the user the information
                        ocprint.print_info(f"Removing files for '{dir}'")
                        # For each file
                        for p2rankFile in p2rankFiles:
                            # Remove the folder and its contets
                            shutil.rmtree(p2rankFile)
                else:
                    ocprint.print_error(f"The algorithm '{chosenAlgorithm}' has not been found for the protein '{dir}'.")
                    ocprint.print_error_log(f"The algorithm '{chosenAlgorithm}' has not been found for the protein '{dir}'.", f"{logdir}/{chosenArchive}_conversion_report.log")
            else:
                ocprint.printv(f"Nothing to convert for '{dir}'. Skipping...")
                continue
    return None
