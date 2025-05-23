#!/usr/bin/env python3

# Description
###############################################################################
'''
This module is responsible for digest processing.

It is imported as:

import OCDocker.Processing.Postprocessing.Digest as ocdigest
'''

# Imports
###############################################################################
import gc
import os

from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Tuple, Union

import OCDocker.Docking.Future.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.Printing as ocprint

from OCDocker.Initialise import *

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

def __core_generate_digest(path: str, ligandDir: str, archive: str, overwrite: bool, digestFormat: str = "json") -> int:
    '''Generate the digest file for a given protein and ligand.

    Parameters
    ----------
    path : str
        The path to the protein directory.
    ligandDir : str
        If the ligand is not in the same directory as the receptor, this is the path to the ligand directory. By default "". If this is not empty, the ligand will be searched in this directory, otherwise, it will be searched in the same directory as the receptor.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    overwrite : bool
        If the docking output already exists, should it be overwritten?
    digestFormat : str, optional
        The format of the digest file. By default "json".

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Get the protein name (which is the last directory in the path)
    ptn = path.split("/")[-1]

    # If is the index directory, ignore
    if ptn in ['index']:
        return ocerror.Error.unnalowed_dir()

    ligandDescriptorPath = f"{ligandDir}/ligand_descriptors.json"

    # If the complex has descriptor files for ligand
    if os.path.isfile(ligandDescriptorPath):
        # Run for gnina
        logPath = f"{ligandDir}/gninaFiles/gnina_0.log" # TODO: add support to multiple boxes/runs
        _ = ocgnina.generate_digest(f"{ligandDir}/dockingDigest.json", logPath, overwrite = overwrite, digestFormat = digestFormat)
        # Run for vina
        logPath = f"{ligandDir}/vinaFiles/vina_0.log" # TODO: add support to multiple boxes/runs
        _ = ocvina.generate_digest(f"{ligandDir}/dockingDigest.json", logPath, overwrite = overwrite, digestFormat = digestFormat)
        # Run for smina
        logPath = f"{ligandDir}/sminaFiles/smina_0.log" # TODO: add support to multiple boxes/runs
        _ = ocsmina.generate_digest(f"{ligandDir}/dockingDigest.json", logPath, overwrite = overwrite, digestFormat = digestFormat)
        # Run for PLANTS
        logPath = f"{ligandDir}/plantsFiles/run/bestranking.csv" # TODO: add support to multiple boxes/runs
        _ = ocplants.generate_digest(f"{ligandDir}/dockingDigest.json", logPath, overwrite = overwrite, digestFormat = digestFormat)
    else:
        errMsg = f"There is no ligand descriptor json file for the protein in the path '{ligandDescriptorPath}'."
        ocprint.print_error_log(errMsg, f"{logdir}/{archive}_docking_digest_run_report_ERROR.log")
        return ocerror.Error.receptor_or_ligand_descriptor_does_not_exist(errMsg, level = ocerror.ReportLevel.ERROR)

    return ocerror.Error.ok()

def __thread_generate_digest(arguments: list) -> int:
    '''Thread aid function to call __core_generate_digest.

    Parameters
    ----------
    arguments : list
        The arguments to be passed to __core_generate_digest.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Call the core dock function passing the arguments correctly
        returnState = __core_generate_digest(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])

    return returnState

def __generate_digest_parallel(complexList: List[Tuple[str, List[str]]], archive: str, overwrite: bool, digestFormat: str, desc: str) -> int:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    complexList : List[Tuple[str, List[str]]]
        A list of tuples with the path to the protein directory and a list of ligand directories.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    digestFormat : str
        Which digest format will be used [json].
    overwrite : bool
        If the docking output already exists, should it be overwritten?
    desc : str
        The description of the progress bar.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []

    # For each file in complexList
    for cl in complexList:
        # Now loop over the ligands of this protein
        for ligandDir in cl[1]:
            # Add the arguments to the list (creating one execution for each pair receptor-ligand)
            arguments.append((cl[0], ligandDir, archive, overwrite, digestFormat))

    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(available_cores) as p:
            # Perform the multi process
            for _ in tqdm(p.imap_unordered(__thread_generate_digest, arguments), total = len(arguments), desc = desc):
                # Clear the memory
                gc.collect()
    except IOError as e:
        errMsg = f"Problem while generating docking digest in parallel. Exception: {e}"
        ocprint.print_error_log(errMsg, f"{logdir}/{archive}_docking_report.log")
        return ocerror.Error.docking_failed(errMsg, level = ocerror.ReportLevel.ERROR)

    # Return
    return ocerror.Error.ok() # FIXME: This should be changed to return the error code in a way to track all docking ocerror.Error

def __generate_digest_no_parallel(complexList: List[Tuple[str, List[str]]], archive: str, overwrite: bool, digestFormat: str, desc: str) -> int:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_generate_digest function.

    Parameters
    ----------
    complexList : List[Tuple[str, List[str]]]
        A list of tuples with the path to the protein directory and a list of ligand directories.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    digestFormat : str
        Which digest format will be used [json].
    overwrite : bool
        If the docking output already exists, should it be overwritten?
    desc : str
        The description of the progress bar.
    
    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # For each file in dirs
        for cl in tqdm(iterable = complexList, total = len(complexList), desc=desc):
            for ligandDir in cl[1]:
                # Call the core dock function (shared between parallel and not parallel)
                __core_generate_digest(cl[0], ligandDir, archive, overwrite, digestFormat)

            # Clear the memory
            gc.collect()
        # Clear the memory
        gc.collect()

    return ocerror.Error.ok() # FIXME: This should be changed to return the error code in a way to track all docking ocerror.Error

def __generate_digest_single(complex: Tuple[str, List[str]], archive: str, overwrite: bool, digestFormat: str, desc: str) -> int:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_generate_digest function.

    Parameters
    ----------
    complex : List[Tuple[str, List[str]]]
        A tuple with the path to the protein directory and a list of ligand directories.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    digestFormat : str
        Which digest format will be used [json].
    overwrite : bool
        If the docking output already exists, should it be overwritten?
    desc : str
        The description of the progress bar.
    
    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # For each file in dirs
    for ligandDir in tqdm(iterable = complex[1], total = len(complex[1]), desc=desc):
        # Call the core dock function (shared between parallel and not parallel)
        __core_generate_digest(complex[0], ligandDir, archive, overwrite, digestFormat)

        # Clear the memory
        gc.collect()

    return ocerror.Error.ok() # FIXME: This should be changed to return the error code in a way to track all docking ocerror.Error

## Public ##

def generate_digest(paths: Union[List[Tuple[str, List[str]]], Tuple[str, List[str]]], archive: str, overwrite: bool, digestFormat: str = "json") -> None:
    '''Generate the digest for the docking output.

    Parameters
    ----------
    paths : List[Tuple[str, List[str]]] | Tuple[str, List[str]]
        The list of directories or the directory to be processed.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    overwrite : bool
        If the docking output already exists, should it be overwritten?
    digestFormat : str, optional
        Which digest format will be used [json], by default "json".
    '''

    # Set the label
    label = f"Processing {archive}"

    # If the path is a list
    if isinstance(paths, list):

        # If logfiles exists, backup them
        oclogging.backup_log(f"{archive}_docking_digest_run_report_ERROR")

        # Check if multiprocessing is enabled
        if multiprocess:
            # Prepare the pdbbind
            __generate_digest_parallel(paths, archive, overwrite, digestFormat, label)
        else:
            # Prepare the database
            __generate_digest_no_parallel(paths, archive, overwrite, digestFormat, label)
    else:
        __generate_digest_single(paths, archive, overwrite, digestFormat, label)

