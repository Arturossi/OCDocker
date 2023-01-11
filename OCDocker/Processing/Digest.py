#!/usr/lib/python3

# Imports
###############################################################################
import gc
import os
import time

from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Tuple


from OCDocker.Initialise import *

import OCDocker.Toolbox as octools
import OCDocker.Docking.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina

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
This module is responsible for digest processing.

It is imported as:

import OCDocker.Processing.Digest as ocdigest
'''

# Classes
###############################################################################


# Functions
###############################################################################
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

    Raises
    ------
    None
    '''

    # Get the protein name (which is the last directory in the path)
    ptn = path.split("/")[-1]

    # If is the index directory, ignore
    if ptn in ['index']:
        return errors.unnalowed_dir()

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
        octools.print_error_log(errMsg, f"{logdir}/{archive}_docking_digest_run_report_ERROR.log")
        return errors.receptor_or_ligand_descriptor_does_not_exist(errMsg, level = "error")

    return errors.ok()

def __thread_generate_digest(arguments: list) -> int:
    '''Thread aid function to call __core_run_dock.

    Parameters
    ----------
    arguments : list
        The arguments to be passed to __core_run_dock.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core dock function passing the arguments correctly
        returnState = __core_generate_digest(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])

    return returnState

def generate_digest_parallel(complexList: List[Tuple[str, List[str]]], archive: str, overwrite: bool, digestFormat: str, desc: str) -> int:
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

    Raises
    ------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []

    # For each file in complexList
    for cl in complexList:
        # Now loop over the ligands of this protein
        for ligandDir in cl[1]:
            # Add the arguments to the list (creating one execution for each pair receptor-ligand)
            arguments.append((cl[0], ligandDir, archive, overwrite, digestFormat))

    # If logfile exists, backup it (for error and warnings)
    if os.path.isfile(f"{logdir}/{archive}_docking_digest_report_ERROR.log"):
        if not os.path.isdir(f"{logdir}/{archive}_docking_digest_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_docking_digest_report_past")
        os.rename(f"{logdir}/{archive}_docking_digest_report_ERROR.log", f"{logdir}/{archive}_docking_digest_report_past/{archive}_docking_digest_report_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")

    if os.path.isfile(f"{logdir}/{archive}_docking_digest_report_WARNING.log"):
        if not os.path.isdir(f"{logdir}/{archive}_docking_digest_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_docking_digest_report_past")
        os.rename(f"{logdir}/{archive}_docking_digest_report_WARNING.log", f"{logdir}/{archive}_docking_digest_report_past/{archive}_docking_digest_report_WARNING_{time.strftime('%d%m%Y-%H%M%S')}.log")

    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for _ in tqdm(p.imap_unordered(__thread_generate_digest, arguments), total = len(arguments), desc = desc):
                # Clear the memory
                gc.collect()
    except IOError as e:
        errMsg = f"Problem while generating docking digest in parallel. Exception: {e}"
        octools.print_error_log(errMsg, f"{logdir}/{archive}_docking_report.log")
        return errors.docking_failed(errMsg, level = "error")

    # Return
    return errors.ok() # FIXME: This should be changed to return the error code in a way to track all docking errors

def generate_digest_no_parallel(complexList: List[Tuple[str, List[str]]], archive: str, overwrite: bool, digestFormat: str, desc: str) -> int:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_run_dock function.

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

    Raises
    ------
    None
    '''

    # If logfile exists, backup it (for error and warnings)
    if os.path.isfile(f"{logdir}/{archive}_docking_digest_report_ERROR.log"):
        if not os.path.isdir(f"{logdir}/{archive}_docking_digest_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_docking_digest_report_past")
        os.rename(f"{logdir}/{archive}_docking_digest_report_ERROR.log", f"{logdir}/{archive}_docking_digest_report_past/{archive}_docking_digest_report_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")

    if os.path.isfile(f"{logdir}/{archive}_docking_digest_report_WARNING.log"):
        if not os.path.isdir(f"{logdir}/{archive}_docking_digest_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_docking_digest_report_past")
        os.rename(f"{logdir}/{archive}_docking_digest_report_WARNING.log", f"{logdir}/{archive}_docking_digest_report_past/{archive}_docking_digest_report_WARNING_{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # For each file in dirs
        for cl in tqdm(iterable = complexList, total = len(complexList), desc=desc):
            for ligandDir in cl[1]:
                # Call the core dock function (shared between parallel and not parallel)
                __core_generate_digest(cl[0], ligandDir, archive, overwrite, digestFormat)

            # Clear the memory
            gc.collect()
        # Clear the memory
        gc.collect()

    return errors.ok() # FIXME: This should be changed to return the error code in a way to track all docking errors
