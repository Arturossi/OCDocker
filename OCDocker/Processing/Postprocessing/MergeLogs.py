#!/usr/lib/python3

# Description
###############################################################################
'''
This module is responsible for digest processing.

It is imported as:

import OCDocker.Processing.Postprocessing.MergeLogs as ocmergelogs
'''

# Imports
###############################################################################
import errno
import gc
import os
import vaex

import numpy as np
import vaex.dataframe as vdf

from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.Printing as ocprint

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

def __core_merge_descriptors_in_dataframe(processDirPackage: Tuple[str, str]) -> vdf.DataFrameLocal:
    '''Reads the descriptor and receptor json then parse them into a dataframe.

    Parameters
    ----------
    processDirPackage : Tuple(str, str)
        Tuple containing the processDir and the package. The tuple is in the format: (processDir, receptor_descriptor_path).
    archive : str
        Which archive will be processed [dudez, pdbbind].

    Returns
    -------
    vdf.DataFrameLocal
        DataFrame containing the results of the docking.

    Raises
    ------
    None
    '''

    # Unpack the tuple
    processDir, receptor_descriptor_path = processDirPackage

    # Find ptn name
    ptn = os.path.dirname(receptor_descriptor_path).split(os.path.sep)[-1]

    # Find which kind of archive it will be
    ligand_descriptor_path = f"{processDir}/ligand_descriptors.json"

    try:
        # Check if there is the receptor json, if yes, load it
        if os.path.isfile(receptor_descriptor_path):
            receptor_descriptors = ocr.read_descriptors_from_json(receptor_descriptor_path, returnVaex = True)
            # Nasty fix for descriptors with count
            for descriptor in receptor_descriptors.column_names: # type: ignore
                if "count" in descriptor and np.isnan(receptor_descriptors[descriptor].values[0]): # type: ignore
                    # If any count descriptor is NaN then set it to 0
                    receptor_descriptors[descriptor].values[0] = 0 # type: ignore
        else:
            receptor_descriptors = None
            _ = errors.file_do_not_exist(f"The file '{receptor_descriptor_path}' does not exist!")
    except IOError as e:
        if e.errno == errno.EPIPE:
            _ = errors.broken_pipe(message=f"Found a broken PIPE error while reading the file '{receptor_descriptor_path}': {e}")

    try:
        # Check if there is the ligand json, if yes, load it
        if os.path.isfile(ligand_descriptor_path):
            ligand_descriptors = ocl.read_descriptors_from_json(ligand_descriptor_path, returnVaex = True)
        else:
            ligand_descriptors = None
            _ = errors.file_do_not_exist(f"The file '{ligand_descriptor_path}' does not exist!")
    except IOError as e:
        if e.errno == errno.EPIPE:
            _ = errors.broken_pipe(message=f"Found a broken PIPE error while reading the file '{ligand_descriptor_path}': {e}")

    # Initiate the dataframe
    df = vaex.from_dict({ "Protein": [ptn] })

    # If the receptor descriptor is not empty
    if receptor_descriptors: # type: ignore
        # Merge the receptor descriptors
        df = df.join(receptor_descriptors) # type: ignore
    
    # If the ligand descriptor is not empty
    if ligand_descriptors: # type: ignore
        # Merge the ligand descriptors
        df = df.join(ligand_descriptors) # type: ignore

    # Return the single row dataframe
    return df

def __thread_merge_descriptors_in_dataframe_parallel(arguments: Tuple[Tuple[str, str], str]) -> vdf.DataFrameLocal:
    '''Thread aid function to call __core_merge_descriptors_in_dataframe.

    Parameters
    ----------
    arguments : Tuple[Tuple[str, str], str]
        Tuple containing the directory where the files are stored and the receptor descriptor json file and the archive type.
    
    Returns
    -------
    vdf.DataFrameLocal
        Dataframe with the descriptors of the protein.

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_merge_descriptors_in_dataframe(arguments[0])

def __merge_descriptors_in_dataframe_parallel(dirs: List[Tuple[str, str]], desc: str) -> vdf.DataFrameLocal:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    dirs : List[Tuple[str, str]]
        Tuple containing the directory where the files are stored and the receptor descriptor json file.
    desc : str
        Description of the process.

    Returns
    -------
    vdf.DataFrameLocal
        Dataframe with the descriptors of the proteins.

    Raises
    ------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []

    # For each file in the glob
    for d in dirs:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((d, None))

    # List with all protein data
    ptnList = []

    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for innerData in tqdm(p.imap_unordered(__thread_merge_descriptors_in_dataframe_parallel, arguments), total = len(arguments), desc = desc):
                # Update the dict with the result from the called function
                ptnList.append(innerData)
                # Clear the memory
                gc.collect()
    except IOError as e:
        ocprint.print_error_log(f"Problem while mergin descriptors in parallel. Exception: {e}", f"{logdir}/read_log_ERROR_report.log")
        ocprint.print_error(f"Problem while mergin descriptors in parallel. Exception: {e}")

    return vaex.concat(ptnList) # type: ignore

def __merge_descriptors_in_dataframe_no_parallel(dirs: List[Tuple[str, str]], desc: str) -> vdf.DataFrameLocal:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_read_log function.

    Parameters
    ----------
    dirs : List[Tuple[str, str]]
        Tuple containing the directory where the files are stored and the receptor descriptor json file.
    desc : str
        Description of the process.

    Returns
    -------
    vdf.DataFrameLocal
        Dataframe with the descriptors of the proteins.

    Raises
    ------
    None
    '''

    # List to store the read data
    ptnList = []

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        for dir in tqdm(iterable = dirs, total = len(dirs), desc = desc):
            # Call the core read log function (shared between parallel and not parallel) and store the data into the DataFrame
            ptnList.append(__core_merge_descriptors_in_dataframe(dir))
            # Clear the memory
            gc.collect()

    return vaex.concat(ptnList) # type: ignore

def __merge_descriptors_in_dataframe_single(path: Tuple[str, str]) -> vdf.DataFrameLocal:
    '''Warper to prepare the jobs, recieves a directory, and pass it to the __core_prepare function.

    TODO: Add the support to custom databases.

    Parameters
    ----------
    path : Tuple[str, str]
        A tuple with the directory and the ligand type (ligand, decoy, candidate).

    Returns
    -------
    vdf.DataFrameLocal
        Dataframe with the descriptors of the proteins.

    Raises
    ------
    None
    '''

    # Read the log
    return __core_merge_descriptors_in_dataframe(path)

## Public ##

def merge_descriptors_in_dataframe(paths: Union[List[Tuple[str, str]], List[Tuple[str, str]]], archive: str) -> vdf.DataFrameLocal:
    '''Merge the descriptors with the result for the log files.

    Parameters
    ----------
    paths : List[Tuple[str, str]] | Tuple[str, str]
        The list of tuples or a single tuple containing the directories and the receptor descriptor path.
    archive : str
        The archive name. Options are [dudez, pdbbind].

    Returns
    -------
    vdf.DataFrameLocal
        Dataframe with the descriptors of the proteins.
    '''

    # If the path is a list
    if isinstance(paths, list):
        # If logfile exists, backup it
        oclogging.backup_log("read_log_ERROR_report")

        # Set the label
        label = f"Processing {archive}"

        # Check if multiprocessing is enabled
        if args.multiprocess:
            # Prepare the pdbbind
            return __merge_descriptors_in_dataframe_parallel(paths, label)
        else:
            # Prepare the database
            return __merge_descriptors_in_dataframe_no_parallel(paths, label)
    else:
        return __merge_descriptors_in_dataframe_single(paths)
