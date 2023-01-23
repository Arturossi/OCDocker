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
import OCDocker.Toolbox.FilesFolders as ocff
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

def __merge_descriptors_in_dataframe_parallel(paths: List[Tuple[str, str]], receptorDataFile: str, ptn: str, saveChunk: int, desc: str, datafileFormat: str = "hdf5", overwrite: bool = False) -> vdf.DataFrameLocal:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    paths : List[Tuple[str, str]]
        Tuple containing the directory where the files are stored and the receptor descriptor json file.
    receptorDataFile : str
        Path to the receptor data file.
    ptn : str
        Protein name.
    saveChunk : int
        The number of iterations to perform before saving the data.
    desc : str
        Description of the process.
    datafileFormat : str, optional
        Format of the datafile, by default "hdf5". TODO: Add support for csv.
    overwrite : bool, optional
        If True, then it will overwrite the datafile, by default False.

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

    # Check the datafile format
    df = __check_datafile_format(datafileFormat, receptorDataFile)

    if df is None:
        errMsg = f"Problem while reading the receptor file data '{receptorDataFile}'."
        # Log the error
        ocprint.print_error(errMsg)
        ocprint.print_error_log(errMsg, f"{logdir}/merge_log_ERROR_report.log")
        return vaex.from_dict({})

    # For each file in the glob
    for path in tqdm(iterable = paths, total = len(paths), desc = f"Pre processing protein '{ptn}' paths."):
        # Get protein and ligand names
        pathSplited = path[0].split(os.path.sep)
        lgd = pathSplited[-1]

        # Found flag
        found = False

        # Check if the file exists
        if type(df) != int:
            # For each line in the vaex dataframe
            for i in range(len(df)): # type: ignore
                # Check if the protein and ligand are the same as the ones in the datafile
                if df["Protein"].values[i].as_py() == ptn and df["Ligand"].values[i].as_py() == lgd: # type: ignore
                    found = True
                    break

        if not found or overwrite:
            # Append a tuple containing the file name and ovewrite flag to the arguments list
            arguments.append((path, None))
        
    # List with all protein data
    ptnList = []

    # Counter for the iterations
    i = 0

    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for innerData in tqdm(p.imap_unordered(__thread_merge_descriptors_in_dataframe_parallel, arguments), total = len(arguments), desc = desc):
                # Update the dict with the result from the called function
                ptnList.append(innerData)# Increment the counter
                i += 1
                # Check if the counter is greater or equal of saveChunk
                if i >= saveChunk:
                    # Convert the ptnList to a dataframe
                    df = vaex.concat(ptnList)

                    # Save it
                    df.export_hdf5(receptorDataFile)

                    # Reset the counter
                    i = 0

                # Clear the memory
                gc.collect()
            # If on finish and i is not 0
            if i != 0:
                # Convert the ptnList to a dataframe
                innerdf = vaex.concat(ptnList)

                # Save the data
                innerdf.export_hdf5(receptorDataFile)
    except IOError as e:
        errMsg = f"Problem while merging descriptors in parallel. Exception: {e}"
        ocprint.print_error_log(errMsg, f"{logdir}/read_log_ERROR_report.log")
        ocprint.print_error(errMsg)

    return vaex.concat(ptnList) # type: ignore

def __merge_descriptors_in_dataframe_no_parallel(paths: List[Tuple[str, str]], receptorDataFile: str, ptn: str, saveChunk: int, desc: str, datafileFormat: str = "hdf5", overwrite: bool = False) -> vdf.DataFrameLocal:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_read_log function.

    Parameters
    ----------
    paths : List[Tuple[str, str]]
        Tuple containing the directory where the files are stored and the receptor descriptor json file.
    receptorDataFile : str
        Path to the receptor data file.
    ptn : str
        Protein name.
    saveChunk : int
        The number of iterations to perform before saving the data.
    desc : str
        Description of the process.
    datafileFormat : str, optional
        Format of the datafile, by default "hdf5".
    overwrite : bool, optional
        If True, then it will overwrite the datafile, by default False.

    Returns
    -------
    vdf.DataFrameLocal
        Dataframe with the descriptors of the proteins.

    Raises
    ------
    None
    '''

    # Check the datafile format
    df = __check_datafile_format(datafileFormat, receptorDataFile)

    # If df is None
    if df is None:
        errMsg = f"Problem while reading the receptor file data '{receptorDataFile}'."
        # Log the error
        ocprint.print_error(errMsg)
        ocprint.print_error_log(errMsg, f"{logdir}/merge_log_ERROR_report.log")
        # Return an empty vaex dict
        return vaex.from_dict({})

    # List to store the read data
    ptnList = []

    # Counter for the iterations
    i = 0

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        for path in tqdm(iterable = paths, total = len(paths), desc = desc):
            # Get the ligand type
            pathSplited = path[0].split(os.path.sep)
            lgd = pathSplited[-1]
            
            # Found flag
            found = False

            # Check if the file exists
            if type(df) != int:
                # For each line in the vaex dataframe
                for i in range(len(df)): # type: ignore
                    # Check if the protein and ligand are the same as the ones in the datafile
                    if df["Protein"].values[i].as_py() == ptn and df["Ligand"].values[i].as_py() == lgd: # type: ignore
                        found = True
                        break

            if not found or overwrite:
                # Add 1 to the counter
                i += 1
                # Call the core read log function (shared between parallel and not parallel) and store the data into the DataFrame
                ptnList.append(__core_merge_descriptors_in_dataframe(path))
                # Check if the counter is greater or equal of saveChunk
                if i >= saveChunk:
                    # Convert the ptnList to a dataframe
                    innerdf = vaex.concat(ptnList)

                    # Save the data
                    innerdf.export_hdf5(receptorDataFile)

                    # Reset the counter
                    i = 0
            
            # Clear the memory
            gc.collect()
        # If on finish and i is not 0
        if i != 0:
            # Convert the ptnList to a dataframe
            innerdf = vaex.concat(ptnList)

            # Save the data
            innerdf.export_hdf5(receptorDataFile)

    return vaex.concat(ptnList) # type: ignore

def __merge_descriptors_in_dataframe_single(path: Tuple[str, str], receptorDataFile: str, ptn: str, saveChunk: int, datafileFormat: str = "hdf5", overwrite: bool = False, savedf: bool = False) -> vdf.DataFrameLocal:
    '''Warper to prepare the jobs, recieves a directory, and pass it to the __core_prepare function.

    TODO: Add the support to custom databases.

    Parameters
    ----------
    path : Tuple[str, str]
        A tuple with the directory and the ligand type (ligand, decoy, candidate).
    receptorDataFile : str
        Path to the receptor data file.
    ptn : str
        Protein name.
    saveChunk : int
        The number of iterations to perform before saving the data.
    datafileFormat : str, optional
        Format of the datafile, by default "hdf5".
    overwrite : bool, optional
        If True, then it will overwrite the datafile, by default False.
    savedf : bool, optional
        If True, then it will save the dataframe, by default False.

    Returns
    -------
    vdf.DataFrameLocal
        Dataframe with the descriptors of the proteins.

    Raises
    ------
    None
    '''

    # Check the datafile format
    df = __check_datafile_format(datafileFormat, receptorDataFile)

    # If df is None
    if df is None:
        errMsg = f"Problem while reading the receptor file data '{receptorDataFile}'."
        # Log the error
        ocprint.print_error(errMsg)
        ocprint.print_error_log(errMsg, f"{logdir}/merge_log_ERROR_report.log")
        # Return an empty vaex dict
        return vaex.from_dict({})
    
    # Get the ligand type
    pathSplited = path[0].split(os.path.sep)
    lgd = pathSplited[-1]

    # Found flag
    found = False
    
    # Check if the file exists
    if type(df) != int:
        # For each line in the vaex dataframe
        for i in range(len(df)): # type: ignore
            # Check if the protein and ligand are the same as the ones in the datafile
            if df["Protein"].values[i].as_py() == ptn and df["Ligand"].values[i].as_py() == lgd: # type: ignore
                found = True
                break
    else:
        # If the file does not exist, then the empty dataframe that will be concatenated
        df = vaex.from_dict({})

    if not found or overwrite:
        # Call the core read log function and concatenate its results to the vaex dataframe
        return vaex.concat([df, __core_merge_descriptors_in_dataframe(path)]) # type: ignore
    else:
        # Update the data with information from the hdf5 file
        return df # type: ignore

def __check_datafile_format(datafileFormat: str, receptorDataFile: str) -> Union[vdf.DataFrameLocal, int, None]:
    '''Check if the datafile format is supported. If the file does not exist and has a valid extension, then it will create it.

    Parameters
    ----------
    datafileFormat : str
        Format of the datafile.

    Returns
    -------
    vdf.DataFrameLocal | int | None
        Dataframe with the descriptors of the proteins. If no file exists, will return an int (errors.fileDoNotExistCode). If unsupported, then it will return None.

    Raises
    ------
    ValueError
        If the datafile format is not supported.
    '''

    # Check the datafile format
    if datafileFormat.lower() in ["hdf5"]:
        # If the file does not exist, return an empty dataframe
        if not os.path.isfile(receptorDataFile):
            return errors.file_do_not_exist(f"File '{receptorDataFile}' does not exist.", level = "warn")
        # Read the hdf5 file
        return vaex.open(receptorDataFile)
    else:   	
        _ = errors.unsupported_extension(f"Unsupported datafile format: {datafileFormat}. Supported formats are: [hdf5].")
        return None

## Public ##

def merge_descriptors_in_dataframe(paths: Union[List[Tuple[str, str]], Tuple[str, str]], receptorDataFile: str, ptn: str, archive: str, saveChunk: int = 100, datafileFormat: str = "hdf5", overwrite: bool = False, savedf: bool = False) -> vdf.DataFrameLocal:
    '''Merge the descriptors with the result for the log files.

    Parameters
    ----------
    paths : List[Tuple[str, str]] | Tuple[str, str]
        The list of tuples or a single tuple containing the directories and the receptor descriptor path.
    receptorDataFile : str
        Path to the receptor data file.
    ptn : str
        Protein name.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    saveChunk : int, optional
        The number of iterations to perform before saving the data, by default 100.
    datafileFormat : str, optional
        The format of the datafile. Options are [hdf5, csv], by default "hdf5"
    overwrite : bool, optional
        If True, then it will overwrite the datafile, by default False.
    savedf : bool, optional
        If True, then it will save the dataframe, by default False. Only works if paths is not a list of tuples.

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
        label = f"Processing {ptn}"

        # Check if multiprocessing is enabled
        if args.multiprocess:
            # Prepare the pdbbind
            return __merge_descriptors_in_dataframe_parallel(paths, receptorDataFile, ptn, saveChunk , label, datafileFormat = datafileFormat, overwrite = overwrite)
        else:
            # Prepare the database
            return __merge_descriptors_in_dataframe_no_parallel(paths, receptorDataFile, ptn, saveChunk , label, datafileFormat = datafileFormat, overwrite = overwrite)
    else:
        return __merge_descriptors_in_dataframe_single(paths, receptorDataFile, ptn, saveChunk, datafileFormat = datafileFormat, overwrite = overwrite, savedf = savedf)
