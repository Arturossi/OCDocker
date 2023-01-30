#!/usr/lib/python3

# Description
###############################################################################
'''
This module is responsible for digest processing.

It is imported as:

import OCDocker.Processing.Postprocessing.ReadLogs as ocreadlogs
'''

# Imports
###############################################################################
import gc
import os
import time
import vaex

import numpy as np
import vaex.dataframe as vdf

from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List, Tuple, TypeVar, Union

from OCDocker.Initialise import *

import OCDocker.Docking.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.Printing as ocprint

# Typevars
###############################################################################
TvinaData = TypeVar('TvinaData', bound='vinaData')
TgninaData = TypeVar('TgninaData', bound='gninaData')
TsminaData = TypeVar('TsminaData', bound='sminaData')
TplantsData = TypeVar('TplantsData', bound='plantsData')

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
class vinaData:
    '''Class to hold the data of the Vina log files.'''

    def __init__(self, vina_pose: List[float] = [], vina_affinity: List[float] = [], withNaN: bool = False) -> None:
        '''Initializes the class.

        Parameters
        ----------
        vina_pose : List[float], optional
            A list of the poses of the ligand in the Vina log files. (default is [])
        vina_affinity : List[float], optional
            A list of the affinities of the ligand in the Vina log files. (default is [])
        withNaN : bool, optional
            If True, the class will be initialized with np.NaN values. (default is False)

        Returns
        -------
        None

        Raises
        ------
        None
        '''
        
        if withNaN:
            self.empty_with_nan()
        else:
            self.vina_pose = vina_pose
            self.vina_affinity = vina_affinity
    
        return None

    def __to_dict__(self) -> Dict[str, List[float]]:
        '''Returns a dict of the class.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, List[float]]
            A dict of the class.

        Raises
        ------
        None
        '''

        return { "vina_pose": self.vina_pose, "vina_affinity": self.vina_affinity }

    def empty_with_nan(self: TvinaData) -> TvinaData:
        ''' Empties the class using the np.NaN value.

        Parameters
        ----------
        None

        Returns
        -------
        TvinaData
            The class with the np.NaN values.

        Raises
        ------
        None
        '''

        self.vina_pose = [np.NaN]
        self.vina_affinity = [np.NaN]

        return self

class gninaData:
    '''Class to hold the data of the Gnina log files.'''

    def __init__(self, gnina_pose: List[float] = [], gnina_affinity: List[float] = [], withNaN: bool = False) -> None:
        '''Initializes the class.

        Parameters
        ----------
        gnina_pose : List[float], optional
            A list of the poses of the ligand in the Gnina log files. (default is [])
        gnina_affinity : List[float], optional
            A list of the affinities of the ligand in the Gnina log files. (default is [])
        withNaN : bool, optional
            If True, the class will be initialized with np.NaN values. (default is False)

        Returns
        -------
        None

        Raises
        ------
        None
        '''
        
        if withNaN:
            self.empty_with_nan()
        else:
            self.gnina_pose = gnina_pose
            self.gnina_affinity = gnina_affinity
    
        return None

    def __to_dict__(self) -> Dict[str, List[float]]:
        '''Returns a dict of the class.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, List[float]]
            A dict of the class.

        Raises
        ------
        None
        '''

        return { "gnina_pose": self.gnina_pose, "gnina_affinity": self.gnina_affinity }

    def empty_with_nan(self: TgninaData) -> TgninaData:
        ''' Empties the class using the np.NaN value.

        Parameters
        ----------
        None

        Returns
        -------
        TgninaData
            The class with the np.NaN values.

        Raises
        ------
        None
        '''

        self.gnina_pose = [np.NaN]
        self.gnina_affinity = [np.NaN]

        return self

class sminaData:
    '''Class to hold the data of the Smina log files.'''

    def __init__(self, smina_pose: List[float] = [], smina_affinity: List[float] = [], withNaN: bool = False) -> None:
        '''Initializes the class.

        Parameters
        ----------
        smina_pose : List[float], optional
            A list of the poses of the ligand in the Smina log files. (default is [])
        smina_affinity : List[float], optional
            A list of the affinities of the ligand in the Smina log files. (default is [])
        withNaN : bool, optional
            If True, the class will be initialized with np.NaN values. (default is False)

        Returns
        -------
        None

        Raises
        ------
        None
        '''
        
        if withNaN:
            self.empty_with_nan()
        else:
            self.smina_pose = smina_pose
            self.smina_affinity = smina_affinity
    
        return None

    def __to_dict__(self) -> Dict[str, List[float]]:
        '''Returns a dict of the class.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, List[float]]
            A dict of the class.

        Raises
        ------
        None
        '''

        return { "smina_pose": self.smina_pose, "smina_affinity": self.smina_affinity }

    def empty_with_nan(self: TsminaData) -> TsminaData:
        ''' Empties the class using the np.NaN value.

        Parameters
        ----------
        None

        Returns
        -------
        TsminaData
            The class with the np.NaN values.

        Raises
        ------
        None
        '''

        self.smina_pose = [np.NaN]
        self.smina_affinity = [np.NaN]

        return self

class plantsData:
    '''Class to hold the data of the PLANTS log files.'''

    def __init__(self, PLANTS_TOTAL_SCORE: List[float] = [], PLANTS_SCORE_RB_PEN: List[float] = [], PLANTS_SCORE_NORM_HEVATOMS: List[float] = [], PLANTS_SCORE_NORM_CRT_HEVATOMS: List[float] = [], PLANTS_SCORE_NORM_WEIGHT: List[float] = [], PLANTS_SCORE_NORM_CRT_WEIGHT: List[float] = [], PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS: List[float] = [], withNaN: bool = False) -> None:
        '''Initializes the class.

        Parameters
        ----------
        PLANTS_TOTAL_SCORE : List[float], optional
            A list of the total scores of the ligand in the PLANTS log files. (default is [])
        PLANTS_SCORE_RB_PEN : List[float], optional
            A list of the rigid body penalty scores of the ligand in the PLANTS log files. (default is [])
        PLANTS_SCORE_NORM_HEVATOMS : List[float], optional
            A list of the normalized heavy atoms scores of the ligand in the PLANTS log files. (default is [])
        PLANTS_SCORE_NORM_CRT_HEVATOMS : List[float], optional
            A list of the normalized critical heavy atoms scores of the ligand in the PLANTS log files. (default is [])
        PLANTS_SCORE_NORM_WEIGHT : List[float], optional
            A list of the normalized weight scores of the ligand in the PLANTS log files. (default is [])
        PLANTS_SCORE_NORM_CRT_WEIGHT : List[float], optional
            A list of the normalized critical weight scores of the ligand in the PLANTS log files. (default is [])
        PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS : List[float], optional
            A list of the rigid body penalty normalized critical heavy atoms scores of the ligand in the PLANTS log files. (default is [])
        withNaN : bool, optional
            If True, the class will be initialized with np.NaN values. (default is False)

        Returns
        -------
        None

        Raises
        ------
        None
        '''
        
        if withNaN:
            self.empty_with_nan()
        else:
            self.PLANTS_TOTAL_SCORE = PLANTS_TOTAL_SCORE
            self.PLANTS_SCORE_RB_PEN = PLANTS_SCORE_RB_PEN
            self.PLANTS_SCORE_NORM_HEVATOMS = PLANTS_SCORE_NORM_HEVATOMS
            self.PLANTS_SCORE_NORM_CRT_HEVATOMS = PLANTS_SCORE_NORM_CRT_HEVATOMS
            self.PLANTS_SCORE_NORM_WEIGHT = PLANTS_SCORE_NORM_WEIGHT
            self.PLANTS_SCORE_NORM_CRT_WEIGHT = PLANTS_SCORE_NORM_CRT_WEIGHT
            self.PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS = PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS
        
        return None
    
    def __to_dict__(self) -> Dict[str, List[float]]:
        '''Returns a dict of the class.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, List[float]]
            A dict of the class.

        Raises
        ------
        None
        '''

        return { "PLANTS_TOTAL_SCORE": self.PLANTS_TOTAL_SCORE, "PLANTS_SCORE_RB_PEN": self.PLANTS_SCORE_RB_PEN, "PLANTS_SCORE_NORM_HEVATOMS": self.PLANTS_SCORE_NORM_HEVATOMS, "PLANTS_SCORE_NORM_CRT_HEVATOMS": self.PLANTS_SCORE_NORM_CRT_HEVATOMS, "PLANTS_SCORE_NORM_WEIGHT": self.PLANTS_SCORE_NORM_WEIGHT, "PLANTS_SCORE_NORM_CRT_WEIGHT": self.PLANTS_SCORE_NORM_CRT_WEIGHT, "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": self.PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS }

    def empty_with_nan(self: TplantsData) -> TplantsData:
        ''' Empties the class using the np.NaN value.

        Parameters
        ----------
        None

        Returns
        -------
        TplantsData
            The class with the np.NaN values.

        Raises
        ------
        None
        '''

        self.PLANTS_TOTAL_SCORE = [np.NaN]
        self.PLANTS_SCORE_RB_PEN = [np.NaN]
        self.PLANTS_SCORE_NORM_HEVATOMS = [np.NaN]
        self.PLANTS_SCORE_NORM_CRT_HEVATOMS = [np.NaN]
        self.PLANTS_SCORE_NORM_WEIGHT = [np.NaN]
        self.PLANTS_SCORE_NORM_CRT_WEIGHT = [np.NaN]
        self.PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS = [np.NaN]

        return self

# Functions
###############################################################################
## Private ##

def __core_read_log(processDirData: Tuple[str, str]) -> Dict[str, vdf.DataFrameLocal]:
    '''Reads Vina, Smina and PLANTS logs and then return a dict of dataframes.

    Parameters
    ----------
    processDirData : Tuple[str, str]
        A tuple with the directory and the ligand name.

    Returns
    -------
     Dict[str, vdf.DataFrameLocal]
        A dictionary containing the dataframes of the logs.
        A dict of dicts of dataframes. Each element of the first dict is a complex protein-ligand, and each element of the second dict is a docking algorithm results.

    Raises
    ------
    None
    '''

    # Unpack the tuple
    processDir, tp = processDirData

    # Get protein and ligand names
    processSplited = processDir.split(os.path.sep)
    ptn = processSplited[-4]
    lgd = processSplited[-1]

    # Create docking objects
    dt = { "vina": vinaData(), "gnina": gninaData(), "smina": sminaData(), "plants": plantsData() }

    # Get run number
    runNumber = 0 # TODO: Add support to multiple runs

    # Dict to hold the protein data
    proteinData = { f"{ptn}-{lgd}": None }

    # Create the maxLenList
    maxLen = 1

    # For each key, value in dt
    for key, value in dt.items():
        # Get the log path
        if key in ["smina"]:
            logPath = f"{processDir}/{key}Files/{key}.log"
        elif key in ["plants"]:
            logPath = f"{processDir}/{key}Files/run/bestranking.csv"
        else:
            logPath = f"{processDir}/{key}Files/{key}_{runNumber}.log"
        # Check if exists
        if os.path.isfile(logPath):
            gendict = {}
            if key == "vina":
                # Read the log into dict
                gendict = ocvina.read_log(logPath)
            elif key == "gnina":
                # Read the log into dict
                gendict = ocgnina.read_log(logPath)
            elif key == "smina":
                # Read the log into dict
                gendict = ocsmina.read_log(logPath)
            elif key == "plants":
                # Read the log into dict
                gendict = ocplants.read_log(logPath)

            # First loop iteration flag
            first = True

            # For each key, value in vinaDict
            for key2, value2 in gendict.items():
                # Set it back in the class
                setattr(dt[key], key2, [value2[runNumber]])
                if first:
                    # Set the maxLen as the biggest size among the current maxLen and the len of the first class attribute
                    maxLen = max(maxLen, len([value2[runNumber]]))
                    # Turn the flag off
                    first = False
        else:
            _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its {key} output.")
            # Set the elements in value as np.NaN
            value = value.empty_with_nan()

    # Add the concatenated the dicts. The single elements are repeated to match the largest dict to the proteinData dict using ptn as the key
    proteinData = vaex.from_dict(
        {
            **{
                "Protein": [ptn for _ in range(maxLen)],
                "Ligand": [lgd for _ in range(maxLen)],
                "type": [tp for _ in range(maxLen)],
                "Complex": [f"{ptn}-{lgd}" for _ in range(maxLen)]
            },
            **dt["vina"].__to_dict__(),
            **dt["smina"].__to_dict__(),
            **dt["gnina"].__to_dict__(),
            **dt["plants"].__to_dict__()
        }
    )

    # Clean the memory
    del dt

    # Return the proteinData dict
    return proteinData # type: ignore

def __thread_read_log_parallel(arguments: Tuple[Tuple[str, str]]) -> Dict[str, vdf.DataFrameLocal]:
    '''Thread aid function to call __core_read_log.

    Parameters
    ----------
    arguments : Tuple[Tuple[str, str]]
        A tuple with the directory and the ligand type (ligand, decoy, candidate).

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary containing the dataframes of the logs.

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_read_log(arguments[0])

def __read_log_parallel(paths: List[Tuple[str, str]], desc: str, ptn: str, saveChunk: int, hdf5Path: str, overwrite: bool) -> Dict[str, vdf.DataFrameLocal]:
    '''Read the logs of the ligands in parallel.

    Parameters
    ----------
    paths : List[Tuple[str, str]]
        A list of tuples with the directory and the ligand type (ligand, decoy, candidate).
    desc : str
        The description to be used in the tqdm progress bar.
    ptn : str
        The protein name.
    saveChunk : int
        The number of iterations to perform before saving the data.
    hdf5Path : str
        The path to the hdf5 file.
    overwrite : bool
        A flag to indicate if the data should be overwritten.

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary containing the dataframes of the logs.

    Raises
    ------
    '''

    # Check if the hdf5 file exists or if the overwrite flag is True
    if os.path.isfile(hdf5Path) or overwrite:
        # Load the hdf5 file
        dockingResults = vaex.open(hdf5Path)
    else:
        # Set the dockingResults as None
        dockingResults = None

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []

    # Counter for the iterations
    i = 0

    # For each file in the glob
    for path in paths:
        # Get protein and ligand names
        pathSplited = path[0].split(os.path.sep)
        lgd = pathSplited[-1]

        # Create the key
        key = f"{ptn}-{lgd}"

        # Check if the dockingResults is not None
        if dockingResults is not None:
            # Check if the key is not in the Complex coulmn
            if key not in dockingResults["Complex"].values.to_pylist():
                # Append a tuple containing the file name and ovewrite flag to the arguments list
                arguments.append((path, None))
        else:
            # Append a tuple containing the file name and ovewrite flag to the arguments list
            arguments.append((path, None))

    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for data in tqdm(p.imap_unordered(__thread_read_log_parallel, arguments), total = len(arguments), desc = desc):
                # Check if the dockingResults is None
                if dockingResults is None:
                    # Set the dockingResults as the data
                    dockingResults = data
                else:
                    # Concatenate the data
                    dockingResults = vaex.concat([dockingResults, data])
                # Clean the memory
                del data
                # Increment the counter
                i += 1
                # Check if the counter is greater or equal of saveChunk
                if i >= saveChunk:
                    # Save the dockingResults
                    dockingResults.export_hdf5(hdf5Path) # type: ignore
                    # Reset the counter
                    i = 0
                gc.collect()
            if i > 0:
                # Save the data
                dockingResults.export_hdf5(hdf5Path) # type: ignore
                
    except IOError as e:
        errMsg = f"Problem while reading logs in parallel. Exception: {e}"
        ocprint.print_error_log(errMsg, f"{logdir}/read_log_ERROR_report.log")
        ocprint.print_error(errMsg)

    # Sleep for 0.33 second
    time.sleep(0.33)
    return dockingResults # type: ignore

def __read_log_no_parallel(paths: List[Tuple[str, str]], desc: str, ptn: str, saveChunk: int, hdf5Path: str, overwrite: bool) -> Dict[str, vdf.DataFrameLocal]:
    '''Read the logs of the docking results for the ligands in serial.

    Parameters
    ----------
    paths : List[Tuple[str, str]]
        A list of tuples with the directory and the ligand type (ligand, decoy, candidate).
    desc : str
        The description to be used in the tqdm progress bar.
    ptn : str
        The protein name.
    saveChunk : int
        The number of iterations to perform before saving the data.
    hdf5Path : str
        The path to the hdf5 file.
    overwrite : bool
        A flag to indicate if the data should be overwritten.

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary with the protein name as the key and a dictionary with the vina, smina and plants dataframes as the value.
    
    Raises
    ------
    None
    '''

    # Check if the hdf5 file exists or if the overwrite flag is True
    if os.path.isfile(hdf5Path) or overwrite:
        # Load the hdf5 file
        dockingResults = vaex.open(hdf5Path)
    else:
        # Set the dockingResults as None
        dockingResults = None

    # Counter for the iterations
    i = 0

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        for path, tp in tqdm(iterable = paths, total = len(paths), desc = desc):
            # Get protein and ligand names
            pathSplited = path.split(os.path.sep)
            lgd = pathSplited[-1]

            # Create the key
            key = f"{ptn}-{lgd}"

            # Check if the key from data is in the hdf5 file or if overwrite is True
            if dockingResults is None or key not in dockingResults["Complex"].values.to_pylist() or overwrite: # type: ignore
                # Add 1 to the counter
                i += 1
                # Check if the dockingResults is None
                if dockingResults is None:
                    # Set the dockingResults as the data
                    dockingResults = __core_read_log((path, tp))
                else:
                    # Concatenate the data
                    dockingResults = vaex.concat([dockingResults, __core_read_log((path, tp))]) 
                # Check if the counter is greater or equal of saveChunk
                if i >= saveChunk:
                    # Save the data
                    dockingResults.export_hdf5(hdf5Path) # type: ignore
                    # Reset the counter
                    i = 0
            if i > 0:
                # Save the data
                dockingResults.export_hdf5(hdf5Path) # type: ignore

            # Clear the memory
            gc.collect()
    return dockingResults # type: ignore

def __read_log_single(path: Tuple[str, str], ptn: str, hdf5Path: str, overwrite: bool) -> Dict[str, vdf.DataFrameLocal]:
    '''Warper to prepare the jobs, recieves a directory, and pass it to the __core_read_log function.

    TODO: Add the support to custom databases.

    Parameters
    ----------
    path : Tuple[str, str]
        A tuple with the directory and the ligand type (ligand, decoy, candidate).
    ptn : str
        The protein name.
    hdf5Path : str
        The path to the hdf5 file.
    overwrite : bool
        If True, the hdf5 file will be overwritten.

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary with the protein name as the key and a dictionary with the vina, smina and plants dataframes as the value.

    Raises
    ------
    None
    '''

    # Split the tuple
    processPath, tp = path

    # Parameterize the hdf5 path (This routine should be called for the same receptor) removing the last three directories of the path
    pathSplited = processPath.split(os.path.sep)
    ptn = pathSplited[-4]
    lgd = pathSplited[-1]

    # Load the hdf5 file
    dockingResults = ocff.from_hdf5(hdf5Path)

    # Create the key
    key = f"{ptn}-{lgd}"

    # Check if the dockingResults is not None
    if dockingResults is not None:
        # Check if the key from data is in the hdf5 file or if overwrite is True
        if key not in list(dockingResults.keys()) or overwrite:
            # Read the log (only read if is necessary)
            data = __core_read_log(path)

            # Update the dockingResults with the data
            dockingResults[key] = data[key]
            # Save the dockingResults to the hdf5 file
            ocff.to_hdf5(dockingResults, hdf5Path)

            # Read the log
            return data
        # Return the already computed data
        return dockingResults[key]
    
    return get_vaex_empty_log_data(ptn, lgd, tp)

## Public ##

def get_vaex_empty_log_data(ptn: str, lgd: str, tp: str) -> Dict[str, vdf.DataFrameLocal]:
    '''Get an empty vaex dataframe with the columns of the data.

    Parameters
    ----------
    ptn : str
        The protein name.
    lgd : str
        The ligand name.
    tp : str
        The ligand type (ligand, decoy, candidate).

    Returns
    -------
    vdf.DataFrameLocal
        A dictionary with the protein name as the key and a dictionary with the vina, smina and plants dataframes as the value.
    '''

    # Instantiate all the classes with np.NaN
    vina = vinaData(withNaN = True)
    smina = sminaData(withNaN = True)
    gnina = gninaData(withNaN = True)
    plants = plantsData(withNaN = True)

    proteinData = {}

    proteinData[f"{ptn}-{lgd}"] = vaex.from_dict(
        {
            **{
                "Protein": [ptn],
                "Ligand": [lgd],
                "type": [tp]
            },
            **vina.__to_dict__(),
            **smina.__to_dict__(),
            **gnina.__to_dict__(),
            **plants.__to_dict__()
        }
    )
    
    return proteinData

def read_logs(paths: Union[List[Tuple[str, str]], List[Tuple[str, str]]], archive: str, ptn: str, saveChunk: int = 100, overwrite: bool = False) -> Dict[str, vdf.DataFrameLocal]:
    '''Read the logs of the docking results for the ligands. 
    
    IMPORTANT: If passing a list, ensure that all the paths are related to the same receptor.

    Parameters
    ----------
    paths : Tuple[str] | str
        The list of directories or the directory to be processed.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    ptn : str
        The protein name.
    saveChunk : int, optional
        The number of lines to be read before saving the data. The default is 100. (Not applicable if the paths is not a list!)
    overwrite : bool, optional
        If True overwrites the files, if False does not overwrite the files. The default is False.
    
    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary with the protein name as the key and a dictionary with the vina, smina and plants dataframes as the value.

    '''

    # If the path is a list
    if isinstance(paths, list):

        # If logfile exists, backup it
        oclogging.backup_log("read_log_ERROR_report")

        # Set the label
        label = f"Processing {archive}: {ptn}"

        # Get the first path of paths and then remove the last 3 directories
        hdf5Path = f"{os.path.dirname(os.path.dirname(os.path.dirname(paths[0][0])))}/{ptn}_docking_results.hdf5"

        # Check if multiprocessing is enabled
        if args.multiprocess:
            # Prepare the pdbbind
            return __read_log_parallel(paths, label, ptn, saveChunk, hdf5Path, overwrite)
        else:
            # Prepare the database
            return __read_log_no_parallel(paths, label, ptn, saveChunk, hdf5Path, overwrite)
    else:
        # Get the first path of paths and then remove the last 3 directories
        hdf5Path = os.path.dirname(os.path.dirname(os.path.dirname(paths[0])))

        return __read_log_single(paths, ptn, hdf5Path, overwrite)
