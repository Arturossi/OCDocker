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
import json
import os
from glob import glob

from multiprocessing import Pool
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import OCDocker.Docking.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Error as ocerror
import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Validation as ocvalidation

from OCDocker.Config import get_config
from OCDocker.Processing.GarbageCollection import (
    collect_periodically as __collect_periodically,
    gc_collect_interval as __gc_collect_interval,
    pool_chunksize as __pool_chunksize,
)

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

def __load_digest_data(digest_path: str, digest_format: str) -> Union[Dict[str, Any], int]:
    '''Load an existing digest file or initialize an empty digest payload.

    Parameters
    ----------
    digest_path : str
        Path to the digest file.
    digest_format : str
        Digest format identifier (currently ``json``).

    Returns
    -------
    Dict[str, Any] | int
        The loaded/initialized digest mapping, or an error code.
    '''

    if not ocvalidation.validate_digest_extension(digest_path, digest_format):
        return ocerror.Error.unsupported_extension(
            f"The provided extension '{digest_format}' is not supported.",
            level = ocerror.ReportLevel.ERROR,
        )

    if os.path.isfile(digest_path):
        if digest_format == "json":
            try:
                with open(digest_path, "r", encoding = "utf-8") as handle:
                    digest_data = json.load(handle)
            except (OSError, IOError, FileNotFoundError, json.JSONDecodeError):
                return ocerror.Error.file_not_exist(
                    f"Could not read the digest file '{digest_path}'.",
                    level = ocerror.ReportLevel.ERROR,
                )
            if not isinstance(digest_data, dict):
                return ocerror.Error.wrong_type(
                    f"The digest file '{digest_path}' is not valid.",
                    level = ocerror.ReportLevel.ERROR,
                )
            return digest_data

    digest_data = ocff.empty_docking_digest(digest_path, overwrite = True, digestFormat = "")
    if isinstance(digest_data, dict):
        return digest_data
    return ocerror.Error.wrong_type(
        f"The docking digest file '{digest_path}' is not valid.",
        level = ocerror.ReportLevel.ERROR,
    )


def __write_digest_data(digest_path: str, digest_data: Dict[str, Any], digest_format: str) -> int:
    '''Persist digest payload to disk.

    Parameters
    ----------
    digest_path : str
        Path to the digest file.
    digest_data : Dict[str, Any]
        In-memory digest payload to serialize.
    digest_format : str
        Digest format identifier (currently ``json``).

    Returns
    -------
    int
        The exit code of the operation (based on the Error.py code table).
    '''

    if digest_format == "json":
        try:
            with open(digest_path, "w", encoding = "utf-8") as handle:
                json.dump(digest_data, handle)
        except (OSError, IOError, PermissionError):
            return ocerror.Error.write_file(
                f"Could not write the digest file '{digest_path}'.",
                level = ocerror.ReportLevel.ERROR,
            )
    return ocerror.Error.ok()


def __get_digest_target(digest_data: Dict[str, Any], box_id: Optional[str] = None) -> Dict[str, Any]:
    '''Return the target digest section for score updates.

    Parameters
    ----------
    digest_data : Dict[str, Any]
        Root digest payload.
    box_id : str, optional
        Box identifier. If provided, updates are applied inside this box entry.

    Returns
    -------
    Dict[str, Any]
        Mutable target mapping where parsed scores should be merged.
    '''

    if not box_id:
        return digest_data

    box_key = str(box_id)
    box_data = digest_data.get(box_key)
    if not isinstance(box_data, dict):
        box_data = {}
        digest_data[box_key] = box_data
    return box_data


def __merge_pose_entries(target: Dict[str, Any], incoming: Dict[Any, Any]) -> None:
    '''Merge parsed engine scores by pose into a target digest mapping.

    Parameters
    ----------
    target : Dict[str, Any]
        Target digest section to be updated in-place.
    incoming : Dict[Any, Any]
        Parsed engine payload keyed by pose identifier.

    Returns
    -------
    None
        This function updates ``target`` in-place.
    '''

    if not isinstance(incoming, dict):
        return

    for pose_key, pose_values in incoming.items():
        if not isinstance(pose_values, dict):
            continue
        normalized_key = str(pose_key)
        existing_values = target.get(normalized_key)
        if not isinstance(existing_values, dict):
            existing_values = {}
        existing_values.update(pose_values)
        target[normalized_key] = existing_values


def __merge_engine_log(
    target: Dict[str, Any],
    reader: Callable[[str], Dict[Any, Any]],
    log_path: str,
) -> None:
    '''Read one engine log and merge parsed scores into a target section.

    Parameters
    ----------
    target : Dict[str, Any]
        Target digest section to be updated in-place.
    reader : Callable[[str], Dict[Any, Any]]
        Engine log parser function.
    log_path : str
        Log file path to parse.

    Returns
    -------
    None
        This function updates ``target`` in-place.
    '''

    try:
        parsed = reader(log_path)
    except Exception:
        parsed = {}
    if isinstance(parsed, dict):
        __merge_pose_entries(target, parsed)


def __core_generate_digest(path: str, ligandDir: str, archive: str, overwrite: bool, digestFormat: str = "json", all_boxes: bool = False) -> int:
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
        return ocerror.Error.unallowed_dir()

    ligandDescriptorPath = f"{ligandDir}/ligand_descriptors.json"

    # If the complex has descriptor files for ligand
    if os.path.isfile(ligandDescriptorPath):
        digest_path = f"{ligandDir}/dockingDigest.json"
        digest_data_or_error = __load_digest_data(digest_path, digestFormat)
        if not isinstance(digest_data_or_error, dict):
            return int(digest_data_or_error)
        digest_data: Dict[str, Any] = digest_data_or_error

        if all_boxes:
            boxes = sorted(glob(f"{ligandDir}/boxes/box*.pdb"))
            for box_file in boxes:
                box_id = os.path.splitext(os.path.basename(box_file))[0]
                target = __get_digest_target(digest_data, box_id)
                # Run for gnina
                logPath = f"{ligandDir}/gninaFiles/{box_id}/gnina_0.log"
                __merge_engine_log(target, ocgnina.read_log, logPath)
                # Run for vina
                logPath = f"{ligandDir}/vinaFiles/{box_id}/vina_0.log"
                __merge_engine_log(target, ocvina.read_log, logPath)
                # Run for smina
                logPath = _resolve_smina_log(f"{ligandDir}/sminaFiles/{box_id}")
                __merge_engine_log(target, ocsmina.read_log, logPath)
                # Run for PLANTS
                logPath = f"{ligandDir}/plantsFiles/{box_id}/run/bestranking.csv"
                __merge_engine_log(target, ocplants.read_log, logPath)
        else:
            target = __get_digest_target(digest_data)
            # Run for gnina
            logPath = f"{ligandDir}/gninaFiles/gnina_0.log"
            __merge_engine_log(target, ocgnina.read_log, logPath)
            # Run for vina
            logPath = f"{ligandDir}/vinaFiles/vina_0.log"
            __merge_engine_log(target, ocvina.read_log, logPath)
            # Run for smina
            logPath = _resolve_smina_log(f"{ligandDir}/sminaFiles")
            __merge_engine_log(target, ocsmina.read_log, logPath)
            # Run for PLANTS
            logPath = f"{ligandDir}/plantsFiles/run/bestranking.csv"
            __merge_engine_log(target, ocplants.read_log, logPath)

        write_code = __write_digest_data(digest_path, digest_data, digestFormat)
        if write_code != ocerror.ErrorCode.OK:
            return int(write_code)
    else:
        errMsg = f"There is no ligand descriptor json file for the protein in the path '{ligandDescriptorPath}'."
        config = get_config()
        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_docking_digest_run_report_ERROR.log")
        return ocerror.Error.receptor_or_ligand_descriptor_not_exist(errMsg, level = ocerror.ReportLevel.ERROR)

    return ocerror.Error.ok()

def __generate_digest_no_parallel(complexList: List[Tuple[str, List[str]]], archive: str, overwrite: bool, digestFormat: str, desc: str, all_boxes: bool) -> int:
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

    # Track error codes from all digest operations
    error_codes: List[int] = []

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        collect_every = __gc_collect_interval(len(complexList))
        # For each file in dirs
        for i, cl in enumerate(tqdm(iterable = complexList, total = len(complexList), desc=desc), start=1):
            for ligandDir in cl[1]:
                # Call the core dock function (shared between parallel and not parallel)
                return_code = __core_generate_digest(cl[0], ligandDir, archive, overwrite, digestFormat, all_boxes)
                # Track non-zero error codes
                if return_code != ocerror.ErrorCode.OK:
                    error_codes.append(return_code)

            # Large batches benefit from less frequent explicit GC.
            __collect_periodically(i, collect_every, gc.collect)

    gc.collect()

    # Return the most severe error code, or OK if all succeeded
    if error_codes:
        # Return the first non-OK error code (errors are already logged by core functions)
        return int(error_codes[0])
    return ocerror.Error.ok()

def __generate_digest_parallel(complexList: List[Tuple[str, List[str]]], archive: str, overwrite: bool, digestFormat: str, desc: str, all_boxes: bool) -> int:
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

    total_jobs = sum(len(cl[1]) for cl in complexList)
    arguments = (
        (cl[0], ligandDir, archive, overwrite, digestFormat, all_boxes)
        for cl in complexList
        for ligandDir in cl[1]
    )

    # Track error codes from all digest operations
    error_codes: List[int] = []

    try:
        # Create a Thread pool with the maximum available_cores
        config = get_config()
        with Pool(config.available_cores) as p:
            collect_every = __gc_collect_interval(total_jobs)
            chunksize = __pool_chunksize(total_jobs, config.available_cores)
            # Perform the multi process and collect return codes
            for i, return_code in enumerate(tqdm(p.imap_unordered(__thread_generate_digest, arguments, chunksize=chunksize), total = total_jobs, desc = desc), start=1):
                # Track non-zero error codes
                if return_code != ocerror.ErrorCode.OK:
                    error_codes.append(return_code)
                # Large batches benefit from less frequent explicit GC.
                __collect_periodically(i, collect_every, gc.collect)
    except IOError as e:
        errMsg = f"Problem while generating docking digest in parallel. Exception: {e}"
        config = get_config()
        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_docking_report.log")
        return ocerror.Error.docking_failed(errMsg, level = ocerror.ReportLevel.ERROR)

    gc.collect()

    # Return the most severe error code, or OK if all succeeded
    if error_codes:
        # Return the first non-OK error code (errors are already logged by core functions)
        return int(error_codes[0])
    return ocerror.Error.ok()

def __generate_digest_single(complex: Tuple[str, List[str]], archive: str, overwrite: bool, digestFormat: str, desc: str, all_boxes: bool) -> int:
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

    # Track error codes from all digest operations
    error_codes: List[int] = []

    # For each file in dirs
    for ligandDir in tqdm(iterable = complex[1], total = len(complex[1]), desc=desc):
        # Call the core dock function (shared between parallel and not parallel)
        return_code = __core_generate_digest(complex[0], ligandDir, archive, overwrite, digestFormat, all_boxes)
        # Track non-zero error codes
        if return_code != ocerror.ErrorCode.OK:
            error_codes.append(return_code)

        # Clear the memory
        gc.collect()

    # Return the most severe error code, or OK if all succeeded
    if error_codes:
        # Return the first non-OK error code (errors are already logged by core functions)
        return int(error_codes[0])
    return ocerror.Error.ok()

def __thread_generate_digest(arguments: Tuple[str, str, str, bool, str, bool]) -> int:
    '''Thread aid function to call __core_generate_digest.

    Parameters
    ----------
    arguments : Tuple[str, str, str, bool, str, bool]
        The arguments to be passed to __core_generate_digest.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Call the core dock function passing the arguments correctly
        returnState = __core_generate_digest(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5])

    return returnState

def _resolve_smina_log(run_dir: str) -> str:
    '''Resolve smina log path with backward-compatible fallback.'''
    log_path = f"{run_dir}/smina.log"
    if os.path.isfile(log_path):
        return log_path
    return f"{run_dir}/smina_0.log"
## Public ##


def generate_digest(paths: Union[List[Tuple[str, List[str]]], Tuple[str, List[str]]], archive: str, overwrite: bool, digestFormat: str = "json", all_boxes: bool = False) -> None:
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
        config = get_config()
        if config.multiprocess:
            # Prepare the pdbbind
            __generate_digest_parallel(paths, archive, overwrite, digestFormat, label, all_boxes)
        else:
            # Prepare the database
            __generate_digest_no_parallel(paths, archive, overwrite, digestFormat, label, all_boxes)
    else:
        __generate_digest_single(paths, archive, overwrite, digestFormat, label, all_boxes)
