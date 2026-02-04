#!/usr/bin/env python3

# Description
###############################################################################
'''
This module is responsible for docking running.

It is imported as:

import OCDocker.Processing.Dock as ocdock
'''

# Imports
###############################################################################
import gc
import os
import shutil

import multiprocessing as mp

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from typing import Any, List, Protocol, Tuple, Union

import OCDocker.Docking.Future.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Error as ocerror
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Validation as ocvalidation

from OCDocker.Config import get_config

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

class _Lock(Protocol):
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...
    def acquire(self, *args: Any, **kwargs: Any) -> bool: ...
    def release(self) -> None: ...


DockArgs = Tuple[str, str, str, str, _Lock, bool, str, bool]


def __core_run_dock(path: str, ligandDir: str, archive: str, dockingAlgorithm: str, lock: _Lock, overwrite: bool, digestFormat: str = "json", all_boxes: bool = False) -> int:
    '''Performs the docking.

    Parameters
    ----------
    path : str
        The path to the protein directory.
    ligandDir : str
        If the ligand is not in the same directory as the receptor, this is the path to the ligand directory. By default "". If this is not empty, the ligand will be searched in this directory, otherwise, it will be searched in the same directory as the receptor.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    dockingAlgorithm : str
        Which docking algorithm will be used [gnina, vina, smina, plants].
    lock : Lock
        The lock used to synchronize file access.
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

    # Set receptor data
    receptorPath = f"{path}/receptor.pdb"
    receptorDescriptorPath = f"{path}/receptor_descriptors.json"

    # Set ligand data
    if archive == "dudez":
        ligandPath = f"{ligandDir}/ligand.smi"
    else:
        ligandPath = f"{ligandDir}/ligand.mol2"

    ligandDescriptorPath = f"{ligandDir}/ligand_descriptors.json"

    # If the complex has descriptor files for both ligand and receptor
    if os.path.isfile(receptorDescriptorPath) and os.path.isfile(ligandDescriptorPath):
        # Find the protein name
        ptn = receptorPath.split(os.path.sep)[-2]

        # Get the box path TODO: add support to multiple boxes
        boxPath = f"{ligandDir}/boxes/box0.pdb"
        if all_boxes:
            if len(glob(f"{ligandDir}/boxes/box*.pdb")) == 0:
                errMsg = f"No box files found under '{ligandDir}/boxes'. Skipping docking for ligand '{ligandDir}'."
                config = get_config()
                ocprint.print_warning_log(errMsg, f"{config.logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                ocprint.print_warning(errMsg)
                return ocerror.Error.skip()
        else:
            if not os.path.isfile(boxPath):
                errMsg = f"No box file found at '{boxPath}'. Skipping docking for ligand '{ligandDir}'."
                config = get_config()
                ocprint.print_warning_log(errMsg, f"{config.logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                ocprint.print_warning(errMsg)
                return ocerror.Error.skip()

        # Initialise an return state
        returnState = 0

        dockingAlgorithms = {
            "gnina": __run_gnina,
            "vina": __run_vina,
            "smina": __run_smina,
            "plants": __run_plants
        }

        # Check if the docking algorithm is valid
        if dockingAlgorithm not in dockingAlgorithms:
            errMsg = f"Wrong docking algorithm. Expected ['gnina', 'vina', 'smina', 'plants'] and got '{dockingAlgorithm}'."

            config = get_config()
            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
            return ocerror.Error.not_supported_docking_algorithm(errMsg, level = ocerror.ReportLevel.ERROR)
        else:
            # Call the docking algorithm function based on the dockingAlgorithms dictionary
            returnState = dockingAlgorithms[dockingAlgorithm](ligandPath, ligandDescriptorPath, receptorPath, receptorDescriptorPath, boxPath, ptn, archive, lock, overwrite = overwrite, digestFormat = digestFormat, all_boxes = all_boxes)

    else:
        if not os.path.isfile(receptorDescriptorPath):
            errMsg = f"There is no receptor descriptor json file for the protein in the path '{receptorDescriptorPath}'. Error found while trying to run the '{dockingAlgorithm}' docking software."
            config = get_config()
            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
            _ = ocerror.Error.receptor_or_ligand_descriptor_not_exist(errMsg, level = ocerror.ReportLevel.ERROR)

        if not os.path.isfile(ligandDescriptorPath):
            errMsg = f"There is no ligand descriptor json file for the protein in the path '{ligandDescriptorPath}'. Error found while trying to run the '{dockingAlgorithm}' docking software."
            config = get_config()
            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
            _ = ocerror.Error.receptor_or_ligand_descriptor_not_exist(errMsg, level = ocerror.ReportLevel.ERROR)

        return ocerror.Error.receptor_or_ligand_descriptor_not_exist()

    # Check if the docking was successful
    if returnState != 0:
        errMsg = f"Error found while trying to run the '{dockingAlgorithm}' docking software."

        config = get_config()
        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
        return ocerror.Error.docking_failed(errMsg, level = ocerror.ReportLevel.ERROR)

    return ocerror.Error.ok()

def __list_boxes(ligandDir: str, all_boxes: bool) -> List[Tuple[str, str]]:
    '''List box IDs and paths for a ligand directory.'''
    if all_boxes:
        boxes = sorted(glob(f"{ligandDir}/boxes/box*.pdb"))
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in boxes]
    box0 = os.path.join(ligandDir, "boxes", "box0.pdb")
    return [("box0", box0)] if os.path.isfile(box0) else []

def __normalize_run_result(result: Union[int, Tuple[int, str]]) -> Tuple[int, str]:
    '''Normalize subprocess return values to (exit_code, stderr).'''
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, ""

def __run_dock_no_parallel(complexList: List[Tuple[str, List[str]]], archive: str, dockingAlgorithm: str, overwrite: bool, digestFormat: str, desc: str, all_boxes: bool) -> int:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_run_dock function.

    Parameters
    ----------
    complexList : List[Tuple[str, List[str]]]
        A list of tuples with the path to the protein directory and a list of ligand directories.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    dockingAlgorithm : str
        Which docking algorithm will be used [vina, smina, plants].
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

    # Track error codes from all docking operations
    error_codes = []

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Reuse a single lock across the sequential docking tasks
        lock = mp.Lock()

        # For each file in dirs
        for cl in tqdm(iterable = complexList, total = len(complexList), desc=desc):
            for ligandDir in cl[1]:
                # Call the core dock function (shared between parallel and not parallel)
                return_code = __core_run_dock(cl[0], ligandDir, archive, dockingAlgorithm, lock, overwrite, digestFormat, all_boxes)
                # Track non-zero error codes
                if return_code != ocerror.ErrorCode.OK:
                    error_codes.append(return_code)

            # Clear the memory
            gc.collect()
        # Clear the memory
        gc.collect()

    # Return the most severe error code, or OK if all succeeded
    if error_codes:
        # Return the first non-OK error code (errors are already logged by core functions)
        return error_codes[0]
    return ocerror.Error.ok()

def __run_dock_parallel(complexList: List[Tuple[str, List[str]]], archive: str, dockingAlgorithm: str, overwrite: bool, digestFormat: str, desc: str, all_boxes: bool) -> int:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    complexList : List[Tuple[str, List[str]]]
        A list of tuples with the path to the protein directory and a list of ligand directories.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    dockingAlgorithm : str
        Which docking algorithm will be used [vina, smina, plants].
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
    raw_arguments: List[Tuple[str, str, str, str, bool, str, bool]] = []

    # For each file in complexList
    for cl in complexList:
        # Now loop over the ligands of this protein
        for ligandDir in cl[1]:
            # Add the arguments to the list (creating one execution for each pair receptor-ligand)
            raw_arguments.append((cl[0], ligandDir, archive, dockingAlgorithm, overwrite, digestFormat, all_boxes))

    # Track error codes from all docking operations
    error_codes = []

    try:
        # Create a shared lock for all workers to guard shared outputs
        config = get_config()
        shared_lock = mp.Lock()
        # Inject lock into arguments
        arguments: List[DockArgs] = [
            (arg[0], arg[1], arg[2], arg[3], shared_lock, arg[4], arg[5], arg[6])
            for arg in raw_arguments
        ]
        # Create a Thread pool with the maximum available_cores
        with Pool(config.available_cores) as p:
            # Perform the multi process and collect return codes
            for return_code in tqdm(p.imap_unordered(__thread_run_dock_parallel, arguments), total = len(arguments), desc = desc):
                # Track non-zero error codes
                if return_code != ocerror.ErrorCode.OK:
                    error_codes.append(return_code)
                # Clear the memory
                gc.collect()
    except IOError as e:
        config = get_config()
        ocprint.print_error_log(f"Problem while running docking software {dockingAlgorithm} in parallel. Exception: {e}", f"{config.logdir}/{archive}_docking_report.log")
        return ocerror.Error.docking_failed(f"Problem while running docking software {dockingAlgorithm} in parallel. Exception: {e}", level = ocerror.ReportLevel.ERROR)

    # Return the most severe error code, or OK if all succeeded
    if error_codes:
        # Return the first non-OK error code (errors are already logged by core functions)
        # In case of multiple errors, return the first one encountered
        return error_codes[0]
    return ocerror.Error.ok()

def __run_gnina(ligandPath: str, ligandDescriptorPath: str, receptorPath: str, receptorDescriptorPath: str, boxPath: str, ptn: str, archive: str, lock: _Lock, overwrite: bool = False, digestFormat: str = "json", all_boxes: bool = False) -> int:
    '''Runs gnina.

    Parameters
    ----------
    ligandPath : str);
        The ligand directory path.
    ligandDescriptorPath : str
        The ligand descriptor path.
    receptorPath : str
        The receptor directory path.
    receptorDescriptorPath : str
        The receptor descriptor path.
    boxPath : str
        The box directory.
    ptn : str
        The protein name.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    lock : Lock
        The lock used to synchronize file access.
    overwrite : bool, optional
        If True, overwrite the output file. Defaults to False.
    digestFormat : str, optional
        The digest format. Options are [json]. Defaults to "json".

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Get the ligand Dir
    ligandDir = os.path.dirname(ligandPath)

    # Flag to denote if its needed to run this protein through gnina
    needToRun = False
    # Check if the gninaFiles directory exists
    if not os.path.isdir(f"{ligandDir}/gninaFiles"):
        errMsg = f"The directory '{ligandDir}/gninaFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok."

        config = get_config()
        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_ERROR.log")
        return ocerror.Error.dir_not_exist(errMsg, level = ocerror.ReportLevel.ERROR)

    boxes = __list_boxes(ligandDir, all_boxes)
    use_box_id = all_boxes and len(boxes) > 1
    # Get the folder for each run
    runPaths: List[Tuple[str, str, str]] = []
    for box_id, _box_path in boxes:
        if use_box_id:
            runPath = f"{ligandDir}/gninaFiles/{box_id}"
            os.makedirs(runPath, exist_ok=True)
        else:
            runPath = f"{ligandDir}/gninaFiles"
        runPaths.append((runPath, box_id, _box_path))

    # Check if all files have been processed
    for runPath, _box_id, _box_path in runPaths:
        # Get the run number
        runNumber = 0 # TODO: add support to multiple runs, currently only 0, the code should be something like: runPath.split(os.path.sep)[-1]

        # If the output does not exist or overwrite flag is set to true
        if overwrite or not os.path.isfile(f"{runPath}/gnina_{runNumber}.log") or not os.path.isfile(f"{runPath}/gnina_{runNumber}.pdbqt"):
            needToRun = True
            break

    # If is needed to run (at least one protein)
    if needToRun:
        # Get the ligand name
        lig = os.path.split(os.path.dirname(ligandPath))[-1]

        # Start the lock with statement
        with lock:
            # Read the receptor and the ligand
            receptor = ocr.Receptor(receptorPath, from_json_descriptors = receptorDescriptorPath, name = f"{ptn}_receptor")
            ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptorPath, name = f"{ptn}_{lig}_ligand")

        # If receptor and ligand are not null
        if receptor and ligand:
            # For each path in the paths array (will be more than on in case of multiple boxes)
            for runPath, box_id, _box_path in runPaths:
                # Get the run number
                runNumber = 0 # TODO: add support to multiple runs, currently only 0, the code should be something like: runPath.split(os.path.sep)[-1]

                # Set the prepared receptor and ligand paths
                preparedReceptorPath = f"{os.path.dirname(receptorPath)}/prepared_receptor.pdbqt"
                preparedLigandPath = f"{ligandDir}/prepared_ligand.pdbqt"

                # Parameterizing paths
                gninaLog = f"{runPath}/gnina_{runNumber}.log"
                gninaOutput = f"{runPath}/gnina_{runNumber}.pdbqt"

                # Start the lock with statement
                with lock:
                    # Create the gnina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                    gnina = ocgnina.Gnina(f"{runPath}/conf_gnina.conf", _box_path if use_box_id else boxPath, receptor, preparedReceptorPath, ligand, preparedLigandPath, gninaLog, gninaOutput, name = f"{ptn}_run_{runNumber}", overwrite_config = overwrite)

                # Check if the gnina object has been correctly created
                if not gnina:
                    errMsg = f"Could not generate gnina object for the protein in dir '{ligandPath}'. Error found while trying to run the 'gnina' docking software."

                    config = get_config()
                    ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_ERROR.log")
                    return ocerror.Error.docking_object_not_generated(errMsg, level = ocerror.ReportLevel.ERROR)

                # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(gnina.prepared_ligand) or os.path.getsize(gnina.prepared_ligand) == 0 or not ocvalidation.is_molecule_valid(gnina.prepared_ligand):
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare ligand
                            result = gnina.run_prepare_ligand(overwrite=overwrite)
                            # If result is a tuple
                            if isinstance(result, tuple):
                                # If the result is not 0
                                if result[0] != 0:
                                    # Throw the generic Exception
                                    raise Exception(result[1])
                            # Otherwise is an int
                            else:
                                # If the result is not 0
                                if result != 0:
                                    # Throw the generic Exception
                                    raise Exception("The prepare ligand routine returned an error code different than 0.")

                        except Exception as e:
                            errMsg = f"Could not run the prepare ligand routine for the protein in dir '{gnina.input_ligand_path}'. Error found while trying to run the 'gnina' docking software. Error: {e}"

                            config = get_config()
                            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_ERROR.log")
                            return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                    # Check again if the generated ligand has size 0 or is invalid
                    if os.path.getsize(gnina.prepared_ligand) == 0 or not ocvalidation.is_molecule_valid(gnina.prepared_ligand):
                        prep_cmd = gnina.preparation_strategy.get_ligand_command(gnina.input_ligand_path, gnina.prepared_ligand)
                        errMsg = f"The prepare ligand script has made an output of 0kb again for ligand '{gnina.prepared_ligand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(prep_cmd)}"

                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_ERROR.log")
                        return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(gnina.prepared_receptor) or not ocvalidation.is_molecule_valid_with_retry(gnina.prepared_receptor):
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare receptor
                            result = gnina.run_prepare_receptor(overwrite=overwrite)
                            # If result is a tuple
                            if isinstance(result, tuple):
                                # If the result is not 0
                                if result[0] != 0:
                                    # Throw the generic Exception
                                    raise Exception(result[1])
                            # Otherwise is an int
                            else:
                                # If the result is not 0
                                if result != 0:
                                    # Throw the generic Exception
                                    raise Exception("The prepare receptor routine returned an error code different than 0.")
                        except Exception as e:
                            errMsg = f"Could not run the prepare receptor routine for the protein in dir '{gnina.input_receptor_path}'. Error found while trying to run the 'gnina' docking software. Error: {e}"

                            config = get_config()
                            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_ERROR.log")
                            return ocerror.Error.receptor_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                    # Check if the generated receptor has size 0 or is invalid
                    if not ocvalidation.is_molecule_valid_with_retry(gnina.prepared_receptor):
                        prep_cmd = gnina.preparation_strategy.get_receptor_command(gnina.input_receptor_path, gnina.prepared_receptor)
                        errMsg = f"The prepare receptor has made an output of 0kb for receptor '{gnina.prepared_receptor}' or is not valid... Here is its command line so you might be able to debug it by hand.\n{' '.join(prep_cmd)}"

                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_ERROR.log")
                        return ocerror.Error.receptor_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                # Check if gnina output exists
                if overwrite or not os.path.isfile(gninaOutput) or not os.path.isfile(gninaLog):
                    # Run gnina
                    run_result = gnina.run_gnina()
                    exit_code, stderr = __normalize_run_result(run_result)
                    if exit_code != ocerror.ErrorCode.OK:
                        errMsg = f"Gnina failed for '{ptn}' run '{runNumber}'."
                        if stderr:
                            errMsg += f" Stderr: {stderr}"
                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_ERROR.log")
                        return exit_code

                    # Append to the digest the results
                    _ = ocgnina.generate_digest(f"{ligandDir}/dockingDigest.json", gnina.gnina_log, overwrite = overwrite, digestFormat = digestFormat, box_id = box_id if use_box_id else None)
                else:
                    errMsg = f"The gnina output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '{runPath}/gnina_{runNumber}.log' path. Gnina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true"

                    config = get_config()
                    ocprint.print_warning_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_WARNING.log")
                    ocprint.print_warning(errMsg)
        else:
            errMsg = f"Could not generate receptor or ligand object for the protein in dir '{ligandPath}'. Error found while trying to run the 'gnina' docking software."

            config = get_config()
            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_ERROR.log")
            return ocerror.Error.receptor_or_ligand_not_generated(errMsg, level = ocerror.ReportLevel.ERROR)
    else:
        errMsg = f"The gnina output for '{ptn}' for all boxes is already generated and you can check it at the '{ligandPath}/gninaFiles/*/gnina_<runNumber>.log' path. Gnina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

        config = get_config()
        ocprint.print_warning_log(errMsg, f"{config.logdir}/{archive}_gnina_run_report_WARNING.log")
        ocprint.print_warning(errMsg)

    return ocerror.Error.ok()

def __run_plants(ligandPath: str, ligandDescriptorPath: str, receptorPath: str, receptorDescriptorPath: str, boxPath: str, ptn: str, archive: str, lock: _Lock, overwrite: bool = False, digestFormat: str = "json", all_boxes: bool = False) -> int:
    '''Runs PLANTS.

    Parameters
    ----------
    ligandPath : str
        The ligand directory path.
    ligandDescriptorPath : str
        The ligand descriptor path.
    receptorPath : str
        The receptor directory path.
    receptorDescriptorPath : str
        The receptor descriptor path.
    boxPath : str
        The box directory.
    ptn : str
        The protein name.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    lock : Lock
        The lock used to synchronize file access
    overwrite : bool, optional
        If True, overwrite the output file. Defaults to False.
    digestFormat : str, optional
        The digest format. Options are [json, csv]. Defaults to "json".

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Get the ligand Dir
    ligandDir = os.path.dirname(ligandPath)

    # Check if plantsFiles does not exist
    if not os.path.isdir(f"{ligandDir}/plantsFiles/"):
        errMsg = f"The directory '{ligandDir}/plantsFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok."

        config = get_config()
        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_ERROR.log")
        return ocerror.Error.dir_not_exist(errMsg, level = ocerror.ReportLevel.ERROR)

    # Flag to denote if its needed to run this protein through plants
    needToRun = False
    boxes = __list_boxes(ligandDir, all_boxes)
    use_box_id = all_boxes and len(boxes) > 1
    # Get the folder for each run
    runPaths: List[Tuple[str, str, str]] = []
    for box_id, _box_path in boxes:
        if use_box_id:
            runPath = f"{ligandDir}/plantsFiles/{box_id}"
            os.makedirs(runPath, exist_ok=True)
        else:
            runPath = f"{ligandDir}/plantsFiles"
        runPaths.append((runPath, box_id, _box_path))
    # Check if all files have been processed
    for runPath, _box_id, _box_path in runPaths:
        # Get the run number
        #runNumber = runPath.split(os.path.sep)[-1]
        # Parameterizing paths
        plantsOutput = f"{runPath}/run"
        plantsRankingCsv = f"{plantsOutput}/ranking.csv"
        # If the output dir or the output file does not exist or overwrite flag is true
        if overwrite or not os.path.isdir(plantsOutput) or not os.path.isfile(plantsRankingCsv):
            needToRun = True
            break

    # If is needed to run (at least one protein)
    if needToRun:
        # Separate the extension from file path
        mol2Path, _ = os.path.splitext(receptorPath)

        # Start the lock with statement
        with lock:
            # Read the receptor and the ligand (passing the mol2!!!)
            receptor = ocr.Receptor(receptorPath, mol2_path = f"{mol2Path}.mol2", from_json_descriptors = receptorDescriptorPath, name = f"{ptn}_receptor")
            ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptorPath, name = f"{ptn}_ligand")

        # If receptor and ligand are not null
        if receptor and ligand:
            # Set the prepared receptor and ligand paths
            preparedReceptorPath = f"{os.path.dirname(receptorPath)}/prepared_receptor.mol2"
            preparedLigandPath = f"{ligandDir}/prepared_ligand.mol2"

            # For each path in the paths array (will be more than on in case of multiple boxes)
            for runPath, box_id, _box_path in runPaths:
                # Get the run number
                runNumber = 0 # TODO: add support to multiple runs, currently only 0, the code should be something like: runPath.split(os.path.sep)[-1]

                # Parameterizing paths
                plantsLog = f"{runPath}/plants_{runNumber}.log"
                plantsOutput = f"{runPath}/run"
                plantsRankingCsv = f"{plantsOutput}/ranking.csv"

                # Start the lock with statement
                with lock:
                    # Create the smina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                    plants = ocplants.PLANTS(f"{runPath}/conf_plants.txt", _box_path if use_box_id else boxPath, receptor, preparedReceptorPath, ligand, preparedLigandPath, plantsLog, plantsOutput, name=f"{ptn} PLANTS", overwrite_config = overwrite)

                # Check if the smina object has been correctly created
                if not plants:
                    errMsg = f"Could not generate plants object for the protein in dir '{ligandDir}'. Error found while trying to run the 'PLANTS' docking software."

                    config = get_config()
                    ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_ERROR.log")
                    return ocerror.Error.docking_object_not_generated(errMsg, level = ocerror.ReportLevel.ERROR)

                # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(plants.prepared_ligand) or os.path.getsize(plants.prepared_ligand) == 0 or not ocvalidation.is_molecule_valid(plants.prepared_ligand):
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare ligand
                            result = plants.run_prepare_ligand(overwrite=overwrite)
                            # If result is a tuple
                            if isinstance(result, tuple):
                                # If the result is not 0
                                if result[0] != 0:
                                    # Throw the generic Exception
                                    raise Exception(result[1])
                            # Otherwise is an int
                            else:
                                # If the result is not 0
                                if result != 0:
                                    # Throw the generic Exception
                                    raise Exception("The prepare ligand routine returned an error code different than 0.")
                        except Exception as e:
                            errMsg = f"Could not run the prepare ligand routine for the protein in dir '{plants.input_ligand_path}'. Error found while trying to run the 'PLANTS' docking software. Error: {e}"

                            config = get_config()
                            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_ERROR.log")
                            return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                    # Check if the generated ligand has size 0 or is invalid
                    if os.path.getsize(plants.prepared_ligand) == 0 or not ocvalidation.is_molecule_valid(plants.prepared_ligand):
                        prep_cmd = plants.preparation_strategy.get_ligand_command(plants.input_ligand_path, plants.prepared_ligand)
                        errMsg = f"SPORES has made an output of 0kb again for ligand '{plants.prepared_ligand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(prep_cmd)}"

                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_ERROR.log")
                        return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(plants.prepared_receptor) or not ocvalidation.is_molecule_valid_with_retry(plants.prepared_receptor):
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare receptor
                            result = plants.run_prepare_receptor(overwrite=overwrite)
                            # If result is a tuple
                            if isinstance(result, tuple):
                                # If the result is not 0
                                if result[0] != 0:
                                    # Throw the generic Exception
                                    raise Exception(result[1])
                            # Otherwise is an int
                            else:
                                # If the result is not 0
                                if result != 0:
                                    # Throw the generic Exception
                                    raise Exception("The prepare receptor routine returned an error code different than 0.")
                        except Exception as e:
                            errMsg = f"Could not run the prepare receptor routine for the protein in dir '{plants.input_receptor_path}'. Error found while trying to run the 'PLANTS' docking software. Error: {e}"

                            config = get_config()
                            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_ERROR.log")
                            return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                    # Check if the generated receptor has size 0 or is invalid
                    if not ocvalidation.is_molecule_valid_with_retry(plants.prepared_receptor):
                        prep_cmd = plants.preparation_strategy.get_receptor_command(plants.input_receptor_path, plants.prepared_receptor)
                        errMsg = f"SPORES has made an output of 0kb for receptor '{plants.prepared_receptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(prep_cmd)}"
                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_ERROR.log")
                        return ocerror.Error.receptor_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                # Check if PLANTS output exists and its size is not 0
                if overwrite or not os.path.isdir(plantsOutput) or not os.path.isfile(plantsRankingCsv) or os.path.getsize(plantsRankingCsv) == 0:
                    # If there is already a PLANTS output (PLANTS do not run if the folder is already created. And knowing that PLANTS will ALWAYS run if this code is interpreted, just delete the folder if it exists and lets avoid headaches)
                    if os.path.isdir(plantsOutput):
                        # Remove the folder and its contets
                        shutil.rmtree(plantsOutput)

                    # Run PLANTS
                    run_result = plants.run_plants(overwrite=overwrite)
                    exit_code, stderr = __normalize_run_result(run_result)
                    if exit_code != ocerror.ErrorCode.OK:
                        errMsg = f"PLANTS failed for '{ptn}' run '{runNumber}'."
                        if stderr:
                            errMsg += f" Stderr: {stderr}"
                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_ERROR.log")
                        return exit_code

                    # Append to the digest the results
                        _ = ocplants.generate_digest(f"{ligandDir}/dockingDigest.json", plants.plants_log, overwrite = overwrite, digestFormat = digestFormat, box_id = box_id if use_box_id else None)
                else:
                    errMsg = f"The PLANTS output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '*/run/plants_<runNumber>.log' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

                    config = get_config()
                    ocprint.print_warning_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_WARNING.log")
                    ocprint.print_warning(errMsg)
        else:
            errMsg = f"Could not generate receptor or ligand object for the protein in dir '{ligandDir}'. Error found while trying to run the 'plants' docking software."

            config = get_config()
            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_ERROR.log")
            return ocerror.Error.receptor_or_ligand_not_generated(errMsg, level = ocerror.ReportLevel.ERROR)
    else:
        errMsg = f"The PLANTS output for '{ptn}' is already generated and you can check it at the '{ligandDir}/plantsFiles' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

        config = get_config()
        ocprint.print_warning_log(errMsg, f"{config.logdir}/{archive}_plants_run_report_WARNING.log")
        ocprint.print_warning(errMsg)

    return ocerror.Error.ok()

def __run_smina(ligandPath: str, ligandDescriptorPath: str, receptorPath: str, receptorDescriptorPath: str, boxPath: str, ptn: str, archive: str, lock: _Lock, overwrite: bool = False, digestFormat: str = "json", all_boxes: bool = False) -> int:
    '''Runs SMINA.

    Parameters
    ----------
    ligandPath : str
        The ligand directory path.
    ligandDescriptorPath : str
        The ligand descriptor path.
    receptorPath : str
        The receptor directory path.
    receptorDescriptorPath : str
        The receptor descriptor path.
    boxPath : str
        The box directory.
    ptn : str
        The protein name.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    lock : Lock
        The lock used to synchronize file access.
    overwrite : bool, optional
        If True, overwrite the output file. Defaults to False.
    digestFormat : str, optional
        The digest format. Options are [json, csv]. Defaults to "json".
    all_boxes : bool, optional
        If True, process all box*.pdb files under the ligand directory.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Get the ligand Dir
    ligandDir = os.path.dirname(ligandPath)

    # Check if sminaFiles does not exist
    if not os.path.isdir(f"{ligandDir}/sminaFiles"):
        errMsg = f"The directory '{ligandDir}/sminaFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok."

        config = get_config()
        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_smina_run_report_ERROR.log")
        return ocerror.Error.dir_not_exist(errMsg, level = ocerror.ReportLevel.ERROR)

    boxes = __list_boxes(ligandDir, all_boxes)
    use_box_id = all_boxes and len(boxes) > 1

    # Get the folder for each run
    runPaths: List[Tuple[str, str, str]] = []
    for box_id, _box_path in boxes:
        if use_box_id:
            runPath = f"{ligandDir}/sminaFiles/{box_id}"
            os.makedirs(runPath, exist_ok=True)
        else:
            runPath = f"{ligandDir}/sminaFiles"
        runPaths.append((runPath, box_id, _box_path))

    # If is needed to run (overwrite is set or no output is produced)
    needToRun = False
    for runPath, _box_id, _box_path in runPaths:
        sminaLog = f"{runPath}/smina.log"
        sminaOutput = f"{runPath}/smina.pdbqt"
        if overwrite or not os.path.isfile(sminaLog) or not os.path.isfile(sminaOutput):
            needToRun = True
            break

    if needToRun:
        # Get the ligand name
        lig = os.path.split(os.path.dirname(ligandPath))[-1]

        # Start the lock with statement
        with lock:
            # Read the receptor and the ligand
            receptor = ocr.Receptor(receptorPath, from_json_descriptors = receptorDescriptorPath, name = f"{ptn}_receptor")
            ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptorPath, name = f"{ptn}_ligand")

        # If receptor and ligand are not null
        if receptor and ligand:
            # Set the prepared receptor and ligand paths
            preparedReceptorPath = f"{os.path.dirname(receptorPath)}/prepared_receptor.pdbqt"
            preparedLigandPath = f"{ligandDir}/prepared_ligand.pdbqt"

            # For each path in the paths array (will be more than one in case of multiple boxes)
            for runPath, box_id, _box_path in runPaths:
                # Parameterizing paths
                sminaLog = f"{runPath}/smina.log"
                sminaOutput = f"{runPath}/smina.pdbqt"

                # Start the lock with statement
                with lock:
                    # Create the smina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                    smina = ocsmina.Smina(f"{runPath}/conf_smina.conf", _box_path if use_box_id else boxPath, receptor, preparedReceptorPath, ligand, preparedLigandPath, sminaLog, sminaOutput, name=f"{ptn}_smina", overwrite_config = overwrite)

                # Check if the smina object has been correctly created
                if not smina:
                    errMsg = f"Could not generate smina object for the protein in dir '{ligandPath}'. Error found while trying to run the 'smina' docking software."

                    config = get_config()
                    ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_smina_run_report_ERROR.log")
                    return ocerror.Error.docking_object_not_generated(errMsg, level = ocerror.ReportLevel.ERROR)

                # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(smina.prepared_ligand) or os.path.getsize(smina.prepared_ligand) == 0 or not ocvalidation.is_molecule_valid(smina.prepared_ligand):
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare ligand
                            result = smina.run_prepare_ligand(overwrite=overwrite)
                            # If result is a tuple
                            if isinstance(result, tuple):
                                # If the result is not 0
                                if result[0] != 0:
                                    # Throw the generic Exception
                                    raise Exception(result[1])
                            # Otherwise is an int
                            else:
                                # If the result is not 0
                                if result != 0:
                                    # Throw the generic Exception
                                    raise Exception("The prepare ligand routine returned an error code different than 0.")
                        except Exception as e:
                            errMsg = f"Could not run the prepare ligand routine for the protein in dir '{smina.input_ligand_path}'. Error found while trying to run the 'smina' docking software. Error: {e}"

                            config = get_config()
                            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_smina_run_report_ERROR.log")
                            return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                    # Check if the generated ligand has size 0 or is invalid
                    if os.path.getsize(smina.prepared_ligand) == 0 or not ocvalidation.is_molecule_valid(smina.prepared_ligand):
                        prep_cmd = smina.preparation_strategy.get_ligand_command(smina.input_ligand_path, smina.prepared_ligand)
                        errMsg = f"The prepare ligand script has made an output of 0kb for ligand '{smina.prepared_ligand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(prep_cmd)}"

                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_smina_run_report_ERROR.log")
                        return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(smina.prepared_receptor) or not ocvalidation.is_molecule_valid_with_retry(smina.prepared_receptor):
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare receptor
                            result = smina.run_prepare_receptor(overwrite=overwrite)
                            # If result is a tuple
                            if isinstance(result, tuple):
                                # If the result is not 0
                                if result[0] != 0:
                                    # Throw the generic Exception
                                    raise Exception(result[1])
                            # Otherwise is an int
                            else:
                                # If the result is not 0
                                if result != 0:
                                    # Throw the generic Exception
                                    raise Exception("The prepare receptor routine returned an error code different than 0.")
                        except Exception as e:
                            errMsg = f"Could not run the prepare receptor routine for the protein in dir '{smina.input_receptor_path}'. Error found while trying to run the 'smina' docking software. Error: {e}"

                            config = get_config()
                            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_smina_run_report_ERROR.log")
                            return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                    # Check if the generated receptor has size 0 or is invalid
                    if not ocvalidation.is_molecule_valid_with_retry(smina.prepared_receptor):
                        prep_cmd = smina.preparation_strategy.get_receptor_command(smina.input_receptor_path, smina.prepared_receptor)
                        errMsg = f"The prepare receptor has made an output of 0kb for receptor '{smina.prepared_receptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(prep_cmd)}"

                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_smina_run_report_ERROR.log")
                        return ocerror.Error.receptor_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                # Run smina (no need to recheck for overwrite or output existance because it is already done some lines ago)
                run_result = smina.run_smina(overwrite=overwrite)
                exit_code, stderr = __normalize_run_result(run_result)
                if exit_code != ocerror.ErrorCode.OK:
                    errMsg = f"Smina failed for '{ptn}'."
                    if stderr:
                        errMsg += f" Stderr: {stderr}"
                    config = get_config()
                    ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_smina_run_report_ERROR.log")
                    return exit_code

                # Append to the digest the results
                _ = ocsmina.generate_digest(f"{ligandDir}/dockingDigest.json", smina.smina_log, overwrite = overwrite, digestFormat = digestFormat, box_id = box_id if use_box_id else None)
        else:
            errMsg = f"Could not generate receptor or ligand object for the protein in dir '{ligandPath}'. Error found while trying to run the 'smina' docking software."

            config = get_config()
            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_smina_run_report_ERROR.log")
            return ocerror.Error.receptor_or_ligand_not_generated(errMsg, level = ocerror.ReportLevel.ERROR)
    else:
        errMsg = f"The smina output for '{ptn}' is already generated and you can check it at the '{ligandDir}/sminaFiles' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

        config = get_config()
        ocprint.print_warning_log(errMsg, f"{config.logdir}/{archive}_smina_run_report_WARNING.log")
        ocprint.print_warning(errMsg)

    return ocerror.Error.ok()

def __run_vina(ligandPath: str, ligandDescriptorPath: str, receptorPath: str, receptorDescriptorPath: str, boxPath: str, ptn: str, archive: str, lock: _Lock, overwrite: bool = False, digestFormat: str = "json", all_boxes: bool = False) -> int:
    '''Runs vina.

    Parameters
    ----------
    ligandPath : str
        The ligand directory path.
    ligandDescriptorPath : str
        The ligand descriptor path.
    receptorPath : str
        The receptor directory path.
    receptorDescriptorPath : str
        The receptor descriptor path.
    boxPath : str
        The box directory.
    ptn : str
        The protein name.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    lock : Lock
        The lock used to synchronize file access.
    overwrite : bool, optional
        If True, overwrite the output file. Defaults to False.
    digestFormat : str, optional
        The digest format. Options are [json]. Defaults to "json".

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Get the ligand Dir
    ligandDir = os.path.dirname(ligandPath)

    # Flag to denote if its needed to run this protein through vina
    needToRun = False
    # Check if the vinaFiles directory exists
    if not os.path.isdir(f"{ligandDir}/vinaFiles"):
        errMsg = f"The directory '{ligandDir}/vinaFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok."

        config = get_config()
        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_ERROR.log")
        return ocerror.Error.dir_not_exist(errMsg, level = ocerror.ReportLevel.ERROR)

    boxes = __list_boxes(ligandDir, all_boxes)
    use_box_id = all_boxes and len(boxes) > 1
    # Get the folder for each run
    runPaths: List[Tuple[str, str, str]] = []
    for box_id, _box_path in boxes:
        if use_box_id:
            runPath = f"{ligandDir}/vinaFiles/{box_id}"
            os.makedirs(runPath, exist_ok=True)
        else:
            runPath = f"{ligandDir}/vinaFiles"
        runPaths.append((runPath, box_id, _box_path))

    # Check if all files have been processed
    for runPath, _box_id, _box_path in runPaths:
        # Get the run number
        runNumber = 0 # TODO: add support to multiple runs, currently only 0, the code should be something like: runPath.split(os.path.sep)[-1]

        # If the output does not exist or overwrite flag is set to true
        if overwrite or not os.path.isfile(f"{runPath}/vina_{runNumber}.log") or not os.path.isfile(f"{runPath}/vina_{runNumber}.pdbqt"):
            needToRun = True
            break

    # If is needed to run (at least one protein)
    if needToRun:
        # Get the ligand name
        lig = os.path.split(os.path.dirname(ligandPath))[-1]

        # Start the lock with statement
        with lock:
            # Read the receptor and the ligand
            receptor = ocr.Receptor(receptorPath, from_json_descriptors = receptorDescriptorPath, name = f"{ptn}_receptor")
            ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptorPath, name = f"{ptn}_{lig}_ligand")

        # If receptor and ligand are not null
        if receptor and ligand:
            # For each path in the paths array (will be more than on in case of multiple boxes)
            for runPath, box_id, _box_path in runPaths:
                # Get the run number
                runNumber = 0 # TODO: add support to multiple runs, currently only 0, the code should be something like: runPath.split(os.path.sep)[-1]

                # Set the prepared receptor and ligand paths
                preparedReceptorPath = f"{os.path.dirname(receptorPath)}/prepared_receptor.pdbqt"
                preparedLigandPath = f"{ligandDir}/prepared_ligand.pdbqt"

                # Parameterizing paths
                vinaLog = f"{runPath}/vina_{runNumber}.log"
                vinaOutput = f"{runPath}/vina_{runNumber}.pdbqt"

                # Start the lock with statement
                with lock:
                    # Create the vina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                    vina = ocvina.Vina(f"{runPath}/conf_vina.conf", _box_path if use_box_id else boxPath, receptor, preparedReceptorPath, ligand, preparedLigandPath, vinaLog, vinaOutput, name = f"{ptn}_run_{runNumber}", overwrite_config = overwrite)

                # Check if the vina object has been correctly created
                if not vina:
                    errMsg = f"Could not generate vina object for the protein in dir '{ligandPath}'. Error found while trying to run the 'vina' docking software."

                    config = get_config()
                    ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_ERROR.log")
                    return ocerror.Error.docking_object_not_generated(errMsg, level = ocerror.ReportLevel.ERROR)

                # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(vina.prepared_ligand) or os.path.getsize(vina.prepared_ligand) == 0 or not ocvalidation.is_molecule_valid(vina.prepared_ligand):
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare ligand
                            result = vina.run_prepare_ligand(useOpenBabel = False, overwrite=overwrite) # useOpenBabel has proven to be a dangerous option, it is better to avoid its use for
                            # If result is a tuple
                            if isinstance(result, tuple):
                                # If the result is not 0
                                if result[0] != 0:
                                    # Throw the generic Exception
                                    raise Exception(result[1])
                            # Otherwise is an int
                            else:
                                # If the result is not 0
                                if result != 0:
                                    # Throw the generic Exception
                                    raise Exception("The prepare ligand routine returned an error code different than 0.")

                        except Exception as e:
                            errMsg = f"Could not run the prepare ligand routine for the protein in dir '{vina.input_ligand_path}'. Error found while trying to run the 'vina' docking software. Error: {e}"

                            config = get_config()
                            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_ERROR.log")
                            return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                    # Check again if the generated ligand has size 0 or is invalid
                    if os.path.getsize(vina.prepared_ligand) == 0 or not ocvalidation.is_molecule_valid(vina.prepared_ligand):
                        prep_cmd = vina.preparation_strategy.get_ligand_command(vina.input_ligand_path, vina.prepared_ligand)
                        errMsg = f"The prepare ligand script has made an output of 0kb again for ligand '{vina.prepared_ligand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(prep_cmd)}"

                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_ERROR.log")
                        return ocerror.Error.ligand_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(vina.prepared_receptor) or not ocvalidation.is_molecule_valid_with_retry(vina.prepared_receptor):
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare receptor
                            result = vina.run_prepare_receptor(useOpenBabel = False, overwrite=overwrite) # useOpenBabel has proven to be a dangerous option, it is better to avoid its use for now
                            # If result is a tuple
                            if isinstance(result, tuple):
                                # If the result is not 0
                                if result[0] != 0:
                                    # Throw the generic Exception
                                    raise Exception(result[1])
                            # Otherwise is an int
                            else:
                                # If the result is not 0
                                if result != 0:
                                    # Throw the generic Exception
                                    raise Exception("The prepare receptor routine returned an error code different than 0.")
                        except Exception as e:
                            errMsg = f"Could not run the prepare receptor routine for the protein in dir '{vina.input_receptor_path}'. Error found while trying to run the 'vina' docking software. Error: {e}"

                            config = get_config()
                            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_ERROR.log")
                            return ocerror.Error.receptor_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                    # Check if the generated receptor has size 0 or is invalid
                    if not ocvalidation.is_molecule_valid_with_retry(vina.prepared_receptor):
                        prep_cmd = vina.preparation_strategy.get_receptor_command(vina.input_receptor_path, vina.prepared_receptor)
                        errMsg = f"The prepare receptor has made an output of 0kb for receptor '{vina.prepared_receptor}' or is not valid... Here is its command line so you might be able to debug it by hand.\n{' '.join(prep_cmd)}"

                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_ERROR.log")
                        return ocerror.Error.receptor_not_prepared(errMsg, level = ocerror.ReportLevel.ERROR)

                # Check if vina output exists
                if overwrite or not os.path.isfile(vinaOutput) or not os.path.isfile(vinaLog):
                    # Run vina
                    run_result = vina.run_vina(overwrite=overwrite)
                    exit_code, stderr = __normalize_run_result(run_result)
                    if exit_code != ocerror.ErrorCode.OK:
                        errMsg = f"Vina failed for '{ptn}' run '{runNumber}'."
                        if stderr:
                            errMsg += f" Stderr: {stderr}"
                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_ERROR.log")
                        return exit_code

                    # Append to the digest the results
                    _ = ocvina.generate_digest(f"{ligandDir}/dockingDigest.json", vina.vina_log, overwrite = overwrite, digestFormat = digestFormat, box_id = box_id if use_box_id else None)
                else:
                    errMsg = f"The vina output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '{runPath}/vina_{runNumber}.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true"

                    config = get_config()
                    ocprint.print_warning_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_WARNING.log")
                    ocprint.print_warning(errMsg)
        else:
            errMsg = f"Could not generate receptor or ligand object for the protein in dir '{ligandPath}'. Error found while trying to run the 'vina' docking software."

            config = get_config()
            ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_ERROR.log")
            return ocerror.Error.receptor_or_ligand_not_generated(errMsg, level = ocerror.ReportLevel.ERROR)
    else:
        errMsg = f"The vina output for '{ptn}' for all boxes is already generated and you can check it at the '{ligandPath}/vinaFiles/*/vina_<runNumber>.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

        config = get_config()
        ocprint.print_warning_log(errMsg, f"{config.logdir}/{archive}_vina_run_report_WARNING.log")
        ocprint.print_warning(errMsg)

    return ocerror.Error.ok()

def __thread_run_dock_parallel(arguments: DockArgs) -> int:
    '''Thread aid function to call __core_run_dock.

    Parameters
    ----------
    arguments : list
        The arguments to be passed to __core_run_dock.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Call the core dock function passing the arguments correctly
        path, ligand_dir, archive, algorithm, lock, overwrite, digest_format, all_boxes = arguments
        returnState = __core_run_dock(path, ligand_dir, archive, algorithm, lock, overwrite, digest_format, all_boxes)

    return returnState
## Public ##
def run_dock(paths: Union[List[Tuple[str, List[str]]], Tuple[str, List[str]]], archive: str, dockingAlgorithm: str, overwrite: bool, digestFormat: str, all_boxes: bool = False) -> int:
    '''Run the docking software in parallel or not, based on the multiprocessing flag and input path.

    Parameters
    ----------
    paths : List[Tuple[str, List[str]]] | Tuple[str, List[str]]
        The list of directories or the directory to be processed.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    dockingAlgorithm : str
        Which docking algorithm will be used [vina, smina, plants].
    digestFormat : str
        Which digest format will be used [json].
    overwrite : bool
        If the docking output already exists, should it be overwritten?
    '''

    # Set the description
    label = f"Processing {archive}"

    # If logfiles exists, backup them
    oclogging.backup_log(f"{archive}_{dockingAlgorithm}_run_report_ERROR")
    oclogging.backup_log(f"{archive}_{dockingAlgorithm}_run_report_WARNING")

    # Get config
    config = get_config()

    # If the path is a list
    if isinstance(paths, list):
        # Check if multiprocessing is enabled
        if config.multiprocess:
            # Prepare the pdbbind
            return __run_dock_parallel(paths, archive, dockingAlgorithm, overwrite, digestFormat, label, all_boxes)
        else:
            # Prepare the database
            return __run_dock_no_parallel(paths, archive, dockingAlgorithm, overwrite, digestFormat, label, all_boxes)
    else:
        # Check if multiprocessing is enabled
        if config.multiprocess:
            # Prepare the pdbbind
            return __run_dock_parallel([paths], archive, dockingAlgorithm, overwrite, digestFormat, label, all_boxes)
        else:
            # Prepare the database
            return __run_dock_no_parallel([paths], archive, dockingAlgorithm, overwrite, digestFormat, label, all_boxes)

# Aliases
###############################################################################
run_docking = run_dock
