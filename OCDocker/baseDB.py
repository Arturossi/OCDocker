#!/usr/lib/python3

# Imports
###############################################################################
import errno
import gc
import os
import time
import vaex

import numpy as np
import vaex.dataframe as vdf

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox as octools
import OCDocker.Docking.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Processing.Dock as ocdock
import OCDocker.Processing.Digest as ocdigest
import OCDocker.Processing.Prepare as ocprepare
import OCDocker.Processing.p2rank as ocp2rank
import OCDocker.Processing.Postprocessing.Readlogs as ocreadlogs



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
Sets of classes and functions that are used as base for all databases. It
contains functions that are common to all databases.

They are imported as:

import OCDocker.baseDB as ocbdb
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##
### Read logs

### Merge descriptors in dataframe
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
    with octools.redirect_to_tqdm():
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

    # If logfile exists, backup it for vina, smina and plants (for error and warnings)
    if os.path.isfile(f"{logdir}/read_log_ERROR.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            octools.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/read_log_ERROR.log", f"{logdir}/read_log_past/read_log_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")

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
        octools.print_error_log(f"Problem while mergin descriptors in parallel. Exception: {e}", f"{logdir}/read_log_ERROR_report.log")
        octools.print_error(f"Problem while mergin descriptors in parallel. Exception: {e}")

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

    # If logfile exists, backup it for vina, smina and plants (for error and warnings)
    if os.path.isfile(f"{logdir}/vina_read_log_ERROR.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            octools.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/vina_read_log_ERROR.log", f"{logdir}/read_log_past/vina_read_log_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")
    if os.path.isfile(f"{logdir}/smina_read_log_ERROR.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            octools.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/smina_read_log_ERROR.log", f"{logdir}/read_log_past/smina_read_log_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")
    if os.path.isfile(f"{logdir}/plants_read_log_ERROR.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            octools.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/plants_read_log_ERROR.log", f"{logdir}/read_log_past/plants_read_log_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for dir in tqdm(iterable = dirs, total = len(dirs), desc = desc):
            # Call the core read log function (shared between parallel and not parallel) and store the data into the DataFrame
            ptnList.append(__core_merge_descriptors_in_dataframe(dir))
            # Clear the memory
            gc.collect()

    return vaex.concat(ptnList) # type: ignore


## Public ##
"""def verify_integrity(chosenArchive: str, spacing: float = 0.33) -> None:
    '''Verifies the integrity of the desired database. TODO: remake this function to use the new database structure.

    Parameters
    ----------
    chosenArchive : str
        The name of the archive to verify the integrity.
    spacing : float, optional
        The spacing between the progress bar and the text, by default 0.33

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Verify the integrity of the database
    octools.printv(f"Verifiying the integrity of the {chosenArchive} database")

    # Get all dirs paths in the database
    dirs = glob(f"{chosenArchive}/*")

    # Counter for failed proteins
    failed = 0

    # Parameterizing the amount of directories
    lenDirs = len(dirs)

    # Find the archive type
    archive = chosenArchive.split(os.path.sep)[-1].lower()

    # If logfile exists, backup it
    if os.path.isfile(f"{logdir}/{archive}_integrity_report.log"):
        if not os.path.isdir(f"{logdir}/{archive}_integrity_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_integrity_past")
        os.rename(f"{logdir}/{archive}_integrity_report.log", f"{logdir}/{archive}_integrity_past/{archive}_integrity_report_{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Redirect output to tqdm.write
    with octools.redirect_to_tqdm():
        # For each directory in the database folder
        for dir in tqdm(iterable=dirs, total=lenDirs):
            # If is the index path
            if os.path.basename(dir) in ['index']:
                # Skip it
                continue

            # Parameterizing paths
            p2rankDir = f"{dir}/p2rank"
            vinaDir = f"{dir}/vinaFiles"
            plantsDir = f"{dir}/plantsFiles"

            # Find protein name
            ptn = dir.split(os.path.sep)[-1]

            # Set the input file name path and set the input file name path
            if archive == "dudez":
                fin = f"{dir}/rec.crg.pdb"
                ligand = f""
            elif archive == "pdbbind":
                fin = f"{dir}/{ptn}_protein.pdb"
                ligand = f"{dir}/{ptn}_ligand.mol2"
            else:
                octools.print_error(f"Unknown archive type, expected one of the following ['dudez', 'pdbbind'] and got '{archive}'.")
                return

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
                    octools.print_error_log(f"Unable to generate the p2rank dir for '{dir}'... Error code {errorCode}.", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
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
                    octools.print_error_log(f"Unable to generate the vinaFiles dir for '{dir}'... Error code {errorCode}.", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            # If has no plantsFiles dir
            if not os.path.isdir(plantsDir):
                octools.print_warning(f"The protein '{dir}' has no plantsFiles folder. Trying to fix...")

                # Create the p2rank output dir
                errorCode = octools.safe_create_dir(plantsDir)

                if os.path.isdir(plantsDir):
                    octools.print_success(f"The plantsFiles dir has been generated for '{dir}'.")
                else:
                    octools.print_error(f"Unable to generate the plantsFiles dir for '{dir}'... Error code {errorCode}.")
                    octools.print_error_log(f"Unable to generate the plantsFiles dir for '{dir}'... Error code {errorCode}.", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            octools.printv(f"Checking files for the protein '{dir}'")

            # Check how many boxes are in the p2rankDir
            boxes = glob(f"{p2rankDir}/box*.pdb")
            boxCount = len(boxes)

            # If there is no box in the p2rank output, p2rank will run
            if boxCount == 0:
                octools.print_warning(f"The protein '{dir}' has no box file. Trying to fix...")

                # Run p2rank
                ocp2rank.p2rank_no_parallel([dir], False, desc = "Running p2rank")

                # Check how many boxes are in the p2rankDir (again)
                boxes = glob(f"{p2rankDir}/box*.pdb")
                boxCount = len(boxes)

                if boxCount > 0:
                    octools.print_success(f"Box files generated for '{dir}'.")
                else:
                    octools.print_error(f"The protein '{dir}' still has no box file.")
                    octools.print_error_log(f"The protein '{dir}' still has no box file.", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            # If there is not the same amount of box files as folders in vinaFiles folder
            if len([d for d in glob(f"{vinaDir}/*") if os.path.isdir(d)]) < boxCount:
                octools.print_warning(f"The protein '{dir}' has not the same amount of vina conf files as the amount of box files. Trying to fix...")
                # If vina is needed, the input should be the prepared receptor
                preparedReceptor = f"{dir}/{ptn}_protein.pdbqt"

                boxPath = "" # TODO: Fix this

                # Create a lock for multithreading
                lock = Lock()
                # Start the lock with statement
                with lock:
                    # Run vina
                    ocvina.generate_vina_files_database(dir, fin, boxPath = boxPath)

                # If there is not the same amount of box files as folders in vinaFiles folder (again)
                if len([d for d in glob(f"{vinaDir}/*") if os.path.isdir(d)]) == boxCount:
                    octools.print_success(f"Conf files generated for '{dir}'.")
                else:
                    octools.print_error(f"Unable to generate the vina conf files for '{dir}'...")
                    octools.print_error_log(f"Unable to generate the vina conf files for '{dir}'...", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            # If there is not the same amount of box files as folders in plantsFiles folder
            if len([d for d in glob(f"{plantsDir}/*") if os.path.isdir(d)]) < boxCount:
                octools.print_warning(f"The protein '{dir}' has not the same amount of PLANTS conf files as the amount of box files. Trying to fix...")
                # If PLANTS is needed, the input should be the prepared receptor and ligand
                preparedReceptor = f"{dir}/{ptn}_protein_prepared.mol2"
                preparedLigand = f"{dir}/{ptn}_ligand_prepared.mol2"

                boxPath = "" # TODO: fix the path

                # Create a lock for multithreading
                lock = Lock()
                # Start the lock with statement
                with lock:
                    # Generate box files
                    ocplants.generate_plants_files_database(dir, fin, ligand, spacing, boxPath = boxPath)

                # If there is not the same amount of box files as folders in vinaFiles folder (again)
                if len([d for d in glob(f"{plantsDir}/*") if os.path.isdir(d)]):
                    octools.print_success(f"PLANTS conf files generated for '{dir}'.")
                else:
                    octools.print_error(f"Unable to generate the PLANTS conf files for '{dir}'...")
                    octools.print_error_log(f"Unable to generate the PLANTS conf files for '{dir}'...", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            # If is the pdbbind files
            if archive == "pdbbind":
                # If there is no descriptor file for the ligand or its size is 0
                if not os.path.isfile(f"{dir}/ligand_descriptors.json") or os.path.getsize(f"{dir}/ligand_descriptors.json") == 0:
                    # Generate it
                    ocprepare.prepare(paths, False, archive, sanitize = True, spacing = spacing)
                    __prepare_molecule(f"{dir}/{ptn}_ligand.mol2", False, "ligand", archive, sanitize = True)
                    # If the file still does not exists...
                    if not os.path.isfile(f"{dir}/ligand_descriptors.json") or os.path.getsize(f"{dir}/ligand_descriptors.json") == 0:
                        # REPORT
                        octools.print_error(f"Unable to generate the ligand descriptor file for '{dir}'...")
                        octools.print_error_log(f"Unable to generate the ligand descriptor file dir for '{dir}'...", f"{logdir}/{archive}_integrity_report.log")
                        failed = failed + 1
                        continue

                # If there is no descriptor file for the receptor or its size is 0
                if not os.path.isfile(f"{dir}/{ptn}_protein_descriptors.json") or os.path.getsize(f"{dir}/{ptn}_protein_descriptors.json") == 0:
                    # Generate it
                    ocprepare.prepare_molecule(f"{dir}/{ptn}_protein.pdb", False, "receptor", archive, sanitize = True)
                    # If the file still does not exists...
                    if not os.path.isfile(f"{dir}/{ptn}_protein_descriptors.json") or os.path.getsize(f"{dir}/{ptn}_protein_descriptors.json") == 0:
                        # REPORT
                        octools.print_error(f"Unable to generate the receptor descriptor file for '{dir}'...")
                        octools.print_error_log(f"Unable to generate the receptor descriptor file dir for '{dir}'...", f"{logdir}/{archive}_integrity_report.log")
                        failed = failed + 1
                        continue

    octools.printv(f"Integrity check of the PDBbind database accomplished. Success rate: {((lenDirs - failed) / lenDirs) * 100}% ({(lenDirs - failed)}/{lenDirs})")
    return None
"""

def prepare(archive: str, overwrite: bool = False, spacing: float = 0.33, sanitize: bool = True) -> None:
    '''Prepares the database.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind].
    overwrite : bool, optional
        If True overwrites the files, if False does not overwrite the files. The default is False.
    spacing : float, optional
        The spacing to be used in the grid. The default is 0.33.
    sanitize : bool, optional
        If True sanitizes the ligands, if False does not sanitize the ligands. The default is True.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Make archive lowercase
    archive = os.path.basename(archive).lower()

    # Find which kind of archive it will be
    if archive == "dudez":
        chosenArchive = dudez_archive
        label = f"DUDEz proteins"
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
        label = "PDBbind proteins"
        # Get all paths in the database filtering for pdbbind
    else:
        octools.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")
        return None

    # Get all paths in the database
    paths = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index']]

    # Generate boxes for all receptors
    octools.printv("Generating information regarding possible ligand site.")

    # Prepare it
    ocprepare.prepare(paths, overwrite, archive, sanitize, spacing)

    return None

def run_p2rank(archive: str, overwrite: bool = False) -> None:
    '''Runs P2Rank in the desired database.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind].
    overwrite : bool, optional
        If True overwrites the files, if False does not overwrite the files. The default is False.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Find which kind of archive it will be
    if archive.lower() == "dudez":
        chosenArchive = dudez_archive
    elif archive.lower() == "pdbbind":
        chosenArchive = pdbbind_archive
    else:
        octools.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")
        return None

    # Get all paths paths in the database
    paths = glob(f"{chosenArchive}/*")

    # Generate boxes for all receptors
    octools.printv("Generating P2Rank files.")
    # Run p2rank
    ocp2rank.run_p2rank(paths, overwrite)

    return None

def run_dock(archive: str, dockingAlgorithm: str, digestFormat: str = "json", overwrite: bool = False) -> int:
    '''Run docking.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind].
    dockingAlgorithm : str
        The docking algorithm to be used. The options are [vina, smina, plants].
    digestFormat : str, optional
        The format of the digest file. The options are [json]. The default is "json".
    overwrite : bool, optional
        If True overwrites the files, if False does not overwrite the files. The default is False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Make archive lowercase
    archive = os.path.basename(archive).lower()

    # TODO: add support to custom databases
    # Find which kind of archive it will be
    if archive == "dudez":
        chosenArchive = dudez_archive
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
    else:
        return errors.not_supported_archive(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")

    # TODO: add support to more docking algorithms
    # Check if the docking algorithm is valid
    if dockingAlgorithm not in ["gnina", "vina", "smina", "plants"]:
        return errors.not_supported_docking_algorithm(f"Docking software not recognized. Expected ('gnina', 'vina', 'smina', 'plants') and got '{dockingAlgorithm}'.")

    # Get all dirs paths in the database
    ptnDirs = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index']]

    # Create the complex list
    complexList = []
    
    # For each dir in dirs, let's grab all ligands
    for ptnDir in ptnDirs:
        # Parameterize paths
        ligands = f"{ptnDir}/compounds/ligands"
        decoys = f"{ptnDir}/compounds/decoys"
        candidates = f"{ptnDir}/compounds/candidates"

        # Append to the complex list the merged ligandAlternative list with the list with ligands, decoys and candidates. This is made because each receptor must have its own list of ligands, decoys and candidates, otherwise the docking could be done with the same ligands, decoys and candidates for all receptors making everything out of control.
        complexList.append((ptnDir, glob(f"{ligands}/*") + glob(f"{decoys}/*") + glob(f"{candidates}/*")))
    
    # Run docking
    return ocdock.run_dock(complexList, archive, dockingAlgorithm, overwrite, digestFormat)

def read_logs(archive: str, picklePath: str = "") -> Union[Dict[str, vdf.DataFrameLocal], None]:
    '''Reads database logfiles returning a dict of dicts of vdf.DataFrameLocal.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind].
    picklePath : str, optional
        The path to the pickle file. The default is "". If the picklePath is not empty, the function will write the data to the pickle file, otherwise will return the data.

    Returns
    -------
    Dict[str, vdf.DataFrameLocal] | None
        A dict of vdf.DataFrameLocal. If failed, returns None.

    Raises
    ------
    None
    '''

    # Make archive lowercase
    archive = os.path.basename(archive).lower()
    # Find which kind of archive it will be
    if archive == "dudez":
        chosenArchive = dudez_archive
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
    else:
        octools.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")
        return None
        
    # Create an empty list for all directories to be processed
    processDirs = []

    # For each dir in chosenArchive
    for ptnDir in glob(f"{chosenArchive}/*"):
        # Check if is a dir (just in case) and if its name is not one of the ones we want to skip
        if os.path.isdir(ptnDir) and os.path.basename(ptnDir.split(os.path.sep)[-1]) not in ['index']:
            ligands = f"{ptnDir}/compounds/ligands"
            decoys = f"{ptnDir}/compounds/decoys"
            candidates = f"{ptnDir}/compounds/candidates"

            # Add all subdirs (one for each ligand) from all 3 folders as a tuple (dir, type) to the processDirs list
            processDirs += [(processDir, "ligand") for processDir in glob(f"{ligands}/*") if os.path.isdir(processDir)]
            processDirs += [(processDir, "decoy") for processDir in glob(f"{decoys}/*") if os.path.isdir(processDir)]
            processDirs += [(processDir, "candidate") for processDir in glob(f"{candidates}/*") if os.path.isdir(processDir)]
        
    # Read the logs
    data = ocreadlogs.read_logs(processDirs, archive)

    # If user asked for a pickle file
    if picklePath:
        # Check if data is not empty
        if data:
            # Try to write it
            try:
                octools.to_pickle(picklePath, data)
                octools.print_success(f"The file '{picklePath}' has been successfully written.")
            except Exception as e:
                octools.print_error(f"Could not write the file '{picklePath}'. Error: {e}")
        else:
            octools.print_warning(f"The data object is not defined! There is no reason to write it as a pickle. Aborting...")
        # Return nothing
        return None
    # Return the data
    return data

def generate_dock_result_csv(archive: str, csv_path: str, log_dumps: Union[Dict[str, vdf.DataFrameLocal], None] = None) -> None:
    '''Uses the structure from read_logs to generate an output for all docking softwares.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind].
    csv_path : str
        The path to the csv file.
    log_dumps : Dict[str, vdf.DataFrameLocal] | None, optional
        The data from the logfiles. If None, will use the read_logs function to get the data. The default is None.

    Returns
    -------
    None

    Raises
    ------
    None
    '''
 
    # Check if log_dumps is None
    if not log_dumps:
        # Read the log files
        log_dumps = read_logs(archive)

    data = vaex.concat(list(log_dumps.values())) # type: ignore

    # Check if data is not empty
    if data:
        data.export_csv(path = csv_path, backend = 'arrow') # type: ignore

    return None

def merge_descriptors_in_dataframe(archive: str, readMode: str = "hdf5", saveMode: str = "hdf5", picklenize: bool = False, returnDf: bool = False, skipMergePicklePath: str = "", verboseOperations: bool = False) -> Union[vdf.DataFrameLocal, None]:
    '''Reads all the descriptors jsons and return a pd.DataFrame.

    Parameters
    ----------
    archive : str
        The archive to be prepared. Can be "hdf5" or "csv", by default "hdf5".
    saveMode : str, optional
        The mode to save the dataframe. Can be "hdf5", "csv" or "", by default "hdf5". If empty, will not save the dataframe.
    picklenize : bool, optional
        If True, will save the dataframe as a pickle file in different steps during the execution. The default is False.
    returnDf : bool, optional
        If True, will return the dataframe. The default is False.
    skipMergePicklePath : str, optional
        If not empty, will skip the merge and will try to read the pickle file. The default is "".
    verboseOperations : bool, optional
        If True, will print the operations being done. The default is False. This is useful for debugging.
    
    Returns
    -------
    vdf.DataFrameLocal | None
        The dataframe with all the descriptors.

    Raises
    ------
    None
    '''

    # Make archive lowercase
    archive = os.path.basename(archive).lower()

    # Find which kind of archive it will be
    if archive == "dudez":
        chosenArchive = dudez_archive
        # Parameterize the csvs paths
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
    else:
        octools.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")
        return None

    # Parameterize the out paths (parsed_archive is defined in Initialise.py)
    if saveMode.lower() == "hdf5":
        file_path_out = f"{parsed_archive}/{archive}_complete.hdf5"
    elif saveMode.lower() == "csv":
        file_path_out = f"{parsed_archive}/{archive}_complete.csv"
    elif saveMode == "":
        file_path_out = ""
    else:
        octools.print_error(f"Not valid save mode. Expected one of ['csv', 'hdf5', ''] and found {saveMode}.")
        return None
    
    # Parameterize the in paths (parsed_archive is defined in Initialise.py)
    if readMode.lower() == "hdf5":
        file_path_in = f"{parsed_archive}/{archive}.hdf5"
    elif readMode.lower() == "csv":
        file_path_in = f"{parsed_archive}/{archive}.csv"
    else:
        octools.print_error(f"Not valid read mode. Expected one of ['csv', 'hdf5'] and found {readMode}.")

        return None

    # If the user asked to skip the merge passing a pickle path    
    if not skipMergePicklePath:
        # Create an empty list for all directories to be processed
        processDirs = []

        # For each dir in chosenArchive
        for ptnDir in glob(f"{chosenArchive}/*"):
            # Check if is a dir (just in case) and if its name is not one of the ones we want to skip
            if os.path.isdir(ptnDir) and os.path.basename(ptnDir.split(os.path.sep)[-1]) not in ["index"]:
                # Parameterize paths
                ligands = f"{ptnDir}/compounds/ligands"
                decoys = f"{ptnDir}/compounds/decoys"
                candidates = f"{ptnDir}/compounds/candidates"

                # Parameterize the receptor descriptor path
                receptor_descriptor_path = f"{ptnDir}/receptor_descriptors.json"

                processDirs += [(processDir, receptor_descriptor_path) for processDir in glob(f"{ligands}/*") if os.path.isdir(processDir)]
                processDirs += [(processDir, receptor_descriptor_path) for processDir in glob(f"{decoys}/*") if os.path.isdir(processDir)]
                processDirs += [(processDir, receptor_descriptor_path) for processDir in glob(f"{candidates}/*") if os.path.isdir(processDir)]
        
        # Make data be None (in case of failure)
        data = None
        
        # Decide if multprocessing will be used
        if args.multiprocess:
            data = __merge_descriptors_in_dataframe_parallel(processDirs, f"Processing {archive}")
        else:
            data = __merge_descriptors_in_dataframe_no_parallel(processDirs, f"Processing {archive}")
    else:
        # Try to read the pickle
        try:
            data = octools.from_pickle(skipMergePicklePath)
        except:
            octools.print_error(f"Could not read the pickle file '{skipMergePicklePath}'.")
            return None

    # Check if data is pd.DataFrame type and is not empty
    if type(data) == vdf.DataFrameLocal: # type: ignore
        # Try to write the csv
        try:
            # If picklenize is true, save as pickle in this step
            if picklenize:
                octools.to_pickle(f"{parsed_archive}/{archive}_merged_descriptors.pickle", data)

            # Rename the name column from data dataframe
            #data.rename("Name", "Ligand") # type: ignore
            
            if args.output_level > 2 or verboseOperations:
                with vaex.progress.tree("rich", title="Merging dataframes"): # type: ignore
                    if readMode == "hdf5":
                        # Read the hdf5 from input file
                        ptndf = vaex.open(file_path_in)
                    else:
                        # Read the csv from input file
                        ptndf = vaex.read_csv(file_path_in)
            else:
                octools.print_info(f"Reading {file_path_in}...")
                if readMode == "hdf5":
                    # Read the csv from input file
                    ptndf = vaex.open(file_path_in)
                else:
                    # Read the csv from input file
                    ptndf = vaex.read_csv(file_path_in)

            # Generate and materialize the Complex column for ptndf and data from "Protein" and "Ligand" columns then drop them
            ptndf["Complex"] = ptndf["Protein"] + "-" + ptndf["Ligand"] # type: ignore
            _ = ptndf.materialize("Complex", inplace = True) # type: ignore
            ptndf = ptndf.drop(["Protein", "Ligand"]) # type: ignore

            data["Complex"] = data["Protein"] + "-" + data["Ligand"] # type: ignore
            _ = data.materialize("Complex", inplace = True) # type: ignore
            data = data.drop(["Protein", "Ligand", "Name"]) # type: ignore
            
            # If verbose
            if args.output_level > 2 or verboseOperations:
                with vaex.progress.tree("rich", title="Merging dataframes"): # type: ignore
                    # Merge both DataFrames using the Complex column as a comparer
                    data = ptndf.join(data, on = "Complex", how = "left") # type: ignore
            else:
                octools.print_info("Merging dataframes...")
                # Merge both DataFrames using the Protein column as a comparer
                data = ptndf.join(data, on = "Complex", how = "left") # type: ignore

            # Drop the poses columns since they are the same for all the rows (will be used when the support to multiple poses is added)
            data = data.drop(["vina_pose", "smina_pose", "gnina_pose"]) # TODO: Add support for multiple poses and remove this line
            
            # If saveCsv is True, save the csv
            if saveMode:
                if args.output_level > 2 or verboseOperations:
                    with vaex.progress.tree("rich", title="Saving dataframe"): # type: ignore
                        if saveMode == "hdf5":
                            # Write the data to a new hdf5 file
                            data.export_hdf5(file_path_out)
                        else:
                            # Write the data to a new csv file
                            data.export_csv(file_path_out, backend = "arrow")
                else:
                    octools.print_info(f"Writing the file '{file_path_out}'...")
                    if saveMode == "hdf5":
                        # Write the data to a new hdf5 file
                        data.export_hdf5(file_path_out)
                    else:
                        # Write the data to a new csv file
                        data.export_csv(file_path_out, backend = "arrow")

                octools.print_success(f"The file '{file_path_out}' has been successfully written.")

        except Exception as e:
            octools.print_error(f"Could not write the file '{file_path_out}'. Error: {e}")

            # Return Nothing
            return None
    else:
        octools.print_warning(f"The data object is not defined! There is no reason to write it. Aborting...")

        # Return nothing
        return None

    if returnDf:
        # Return the data
        return data

    return None

