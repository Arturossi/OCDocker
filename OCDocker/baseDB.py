#!/usr/lib/python3

# Description
###############################################################################
'''
Sets of classes and functions that are used as base for all databases. It
contains functions that are common to all databases.

They are imported as:

import OCDocker.baseDB as ocbdb
'''

# Imports
###############################################################################
import os
import vaex

import vaex.dataframe as vdf

from glob import glob
from typing import Dict, Union

from OCDocker.Initialise import *

import OCDocker.Processing.Dock as ocdock
import OCDocker.Processing.Postprocessing.Digest as ocdigest
import OCDocker.Processing.Preprocessing.Prepare as ocprepare
import OCDocker.Processing.Preprocessing.p2rank as ocp2rank
import OCDocker.Processing.Postprocessing.ReadLogs as ocreadlogs
import OCDocker.Processing.Postprocessing.MergeLogs as ocmergelogs

import OCDocker.Toolbox.FilesFolders as ocff
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
    ocprint.printv(f"Verifiying the integrity of the {chosenArchive} database")

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
                ocprint.print_error(f"Unknown archive type, expected one of the following ['dudez', 'pdbbind'] and got '{archive}'.")
                return

            ocprint.printv(f"Checking directories for the protein '{dir}'.")

            # If has no p2rank dir
            if not os.path.isdir(p2rankDir):
                ocprint.print_warning(f"The protein '{dir}' has no p2rank folder. Trying to fix...")

                # Create the p2rank output dir
                errorCode = octools.safe_create_dir(p2rankDir)

                if os.path.isdir(p2rankDir):
                    ocprint.print_success(f"The p2rank dir has been generated for '{dir}'.")
                else:
                    ocprint.print_error(f"Unable to generate the p2rank dir for '{dir}'... Error code {errorCode}.")
                    ocprint.print_error_log(f"Unable to generate the p2rank dir for '{dir}'... Error code {errorCode}.", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            # If has no vinaFiles dir
            if not os.path.isdir(vinaDir):
                ocprint.print_warning(f"The protein '{dir}' has no vinaFiles folder. Trying to fix...")

                # Create the p2rank output dir
                errorCode = octools.safe_create_dir(vinaDir)

                if os.path.isdir(vinaDir):
                    ocprint.print_success(f"The vinaFiles dir has been generated for '{dir}'.")
                else:
                    ocprint.print_error(f"Unable to generate the vinaFiles dir for '{dir}'... Error code {errorCode}.")
                    ocprint.print_error_log(f"Unable to generate the vinaFiles dir for '{dir}'... Error code {errorCode}.", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            # If has no plantsFiles dir
            if not os.path.isdir(plantsDir):
                ocprint.print_warning(f"The protein '{dir}' has no plantsFiles folder. Trying to fix...")

                # Create the p2rank output dir
                errorCode = octools.safe_create_dir(plantsDir)

                if os.path.isdir(plantsDir):
                    ocprint.print_success(f"The plantsFiles dir has been generated for '{dir}'.")
                else:
                    ocprint.print_error(f"Unable to generate the plantsFiles dir for '{dir}'... Error code {errorCode}.")
                    ocprint.print_error_log(f"Unable to generate the plantsFiles dir for '{dir}'... Error code {errorCode}.", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            ocprint.printv(f"Checking files for the protein '{dir}'")

            # Check how many boxes are in the p2rankDir
            boxes = glob(f"{p2rankDir}/box*.pdb")
            boxCount = len(boxes)

            # If there is no box in the p2rank output, p2rank will run
            if boxCount == 0:
                ocprint.print_warning(f"The protein '{dir}' has no box file. Trying to fix...")

                # Run p2rank
                ocp2rank.p2rank_no_parallel([dir], False, desc = "Running p2rank")

                # Check how many boxes are in the p2rankDir (again)
                boxes = glob(f"{p2rankDir}/box*.pdb")
                boxCount = len(boxes)

                if boxCount > 0:
                    ocprint.print_success(f"Box files generated for '{dir}'.")
                else:
                    ocprint.print_error(f"The protein '{dir}' still has no box file.")
                    ocprint.print_error_log(f"The protein '{dir}' still has no box file.", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            # If there is not the same amount of box files as folders in vinaFiles folder
            if len([d for d in glob(f"{vinaDir}/*") if os.path.isdir(d)]) < boxCount:
                ocprint.print_warning(f"The protein '{dir}' has not the same amount of vina conf files as the amount of box files. Trying to fix...")
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
                    ocprint.print_success(f"Conf files generated for '{dir}'.")
                else:
                    ocprint.print_error(f"Unable to generate the vina conf files for '{dir}'...")
                    ocprint.print_error_log(f"Unable to generate the vina conf files for '{dir}'...", f"{logdir}/{archive}_integrity_report.log")
                    failed = failed + 1
                    continue

            # If there is not the same amount of box files as folders in plantsFiles folder
            if len([d for d in glob(f"{plantsDir}/*") if os.path.isdir(d)]) < boxCount:
                ocprint.print_warning(f"The protein '{dir}' has not the same amount of PLANTS conf files as the amount of box files. Trying to fix...")
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
                    ocprint.print_success(f"PLANTS conf files generated for '{dir}'.")
                else:
                    ocprint.print_error(f"Unable to generate the PLANTS conf files for '{dir}'...")
                    ocprint.print_error_log(f"Unable to generate the PLANTS conf files for '{dir}'...", f"{logdir}/{archive}_integrity_report.log")
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
                        ocprint.print_error(f"Unable to generate the ligand descriptor file for '{dir}'...")
                        ocprint.print_error_log(f"Unable to generate the ligand descriptor file dir for '{dir}'...", f"{logdir}/{archive}_integrity_report.log")
                        failed = failed + 1
                        continue

                # If there is no descriptor file for the receptor or its size is 0
                if not os.path.isfile(f"{dir}/{ptn}_protein_descriptors.json") or os.path.getsize(f"{dir}/{ptn}_protein_descriptors.json") == 0:
                    # Generate it
                    ocprepare.prepare_molecule(f"{dir}/{ptn}_protein.pdb", False, "receptor", archive, sanitize = True)
                    # If the file still does not exists...
                    if not os.path.isfile(f"{dir}/{ptn}_protein_descriptors.json") or os.path.getsize(f"{dir}/{ptn}_protein_descriptors.json") == 0:
                        # REPORT
                        ocprint.print_error(f"Unable to generate the receptor descriptor file for '{dir}'...")
                        ocprint.print_error_log(f"Unable to generate the receptor descriptor file dir for '{dir}'...", f"{logdir}/{archive}_integrity_report.log")
                        failed = failed + 1
                        continue

    ocprint.printv(f"Integrity check of the PDBbind database accomplished. Success rate: {((lenDirs - failed) / lenDirs) * 100}% ({(lenDirs - failed)}/{lenDirs})")
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

    # Find which kind of archive it will be
    if archive.lower() == "dudez":
        chosenArchive = dudez_archive
    elif archive.lower() == "pdbbind":
        chosenArchive = pdbbind_archive
    else:
        ocprint.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")
        return None

    # Get all paths in the database
    paths = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index']]

    # Generate boxes for all receptors
    ocprint.printv("Generating information regarding possible ligand site.")

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
        ocprint.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")
        return None

    # Get all paths paths in the database
    paths = glob(f"{chosenArchive}/*")

    # Generate boxes for all receptors
    ocprint.printv("Generating P2Rank files.")
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

def read_logs(archive: str, saveChunk: int = 100, overwrite: bool = False) -> Union[Dict[str, vdf.DataFrameLocal], None]:
    '''Reads database logfiles returning a dict of dicts of vdf.DataFrameLocal.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind].
    saveChunk : int, optional
        The number of lines to be read before saving the data. The default is 100.
    overwrite : bool, optional
        If True overwrites the files, if False does not overwrite the files. The default is False.

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
        ocprint.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")
        return None
    
    data = {}
        
    # For each dir in chosenArchive
    for ptnDir in glob(f"{chosenArchive}/*"):
        # Create an empty list for all directories to be processed
        processDirs = []

        # Check if is a dir (just in case) and if its name is not one of the ones we want to skip
        if os.path.isdir(ptnDir) and os.path.basename(ptnDir.split(os.path.sep)[-1]) not in ['index']:
            ligands = f"{ptnDir}/compounds/ligands"
            decoys = f"{ptnDir}/compounds/decoys"
            candidates = f"{ptnDir}/compounds/candidates"

            # Add all subdirs (one for each ligand) from all 3 folders as a tuple (dir, type) to the processDirs list
            processDirs += [(processDir, "ligand") for processDir in glob(f"{ligands}/*") if os.path.isdir(processDir)]
            processDirs += [(processDir, "decoy") for processDir in glob(f"{decoys}/*") if os.path.isdir(processDir)]
            processDirs += [(processDir, "candidate") for processDir in glob(f"{candidates}/*") if os.path.isdir(processDir)]
        
        # Get the protein name out from the path
        proteinName = os.path.basename(ptnDir)
        
        # Read the logs and concatenate the results with data
        innerData = ocreadlogs.read_logs(processDirs, archive, proteinName, saveChunk = saveChunk, overwrite = overwrite)

        # Check if data is not empty
        if innerData:
            # Merge innerData into data
            data = {**data, **innerData}
        else:
            ocprint.print_warning(f"The data object is not defined! There is no reason to append it to data list. Skipping...")
            continue

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

def merge_descriptors_in_dataframe(archive: str, readMode: str = "hdf5", saveMode: str = "hdf5", picklenize: bool = False, returnDf: bool = False, skipMergePicklePath: str = "", saveChunk: int = 100, datafileFormat: str = "hdf5", verboseOperations: bool = False, overwrite: bool = False) -> Union[vdf.DataFrameLocal, None]:
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
    saveChunk : int, optional
        The chunk size to save the dataframe. The default is 100.
    datafileFormat : str, optional
        The format of the data file. The default is "hdf5". TODO: Add csv support.
    verboseOperations : bool, optional
        If True, will print the operations being done. The default is False. This is useful for debugging.
    overwrite : bool, optional
        If True, will overwrite the database files. The default is False.
    
    Returns
    -------
    vdf.DataFrameLocal | None
        The dataframe with all the descriptors.

    Raises
    ------
    None
    '''

    # Find which kind of archive it will be
    if archive.lower() == "dudez":
        chosenArchive = dudez_archive
        # Parameterize the csvs paths
    elif archive.lower() == "pdbbind":
        chosenArchive = pdbbind_archive
    else:
        ocprint.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found '{archive}'.")
        return None

    # Parameterize the out paths (parsed_archive is defined in Initialise.py)
    if saveMode.lower() == "hdf5":
        file_path_out = f"{parsed_archive}/{archive}_complete.hdf5"
    elif saveMode.lower() == "csv":
        file_path_out = f"{parsed_archive}/{archive}_complete.csv"
    elif saveMode == "":
        file_path_out = ""
    else:
        ocprint.print_error(f"Not valid save mode. Expected one of ['csv', 'hdf5', ''] and found {saveMode}.")
        return None
    
    # Parameterize the in paths (parsed_archive is defined in Initialise.py)
    if readMode.lower() == "hdf5":
        file_path_in = f"{parsed_archive}/{archive}.hdf5"
    elif readMode.lower() == "csv":
        file_path_in = f"{parsed_archive}/{archive}.csv"
    else:
        ocprint.print_error(f"Not valid read mode. Expected one of ['csv', 'hdf5'] and found {readMode}.")
        return None

    # Create an empty list for all directories to be processed
    processDirs = []

    # Create the data list
    data = []

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

        # Extract the ptn name
        ptn = os.path.basename(ptnDir.split(os.path.sep)[-1])

        # Merge the descriptors and append its results to the data list
        data.append(ocmergelogs.merge_descriptors_in_dataframe(processDirs, file_path_out, ptn, archive, saveChunk = saveChunk, datafileFormat = datafileFormat, overwrite = overwrite))

        # Merge the list elements into a single vaex df
        data = vaex.concat(data)

    # Check if data is pd.DataFrame type and is not empty
    if type(data) == vdf.DataFrameLocal: # type: ignore
        # Try to write the csv
        try:
            # If picklenize is true, save as pickle in this step
            if picklenize:
                ocff.to_pickle(f"{parsed_archive}/{archive}_merged_descriptors.pickle", data) # type: ignore

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
                ocprint.print_info(f"Reading {file_path_in}...")
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
                ocprint.print_info("Merging dataframes...")
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
                    ocprint.print_info(f"Writing the file '{file_path_out}'...")
                    if saveMode == "hdf5":
                        # Write the data to a new hdf5 file
                        data.export_hdf5(file_path_out)
                    else:
                        # Write the data to a new csv file
                        data.export_csv(file_path_out, backend = "arrow")

                ocprint.print_success(f"The file '{file_path_out}' has been successfully written.")

        except Exception as e:
            ocprint.print_error(f"Could not write the file '{file_path_out}'. Error: {e}")

            # Return Nothing
            return None
    else:
        ocprint.print_warning(f"The data object is not defined! There is no reason to write it. Aborting...")

        # Return nothing
        return None

    if returnDf:
        # Return the data
        return data

    return None

