#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to update the OCDocker database.

They are imported as:

import OCDocker.Database as ocdatabase
'''

# Imports
###############################################################################
import gc
import os
import mimetypes
import shutil

import textwrap as tw

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from threading import Lock
from typing import List, Tuple

from OCDocker.Initialise import *

import OCDocker.DB.DUDEz as ocdudez
import OCDocker.DB.PDBbind as ocpdbbind
import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.Downloading as ocdown
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
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

#### Process DUDEz
def __core_process_dudez(target: str, overwrite: bool) -> None:
    '''Core function to process the DUDEz database.

    Parameters
    ----------
    target : str
        The path where the files are.
    overwrite : bool
        Flag to tell if files should be overwritten.

    Returns
    -------
    None
    '''

    # Get the target name
    target_name = os.path.basename(target)
    # Process the ligands
    ocprint.printv(f"Processing the ligands for {target_name}")
    # Parameterize the compounds path
    targetc = os.path.join(target, "compounds")
    # Create the compound folder (will hold all compounds, no matter if they are ligand or decoy)
    _ = ocff.safe_create_dir(targetc)
    # List to hold the tuples for each processing that will be made
    process_list = ["ligands", "decoys"]
    # For each data
    for data in process_list:
        # Print which file is being processed
        ocprint.printv(f"Processing {target}/{data}.smi")
        # Create the ligands folder
        _ = ocff.safe_create_dir(f"{targetc}/{data}")
        # Create a lock for multithreading
        lock = Lock()
        # Start the lock with statement
        with lock:
            # Process the ligands, splitting them into the multiple files
            with open(f"{target}/{data}.smi", 'r') as f:
                for line in f:
                    # Get the smiles and name of the ligand
                    smiles, name = line.split()
                    # Check if there is already a folder with the ligand name (to warn the user)
                    if os.path.isdir(f"{targetc}/{data}/{name}"):
                        ocprint.print_warning(f"The ligand {name} already exists in the {data[0]} dataser. You may not need to process the {data[1]}.smi file again. By the way... I am just warning you.")

                    # Create the ligand folder using its name
                    _ = ocff.safe_create_dir(f"{targetc}/{data}/{name}")
                    
                    # Test if the file exists
                    if overwrite or not os.path.isfile(f"{targetc}/{data}/{name}/ligand.mol2"):
                        # Check if the outputfile exists
                        if os.path.isfile(f"{targetc}/{data}/{name}/ligand.mol2"):
                            # Remove the file
                            os.remove(f"{targetc}/{data}/{name}/ligand.mol2")
                        # Convert it to mol2 (NOTE: There are many molecules with SAME name... currently I am not handling this. I am just accounting the first molecule and discarding the others. IMPORTANT: Error messages WILL pop while processing the data here! They may be safe to ignore, I guess...)
                        _ = occonversion.convertMolsFromString(smiles, f"{targetc}/{data}/{name}/ligand.mol2")
                        # Save a smiles file (to avoid compatibility issues)
                        with open(f"{targetc}/{data}/{name}/ligand.smi", 'w') as f:
                            f.write(f"{smiles}")
                    else:
                        ocprint.print_warning(f"File '{targetc}/{data}/{name}/ligand.mol2' already exists. Skipping...")

    return None

def __thread_process_dudez(arguments: Tuple[str, bool]) -> None:
    '''Thread aid function to call __core_process_dudez.

    Parameters
    ----------
    arguments : Tuple[str, bool]
        The arguments to be passed to __core_process_dudez.

    Returns
    -------
    None
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Call core prepare function (shared between thread and no thread)
        return __core_process_dudez(arguments[0], arguments[1])

def __process_dudez_parallel(targets: List[str], overwrite: bool, desc: str) -> None:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    targets : List[str]
        The list of directories to be processed.
    overwrite : bool
        Flag to tell if files should be overwritten.
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for target in targets:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((target, overwrite))
    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(available_cores) as p:
            # Perform the multi process
            for _ in tqdm(p.imap_unordered(__thread_process_dudez, arguments), total = len(arguments), desc = desc):
                # Clear the memory
                gc.collect()
    except IOError as e:
        ocprint.print_error(f"Problem while processing DUDEz in parallel. Exception: {e}")

    return None

def __process_dudez_no_parallel(targets: List[str], overwrite: bool, desc: str) -> None:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_process_dudez function.

    Parameters
    ----------
    targets : List[str]
        The list of directories to be processed.
    overwrite : bool
        Flag to tell if files should be overwritten.
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    None
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        for target in tqdm(iterable=targets, total=len(targets), desc=desc):
            # Call the core prepare function
            __core_process_dudez(target, overwrite)
            # Clear the memory
            gc.collect()
    return None

#### Download DUDEz
def __core_download_dudez(target: str, overwrite: bool) -> None:
    '''Downloads the DUDEz database.

    Parameters
    ----------
    target : str
        The target directory to download the database.
    overwrite : bool
        Flag to tell if files should be overwritten.

    Returns
    -------
    None
    '''

    # Trying to fix dudez lazy webmasters mistakes
    if target == "D4":
        target2 = "DRD4"
    else:
        target2 = target

    # Create a folder for the target in the archive
    _ = ocff.safe_create_dir(f"{dudez_archive}/{target2}")

    # Check if the target receptor does not exists or the user wants to overwrite it
    if not os.path.isfile(f"{dudez_archive}/{target2}/receptor.pdb") or overwrite:
        # Download the target receptor
        ocdown.download_url(f"{dudez_download}/DOCKING_GRIDS_AND_POSES/{target2}/rec.crg.pdb", f"{dudez_archive}/{target2}/receptor.pdb")

    # Check if the reference ligand does not exists or the user wants to overwrite it
    if not os.path.isfile(f"{dudez_archive}/{target2}/ligand.mol2") or overwrite:
        # Download the target receptor
        ocdown.download_url(f"{dudez_download}/DOCKING_GRIDS_AND_POSES/{target2}/xtal-lig.pdb", f"{dudez_archive}/{target2}/reference_ligand.pdb")

    # Check if the target dudez ligands does not exists or the user wants to overwrite it
    if not os.path.isfile(f"{dudez_archive}/{target2}/ligands.smi") or overwrite:
        # Download the dudeZ ligands
        ocdown.download_url(f"{dudez_download}/DUDE-Z-benchmark-grids/{target}/ligands.smi", f"{dudez_archive}/{target2}/ligands.smi")

    # Check if the target dudez decoys does not exists or the user wants to overwrite it
    if not os.path.isfile(f"{dudez_archive}/{target2}/decoys.smi") or overwrite:
        # Download the dudeZ ligands
        ocdown.download_url(f"{dudez_download}/DUDE-Z-benchmark-grids/{target}/decoys.smi", f"{dudez_archive}/{target2}/decoys.smi")

    # Download the Extrema set TODO: Currently not used
    #ocdown.download_url(f"{dudez_download}/extrema/{target}/minus2/{target}_minus2.smi", f"{dudez_archive}/{target2}/extrema_minus2.smi")
    #ocdown.download_url(f"{dudez_download}/extrema/{target}/minus1/{target}_minus1.smi", f"{dudez_archive}/{target2}/extrema_minus1.smi")
    #ocdown.download_url(f"{dudez_download}/extrema/{target}/neutral/{target}_neutral.smi", f"{dudez_archive}/{target2}/extrema_neutral.smi")
    #ocdown.download_url(f"{dudez_download}/extrema/{target}/plus1/{target}_plus1.smi", f"{dudez_archive}/{target2}/extrema_plus1.smi")
    #ocdown.download_url(f"{dudez_download}/extrema/{target}/plus2/{target}_plus2.smi", f"{dudez_archive}/{target2}/extrema_plus2.smi")

    return None

def __thread_download_dudez(arguments: Tuple[str, bool]) -> None:
    '''Thread aid function to call __core_download_dudez.

    Parameters
    ----------
    arguments : Tuple[str, bool]
        The arguments to be passed to __core_download_dudez.

    Returns
    -------
    None
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Call core prepare function (shared between thread and no thread)
        return __core_download_dudez(arguments[0], arguments[1])

def __download_dudez_parallel(targets: List[str], overwrite: bool, desc: str) -> None:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    targets : List[str]
        The list of directories to be processed.
    overwrite : bool
        Flag to tell if files should be overwritten.
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for target in targets:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((target, overwrite))
    # Create a Thread pool with the maximum available_cores
    with Pool(available_cores) as p:
        # Perform the multi process
        for _ in tqdm(p.imap_unordered(__thread_download_dudez, arguments), total = len(arguments), desc = desc):
            # Clear the memory
            gc.collect()

    return None

def __download_dudez_no_parallel(targets: List[str], overwrite: bool, desc: str) -> None:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_download_dudez function.

    Parameters
    ----------
    targets : List[str]
        The list of directories to be processed.
    overwrite : bool
        Flag to tell if files should be overwritten.
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    None
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        for target in tqdm(iterable=targets, total=len(targets), desc=desc):
            # Call the core prepare function
            __core_download_dudez(target, overwrite)
            # Clear the memory
            gc.collect()

    return None

## Public ##
def create_directories() -> None:
    '''Create necessary dirs.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    # Create the base dir
    _ = ocff.safe_create_dir(ocdb_path)
    # Create the pdbbind dir
    _ = ocff.safe_create_dir(pdbbind_archive)
    # Create the dudez dir
    _ = ocff.safe_create_dir(dudez_archive)
    # Create the Parsed dir
    _ = ocff.safe_create_dir(parsed_archive)

    return None

def update_DUDEz(overwrite: bool = False, download: bool = True, multiprocess: bool = True) -> int:
    '''Updates the DUDE-Z database.

    Parameters
    ----------
    overwrite : bool
        Flag to tell if files should be overwritten.
    download : bool
        Flag to tell if the database should be downloaded.
    multiprocess : bool
        Flag to tell if the download should be done in parallel.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # If mimetypes are not inited yet
    if mimetypes.inited == False:
        # Init mimetypes
        mimetypes.init()

    # Create tmp dir for download
    _ = ocff.safe_create_dir("./tmp")

    ocprint.printv("Downloading the DUDE-Z database")

    # Download the benchmark grids indexes
    ocdown.download_url(f"{dudez_download}/DUDE-Z-benchmark-grids/DUDE-Z_targets", f"{tmpDir}/DUDE-Z_targets")

    # Initialise an empty list to store the targets
    targets = []
    # Read the targets into a list
    with open(f"{tmpDir}/DUDE-Z_targets", 'r') as f:
        targets = f.read().splitlines()

    # Check if the target list is empty
    if len(targets) == 0:
        return ocerror.Error.file_not_exist("The target list is empty. Something went wrong with the download.", ocerror.ReportLevel.ERROR) # type: ignore

    # If the download flag is set
    if download:
        # Download all sets
        ocprint.printv("Downloading the datasets.")

        # Check multiprocessing is enabled
        if multiprocess:
            # Call the multiprocessing function
            __download_dudez_parallel(targets, overwrite, "Downloading DUDE-Z database")
        else:
            # Call the single process function
            __download_dudez_no_parallel(targets, overwrite, "Downloading DUDE-Z database")

    # Process each target
    targets = [d for d in glob(f"{dudez_archive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['goldilocks', 'tmp']]

    # Rename each protein protein file from rec.crg.pdb to {name}_protein.pdb
    for target in tqdm(targets, desc="Renaming protein files"):
        # Get the target name
        name = os.path.basename(target)
        # Get the protein file
        protein = glob(f"{target}/rec.crg.pdb")
        # If the protein file exists
        if len(protein) > 0:
            # Rename the file
            os.rename(protein[0], f"{target}/{name}_protein.pdb")

    # Check multiprocessing is enabled
    if multiprocess:
        # Call the multiprocessing function NOTE: the extrema files are not being download for now
        __process_dudez_parallel(targets, overwrite, "Processing DUDE-Z database")
    else:
        # Call the single process function NOTE: the extrema files are not being download for now
        __process_dudez_no_parallel(targets, overwrite, "Processing DUDE-Z database")

    # Delete the downloaded file
    ocprint.printv("Deleting the downloaded file.")
    os.remove(f"{tmpDir}/DUDE-Z_targets")

    # Prepare the DUDEz database
    ocdudez.prepare()

    return ocerror.Error.ok() # type: ignore

def update_PDBbind(overwrite: bool = False, deleteTar: bool = True, silentMode: str = "") -> int:
    '''Updates the PDBbind database from the Protein-ligand complexes: The refined set.

    Parameters
    ----------
    deleteTar : bool, optional
        Flag to tell if the tar files should be deleted after extraction. The default is True.
    silentMode : str, optional
        Flag to tell the behaviour of the function. The default is "", which means no silent. Options are "continue", "skip" and "".   

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # If mimetypes are not inited yet
    if mimetypes.inited == False:
        # Init mimetypes
        mimetypes.init()
        
    # Parameterizing the topics (this sounds strange but one large string concatenation was bugging the IDE)
    t1 = f"- Go to the PDBbind website ({clrs['c']}http://www.pdbbind.org.cn/download.php{clrs['n']})."

    #t2 = f"- Download the{clrs['c']} Protein-ligand complexes: The refined set{clrs['n']} (it may have the number 3 as its index), untar it and put all the protein folders folder inside the{clrs['y']} {pdbbind_archive}{clrs['n']} folder."
    #t2 += f" The folders{clrs['y']} readme{clrs['n']} and{clrs['y']} index{clrs['n']} should be{clrs['r']} deleted{clrs['n']}."

    t2 = f"- Download the{clrs['c']} Protein-ligand complexes: The refined set{clrs['n']} (it may have the number 3 as its index)."

    t3 = f"- Then provide the full path to it or put the file inside the{clrs['y']} {pdbbind_archive}{clrs['n']} folder and type continue (please, make sure that the downloaded file is the{clrs['c']} ONLY{clrs['n']} file inside the{clrs['y']} {pdbbind_archive}{clrs['n']} folder). If you want to skip the PDBbind update, type 'skip' (without quotes) and press enter. "

    # If silent mode is off
    if silentMode == "":
        # Since no rsync option to update pdbbind database has been found you have to manually download/untar the files and put them inside the database folder
        print(tw.dedent("""
                Unfortunately this step has not been able to be completely automatized... :(
        Please, we kindly ask you to perform the following steps to update the PDBbind database

        """ + t1 + """

        """ + t2 + """

        """ + t3 + """

        """))

    # Infinite loop (user can break it by sending an empty answer)
    while True:
        # If silent mode is on
        if silentMode:
            # Set the option to continue
            opt = silentMode
        else:
            # Check the options
            opt = input("Once these steps are done, type 'continue' and press enter to continue. To skip, type 'skip' To cancel just press enter without typing nothing.\n")

            # If there is quotes or double quotes in the path
            if "'" in opt or '"' in opt:
                # Remove them
                opt = opt.replace('"', "").replace("'", "")

        # If the option in lowercase is in the continue list (traductions may enter here)
        if opt.lower() in ["continue", "continuar"]:
            ocprint.printv("Continuing the update proces...")
            # Find the pdbbindTar file
            pdbbindTar = glob(f"{pdbbind_archive}/*.tar.gz")[0]

            # Since everything is right, start to untar/ungz them and delete source .tar.gz file
            _ = ocff.untar(pdbbindTar, out_path = f"{pdbbind_archive}", delete = deleteTar)

            # Check if there is a refined-set folder
            if os.path.isdir(f"{pdbbind_archive}/refined-set"):
                # For each file inside the refined-set folder
                for filename in os.listdir(os.path.join(pdbbind_archive, "refined-set")):
                    # Remove unwanted dirs here
                    if filename in ["readme"]:
                        # Skip it
                        continue

                    # Parameterize the destination path
                    destPath = f"{pdbbind_archive}/{filename}"

                    # Move it to the parent folder
                    shutil.move(f"{pdbbind_archive}/refined-set/{filename}", destPath)

                    # If is not index (it is a special folder)
                    if filename != "index":
                        # Create the compounds folder inside the protein folder
                        _ = ocff.safe_create_dir(f"{destPath}/compounds")
                        # Create the ligands folder inside the compounds folder (PDBbind only has one ligand per protein)
                        _ = ocff.safe_create_dir(f"{destPath}/compounds/ligands")
                        # Create the ligand folder inside the ligands folder (yes, generic name until I find a better one)
                        _ = ocff.safe_create_dir(f"{destPath}/compounds/ligands/ligand")
                        # Create the boxes folder inside the ligand folder
                        _ = ocff.safe_create_dir(f"{destPath}/compounds/ligands/ligand/boxes")
                        
                        # Make a copy of the ligands to serve as reference and then move the ligand files to the ligands folder (mol2 and sdf)
                        shutil.copy(f"{destPath}/{filename}_ligand.mol2", f"{destPath}/reference_ligand.mol2")
                        shutil.copy(f"{destPath}/{filename}_ligand.sdf", f"{destPath}/reference_ligand.sdf")
                        shutil.move(f"{destPath}/{filename}_ligand.mol2", f"{destPath}/compounds/ligands/ligand/ligand.mol2")
                        shutil.move(f"{destPath}/{filename}_ligand.sdf", f"{destPath}/compounds/ligands/ligand/ligand.sdf")

                        # Rename the protein file
                        shutil.move(f"{destPath}/{filename}_protein.pdb", f"{destPath}/receptor.pdb")

                        _ = ocmolproc.make_only_ATOM_and_CRYST_pdb(f"{destPath}/receptor.pdb")

                        # Remove all the unwanted files
                        unwanteds = [("pocket", "pdb")]
                        for unwanted in unwanteds:
                            # If the file exists
                            if os.path.isfile(f"{destPath}/{filename}_{unwanted[0]}.{unwanted[1]}"):
                                # Remove it
                                os.remove(f"{destPath}/{filename}_{unwanted[0]}.{unwanted[1]}")

                # Remove the refined-set folder
                shutil.rmtree(f"{pdbbind_archive}/refined-set")

            # Check if there is a readme file
            if os.path.isfile(f"{pdbbind_archive}/README.txt"):
                # Delete it
                os.remove(f"{pdbbind_archive}/README.txt")

            # Exit the loop
            break

        elif opt.lower() in ["skip", "pular"]:
            ocprint.printv(f"The user decided to skip this update. Skipping!!!")
            return ocerror.Error.ok() # type: ignore

        elif opt == "":
            rcode = ocerror.Error.abort("User aborted the update.") # type: ignore
            quit(rcode)

        else:
            ocprint.printv(f"Please use a valid answer!")
            ocprint.printv("- 'continue': To continue.")
            ocprint.printv("- 'skip':     To skip.")
            ocprint.printv("- '':         To quit.")
            continue

    # Prepare the PDBbind database
    ocpdbbind.prepare()

    return ocerror.Error.ok() # type: ignore

def update_databases() -> None:
    '''Calls all the database update functions sequentially.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    
    # Start the mimetypes
    mimetypes.init()

    print("\n\nUpdating ALL databases.\n")
    create_directories()

    print("Updating PDBbind database...")
    _ = update_PDBbind()
    print("\n\nDone updating PDBbind!\n")

    print("Updating DUDEz database...")
    _ = update_DUDEz(overwrite = overwrite, multiprocess = multiprocess)
    print("\n\nDone updating DUDEz!\n")

    print("\n\nDone updating ALL databases.\n")

    return None
