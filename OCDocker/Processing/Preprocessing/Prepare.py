#!/usr/bin/env python3

# Description
###############################################################################
'''
This module is responsible for digest processing.

It is imported as:

import OCDocker.Processing.Preprocessing.Prepare as ocprepare
'''

# Imports
###############################################################################
import gc
import os
import rdkit
import shutil

from glob import glob
from multiprocessing import Pool
from threading import Lock
from tqdm import tqdm
from typing import List, Tuple, Union

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Docking.Future.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##


def __prepare_molecule(mol: rdkit.Chem.rdchem.Mol, overwrite: bool, moltype: str, dbName: str, sanitize: bool, molName: str = "", targetCentroid: Union[Tuple[float, float, float], rdkit.Geometry.rdGeometry.Point3D] = None, alternativeLigand: rdkit.Chem.rdchem.Mol = None) -> None: # type: ignore
    '''Prepares a molecule, generating output to docking software.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to be prepared.
    overwrite : bool
        Flag for demanding file overwrite.
    moltype : str
        Type of the molecule to be prepared.
    dbName : str
        Name of the database.
    sanitize : bool
        Flag for demanding molecule sanitization.
    molName : str, optional
        Name of the molecule.
    targetCentroid : Tuple[float, float, float] | rdkit.Geometry.rdGeometry.Point3D, optional
        Centroid of the target. If not provided, the centroid of the molecule will be used.
    alternativeLigand : rdkit.Chem.rdchem.Mol, optional
        Alternative ligand to be used in the preparation.

    Returns
    -------
    None
    '''

    # Find its name and path
    if type(mol) == tuple:
        molPath = os.path.split(mol[0])[0]
    else:
        molPath = os.path.split(mol)[0]

    # Check if the molName was provided
    if molName == "":
        # Set the molname as the molType
        molName = moltype
    
    if overwrite or not os.path.isfile(f"{molPath}/{moltype}_descriptors.json"):
        if moltype == "ligand":
            # Safe create dockingFiles dirs
            _ = ocff.safe_create_dir(f"{molPath}/plantsFiles")
            _ = ocff.safe_create_dir(f"{molPath}/vinaFiles")
            _ = ocff.safe_create_dir(f"{molPath}/sminaFiles")
            _ = ocff.safe_create_dir(f"{molPath}/gninaFiles")

            try:
                # Create a lock for multithreading
                lock = Lock()
                # Start the lock with statement
                with lock:
                    # Create the ligand object
                    m = ocl.Ligand(mol, molName, sanitize = sanitize)
                    # Test if the Radius of Gyration is None
                    if not m.RadiusOfGyration: # type: ignore
                        # Print a warning
                        ocprint.print_warning(f"The ligand '{molName}' has a Radius of Gyration of None, trying to load its alternative ligand.")
                        # If so, try to load the alternative ligand
                        if alternativeLigand:
                            # Create the ligand object
                            m = ocl.Ligand(alternativeLigand, molName, sanitize = sanitize)
                            # Check the radius of gyration again
                            if not m.RadiusOfGyration: # type: ignore
                                # If it is still None, print a warning and return
                                ocprint.print_warning(f"The ligand '{molName}' has a Radius of Gyration of None, even with the alternative ligand, skipping.")
                        else:
                            # Print a warning
                            ocprint.print_warning(f"The ligand '{molName}' has a Radius of Gyration of None and no alternative ligand was provided.")

                    # Create a box around the ligand
                    m.create_box(centroid = targetCentroid, overwrite = overwrite)
            # If m is not valid
            except Exception as e:
                errMsg = f"The molecule '{mol}' could not be parsed!"

                _ = ocerror.Error.parse_molecule(errMsg, level = ocerror.ReportLevel.ERROR) # type: ignore
                config = get_config()
                ocprint.print_error_log(errMsg, f"{config.logdir}/{dbName}_error_Parse.log")
                return None

        elif moltype == "receptor":
                # If is a tuple
                if type(mol) == tuple:
                    try:
                        # Create a lock for multithreading
                        lock = Lock()
                        # Start the lock with statement
                        with lock:
                            # Check if the extension is pdb
                            if mol[0].endswith(".pdb"):
                                # Clean the receptor
                                _ = ocmolproc.make_only_ATOM_and_CRYST_pdb(structurePath = mol[0])
                            # Create the receptor object
                            m = ocr.Receptor(mol[0], molName, mol2Path = mol[1])
                    except Exception as e:
                        errMsg = f"The molecule '{mol[0]}' could not be parsed! Error {e}"

                        _ = ocerror.Error.parse_molecule(errMsg, level = ocerror.ReportLevel.ERROR) # type: ignore
                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{dbName}_error_Parse.log")
                        return None
                else:
                    try:
                        # Create a lock for multithreading
                        lock = Lock()
                        # Start the lock with statement
                        with lock:
                            # Create the receptor object
                            m = ocr.Receptor(mol, molName)
                    except Exception as e:
                        errMsg = f"The molecule '{mol}' could not be parsed! Error {e}"

                        _ = ocerror.Error.parse_molecule(errMsg, level = ocerror.ReportLevel.ERROR) # type: ignore
                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{dbName}_error_Parse.log")
                        return None
        else:
            _ = ocerror.Error.unknown("Unknown molecule type", level = ocerror.ReportLevel.ERROR) # type: ignore
            return None

        # Test if the molecule is valid
        if not m or not m.is_valid():
            errMsg = f"The molecule '{mol}' is not valid! Its descriptors are malformed. Please check it manually!"

            _ = ocerror.Error.malformed_molecule(errMsg, level = ocerror.ReportLevel.ERROR) # type: ignore
            ocprint.print_error_log(errMsg, f"{logdir}/{dbName}_error_Parse.log")
        else:
            # Export its descriptors
            _ = m.to_json(overwrite)
            
    # Return
    return None


def __sub_core_prepare(dirsToProcess: str, dbName: str, overwrite: bool, mols : List[str] = [], sanitize: bool = True,  targetCentroid: Union[Tuple[float, float, float], rdkit.Geometry.rdGeometry.Point3D] = None) -> List[str]: # type: ignore
    '''Runs the prepare function for the dudez database subsets.

    Parameters
    ----------
    dirsToProcess : str
        Path to the directory to be processed.
    dbName : str
        Name of the database.
    overwrite : bool
        Flag for demanding file overwrite.
    mols : List[str], optional
        List of molecules to be processed. If empty, all folders are inside dirsToProcess are assumed to be molecules and are processed.
    sanitize : bool, optional
        Flag for demanding molecule sanitization, by default True.
    targetCentroid : Tuple[float, float, float] | rdkit.Geometry.rdGeometry.Point3D, optional
        Centroid of the target. If not provided, the centroid of the molecule will be used.

    Returns
    -------
    List[str]
        List of molecule directories.
    '''

    # Check if mols is empty
    if mols:
        # If not, create each dir with the molecule and then move the molecule to it
        for mol in mols:
            # Get the molecule name and path
            _, molName = os.path.split(mol)
            # Remove the extension
            molTmp = molName.split(".")
            # Checage to support files with multiple dots
            if len(molTmp) > 2:
                molName = ".".join(molTmp[:-1])
            else:
                molName = molTmp[0]
            # Create the dir
            _ = ocff.safe_create_dir(f"{mol}/{molName}")
            # Move the molecule to it
            shutil.move(mol, f"{mol}/{molName}/ligand.{molTmp[-1]}")  # type: ignore

    # Get the list of dirs to process
    processDirs = [dirToProcess for dirToProcess in glob(f"{dirsToProcess}/*") if os.path.isdir(dirToProcess)]

    # For each directory (check to see if it is needed to generate descriptors)
    for processDir in processDirs:
        # Safe create docking Files dirs
        _ = ocff.safe_create_dir(f"{processDir}/plantsFiles")
        _ = ocff.safe_create_dir(f"{processDir}/vinaFiles")
        _ = ocff.safe_create_dir(f"{processDir}/sminaFiles")
        _ = ocff.safe_create_dir(f"{processDir}/gninaFiles")

        # Check if the dbName is PDBbind
        if dbName.lower() in ["pdbbind"]:
            # Set the fligand name as the ligand file path
            fligand = f"{processDir}/ligand.sdf"
            alternativeLigand = f"{processDir}/ligand.mol2"
            # For each ligand (don't use parallel, since there is no need)
            __prepare_molecule(fligand, overwrite, "ligand", dbName, sanitize = sanitize, targetCentroid = targetCentroid, alternativeLigand = alternativeLigand)
        else:
            # Set the fligand name as the ligand file path (use mol2)
            fligand = f"{processDir}/ligand.smi"
            # For each ligand (don't use parallel, since there is no need)
            __prepare_molecule(fligand, overwrite, "ligand", dbName, sanitize = sanitize, targetCentroid = targetCentroid)

    return processDirs


def __core_prepare(path: str, overwrite: bool, archive: str, sanitize: bool, spacing: float, targetCentroid: Union[Tuple[float, float, float], rdkit.Geometry.rdGeometry.Point3D] = None) -> int: # type: ignore
    '''Prepares a database entry to be run in multiple docking software.

    Parameters
    ----------
    path : str
        Path to the database directory.
    overwrite : bool
        Flag for demanding file overwrite.
    archive : str
        Which archive to use. Options are [dudez, pdbbind].
    sanitize : bool
        Flag for demanding molecule sanitization.
    spacing : float
        Spacing to enlarge the radius of the sphere used in PLANTS conf file. Ranges from 0 to 1
    targetCentroid : Tuple[float, float, float] | rdkit.Geometry.rdGeometry.Point3D, optional
        Centroid of the target. If not provided, the centroid of the ligand will be used.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Check if the basename of the working directory is not in the list of ignored directories
    if os.path.basename(path) in ['index']:
        # Skip it
        return ocerror.Error.unnalowed_dir() # type: ignore

    # Set the input file name path
    fin = f"{path}/receptor.pdb"
    fout = f"{path}/receptor.mol2"

    # Set the prepared receptor name
    preparedReceptorMol2 = f"{path}/prepared_receptor.mol2"
    preparedReceptorPdbqt = f"{path}/prepared_receptor.pdbqt"

    # Prepare the receptor
    __prepare_molecule((fin, fout), overwrite, "receptor", archive, sanitize = sanitize)

    # Parameterize the compounds folders
    ligands_d = os.path.join(path, "compounds", "ligands")       # known ligands
    decoys_d = os.path.join(path, "compounds", "decoys")         # known decoys
    candidates_d = os.path.join(path, "compounds", "candidates") # unknown ligands

    # Check if there is no target centroid data
    if targetCentroid is None:
        # Parameterize the reference ligand extensions in a list (in order of preference)
        ref_ligand_exts = ["mol2", "sdf", "pdb"]

        # Set the target centroid to None
        targetCentroid = None

        # For each extension in the list
        for ref_ligand_ext in ref_ligand_exts:
            # Parameterize the reference ligand path
            ref_ligand = os.path.join(path, f"reference_ligand.{ref_ligand_ext}")

            # Check if the reference ligand does not exist (extensions in order: pdb, mol2)
            if os.path.isfile(ref_ligand):
                try:
                    # Set the target centroid as the centroid of the ligand from the mol2 file
                    targetCentroid = ocl.get_centroid(ref_ligand, sanitize = sanitize)
                    
                    # Check if the target centroid is None
                    if not targetCentroid:
                        # Print a warning
                        ocprint.print_warning(message = f"WARNING: The centroid of the reference ligand in path '{path}' could not be calculated. The centroid of the receptor will be used instead.")
                        # Force the next iteration
                        continue

                    # Reference ligand found and read, break the loop
                    break
                except Exception as e:
                    # Print the error
                    ocprint.print_error(f"Problems parsing the reference ligand file: {ref_ligand}. Error: {e}")
        
        # Check if the target centroid is still None
        if targetCentroid is None:
            return ocerror.Error.file_not_exist(f"Could not find the file '{' or '.join([os.path.join(path, f'reference_ligand.{ref_ligand_ext}') for ref_ligand_ext in ref_ligand_exts])}' for the molecule '{path}' or the provided files are not valid and a target centroid has not been provided. This molecule will not be processed.", level = ocerror.ReportLevel.ERROR) # type: ignore

    # Create an empty list to hold all dirs to be processed
    processDirs = []

    # If the archive is dudez
    if archive == "dudez":
        # Set the ligand extension to .smi
        ligandExt = ".smi"
    else:
        # Set the ligand extension to .mol2
        ligandExt = ".mol2"
        
    # Check if the ligands dir exists
    if os.path.isdir(ligands_d):
        # For each molecule in ligands dir
        mols = glob(f"{ligands_d}/*.{ligandExt}")
        # Append the dir to the list of dirs to be processed
        processDirs += __sub_core_prepare(ligands_d, archive, overwrite, mols, sanitize, targetCentroid = targetCentroid)

    # Check if the decoys dir exists
    if os.path.isdir(decoys_d):
        # For each molecule in dudez decoy dir
        mols = glob(f"{decoys_d}/*.{ligandExt}")
        # Append the dir to the list of dirs to be processed
        processDirs += __sub_core_prepare(decoys_d, archive, overwrite, mols, sanitize, targetCentroid = targetCentroid)
    
    # Check if the candidates dir exists
    if os.path.isdir(candidates_d):
        # For each molecule in dudez candidate dir
        mols = glob(f"{candidates_d}/*.{ligandExt}")
        # Append the dir to the list of dirs to be processed
        processDirs += __sub_core_prepare(candidates_d, archive, overwrite, mols, sanitize, targetCentroid = targetCentroid)

    # For each dir to be processed
    for processDir in processDirs:
        # Check if there is a box for the ligand
        boxCount = len(glob(f"{processDir}/boxes/box*.pdb"))

        # If overwrite mode is on or there is not the same amount of box files as folders in gninaFiles folder
        if boxCount == 0 or len(glob(f"{processDir}/gninaFiles/*")) != boxCount or len(glob(f"{processDir}/gninaFiles/*")) == 0 or overwrite:
            # Create a lock for multithreading
            lock = Lock()
            # Start the lock with statement
            with lock:
                # Create the vina inputs from the boxes
                ocgnina.gen_gnina_conf(f"{processDir}/boxes/box0.pdb", f"{processDir}/sminaFiles/conf_smina.conf", preparedReceptorPdbqt)
        else:
            ocprint.print_info(f"The protein '{processDir}' already has its gnina file generated, skipping its execution.")
        
        # If overwrite mode is on or there is not the same amount of box files as folders in vinaFiles folder
        if boxCount == 0 or len(glob(f"{processDir}/vinaFiles/*")) != boxCount or len(glob(f"{processDir}/vinaFiles/*")) == 0 or overwrite:
            # Create a lock for multithreading
            lock = Lock()
            # Start the lock with statement
            with lock:
                # Create the vina inputs from the boxes
                ocvina.generate_vina_files_database(processDir, preparedReceptorPdbqt, boxPath = f"{processDir}/boxes")
        else:
            ocprint.print_info(f"The protein '{processDir}' already has its vina file generated, skipping its execution.")

        # If overwrite mode is on or there is not the same amount of box files as folders in plantsFiles folder
        if boxCount == 0 or len(glob(f"{processDir}/plantsFiles/*")) != boxCount or len(glob(f"{processDir}/plantsFiles/*")) == 0 or overwrite:
            # Set the fligand variable to the dir + ligandName + .mol2
            fligand = f"{processDir}/ligand.mol2"
            # Create a lock for multithreading
            lock = Lock()
            # Start the lock with statement
            with lock:
                # Create the PLANTS inputs from the boxes
                ocplants.generate_plants_files_database(processDir, preparedReceptorMol2, fligand, spacing, boxPath = f"{processDir}/boxes")
        else:
            ocprint.print_info(f"The protein '{processDir}' already has its PLANTS file generated, skipping its execution.")

        # If overwrite mode is on or there not any conf file in the sminaFiles folder
        if len(glob(f"{processDir}/sminaFiles/*.conf")) == 0 or overwrite:
            # Create a lock for multithreading
            lock = Lock()
            # Start the lock with statement
            with lock:
                # Create the smina inputs
                ocsmina.gen_smina_conf(f"{processDir}/boxes/box0.pdb", f"{processDir}/sminaFiles/conf_smina.conf", preparedReceptorPdbqt)
        else:
            ocprint.print_info(f"The protein '{processDir}' already has its smina file generated, skipping its execution.")

    return ocerror.Error.ok() # type: ignore


def __thread_prepare(arguments: Tuple[str, bool, str, bool, float]) -> int:
    '''Thread aid function to call __core_prepare.

    Parameters
    ----------
    arguments : Tuple[str, bool, str, bool, float]
        The arguments to be passed to __core_prepare. Its arguments are: (path, overwrite, archive, sanitize, spacing). See __core_prepare for more information.

    Returns
    -------
    int
        The error code. See octools.error_codes for more information.
    '''
    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        # Call core prepare function (shared between thread and no thread)
        return __core_prepare(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])


def __prepare_parallel(paths: List[str], overwrite: bool, archive: str, sanitize: bool, spacing: float, desc: str) -> None:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    TODO: Add the support to custom databases.

    Parameters
    ----------
    paths : List[str]
        The list of directories to be processed.
    overwrite : bool
        If True, the function will overwrite the files if they already exists.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    sanitize : bool
        If True, the function will sanitize the molecules.
    spacing : float
        The spacing value used to enlarge the radius of the sphere used in PLANTS file. Ranges from 0 to 1.
    desc : str
        The description to be used in the tqdm progress bar.
    
    Returns
    -------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    
    # For each file in the glob
    for path in paths:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((path, overwrite, archive, sanitize, spacing))

    try:
        # Create a Thread pool with the maximum available_cores
        config = get_config()
        with Pool(config.available_cores) as p:
            # Perform the multi process
            for _ in tqdm(p.imap_unordered(__thread_prepare, arguments), total = len(arguments), desc = desc):
                # Clear the memory
                gc.collect()
    except IOError as e:
        errMsg = f"Problem while preparing {archive}. Exception: {e}"
        config = get_config()
        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_prepare_report.log")
        ocprint.print_error(errMsg)
    
    return None


def __prepare_no_parallel(paths: List[str], overwrite: bool, archive: str, sanitize: bool, spacing: float, desc: str) -> None:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_prepare function.

    TODO: Add the support to custom databases.

    Parameters
    ----------
    paths : List[str]
        The list of directories to be processed.
    overwrite : bool
        If True, the function will overwrite the files if they already exists.
    archive: str
        The archive name. Options are [dudez, pdbbind].
    sanitize : bool
        If True, the function will sanitize the molecules.
    spacing : float
        The spacing value used to enlarge the radius of the sphere used in PLANTS file. Ranges from 0 to 1.
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    None
    '''

    # Redirect all prints to tqdm.write
    with ocbasetools.redirect_to_tqdm():
        for path in tqdm(iterable=paths, total=len(paths), desc=desc):
            # Call the core prepare function
            __core_prepare(path, overwrite, archive, sanitize, spacing)
            # Clear the memory
            gc.collect()

    return None


def __prepare_single(path: str, overwrite: bool, archive: str, sanitize: bool, spacing: float) -> None:
    '''Warper to prepare the jobs, recieves a directory, and pass it to the __core_prepare function.

    TODO: Add the support to custom databases.

    Parameters
    ----------
    paths : str
        The directory to be processed.
    overwrite : bool
        If True, the function will overwrite the files if they already exists.
    archive: str
        The archive name. Options are [dudez, pdbbind].
    sanitize : bool
        If True, the function will sanitize the molecules.
    spacing : float
        The spacing value used to enlarge the radius of the sphere used in PLANTS file. Ranges from 0 to 1.

    Returns
    -------
    None
    '''

    __core_prepare(path, overwrite, archive, sanitize, spacing)
    gc.collect()

    return None





## Public ##


def prepare(paths: Union[List[str], str], overwrite: bool, archive: str, sanitize: bool, spacing: float) -> None:
    '''Prepare the files to be used in the docking process.

    Parameters
    ----------
    paths : List[str] | str
        The list of directories or the directory to be processed.
    overwrite : bool
        If True, the function will overwrite the files if they already exists.
    archive : str
        The archive name. Options are [dudez, pdbbind].
    sanitize : bool
        If True, the function will sanitize the molecules.
    spacing : float
        The spacing value used to enlarge the radius of the sphere used in PLANTS file. Ranges from 0 to 1.
    '''

    # If the path is a list
    if isinstance(paths, list):
        # Backup log
        oclogging.backup_log(f"{archive}_prepare_report")
        
        # Set the description
        label = f"Preparing {archive}"

        # Check if multiprocessing is enabled
        config = get_config()
        if config.multiprocess:
            # Prepare the pdbbind
            __prepare_parallel(paths, overwrite, archive, sanitize, spacing, label)
        else:
            # Prepare the database
            __prepare_no_parallel(paths, overwrite, archive, sanitize, spacing, label)
    else:
        __prepare_single(paths, overwrite, archive, sanitize, spacing)
