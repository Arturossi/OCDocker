#!/usr/lib/python3

# Imports
###############################################################################
import errno
import gc
import os
import time
import rdkit
import shutil
import vaex

import numpy as np
import pandas as pd
import vaex.dataframe as vdf

from glob import glob
from multiprocessing import Pool
from threading import Lock
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
import OCDocker.ExternalTools.runprank as runprank


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
### p2rank
def __run_p2rank(dir: str, fin: str, overwrite: bool = False) -> None:
    '''Runs p2rank for a given directory.

    Parameters
    ----------
    dir : str
        Path where the data is.
    fin : str
        PDB file as input.
    overwrite : bool, optional
        Flag for demanding file overwrite, by default False.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Set the output path
    fout = f"{dir}/p2rank"

    # Algorithms to be analyzed (Only Agglomerative Clustering)
    algorithms = {
        "AffinityPropagation": False,
        "AgglomerativeClustering": True,
        "Birch": False,
        "DBSCAN": False,
        "KMeans": False,
        "MeanShift": False,
        "MiniBatchKMeans": False,
        "NoCluster": False,
        "OPTICS": False,
        "SpectralClustering": False
    }

    # Create a lock for multithreading
    lock = Lock()
    # Start the lock with statement
    with lock:
        try:
            # Run p2rank
            runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = True if args.output_level >= 3 else False, overwrite = overwrite)
        except Exception as e:
            octools.print_warning(f"The protein '{dir}' had a problem while running p2rank. Retrying to run p2rank. Exception: {e}")
            runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = True if args.output_level >= 3 else False, overwrite = overwrite)

    return None

def __core_p2rank(dir: str, overwrite: bool, archive: str) -> None:
    '''Prepares a database entry to be run in multiple docking software.

    Parameters
    ----------
    dir : str
        Path where the data is.
    overwrite : bool
        Flag for demanding file overwrite.
    archive : str
        Which archive will be processed [dudez, pdbbind].

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    fin = ""

    if archive == "dudez":
        # Set the input file name path
        fin = f"{dir}/rec.crg.pdb"
    elif archive == "pdbbind":
        # If is the index path
        if os.path.basename(dir) in ['index']:
            # Skip it
            return
        # Find the protein name
        ptn = dir.split(os.path.sep)[-1]
        # Set the input file name path (to generate the box and data about the protein)
        fin = f"{dir}/{ptn}_protein.pdb"
    # Set the output path
    fout = f"{dir}/p2rank"
    # Create the p2rank output dir
    _ = octools.safe_create_dir(fout)
    # Parameterizing box count
    boxCount = len(glob(f"{fout}/box*.pdb"))
    # If overwrite mode is on or there is no box in the p2rank output, p2rank will run
    if boxCount == 0 or overwrite:
        # Run p2rank
        __run_p2rank(dir, fin, overwrite=overwrite) 
    else:
        octools.print_info(f"The protein '{dir}' already has its p2rank output generated, skipping its execution.")

    return None

def __thread_p2rank(arguments: Tuple[str, bool, str]) -> None:
    '''Thread aid function to call __core_p2rank.

    Parameters
    ----------
    arguments : Tuple[str, bool, str]
        Tuple with the arguments to be passed to __core_p2rank. The arguments are: (dir, overwrite, archive). Where dir is the path where the data is, overwrite is a flag for demanding file overwrite and archive is which archive will be processed [dudez, pdbbind].

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call core prepare function (shared between thread and no thread)
        return __core_p2rank(arguments[0], arguments[1], arguments[2])

def __p2rank_parallel(dirs: List[str], overwrite: bool, archive: str, desc: str) -> None:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    dirs: List[str]
        List of directories to be processed.
    overwrite: bool
        Flag for demanding file overwrite.
    archive: str
        Which archive will be processed [dudez, pdbbind].
    desc: str
        Description to be used in the progress bar.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for dir in dirs:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((dir, overwrite, archive))
    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for _ in tqdm(p.imap_unordered(__thread_p2rank, arguments), total = len(arguments), desc = desc):
                # Clear the memory
                gc.collect()
    except IOError as e:
        octools.print_error_log(f"Problem while running p2rank in parallel. Exception: {e}", f"{logdir}/{archive}_p2rank_report.log")
        octools.print_error(f"Problem while running p2rank in parallel. Exception: {e}")

    # Return
    return None

def __p2rank_no_parallel(dirs: List[str], overwrite: bool, archive: str, desc: str) -> None:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_p2rank function.

    Parameters
    ----------
    dirs: List[str]
        List of directories to be processed.
    overwrite: bool
        Flag for demanding file overwrite.
    archive: str
        Which archive will be processed [dudez, pdbbind].
    desc: str
        Description to be used in the progress bar.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for dir in tqdm(iterable=dirs, total=len(dirs), desc=desc):
            # Call the core p2rank function
            __core_p2rank(dir, overwrite, archive)
            # Clear the memory
            gc.collect()
    return None

### Prepare
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

    Raises
    ------
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
            _ = octools.safe_create_dir(f"{molPath}/plantsFiles")
            _ = octools.safe_create_dir(f"{molPath}/vinaFiles")
            _ = octools.safe_create_dir(f"{molPath}/sminaFiles")
            _ = octools.safe_create_dir(f"{molPath}/gninaFiles")

            try:
                # Create a lock for multithreading
                lock = Lock()
                # Start the lock with statement
                with lock:
                    # Create the ligand object
                    m = ocl.Ligand(mol, molName, sanitize = sanitize)
                    # Test if the Radius of Gyration is None
                    if not m.RadiusOfGyration:
                        # Print a warning
                        octools.print_warning(f"The ligand '{molName}' has a Radius of Gyration of None, trying to load its alternative ligand.")
                        # If so, try to load the alternative ligand
                        if alternativeLigand:
                            # Create the ligand object
                            m = ocl.Ligand(alternativeLigand, molName, sanitize = sanitize)
                            # Check the radius of gyration again
                            if not m.RadiusOfGyration:
                                # If it is still None, print a warning and return
                                octools.print_warning(f"The ligand '{molName}' has a Radius of Gyration of None, even with the alternative ligand, skipping.")
                        else:
                            # Print a warning
                            octools.print_warning(f"The ligand '{molName}' has a Radius of Gyration of None and no alternative ligand was provided.")

                    # Create a box around the ligand
                    m.create_box(centroid = targetCentroid, overwrite = overwrite)
            # If m is not valid
            except Exception as e:
                errMsg = f"The molecule '{mol}' could not be parsed!"

                _ = errors.parse_molecule(errMsg, "error")
                octools.print_error_log(errMsg, f"{logdir}/{dbName}_error_Parse.log")
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
                                _ = octools.make_only_ATOM_and_CRYST_pdb(structurePath = mol[0])
                            # Create the receptor object
                            m = ocr.Receptor(mol[0], molName, mol2Path = mol[1])
                    except Exception as e:
                        errMsg = f"The molecule '{mol[0]}' could not be parsed! Error {e}"

                        _ = errors.parse_molecule(errMsg, "error")
                        octools.print_error_log(errMsg, f"{logdir}/{dbName}_error_Parse.log")
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

                        _ = errors.parse_molecule(errMsg, "error")
                        octools.print_error_log(errMsg, f"{logdir}/{dbName}_error_Parse.log")
                        return None
        else:
            _ = errors.unknown("Unknown molecule type", "error")
            return None

        # Test if the molecule is valid
        if not m or not m.is_valid():
            errMsg = f"The molecule '{mol}' is not valid! Its descriptors are malformed. Please check it manually!"

            _ = errors.malformed_molecule(errMsg, "error")
            octools.print_error_log(errMsg, f"{logdir}/{dbName}_error_Parse.log")
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

    Raises
    ------
    None
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
            _ = octools.safe_create_dir(f"{mol}/{molName}")
            # Move the molecule to it
            shutil.move(mol, f"{mol}/{molName}/ligand.{molTmp[-1]}")  # type: ignore

    # Get the list of dirs to process
    processDirs = [dirToProcess for dirToProcess in glob(f"{dirsToProcess}/*") if os.path.isdir(dirsToProcess)]

    # For each directory (check to see if it is needed to generate descriptors)
    for processDir in processDirs:
        # Safe create docking Files dirs
        _ = octools.safe_create_dir(f"{processDir}/plantsFiles")
        _ = octools.safe_create_dir(f"{processDir}/vinaFiles")
        _ = octools.safe_create_dir(f"{processDir}/sminaFiles")
        _ = octools.safe_create_dir(f"{processDir}/gninaFiles")

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

    Raises
    ------
    None
    '''

    # Check if the basename of the working directory is not in the list of ignored directories
    if os.path.basename(path) in ['index']:
        # Skip it
        return errors.unnalowed_dir()

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
                        octools.print_warning(message = f"WARNING: The centroid of the reference ligand in path '{path}' could not be calculated. The centroid of the receptor will be used instead.")
                        # Force the next iteration
                        continue

                    # Reference ligand found and read, break the loop
                    break
                except Exception as e:
                    # Print the error
                    octools.print_error(f"Problems parsing the reference ligand file: {ref_ligand}. Error: {e}")
        
        # Check if the target centroid is still None
        if targetCentroid is None:
            return errors.file_do_not_exist(f"Could not find the file '{' or '.join([os.path.join(path, f'reference_ligand.{ref_ligand_ext}') for ref_ligand_ext in ref_ligand_exts])}' for the molecule '{path}' or the provided files are not valid and a target centroid has not been provided. This molecule will not be processed.", level = "error")            

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

    ''' P2Rank is not used yet
    # Set the output path
    fout = f"{path}/p2rank"
    # Create the p2rank output dir
    _ = octools.safe_create_dir(fout)
    # Parameterizing box count
    boxCount = len(glob(f"{fout}/box*.pdb"))
    # If overwrite mode is on or there is no box in the p2rank output, p2rank will run
    if boxCount == 0 or overwrite:
        # Run p2rank
        __run_p2rank(path, fin, overwrite=overwrite)
    else:
        octools.print_info(f"The protein '{path}' already has its p2rank output generated, skipping its execution.")
    '''

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
            octools.print_info(f"The protein '{processDir}' already has its gnina file generated, skipping its execution.")
        
        # If overwrite mode is on or there is not the same amount of box files as folders in vinaFiles folder
        if boxCount == 0 or len(glob(f"{processDir}/vinaFiles/*")) != boxCount or len(glob(f"{processDir}/vinaFiles/*")) == 0 or overwrite:
            # Create a lock for multithreading
            lock = Lock()
            # Start the lock with statement
            with lock:
                # Create the vina inputs from the boxes
                ocvina.generate_vina_files_database(processDir, preparedReceptorPdbqt, boxPath = f"{processDir}/boxes")
        else:
            octools.print_info(f"The protein '{processDir}' already has its vina file generated, skipping its execution.")

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
            octools.print_info(f"The protein '{processDir}' already has its PLANTS file generated, skipping its execution.")

        # If overwrite mode is on or there not any conf file in the sminaFiles folder
        if len(glob(f"{processDir}/sminaFiles/*.conf")) == 0 or overwrite:
            # Create a lock for multithreading
            lock = Lock()
            # Start the lock with statement
            with lock:
                # Create the smina inputs
                ocsmina.gen_smina_conf(f"{processDir}/boxes/box0.pdb", f"{processDir}/sminaFiles/conf_smina.conf", preparedReceptorPdbqt)
        else:
            octools.print_info(f"The protein '{processDir}' already has its smina file generated, skipping its execution.")

    return errors.ok()

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

    Raises
    ------
    None
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call core prepare function (shared between thread and no thread)
        return __core_prepare(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])

def __prepare_parallel(dirs: List[str], overwrite: bool, archive: str, sanitize: bool, spacing: float, desc: str) -> None:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    TODO: Add the support to custom databases.

    Parameters
    ----------
    dirs : List[str]
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

    Raises
    ------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    
    # For each file in the glob
    for dir in dirs:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((dir, overwrite, archive, sanitize, spacing))

    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for _ in tqdm(p.imap_unordered(__thread_prepare, arguments), total = len(arguments), desc = desc):
                # Clear the memory
                gc.collect()
    except IOError as e:
        octools.print_error_log(f"Problem while preparing {archive}. Exception: {e}", f"{logdir}/{archive}_prepare_report.log")
        octools.print_error(f"Problem while preparing {archive}. Exception: {e}")
    
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

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for path in tqdm(iterable=paths, total=len(paths), desc=desc):
            # Call the core prepare function
            __core_prepare(path, overwrite, archive, sanitize, spacing)
            # Clear the memory
            gc.collect()

    return None


### Docking
def __run_gnina(ligandPath: str, ligandDescriptorPath: str, receptorPath: str, receptorDescriptorPath: str, boxPath: str, ptn: str, archive: str, overwrite: bool = False) -> int:
    '''Runs gnina.

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
    overwrite : bool, optional
        If True, overwrite the output file. Defaults to False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Get the ligand Dir
    ligandDir = os.path.dirname(ligandPath)

    # Flag to denote if its needed to run this protein through gnina
    needToRun = False
    # Check if the gninaFiles directory exists
    if not os.path.isdir(f"{ligandDir}/gninaFiles"):
        errMsg = f"The directory '{ligandDir}/gninaFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok."

        octools.print_error_log(errMsg, f"{logdir}/{archive}_gnina_run_report_ERROR.log")
        return errors.dir_does_not_exists(errMsg, level = "error")

    # Get the folder for each run
    runPaths = [f"{ligandDir}/gninaFiles"] # glob(f"{ligandDir}/gninaFiles/*") # TODO: Add support for multiple runs

    # Check if all files have been processed
    for runPath in runPaths:
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
        # Create a lock for multithreading
        lock = Lock()
        # Start the lock with statement
        with lock:
            # Read the receptor and the ligand
            receptor = ocr.Receptor(receptorPath, from_json_descriptors = receptorDescriptorPath, name = f"{ptn}_receptor")
            ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptorPath, name = f"{ptn}_{lig}_ligand")
        
        # If receptor and ligand are not null
        if receptor and ligand:
            # For each path in the paths array (will be more than on in case of multiple boxes)
            for runPath in runPaths:
                # Get the run number
                runNumber = 0 # TODO: add support to multiple runs, currently only 0, the code should be something like: runPath.split(os.path.sep)[-1]

                # Set the prepared receptor and ligand paths
                preparedReceptorPath = f"{os.path.dirname(receptorPath)}/prepared_receptor.pdbqt"
                preparedLigandPath = f"{ligandDir}/prepared_ligand.pdbqt"

                # Parameterizing paths
                gninaLog = f"{runPath}/gnina_{runNumber}.log"
                gninaOutput = f"{runPath}/gnina_{runNumber}.pdbqt"

                # Create a lock for multithreading
                lock = Lock()
                # Start the lock with statement
                with lock:
                    # Create the gnina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                    gnina = ocgnina.Gnina(f"{runPath}/conf_gnina.conf", boxPath, receptor, preparedReceptorPath, ligand, preparedLigandPath, gninaLog, gninaOutput, name = f"{ptn}_run_{runNumber}", overwriteConfig = overwrite)

                # Check if the gnina object has been correctly created
                if not gnina:
                    errMsg = f"Could not generate gnina object for the protein in dir '{ligandPath}'. Error found while trying to run the 'gnina' docking software."

                    octools.print_error_log(errMsg, f"{logdir}/{archive}_gnina_run_report_ERROR.log")
                    return errors.docking_object_not_generated(errMsg, level = "error")

                # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(gnina.preparedLigand) or os.path.getsize(gnina.preparedLigand) == 0 or not octools.is_molecule_valid(gnina.preparedLigand):
                    # Create a lock for multithreading
                    lock = Lock()
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare ligand
                            result = gnina.run_prepare_ligand()
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
                            errMsg = f"Could not run the prepare ligand routine for the protein in dir '{gnina.inputLigandPath}'. Error found while trying to run the 'gnina' docking software. Error: {e}"

                            octools.print_error_log(errMsg, f"{logdir}/{archive}_gnina_run_report_ERROR.log")
                            return errors.ligand_not_prepared(errMsg, level = "error")

                    # Check again if the generated ligand has size 0 or is invalid
                    if os.path.getsize(gnina.preparedLigand) == 0 or not octools.is_molecule_valid(gnina.preparedLigand):
                        errMsg = f"The prepare ligand script has made an output of 0kb again for ligand '{gnina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(gnina.prepareLigandCmd)}"

                        octools.print_error_log(errMsg, f"{logdir}/{archive}_gnina_run_report_ERROR.log")
                        return errors.ligand_not_prepared(errMsg, level = "error")

                # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(gnina.preparedReceptor) or os.path.getsize(gnina.preparedReceptor) == 0 or not octools.is_molecule_valid(gnina.preparedReceptor):
                    # Create a lock for multithreading
                    lock = Lock()
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare receptor
                            result = gnina.run_prepare_receptor()
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
                            errMsg = f"Could not run the prepare receptor routine for the protein in dir '{gnina.inputReceptorPath}'. Error found while trying to run the 'gnina' docking software. Error: {e}"

                            octools.print_error_log(errMsg, f"{logdir}/{archive}_gnina_run_report_ERROR.log")
                            return errors.receptor_not_prepared(errMsg, level = "error")

                    # Check if the generated receptor has size 0 or is invalid
                    if os.path.getsize(gnina.preparedReceptor) == 0 or not octools.is_molecule_valid(gnina.preparedReceptor):
                        errMsg = f"The prepare receptor has made an output of 0kb for receptor '{gnina.preparedReceptor}' or is not valid... Here is its command line so you might be able to debug it by hand.\n{' '.join(gnina.prepareReceptorCmd)}"

                        octools.print_error_log(errMsg, f"{logdir}/{archive}_gnina_run_report_ERROR.log")
                        return errors.receptor_not_prepared(errMsg, level = "error")

                # Check if gnina output exists
                if overwrite or not os.path.isfile(gninaOutput) or not os.path.isfile(gninaLog):
                    # Create a lock for multithreading
                    lock = Lock()
                    # Start the lock with statement
                    with lock:
                        # Run gnina
                        gnina.run_gnina()
                else:
                    errMsg = f"The gnina output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '{runPath}/gnina_{runNumber}.log' path. Gnina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true"

                    octools.print_warning_log(errMsg, f"{logdir}/{archive}_gnina_run_report_WARNING.log")
                    octools.print_warning(errMsg)
        else:
            errMsg = f"Could not generate receptor or ligand object for the protein in dir '{ligandPath}'. Error found while trying to run the 'gnina' docking software."

            octools.print_error_log(errMsg, f"{logdir}/{archive}_gnina_run_report_ERROR.log")
            return errors.receptor_or_ligand_not_generated(errMsg, level = "error")
    else:
        errMsg = f"The gnina output for '{ptn}' for all boxes is already generated and you can check it at the '{ligandPath}/gninaFiles/*/gnina_<runNumber>.log' path. Gnina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

        octools.print_warning_log(errMsg, f"{logdir}/{archive}_gnina_run_report_WARNING.log")
        octools.print_warning(errMsg)

    return errors.ok()

def __run_vina(ligandPath: str, ligandDescriptorPath: str, receptorPath: str, receptorDescriptorPath: str, boxPath: str, ptn: str, archive: str, overwrite: bool = False) -> int:
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
    overwrite : bool, optional
        If True, overwrite the output file. Defaults to False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Get the ligand Dir
    ligandDir = os.path.dirname(ligandPath)

    # Flag to denote if its needed to run this protein through vina
    needToRun = False
    # Check if the vinaFiles directory exists
    if not os.path.isdir(f"{ligandDir}/vinaFiles"):
        errMsg = f"The directory '{ligandDir}/vinaFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok."

        octools.print_error_log(errMsg, f"{logdir}/{archive}_vina_run_report_ERROR.log")
        return errors.dir_does_not_exists(errMsg, level = "error")

    # Get the folder for each run
    runPaths = [f"{ligandDir}/vinaFiles"] # glob(f"{ligandDir}/vinaFiles/*") # TODO: Add support for multiple runs

    # Check if all files have been processed
    for runPath in runPaths:
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
        # Create a lock for multithreading
        lock = Lock()
        # Start the lock with statement
        with lock:
            # Read the receptor and the ligand
            receptor = ocr.Receptor(receptorPath, from_json_descriptors = receptorDescriptorPath, name = f"{ptn}_receptor")
            ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptorPath, name = f"{ptn}_{lig}_ligand")
        
        # If receptor and ligand are not null
        if receptor and ligand:
            # For each path in the paths array (will be more than on in case of multiple boxes)
            for runPath in runPaths:
                # Get the run number
                runNumber = 0 # TODO: add support to multiple runs, currently only 0, the code should be something like: runPath.split(os.path.sep)[-1]

                # Set the prepared receptor and ligand paths
                preparedReceptorPath = f"{os.path.dirname(receptorPath)}/prepared_receptor.pdbqt"
                preparedLigandPath = f"{ligandDir}/prepared_ligand.pdbqt"

                # Parameterizing paths
                vinaLog = f"{runPath}/vina_{runNumber}.log"
                vinaOutput = f"{runPath}/vina_{runNumber}.pdbqt"

                # Create a lock for multithreading
                lock = Lock()
                # Start the lock with statement
                with lock:
                    # Create the vina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                    vina = ocvina.Vina(f"{runPath}/conf_vina.conf", boxPath, receptor, preparedReceptorPath, ligand, preparedLigandPath, vinaLog, vinaOutput, name = f"{ptn}_run_{runNumber}", overwriteConfig = overwrite)

                # Check if the vina object has been correctly created
                if not vina:
                    errMsg = f"Could not generate vina object for the protein in dir '{ligandPath}'. Error found while trying to run the 'vina' docking software."

                    octools.print_error_log(errMsg, f"{logdir}/{archive}_vina_run_report_ERROR.log")
                    return errors.docking_object_not_generated(errMsg, level = "error")

                # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(vina.preparedLigand) or os.path.getsize(vina.preparedLigand) == 0 or not octools.is_molecule_valid(vina.preparedLigand):
                    # Create a lock for multithreading
                    lock = Lock()
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare ligand
                            result = vina.run_prepare_ligand(useOpenBabel = False) # useOpenBabel has proven to be a dangerous option, it is better to avoid its use for
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
                            errMsg = f"Could not run the prepare ligand routine for the protein in dir '{vina.inputLigandPath}'. Error found while trying to run the 'vina' docking software. Error: {e}"

                            octools.print_error_log(errMsg, f"{logdir}/{archive}_vina_run_report_ERROR.log")
                            return errors.ligand_not_prepared(errMsg, level = "error")

                    # Check again if the generated ligand has size 0 or is invalid
                    if os.path.getsize(vina.preparedLigand) == 0 or not octools.is_molecule_valid(vina.preparedLigand):
                        errMsg = f"The prepare ligand script has made an output of 0kb again for ligand '{vina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareLigandCmd)}"

                        octools.print_error_log(errMsg, f"{logdir}/{archive}_vina_run_report_ERROR.log")
                        return errors.ligand_not_prepared(errMsg, level = "error")

                # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(vina.preparedReceptor) or os.path.getsize(vina.preparedReceptor) == 0 or not octools.is_molecule_valid(vina.preparedReceptor):
                    # Create a lock for multithreading
                    lock = Lock()
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare receptor
                            result = vina.run_prepare_receptor(useOpenBabel = False) # useOpenBabel has proven to be a dangerous option, it is better to avoid its use for now
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
                            errMsg = f"Could not run the prepare receptor routine for the protein in dir '{vina.inputReceptorPath}'. Error found while trying to run the 'vina' docking software. Error: {e}"

                            octools.print_error_log(errMsg, f"{logdir}/{archive}_vina_run_report_ERROR.log")
                            return errors.receptor_not_prepared(errMsg, level = "error")

                    # Check if the generated receptor has size 0 or is invalid
                    if os.path.getsize(vina.preparedReceptor) == 0 or not octools.is_molecule_valid(vina.preparedReceptor):
                        errMsg = f"The prepare receptor has made an output of 0kb for receptor '{vina.preparedReceptor}' or is not valid... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareReceptorCmd)}"

                        octools.print_error_log(errMsg, f"{logdir}/{archive}_vina_run_report_ERROR.log")
                        return errors.receptor_not_prepared(errMsg, level = "error")

                # Check if vina output exists
                if overwrite or not os.path.isfile(vinaOutput) or not os.path.isfile(vinaLog):
                    # Create a lock for multithreading
                    lock = Lock()
                    # Start the lock with statement
                    with lock:
                        # Run vina
                        vina.run_vina()
                else:
                    errMsg = f"The vina output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '{runPath}/vina_{runNumber}.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true"

                    octools.print_warning_log(errMsg, f"{logdir}/{archive}_vina_run_report_WARNING.log")
                    octools.print_warning(errMsg)
        else:
            errMsg = f"Could not generate receptor or ligand object for the protein in dir '{ligandPath}'. Error found while trying to run the 'vina' docking software."

            octools.print_error_log(errMsg, f"{logdir}/{archive}_vina_run_report_ERROR.log")
            return errors.receptor_or_ligand_not_generated(errMsg, level = "error")
    else:
        errMsg = f"The vina output for '{ptn}' for all boxes is already generated and you can check it at the '{ligandPath}/vinaFiles/*/vina_<runNumber>.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

        octools.print_warning_log(errMsg, f"{logdir}/{archive}_vina_run_report_WARNING.log")
        octools.print_warning(errMsg)

    return errors.ok()


def __run_smina(ligandPath: str, ligandDescriptorPath: str, receptorPath: str, receptorDescriptorPath: str, boxPath: str, ptn: str, archive: str, overwrite: bool = False) -> int:
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
    overwrite : bool, optional
        If True, overwrite the output file. Defaults to False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Get the ligand Dir
    ligandDir = os.path.dirname(ligandPath)

    # Set the run path (attatched to the ligand)
    runPath = f"{ligandDir}/sminaFiles"

    # Parameterizing paths
    sminaLog = f"{runPath}/smina.log"
    sminaOutput = f"{runPath}/smina.pdbqt"

    # Check if sminaFiles does not exist
    if not os.path.isdir(runPath):
        errMsg = f"The directory '{runPath}' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok."

        octools.print_error_log(errMsg, f"{logdir}/{archive}_smina_run_report_ERROR.log")
        return errors.dir_does_not_exists(errMsg, level = "error")

    # If is needed to run (overwrite is set or no output is produced)
    if overwrite or not os.path.isfile(sminaLog) or not os.path.isfile(sminaOutput):
        # Get the ligand name
        lig = os.path.split(os.path.dirname(ligandPath))[-1]
        # Create a lock for multithreading
        lock = Lock()
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

            # Create a lock for multithreading
            lock = Lock()
            # Start the lock with statement
            with lock:
                # Create the smina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                smina = ocsmina.Smina(f"{runPath}/conf_smina.conf", boxPath, receptor, preparedReceptorPath, ligand, preparedLigandPath, sminaLog, sminaOutput, name=f"{ptn}_smina", overwriteConfig = overwrite)

            # Check if the smina object has been correctly created
            if not smina:
                errMsg = f"Could not generate smina object for the protein in dir '{ligandPath}'. Error found while trying to run the 'smina' docking software."

                octools.print_error_log(errMsg, f"{logdir}/{archive}_smina_run_report_ERROR.log")
                return errors.docking_object_not_generated(errMsg, level = "error")

            # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
            if overwrite or not os.path.isfile(smina.preparedLigand) or os.path.getsize(smina.preparedLigand) == 0 or not octools.is_molecule_valid(smina.preparedLigand):
                # Create a lock for multithreading
                lock = Lock()
                # Start the lock with statement
                with lock:
                    try:
                        # Run the prepare ligand
                        result = smina.run_prepare_ligand()
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
                        errMsg = f"Could not run the prepare ligand routine for the protein in dir '{smina.inputLigandPath}'. Error found while trying to run the 'smina' docking software. Error: {e}"

                        octools.print_error_log(errMsg, f"{logdir}/{archive}_smina_run_report_ERROR.log")
                        return errors.ligand_not_prepared(errMsg, level = "error")

                # Check if the generated ligand has size 0 or is invalid
                if os.path.getsize(smina.preparedLigand) == 0 or not octools.is_molecule_valid(smina.preparedLigand):
                    errMsg = f"The prepare ligand script has made an output of 0kb for ligand '{smina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareLigandCmd)}"

                    octools.print_error_log(errMsg, f"{logdir}/{archive}_smina_run_report_ERROR.log")
                    return errors.ligand_not_prepared(errMsg, level = "error")
                    
            # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
            if overwrite or not os.path.isfile(smina.preparedReceptor) or os.path.getsize(smina.preparedReceptor) == 0 or not octools.is_molecule_valid(smina.preparedReceptor):
                # Create a lock for multithreading
                lock = Lock()
                # Start the lock with statement
                with lock:
                    try:
                        # Run the prepare receptor
                        result = smina.run_prepare_receptor()
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
                        errMsg = f"Could not run the prepare receptor routine for the protein in dir '{smina.inputReceptorPath}'. Error found while trying to run the 'smina' docking software. Error: {e}"

                        octools.print_error_log(errMsg, f"{logdir}/{archive}_smina_run_report_ERROR.log")
                        return errors.ligand_not_prepared(errMsg, level = "error")

                # Check if the generated receptor has size 0 or is invalid
                if os.path.getsize(smina.preparedReceptor) == 0 or not octools.is_molecule_valid(smina.preparedReceptor):
                    errMsg = f"The prepare receptor has made an output of 0kb for receptor '{smina.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareReceptorCmd)}"

                    octools.print_error_log(errMsg, f"{logdir}/{archive}_smina_run_report_ERROR.log")
                    return errors.receptor_not_prepared(errMsg, level = "error")

            # Create a lock for multithreading
            lock = Lock()
            # Start the lock with statement
            with lock:
                # Run smina (no need to recheck for overwrite or output existance because it is already done some lines ago)
                smina.run_smina()
        else:
            errMsg = f"Could not generate receptor or ligand object for the protein in dir '{ligandPath}'. Error found while trying to run the 'smina' docking software."

            octools.print_error_log(errMsg, f"{logdir}/{archive}_smina_run_report_ERROR.log")
            return errors.receptor_or_ligand_not_generated(errMsg, level = "error")
    else:
        errMsg = f"The smina output for '{ptn}' is already generated and you can check it at the '{sminaLog}' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

        octools.print_warning_log(errMsg, f"{logdir}/{archive}_smina_run_report_WARNING.log")
        octools.print_warning(errMsg)
    
    return errors.ok()

def __run_plants(ligandPath: str, ligandDescriptorPath: str, receptorPath: str, receptorDescriptorPath: str, boxPath: str, ptn: str, archive: str, overwrite: bool = False) -> int:
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
    overwrite : bool, optional
        If True, overwrite the output file. Defaults to False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Get the ligand Dir
    ligandDir = os.path.dirname(ligandPath)

    # Check if plantsFiles does not exist
    if not os.path.isdir(f"{ligandDir}/plantsFiles/"):
        errMsg = f"The directory '{ligandDir}/plantsFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok."

        octools.print_error_log(errMsg, f"{logdir}/{archive}_plants_run_report_ERROR.log")
        return errors.dir_does_not_exists(errMsg, level = "error")

    # Flag to denote if its needed to run this protein through plants
    needToRun = False
    # Get the folder for each run
    runPaths = [f"{ligandDir}/plantsFiles"] # glob(f"{ligandDir}/plantsFiles/*") # TODO: add support for multiple runs
    # Check if all files have been processed
    for runPath in runPaths:
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

        # Create a lock for multithreading
        lock = Lock()
        # Start the lock with statement
        with lock:
            # Read the receptor and the ligand (passing the mol2!!!)
            receptor = ocr.Receptor(receptorPath, mol2Path = f"{mol2Path}.mol2", from_json_descriptors = receptorDescriptorPath, name = f"{ptn}_receptor")
            ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptorPath, name = f"{ptn}_ligand")

        # If receptor and ligand are not null
        if receptor and ligand:
            # Set the prepared receptor and ligand paths
            preparedReceptorPath = f"{os.path.dirname(receptorPath)}/prepared_receptor.mol2"
            preparedLigandPath = f"{ligandDir}/prepared_ligand.mol2"

            # For each path in the paths array (will be more than on in case of multiple boxes)
            for runPath in runPaths:
                # Get the run number
                runNumber = 0 # TODO: add support to multiple runs, currently only 0, the code should be something like: runPath.split(os.path.sep)[-1]

                # Parameterizing paths
                plantsLog = f"{runPath}/plants_{runNumber}.log"
                plantsOutput = f"{runPath}/run"
                plantsRankingCsv = f"{plantsOutput}/ranking.csv"
                
                # Create a lock for multithreading
                lock = Lock()
                # Start the lock with statement
                with lock:
                    # Create the smina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                    plants = ocplants.PLANTS(f"{runPath}/conf_plants.txt", boxPath, receptor, preparedReceptorPath, ligand, preparedLigandPath, plantsLog, plantsOutput, name=f"{ptn} PLANTS", overwriteConfig = overwrite)

                # Check if the smina object has been correctly created
                if not plants:
                    errMsg = f"Could not generate plants object for the protein in dir '{ligandDir}'. Error found while trying to run the 'PLANTS' docking software."

                    octools.print_error_log(errMsg, f"{logdir}/{archive}_plants_run_report_ERROR.log")
                    return errors.docking_object_not_generated(errMsg, level = "error")

                # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(plants.preparedLigand) or os.path.getsize(plants.preparedLigand) == 0 or not octools.is_molecule_valid(plants.preparedLigand):
                    # Create a lock for multithreading
                    lock = Lock()
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare ligand
                            result = plants.run_prepare_ligand()
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
                            errMsg = f"Could not run the prepare ligand routine for the protein in dir '{plants.inputLigandPath}'. Error found while trying to run the 'PLANTS' docking software. Error: {e}"

                            octools.print_error_log(errMsg, f"{logdir}/{archive}_plants_run_report_ERROR.log")
                            return errors.ligand_not_prepared(errMsg, level = "error")

                    # Check if the generated ligand has size 0 or is invalid
                    if os.path.getsize(plants.preparedLigand) == 0 or not octools.is_molecule_valid(plants.preparedLigand):
                        errMsg = f"SPORES has made an output of 0kb again for ligand '{plants.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(plants.prepareLigandCmd)}"

                        octools.print_error_log(errMsg, f"{logdir}/{archive}_plants_run_report_ERROR.log")
                        return errors.ligand_not_prepared(errMsg, level = "error")

                # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                if overwrite or not os.path.isfile(plants.preparedReceptor) or os.path.getsize(plants.preparedReceptor) == 0 or not octools.is_molecule_valid(plants.preparedReceptor):
                    # Create a lock for multithreading
                    lock = Lock()
                    # Start the lock with statement
                    with lock:
                        try:
                            # Run the prepare receptor
                            result = plants.run_prepare_receptor()
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
                            errMsg = f"Could not run the prepare receptor routine for the protein in dir '{plants.inputReceptorPath}'. Error found while trying to run the 'PLANTS' docking software. Error: {e}"

                            octools.print_error_log(errMsg, f"{logdir}/{archive}_plants_run_report_ERROR.log")
                            return errors.ligand_not_prepared(errMsg, level = "error")

                    # Check if the generated receptor has size 0 or is invalid
                    if os.path.getsize(plants.preparedReceptor) == 0 or not octools.is_molecule_valid(plants.preparedReceptor):
                        errMsg = f"SPORES has made an output of 0kb for receptor '{plants.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(plants.prepareReceptorCmd)}"
                        octools.print_error_log(errMsg, f"{logdir}/{archive}_plants_run_report_ERROR.log")
                        return errors.receptor_not_prepared(errMsg, level = "error")

                # Check if PLANTS output exists and its size is not 0
                if overwrite or not os.path.isdir(plantsOutput) or not os.path.isfile(plantsRankingCsv) and not os.path.getsize(plantsRankingCsv) == 0:
                    # If there is already a PLANTS output (PLANTS do not run if the folder is already created. And knowing that PLANTS will ALWAYS run if this code is interpreted, just delete the folder if it exists and lets avoid headaches)
                    if os.path.isdir(plantsOutput):
                        # Remove the folder and its contets
                        shutil.rmtree(plantsOutput)

                    # Create a lock for multithreading
                    lock = Lock()
                    # Start the lock with statement
                    with lock:
                        # Run PLANTS
                        plants.run_plants(overwrite=overwrite)
                else:
                    errMsg = f"The PLANTS output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '*/run/plants_<runNumber>.log' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

                    octools.print_warning_log(errMsg, f"{logdir}/{archive}_plants_run_report_WARNING.log")
                    octools.print_warning(errMsg)
        else:
            errMsg = f"Could not generate receptor or ligand object for the protein in dir '{ligandDir}'. Error found while trying to run the 'plants' docking software."

            octools.print_error_log(errMsg, f"{logdir}/{archive}_plants_run_report_ERROR.log")
            return errors.receptor_or_ligand_not_generated(errMsg, level = "error")
    else:
        errMsg = f"The PLANTS output for '{ptn}' is already generated and you can check it at the '{ligandDir}/plantsFiles' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true."

        octools.print_warning_log(errMsg, f"{logdir}/{archive}_plants_run_report_WARNING.log")
        octools.print_warning(errMsg)
    
    return errors.ok()
        
def __core_run_dock(path: str, ligandDir: str, archive: str, dockingAlgorithm: str, overwrite: bool) -> int:
    '''Performs the docking.

    Parameters
    ----------
    path : str
        The path to the protein directory.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    dockingAlgorithm : str
        Which docking algorithm will be used [gnina, vina, smina, plants].
    overwrite : bool
        If the docking output already exists, should it be overwritten?
    ligandDir : str
        If the ligand is not in the same directory as the receptor, this is the path to the ligand directory. By default "". If this is not empty, the ligand will be searched in this directory, otherwise, it will be searched in the same directory as the receptor.

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

        # Initialize an return state 
        returnState = 0
        
        if dockingAlgorithm == "gnina":
            returnState = __run_gnina(ligandPath, ligandDescriptorPath, receptorPath, receptorDescriptorPath, boxPath, ptn, archive, overwrite = overwrite)
        elif dockingAlgorithm == "vina":
            returnState = __run_vina(ligandPath, ligandDescriptorPath, receptorPath, receptorDescriptorPath, boxPath, ptn, archive, overwrite = overwrite)
        elif dockingAlgorithm == "smina":
            returnState = __run_smina(ligandPath, ligandDescriptorPath, receptorPath, receptorDescriptorPath, boxPath, ptn, archive, overwrite = overwrite)
        elif dockingAlgorithm == "plants":
            returnState = __run_plants(ligandPath, ligandDescriptorPath, receptorPath, receptorDescriptorPath, boxPath, ptn, archive, overwrite = overwrite)
        else:
            errMsg = f"Wrong docking algorithm. Expected ['gnina', 'vina', 'smina', 'plants'] and got '{dockingAlgorithm}'."

            octools.print_error_log(errMsg, f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
            return errors.receptor_or_ligand_descriptor_does_not_exist(errMsg, level = "error")
    else:
        if not os.path.isfile(receptorDescriptorPath):
            errMsg = f"There is no receptor descriptor json file for the protein in the path '{receptorDescriptorPath}'. Error found while trying to run the '{dockingAlgorithm}' docking software."
            octools.print_error_log(errMsg, f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
            _ = errors.receptor_or_ligand_descriptor_does_not_exist(errMsg, level = "error")

        if not os.path.isfile(ligandDescriptorPath):
            errMsg = f"There is no ligand descriptor json file for the protein in the path '{ligandDescriptorPath}'. Error found while trying to run the '{dockingAlgorithm}' docking software."
            octools.print_error_log(errMsg, f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
            _ = errors.receptor_or_ligand_descriptor_does_not_exist(errMsg, level = "error")
        return errors.receptor_or_ligand_descriptor_does_not_exist()

    # Check if the docking was successful
    if returnState != 0:
        errMsg = f"Error found while trying to run the '{dockingAlgorithm}' docking software."

        octools.print_error_log(errMsg, f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
        return errors.docking_failed(errMsg, level = "error")

    return errors.ok()

def __thread_run_dock_parallel(arguments: list) -> int:
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
        returnState = __core_run_dock(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])

    return returnState

def __run_dock_parallel(complexList: List[Tuple[str, List[str]]], archive: str, dockingAlgorithm: str, overwrite: bool, desc: str) -> int:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    complexList : List[Tuple[str, List[str]]]
        A list of tuples with the path to the protein directory and a list of ligand directories.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    dockingAlgorithm : str
        Which docking algorithm will be used [vina, smina, plants].
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
            arguments.append((cl[0], ligandDir, archive, dockingAlgorithm, overwrite))

    # If logfile exists, backup it (for error and warnings)
    if os.path.isfile(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log"):
        if not os.path.isdir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past/{archive}_{dockingAlgorithm}_run_report_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")

    if os.path.isfile(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log"):
        if not os.path.isdir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past/{archive}_{dockingAlgorithm}_run_report_WARNING_{time.strftime('%d%m%Y-%H%M%S')}.log")

    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for _ in tqdm(p.imap_unordered(__thread_run_dock_parallel, arguments), total = len(arguments), desc = desc):
                # Clear the memory
                gc.collect()
    except IOError as e:
        octools.print_error_log(f"Problem while running docking software {dockingAlgorithm} in parallel. Exception: {e}", f"{logdir}/{archive}_docking_report.log")
        return errors.docking_failed(f"Problem while running docking software {dockingAlgorithm} in parallel. Exception: {e}", level = "error")

    # Return
    return errors.ok() # FIXME: This should be changed to return the error code in a way to track all docking errors

def __run_dock_no_parallel(complexList: List[Tuple[str, List[str]]], archive: str, dockingAlgorithm: str, overwrite: bool, desc: str) -> int:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_run_dock function.

    Parameters
    ----------
    complexList : List[Tuple[str, List[str]]]
        A list of tuples with the path to the protein directory and a list of ligand directories.
    archive : str
        Which archive will be processed [dudez, pdbbind].
    dockingAlgorithm : str
        Which docking algorithm will be used [vina, smina, plants].
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
    if os.path.isfile(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log"):
        if not os.path.isdir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past/{archive}_{dockingAlgorithm}_run_report_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")

    if os.path.isfile(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log"):
        if not os.path.isdir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past/{archive}_{dockingAlgorithm}_run_report_WARNING_{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # For each file in dirs
        for cl in tqdm(iterable = complexList, total = len(complexList), desc=desc):
            for ligandDir in cl[1]:
                # Call the core dock function (shared between parallel and not parallel)
                __core_run_dock(cl[0], ligandDir, archive, dockingAlgorithm, overwrite)

            # Clear the memory
            gc.collect()
        # Clear the memory
        gc.collect()

    return errors.ok() # FIXME: This should be changed to return the error code in a way to track all docking errors


### Read logs
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

    # Create docking dicts
    vinaDict = { "vina_pose": [], "vina_affinity": [] }
    gninaDict = { "gnina_pose": [], "gnina_affinity": [] }
    sminaDict = { "smina_pose": [], "smina_affinity": [] }
    plantsDict = { "PLANTS_TOTAL_SCORE": [], "PLANTS_SCORE_RB_PEN": [], "PLANTS_SCORE_NORM_HEVATOMS": [], "PLANTS_SCORE_NORM_CRT_HEVATOMS": [], "PLANTS_SCORE_NORM_WEIGHT": [], "PLANTS_SCORE_NORM_CRT_WEIGHT": [], "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": [] }

    # Get run number
    runNumber = 0 # TODO: Add support to multiple runs

    # Dict to hold the protein data
    proteinData = { f"{ptn}-{lgd}": None }

    # VINA
    vinaDir = f"{processDir}/vinaFiles"
    # Parameterize the log path
    logPath = f"{vinaDir}/vina_{runNumber}.log"

    # Check if exists
    if os.path.isfile(logPath):
        # Read the log into dict
        gendict = ocvina.read_vina_log(logPath)

        # For each key, value in vinaDict
        for key, value in gendict.items():
            # Append the value to the vina dict
            vinaDict[key].append(value[0])
    else:
        _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its vina output.")
        # Set the elements in vinaDict as np.NaN
        vinaDict = { "vina_pose": [np.NaN], "vina_affinity": [np.NaN] }


    # SMINA
    sminaDir = f"{processDir}/sminaFiles"
    # Parameterize the log path
    logPath = f"{sminaDir}/smina.log"

    # Check if smina log exists
    if os.path.isfile(logPath): # TODO: Add support to multiple runs
        # Read the log into dict
        gendict = ocsmina.read_smina_log(logPath)

        # For each key, value in sminaDict
        for key, value in gendict.items():
            # Append the value to the smina dict
            sminaDict[key].append(value[0])
    else:
        _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its SMINA output.")
        # Set the elements in sminaDict as np.NaN
        sminaDict = { "smina_pose": [np.NaN], "smina_affinity": [np.NaN] }

    # GNINA
    gninaDir = f"{processDir}/gninaFiles"
    # Parameterize the log path
    logPath = f"{gninaDir}/gnina_{runNumber}.log"

    # Check if exists
    if os.path.isfile(logPath):
        # Read the log into dict
        gendict = ocgnina.read_gnina_log(logPath)

        # For each key, value in gninaDict
        for key, value in gendict.items():
            # Append the value to the gnina dict
            gninaDict[key].append(value[0])
    else:
        _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its gnina output.")
        # Set the elements in gninaDict as np.NaN
        gninaDict = { "gnina_pose": [np.NaN], "gnina_affinity": [np.NaN] }

    # PLANTS
    plantsDir = f"{processDir}/plantsFiles"
    # Parameterize the log path
    logPath = f"{plantsDir}/run/bestranking.csv"

    # Check if exists
    if os.path.isfile(logPath):
        # Read the log into dict
        gendict = ocplants.read_plants_log(logPath)

        # For each key, value in plantsDict
        for key, value in gendict.items():
            # Append the value to the plants dict
            plantsDict[key].append(value[0])
    else:
        _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its PLANTS output.")
        # Set the elements in plantsDict as np.NaN
        plantsDict = { "PLANTS_TOTAL_SCORE": [np.NaN], "PLANTS_SCORE_RB_PEN": [np.NaN], "PLANTS_SCORE_NORM_HEVATOMS": [np.NaN], "PLANTS_SCORE_NORM_CRT_HEVATOMS": [np.NaN], "PLANTS_SCORE_NORM_WEIGHT": [np.NaN], "PLANTS_SCORE_NORM_CRT_WEIGHT": [np.NaN], "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": [np.NaN] }

    # Create the maxLenList
    maxLenList = []

    # Add each score to the list its len greater than 0 (never should be negative)
    if len(vinaDict["vina_pose"]) > 0:
        maxLenList.append(len(vinaDict["vina_pose"]))
    if len(sminaDict["smina_pose"]) > 0:
        maxLenList.append(len(sminaDict["smina_pose"]))
    if len(gninaDict["gnina_pose"]) > 0:
        maxLenList.append(len(gninaDict["gnina_pose"]))
    if len(plantsDict["PLANTS_TOTAL_SCORE"]) > 0:
        maxLenList.append(len(plantsDict["PLANTS_TOTAL_SCORE"]))
    
    # Check if the list is empty
    if len(maxLenList) == 0:
        # Set the maxLen to 1
        maxLen = 1
    else:
        # Get the number of elements of the dict with the largest number of elements
        maxLen = max(maxLenList)

    # Add the concatenated the dicts. The single elements are repeated to match the largest dict to the proteinData dict using ptn as the key
    proteinData[f"{ptn}-{lgd}"] = vaex.from_dict(
        {
            **{
                "Protein": [ptn for _ in range(maxLen)],
                "Ligand": [lgd for _ in range(maxLen)],
                "type": [tp for _ in range(maxLen)]
            },
            **vinaDict,
            **sminaDict,
            **gninaDict,
            **plantsDict
        }
    )

    # Clean the memory
    del vinaDict, sminaDict, plantsDict#, gninaDict

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
    with octools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_read_log(arguments[0])

def __read_log_parallel(ptnDirs: List[Tuple[str, str]], desc: str) -> Dict[str, vdf.DataFrameLocal]:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    ptnDirs : List[Tuple[str, str]]
        A list of tuples with the directory and the ligand type (ligand, decoy, candidate).
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary containing the dataframes of the logs.

    Raises
    ------
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []

    # For each file in the glob
    for ptnDir in ptnDirs:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((ptnDir, None))

    # If logfile exists, backup it for vina, smina and plants (for error and warnings)
    if os.path.isfile(f"{logdir}/read_log_ERROR.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            octools.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/read_log_ERROR.log", f"{logdir}/read_log_past/read_log_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Dict to store the read data
    data = {}

    try:
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Perform the multi process
            for innerData in tqdm(p.imap_unordered(__thread_read_log_parallel, arguments), total = len(arguments), desc = desc):
                # Get the key from innerData
                key = list(innerData.keys())[0]
                # Set the value of the key in data to the value of the key in innerData
                data[key] = innerData[key]
                # Clean the memory
                del innerData
                gc.collect()
                
    except IOError as e:
        octools.print_error_log(f"Problem while reading logs in parallel. Exception: {e}", f"{logdir}/read_log_ERROR_report.log")
        octools.print_error(f"Problem while reading logs in parallel. Exception: {e}")

    return data

def __read_log_no_parallel(ptnDirs: List[Tuple[str, str]], desc: str) -> Dict[str, vdf.DataFrameLocal]:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_read_log function.

    Parameters
    ----------
    ptnDirs : List[Tuple[str, str]]
        A list of tuples with the directory and the ligand type (ligand, decoy, candidate).
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    Dict[str, vdf.DataFrameLocal]
        A dictionary with the protein name as the key and a dictionary with the vina, smina and plants dataframes as the value.
    
    Raises
    ------
    None
    '''

    # Dict to store the read data
    data = {}

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
        for ptnDir, tp in tqdm(iterable = ptnDirs, total = len(ptnDirs), desc = desc):
            # Call the core read log function (shared between parallel and not parallel) and store the data into the data dict
            data.update(__core_read_log((ptnDir, tp)))
            # Clear the memory
            gc.collect()

    return data

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
def verify_integrity(chosenArchive: str, spacing: float = 0.33) -> None:
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
                __run_p2rank(dir, fin)

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
                    __prepare_molecule(f"{dir}/{ptn}_protein.pdb", False, "receptor", archive, sanitize = True)
                    # If the file still does not exists...
                    if not os.path.isfile(f"{dir}/{ptn}_protein_descriptors.json") or os.path.getsize(f"{dir}/{ptn}_protein_descriptors.json") == 0:
                        # REPORT
                        octools.print_error(f"Unable to generate the receptor descriptor file for '{dir}'...")
                        octools.print_error_log(f"Unable to generate the receptor descriptor file dir for '{dir}'...", f"{logdir}/{archive}_integrity_report.log")
                        failed = failed + 1
                        continue

    octools.printv(f"Integrity check of the PDBbind database accomplished. Success rate: {((lenDirs - failed) / lenDirs) * 100}% ({(lenDirs - failed)}/{lenDirs})")
    return None

def convert_debug_to_production(chosenArchive: str, chosenAlgorithm: str = "ac", strict: bool = False, removeDebug: bool = False) -> None:
    '''Converts debug folders to production mode. It is required to choose an algorithm which will be used furtherly in the pipeline.

    Parameters
    ----------
    chosenArchive : str
        The archive to be converted. The options are [dudez, pdbbind].
    chosenAlgorithm : str, optional
        The algorithm to be used in the pipeline. The default is "ac". The short code for the algorithms are
            AffinityPropagation: ap, 
            AgglomerativeClustering: ac
            Birch: bi
            DBSCAN: db
            KMeans:  km
            MeanShift: ms
            MiniBatchKMeans: mb
            NoCluster: na
            OPTICS: op
            SpectralClustering: sc
            Ward: wa
    strict : bool, optional
        If True does not convert the data even if there is only one dir, if False will convert the data if the protein has only one dir (this is good when you ran with only one algorithm, some proteins may have been run with "na"). The default is False.
    removeDebug : bool, optional
        If True removes the debug folder after the conversion, if False keeps the debug folder. The default is False.
    '''

    # Generate boxes for all receptors
    octools.printv("Converting p2rank debug to production file tree.")

    # Get all dirs paths in the DUDEz database
    dirs = glob(f"{chosenArchive}/*")

    # Set the allowed values
    allowed = ["ap", "ac", "bi", "db", "km", "ms", "mb", "na", "op", "sc", "wa"]

    # Redirect output to tqdm.write
    with octools.redirect_to_tqdm():
        # For each directory in the database folder
        for dir in tqdm(iterable=dirs, total=len(dirs)):
            # Print text
            octools.printv(f"Processing '{dir}'.")

            # Parameterize the p2rank dir
            p2rankDir = f"{dir}/p2rank"

            # Flag to check if the algorithm folder has been found
            hasDir = False

            # Get all the dirs which are in the allowed values
            p2rankFiles = [d for d in glob(f"{p2rankDir}/*") if octools.is_algorithm_allowed(d) and os.path.isdir(d)]

            # Parameterize the amount of dirs
            p2rankFilesLen = len(p2rankFiles)

            # If there is any dir
            if p2rankFilesLen > 0:
                # If there is only one file
                if p2rankFilesLen == 1 and not strict:
                    octools.print_info(f"There is only one file.")
                    # Set the hasDir as true
                    hasDir = True
                    # Get the boxes
                    boxes = glob(f"{p2rankFiles[0]}/*")
                    # If no box is found (folders WILL NOT BE REMOVED)
                    if len(boxes) < 1:
                        octools.print_error(f"The protein '{dir}' has no box!!!!!")
                        octools.print_error_log(f"The protein '{dir}' has no box!!!!!", f"{logdir}/{chosenArchive}_conversion_report.log")
                        continue
                    # Get the algorithm name
                    algorithm = p2rankFiles[0].split(os.path.sep)[-1]
                    # For each box found
                    for box in boxes:
                        # Create the destination box name
                        boxDest = os.path.basename(box).replace(f"_{algorithm}","")
                        # Copy the box to the parent directory
                        shutil.copyfile(box, f"{p2rankDir}/{boxDest}")
                else:
                    for p2rankFile in p2rankFiles:
                        # Get the algorithm name
                        algorithm = p2rankFile.split(os.path.sep)[-1]
                        if algorithm == chosenAlgorithm:
                            # Set the hasDir as true
                            hasDir = True
                            # Get the boxes
                            boxes = glob(f"{p2rankFile}/*")
                            # If no box is found (folders WILL NOT BE REMOVED)
                            if len(boxes) < 1:
                                octools.print_error(f"The protein '{dir}' has no box!!!!!")
                                octools.print_error_log(f"The protein '{dir}' has no box!!!!!", f"{logdir}/{chosenArchive}_conversion_report.log")
                                continue
                            # Get the algorithm name
                            algorithm = p2rankFile.split(os.path.sep)[-1]
                # If the algorithm folder has been found
                if hasDir:
                    # Check if remove is set
                    if removeDebug:
                        # Print to the user the information
                        octools.print_info(f"Removing files for '{dir}'")
                        # For each file
                        for p2rankFile in p2rankFiles:
                            # Remove the folder and its contets
                            shutil.rmtree(p2rankFile)
                else:
                    octools.print_error(f"The algorithm '{chosenAlgorithm}' has not been found for the protein '{dir}'.")
                    octools.print_error_log(f"The algorithm '{chosenAlgorithm}' has not been found for the protein '{dir}'.", f"{logdir}/{chosenArchive}_conversion_report.log")
            else:
                octools.printv(f"Nothing to convert for '{dir}'. Skipping...")
                continue
    return None

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

    # If is multiprocess
    if args.multiprocess:
        # Prepare the pdbbind
        __prepare_parallel(paths, overwrite, archive, sanitize, spacing, label)
    else:
        # Prepare the database
        __prepare_no_parallel(paths, overwrite, archive, sanitize, spacing, label)

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

    # Make archive lowercase
    archive = os.path.basename(archive).lower()

    # Find which kind of archive it will be
    if archive == "dudez":
        chosenArchive = dudez_archive
        label = f"DUDEz proteins"
        # Get all dirs paths in the database
        dirs = glob(f"{chosenArchive}/*")
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
        label = "PDBbind proteins"
        # Get all dirs paths in the database filtering for pdbbind
        dirs = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index']]
    else:
        octools.print_error(f"Not valid archive type. Expected one of ['dudez', 'pdbbind'] and found {archive}.")
        return None

    # Generate boxes for all receptors
    octools.printv("Generating P2Rank files.")
    # If is multiprocess
    if args.multiprocess:
        # Run the p2rank in parallel
        __p2rank_parallel(dirs, overwrite, archive, label)
    else:
        # Run the p2rank in serial
        __p2rank_no_parallel(dirs, overwrite, archive, label)
    return None

def run_dock(archive: str, dockingAlgorithm: str, overwrite: bool = False) -> int:
    '''Run docking.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind].
    dockingAlgorithm : str
        The docking algorithm to be used. The options are [vina, smina, plants].
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
        
    # Decide if multprocessing will be used
    if args.multiprocess:
        return __run_dock_parallel(complexList, archive, dockingAlgorithm, overwrite, f"Processing {archive}")
    else:
        return __run_dock_no_parallel(complexList, archive, dockingAlgorithm, overwrite, f"Processing {archive}")

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
        
    # Make data be None (in case of failure)
    data = None

    # Decide if multprocessing will be used
    if args.multiprocess:
        data = __read_log_parallel(processDirs, f"Processing {archive}")
    else:
        data = __read_log_no_parallel(processDirs, f"Processing {archive}")

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

