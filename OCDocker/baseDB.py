#!/usr/lib/python3

# Imports
###############################################################################
from genericpath import isfile
import os
import gc
import time
import shutil
import rdkit
from glob import glob
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import pandas as pd

from OCDocker.Initialise import *
import OCDocker.Ligand as ocl
import OCDocker.Vina as ocvina
import OCDocker.Receptor as ocr
import OCDocker.Smina as ocsmina
import OCDocker.PLANTS as ocplants
import OCDocker.Toolbox as octools
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
        Which archive will be processed [dudez, pdbbind, astex].

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    fin = ""

    if archive == "astex":
        # Set the input file name path
        fin = f"{dir}/protein.pdb"
    elif archive == "dudez":
        # Set the input file name path
        fin = f"{dir}/rec.crg.pdb"
    elif archive == "pdbbind":
        # If is the index path
        if os.path.basename(dir) in ['index', 'db']:
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
        Tuple with the arguments to be passed to __core_p2rank. The arguments are: (dir, overwrite, archive). Where dir is the path where the data is, overwrite is a flag for demanding file overwrite and archive is which archive will be processed [dudez, pdbbind, astex].

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
    # Return
    return None

def __p2rank_parallel(dirs: List[str], overwrite: bool, archive: str, desc: str) -> None:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    dirs: List[str]
        List of directories to be processed.
    overwrite: bool
        Flag for demanding file overwrite.
    archive: str
        Which archive will be processed [dudez, pdbbind, astex].
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
    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for _ in tqdm(p.imap_unordered(__thread_p2rank, arguments), total = len(arguments), desc = desc):
            # Clear the memory
            gc.collect()
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
        Which archive will be processed [dudez, pdbbind, astex].
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


### Box handling
def __run_create_vina_conf_from_box(path: str, fin: str, boxPath: str = "") -> None:
    '''Creates vina conf file from box.

    Parameters
    ----------
    path : str
        The path to the folder where the files will be generated.
    fin : str
        The path of the protein.
    boxPath : str
        The path to the box file. If empty, it will try to look for a p2rank dir inside <path>.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Run vina
    ocvina.generate_vina_files_database(path, fin, boxPath = boxPath)

    return None

def __run_create_plants_conf_from_box(path: str, fin: str, ligand: str, spacing: float, boxPath: str = "") -> None:
    '''Creates PLANTS conf file from box.

    Parameters
    ----------
    path : str
        Directory of the protein to run p2rank.
    protein : str
        Protein path.
    ligand : str
        Ligand name to be used in conf file.
    spacing : float
        Extra spacing.
    boxPath : str
        The path to the box file. If empty, it will try to look for a p2rank dir inside <path>.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Run vina
    ocplants.generate_plants_files_database(path, fin, ligand, spacing, boxPath = boxPath)

    return None


### Prepare
def __prepare_molecule(mol: rdkit.Chem.rdchem.Mol, overwrite: bool, moltype: str, dbName: str, sanitize: bool, molName: str = "molecule", targetCentroid: Union[Tuple[float, float, float], rdkit.Geometry.rdGeometry.Point3D] = None) -> None: # type: ignore
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
    molName : str
        Name of the molecule.
    targetCentroid : Tuple[float, float, float] | rdkit.Geometry.rdGeometry.Point3D, optional
        Centroid of the target. If not provided, the centroid of the molecule will be used.

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
    
    if overwrite or not os.path.isfile(f"{molPath}/{moltype}_descriptors.json"):
        if moltype == "ligand":
            # Safe create plantsFiles, vinaFiles and sminaFiles dirs
            _ = octools.safe_create_dir(f"{molPath}/plantsFiles")
            _ = octools.safe_create_dir(f"{molPath}/vinaFiles")
            _ = octools.safe_create_dir(f"{molPath}/sminaFiles")
            try:
                # Create the ligand object
                m = ocl.Ligand(mol, molName, sanitize = sanitize)
                # Create a box around the ligand
                m.create_box(centroid = targetCentroid, overwrite = overwrite)
            # If m is not valid
            except Exception as e:
                _ = errors.parse_molecule(f"The molecule '{mol}' could not be parsed!", "error")
                octools.print_error_log(f"The molecule '{mol}' could not be parsed!", f"{logdir}/{dbName}_error_Parse.log")
                return None
        elif moltype == "receptor":
                # If is a tuple
                if type(mol) == tuple:
                    try:
                        # Create the receptor object
                        m = ocr.Receptor(mol[0], molName, mol2Path = mol[1])
                    except Exception as e:
                        _ = errors.parse_molecule(f"The molecule '{mol[0]}' could not be parsed! Error {e}", "error")
                        octools.print_error_log(f"The molecule '{mol[0]}' could not be parsed! Error {e}", f"{logdir}/{dbName}_error_Parse.log")
                        return None
                else:
                    try:
                        # Create the receptor object
                        m = ocr.Receptor(mol, molName)
                    except Exception as e:
                        _ = errors.parse_molecule(f"The molecule '{mol}' could not be parsed! Error {e}", "error")
                        octools.print_error_log(f"The molecule '{mol}' could not be parsed! Error {e}", f"{logdir}/{dbName}_error_Parse.log")
                        return None
        else:
            _ = errors.unknown("Unknown molecule type", "error")
            return None

        # Test if the molecule is valid
        if not m or not m.is_valid():
            _ = errors.malformed_molecule(f"The molecule '{mol}' is not valid! Its descriptors are malformed. Please check it manually!", "error")
            octools.print_error_log(f"The molecule '{mol}' is not valid! Its descriptors are malformed. Please check it manually!", f"{logdir}/{dbName}_error_Parse.log")
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
            shutil.move(mol, f"{mol}/{molName}/ligand.{molTmp[-1]}")

    # Get the list of dirs to process
    processDirs = [dirToProcess for dirToProcess in glob(f"{dirsToProcess}/*") if os.path.isdir(dirsToProcess)]

    # For each directory (check to see if it is needed to generate descriptors)
    for processDir in processDirs:
        # Set the fligand name as the ligand file path
        fligand = f"{processDir}/ligand.smi"
        # Safe create plantsFiles, vinaFiles and sminaFiles dirs
        _ = octools.safe_create_dir(f"{processDir}/vinaFiles")
        _ = octools.safe_create_dir(f"{processDir}/sminaFiles")
        _ = octools.safe_create_dir(f"{processDir}/plantsFiles")
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
        Which archive to use. Options are [dudez, pdbbind, astex].
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
    preparedReceptor = f"{path}/receptor_prepared.mol2"

    # Prepare the receptor
    __prepare_molecule((fin, fout), overwrite, "receptor", archive, sanitize = sanitize)

    # Parameterize the compounds folders
    ligands_d = os.path.join(path, "compounds", "ligands")       # known ligands
    decoys_d = os.path.join(path, "compounds", "decoys")         # known decoys
    candidates_d = os.path.join(path, "compounds", "candidates") # unknown ligands

    # Check if there is no target centroid data
    if targetCentroid is None:
        # Parameterize the reference ligand name (pdb and mol2)
        ref_ligand_pdb = os.path.join(f"{path}/reference_ligand.pdb")
        ref_ligand_mol2 = os.path.join(f"{path}/reference_ligand.mol2")

        # Check if the reference ligand does not exist (extensions in order: pdb, mol2)
        if os.path.isfile(ref_ligand_pdb):
            # Set the target centroid as the centroid of the ligand from the pdb file
            targetCentroid = ocl.get_centroid(ref_ligand_pdb, sanitize = sanitize)
        elif os.path.isfile(ref_ligand_mol2):
            # Set the target centroid as the centroid of the ligand from the mol2 file
            targetCentroid = ocl.get_centroid(ref_ligand_mol2, sanitize = sanitize)
        else:
            #octools.print_error_log(f"Could not find the file '{ref_ligand_pdb}' or '{ref_ligand_mol2}' for the molecule '{path}' and a target centroid has not been provided. This molecule will not be processecvcd.", f"{logdir}/{archive}_error_Parse.log")
            return errors.file_do_not_exist(f"Could not find the file '{ref_ligand_pdb}' or '{ref_ligand_mol2}' for the molecule '{path}' and a target centroid has not been provided. This molecule will not be processed.", level = "error")

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

    print(processDirs)
    # For each dir to be processed
    for processDir in processDirs:
        # Check if there is a box for the ligand
        boxCount = len(glob(f"{processDir}/boxes/box*.pdb"))
        
        print(boxCount)
        print(len(glob(f"{processDir}/vinaFiles/*")))
        print(len(glob(f"{processDir}/vinaFiles/*")) == 0)

        # If overwrite mode is on or there is not the same amount of box files as folders in vinaFiles folder
        if boxCount == 0 or len(glob(f"{processDir}/vinaFiles/*")) != boxCount or len(glob(f"{processDir}/vinaFiles/*")) == 0 or overwrite:
            # Create the vina inputs from the boxes
            print(f"ocvina.generate_vina_files_database({processDir}, '{fin}', boxPath = '{processDir}/boxes/box0.pdb')")
            ocvina.generate_vina_files_database({processDir}, fin, boxPath = f"{processDir}/boxes/box0.pdb")
        else:
            octools.print_info(f"The protein '{processDir}' already has its vina file generated, skipping its execution.")

        # If overwrite mode is on or there is not the same amount of box files as folders in vinaFiles folder
        if boxCount == 0 or len(glob(f"{processDir}/plantsFiles/*")) != boxCount or len(glob(f"{processDir}/plantsFiles/*")) == 0 or overwrite:
            # Set the fligand variable to the dir + ligandName + .mol2
            fligand = f"{processDir}/ligand.mol2"
            # Create the PLANTS inputs from the boxes
            print(f"ocplants.generate_plants_files_database(f'{processDir}/plantsFiles/conf_plants.conf', '{fin}', boxPath = '{processDir}/boxes/box0.pdb')")
            ocplants.generate_plants_files_database(f"{processDir}/plantsFiles/conf_plants.conf", preparedReceptor, fligand, spacing, boxPath = f"{processDir}/boxes/box0.pdb")
        else:
            octools.print_info(f"The protein '{processDir}' already has its PLANTS file generated, skipping its execution.")

        # If overwrite mode is on or there not any conf file in the sminaFiles folder
        if len(glob(f"{processDir}/sminaFiles/*.conf")) == 0 or overwrite:
            # Create the smina d
            # Create the smina inputs
            ocsmina.gen_smina_conf(f"{processDir}/sminaFiles/conf_smina.conf", fin)
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
        The archive name. Options are [astex, dudez, pdbbind].
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

    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for _ in tqdm(p.imap_unordered(__thread_prepare, arguments), total = len(arguments), desc = desc):
            # Clear the memory
            gc.collect()
    
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
        The archive name. Options are [astex, dudez, pdbbind].
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


### Get
def __core_get(dir: str, archive: str) -> Union[Tuple[str, ocr.Receptor, ocl.Ligand], None]:
    '''Loads in memory a pair receptor-ligand and then return them in a tuple alongside with the protein name gotten from file path.

    Parameters
    ----------
    dir : str
        The directory to be processed.
    archive : str
        The archive name. Options are [astex, dudez, pdbbind].

    Returns
    -------
    Tuple[str, ocr.Receptor, ocl.Ligand] | None
        The tuple containing the protein name, the receptor and the ligand. If the protein is not valid, returns None.

    Raises
    ------
    None
    '''

    # Find ptn name
    ptn = dir.split(os.path.sep)[-1]
    # If is the index directory, ignore
    if dir in ['index', 'db']:
        return None
    # Find which kind of archive it will be
    if archive == "astex":
        pass
    elif archive == "dudez":
        pass
    elif archive == "pdbbind":
        # Set the input file name path (to generate the box and data about the protein)
        receptorPath = f"{dir}/{ptn}_protein.pdb"
        # Set the ligand file name path (to generate data about the ligand)
        ligandPath = f"{dir}/{ptn}_ligand.mol2"
        # If the complex has all descriptors for protein AND ligand
        if os.path.isfile(f"{dir}/{ptn}_protein_descriptors.json") and os.path.isfile(f"{dir}/{ptn}_ligand_descriptors.json"):
            # Read the receptor and the ligand
            receptor = ocr.Receptor(receptorPath, from_json_descriptors = f"{dir}/{ptn}_protein_descriptors.json", name = f"{ptn}_receptor")
            ligand = ocl.Ligand(ligandPath, from_json_descriptors = f"{dir}/{ptn}_ligand_descriptors.json", name = f"{ptn}_ligand")
            # Return them
            return (ptn, receptor, ligand)
    return None

def __thread_get_parallel(arguments: Tuple[str, str]) -> Union[Tuple[str, ocr.Receptor, ocl.Ligand], None]:
    '''Thread aid function to call __core_get.

    Parameters
    ----------
    arguments : Tuple[str, str]
        The arguments to be passed to __core_get. Its arguments are: (dir, archive). See __core_get for more information.

    Returns
    -------
    Tuple[str, ocr.Receptor, ocl.Ligand] | None
        The tuple containing the protein name, the receptor and the ligand.

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call core get function (shared between thread and not thread)
        return __core_get(arguments[0], arguments[1])

def __get_parallel(dirs: List[str], archive: str, desc: str) -> Dict[str, Tuple[ocr.Receptor, ocl.Ligand]]:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    dirs : List[str]
        The list of directories to be processed.
    archive : str
        The archive name. Options are [astex, dudez, pdbbind].
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    Dict[str, Tuple[ocr.Receptor, ocl.Ligand]]
        The dictionary containing the protein name as key and the tuple containing the receptor and the ligand as value.

    Raises
    ------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for dir in dirs:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((dir, archive))
    # Dict of elements
    databaseDict = dict()
    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for complexData in tqdm(p.imap_unordered(__thread_get_parallel, arguments), total = len(arguments), desc = desc):
            # If the complex data is not empty
            if complexData:
                databaseDict[complexData[0]] = (complexData[1], complexData[2])
            # Clear the memory
            gc.collect()
    # Return
    return databaseDict

def __get_no_parallel(dirs: List[str], archive: str, desc: str) -> Dict[str, Tuple[ocr.Receptor, ocl.Ligand]]:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_get function.

    Parameters
    ----------
    dirs : List[str]
        The list of directories to be processed.
    archive : str
        The archive name. Options are [astex, dudez, pdbbind].
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    Dict[str, Tuple[ocr.Receptor, ocl.Ligand]]
        The dictionary containing the protein name as key and the tuple containing the receptor and the ligand as value.

    Raises
    ------
    None
    '''

    # Dict of elements
    databaseDict = dict()
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for dir in tqdm(iterable=dirs, total=len(dirs), desc=desc):
            # Call the core get function
            data = __core_get(dir, archive)
            # Check if data is None
            if not data:
                octools.print_error(f"Error while processing '{dir}'.")
                continue
            # Add them to the dict using the protein as the key
            databaseDict[data[0]] = (data[1], data[2])
            # Clear the memory
            gc.collect()
        return databaseDict


### Docking
def __sub_core_run_dock(receptorPath: str, ligandPath: str, receptorDir: str, ligandDir: str, archive: str, dockingAlgorithm: str, receptorDescriptor: str, ligandDescriptor: str, overwrite: bool) -> int:
    '''Performs the docking.

    Parameters
    ----------
    receptorPath : str
        The path to the receptor file.
    ligandPath : str
        The path to the ligand file.
    receptorDir : str
        The path to the receptor directory.
    ligandDir : str
        The path to the ligand directory.
    archive : str
        The archive name. Options are [astex, dudez, pdbbind].
    dockingAlgorithm : str
        The docking algorithm to be used. Options are [vina, smina, plants].
    receptorDescriptor : str
        The path to the receptor descriptor file.
    ligandDescriptor : str
        The path to the ligand descriptor file.
    overwrite : bool
        If the docking results should be overwritten.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # If the complex has all descriptors for protein AND ligand
    if os.path.isfile(receptorDescriptor) and os.path.isfile(ligandDescriptor):
        # Find protein name
        ptn = receptorPath.split(os.path.sep)[-1]
        
        # Parameterize the receptor and ligand names
        receptorName = "receptor"
        ligandName = "ligand"

        # If running vina
        if dockingAlgorithm == "vina":
            # Flag to denote if its needed to run this protein through vina
            needToRun = False
            # Check if the vinaFiles directory exists
            if not os.path.isdir(f"{ligandDir}/vinaFiles"):
                octools.print_error_log(f"The directory '{ligandDir}/vinaFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                return errors.dir_does_not_exists(f"The directory '{ligandDir}/vinaFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok.", level = "error")
            # Get the folder for each run
            runPaths = glob(f"{ligandDir}/vinaFiles/*")
            # Check if all files have been processed
            for runPath in runPaths:
                # Get the run number
                runNumber = runPath.split(os.path.sep)[-1]
                # If the output does not exist or overwrite flag is set to true
                if overwrite or not os.path.isfile(f"{runPath}/vina_{runNumber}.log") or not os.path.isfile(f"{runPath}/vina_{runNumber}.pdbqt"):
                    needToRun = True
                    break
            # If is needed to run (at least one protein)
            if needToRun:
                # Read the receptor and the ligand
                receptor = ocr.Receptor(receptorPath, from_json_descriptors = receptorDescriptor, name = f"{ptn}_receptor")
                ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptor, name = f"{ptn}_ligand")
                # If receptor and ligand are not null
                if receptor and ligand:
                    # For each path in the paths array (will be more than on in case of multiple boxes)
                    for runPath in runPaths:
                        # Get the run number
                        runNumber = runPath.split(os.path.sep)[-1]
                        # Parameterizing paths
                        vinaLog = f"{runPath}/vina_{runNumber}.log"
                        vinaOutput = f"{runPath}/vina_{runNumber}.pdbqt"
                        # Create the vina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                        vina = ocvina.Vina(f"{runPath}/conf_vina.txt", f"{receptorDir}/p2rank/box{runNumber}.pdb", receptor, f"{receptorDir}/{receptorName}.pdbqt", ligand, f"{ligandDir}/{ligandName}.pdbqt", vinaLog, vinaOutput, name=f"{ptn}_run_{runNumber}")
                        # Check if the vina object has been correctly created
                        if not vina:
                            octools.print_error_log(f"Could not generate vina object for the protein in dir '{ligandDir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                            return errors.docking_object_not_generated(f"Could not generate vina object for the protein in dir '{ligandDir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                        # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                        if overwrite or not os.path.isfile(vina.preparedLigand) or os.path.getsize(vina.preparedLigand) == 0 or not octools.is_molecule_valid(vina.preparedLigand):
                            # Run the prepare ligand
                            _ = vina.run_prepare_ligand(useOpenBabel=False) # useOpenBabel has proven to be a dangerous option, it is better to avoid its use for
                            # Check if the generated ligand has size 0 or is invalid
                            if os.path.getsize(vina.preparedLigand) == 0 or not octools.is_molecule_valid(vina.preparedLigand):
                                octools.print_warning_log(f"The prepare ligand script has made an output of 0kb for ligand '{vina.preparedLigand}', this is wierd. Trying to run it again.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                                octools.print_warning(f"The prepare ligand script has made an output of 0kb for ligand '{vina.preparedLigand}', this is wierd. Trying to run it again.")
                                # Run again the prepare ligand
                                _ = vina.run_prepare_ligand(useOpenBabel=False) # useOpenBabel has proven to be a dangerous option, it is better to avoid its use for
                                # Check again if the generated ligand has size 0 or is invalid
                                if os.path.getsize(vina.preparedLigand) == 0 or not octools.is_molecule_valid(vina.preparedLigand):
                                    octools.print_error_log(f"The prepare ligand script has made an output of 0kb again for ligand '{vina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareLigandCmd)}", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                                    return errors.ligand_not_prepared(f"The prepare ligand script has made an output of 0kb again for ligand '{vina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareLigandCmd)}", level = "error")
                        # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                        if overwrite or not os.path.isfile(vina.preparedReceptor) or os.path.getsize(vina.preparedReceptor) == 0 or not octools.is_molecule_valid(vina.preparedReceptor):
                            # Run the prepare receptor
                            _ = vina.run_prepare_receptor(useOpenBabel=False) # useOpenBabel has proven to be a dangerous option, it is better to avoid its use for now
                            # Check if the generated receptor has size 0 or is invalid
                            if os.path.getsize(vina.preparedReceptor) == 0 or not octools.is_molecule_valid(vina.preparedReceptor):
                                octools.print_warning_log(f"The prepare receptor has made an output of 0kb for ligand '{vina.preparedReceptor}', this is wierd. Trying to run it again.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                                octools.print_warning(f"The prepare receptor has made an output of 0kb for ligand '{vina.preparedReceptor}', this is wierd. Trying to run it again.")
                                # Run again the prepare receptor
                                _ = vina.run_prepare_receptor(useOpenBabel=False) # useOpenBabel has proven to be a dangerous option, it is better to avoid its use for
                                # Check again if the generated receptor has size 0 or is invalid
                                if os.path.getsize(vina.preparedReceptor) == 0 or not octools.is_molecule_valid(vina.preparedReceptor):
                                    octools.print_error_log(f"The prepare receptor has made an output of 0kb again for receptor '{vina.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareReceptorCmd)}", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                                    return errors.receptor_not_prepared(f"The prepare receptor has made an output of 0kb again for receptor '{vina.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareReceptorCmd)}", level = "error")
                        # Check if vina output exists
                        if overwrite or not os.path.isfile(vinaOutput) or not os.path.isfile(vinaLog):
                            # Run vina
                            vina.run_vina()
                        else:
                            octools.print_warning_log(f"The vina output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '{runPath}/vina_{runNumber}.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                            octools.print_warning(f"The vina output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '{runPath}/vina_{runNumber}.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true")
                else:
                    octools.print_error_log(f"Could not generate receptor or ligand object for the protein in dir '{ligandDir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                    return errors.receptor_or_ligand_not_generated(f"Could not generate receptor or ligand object for the protein in dir '{ligandDir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
            else:
                octools.print_warning_log(f"The vina output for '{ptn}' for all boxes is already generated and you can check it at the '{ligandDir}/vinaFiles/*/vina_<runNumber>.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                octools.print_warning(f"The vina output for '{ptn}' for all boxes is already generated and you can check it at the '{ligandDir}/vinaFiles/*/vina_<runNumber>.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
        elif dockingAlgorithm == "smina":
            # Set the run path
            runPath = f"{ligandDir}/sminaFiles/"
            # Parameterizing paths
            sminaLog = f"{runPath}/smina.log"
            sminaOutput = f"{runPath}/smina.pdbqt"
            # Check if sminaFiles does not exist
            if not os.path.isdir(runPath):
                octools.print_error_log(f"The directory '{runPath}' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                return errors.dir_does_not_exists(f"The directory '{runPath}' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok.", level = "error")
            # If is needed to run (overwrite is set or no output is produced)
            if overwrite or not os.path.isfile(sminaLog) or not os.path.isfile(sminaOutput):
                # Read the receptor and the ligand
                receptor = ocr.Receptor(receptorPath, from_json_descriptors = receptorDescriptor, name = f"{ptn}_receptor")
                ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptor, name = f"{ptn}_ligand")
                # If receptor and ligand are not null
                if receptor and ligand:
                    # Create the smina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                    smina = ocsmina.Smina(f"{runPath}/conf_smina.txt", receptor, f"{receptorDir}/{ptn}_protein.pdbqt", ligand, f"{ligandDir}/{ptn}_ligand.pdbqt", sminaLog, sminaOutput, name=f"{ptn}_smina")
                    # Check if the smina object has been correctly created
                    if not smina:
                        octools.print_error_log(f"Could not generate smina object for the protein in dir '{ligandPath}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                        return errors.docking_object_not_generated(f"Could not generate smina object for the protein in dir '{ligandDir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                    # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                    if overwrite or not os.path.isfile(smina.preparedLigand) or os.path.getsize(smina.preparedLigand) == 0 or not octools.is_molecule_valid(smina.preparedLigand):
                        # Run the prepare ligand
                        _ = smina.run_prepare_ligand()
                        # Check if the generated ligand has size 0 or is invalid
                        if os.path.getsize(smina.preparedLigand) == 0 or not octools.is_molecule_valid(smina.preparedLigand):
                            octools.print_warning_log(f"The prepare ligand script has made an output of 0kb for ligand '{smina.preparedLigand}', this is wierd. Trying to run it again.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                            octools.print_warning(f"The prepare ligand script has made an output of 0kb for ligand '{smina.preparedLigand}', this is wierd. Trying to run it again.")
                            # Run again the prepare ligand
                            _ = smina.run_prepare_ligand()
                            # Check again if the generated ligand has size 0 or is invalid
                            if os.path.getsize(smina.preparedLigand) == 0 or not octools.is_molecule_valid(smina.preparedLigand):
                                octools.print_error_log(f"The prepare ligand script has made an output of 0kb again for ligand '{smina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareLigandCmd)}", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                                return errors.ligand_not_prepared(f"The prepare ligand script has made an output of 0kb again for ligand '{smina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareLigandCmd)}", level = "error")
                    # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                    if overwrite or not os.path.isfile(smina.preparedReceptor) or os.path.getsize(smina.preparedReceptor) == 0 or not octools.is_molecule_valid(smina.preparedReceptor):
                        # Run the prepare receptor
                        _ = smina.run_prepare_receptor()
                        # Check if the generated receptor has size 0 or is invalid
                        if os.path.getsize(smina.preparedReceptor) == 0 or not octools.is_molecule_valid(smina.preparedReceptor):
                            octools.print_warning_log(f"The prepare receptor has made an output of 0kb for ligand '{smina.preparedReceptor}', this is wierd. Trying to run it again.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                            octools.print_warning(f"The prepare receptor has made an output of 0kb for ligand '{smina.preparedReceptor}', this is wierd. Trying to run it again.")
                            # Run again the prepare receptor
                            _ = smina.run_prepare_receptor()
                            # Check again if the generated receptor has size 0 or is invalid
                            if os.path.getsize(smina.preparedReceptor) == 0 or not octools.is_molecule_valid(smina.preparedReceptor):
                                octools.print_error_log(f"The prepare receptor has made an output of 0kb again for receptor '{smina.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareReceptorCmd)}", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                                return errors.receptor_not_prepared(f"The prepare receptor has made an output of 0kb again for receptor '{smina.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareReceptorCmd)}", level = "error")
                    # Run smina (no need to recheck for overwrite or output existance because it is already done some lines ago)
                    smina.run_smina()
                else:
                    octools.print_error_log(f"Could not generate receptor or ligand object for the protein in dir '{ligandDir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                    return errors.receptor_or_ligand_not_generated(f"Could not generate receptor or ligand object for the protein in dir '{ligandDir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
            else:
                octools.print_warning_log(f"The smina output for '{ptn}' is already generated and you can check it at the '{sminaLog}' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                octools.print_warning(f"The smina output for '{ptn}' is already generated and you can check it at the '{sminaLog}' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
        elif dockingAlgorithm == "plants":
            # Flag to denote if its needed to run this protein through plants
            needToRun = False
            # Check if plantsFiles does not exist
            if not os.path.isdir(f"{ligandDir}/plantsFiles/"):
                octools.print_error_log(f"The directory '{ligandDir}/plantsFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                return errors.dir_does_not_exists(f"The directory '{ligandDir}/plantsFiles/' does not exist! Please ensure its existance before running this function. NOTE: You may need to run the verify_integrity routine to help to ensure that all files are ok.", level = "error")
            # Get the folder for each run
            runPaths = glob(f"{ligandDir}/plantsFiles/*")
            # Check if all files have been processed
            for runPath in runPaths:
                # Get the run number
                runNumber = runPath.split(os.path.sep)[-1]
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
                mol2Path, file_extension = os.path.splitext(receptorPath)
                # Read the receptor and the ligand (passing the mol2!!!)
                receptor = ocr.Receptor(receptorPath, mol2Path = f"{mol2Path}.mol2", from_json_descriptors = receptorDescriptor, name = f"{ptn}_receptor")
                ligand = ocl.Ligand(ligandPath, from_json_descriptors = ligandDescriptor, name = f"{ptn}_ligand")
                # If receptor and ligand are not null
                if receptor and ligand:
                    # For each path in the paths array (will be more than on in case of multiple boxes)
                    for runPath in runPaths:
                        # Get the run number
                        runNumber = runPath.split(os.path.sep)[-1]
                        # Parameterizing paths
                        plantsLog = f"{runPath}/plants_{runNumber}.log"
                        plantsOutput = f"{runPath}/run"
                        plantsRankingCsv = f"{plantsOutput}/ranking.csv"
                        # Create the smina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                        plants = ocplants.PLANTS(f"{runPath}/conf_plants.txt", f"{receptorDir}/p2rank/box{runNumber}.pdb", receptor, f"{receptorDir}/{receptorName}_prepared.mol2", ligand, f"{ligandDir}/{ptn}_ligand_prepared.mol2", plantsLog, plantsOutput, name=f"{ptn} PLANTS")
                        # Check if the smina object has been correctly created
                        if not plants:
                            octools.print_error_log(f"Could not generate plants object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                            return errors.docking_object_not_generated(f"Could not generate plants object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                        # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                        if overwrite or not os.path.isfile(plants.preparedLigand) or os.path.getsize(plants.preparedLigand) == 0 or not octools.is_molecule_valid(plants.preparedLigand):
                            # Run the prepare ligand
                            _ = plants.run_prepare_ligand()
                            # Check if the generated ligand has size 0 or is invalid
                            if os.path.getsize(plants.preparedLigand) == 0 or not octools.is_molecule_valid(plants.preparedLigand):
                                octools.print_warning_log(f"SPORES has made an output of 0kb for ligand '{plants.preparedLigand}', this is wierd. Trying to run it again.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                                octools.print_warning(f"SPORES has made an output of 0kb for ligand '{plants.preparedLigand}', this is wierd. Trying to run it again.")
                                # Run again the prepare ligand
                                _ = plants.run_prepare_ligand()
                                # Check again if the generated ligand has size 0 or is invalid
                                if os.path.getsize(plants.preparedLigand) == 0 or not octools.is_molecule_valid(plants.preparedLigand):
                                    octools.print_error_log(f"SPORES has made an output of 0kb again for ligand '{plants.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(plants.prepareLigandCmd)}", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                                    return errors.ligand_not_prepared(f"SPORES has made an output of 0kb again for ligand '{plants.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(plants.prepareLigandCmd)}", level = "error")
                        # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                        if overwrite or not os.path.isfile(plants.preparedReceptor) or os.path.getsize(plants.preparedReceptor) == 0 or not octools.is_molecule_valid(plants.preparedReceptor):
                            # Run the prepare receptor
                            _ = plants.run_prepare_receptor()
                            # Check if the generated receptor has size 0 or is invalid
                            if os.path.getsize(plants.preparedReceptor) == 0 or not octools.is_molecule_valid(plants.preparedReceptor):
                                octools.print_warning_log(f"SPORES has made an output of 0kb for ligand '{plants.preparedReceptor}', this is wierd. Trying to run it again.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                                octools.print_warning(f"SPORES has made an output of 0kb for ligand '{plants.preparedReceptor}', this is wierd. Trying to run it again.")
                                # Run again the prepare receptor
                                _ = plants.run_prepare_receptor()
                                # Check again if the generated receptor has size 0 or is invalid
                                if os.path.getsize(plants.preparedReceptor) == 0 or not octools.is_molecule_valid(plants.preparedReceptor):
                                    octools.print_error_log(f"SPORES has made an output of 0kb again for receptor '{plants.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(plants.prepareReceptorCmd)}", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                                    return errors.receptor_not_prepared(f"SPORES has made an output of 0kb again for receptor '{plants.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(plants.prepareReceptorCmd)}", level = "error")
                        # Check if PLANTS output exists and its size is not 0
                        if overwrite or not os.path.isdir(plantsOutput) or not os.path.isfile(plantsRankingCsv) and not os.path.getsize(plantsRankingCsv) == 0:
                            # If there is already a PLANTS output (PLANTS do not run if the folder is already created. And knowing that PLANTS will ALWAYS run if this code is interpreted, just delete the folder if it exists and lets avoid headaches)
                            if os.path.isdir(plantsOutput):
                                # Remove the folder and its contets
                                shutil.rmtree(plantsOutput)
                            # Run PLANTS
                            plants.run_plants(overwrite=overwrite)
                        else:
                            octools.print_warning_log(f"The PLANTS output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '*/run/plants_<runNumber>.log' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                            octools.print_warning(f"The PLANTS output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '*/run/plants_<runNumber>.log' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
                else:
                    octools.print_error_log(f"Could not generate receptor or ligand object for the protein in dir '{ligandDir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
                    return errors.receptor_or_ligand_not_generated(f"Could not generate receptor or ligand object for the protein in dir '{ligandDir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
            else:
                octools.print_warning_log(f"The PLANTS output for '{ptn}' is already generated and you can check it at the '{ligandDir}/plantsFiles' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log")
                octools.print_warning(f"The PLANTS output for '{ptn}' is already generated and you can check it at the '{ligandDir}/plantsFiles' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
        else:
            octools.print_error_log(f"Wrong docking algorithm. Expected ['vina', 'smina', 'plants'] and got '{dockingAlgorithm}'.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
            return errors.receptor_or_ligand_descriptor_does_not_exist(f"Wrong docking algorithm. Expected ['vina', 'smina', 'plants'] and got '{dockingAlgorithm}'.", level = "error")
    else:
        if not os.path.isfile(receptorDescriptor):
            octools.print_error_log(f"There is no receptor descriptor json file for the protein in the path '{receptorDescriptor}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
            _ = errors.receptor_or_ligand_descriptor_does_not_exist(f"There is no receptor descriptor for the protein in the path '{receptorDescriptor}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
        if not os.path.isfile(ligandDescriptor):
            octools.print_error_log(f"There is no ligand descriptor json file for the protein in the path '{ligandDescriptor}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log")
            _ = errors.receptor_or_ligand_descriptor_does_not_exist(f"There is no ligand descriptor for the protein in the path '{ligandDescriptor}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
        return errors.receptor_or_ligand_descriptor_does_not_exist()
    return errors.ok()

def __core_run_dock(d: str, ligandDir: str, archive: str, dockingAlgorithm: str, overwrite: bool) -> int:
    '''Performs the docking.

    Parameters
    ----------
    d : str
        The path to the protein directory.
    archive : str
        Which archive will be processed [dudez, pdbbind, astex].
    dockingAlgorithm : str
        Which docking algorithm will be used [vina, smina, plants].
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

    # If is the index directory, ignore
    if d in ['index', 'db']:
        return errors.unnalowed_dir()

    # Find protein name
    ptn = d.split(os.path.sep)[-1]

    # Set receptor data
    receptorPath = f"{d}/receptor.pdb"
    receptorDescriptor = f"{d}/receptor_descriptors.json"

    # Set ligand data
    ligandPath = f"{ligandDir}/ligand.smi"
    ligandDescriptor = f"{ligandDir}/ligand_descriptors.json"

    # Run the docking sub core routine for the chosen archive and algorithm
    return __sub_core_run_dock(receptorPath, ligandPath, d, ligandDir, archive, dockingAlgorithm, receptorDescriptor, ligandDescriptor, overwrite)

def __thread_run_dock_parallel(arguments):
    '''Thread aid function to call __core_run_dock.

    Parameters
    ----------
    arguments : list
        The arguments to be passed to __core_run_dock.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core dock function passing the arguments correctly
        __core_run_dock(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])
    return None

def __run_dock_parallel(dirs: List[str], ligandDirs: List[str], archive: str, dockingAlgorithm: str, overwrite: bool, desc: str) -> int:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    dirs : List[str]
        A list of directories where the files are stored.
    ligandDirs : List[str]
        A list of directories where the ligands are stored.
    archive : str
        Which archive will be processed [dudez, pdbbind, astex].
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

    # For each file in dirs
    for d in dirs:
        # Now loop over the ligands of this protein
        for ligandDir in ligandDirs:
            # Add the arguments to the list (creating one execution for each pair receptor-ligand)
            arguments.append((d, ligandDir, archive, dockingAlgorithm, overwrite))

    # If logfile exists, backup it (for error and warnings)
    if os.path.isfile(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log"):
        if not os.path.isdir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_ERROR.log", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past/{archive}_{dockingAlgorithm}_run_report_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")
    if os.path.isfile(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log"):
        if not os.path.isdir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/{archive}_{dockingAlgorithm}_run_report_WARNING.log", f"{logdir}/{archive}_{dockingAlgorithm}_run_report_past/{archive}_{dockingAlgorithm}_run_report_WARNING_{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for _ in tqdm(p.imap_unordered(__thread_run_dock_parallel, arguments), total = len(arguments), desc = desc):
            # Clear the memory
            gc.collect()

    # Return
    return errors.ok()

def __run_dock_no_parallel(dirs: List[str], ligandDirs: List[str], archive: str, dockingAlgorithm: str, overwrite: bool, desc: str) -> int:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_run_dock function.

    Parameters
    ----------
    dirs : List[str]
        A list of directories where the files are stored.
    ligandDirs : List[str]
        A list of directories where the ligands are stored.
    archive : str
        Which archive will be processed [dudez, pdbbind, astex].
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
        for d in tqdm(iterable=dirs, total=len(dirs), desc=desc):
            for ligandDir in ligandDirs:
                # Call the core dock function (shared between parallel and not parallel)
                __core_run_dock(d, ligandDir, archive, dockingAlgorithm, overwrite)
            # Clear the memory
            gc.collect()
        # Clear the memory
        gc.collect()

    return errors.ok()


### Read logs
def __core_read_log(processDirData: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    '''Reads Vina, Smina and PLANTS logs and then return a dict of dataframes.

    Parameters
    ----------
    processDirData : Tuple[str, str]
        A tuple with the directory and the ligand name.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        A dict of dicts of dataframes. Each element of the first dict is a complex protein-ligand, and each element of the second dict is a docking algorithm results.

    Raises
    ------
    None
    '''

    # Unpack the tuple
    processDir, tp = processDirData

    # Get protein and ligand names
    ptn = processDir.split(os.path.sep)[-3]
    lgd = processDir.split(os.path.sep)[-1]

    # Create Vina, Smina and PLANTS dataframes
    vinadf = pd.DataFrame(columns=["mode", "affinity", "rmsd_lb_best_mode", "rmsd_ub_best_mode"])
    sminadf = pd.DataFrame(columns=["mode", "affinity", "rmsd_lb_best_mode", "rmsd_ub_best_mode"])
    plantsdf = pd.DataFrame(columns=["LIGAND_ENTRY", "TOTAL_SCORE", "SCORE_RB_PEN", "SCORE_NORM_HEVATOMS", "SCORE_NORM_CRT_HEVATOMS", "SCORE_NORM_WEIGHT", "SCORE_NORM_CRT_WEIGHT", "SCORE_RB_PEN_NORM_CRT_HEVATOMS"])
    # Dict to hold the protein data
    proteinData = {}

    # Get all vina directories (0, 1, 2...)
    vinaDirs = glob(f"{processDir}/vinaFiles/*")
    # For each dir in vinaDirs
    for vinaDir in vinaDirs:
        # Get run number
        runNumber = vinaDir.split(os.path.sep)[-1]
        # Parameterize the log path
        logPath = f"{vinaDir}/vina_{runNumber}.log"
        # Check if exists
        if os.path.isfile(logPath):
            # Read the log into dataframe
            df = ocvina.read_vina_log(logPath)
            # Check if df is a dataframe
            if isinstance(df, pd.DataFrame):
                # Concatenate df and vinadf
                vinadf = pd.concat([vinadf, df], ignore_index=True)
            else:
                _ = errors.wrong_type(f"The file '{logPath}' could not be read.")
        else:
            _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its vina output.")
    # Get all vina directories (0, 1, 2...)
    plantsDirs = glob(f"{processDir}/plantsFiles/*")
    # For each dir in plantsDir
    for plantsDir in plantsDirs:
        # Get run number
        runNumber = plantsDir.split(os.path.sep)[-1]
        # Parameterize the log path
        logPath = f"{plantsDir}/run/ranking.csv"
        # Check if exists
        if os.path.isfile(logPath):
            # Read the log into dataframe
            df = ocplants.read_plants_log(logPath)
            if isinstance(df, pd.DataFrame):
                # Concatenate df and plantsdf
                plantsdf = pd.concat([plantsdf, df], ignore_index=True)
            else:
                _ = errors.wrong_type(f"The file '{logPath}' could not be read.")
        else:
            _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its PLANTS output.")
    # Parameterize the log path
    logPath = f"{processDir}/sminaFiles/smina.log"
    # Check if smina log exists
    if os.path.isfile(logPath):
        # Read the log into dataframe
        df = ocsmina.read_smina_log(logPath)
        if isinstance(df, pd.DataFrame):
            # Concatenate df and plantsdf
            sminadf = df
        else:
            _ = errors.wrong_type(f"The file '{logPath}' could not be read.")
    else:
        _ = errors.file_do_not_exist(f"The file '{logPath}' does not exist. Could not read its SMINA output.")
    # Create a dataFrame to the type
    df = pd.DataFrame([[ptn, lgd, tp]], columns=['Protein', 'Ligand', 'type'])
    # Add the protein data to the proteinData dict using ptn as the key
    proteinData[f"{ptn}-{lgd}"] = {"vina": vinadf, "smina": sminadf, "plants": plantsdf, "type": df}

    # Return the proteinData dict
    return proteinData

def __thread_read_log_parallel(arguments: Tuple[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    '''Thread aid function to call __core_read_log.

    Parameters
    ----------
    arguments : Tuple[str]
        A tuple with the directory.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        A dict of dicts of dataframes. Each element of the first dict is a complex protein-ligand, and each element of the second dict is a docking algorithm results.

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_read_log(arguments[0])

def __read_log_parallel(ptnDirs: List[str], desc: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    ptnDirs : List[str]
        A list of directories to be processed.
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        A dict of dicts of dataframes. Each element of the first dict is a complex protein-ligand, and each element of the second dict is a docking algorithm results.

    Raises
    ------
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for ptnDir in ptnDirs:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((ptnDir))

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

    # Dict to store the read data
    data = {}
    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for innerData in tqdm(p.imap_unordered(__thread_read_log_parallel, arguments), total = len(arguments), desc = desc):
            # Update the dict with the result from the called function
            data.update(innerData)
            # Clear the memory
            gc.collect()

    return data

def __read_log_no_parallel(ptnDirs: List[str], desc: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_read_log function.

    Parameters
    ----------
    ptnDirs : List[str]
        A list of directories to be processed.
    desc : str
        The description to be used in the tqdm progress bar.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
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
        for ptnDir in tqdm(iterable = ptnDirs, total = len(ptnDirs), desc = desc):
            # Call the core read log function (shared between parallel and not parallel) and store the data into the data dict
            data.update(__core_read_log(ptnDir))
            # Clear the memory
            gc.collect()

    return data


### Parse into csv
def __core_generate_dock_result_csv(processDir: str, log_dump: Dict[str, pd.DataFrame], ptn: str, ligand: str, archive: str) -> pd.DataFrame:
    '''Reads Vina, Smina and PLANTS logs and then return a dict of dataframes.

    Parameters
    ----------
    processDir : str
        The directory where the logs are.
    log_dump : Dict[str, pd.DataFrame]
        The log dump for the complex ptn-ligand.
    ptn : str
        The protein name.
    ligand : str
        The ligand name.
    archive : str
        Which archive will be processed [dudez, pdbbind, astex].

    Returns
    -------
    pd.DataFrame
        A dataframe with the results.

    Raises
    ------
    None
    '''

    # The new dataframe
    df = pd.DataFrame(columns=["Protein", "Ligand", "vina_affinity", "smina_affinity", "plants_TOTAL_SCORE", "plants_SCORE_RB_PEN", "plants_SCORE_NORM_HEVATOMS", "plants_SCORE_NORM_CRT_HEVATOMS", "plants_SCORE_NORM_WEIGHT", "plants_SCORE_NORM_CRT_WEIGHT", "plants_SCORE_RB_PEN_NORM_CRT_HEVATOMS", "vina_rmsd", "smina_rmsd", "plants_rmsd"])

    # List to work with vina/smina/PLANTS data
    vinaData = []
    sminaData = []
    plantsData = []

    # If the vina dataframe is not empty
    if not log_dump['vina'].empty:
        # Get all vina directories (0, 1, 2...)
        vinaDirs = glob(f"{processDir}/vinaFiles/*")
        for vinaDir in vinaDirs:
            # Get run number
            runNumber = vinaDir.split(os.path.sep)[-1]
            # Try to load the mol2, if fails, try the .sdf
            try:
                # Get the RMSD
                rmsd = octools.get_rmsd(f"{processDir}/{ligand}.mol2", f"{processDir}/vinaFiles/{runNumber}/vina_{runNumber}.pdbqt")
                # Check the RMSD type
                if type(rmsd) == float:
                    vinaData.append([rmsd])
                else:
                    # Find and concatenate the RMSDs
                    vinaData += rmsd # type: ignore
            except Exception as e:
                try:
                    octools.print_warning(f"Possibly I could not load the '{ligand}.mol2', trying to load the '{ligand}.sdf' instead. Error: {e}")
                    # Get the RMSD (using the sdf)
                    rmsd = octools.get_rmsd(f"{processDir}/{ligand}.sdf", f"{processDir}/vinaFiles/{runNumber}/vina_{runNumber}.pdbqt")
                    # Check the RMSD type
                    if type(rmsd) == float:
                        vinaData.append([runNumber, rmsd])
                    else:
                        # Find and concatenate the RMSDs
                        vinaData += rmsd # type: ignore
                except Exception as e2:
                    octools.print_error(f"Problems while processing the Vina output for the protein '{processDir}'")
                    octools.print_error_log(f"Problems while processing the Vina output for the protein '{processDir}'. Error: {e2}", f"{logdir}/{archive}_dock_result_ERROR.log")

    # If the vina dataframe is not empty
    if not log_dump['smina'].empty:
        # Try to load the mol2, if fails, try the .sdf
        try:
            # Get the RMSD
            rmsd = octools.get_rmsd(f"{processDir}/{ligand}.mol2", f"{processDir}/sminaFiles/smina.pdbqt")
            # Check the RMSD type
            if type(rmsd) == float:
                sminaData.append([rmsd])
            else:
                # Find and concatenate the RMSDs
                sminaData += rmsd # type: ignore
        except Exception as e:
            try:
                octools.print_warning(f"Possibly I could not load the '{ligand}.mol2', trying to load the '{ligand}.sdf' instead. Error: {e}")
                # Get the RMSD
                rmsd = octools.get_rmsd(f"{processDir}/{ligand}.sdf", f"{processDir}/sminaFiles/smina.pdbqt")
                # Check the RMSD type
                if type(rmsd) == float:
                    sminaData.append([rmsd])
                else:
                    # Find and concatenate the RMSDs
                    sminaData += rmsd # type: ignore
            except Exception as e2:
                octools.print_error(f"Problems while processing the Smina output for the protein '{processDir}'")
                octools.print_error_log(f"Problems while processing the Smina output for the protein '{processDir}'. Error: {e2}", f"{logdir}/{archive}_dock_result_ERROR.log")

    # If the plants dataframe is not empty
    if not log_dump['plants'].empty:
        # Get all PLANTS directories (0, 1, 2...)
        plantsDirs = glob(f"{processDir}/plantsFiles/*")
        for plantsDir in plantsDirs:
            # Get run number
            runNumber = plantsDir.split(os.path.sep)[-1]
            # For each ligand which is in the list
            for l in glob(f"{processDir}/plantsFiles/{runNumber}/run/*[0-9].mol2"):
                # Try to load the mol2, if fails, try the .sdf
                try:
                    # Get the RMSD
                    rmsd = octools.get_rmsd(f"{processDir}/{ligand}.mol2", l)
                    # Check the RMSD type
                    if type(rmsd) == float:
                        plantsData.append([rmsd])
                    else:
                        # Find and concatenate the RMSDs
                        plantsData += rmsd # type: ignore
                except Exception as e:
                    try:
                        octools.print_warning(f"Possibly I could not load the '{ligand}.mol2', trying to load the '{ligand}.sdf' instead. Error: {e}")
                        # Get the RMSD
                        rmsd = octools.get_rmsd(f"{processDir}/{ligand}.sdf", l)
                        # Check the RMSD type
                        if type(rmsd) == float:
                            plantsData.append([rmsd])
                        else:
                            # Find and concatenate the RMSDs
                            plantsData += rmsd # type: ignore
                    except Exception as e2:
                        octools.print_error(f"Problems while processing the PLANTS output for the protein '{processDir}'")
                        octools.print_error_log(f"Problems while processing the PLANTS output for the protein '{processDir}'. Error: {e2}", f"{logdir}/{archive}_dock_result_ERROR.log")

    # For each software, if not empty, determine which is the minimum value and which index it belongs and then select the corresponding line in the DataFrame
    if vinaData:
        minRMSD_vina = min(vinaData)
        index_vina = vinaData.index(minRMSD_vina)
        vinaList = log_dump['vina'][['affinity']].iloc[[index_vina]].values[0].tolist()
    else:
        vinaList = [np.NaN]
        minRMSD_vina = np.NaN

    # For each software, if not empty, determine which is the minimum value and which index it belongs and then select the corresponding line in the DataFrame
    if sminaData:
        minRMSD_smina = min(sminaData)
        index_smina = sminaData.index(minRMSD_smina)
        sminaList = log_dump['smina'][['affinity']].iloc[[index_smina]].values[0].tolist()
    else:
        sminaList = [np.NaN]
        minRMSD_smina = np.NaN

    # For each software, if not empty, determine which is the minimum value and which index it belongs and then select the corresponding line in the DataFrame
    if plantsData:
        minRMSD_plants = min(plantsData)
        index_plants = plantsData.index(minRMSD_plants)
        plantsList = log_dump['plants'][["TOTAL_SCORE", "SCORE_RB_PEN", "SCORE_NORM_HEVATOMS", "SCORE_NORM_CRT_HEVATOMS", "SCORE_NORM_WEIGHT", "SCORE_NORM_CRT_WEIGHT", "SCORE_RB_PEN_NORM_CRT_HEVATOMS"]].iloc[[index_plants]].values[0].tolist()
    else:
        plantsList = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
        minRMSD_plants = np.NaN

    # Append the data to the DataFrame
    #df.loc[len(df), df.columns] = [ptn] + [ligand] + vinaList + sminaList + plantsList + [minRMSD_vina, minRMSD_smina, minRMSD_plants]
    df.append(pd.DataFrame([ptn] + [ligand] + vinaList + sminaList + plantsList + [minRMSD_vina, minRMSD_smina, minRMSD_plants], columns = df.columns))

    return df

def __thread_generate_dock_result_csv_parallel(arguments: Tuple[str, Dict[str, pd.DataFrame], str, str, str]) -> pd.DataFrame:
    '''Thread aid function to call __core_generate_dock_result_csv.

    Parameters
    ----------
    arguments : Tuple[str, Dict[str, pd.DataFrame], str, str, str]
        Tuple containing the arguments for the __core_generate_dock_result_csv function. The arguments are: (ptn, ligand, processDir, logdir, archive).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the docking.

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_generate_dock_result_csv(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])

def __generate_dock_result_csv_parallel(processDirs: List[Tuple[str, str, str, Dict[str, pd.DataFrame]]], archive: str, desc: str) -> pd.DataFrame:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    processDirs : List[Tuple[str, str, str, Dict[str, pd.DataFrame]]]
        Dictionary containing the directories to process and the ligands to process. The dictionary is in the format: {ptn-ligand: log_dump}.
    archive : str
        Which archive will be processed [dudez, pdbbind, astex].
    desc : str
        Description to be displayed in the progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the docking.

    Raises
    ------
    None
    '''

    # If logfile exists, backup it for vina, smina and plants (for error and warnings)
    if os.path.isfile(f"{logdir}/{archive}_dock_result_ERROR.log"):
        if not os.path.isdir(f"{logdir}/generate_dock_result_csv_past"):
            octools.safe_create_dir(f"{logdir}/generate_dock_result_csv_past")
        os.rename(f"{logdir}/{archive}_dock_result_ERROR.log", f"{logdir}/read_log_past/{archive}_dock_result_ERROR.{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    
    # For each file in the glob
    for processDir, ptn, ligand, log_dump in processDirs:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((processDir, log_dump, ptn, ligand, archive))

    # Result DataFrame list
    dfList = []
    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for line in tqdm(p.imap_unordered(__thread_generate_dock_result_csv_parallel, arguments), total = len(arguments), desc = desc):
            # Append the result to the dataframe list
            dfList.append(line)
            # Clear the memory
            gc.collect()
    return pd.concat(dfList)

def __generate_dock_result_csv_no_parallel(processDirs: List[Tuple[str, str, str, Dict[str, pd.DataFrame]]], archive: str, desc: str) -> pd.DataFrame:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_generate_dock_result_csv function.

    Parameters
    ----------
    processDirs : List[Tuple[str, str, str, Dict[str, pd.DataFrame]]]
        Dictionary containing the directories to process and the ligands to process. The dictionary is in the format: {ptn: log_dump}.
    archive : str
        Which archive will be processed [dudez, pdbbind, astex].
    desc : str
        Description to be displayed in the progress bar.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the docking.

    Raises
    ------
    None
    '''

    # If logfile exists, backup it for vina, smina and plants (for error and warnings)
    if os.path.isfile(f"{logdir}/{archive}_dock_result_ERROR..log"):
        if not os.path.isdir(f"{logdir}/generate_dock_result_csv_past"):
            octools.safe_create_dir(f"{logdir}/generate_dock_result_csv_past")
        os.rename(f"{logdir}/{archive}_dock_result_ERROR.log", f"{logdir}/read_log_past/{archive}_dock_result_ERROR.{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Result DataFrame list
    dfList = []

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for processDir, ptn, ligand, log_dump in tqdm(iterable = processDirs, total = len(processDirs), desc = desc):
            # Call the core read log function (shared between parallel and not parallel) and assign it to the line
            dfList.append(__core_generate_dock_result_csv(processDir, log_dump, ptn, ligand, archive))
            # Clear the memory
            gc.collect()

    return pd.concat(dfList)


### Merge descriptors in dataframe
def __core_merge_descriptors_in_dataframe(processDirPackage: Tuple[str, str], archive: str) -> pd.DataFrame:
    '''Reads the descriptor and receptor json then parse them into a dataframe.

    Parameters
    ----------
    processDirPackage : Tuple(str, str)
        Tuple containing the processDir and the package. The tuple is in the format: (processDir, receptor_descriptor_path).
    archive : str
        Which archive will be processed [dudez, pdbbind, astex].

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the docking.

    Raises
    ------
    None
    '''

    # Unpack the tuple
    processDir, receptor_descriptor_path = processDirPackage

    # Find ptn name
    ptn = os.path.dirname(receptor_descriptor_path).split(os.path.sep)[-1]

    #region Create an empty dataframe with all descriptors
    ptndf = pd.DataFrame(columns=["Protein", "AUTOCORR2D_1", "AUTOCORR2D_2", "AUTOCORR2D_3", "AUTOCORR2D_4", "AUTOCORR2D_5", "AUTOCORR2D_6", "AUTOCORR2D_7", "AUTOCORR2D_8", "AUTOCORR2D_9", "AUTOCORR2D_10", "AUTOCORR2D_11", "AUTOCORR2D_12", "AUTOCORR2D_13", "AUTOCORR2D_14", "AUTOCORR2D_15", "AUTOCORR2D_16", "AUTOCORR2D_17", "AUTOCORR2D_18", "AUTOCORR2D_19", "AUTOCORR2D_20", "AUTOCORR2D_21", "AUTOCORR2D_22", "AUTOCORR2D_23", "AUTOCORR2D_24", "AUTOCORR2D_25", "AUTOCORR2D_26", "AUTOCORR2D_27", "AUTOCORR2D_28", "AUTOCORR2D_29", "AUTOCORR2D_30", "AUTOCORR2D_31", "AUTOCORR2D_32", "AUTOCORR2D_33", "AUTOCORR2D_34", "AUTOCORR2D_35", "AUTOCORR2D_36", "AUTOCORR2D_37", "AUTOCORR2D_38", "AUTOCORR2D_39", "AUTOCORR2D_40", "AUTOCORR2D_41", "AUTOCORR2D_42", "AUTOCORR2D_43", "AUTOCORR2D_44", "AUTOCORR2D_45", "AUTOCORR2D_46", "AUTOCORR2D_47", "AUTOCORR2D_48", "AUTOCORR2D_49", "AUTOCORR2D_50", "AUTOCORR2D_51", "AUTOCORR2D_52", "AUTOCORR2D_53", "AUTOCORR2D_54", "AUTOCORR2D_55", "AUTOCORR2D_56", "AUTOCORR2D_57", "AUTOCORR2D_58", "AUTOCORR2D_59", "AUTOCORR2D_60", "AUTOCORR2D_61", "AUTOCORR2D_62", "AUTOCORR2D_63", "AUTOCORR2D_64", "AUTOCORR2D_65", "AUTOCORR2D_66", "AUTOCORR2D_67", "AUTOCORR2D_68", "AUTOCORR2D_69", "AUTOCORR2D_70", "AUTOCORR2D_71", "AUTOCORR2D_72", "AUTOCORR2D_73", "AUTOCORR2D_74", "AUTOCORR2D_75", "AUTOCORR2D_76", "AUTOCORR2D_77", "AUTOCORR2D_78", "AUTOCORR2D_79", "AUTOCORR2D_80", "AUTOCORR2D_81", "AUTOCORR2D_82", "AUTOCORR2D_83", "AUTOCORR2D_84", "AUTOCORR2D_85", "AUTOCORR2D_86", "AUTOCORR2D_87", "AUTOCORR2D_88", "AUTOCORR2D_89", "AUTOCORR2D_90", "AUTOCORR2D_91", "AUTOCORR2D_92", "AUTOCORR2D_93", "AUTOCORR2D_94", "AUTOCORR2D_95", "AUTOCORR2D_96", "AUTOCORR2D_97", "AUTOCORR2D_98", "AUTOCORR2D_99", "AUTOCORR2D_100", "AUTOCORR2D_101", "AUTOCORR2D_102", "AUTOCORR2D_103", "AUTOCORR2D_104", "AUTOCORR2D_105", "AUTOCORR2D_106", "AUTOCORR2D_107", "AUTOCORR2D_108", "AUTOCORR2D_109", "AUTOCORR2D_110", "AUTOCORR2D_111", "AUTOCORR2D_112", "AUTOCORR2D_113", "AUTOCORR2D_114", "AUTOCORR2D_115", "AUTOCORR2D_116", "AUTOCORR2D_117", "AUTOCORR2D_118", "AUTOCORR2D_119", "AUTOCORR2D_120", "AUTOCORR2D_121", "AUTOCORR2D_122", "AUTOCORR2D_123", "AUTOCORR2D_124", "AUTOCORR2D_125", "AUTOCORR2D_126", "AUTOCORR2D_127", "AUTOCORR2D_128", "AUTOCORR2D_129", "AUTOCORR2D_130", "AUTOCORR2D_131", "AUTOCORR2D_132", "AUTOCORR2D_133", "AUTOCORR2D_134", "AUTOCORR2D_135", "AUTOCORR2D_136", "AUTOCORR2D_137", "AUTOCORR2D_138", "AUTOCORR2D_139", "AUTOCORR2D_140", "AUTOCORR2D_141", "AUTOCORR2D_142", "AUTOCORR2D_143", "AUTOCORR2D_144", "AUTOCORR2D_145", "AUTOCORR2D_146", "AUTOCORR2D_147", "AUTOCORR2D_148", "AUTOCORR2D_149", "AUTOCORR2D_150", "AUTOCORR2D_151", "AUTOCORR2D_152", "AUTOCORR2D_153", "AUTOCORR2D_154", "AUTOCORR2D_155", "AUTOCORR2D_156", "AUTOCORR2D_157", "AUTOCORR2D_158", "AUTOCORR2D_159", "AUTOCORR2D_160", "AUTOCORR2D_161", "AUTOCORR2D_162", "AUTOCORR2D_163", "AUTOCORR2D_164", "AUTOCORR2D_165", "AUTOCORR2D_166", "AUTOCORR2D_167", "AUTOCORR2D_168", "AUTOCORR2D_169", "AUTOCORR2D_170", "AUTOCORR2D_171", "AUTOCORR2D_172", "AUTOCORR2D_173", "AUTOCORR2D_174", "AUTOCORR2D_175", "AUTOCORR2D_176", "AUTOCORR2D_177", "AUTOCORR2D_178", "AUTOCORR2D_179", "AUTOCORR2D_180", "AUTOCORR2D_181", "AUTOCORR2D_182", "AUTOCORR2D_183", "AUTOCORR2D_184", "AUTOCORR2D_185", "AUTOCORR2D_186", "AUTOCORR2D_187", "AUTOCORR2D_188", "AUTOCORR2D_189", "AUTOCORR2D_190", "AUTOCORR2D_191", "AUTOCORR2D_192", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "EState_VSA10", "EState_VSA11", "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea", "FractionCSP3", "HallKierAlpha", "HeavyAtomMolWt", "HeavyAtomCount", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRadicalElectrons", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "qed", "RingCount", "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SMR_VSA10", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "TPSA", "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "VSA_EState10", "AUTOCORR3D_1", "AUTOCORR3D_2", "AUTOCORR3D_3", "AUTOCORR3D_4", "AUTOCORR3D_5", "AUTOCORR3D_6", "AUTOCORR3D_7", "AUTOCORR3D_8", "AUTOCORR3D_9", "AUTOCORR3D_10", "AUTOCORR3D_11", "AUTOCORR3D_12", "AUTOCORR3D_13", "AUTOCORR3D_14", "AUTOCORR3D_15", "AUTOCORR3D_16", "AUTOCORR3D_17", "AUTOCORR3D_18", "AUTOCORR3D_19", "AUTOCORR3D_20", "AUTOCORR3D_21", "AUTOCORR3D_22", "AUTOCORR3D_23", "AUTOCORR3D_24", "AUTOCORR3D_25", "AUTOCORR3D_26", "AUTOCORR3D_27", "AUTOCORR3D_28", "AUTOCORR3D_29", "AUTOCORR3D_30", "AUTOCORR3D_31", "AUTOCORR3D_32", "AUTOCORR3D_33", "AUTOCORR3D_34", "AUTOCORR3D_35", "AUTOCORR3D_36", "AUTOCORR3D_37", "AUTOCORR3D_38", "AUTOCORR3D_39", "AUTOCORR3D_40", "AUTOCORR3D_41", "AUTOCORR3D_42", "AUTOCORR3D_43", "AUTOCORR3D_44", "AUTOCORR3D_45", "AUTOCORR3D_46", "AUTOCORR3D_47", "AUTOCORR3D_48", "AUTOCORR3D_49", "AUTOCORR3D_50", "AUTOCORR3D_51", "AUTOCORR3D_52", "AUTOCORR3D_53", "AUTOCORR3D_54", "AUTOCORR3D_55", "AUTOCORR3D_56", "AUTOCORR3D_57", "AUTOCORR3D_58", "AUTOCORR3D_59", "AUTOCORR3D_60", "AUTOCORR3D_61", "AUTOCORR3D_62", "AUTOCORR3D_63", "AUTOCORR3D_64", "AUTOCORR3D_65", "AUTOCORR3D_66", "AUTOCORR3D_67", "AUTOCORR3D_68", "AUTOCORR3D_69", "AUTOCORR3D_70", "AUTOCORR3D_71", "AUTOCORR3D_72", "AUTOCORR3D_73", "AUTOCORR3D_74", "AUTOCORR3D_75", "AUTOCORR3D_76", "AUTOCORR3D_77", "AUTOCORR3D_78", "AUTOCORR3D_79", "AUTOCORR3D_80", "Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex", "SASA", "DipoleMoment", "IsoelectricPoint", "InstabilityIndex","GRAVY", "Aromaticity", "__countAA", "countA", "countR", "countN", "countD", "countC", "countQ", "countE", "countG", "countH", "countI", "countL", "countK", "countM", "countF", "countP", "countS", "countT", "countW", "countY", "countV", "TotalAALength", "AvgAALength", "countChain"])
    #endregion

    # Find which kind of archive it will be
    if archive == "astex":
        ligand_descriptor_path = f"{astex_archive}/{ptn}/{ptn}_ligand_descriptors.json"
    elif archive == "dudez":
        ligand_descriptor_path = f"{processDir}/{processDir.split(os.path.sep)[-1]}_descriptors.json"
    elif archive == "pdbbind":
        ligand_descriptor_path = f"{pdbbind_archive}/{ptn}/{ptn}_ligand_descriptors.json"
    else:
        octools.print_error(f"Unknown archive type. Expected one of the following: 'astex', 'dudez', 'pdbbind' and got {archive}.")
        return ptndf
    # Check if there is the receptor json, if yes, load it
    if os.path.isfile(receptor_descriptor_path):
        receptor_descriptors = ocr.read_descriptors_from_json(receptor_descriptor_path, returnDict = True)
    else:
        _ = errors.file_do_not_exist(f"The file {receptor_descriptor_path} does not exist!")
        receptor_descriptors = {}
    # Check if there is the ligand json, if yes, load it
    if os.path.isfile(ligand_descriptor_path):
        ligand_descriptors = ocl.read_descriptors_from_json(ligand_descriptor_path, returnDict = True)
    else:
        _ = errors.file_do_not_exist(f"The file {ligand_descriptor_path} does not exist!")
        ligand_descriptors = {}
    # Create new dict
    all_descriptors = dict()
    # Set Name and Path
    all_descriptors["Protein"] = ptn
    # Merge both descriptors dicts
    all_descriptors = { **all_descriptors, **receptor_descriptors } # type: ignore
    all_descriptors = { **all_descriptors, **ligand_descriptors } # type: ignore
    # Create a temporary pd.DataFrame
    tmpdf = pd.DataFrame(all_descriptors, index=[0])
    # Append the line to the pd.DataFrame
    ptndf = pd.concat([ptndf, tmpdf], ignore_index=True)
    # Return the dataframe with a single row
    return ptndf

def __thread_merge_descriptors_in_dataframe_parallel(arguments: Tuple[Tuple[str, str], str]) -> pd.DataFrame:
    '''Thread aid function to call __core_merge_descriptors_in_dataframe.

    Parameters
    ----------
    arguments : Tuple[Tuple[str, str], str]
        Tuple containing the directory where the files are stored and the receptor descriptor json file and the archive type.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with the descriptors of the protein.

    Raises
    ------
    None
    '''

    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_merge_descriptors_in_dataframe(arguments[0], arguments[1])

def __merge_descriptors_in_dataframe_parallel(dirs: List[Tuple[str, str]], archive: str, desc: str) -> pd.DataFrame:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.

    Parameters
    ----------
    dirs : List[Tuple[str, str]]
        Tuple containing the directory where the files are stored and the receptor descriptor json file.
    archive : str
        Which archive will be processed. [dudez, pdbbind, astex]
    desc : str
        Description of the process.

    Returns
    -------
    pd.DataFrame
        Dataframe with the descriptors of the proteins.

    Raises
    ------
    None
    '''

    # Arguments to pass to each Thread in the Thread Pool
    arguments = []

    # For each file in the glob
    for dir in dirs:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((dir, archive))

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

    # Dataframe with all protein data
    ptndf = pd.DataFrame()

    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for innerData in tqdm(p.imap_unordered(__thread_merge_descriptors_in_dataframe_parallel, arguments), total = len(arguments), desc = desc):
            # Update the dict with the result from the called function
            ptndf = pd.concat([ptndf, innerData])
            # Clear the memory
            gc.collect()

    return ptndf

def __merge_descriptors_in_dataframe_no_parallel(dirs: List[Tuple[str, str]], archive: str, desc: str) -> pd.DataFrame:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_read_log function.

    Parameters
    ----------
    dirs : List[Tuple[str, str]]
        Tuple containing the directory where the files are stored and the receptor descriptor json file.

    Returns
    -------
    pd.DataFrame
        Dataframe with the descriptors of the proteins.

    Raises
    ------
    None
    '''

    # Dict to store the read data
    ptndf = pd.DataFrame()
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
            ptndf = pd.concat([ptndf, __core_merge_descriptors_in_dataframe(dir, archive)])
            # Clear the memory
            gc.collect()
    return ptndf


## Public ##
def verify_integrity(chosenArchive: str, spacing: float = 0.33) -> None:
    '''Verifies the integrity of the desired database

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
            if os.path.basename(dir) in ['index', 'db']:
                # Skip it
                continue

            # Parameterizing paths
            p2rankDir = f"{dir}/p2rank"
            vinaDir = f"{dir}/vinaFiles"
            plantsDir = f"{dir}/plantsFiles"

            # Find protein name
            ptn = dir.split(os.path.sep)[-1]

            # Set the input file name path and set the input file name path
            if archive == "astex":
                fin = f"{dir}/protein.pdb"
            elif archive == "dudez":
                fin = f"{dir}/rec.crg.pdb"
                ligand = f""
            elif archive == "pdbbind":
                fin = f"{dir}/{ptn}_protein.pdb"
                ligand = f"{dir}/{ptn}_ligand.mol2"
            else:
                octools.print_error(f"Unknown archive type, expected one of the following ['astex', 'dudez', 'pdbbind'] and got '{archive}'.")
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
                # Run the vina conf creation from box
                __run_create_vina_conf_from_box(dir, preparedReceptor)

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
                # Generate box files
                __run_create_plants_conf_from_box(dir, preparedReceptor, preparedLigand, spacing)
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
                if not os.path.isfile(f"{dir}/{ptn}_ligand_descriptors.json") or os.path.getsize(f"{dir}/{ptn}_ligand_descriptors.json") == 0:
                    # Generate it
                    __prepare_molecule(f"{dir}/{ptn}_ligand.mol2", False, "ligand", archive, sanitize = True)
                    # If the file still does not exists...
                    if not os.path.isfile(f"{dir}/{ptn}_ligand_descriptors.json") or os.path.getsize(f"{dir}/{ptn}_ligand_descriptors.json") == 0:
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
        The archive to be converted. The options are [dudez, pdbbind, astex].
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
        The archive to be prepared. The options are [dudez, pdbbind, astex].
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
    if archive == "astex":
        chosenArchive = astex_archive
        label = f"Astex proteins"
        # Get all paths in the database
        paths = glob(f"{chosenArchive}/*")
    elif archive == "dudez":
        chosenArchive = dudez_archive
        label = f"DUDEz proteins"
        # Get all paths in the database
        paths = glob(f"{chosenArchive}/*")
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
        label = "PDBbind proteins"
        # Get all paths in the database filtering for pdbbind
        paths = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index']]
    else:
        octools.print_error(f"Not valid archive type. Expected one of ['astex', 'dudez', 'pdbbind'] and found {archive}.")
        return None

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
        The archive to be prepared. The options are [dudez, pdbbind, astex].
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
    if archive == "astex":
        chosenArchive = astex_archive
        label = f"Astex proteins"
        # Get all dirs paths in the database
        dirs = glob(f"{chosenArchive}/*")
    elif archive == "dudez":
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
        octools.print_error(f"Not valid archive type. Expected one of ['astex', 'dudez', 'pdbbind'] and found {archive}.")
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
        The archive to be prepared. The options are [dudez, pdbbind, astex].
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
    # Find which kind of archive it will be
    if archive == "astex":
        chosenArchive = astex_archive
    elif archive == "dudez":
        chosenArchive = dudez_archive
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
    else:
        return errors.not_supported_archive(f"Not valid archive type. Expected one of ['astex', 'dudez', 'pdbbind'] and found {archive}.")

    # Check if the docking algorithm is valid
    if dockingAlgorithm not in ["vina", "smina", "plants"]:
        return errors.not_supported_docking_algorithm(f"Docking software not recognized. Expected ('vina', 'smina', 'plants') and got '{dockingAlgorithm}'.")

    # Get all dirs paths in the database
    ptnDirs = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index', 'db']]

    # Create the alternative dir list
    ligandDirs = []
    # For each dir in dirs, let's grab all ligands
    for ptnDir in ptnDirs:
        # Set the model path
        receptorPath = f"{ptnDir}/rec.crg.pdb"
        # Check if file is a PDB file
        if receptorPath.endswith(".pdb"):
            # Initialise hasCryst1 flag
            hasCryst1 = False

            # Check if modelPath is a valid file
            if os.path.isfile(receptorPath):
                # Open it
                with open(receptorPath, "r") as pdbFile:
                    # For each line in it
                    for line in pdbFile:
                        # Check if starts with CRYST1
                        if line.startswith("CRYST1"):
                            # Set the hasCryst1 flag to True
                            hasCryst1 = True
                            # Since it has been found, break the loop
                            break
                        # If it is not CRYST1, check if it is ATOM
                        elif line.startswith("ATOM"):
                            # If is ATOM and not CRYST1, means that there is no CRYST1 line in the file, so break the loop
                            break
            # If there is no CRYST1 line in the file let's add a generic CRYST1 line then
            if not hasCryst1:
                # Define a generic CRYST1 line
                cryst1 = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
                # Initialise the contnt variable
                content = ""
                # Read the CRYST1 line to the file
                with open(receptorPath, "r") as pdbFile:
                    # Read the file
                    content = pdbFile.read()
                # Write the CRYST1 line to the file
                with open(receptorPath, "w") as pdbFile:
                    # Write the CRYST1 line to the file
                    pdbFile.write(cryst1)
                    # Write the content to the file
                    pdbFile.write(content)
        # Parameterize paths
        ligands = f"{ptnDir}/compounds/ligands"
        decoys = f"{ptnDir}/compounds/decoys"
        candidates = f"{ptnDir}/compounds/candidates"

        # Merge the ligandAlternative list with the list with ligands, decoys and candidates
        ligandDirs = glob(f"{ligands}/*") + glob(f"{decoys}/*") + glob(f"{candidates}/*")

    # Decide if multprocessing will be used
    if args.multiprocess:
        __run_dock_parallel(ptnDirs, ligandDirs, archive, dockingAlgorithm, overwrite, f"Processing {archive}")
    else:
        __run_dock_no_parallel(ptnDirs, ligandDirs, archive, dockingAlgorithm, overwrite, f"Processing {archive}")

    return errors.ok()

def read_logs(archive: str, picklePath: str = "") -> Union[Dict[str, Dict[str, pd.DataFrame]], None]:
    '''Reads database logfiles returning a dict of dicts of pd.DataFrames.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind, astex].
    picklePath : str, optional
        The path to the pickle file. The default is "". If the picklePath is not empty, the function will write the data to the pickle file.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]] | None
        A dict of dicts of pd.DataFrames with the data from the logfiles. If fails, will return None.

    Raises
    ------
    None
    '''

    # Make archive lowercase
    archive = os.path.basename(archive).lower()
    # Find which kind of archive it will be
    if archive == "astex":
        chosenArchive = astex_archive
    elif archive == "dudez":
        chosenArchive = dudez_archive
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
    else:
        octools.print_error(f"Not valid archive type. Expected one of ['astex', 'dudez', 'pdbbind'] and found {archive}.")
        return None
        
    # Create an empty list for all directories to be processed
    processDirs = []

    # For each dir in chosenArchive
    for ptnDir in glob(f"{chosenArchive}/*"):
        # Check if is a dir (just in case) and if its name is not one of the ones we want to skip
        if os.path.isdir(ptnDir) and os.path.basename(ptnDir.split(os.path.sep)[-1]) not in ['index', 'db']:
            # Find ptn name
            ptn = ptnDir.split(os.path.sep)[-1]

            ligands = f"{ptnDir}/compounds/ligands"
            decoys = f"{ptnDir}/compounds/decoys"
            candidates = f"{ptnDir}/compounds/candidates"

            # Add all subdirs (one for each ligand) from all 4 folders as a tuple (dir, ligand_descriptor_path)
            processDirs += [processDir for processDir in glob(f"{ligands}/*") if os.path.isdir(processDir)]
            processDirs += [processDir for processDir in glob(f"{decoys}/*") if os.path.isdir(processDir)]
            processDirs += [processDir for processDir in glob(f"{candidates}/*") if os.path.isdir(processDir)]

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

def generate_dock_result_csv(archive: str, log_dumps: Dict[str, Dict[str, pd.DataFrame]], csv_path: str, chunksize: int = 500) -> None:
    '''Uses the structure from read_logs to generate an output for all docking softwares.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind, astex].
    log_dumps : Dict[str, Dict[str, pd.DataFrame]]
        The data from the logfiles.
    csv_path : str
        The path to the csv file.
    chunksize : int, optional
        The chunksize to be used when writing the csv. The default is 500.

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    # Create an empty list for all directories to be processed
    processDirs = []

    # Check which archive is being used
    if archive == "astex":
        # Get protein dirs
        ptnDirs = [d for d in glob(f"{astex_archive}/*") if os.path.isdir(d)]
    elif archive == "dudez":
        # Get protein dirs
        ptnDirs = [d for d in glob(f"{dudez_archive}/*") if os.path.isdir(d)]
    elif archive == "pdbbind":
        # Get protein dirs
        ptnDirs = [d for d in glob(f"{pdbbind_archive}/*") if os.path.isdir(d)]
    else:
        octools.print_error(f"Unknown archive type. Expected one of the following: 'astex', 'dudez', 'pdbbind' and got {archive}.")
        return None

    # For each protein in proteins
    for ptnDir in ptnDirs:
        # Parameterize paths
        ligands = f"{ptnDir}/compounds/ligands"
        decoys = f"{ptnDir}/compounds/decoys"
        candidates = f"{ptnDir}/compounds/candidates"
        
        # Add all subdirs (one for each ligand) from all 4 folders as a tuple (dir, ligand_name))
        processDirs += [(d, d.split(os.path.sep)[-3], d.split(os.path.sep)[-1], log_dumps[f"{d.split(os.path.sep)[-3]}-{d.split(os.path.sep)[-1]}"]) for d in glob(f"{ligands}/*") if os.path.isdir(d)]
        processDirs += [(d, d.split(os.path.sep)[-3], d.split(os.path.sep)[-1], log_dumps[f"{d.split(os.path.sep)[-3]}-{d.split(os.path.sep)[-1]}"]) for d in glob(f"{decoys}/*") if os.path.isdir(d)]
        processDirs += [(d, d.split(os.path.sep)[-3], d.split(os.path.sep)[-1], log_dumps[f"{d.split(os.path.sep)[-3]}-{d.split(os.path.sep)[-1]}"]) for d in glob(f"{candidates}/*") if os.path.isdir(d)]
        
    # Decide if multprocessing will be used
    if args.multiprocess:
        data = __generate_dock_result_csv_parallel(processDirs, archive, f"Generating docking csv {archive}")
    else:
        data = __generate_dock_result_csv_no_parallel(processDirs, archive, f"Generating docking csv {archive}")
    # Check if data is not empty
    if not data.empty:
        data.to_csv(csv_path, index=False, chunksize=chunksize)
    return None

def merge_descriptors_in_dataframe(archive: str, saveCsv: bool = True) -> Union[pd.DataFrame, None]:
    '''Reads all the descriptors jsons and return a pd.DataFrame.

    Parameters
    ----------
    archive : str
        The archive to be prepared. The options are [dudez, pdbbind, astex].
    saveCsv : bool, optional
        If True, the csv will be saved. The default is True.
    
    Returns
    -------
    pd.DataFrame | None
        The dataframe with all the descriptors.

    Raises
    ------
    None
    '''

    # Make archive lowercase
    archive = os.path.basename(archive).lower()
    # Find which kind of archive it will be
    if archive == "astex":
        chosenArchive = astex_archive
    elif archive == "dudez":
        chosenArchive = dudez_archive
        # Parameterize the csvs paths
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
    else:
        octools.print_error(f"Not valid archive type. Expected one of ['astex', 'dudez', 'pdbbind'] and found {archive}.")
        return None

    # Parameterize the csvs paths (parsed_archive is defined in Initialise.py)
    csv_path_in = f"{parsed_archive}/{archive}.csv"
    csv_path_out = f"{parsed_archive}/{archive}_complete.csv"

    # Create an empty list for all directories to be processed
    processDirs = []

    # For each dir in chosenArchive
    for ptnDir in glob(f"{chosenArchive}/*"):
        # Check if is a dir (just in case) and if its name is not one of the ones we want to skip
        if os.path.isdir(ptnDir) and os.path.basename(ptnDir.split(os.path.sep)[-1]) not in ['index', 'db']:
            # Find ptn name
            ptn = ptnDir.split(os.path.sep)[-1]

            # Parameterize paths
            ligands = f"{ptnDir}/compounds/ligands"
            decoys = f"{ptnDir}/compounds/decoys"
            candidates = f"{ptnDir}/compounds/candidates"

            processDirs += [(processDir, f"{ptnDir}/receptor.json") for processDir in glob(f"{ligands}/*") if os.path.isdir(processDir)]
            processDirs += [(processDir, f"{ptnDir}/receptor.json") for processDir in glob(f"{decoys}/*") if os.path.isdir(processDir)]
            processDirs += [(processDir, f"{ptnDir}/receptor.json") for processDir in glob(f"{candidates}/*") if os.path.isdir(processDir)]

    # Make data be None (in case of failure)
    data = None
    # Decide if multprocessing will be used
    if args.multiprocess:
        data = __merge_descriptors_in_dataframe_parallel(processDirs, archive, f"Processing {archive}")
    else:
        data = __merge_descriptors_in_dataframe_no_parallel(processDirs, archive, f"Processing {archive}")
    # Check if data is pd.DataFrame type and is not empty
    if type(data) == pd.DataFrame and not data.empty:
        # Try to write the csv
        try:
            # Rename the name column from data dataframe
            data = data.rename(columns={'Name': 'Ligand'})
            # Remove unwanted keys
            for k in ["Path", "mol2Path", "__countAA"]:
                if k in data:
                    del data[k]
            # Read the csv from input file
            ptndf = pd.read_csv(csv_path_in)
            # Merge the both DataFrames using the Protein column as a comparer
            data = pd.merge(ptndf, data, on=["Protein", "Ligand"], how="left")
            # If saveCsv is True, save the csv
            if saveCsv:
                # Write the data to a new csv file
                data.to_csv(csv_path_out, index=False)
            octools.print_success(f"The file '{csv_path_out}' has been successfully written.")
        except Exception as e:
            octools.print_error(f"Could not write the file '{csv_path_out}'. Error: {e}")
            # Return Nothing
            return None
    else:
        octools.print_warning(f"The data object is not defined! There is no reason to write it as a pickle. Aborting...")
        # Return nothing
        return None

    # Return the data
    return data
