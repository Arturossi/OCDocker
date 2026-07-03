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
from tqdm import tqdm
from typing import Any, List, Optional, Tuple, Union

import OCDocker.Docking.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Error as ocerror
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Basetools as ocbasetools
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint

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

def __core_prepare(
    path: str,
    overwrite: bool,
    archive: str,
    sanitize: bool,
    spacing: float,
    targetCentroid: Optional[Union[Tuple[float, float, float], rdkit.Geometry.rdGeometry.Point3D]] = None,
    all_boxes: bool = False
) -> int:
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
        return ocerror.Error.unallowed_dir()

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
            return ocerror.Error.file_not_exist(f"Could not find the file '{' or '.join([os.path.join(path, f'reference_ligand.{ref_ligand_ext}') for ref_ligand_ext in ref_ligand_exts])}' for the molecule '{path}' or the provided files are not valid and a target centroid has not been provided. This molecule will not be processed.", level = ocerror.ReportLevel.ERROR)

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
        box_files = sorted(glob(f"{processDir}/boxes/box*.pdb"))
        # Check if there is a box for the ligand
        boxCount = len(box_files)
        if boxCount == 0:
            ocprint.print_warning(f"No box files found for '{processDir}'. Skipping this molecule.")
            continue

        if not all_boxes or len(box_files) <= 1:
            gnina_entries = glob(f"{processDir}/gninaFiles/*")
            vina_entries = glob(f"{processDir}/vinaFiles/*")
            plants_entries = glob(f"{processDir}/plantsFiles/*")
            smina_conf_entries = glob(f"{processDir}/sminaFiles/*.conf")

            # If overwrite mode is on or there is not the same amount of box files as folders in gninaFiles folder
            if overwrite or len(gnina_entries) != boxCount:
                # Create the gnina inputs from the boxes
                ocgnina.gen_gnina_conf(f"{processDir}/boxes/box0.pdb", f"{processDir}/gninaFiles/conf_gnina.conf", preparedReceptorPdbqt)
            else:
                ocprint.print_info(f"The protein '{processDir}' already has its gnina file generated, skipping its execution.")

            # If overwrite mode is on or there is not the same amount of box files as folders in vinaFiles folder
            if overwrite or len(vina_entries) != boxCount:
                # Create the vina inputs from the boxes
                ocvina.generate_vina_files_database(processDir, preparedReceptorPdbqt, boxPath = f"{processDir}/boxes")
            else:
                ocprint.print_info(f"The protein '{processDir}' already has its vina file generated, skipping its execution.")

            # If overwrite mode is on or there is not the same amount of box files as folders in plantsFiles folder
            if overwrite or len(plants_entries) != boxCount:
                # Set the fligand variable to the dir + ligandName + .mol2
                fligand = f"{processDir}/ligand.mol2"
                # Create the PLANTS inputs from the boxes
                ocplants.generate_plants_files_database(processDir, preparedReceptorMol2, fligand, spacing, boxPath = f"{processDir}/boxes")
            else:
                ocprint.print_info(f"The protein '{processDir}' already has its PLANTS file generated, skipping its execution.")

            # If overwrite mode is on or there not any conf file in the sminaFiles folder
            if overwrite or len(smina_conf_entries) == 0:
                # Create the smina inputs
                ocsmina.gen_smina_conf(f"{processDir}/boxes/box0.pdb", f"{processDir}/sminaFiles/conf_smina.conf", preparedReceptorPdbqt)
            else:
                ocprint.print_info(f"The protein '{processDir}' already has its smina file generated, skipping its execution.")
        else:
            # Multi-box mode: generate per-box configs
            for box_file in box_files:
                box_id = os.path.splitext(os.path.basename(box_file))[0]
                gnina_dir = f"{processDir}/gninaFiles/{box_id}"
                vina_dir = f"{processDir}/vinaFiles/{box_id}"
                plants_dir = f"{processDir}/plantsFiles/{box_id}"
                smina_dir = f"{processDir}/sminaFiles/{box_id}"
                _ = ocff.safe_create_dir(gnina_dir)
                _ = ocff.safe_create_dir(vina_dir)
                _ = ocff.safe_create_dir(plants_dir)
                _ = ocff.safe_create_dir(smina_dir)

                # GNINA conf
                ocgnina.gen_gnina_conf(box_file, f"{gnina_dir}/conf_gnina.conf", preparedReceptorPdbqt)
                # Vina conf
                ocvina.box_to_vina(box_file, f"{vina_dir}/conf_vina.conf", preparedReceptorPdbqt)
                # Smina conf
                ocsmina.gen_smina_conf(box_file, f"{smina_dir}/conf_smina.conf", preparedReceptorPdbqt)
                # PLANTS conf
                fligand = f"{processDir}/ligand.mol2"
                ocplants.box_to_plants(box_file, f"{plants_dir}/conf_plants.txt", preparedReceptorMol2, fligand, f"{plants_dir}/run", spacing = spacing)

    return ocerror.Error.ok()

def __prepare_molecule(
    mol: Union[Tuple[str, str], str, rdkit.Chem.rdchem.Mol],
    overwrite: bool,
    moltype: str,
    dbName: str,
    sanitize: bool,
    molName: str = "",
    targetCentroid: Optional[Union[Tuple[float, float, float], rdkit.Geometry.rdGeometry.Point3D]] = None,
    alternativeLigand: Optional[Union[str, rdkit.Chem.rdchem.Mol]] = None
) -> None:
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
    if isinstance(mol, tuple):
        molPath = os.path.split(mol[0])[0]
    elif isinstance(mol, str):
        molPath = os.path.split(mol)[0]
    else:
        molPath = ""

    # Check if the molName was provided
    if molName == "":
        # Set the molname as the molType
        molName = moltype

    if overwrite or not os.path.isfile(f"{molPath}/{moltype}_descriptors.json"):
        m: Any
        if moltype == "ligand":
            # Safe create dockingFiles dirs
            _ = ocff.safe_create_dir(f"{molPath}/plantsFiles")
            _ = ocff.safe_create_dir(f"{molPath}/vinaFiles")
            _ = ocff.safe_create_dir(f"{molPath}/sminaFiles")
            _ = ocff.safe_create_dir(f"{molPath}/gninaFiles")

            try:
                ligand_input: Union[str, rdkit.Chem.rdchem.Mol]
                if isinstance(mol, tuple):
                    ligand_input = mol[0]
                else:
                    ligand_input = mol
                # Create the ligand object
                m = ocl.Ligand(ligand_input, molName, sanitize = sanitize)
                # Test if the Radius of Gyration is None
                if not m.RadiusOfGyration:
                    # Print a warning
                    ocprint.print_warning(f"The ligand '{molName}' has a Radius of Gyration of None, trying to load its alternative ligand.")
                    # If so, try to load the alternative ligand
                    if alternativeLigand:
                        # Create the ligand object
                        m = ocl.Ligand(alternativeLigand, molName, sanitize = sanitize)
                        # Check the radius of gyration again
                        if not m.RadiusOfGyration:
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

                _ = ocerror.Error.parse_molecule(errMsg, level = ocerror.ReportLevel.ERROR)
                config = get_config()
                ocprint.print_error_log(errMsg, f"{config.logdir}/{dbName}_error_Parse.log")
                return None

        elif moltype == "receptor":
                # If is a tuple
                if isinstance(mol, tuple):
                    try:
                        # Check if the extension is pdb
                        if mol[0].endswith(".pdb"):
                            # Clean the receptor
                            _ = ocmolproc.clean_for_dssp(structurePath = mol[0])
                        # Create the receptor object
                        m = ocr.Receptor(mol[0], molName, mol2_path = mol[1])
                    except Exception as e:
                        errMsg = f"The molecule '{mol[0]}' could not be parsed! Error {e}"

                        _ = ocerror.Error.parse_molecule(errMsg, level = ocerror.ReportLevel.ERROR)
                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{dbName}_error_Parse.log")
                        return None
                elif isinstance(mol, str):
                    try:
                        # Create the receptor object
                        m = ocr.Receptor(mol, molName)
                    except Exception as e:
                        errMsg = f"The molecule '{mol}' could not be parsed! Error {e}"

                        _ = ocerror.Error.parse_molecule(errMsg, level = ocerror.ReportLevel.ERROR)
                        config = get_config()
                        ocprint.print_error_log(errMsg, f"{config.logdir}/{dbName}_error_Parse.log")
                        return None
                else:
                    _ = ocerror.Error.wrong_type(
                        f"The receptor '{mol}' has an unsupported type. Expected a file path or (pdb, mol2) tuple.",
                        level=ocerror.ReportLevel.ERROR,
                    )
                    return None
        else:
            _ = ocerror.Error.unknown("Unknown molecule type", level = ocerror.ReportLevel.ERROR)
            return None

        # Test if the molecule is valid
        if not m or not m.is_valid():
            errMsg = f"The molecule '{mol}' is not valid! Its descriptors are malformed. Please check it manually!"

            _ = ocerror.Error.malformed_molecule(errMsg, level = ocerror.ReportLevel.ERROR)
            config = get_config()
            ocprint.print_error_log(errMsg, f"{config.logdir}/{dbName}_error_Parse.log")
        else:
            # Export its descriptors
            _ = m.to_json(overwrite)

    # Return
    return None

def __prepare_no_parallel(paths: List[str], overwrite: bool, archive: str, sanitize: bool, spacing: float, desc: str, all_boxes: bool) -> None:
    '''Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_prepare function.


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
        collect_every = __gc_collect_interval(len(paths))
        for i, path in enumerate(tqdm(iterable=paths, total=len(paths), desc=desc), start=1):
            # Call the core prepare function
            __core_prepare(path, overwrite, archive, sanitize, spacing, all_boxes = all_boxes)
            # Large batches benefit from less frequent explicit GC.
            __collect_periodically(i, collect_every, gc.collect)

    gc.collect()

    return None

def __prepare_parallel(paths: List[str], overwrite: bool, archive: str, sanitize: bool, spacing: float, desc: str, all_boxes: bool) -> None:
    '''Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.


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

    total_jobs = len(paths)
    arguments = (
        (path, overwrite, archive, sanitize, spacing, all_boxes)
        for path in paths
    )

    try:
        # Create a Thread pool with the maximum available_cores
        config = get_config()
        with Pool(config.available_cores) as p:
            collect_every = __gc_collect_interval(total_jobs)
            chunksize = __pool_chunksize(total_jobs, config.available_cores)
            # Perform the multi process
            for i, _ in enumerate(tqdm(p.imap_unordered(__thread_prepare, arguments, chunksize=chunksize), total = total_jobs, desc = desc), start=1):
                # Large batches benefit from less frequent explicit GC.
                __collect_periodically(i, collect_every, gc.collect)
    except IOError as e:
        errMsg = f"Problem while preparing {archive}. Exception: {e}"
        config = get_config()
        ocprint.print_error_log(errMsg, f"{config.logdir}/{archive}_prepare_report.log")
        ocprint.print_error(errMsg)

    gc.collect()

    return None

def __prepare_single(path: str, overwrite: bool, archive: str, sanitize: bool, spacing: float, all_boxes: bool) -> None:
    '''Warper to prepare the jobs, recieves a directory, and pass it to the __core_prepare function.


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

    __core_prepare(path, overwrite, archive, sanitize, spacing, all_boxes = all_boxes)
    gc.collect()

    return None

def __sub_core_prepare(
        dirsToProcess: str,
        dbName: str,
        overwrite: bool,
        mols: Optional[List[str]] = None,
        sanitize: bool = True,
        targetCentroid: Optional[Union[Tuple[float, float, float], rdkit.Geometry.rdGeometry.Point3D]] = None
    ) -> List[str]:
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

    if mols is None:
        mols = []
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
            shutil.move(mol, f"{mol}/{molName}/ligand.{molTmp[-1]}")

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

def __thread_prepare(arguments: Tuple[str, bool, str, bool, float, bool]) -> int:
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
        return __core_prepare(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], all_boxes = arguments[5])
## Public ##


def prepare(paths: Union[List[str], str], overwrite: bool, archive: str, sanitize: bool, spacing: float, all_boxes: bool = False) -> None:
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
            __prepare_parallel(paths, overwrite, archive, sanitize, spacing, label, all_boxes)
        else:
            # Prepare the database
            __prepare_no_parallel(paths, overwrite, archive, sanitize, spacing, label, all_boxes)
    else:
        __prepare_single(paths, overwrite, archive, sanitize, spacing, all_boxes)
