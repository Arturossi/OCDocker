#!/usr/lib/python3

# Imports
###############################################################################
import os
import gc
import time
import shutil
from glob import glob
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
def __run_p2rank(dir, fin, overwrite = False):
    '''
    Runs p2rank for a given directory.
    Input:
      dir       [string]                - Directory of the protein to run p2rank
      fin       [string]                - PDB file as input
      overwrite [bool]   DEFAULT: False - If True, all files will be generated, otherwise will try to optimize the run avoiding to run p2rank
    Return:
      -
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
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = 1 if args.output_level >= 3 else 0, overwrite = overwrite)
    except Exception as e:
        octools.print_warning(f"The protein '{dir}' had a problem while running p2rank. Retrying to run p2rank. Exception: {e}  ")
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = 1 if args.output_level >= 3 else 0, overwrite = overwrite)

    return

def __run_create_vina_conf_from_box(dir, fin):
    '''
    Creates vina conf file from box.
    Input:
      dir [string] - Directory of the protein to run p2rank
      fin [string] - PDB file as input
    Return:
      -
    '''
    # Run vina
    ocvina.generate_vina_files_database(dir, fin)

    return

def __run_create_plants_conf_from_box(dir, fin, ligand, spacing):
    '''
    Creates PLANTS conf file from box.
    Input:
      dir          [string] - Directory of the protein to run p2rank
      protein      [string] - Protein path
      ligand       [string] - Ligand name to be used in conf file
      spacing      [float]  - Extra spacing
    Return:
      -
    '''
    # Run vina
    ocplants.generate_plants_files_database(dir, fin, ligand, spacing)
    return

### Prepare
def __core_prepare(dir, overwrite, archive, sanitize, spacing):
    '''
    Prepares a database entry to be run in multiple docking software.
    Input:
     dir       [string] - Path where the data is
     overwrite [bool]   - Flag for demanding file overwrite
     archive   [string] - Which archive will be processed [dudez, pdbbind, astex]
     sanitize  [string] - Flag to tell if the molecule should be sanitized
     spacing   [float]  - The spacing value used to enlarge the radius of the sphere used in PLANTS file. Ranges from 0 to 1
    Return:
      -
    '''
    if archive == "astex":
        # Set the input file name path
        fin = f"{dir}/protein"

        # Set the ligand input file name path
        lfin = f"{dir}/ligand"

        # If the overwrite flag is true or the receptor pdb file does not exist
        if overwrite or not os.path.isfile(f"{fin}.pdb"):
            # Convert the protein file from mol2 to pdb
            _ = octools.convertMols(f"{fin}.mol2", f"{fin}.pdb")

        # If the overwrite flag is true or the ligand mol2 file does not exists
        if overwrite or not os.path.isfile(f"{lfin}.mol2"):
            # Convert the ligand file from mol to mol2
            _ = octools.convertMols(f"{lfin}.mol", f"{lfin}.mol2")

        # Reset the input file variable
        fin = f"{fin}.pdb"
    elif archive == "dudez":
        # Set the input file name path
        fin = f"{dir}/rec.crg.pdb"

        # Set the 3 dirs containing ligand/decoys
        dudezDir = f"{dir}/DUDE_Z"
        extremaDir = f"{dir}/Extrema"
        goldilocksDir = f"{dir}/Goldilocks"

        # Parameterize paths
        dudezDirLigand = f"{dudezDir}_ligands"
        dudezDirDecoy = f"{dudezDir}_decoys"
        extremaDirDecoy = f"{extremaDir}_decoys"
        goldilocksDirDecoy = f"{goldilocksDir}_decoys"

        # Create the dirs for data from the 3 dirs above
        _ = octools.safe_create_dir(dudezDirLigand)
        _ = octools.safe_create_dir(dudezDirDecoy)
        _ = octools.safe_create_dir(extremaDirDecoy)
        _ = octools.safe_create_dir(goldilocksDirDecoy)

        # Get all mol2 files in dudezDir
        mol2Files = glob(f"{dudezDir}/*.mol2")
        # Separate ligands and decoys
        for mol2File in mol2Files:
            # If there is the string ligand_poses in the link (means that is ligand)
            if "ligand_poses" in mol2File:
                _ = octools.split_and_convert(mol2File, dudezDirLigand, "mol2", overwrite)
            else:
                _ = octools.split_and_convert(mol2File, dudezDirDecoy, "mol2", overwrite)

        # Get all mol2 files in extremaDir
        mol2Files = glob(f"{extremaDir}/*.mol2")
        # Separate ligands and decoys
        for mol2File in mol2Files:
            _ = octools.split_and_convert(mol2File, extremaDirDecoy, "mol2", overwrite)

        # Get all mol2 files in goldilocksDir
        mol2Files = glob(f"{goldilocksDir}/*.mol2")
        # Separate ligands and decoys
        for mol2File in mol2Files:
            _ = octools.split_and_convert(mol2File, goldilocksDirDecoy, "mol2", overwrite)

        # Defining the moltype
        moltype = "ligand"

        # TODO: refazer essa parte aqui para adequar ao novo modelo de preparação
        # For each molecule in dudez ligand dir
        mols = glob(f"{dudezDirLigand}/*.mol2")
        #__prepare_parallel(mols, overwrite, moltype, f"{ptn} DUDEz ligand")
        # For each molecule in dudez decoy dir
        #__prepare_parallel(glob(f"{dudezDirDecoy}/*.mol2"), overwrite, moltype, f"{ptn} DUDEz decoy")
        # For each molecule in extrema decoy dir
        #__prepare_parallel(glob(f"{extremaDirDecoy}/*.mol2"), overwrite, moltype, f"{ptn} extrema decoy")
        # For each molecule in goldilocks decoy dir
        #__prepare_parallel(glob(f"{goldilocksDirDecoy}/*.mol2"), overwrite, moltype, f"{ptn} goldilocks decoy")
    elif archive == "pdbbind":
        # If is the index path
        if os.path.basename(dir) not in ['index', 'db']:
            # Skip it
            return
        # Find the protein name
        ptn = dir.split(os.path.sep)[-1]
        # Set the input file name path (to generate the box and data about the protein)
        fin = f"{dir}/{ptn}_protein.pdb"
        fout = f"{dir}/{ptn}_protein.mol2"
        # Convert the .pdb to .mol2 (for dock6 use)
        _ = octools.convertMols(fin, fout)
        # Set the ligand file name path (to generate data about the ligand)
        fligand = f"{dir}/{ptn}_ligand.mol2"
        # For each ligand (don't use parallel, since there is no need)
        __prepare_molecule(fligand, overwrite, "ligand", archive, f"{ptn} PDBbind ligand")
        # For each Receptor
        __prepare_molecule((fin, fout), overwrite, "receptor", archive, f"{ptn} PDBbind receptor")

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
    # If overwrite mode is on or there is not the same amount of box files as folders in vinaFiles folder
    if len(glob(f"{dir}/vinaFiles/*")) == boxCount or overwrite:
        # Create the vina inputs from the boxes
        ocvina.generate_vina_files_database(dir, fin)
    else:
        octools.print_info(f"The protein '{dir}' already has its vina file generated, skipping its execution.")
    # If overwrite mode is on or there is not the same amount of box files as folders in vinaFiles folder
    if len(glob(f"{dir}/plantsFiles/*")) == boxCount or overwrite:
        # Create the PLANTS inputs from the boxes
        ocplants.generate_plants_files_database(dir, fin, fligand, spacing)
    else:
        octools.print_info(f"The protein '{dir}' already has its PLANTS file generated, skipping its execution.")

    return None

def __thread_prepare(arguments):
    '''
    Thread aid function to call __core_prepare.
    Input:
     arguments [tuple(string, bool, string, string, bool)] - Tuple containing, in this order:
        - [string] The path where the files are
        - [bool]   Flag to tell if files should be overwritten
        - [string] The database name [dudez, pdbbind, astex]
        - [bool]   Flag to tell if the molecule should be sanitized
        - [float]  The spacing value used to enlarge the radius of the sphere used in PLANTS file. Ranges from 0 to 1
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call core prepare function (shared between thread and no thread)
        return __core_prepare(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])
    # Return
    return None

def __prepare_parallel2(filenames, overwrite, moltype, dbName, sanitize, desc):
    '''
    Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.
    Input:
     filenames [string] - List of molecule paths
     overwrite [bool]   - Flag to tell if files should be overwritten
     moltype   [string] - The type of the molecule (ligant or receptor)
     dbName    [string] - The database name (for proper logging) [dudez, pdbbind, astex]
     sanitize  [string] - Flag to tell telling if the molecule should be sanitized
     desc      [string] - The description used in the progress bar
    Return:
      -
    '''
    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for filename in filenames:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((dir, overwrite, moltype, dbName, sanitize))
    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for _ in tqdm(p.imap_unordered(__thread_prepare, arguments), total = len(arguments), desc = desc):
            pass
    # Return
    return None

def __prepare_molecule(mol, overwrite, moltype, dbName, sanitize):
    '''
    Prepares a molecule, generating output to docking software.
    Input:
     mol       [string] - Path to the molecule
     overwrite [bool]   - Flag to tell if files should be overwritten
     moltype   [string] - The type of the molecule (ligant or receptor)
     dbName    [string] - The database name (for proper logging)
     sanitize  [string] - Flag to tell telling if the molecule should be sanitized
    Return:
      -
    '''
    # Find its name and path
    if type(mol) == tuple:
        molPath, molName = os.path.split(mol[0])
    else:
        molPath, molName = os.path.split(mol)
    molName, ext = os.path.splitext(molName)
    if overwrite or not os.path.isfile(f"{molPath}/{molName}_descriptors.json"):
        if moltype == "ligand":
            try:
                # Create the ligand object
                m = ocl.Ligand(mol, molName, sanitize = sanitize)
            # If m is not valid
            except Exception as e:
                # Let's check its extension
                filename, file_extension = os.path.splitext(mol)
                # Check if the extension is .mol2
                if file_extension == ".mol2":
                    # Tell the user the search for another extension (.sdf)
                    _ = errors.parse_molecule(f"The molecule '{mol}' could not be parsed! Trying to change its extension from '.mol2' to '.sdf'.", "warning")
                    octools.print_warning_log(f"The molecule '{mol}' could not be parsed! Trying to change its extension from '.mol2' to '.sdf'.", f"{logdir}/{dbName}_warn_Parse.log")
                    try:
                        # Parse the .sdf file
                        m = ocl.Ligand(f"{filename}.sdf", molName, sanitize = sanitize)
                    except:
                        _ = errors.parse_molecule(f"The molecule '{mol}' could not be parsed!", "error")
                        octools.print_error_log(f"The molecule '{mol}' could not be parsed! .", f"{logdir}/{dbName}_error_Parse.log")
                        return None
                # Check if the extension is .sdf
                elif file_extension == ".sdf":
                    # Tell the user the search for another extension (.mol2)
                    _ = errors.parse_molecule(f"The molecule '{mol}' could not be parsed! Trying to change its extension from '.sdf' to '.mol2'.", "warning")
                    octools.print_warning_log(f"The molecule '{mol}' could not be parsed! Trying to change its extension from '.sdf' to '.mol2'.", f"{logdir}/{dbName}_warn_Parse.log")
                    try:
                        # Parse the .mol2 file
                        m = ocl.Ligand(f"{filename}.sdf", molName, sanitize = sanitize)
                    except:
                        _ = errors.parse_molecule(f"The molecule '{mol}' could not be parsed!", "error")
                        octools.print_error_log(f"The molecule '{mol}' could not be parsed! .", f"{logdir}/{dbName}_error_Parse.log")
                        return None
        elif moltype == "receptor":
            try:
                # If is a tuple
                if type(mol) == tuple:
                    # Create the receptor object
                    m = ocr.Receptor(mol[0], molName, mol2Path = mol[1])
                else:
                    # Create the receptor object
                    m = ocr.Receptor(mol, molName)
            # If m is not valid
            except Exception as e:
                _ = errors.parse_molecule(f"The molecule '{mol}' could not be parsed! Error {e}", "error")
                octools.print_error_log(f"The molecule '{mol}' could not be parsed! Error {e}", f"{logdir}/{dbName}_error_Parse.log")
                return None
        else:
            _ = errors.unkown("Unknown molecule type", "error")
            return None
        # Test if the ligand is valid
        if not m or not m.is_valid():
            _ = errors.malformed_molecule(f"The molecule '{mol}' is not valid! Its descriptors are malformed. Please check it manually!", "error")
            octools.print_error_log(f"The molecule '{mol}' is not valid! Its descriptors are malformed. Please check it manually!", f"{logdir}/{dbName}_error_Parse.log")
        else:
            # Export its descriptors
            _ = m.to_json(overwrite)
    # Return
    return None

def __prepare_parallel(dirs, overwrite, archive, sanitize, spacing, desc):
    '''
    Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.
    Input:
     dirs      [string] - List of paths to process
     overwrite [bool]   - Flag to tell if files should be overwritten
     archive   [string] - The database name (for proper logging) [dudez, pdbbind, astex]
     sanitize  [string] - Flag to tell telling if the molecule should be sanitized
     spacing   [float]  - The spacing value used to enlarge the radius of the sphere used in PLANTS file. Ranges from 0 to 1
     desc      [string] - The description used in the progress bar
    Return:
      -
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
    # Return
    return None

def __prepare_no_parallel(dirs, overwrite, archive, sanitize, spacing, desc):
    '''
    Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_prepare function.
    Input:
     dirs      [string] - List of paths to process
     overwrite [bool]   - Flag to tell if files should be overwritten
     archive   [string] - The database name (for proper logging)
     sanitize  [string] - Flag to tell telling if the molecule should be sanitized
     spacing   [float]  - The spacing value used to enlarge the radius of the sphere used in PLANTS file. Ranges from 0 to 1
     desc      [string] - The description used in the progress bar
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for dir in tqdm(iterable=dirs, total=len(dirs), desc=desc):
            # Call the core prepare function
            __core_prepare(dir, overwrite, archive, sanitize, spacing)
            # Clear the memory
            gc.collect()
    return None

### Get
def __core_get(dir, archive):
    '''
    Loads in memory a pair receptor-ligand and then return them in a tuple alongside with the protein name gotten from file path.
    Input:
     dir     [string] - The directory where the files are stored
     archive [string] - Which archive will be processed [dudez, pdbbind, astex]
    Return:
      -
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

def __thread_get_parallel(arguments):
    '''
    Thread aid function to call __core_get.
    Input:
     arguments [tuple(string, string)] - Tuple containing, in this order:
        - [string] - The directory where the files are stored
        - [string] - Which archive will be processed [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call core get function (shared between thread and not thread)
        return __core_get(arguments[0], arguments[1])

def __get_parallel(dirs, archive, desc):
    '''
    Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.
    Input:
     dirs    [string] - List of paths to process
     archive [string] - The database name (for proper logging)
     desc    [string] - The description used in the progress bar
    Return:
      [dict of tuple of OCDocker.Receptor, OCDocker.Ligand] - Dict of OCDocker.Receptor and OCDocker.Ligand objects having as key the name of the protein.
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

def __get_no_parallel(dirs, archive, desc):
    '''
    Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_get function.
    Input:
     dirs      [string] - List of paths to process
     archive   [string] - The database name (for proper logging)
     desc      [string] - The description used in the progress bar
    Return:
      [dict of tuple of OCDocker.Receptor, OCDocker.Ligand] - Dict of OCDocker.Receptor and OCDocker.Ligand objects having as key the name of the protein.
    '''
    # Dict of elements
    databaseDict = dict()
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for dir in tqdm(iterable=dirs, total=len(dirs), desc=desc):
            # Call the core get function
            data = __core_get(dir, archive)
            # Add them to the dict using the protein as the key
            databaseDict[data[0]] = (data[1], data[2])
            # Clear the memory
            gc.collect()
        return databaseDict

### Docking
def __core_run_dock(dir, archive, dockingAlgorithm, overwrite):
    '''
    Performs the docking.
    Input:
     dir              [string] - The directory where the files are stored
     archive          [string] - Which archive will be processed [dudez, pdbbind, astex]
     dockingAlgorithm [string] - Which docking algorithm will be used [vina, smina, plants]
     overwrite        [bool]   - Flag to tell if files should be overwritten
    Return:
      -
    '''
    # If is the index directory, ignore
    if dir in ['index', 'db']:
        return
    # Find which kind of archive it will be
    if archive == "astex":
        chosenArchive = astex_archive
    elif archive == "dudez":
        chosenArchive = dudez_archive
    elif archive == "pdbbind":
        # Find protein name
        ptn = dir.split(os.path.sep)[-1]
        # Set the input file name path (to generate the box and data about the protein)
        receptorPath = f"{dir}/{ptn}_protein.pdb"
        # Set the ligand file name path (to generate data about the ligand)
        ligandPath = f"{dir}/{ptn}_ligand.mol2"
        # If the complex has all descriptors for protein AND ligand
        if os.path.isfile(f"{dir}/{ptn}_protein_descriptors.json") and os.path.isfile(f"{dir}/{ptn}_ligand_descriptors.json"):
            # If running vina
            if dockingAlgorithm == "vina":
                # Flag to denote if its needed to run this protein through vina
                needToRun = False
                # Get the folder for each run
                runPaths = glob(f"{dir}/vinaFiles/*")
                # Check if all files have been processed
                for runPath in runPaths:
                    # Get the run number
                    runNumber = runPath.split(os.path.sep)[-1]
                    # If the output does not exist or overwrite flag is true
                    if overwrite or not os.path.isfile(f"{runPath}/vina_{runNumber}.log") or not os.path.isfile(f"{runPath}/vina_{runNumber}.pdbqt"):
                        needToRun = True
                        break
                # If is needed to run (at least one protein)
                if needToRun:
                    # Read the receptor and the ligand
                    receptor = ocr.Receptor(receptorPath, from_json_descriptors = f"{dir}/{ptn}_protein_descriptors.json", name = f"{ptn}_receptor")
                    ligand = ocl.Ligand(ligandPath, from_json_descriptors = f"{dir}/{ptn}_ligand_descriptors.json", name = f"{ptn}_ligand")
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
                            vina = ocvina.Vina(f"{runPath}/conf_vina.txt", f"{dir}/p2rank/box{runNumber}.pdb", receptor, f"{dir}/{ptn}_protein.pdbqt", ligand, f"{dir}/{ptn}_ligand.pdbqt", vinaLog, vinaOutput, name=f"{ptn}_run_{runNumber}")
                            # Check if the vina object has been correctly created
                            if not vina:
                                octools.print_error_log(f"Could not generate vina object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                                return errors.docking_object_not_generated(f"Could not generate vina object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                            # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                            if overwrite or not os.path.isfile(vina.preparedLigand) or os.path.getsize(vina.preparedLigand) == 0 or not octools.is_molecule_valid(vina.preparedLigand):
                                # Run the prepare ligand
                                _ = vina.run_prepare_ligand()
                                # Check if the generated ligand has size 0 or is invalid
                                if os.path.getsize(vina.preparedLigand) == 0 or not octools.is_molecule_valid(vina.preparedLigand):
                                    octools.print_warning_log(f"The prepare ligand script has made an output of 0kb for ligand '{vina.preparedLigand}', this is wierd. Trying to run it again.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                                    octools.print_warning(f"The prepare ligand script has made an output of 0kb for ligand '{vina.preparedLigand}', this is wierd. Trying to run it again.")
                                    # Run again the prepare ligand
                                    _ = vina.run_prepare_ligand()
                                    # Check again if the generated ligand has size 0 or is invalid
                                    if os.path.getsize(vina.preparedLigand) == 0 or not octools.is_molecule_valid(vina.preparedLigand):
                                        octools.print_error_log(f"The prepare ligand script has made an output of 0kb again for ligand '{vina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareLigandCmd)}", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                                        return errors.ligand_not_prepared(f"The prepare ligand script has made an output of 0kb again for ligand '{vina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareLigandCmd)}", level = "error")
                            # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                            if overwrite or not os.path.isfile(vina.preparedReceptor) or os.path.getsize(vina.preparedReceptor) == 0 or not octools.is_molecule_valid(vina.preparedReceptor):
                                # Run the prepare receptor
                                _ = vina.run_prepare_receptor()
                                # Check if the generated receptor has size 0 or is invalid
                                if os.path.getsize(vina.preparedReceptor) == 0 or not octools.is_molecule_valid(vina.preparedReceptor):
                                    octools.print_warning_log(f"The prepare receptor has made an output of 0kb for ligand '{vina.preparedReceptor}', this is wierd. Trying to run it again.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                                    octools.print_warning(f"The prepare receptor has made an output of 0kb for ligand '{vina.preparedReceptor}', this is wierd. Trying to run it again.")
                                    # Run again the prepare receptor
                                    _ = vina.run_prepare_receptor()
                                    # Check again if the generated receptor has size 0 or is invalid
                                    if os.path.getsize(smina.preparedReceptor) == 0 or not octools.is_molecule_valid(vina.preparedReceptor):
                                        octools.print_error_log(f"The prepare receptor has made an output of 0kb again for receptor '{vina.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareReceptorCmd)}", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                                        return errors.receptor_not_prepared(f"The prepare receptor has made an output of 0kb again for receptor '{vina.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(vina.prepareReceptorCmd)}", level = "error")
                            # Check if vina output exists
                            if overwrite or not os.path.isfile(vinaOutput) or not os.path.isfile(vinaLog):
                                # Run vina
                                vina.run_vina()
                            else:
                                octools.print_warning_log(f"The vina output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '{runPath}/vina_{runNumber}.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                                octools.print_warning(f"The vina output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '{runPath}/vina_{runNumber}.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true")
                    else:
                        octools.print_error_log(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                        return errors.receptor_or_ligand_not_generated(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                else:
                    octools.print_warning_log(f"The vina output for '{ptn}' for all boxes is already generated and you can check it at the '{dir}/vinaFiles/*/vina_<runNumber>.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                    octools.print_warning(f"The vina output for '{ptn}' for all boxes is already generated and you can check it at the '{dir}/vinaFiles/*/vina_<runNumber>.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
            elif dockingAlgorithm == "smina":
                # Set the run path
                runPath = f"{dir}/sminaFiles/"
                # Parameterizing paths
                sminaLog = f"{runPath}/smina.log"
                sminaOutput = f"{runPath}/smina.pdbqt"
                # Create the smina dir
                _ = octools.safe_create_dir(runPath)
                # If is needed to run (overwrite is set or no output is produced)
                if overwrite or not os.path.isfile(sminaLog) or not os.path.isfile(sminaOutput):
                    # Read the receptor and the ligand
                    receptor = ocr.Receptor(receptorPath, from_json_descriptors = f"{dir}/{ptn}_protein_descriptors.json", name = f"{ptn}_receptor")
                    ligand = ocl.Ligand(ligandPath, from_json_descriptors = f"{dir}/{ptn}_ligand_descriptors.json", name = f"{ptn}_ligand")
                    # If receptor and ligand are not null
                    if receptor and ligand:
                        # Create the smina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                        smina = ocsmina.Smina(f"{runPath}/conf_smina.txt", receptor, f"{dir}/{ptn}_protein.pdbqt", ligand, f"{dir}/{ptn}_ligand.pdbqt", sminaLog, sminaOutput, name=f"{ptn}_smina")
                        # Check if the smina object has been correctly created
                        if not smina:
                            octools.print_error_log(f"Could not generate smina object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                            return errors.docking_object_not_generated(f"Could not generate smina object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                        # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                        if overwrite or not os.path.isfile(smina.preparedLigand) or os.path.getsize(smina.preparedLigand) == 0 or not octools.is_molecule_valid(smina.preparedLigand):
                            # Run the prepare ligand
                            _ = smina.run_prepare_ligand()
                            # Check if the generated ligand has size 0 or is invalid
                            if os.path.getsize(smina.preparedLigand) == 0 or not octools.is_molecule_valid(smina.preparedLigand):
                                octools.print_warning_log(f"The prepare ligand script has made an output of 0kb for ligand '{smina.preparedLigand}', this is wierd. Trying to run it again.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                                octools.print_warning(f"The prepare ligand script has made an output of 0kb for ligand '{smina.preparedLigand}', this is wierd. Trying to run it again.")
                                # Run again the prepare ligand
                                _ = smina.run_prepare_ligand()
                                # Check again if the generated ligand has size 0 or is invalid
                                if os.path.getsize(smina.preparedLigand) == 0 or not octools.is_molecule_valid(smina.preparedLigand):
                                    octools.print_error_log(f"The prepare ligand script has made an output of 0kb again for ligand '{smina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareLigandCmd)}", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                                    return errors.ligand_not_prepared(f"The prepare ligand script has made an output of 0kb again for ligand '{smina.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareLigandCmd)}", level = "error")
                        # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                        if overwrite or not os.path.isfile(smina.preparedReceptor) or os.path.getsize(smina.preparedReceptor) == 0 or not octools.is_molecule_valid(smina.preparedReceptor):
                            # Run the prepare receptor
                            _ = smina.run_prepare_receptor()
                            # Check if the generated receptor has size 0 or is invalid
                            if os.path.getsize(smina.preparedReceptor) == 0 or not octools.is_molecule_valid(smina.preparedReceptor):
                                octools.print_warning_log(f"The prepare receptor has made an output of 0kb for ligand '{smina.preparedReceptor}', this is wierd. Trying to run it again.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                                octools.print_warning(f"The prepare receptor has made an output of 0kb for ligand '{smina.preparedReceptor}', this is wierd. Trying to run it again.")
                                # Run again the prepare receptor
                                _ = smina.run_prepare_receptor()
                                # Check again if the generated receptor has size 0 or is invalid
                                if os.path.getsize(smina.preparedReceptor) == 0 or not octools.is_molecule_valid(smina.preparedReceptor):
                                    octools.print_error_log(f"The prepare receptor has made an output of 0kb again for receptor '{smina.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareReceptorCmd)}", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                                    return errors.receptor_not_prepared(f"The prepare receptor has made an output of 0kb again for receptor '{smina.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(smina.prepareReceptorCmd)}", level = "error")
                        # Run smina (no need to recheck for overwrite or output existance because it is already done some lines ago)
                        smina.run_smina()
                    else:
                        octools.print_error_log(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                        return errors.receptor_or_ligand_not_generated(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                else:
                    octools.print_warning_log(f"The smina output for '{ptn}' is already generated and you can check it at the '{sminaLog}' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                    octools.print_warning(f"The smina output for '{ptn}' is already generated and you can check it at the '{sminaLog}' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
            elif dockingAlgorithm == "plants":
                # Flag to denote if its needed to run this protein through plants
                needToRun = False
                # Get the folder for each run
                runPaths = glob(f"{dir}/plantsFiles/*")
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
                    receptor = ocr.Receptor(receptorPath, mol2Path = f"{mol2Path}.mol2", from_json_descriptors = f"{dir}/{ptn}_protein_descriptors.json", name = f"{ptn}_receptor")
                    ligand = ocl.Ligand(ligandPath, from_json_descriptors = f"{dir}/{ptn}_ligand_descriptors.json", name = f"{ptn}_ligand")
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
                            plants = ocplants.PLANTS(f"{runPath}/conf_plants.txt", f"{dir}/p2rank/box{runNumber}.pdb", receptor, f"{dir}/{ptn}_protein_prepared.mol2", ligand, f"{dir}/{ptn}_ligand_prepared.mol2", plantsLog, plantsOutput, name=f"{ptn} PLANTS")
                            # Check if the smina object has been correctly created
                            if not plants:
                                octools.print_error_log(f"Could not generate plants object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                                return errors.docking_object_not_generated(f"Could not generate plants object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                            # If prepared ligand has the overwrite flag on, does not exists, has size 0 or is not valid
                            if overwrite or not os.path.isfile(plants.preparedLigand) or os.path.getsize(plants.preparedLigand) == 0 or not octools.is_molecule_valid(plants.preparedLigand):
                                # Run the prepare ligand
                                _ = plants.run_prepare_ligand()
                                # Check if the generated ligand has size 0 or is invalid
                                if os.path.getsize(plants.preparedLigand) == 0 or not octools.is_molecule_valid(plants.preparedLigand):
                                    octools.print_warning_log(f"SPORES has made an output of 0kb for ligand '{plants.preparedLigand}', this is wierd. Trying to run it again.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                                    octools.print_warning(f"SPORES has made an output of 0kb for ligand '{plants.preparedLigand}', this is wierd. Trying to run it again.")
                                    # Run again the prepare ligand
                                    _ = plants.run_prepare_ligand()
                                    # Check again if the generated ligand has size 0 or is invalid
                                    if os.path.getsize(plants.preparedLigand) == 0 or not octools.is_molecule_valid(plants.preparedLigand):
                                        octools.print_error_log(f"SPORES has made an output of 0kb again for ligand '{plants.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(plants.prepareLigandCmd)}", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                                        return errors.ligand_not_prepared(f"SPORES has made an output of 0kb again for ligand '{plants.preparedLigand}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(plants.prepareLigandCmd)}", level = "error")
                            # If prepared receptor has the overwrite flag on, does not exists, has size 0 or is not valid
                            if overwrite or not os.path.isfile(plants.preparedReceptor) or os.path.getsize(plants.preparedReceptor) == 0 or not octools.is_molecule_valid(plants.preparedReceptor):
                                # Run the prepare receptor
                                _ = plants.run_prepare_receptor()
                                # Check if the generated receptor has size 0 or is invalid
                                if os.path.getsize(plants.preparedReceptor) == 0 or not octools.is_molecule_valid(plants.preparedReceptor):
                                    octools.print_warning_log(f"SPORES has made an output of 0kb for ligand '{plants.preparedReceptor}', this is wierd. Trying to run it again.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                                    octools.print_warning(f"SPORES has made an output of 0kb for ligand '{plants.preparedReceptor}', this is wierd. Trying to run it again.")
                                    # Run again the prepare receptor
                                    _ = plants.run_prepare_receptor()
                                    # Check again if the generated receptor has size 0 or is invalid
                                    if os.path.getsize(plants.preparedReceptor) == 0 or not octools.is_molecule_valid(plants.preparedReceptor):
                                        octools.print_error_log(f"SPORES has made an output of 0kb again for receptor '{plants.preparedReceptor}'... Here is its command line so you might be able to debug it by hand.\n{' '.join(plants.prepareReceptorCmd)}", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
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
                                octools.print_warning_log(f"The PLANTS output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '*/run/plants_<runNumber>.log' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                                octools.print_warning(f"The PLANTS output for '{ptn}' run '{runNumber}' is already generated and you can check it at the '*/run/plants_<runNumber>.log' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
                    else:
                        octools.print_error_log(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                        return errors.receptor_or_ligand_not_generated(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                else:
                    octools.print_warning_log(f"The PLANTS output for '{ptn}' is already generated and you can check it at the '{dir}' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                    octools.print_warning(f"The PLANTS output for '{ptn}' is already generated and you can check it at the '{dir}' path. PLANTS execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
            else:
                octools.print_error_log(f"Wrong docking algorithm. Expected ['vina', 'smina', 'plants'] and got '{dockingAlgorithm}'.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                return errors.receptor_or_ligand_descriptor_does_not_exist(f"Wrong docking algorithm. Expected ['vina', 'smina', 'plants'] and got '{dockingAlgorithm}'.", level = "error")
        else:
            octools.print_error_log(f"There is no ligand or receptor descriptor json file for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
            return errors.receptor_or_ligand_descriptor_does_not_exist(f"There is no ligand or receptor descriptor for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
    else:
        octools.print_error_log(f"Wrong archive. Only one of the following archives is accepted ['astex', 'dudez', 'pdbbind'] and got '{archive}'.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
        return errors.receptor_or_ligand_descriptor_does_not_exist(f"Wrong archive. Only one of the following archives is accepted ['astex', 'dudez', 'pdbbind'] and got '{archive}'.", level = "error")
    return None

def __thread_run_dock_parallel(arguments):
    '''
    Thread aid function to call __core_run_dock.
    Input:
     arguments [tuple(string, string, string, bool)] - Tuple containing, in this order:
        - [string] - The directory where the files are stored
        - [string] - Which archive will be processed [dudez, pdbbind, astex]
        - [string] - Which docking algorithm will be used [vina, smina, plants]
        - [bool]   - Flag to tell if files should be overwritten
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core dock function passing the arguments correctly
        __core_run_dock(arguments[0], arguments[1], arguments[2], arguments[3])
    return None

def __run_dock_parallel(dirs, archive, dockingAlgorithm, overwrite, desc):
    '''
    Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.
    Input:
     dirs             [string] - List of paths to process
     archive          [string] - The database name (for proper logging)
     dockingAlgorithm [string] - Which docking algorithm will be used [vina, smina, plants]
     overwrite        [bool]   - Flag to tell if files should be overwritten
     desc             [string] - The description used in the progress bar
    Return:
      -
    '''
    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for dir in dirs:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((dir, archive, dockingAlgorithm, overwrite))
    # If logfile exists, backup it (for error and warnings)
    if os.path.isfile(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log"):
        if not os.path.isdir(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past/PDBbind_{dockingAlgorithm}_run_report_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")
    if os.path.isfile(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log"):
        if not os.path.isdir(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past/PDBbind_{dockingAlgorithm}_run_report_WARNING_{time.strftime('%d%m%Y-%H%M%S')}.log")
    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for _ in tqdm(p.imap_unordered(__thread_run_dock_parallel, arguments), total = len(arguments), desc = desc):
            # Clear the memory
            gc.collect()
    # Return
    return None

def __run_dock_no_parallel(dirs, archive, dockingAlgorithm, overwrite, desc):
    '''
    Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_run_dock function.
    Input:
     dirs             [string] - List of paths to process
     archive          [string] - The database name (for proper logging)
     dockingAlgorithm [string] - Which docking algorithm will be used [vina, smina, plants]
     overwrite        [bool]   - Flag to tell if files should be overwritten
     desc             [string] - The description used in the progress bar
    Return:
      -
    '''
    # If logfile exists, backup it (for error and warnings)
    if os.path.isfile(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log"):
        if not os.path.isdir(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past/PDBbind_{dockingAlgorithm}_run_report_ERROR_{time.strftime('%d%m%Y-%H%M%S')}.log")
    if os.path.isfile(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log"):
        if not os.path.isdir(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past"):
            octools.safe_create_dir(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past")
        os.rename(f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_past/PDBbind_{dockingAlgorithm}_run_report_WARNING_{time.strftime('%d%m%Y-%H%M%S')}.log")
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for dir in tqdm(iterable=dirs, total=len(dirs), desc=desc):
            # Call the core dock function (shared between parallel and not parallel)
            __core_dock(dir, archive, dockingAlgorithm, overwrite)
            # Clear the memory
            gc.collect()
    return None

### Read logs
def __core_read_log(dir, archive):
    '''
    Reads Vina, Smina and PLANTS logs and then return a dict of dataframes.
    Input:
     dir     [string] - The directory where the files are stored
     archive [string] - Which archive will be processed [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Find ptn name
    ptn = dir.split(os.path.sep)[-1]
    # Create Vina, Smina and PLANTS dataframes
    vinadf = pd.DataFrame(columns=["mode", "affinity", "rmsd_lb_best_mode", "rmsd_ub_best_mode"])
    sminadf = pd.DataFrame(columns=["mode", "affinity", "rmsd_lb_best_mode", "rmsd_ub_best_mode"])
    plantsdf = pd.DataFrame(columns=["LIGAND_ENTRY", "TOTAL_SCORE", "SCORE_RB_PEN", "SCORE_NORM_HEVATOMS", "SCORE_NORM_CRT_HEVATOMS", "SCORE_NORM_WEIGHT", "SCORE_NORM_CRT_WEIGHT", "SCORE_RB_PEN_NORM_CRT_HEVATOMS"])
    # Get all vina directories (0, 1, 2...)
    vinaDirs = glob(f"{dir}/vinaFiles/*")
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
    plantsDirs = glob(f"{dir}/plantsFiles/*")
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
    logPath = f"{dir}/sminaFiles/smina.log"
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
    # Return a dict with each read data with the protein name as index
    return {ptn: {"vina": vinadf, "smina": sminadf, "plants": plantsdf}}

def __thread_read_log_parallel(arguments):
    '''
    Thread aid function to call __core_read_log.
    Input:
     arguments [tuple(string, string, string, bool)] - Tuple containing, in this order:
        - [string] - The directory where the files are stored
        - [string] - Which archive will be processed [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_read_log(arguments[0], arguments[1])
    return None

def __read_log_parallel(dirs, archive, desc):
    '''
    Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.
    Input:
     dirs    [string] - List of paths to process
     archive [string] - Which archive will be processed [dudez, pdbbind, astex]
     desc    [string] - The description used in the progress bar
    Return:
      [dict of dicts of pd.DataFrame]
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

def __read_log_no_parallel(dirs, archive, desc):
    '''
    Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_read_log function.
    Input:
     dirs    [string] - List of paths to process
     archive [string] - Which archive will be processed [dudez, pdbbind, astex]
     desc    [string] - The description used in the progress bar
    Return:
      [dict of dicts of pd.DataFrame]
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
        for dir in tqdm(iterable = dirs, total = len(dirs), desc = desc):
            # Call the core read log function (shared between parallel and not parallel) and store the data into the data dict
            data.update(__core_read_log(dir, archive))
            # Clear the memory
            gc.collect()
    return None

### Parse into csv
def __core_generate_dock_result_csv(log_dump, ptn, archive):
    '''
    Reads Vina, Smina and PLANTS logs and then return a dict of dataframes.
    Input:
     dir     [string] - The directory where the files are stored
     ptn     [string] - Which protein is being processed
     archive [string] - Which archive will be processed [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Find which kind of archive it will be
    if archive == "astex":
        chosenArchive = astex_archive
    elif archive == "dudez":
        chosenArchive = dudez_archive
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
        reference_ligand = f"{ptn}_ligand.mol2"
        reference_ligand2 = f"{ptn}_ligand.sdf"

    # Set the target dir
    dir = f"{chosenArchive}/{ptn}"

    # The new dataframe
    df = pd.DataFrame(columns=["Protein", "vina_affinity", "smina_affinity", "plants_TOTAL_SCORE", "plants_SCORE_RB_PEN", "plants_SCORE_NORM_HEVATOMS", "plants_SCORE_NORM_CRT_HEVATOMS", "plants_SCORE_NORM_WEIGHT", "plants_SCORE_NORM_CRT_WEIGHT", "plants_SCORE_RB_PEN_NORM_CRT_HEVATOMS", "vina_rmsd", "smina_rmsd", "plants_rmsd"])

    # List to work with vina/smina/PLANTS data
    vinaData = []
    sminaData = []
    plantsData = []

    # If the vina dataframe is not empty
    if not log_dump['vina'].empty:
        # Get all vina directories (0, 1, 2...)
        vinaDirs = glob(f"{dir}/vinaFiles/*")
        for vinaDir in vinaDirs:
            # Get run number
            runNumber = vinaDir.split(os.path.sep)[-1]
            # Try to load the mol2, if fails, try the .sdf
            try:
                # Find and concatenate the RMSDs
                vinaData += octools.get_rmsd(f"{dir}/{reference_ligand}", f"{dir}/vinaFiles/{runNumber}/vina_{runNumber}.pdbqt")
            except Exception as e:
                try:
                    octools.print_warning(f"Possibly I could not load the '{reference_ligand}', trying to load the '{reference_ligand2}' instead. Error: {e}")
                    # Find and concatenate the RMSDs
                    vinaData += octools.get_rmsd(f"{dir}/{reference_ligand2}", f"{dir}/vinaFiles/{runNumber}/vina_{runNumber}.pdbqt")
                except Exception as e2:
                    octools.print_error(f"Problems while processing the Vina output for the protein '{dir}'")
                    octools.print_error_log(f"Problems while processing the Vina output for the protein '{dir}'. Error: {e2}", f"{logdir}/{archive}_dock_result_ERROR.log")

    # If the vina dataframe is not empty
    if not log_dump['smina'].empty:
        # Try to load the mol2, if fails, try the .sdf
        try:
            # Read smina data
            sminaData += octools.get_rmsd(f"{dir}/{reference_ligand}", f"{dir}/sminaFiles/smina.pdbqt")
        except Exception as e:
            try:
                octools.print_warning(f"Possibly I could not load the '{reference_ligand}', trying to load the '{reference_ligand2}' instead. Error: {e}")
                # Find and concatenate the RMSDs
                sminaData += octools.get_rmsd(f"{dir}/{reference_ligand2}", f"{dir}/sminaFiles/smina.pdbqt")
            except Exception as e2:
                octools.print_error(f"Problems while processing the Smina output for the protein '{dir}'")
                octools.print_error_log(f"Problems while processing the Smina output for the protein '{dir}'. Error: {e2}", f"{logdir}/{archive}_dock_result_ERROR.log")

    # If the plants dataframe is not empty
    if not log_dump['plants'].empty:
        # Get all PLANTS directories (0, 1, 2...)
        plantsDirs = glob(f"{dir}/plantsFiles/*")
        for plantsDir in plantsDirs:
            # Get run number
            runNumber = plantsDir.split(os.path.sep)[-1]
            # For each ligand which is in the list
            for ligand in glob(f"{dir}/plantsFiles/{runNumber}/run/{ptn}*[0-9].mol2"):
                # Try to load the mol2, if fails, try the .sdf
                try:
                    # Find and concatenate the RMSDs
                    plantsData += octools.get_rmsd(f"{dir}/{reference_ligand}", ligand)
                except Exception as e:
                    try:
                        octools.print_warning(f"Possibly I could not load the '{reference_ligand}', trying to load the '{reference_ligand2}' instead. Error: {e}")
                        # Find and concatenate the RMSDs
                        plantsData += octools.get_rmsd(f"{dir}/{reference_ligand2}", ligand)
                    except Exception as e2:
                        octools.print_error(f"Problems while processing the PLANTS output for the protein '{dir}'")
                        octools.print_error_log(f"Problems while processing the PLANTS output for the protein '{dir}'. Error: {e2}", f"{logdir}/{archive}_dock_result_ERROR.log")

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
    df.loc[len(df), df.columns] = [ptn] + vinaList + sminaList + plantsList + [minRMSD_vina, minRMSD_smina, minRMSD_plants]

    return df

def __thread_generate_dock_result_csv_parallel(arguments):
    '''
    Thread aid function to call __core_generate_dock_result_csv.
    Input:
     arguments [tuple(string, string, string, bool)] - Tuple containing, in this order:
        - [string] - The directory where the files are stored
        - [string] - Which protein is being processed
        - [string] - Which archive will be processed [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_generate_dock_result_csv(arguments[0], arguments[1], arguments[2])
    return None

def __generate_dock_result_csv_parallel(log_dumps, archive, desc):
    '''
    Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.
    Input:
     log_dumps [dict of dicts of pd.DataFrame] - The dump generated from the read_logs function
     archive   [string]                        - Which archive will be processed [dudez, pdbbind, astex]
     desc      [string]                        - The description used in the progress bar
    Return:
      [dict of dicts of pd.DataFrame]
    '''
    # If logfile exists, backup it for vina, smina and plants (for error and warnings)
    if os.path.isfile(f"{logdir}/{archive}_dock_result_ERROR.log"):
        if not os.path.isdir(f"{logdir}/generate_dock_result_csv_past"):
            octools.safe_create_dir(f"{logdir}/generate_dock_result_csv_past")
        os.rename(f"{logdir}/{archive}_dock_result_ERROR.log", f"{logdir}/read_log_past/{archive}_dock_result_ERROR.{time.strftime('%d%m%Y-%H%M%S')}.log")
    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for ptn, log_dump in log_dumps.items():
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((log_dump, ptn, archive))
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

def __generate_dock_result_csv_no_parallel(log_dumps, archive, desc):
    '''
    Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_generate_dock_result_csv function.
    Input:
     log_dumps [dict of dicts of pd.DataFrame] - The dump generated from the read_logs function
     archive   [string]                        - Which archive will be processed [dudez, pdbbind, astex]
     desc      [string]                        - The description used in the progress bar
    Return:
      [dict of dicts of pd.DataFrame]
    '''
    # If logfile exists, backup it for vina, smina and plants (for error and warnings)
    if os.path.isfile(f"{logdir}/{archive}_dock_result_ERROR..log"):
        if not os.path.isdir(f"{logdir}/generate_dock_result_csv_past"):
            octools.safe_create_dir(f"{logdir}/generate_dock_result_csv_past")
        os.rename(f"{logdir}/{archive}_dock_result_ERROR..log", f"{logdir}/read_log_past/{archive}_dock_result_ERROR.{time.strftime('%d%m%Y-%H%M%S')}.log")
    # Result DataFrame list
    dfList = []
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        for ptn, log_dump in tqdm(iterable = log_dumps.items(), total = len(log_dumps), desc = desc):
            # Call the core read log function (shared between parallel and not parallel) and assign it to the line
            dfList.append(__core_generate_dock_result_csv(log_dump, ptn, archive))
            # Clear the memory
            gc.collect()
    return pd.concat(dfList)

### Merge descriptors in dataframe
def __core_merge_descriptors_in_dataframe(dir, archive):
    '''
    Reads the descriptor and receptor json then parse them into a dataframe.
    Input:
     dir     [string] - The directory where the files are stored
     archive [string] - Which archive will be processed [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Find ptn name
    ptn = dir.split(os.path.sep)[-1]
    # <editor-fold> Create an empty dataframe with all descriptors
    ptndf = pd.DataFrame(columns=["Protein", "AUTOCORR2D_1", "AUTOCORR2D_2", "AUTOCORR2D_3", "AUTOCORR2D_4", "AUTOCORR2D_5", "AUTOCORR2D_6", "AUTOCORR2D_7", "AUTOCORR2D_8", "AUTOCORR2D_9", "AUTOCORR2D_10", "AUTOCORR2D_11", "AUTOCORR2D_12", "AUTOCORR2D_13", "AUTOCORR2D_14", "AUTOCORR2D_15", "AUTOCORR2D_16", "AUTOCORR2D_17", "AUTOCORR2D_18", "AUTOCORR2D_19", "AUTOCORR2D_20", "AUTOCORR2D_21", "AUTOCORR2D_22", "AUTOCORR2D_23", "AUTOCORR2D_24", "AUTOCORR2D_25", "AUTOCORR2D_26", "AUTOCORR2D_27", "AUTOCORR2D_28", "AUTOCORR2D_29", "AUTOCORR2D_30", "AUTOCORR2D_31", "AUTOCORR2D_32", "AUTOCORR2D_33", "AUTOCORR2D_34", "AUTOCORR2D_35", "AUTOCORR2D_36", "AUTOCORR2D_37", "AUTOCORR2D_38", "AUTOCORR2D_39", "AUTOCORR2D_40", "AUTOCORR2D_41", "AUTOCORR2D_42", "AUTOCORR2D_43", "AUTOCORR2D_44", "AUTOCORR2D_45", "AUTOCORR2D_46", "AUTOCORR2D_47", "AUTOCORR2D_48", "AUTOCORR2D_49", "AUTOCORR2D_50", "AUTOCORR2D_51", "AUTOCORR2D_52", "AUTOCORR2D_53", "AUTOCORR2D_54", "AUTOCORR2D_55", "AUTOCORR2D_56", "AUTOCORR2D_57", "AUTOCORR2D_58", "AUTOCORR2D_59", "AUTOCORR2D_60", "AUTOCORR2D_61", "AUTOCORR2D_62", "AUTOCORR2D_63", "AUTOCORR2D_64", "AUTOCORR2D_65", "AUTOCORR2D_66", "AUTOCORR2D_67", "AUTOCORR2D_68", "AUTOCORR2D_69", "AUTOCORR2D_70", "AUTOCORR2D_71", "AUTOCORR2D_72", "AUTOCORR2D_73", "AUTOCORR2D_74", "AUTOCORR2D_75", "AUTOCORR2D_76", "AUTOCORR2D_77", "AUTOCORR2D_78", "AUTOCORR2D_79", "AUTOCORR2D_80", "AUTOCORR2D_81", "AUTOCORR2D_82", "AUTOCORR2D_83", "AUTOCORR2D_84", "AUTOCORR2D_85", "AUTOCORR2D_86", "AUTOCORR2D_87", "AUTOCORR2D_88", "AUTOCORR2D_89", "AUTOCORR2D_90", "AUTOCORR2D_91", "AUTOCORR2D_92", "AUTOCORR2D_93", "AUTOCORR2D_94", "AUTOCORR2D_95", "AUTOCORR2D_96", "AUTOCORR2D_97", "AUTOCORR2D_98", "AUTOCORR2D_99", "AUTOCORR2D_100", "AUTOCORR2D_101", "AUTOCORR2D_102", "AUTOCORR2D_103", "AUTOCORR2D_104", "AUTOCORR2D_105", "AUTOCORR2D_106", "AUTOCORR2D_107", "AUTOCORR2D_108", "AUTOCORR2D_109", "AUTOCORR2D_110", "AUTOCORR2D_111", "AUTOCORR2D_112", "AUTOCORR2D_113", "AUTOCORR2D_114", "AUTOCORR2D_115", "AUTOCORR2D_116", "AUTOCORR2D_117", "AUTOCORR2D_118", "AUTOCORR2D_119", "AUTOCORR2D_120", "AUTOCORR2D_121", "AUTOCORR2D_122", "AUTOCORR2D_123", "AUTOCORR2D_124", "AUTOCORR2D_125", "AUTOCORR2D_126", "AUTOCORR2D_127", "AUTOCORR2D_128", "AUTOCORR2D_129", "AUTOCORR2D_130", "AUTOCORR2D_131", "AUTOCORR2D_132", "AUTOCORR2D_133", "AUTOCORR2D_134", "AUTOCORR2D_135", "AUTOCORR2D_136", "AUTOCORR2D_137", "AUTOCORR2D_138", "AUTOCORR2D_139", "AUTOCORR2D_140", "AUTOCORR2D_141", "AUTOCORR2D_142", "AUTOCORR2D_143", "AUTOCORR2D_144", "AUTOCORR2D_145", "AUTOCORR2D_146", "AUTOCORR2D_147", "AUTOCORR2D_148", "AUTOCORR2D_149", "AUTOCORR2D_150", "AUTOCORR2D_151", "AUTOCORR2D_152", "AUTOCORR2D_153", "AUTOCORR2D_154", "AUTOCORR2D_155", "AUTOCORR2D_156", "AUTOCORR2D_157", "AUTOCORR2D_158", "AUTOCORR2D_159", "AUTOCORR2D_160", "AUTOCORR2D_161", "AUTOCORR2D_162", "AUTOCORR2D_163", "AUTOCORR2D_164", "AUTOCORR2D_165", "AUTOCORR2D_166", "AUTOCORR2D_167", "AUTOCORR2D_168", "AUTOCORR2D_169", "AUTOCORR2D_170", "AUTOCORR2D_171", "AUTOCORR2D_172", "AUTOCORR2D_173", "AUTOCORR2D_174", "AUTOCORR2D_175", "AUTOCORR2D_176", "AUTOCORR2D_177", "AUTOCORR2D_178", "AUTOCORR2D_179", "AUTOCORR2D_180", "AUTOCORR2D_181", "AUTOCORR2D_182", "AUTOCORR2D_183", "AUTOCORR2D_184", "AUTOCORR2D_185", "AUTOCORR2D_186", "AUTOCORR2D_187", "AUTOCORR2D_188", "AUTOCORR2D_189", "AUTOCORR2D_190", "AUTOCORR2D_191", "AUTOCORR2D_192", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "EState_VSA10", "EState_VSA11", "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea", "FractionCSP3", "HallKierAlpha", "HeavyAtomMolWt", "HeavyAtomCount", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRadicalElectrons", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "qed", "RingCount", "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SMR_VSA10", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "TPSA", "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "VSA_EState10", "AUTOCORR3D_1", "AUTOCORR3D_2", "AUTOCORR3D_3", "AUTOCORR3D_4", "AUTOCORR3D_5", "AUTOCORR3D_6", "AUTOCORR3D_7", "AUTOCORR3D_8", "AUTOCORR3D_9", "AUTOCORR3D_10", "AUTOCORR3D_11", "AUTOCORR3D_12", "AUTOCORR3D_13", "AUTOCORR3D_14", "AUTOCORR3D_15", "AUTOCORR3D_16", "AUTOCORR3D_17", "AUTOCORR3D_18", "AUTOCORR3D_19", "AUTOCORR3D_20", "AUTOCORR3D_21", "AUTOCORR3D_22", "AUTOCORR3D_23", "AUTOCORR3D_24", "AUTOCORR3D_25", "AUTOCORR3D_26", "AUTOCORR3D_27", "AUTOCORR3D_28", "AUTOCORR3D_29", "AUTOCORR3D_30", "AUTOCORR3D_31", "AUTOCORR3D_32", "AUTOCORR3D_33", "AUTOCORR3D_34", "AUTOCORR3D_35", "AUTOCORR3D_36", "AUTOCORR3D_37", "AUTOCORR3D_38", "AUTOCORR3D_39", "AUTOCORR3D_40", "AUTOCORR3D_41", "AUTOCORR3D_42", "AUTOCORR3D_43", "AUTOCORR3D_44", "AUTOCORR3D_45", "AUTOCORR3D_46", "AUTOCORR3D_47", "AUTOCORR3D_48", "AUTOCORR3D_49", "AUTOCORR3D_50", "AUTOCORR3D_51", "AUTOCORR3D_52", "AUTOCORR3D_53", "AUTOCORR3D_54", "AUTOCORR3D_55", "AUTOCORR3D_56", "AUTOCORR3D_57", "AUTOCORR3D_58", "AUTOCORR3D_59", "AUTOCORR3D_60", "AUTOCORR3D_61", "AUTOCORR3D_62", "AUTOCORR3D_63", "AUTOCORR3D_64", "AUTOCORR3D_65", "AUTOCORR3D_66", "AUTOCORR3D_67", "AUTOCORR3D_68", "AUTOCORR3D_69", "AUTOCORR3D_70", "AUTOCORR3D_71", "AUTOCORR3D_72", "AUTOCORR3D_73", "AUTOCORR3D_74", "AUTOCORR3D_75", "AUTOCORR3D_76", "AUTOCORR3D_77", "AUTOCORR3D_78", "AUTOCORR3D_79", "AUTOCORR3D_80", "Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex", "SASA", "DipoleMoment", "IsoelectricPoint", "InstabilityIndex","GRAVY", "Aromaticity", "__countAA", "countA", "countR", "countN", "countD", "countC", "countQ", "countE", "countG", "countH", "countI", "countL", "countK", "countM", "countF", "countP", "countS", "countT", "countW", "countY", "countV", "TotalAALength", "AvgAALength", "countChain"])

    # </editor-fold>

    # Find which kind of archive it will be
    if archive == "astex":
        chosenArchive = astex_archive
        recpetor_descriptor = f"{astex_archive}/{ptn}/{ptn}_protein_descriptors.json"
        ligand_descriptor = f"{astex_archive}/{ptn}/{ptn}_ligand_descriptors.json"
    elif archive == "dudez":
        chosenArchive = dudez_archive
        recpetor_descriptor = f"{dudez_archive}/{ptn}/{ptn}_protein_descriptors.json"
        ligand_descriptor = f"{dudez_archive}/{ptn}/{ptn}_ligand_descriptors.json"
    elif archive == "pdbbind":
        chosenArchive = pdbbind_archive
        receptor_descriptor_path = f"{pdbbind_archive}/{ptn}/{ptn}_protein_descriptors.json"
        ligand_descriptor_path = f"{pdbbind_archive}/{ptn}/{ptn}_ligand_descriptors.json"
    else:
        octools.print_error(f"Unknown archive type. Expected one of the following: 'astex', 'dudez', 'pdbbind' and got {archive}.")
        return ptndf
    # Check if there is the receptor and the ligand descriptor json, if yes, load them
    if os.path.isfile(receptor_descriptor_path):
        receptor_descriptors = ocr.read_descriptors_from_json(receptor_descriptor_path, returnDict = True)
        # Remove unwanted keys
        for k in ["Name", "Path", "mol2Path"]
            if k in receptor_descriptors:
                del receptor_descriptors[k]
    else:
        _ = errors.file_do_not_exist(f"The file {receptor_descriptor_path} does not exist!")
        receptor_descriptors = {}
    if os.path.isfile(ligand_descriptor_path):
        ligand_descriptors = ocl.read_descriptors_from_json(ligand_descriptor_path, returnDict = True)
        # Remove unwanted keys
        for k in ["Name", "Path"]
            if k in receptor_descriptors:
                del receptor_descriptors[k]
    else:
        _ = errors.file_do_not_exist(f"The file {ligand_descriptor_path} does not exist!")
        ligand_descriptors = {}
    # Create new dict
    all_descriptors = dict()
    # Set Name and Path
    all_descriptors["Protein"] = ptn
    # Merge both descriptors dicts
    all_descriptors = {**all_descriptors, **receptor_descriptors}
    all_descriptors = {**all_descriptors, **ligand_descriptors}
    tmpdf = pd.DataFrame(all_descriptors, index=[0])
    # Append the line to the DataFrame
    ptndf = pd.concat([ptndf, tmpdf], ignore_index=True)
    # Return the dataframe with a single row
    return ptndf

def __thread_merge_descriptors_in_dataframe_parallel(arguments):
    '''
    Thread aid function to call __core_merge_descriptors_in_dataframe.
    Input:
     arguments [tuple(string, string, string, bool)] - Tuple containing, in this order:
        - [string] - The directory where the files are stored
        - [string] - Which archive will be processed [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Call the core read log function passing the arguments correctly
        return __core_merge_descriptors_in_dataframe(arguments[0], arguments[1])
    return None

def __merge_descriptors_in_dataframe_parallel(dirs, archive, desc):
    '''
    Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.
    Input:
     dirs    [string] - List of paths to process
     archive [string] - Which archive will be processed [dudez, pdbbind, astex]
     desc    [string] - The description used in the progress bar
    Return:
      [pd.DataFrame]
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

def __merge_descriptors_in_dataframe_no_parallel(dirs, archive, desc):
    '''
    Warper to prepare the jobs, recieves a list of directories, and pass one by one, sequentially to the __core_read_log function.
    Input:
     dirs    [string] - List of paths to process
     archive [string] - Which archive will be processed [dudez, pdbbind, astex]
     desc    [string] - The description used in the progress bar
    Return:
      [dict of dicts of pd.DataFrame]
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
            ptndf = pd.concat([ptndf, __core_read_log(dir, archive)])
            # Clear the memory
            gc.collect()
    return ptndf

## Public ##
def verify_integrity(chosenArchive, spacing = 0.33):
    '''
    Verifies the integrity of the desired database
    Input:
     chosenArchive [string]               - Which archive will be processed. [dudez, pdbbind, astex]
     spacing       [float]  DEFAULT: 0.33 - Extra spacing for the sphere in percentage. (To ensure that all the sites will be accounted) (ONLY USED IN Smina)
    Return:
      -
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
    if os.path.isfile(f"{logdir}/PDBbind_integrity_report.log"):
        if not os.path.isdir(f"{logdir}/pdbbind_integrity_past"):
            octools.safe_create_dir(f"{logdir}/pdbbind_integrity_past")
        os.rename(f"{logdir}/PDBbind_integrity_report.log", f"{logdir}/pdbbind_integrity_past/PDBbind_integrity_report_{time.strftime('%d%m%Y-%H%M%S')}.log")

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
                    octools.print_error_log(f"Unable to generate the p2rank dir for '{dir}'... Error code {errorCode}.", f"{logdir}/PDBbind_integrity_report.log")
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
                    octools.print_error_log(f"Unable to generate the vinaFiles dir for '{dir}'... Error code {errorCode}.", f"{logdir}/PDBbind_integrity_report.log")
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
                    octools.print_error_log(f"Unable to generate the plantsFiles dir for '{dir}'... Error code {errorCode}.", f"{logdir}/PDBbind_integrity_report.log")
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
                    octools.print_error_log(f"The protein '{dir}' still has no box file.", f"{logdir}/PDBbind_integrity_report.log")
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
                    octools.print_error_log(f"Unable to generate the vina conf files for '{dir}'...", f"{logdir}/PDBbind_integrity_report.log")
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
                    octools.print_error_log(f"Unable to generate the PLANTS conf files for '{dir}'...", f"{logdir}/PDBbind_integrity_report.log")
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
                        octools.print_error_log(f"Unable to generate the ligand descriptor file dir for '{dir}'...", f"{logdir}/PDBbind_integrity_report.log")
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
                        octools.print_error_log(f"Unable to generate the receptor descriptor file dir for '{dir}'...", f"{logdir}/PDBbind_integrity_report.log")
                        failed = failed + 1
                        continue

    octools.printv(f"Integrity check of the PDBbind database accomplished. Success rate: {((lenDirs - failed) / lenDirs) * 100}% ({(lenDirs - failed)}/{lenDirs})")

    return

def convert_debug_to_production(chosenArchive, chosenAlgorithm = "ac", strict = False, removeDebug = False):
    '''
    Converts debug folders to production mode. It is required to choose an algorithm which will be used furtherly in the pipeline.
    Input:
     chosenArchive   [string]              - Which archive will be processed. [dudez, pdbbind, astex]
     chosenAlgorithm [string] DEFAULT: ac  - The short code for the chosen algorithm. The choices are:
                                                AffinityPropagation: ap
                                                AgglomerativeClustering: ac
                                                Birch: bi
                                                DBSCAN: db
                                                KMeans:  km
                                                MeanShift: ms
                                                MiniBatchKMeans: mb
                                                NoCluster: na
                                                OPTICS: op
                                                SpectralClustering: sc
     strict          [bool] DEFAULT: False - If True does not convert the data even if there is only one dir, if False will convert the data if the protein has only one dir (this is good when you ran with only one algorithm, some proteins may have been run with "na")
     removeDebug     [bool] DEFAULT: False - If True removes debug folders (NO TURNING BACK), if False leave the dirs
    Return:
      -
    '''
    # Generate boxes for all receptors
    octools.printv("Converting p2rank debug to production file tree.")

    # Get all dirs paths in the DUDEz database
    dirs = glob(f"{chosenArchive}/*")

    # Set the allowed values
    allowed = ["ap", "ac", "bi", "db", "km", "ms", "mb", "na", "op", "sc"]

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
                        octools.print_error_log(f"The protein '{dir}' has no box!!!!!", f"{logdir}/PDBbind_conversion_report.log")
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
                        if algoritm == chosenAlgorithm:
                            # Set the hasDir as true
                            hasDir = True
                            # Get the boxes
                            boxes = glob(f"{p2rankFile}/*")
                            # If no box is found (folders WILL NOT BE REMOVED)
                            if len(boxes) < 1:
                                octools.print_error(f"The protein '{dir}' has no box!!!!!")
                                octools.print_error_log(f"The protein '{dir}' has no box!!!!!", f"{logdir}/PDBbind_conversion_report.log")
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
                    octools.print_error_log(f"The algorithm '{chosenAlgorithm}' has not been found for the protein '{dir}'.")
            else:
                octools.printv(f"Nothing to convert for '{dir}'. Skipping...")
                continue
    return

def prepare(archive, overwrite = False, spacing = 0.33, sanitize = True):
    '''
    Prepares the database.
    Input:
     archive   [string]                - Which archive will be processed. [dudez, pdbbind, astex]
     overwrite [bool]   DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
     spacing   [float]  DEFAULT: 0.33  - Extra spacing for the sphere in percentage. (To ensure that all the sites will be accounted)
     sanitize  [bool]   DEFAULT: True  - Flag to denote if the molecule should be sanitized
    Return:
      -
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
        return
    # Generate boxes for all receptors
    octools.printv("Generating information regarding possible ligand site.")
    # If is multiprocess
    if args.multiprocess:
        # If the archive is pdbbind
        if archive == "pdbbind":
            # Get all dirs paths in the database
            dirs = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index', 'db']]
            # Prepare the pdbbind
            __prepare_parallel(dirs, overwrite, archive, sanitize, spacing, "PDBbind proteins")
        else:
            # Get all dirs paths in the database
            dirs = glob(f"{chosenArchive}/*")
            # Prepare the database
            __prepare_parallel(dirs, overwrite, archive, sanitize, spacing, f"{chosenArchive} proteins")
    return None

def run_dock(archive, dockingAlgorithm, overwrite = False):
    '''
    Run docking.
    Input:
     archive          [string]                - Which archive will be processed. [dudez, pdbbind, astex]
     dockingAlgorithm [string]                - Which docking software will be run. [vina, smina, plants]
     overwrite        [bool]   DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
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
    # Check if the docking algorithm is valid
    if dockingAlgorithm not in ["vina", "smina", "plants"]:
        octools.print_error(f"Docking software not recognized. Expected ('vina', 'smina', 'plants') and got '{dockingAlgorithm}'.")
        return None
    # Get all dirs paths in the database
    dirs = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index', 'db']]
    # Decide if multprocessing will be used
    if args.multiprocess:
        __run_dock_parallel(dirs, archive, dockingAlgorithm, overwrite, f"Processing {archive}")
    else:
        __run_dock_no_parallel(dirs, archive, dockingAlgorithm, overwrite)
    return None

def get_database_single_file(archive):
    '''
    Parse the database into a SINGLE serializable object. (Not so good)
    Input:
     archive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      [dict of tuples]
    '''
    # Make archive lowercase
    archive = archive.lower()

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
    # Get all dirs inside the database
    dirs = glob(f"{chosenArchive}/*")
    # Dict of elements
    databaseDict = dict()
    # Decide if multprocessing will be used
    if args.multiprocess:
        databaseDict = __get_parallel(dirs, archive, f"Processing {archive}")
    else:
        databaseDict = __get_no_parallel(dirs, archive)
    return databaseDict

def get_database_multiple_files(archive, sliceSize = 100):
    '''
    Parse the database into multiple serializable objects.
    Input:
     archive   [string]              - Which archive will be processed. [dudez, pdbbind, astex]
     sliceSize [int]    DEFAULT: 100 - Number of elements in each chunk. (Please, always use the same value)
    Return:
      [dict of tuples]
    '''
    # Make archive lowercase
    archive = archive.lower()

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
    # Get all dirs inside the database (except index and db)
    dirs = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index', 'db']]
    # Create the db dir if does not exsit yet
    _ = octools.safe_create_dir(f"{chosenArchive}/db")
    # Slice it into chunks
    chunkedDirs = [dirs[x:x + sliceSize] for x in range(0, len(dirs), sliceSize)]
    # For each chunk
    for i, chunkedDir in enumerate(chunkedDirs):
        if os.path.isfile(f"{chosenArchive}/db/pdbbind_{i}.pickle"):
            octools.print_warning(f"The file '{chosenArchive}/db/pdbbind_{i}.pickle' already exists. Skipping.")
            continue
        # Dict of elements
        databaseDict = dict()
        # Decide if multprocessing will be used
        if args.multiprocess:
            databaseDict = __get_parallel(chunkedDir, archive, f"Processing {archive}")
        else:
            databaseDict = __get_no_parallel(chunkedDir, archive)
        # Test if dabaseDict is fine
        if databaseDict:
            octools.to_pickle(f"{chosenArchive}/db/pdbbind_{i}.pickle", databaseDict)

    return databaseDict

def read_logs(archive, picklePath = ""):
    '''
    Reads database logfiles returning a dict of dicts of pd.DataFrames.
    Input:
     archive    [string]             - Which archive will be processed. [dudez, pdbbind, astex]
     picklePath [string] DEFAULT: "" - The path where to store the pickle file. If empty no pickle file will be generated.
    Return:
     [dict of dicts of pd.DataFrame]
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
    # Get all dirs paths in the database
    dirs = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index', 'db']]
    # Make data be None (in case of failure)
    data = None
    # Decide if multprocessing will be used
    if args.multiprocess:
        data = __read_log_parallel(dirs, archive, f"Processing {archive}")
    else:
        data = __read_log_no_parallel(dirs, archive, f"Processing {archive}")
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
        return
    # Return the data
    return data

def generate_dock_result_csv(archive, log_dumps, csv_path, chunksize=500):
    '''
    Uses the structure from read_logs to generate an output for all docking softwares.
    Input:
     archive   [string]                                     - Which archive will be processed [dudez, pdbbind, astex]
     log_dumps [dict of dicts of pd.DataFrame]              - The dump generated from the read_logs function
     csv_path  [string]                                     - Path to the csv file
     chunksize [int]                           DEFAULT: 500 - Chunk size to write the csv
    Return:
     -
    '''
    # Decide if multprocessing will be used
    if args.multiprocess:
        data = __generate_dock_result_csv_parallel(log_dumps, archive, f"Generating docking csv {archive}")
    else:
        data = __generate_dock_result_csv_no_parallel(log_dumps, archive, f"Generating docking csv {archive}")
    # Check if data is not empty
    if not data.empty:
        data.to_csv(csv_path, index=False, chunksize=chunksize)
    return

def merge_descriptors_in_dataframe(archive, saveCsv=True):
    '''
    Reads all the descriptors jsons and return a pd.DataFrame.
    Input:
     archive [string]               - Which archive will be processed. [dudez, pdbbind, astex]
     saveCsv [bool]   DEFAULT: True - If True will save to the Prepared folder in the database
    Return:
     [pd.DataFrame]
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
    # Get all dirs paths in the database
    dirs = [d for d in glob(f"{chosenArchive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['index', 'db']]

    # Make data be None (in case of failure)
    data = None
    # Decide if multprocessing will be used
    if args.multiprocess:
        data = __merge_descriptors_in_dataframe_parallel(dirs, archive, f"Processing {archive}")
    else:
        data = __merge_descriptors_in_dataframe_no_parallel(dirs, archive, f"Processing {archive}")
    # Check if data is pd.DataFrame type and is not empty
    if type(data) == pd.DataFrame and not data.empty:
        # Try to write the csv
        try:
            # Parameterize the csvs paths
            csv_path_in = f"{parsed_archive}/PDBbind.csv"
            csv_path_out = f"{parsed_archive}/PDBbind_complete.csv"
            # Read the csv from input file
            ptndf = pd.read_csv(csv_path_in)
            # Merge the both DataFrames using the Protein column as a comparer
            data = pd.merge(ptndf, data, on="Protein", how="left")
            # Write the data to a new csv file
            data.to_csv(csv_path_out, index=False)
            octools.print_success(f"The file '{csv_path_out}' has been successfully written.")
        except Exception as e:
            octools.print_error(f"Could not write the file '{csv_path_out}'. Error: {e}")
            # Return Nothing
            return
    else:
        octools.print_warning(f"The data object is not defined! There is no reason to write it as a pickle. Aborting...")
        # Return nothing
        return

    # Return the data
    return data
