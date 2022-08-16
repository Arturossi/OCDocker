#!/usr/lib/python3

# Imports
###############################################################################
import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

from OCDocker.Initialise import *
import OCDocker.Ligand as ocl
import OCDocker.baseDB as ocbdb
import OCDocker.Toolbox as octools

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
Sets of classes and functions that are used to process the DUDE-Z dataset.

They are imported as:

import OCDocker.DUDEz as ocdudez
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##
def __thread_validation(arguments):
    '''
    Function to
    Input:
     arguments [tuple(string, string)] - Tuple containing, in this order:
        - [string] The molecule path
        - [string] The database dir
    Return:
      [int]
        1 - If a problem has been found
        0 - If no problem has been found
    '''
    # Unwarping arguments
    mol = arguments[0]
    database = arguments[1]
    # Get its name
    molName = os.path.splitext(os.path.basename(mol))[0]
    # Create a ligand object of it
    l = ocl.Ligand(mol, molName, sanitize = False, from_json_descriptors = f"{database}/{molName}_descriptors.json")
    # Check its validity
    if not l.is_valid():
        # If not valid, throw an error
        _ = errors.malformed_molecule(f"The molecule '{mol}' has some problems on it. One or more its descriptors are lacking!")
        # Print problems to log
        octools.print_error_log(f"The molecule '{mol}' has some problems on it. One or more its descriptors are lacking!", f"{logdir}/DUDEz_molecule_validation_report.log")
        # Add one more problematic molecule to the counter
        return 1
    return 0

def __inner_validate_database_molecules(database, subset):
    '''
    Validates all the molecules in the DUDEz database.
    Input:
      database [string] - The database dir
      subset   [string] - The database subset (DUDE_Z_ligands, DUDE_Z_decoys, extrema_decoys, goldilocks_decoys)
    Return:
      -
    '''
    problematicMolsNum = 0
    if not os.path.isdir(database):
        _ = errors.dir_does_not_exists(f"The directory '{database}' does not exist. {clrs['p']}PLEASE{clrs['n']}, review this!")
    else:
        # Get all the molecules with 1pt0LD pattern (optimized)
        # TODO: "This is a temporary solution. It should be changed to read the smiles format instead."
        mols = glob(f"{database}/*1pt0LD*.mol2")
        # Arguments to pass to each Thread in the Thread Pool
        arguments = []
        # For each file in the glob
        for mol in mols:
            # Append a tuple containing the file name and ovewrite flag to the arguments list
            arguments.append((mol, database))
        # Count the number of molecules
        lenMols = len(mols)
        # Create a Thread pool with the maximum available_cores
        with Pool(args.available_cores) as p:
            # Redirect output to tqdm.write
            with redirect_to_tqdm():
                # For each molecule in dudezDirLigand (multiThreaded)
                for hasProblem in tqdm(p.imap_unordered(__thread_validation, arguments), total=lenMols, desc = subset):
                    # Add the problem num, it can be 0 for no problem and 1 for some problem
                    problematicMolsNum += hasProblem
        # Parameterize error string
        problematicMolsError = f"In dudez database there are {problematicMolsNum} problematic molecules."
        # If there is any problematic molecule
        if problematicMolsNum > 0:
            # Tell the user to look in the log
            problematicMolsError = f"{problematicMolsError} Find more info in the log stored at '{logdir}/DUDEz_molecule_validation_report.log'."
        else:
            # Cheer the user up
            problematicMolsError = f"{problematicMolsError} Phew!"
        # Print the error (or be happy with no error!)
        octools.print_info(problematicMolsError, force = True)

    return

def __validate_database_molecules():
    '''
    Validates all the molecules in the DUDEz database.
    Input:
      -
    Return:
      -
    '''
    # Get all dirs paths in the database
    dirs = glob(f"{dudez_archive}/*")
    # For each directory
    for dir in dirs:
        # Get target Name
        targetName = dir.split(os.path.sep)[-1]

        # Set the 3 dirs containing ligand/decoys
        dudezDir = f"{dir}/DUDE_Z"
        extremaDir = f"{dir}/Extrema"
        goldilocksDir = f"{dir}/Goldilocks"

        # Parameterize paths
        dudezDirLigand = f"{dudezDir}_ligands"
        dudezDirDecoy = f"{dudezDir}_decoys"
        extremaDirDecoy = f"{extremaDir}_decoys"
        goldilocksDirDecoy = f"{goldilocksDir}_decoys"

        # Check if dudezDirLigand exists
        __inner_validate_database_molecules(dudezDirLigand, f"{targetName} DUDEz ligands")
        # Check if dudezDirDecoy exists
        __inner_validate_database_molecules(dudezDirDecoy, f"{targetName} DUDEz decoys")
        # Check if extremaDirDecoy exists
        __inner_validate_database_molecules(extremaDirDecoy, f"{targetName} extrema decoys")
        # Check if goldilocksDirDecoy exists
        __inner_validate_database_molecules(goldilocksDirDecoy, f"{targetName} goldilocks decoys")

    return

def __paralel_check_repeated_ligands(arguments):
    '''
    Runs the ligand simmilarity check in parallel.
    Input:
      arguments [tuple(Ligand,list(Ligand))] - A tuple with 2 positions, the first is the reference ligand and the list of ligands to be compared.
    Return:
      -
    '''
    # Change the ligand name to a more readable one
    ligand = arguments[0]
    # Change the ligand list to a more readable one
    ligandsToCompare = arguments[1]
    # Check if ligandsToCompare is valid
    if not ligandsToCompare:
        _ = errors.empty(f"The ligand list to the ligand {ligand.path} is empty.", force = True)
        octools.print_error_log(f"The ligand list to the ligand {ligand.path} is empty.", f"{logdir}/DUDEz_database_redundant_residues.log")
        # Skip
        return None
    # Get the ligand path
    ligandPath = os.path.dirname(ligand.path)
    # Create the unique and not unique reference files names
    ligandFileName = f"{ligandPath}/{ligand.name}"
    uniqueFile = f"{ligandFileName}_unique"
    notUniqueReferenceFile = f"{ligandPath}/{ligand.name}_NOTunique"
    # List of not unique files
    notUniqueFiles = []
    # If there is no uniqueFile means that this file has not been checked
    if not os.path.isfile(uniqueFile) and not os.path.isfile(notUniqueReferenceFile):
        # For each ligand in ligand list
        for l in ligandsToCompare:
            # Find the l.path
            lPath = os.path.dirname(l.path)
            # If no target ligand file for uniqueness or not uniquness exists means that this file has not been checked
            if not os.path.isfile(f"{lPath}/{l.name}_unique") and not os.path.isfile(f"{lPath}/{l.name}_NOTunique"):
                # Check if the molecules are the same
                if ligand.is_same_molecule(l, sanitize = False):
                    # If they are the same, print the error to the user
                    octools.print_error(f"The ligand {ligand.path} and the ligand {l.path} might be the same. It is advised to check them manually.", force = True)
                    octools.print_error_log(f"The ligand {ligand.path} and the ligand {l.path} might be the same. It is advised to check them manually.", f"{logdir}/DUDEz_database_redundant_residues.log")
                    # Append the l.path and l.name to notUniqueFilesLit
                    notUniqueFiles.append(f"{lPath}/{l.name}")
        # If the ligand is unique
        if not notUniqueFiles:
            # Write the file
            with open(uniqueFile, "w") as f:
                # Whatever if file was already existing
                try:
                    # Set current time anyway
                    os.utime(uniqueFile, None)
                except OSError:
                    # File deleted between open() and os.utime() calls
                    pass
        else:
            # Write the file
            with open(notUniqueReferenceFile, "w") as f:
                # Whatever if file was already existing
                try:
                    # Write the ligand file name to the list
                    f.write(f"{ligandFileName}\n")
                    # For each element in the not unique path list
                    for notUnique in notUniqueFiles:
                        # Join the paths for future checks and write them to the file
                        f.write(f"{notUnique}\n")
                except OSError:
                    # File deleted between open() and os.utime() calls
                    pass
            # For each other file (since they are the same, write the not unique file will save time and avoid errors)
            for notUnique in notUniqueFiles:
                notUniqueFile = f"{notUnique}_NOTunique"
                # Write the file
                with open(notUniqueFile, "w") as f:
                    # Whatever if file was already existing
                    try:
                        # Write the ligand file name to the list
                        f.write(f"{ligandFileName}\n")
                        # For each element in the not unique path list
                        for notUnique2 in notUniqueFiles:
                            # Join the paths for future checks and write them to the file
                            f.write(f"{notUnique2}\n")
                    except OSError:
                        # File deleted between open() and os.utime() calls
                        pass
    return None

def __check_for_repeated_ligands():
    '''
    Checks if there is any repeated ligand in the DUDEz database.
    Input:
      -
    Return:
      -
    '''
    # Get all dirs paths in the database
    dirs = glob(f"{dudez_archive}/*")
    # List to hold all comparisons
    arguments = []
    # For each directory
    for dir in dirs:
        # Create a ligand list for the currend molecule and fill it with all its ligands
        ligands = get_ligands_from_molecule(dir)
        # For each ligand in the list, get its index
        for i in range(len(ligands)):
            # List of ligands to compare
            innerToCompare = []
            # Get the index of all next elements in the list
            for j in range(i + 1, len(ligands)):
                # Add the element to the ligand list
                innerToCompare.append(ligands[j])
            # If there is a list to compare
            if innerToCompare:
                # Add the tuple to the list
                arguments.append((ligands[i], innerToCompare))
    # Create the pool with available_cores
    with Pool(args.available_cores) as p:
        # Redirect output to tqdm.write
        with redirect_to_tqdm():
            # For each molecule in dudezDirLigand (multiThreaded)
            for _ in tqdm(p.imap_unordered(__paralel_check_repeated_ligands, arguments), total = len(arguments), desc = "DUDEz checking"):
                pass

    return

## Public ##
def get_all_ligands():
    '''
    Gets all the ligands in the DUDEz database.
    Input:
      molecule [string] - Path of the molecule.
    Return:
      list(ocl.Ligand) - A list of all ligands in the DUDEz database
    '''
    # Create a ligand list and fill it with ALL ligands
    ligands = []
    # Get all dirs paths in the database
    dirs = glob(f"{dudez_archive}/*")
    # For each directory
    for dir in dirs:
        octools.print_info(f"Reading the molecule from directory '{dir}'.")
        # Append the new ligands list to the end of the already existant ligands list
        ligands.extend(get_ligands_from_molecule(dir))

    return ligands

def get_ligands_from_molecule(molecule):
    '''
    Gets all the ligands in the DUDEz database.
    Input:
      molecule [string] - Path of the molecule.
    Return:
      list(ocl.Ligand) - A list of all ligands in the DUDEz database
    '''
    # Get target Name
    targetName = molecule.split(os.path.sep)[-1]

    # Set the 3 dirs containing ligand/decoys
    dudezDir = f"{molecule}/DUDE_Z"
    extremaDir = f"{molecule}/Extrema"
    goldilocksDir = f"{molecule}/Goldilocks"

    # Parameterize paths
    dudezDirLigand = f"{dudezDir}_ligands"
    dudezDirDecoy = f"{dudezDir}_decoys"
    extremaDirDecoy = f"{extremaDir}_decoys"
    goldilocksDirDecoy = f"{goldilocksDir}_decoys"

    # Create a ligand list and fill it with ALL the given molecule ligands
    ligands = []

    # Create a list of databases
    databases =[dudezDirLigand, dudezDirDecoy, extremaDirDecoy, goldilocksDirDecoy]

    # For each database in the list
    for db in databases:
        # Get a list of .mol2 molecules
        mols = glob(f"{db}/*.mol2")
        # Redirect output to tqdm.write
        with redirect_to_tqdm():
            # For each .mol2 file in dudezDirLigand directory
            for l in tqdm(iterable = mols, total = len(mols), desc = f"Molecules processed for '{targetName}'."):
                # Find the ligand name in the DUDEz ligand database
                ligandName = os.path.splitext(os.path.basename(l))[0]
                # Append to the ligands list its ligand (without sanitization, because it return errors when there is a N in a cyclic strucuture, my guess)
                ligands.append(ocl.Ligand(l, ligandName, sanitize = False, from_json_descriptors = f"{db}/{ligandName}_descriptors.json"))

    return ligands

def verify_integrity():
    '''
    Verifies the integrity of the DUDEz database
    Input:
      -
    Return:
      -
    '''
    octools.print_info("Verifiying the integrity of the DUDEz database.", force = True)
    ocbdb.verify_integrity(dudez_archive)
    octools.print_info("Verifiying the integrity of the DUDEz ligand candidates.", force = True)
    __validate_database_molecules()
    octools.print_info("Checking for repeated ligands.", force = True)
    __check_for_repeated_ligands()

    return

def convert_debug_to_production(chosenAlgorithm = "ac", strict = False, removeDebug = False):
    '''
    Converts debug folders to production mode. It is required to choose an algorithm which will be used furtherly in the pipeline.
    Input:
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
    ocbdb.convert_debug_to_production(dudez_archive, chosenAlgorithm = chosenAlgorithm, strict = strict, removeDebug = removeDebug)

def prepare(overwrite = False):
    '''
    Prepares the DUDEz database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with the output already generated.
    Return:
      -
    '''
    # Prepare the databse
    ocbdb.prepare("dudez", overwrite = overwrite)
    # Verify its integrity
    #verify_integrity()

def run_p2rank(overwrite = False):
    '''
    Runs P2Rank in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_p2rank("dudez", overwrite = overwrite)

def run_vina(overwrite = False):
    '''
    Runs vina in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_dock("dudez", "vina", overwrite = overwrite)

def run_smina(overwrite = False):
    '''
    Runs smina in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_dock("dudez", "smina", overwrite = overwrite)

def run_plants(overwrite = False):
    '''
    Runs PLANTS in the whole database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    return ocbdb.run_dock("dudez", "plants", overwrite = overwrite)