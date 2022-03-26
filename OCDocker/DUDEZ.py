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
        _ = errors.malformedMolecule(f"The molecule '{mol}' has some problems on it. One or more its descriptors are lacking!")
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
        # Get all the molecules
        mols = glob(f"{database}/*.mol2")
        # Arguments to pass to each Thread in the Thread Pool
        arguments = []
        # For each file in the glob
        for mol in mols:
            # Append a tuple containing the file name and ovewrite flag to the arguments list
            arguments.append((mol, database))
        # Count the number of molecules
        lenMols = len(mols)
        # Create a Thread pool with the maximum available_cores
        p = Pool(args.available_cores)
        # For each molecule in dudezDirLigand (multiThreaded)
        for hasProblem in tqdm(p.imap_unordered(__thread_validation, arguments), total=lenMols, desc = subset):
            # Add the problem num, it can be 0 for no problem and 1 for some problem
            problematicMolsNum += hasProblem
        # Close the pool
        p.close()
        # Wait the pool to join
        p.join()
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
    # For each directory
    for dir in dirs:
        # Create a ligand list for the currend molecule and fill it with all its ligands
        ligands = get_ligands_from_molecule(dir)
        # For each ligand in the list, get its index
        for i in range(len(ligands)):
            # Get the index of all next elements in the list
            for j in range(i + 1, len(ligands)):
                # Check if the molecules are the same
                if ligands[i].is_same_molecule(ligands[j]):
                    # If they are the same, print the error to the user
                    octools.print_error(f"The ligand {ligands[i]} and the ligand {ligands[j]} might be the same. It is advised to check them manually.", force = True)
                    octools.print_error_log(f"The ligand {ligands[i]} and the ligand {ligands[j]} might be the same. It is advised to check them manually.", f"{logdir}/DUDEz_database_redundant_residues.log")

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
        # For each .mol2 file in dudezDirLigand directory
        for l in glob(f"{db}/*.mol2"):
            # Find the ligand name in the DUDEz ligand database
            ligandName = os.path.splitext(os.path.basename(l))[0]
            # Append to the ligands list its ligand
            ligands.append(ocl.Ligand(l, ligandName, from_json_descriptors = f"{db}/{ligandName}_descriptors.json"))

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
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    ocbdb.prepare(dudez_archive, overwrite = overwrite)
