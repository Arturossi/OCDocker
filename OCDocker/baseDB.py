#!/usr/lib/python3

# Imports
###############################################################################
import os
import shutil
from glob import glob
from tqdm import tqdm

from OCDocker.Initialise import *
import OCDocker.Ligand as ocl
import OCDocker.Vina as ocvina
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
def __run_p2rank(dir, fin):
    '''
    Runs p2rank for a given directory.
    Input:
      dir [string] - Directory of the protein to run p2rank.
      fin [string] - PDB file as input.
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
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = args.verbosity)
    except Exception as e:
        octools.print_warning(f"The protein '{dir}' had a problem while running p2rank. Retrying to run p2rank. Exception: {e}  ")
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = args.verbosity)

    return

def __run_create_vina_conf_from_box(dir, fin):
    '''
    Creates vina conf file from box
    Input:
      dir [string] - Directory of the protein to run p2rank.
      fin [string] - PDB file as input.
    Return:
      -
    '''
    # Run vina
    ocvina.generate_vina_files_database(dir, fin)

    return

## Public ##
def verify_integrity(chosenArchive):
    '''
    Verifies the integrity of the DUDEZ database
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Verify the integrity of the database
    octools.printv("Verifiying the integrity of the DUDEZ database")

    # Get all dirs paths in the database
    dirs = glob(f"{chosenArchive}/*")

    # Counter for failed proteins
    failed = 0

    # Parameterizing the amount of directories
    lenDirs = len(dirs)

    # Find the archive type
    archive = chosenArchive.split(os.path.sep)[-1].lower()

    # For each directory in the database folder
    for dir in tqdm(iterable=dirs, total=lenDirs):
        # Parameterizing paths
        p2rankDir = f"{dir}/p2rank"
        vinaDir = f"{dir}/vinaFiles"

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

        octools.printv(f"Checking files for the protein '{dir}'")

        # Check how many boxes are in the p2rankDir
        boxCount = len(glob(f"{p2rankDir}/box*.pdb"))

        # If there is no box in the p2rank output, p2rank will run
        if boxCount == 0:
            octools.print_warning(f"The protein '{dir}' has no box file. Trying to fix...")

            # Run p2rank
            __run_p2rank(dir)

            # Check how many boxes are in the p2rankDir (again)
            boxCount = len(glob(f"{p2rankDir}/box*.pdb"))

            if boxCount > 0:
                octools.print_success(f"Box files generated for '{dir}'.")
            else:
                octools.print_error(f"The protein '{dir}' still has no box file.")
                octools.print_error_log(f"The protein '{dir}' still has no box file.", f"{logdir}/PDBbind_integrity_report.log")
                failed = failed + 1
                continue

        # If there is not the same amount of box files as folders in vinaFiles folder
        if len(glob(f"{dir}/vinaFiles/*")) < boxCount:
            octools.print_warning(f"The protein '{dir}' has not the same amount of vina conf files as the amount of box files. Trying to fix...")

            # Set the input file name path and set the input file name path
            if archive == "astex":
                fin = f"{dir}/protein.pdb"
            elif archive == "dudez":
                fin = f"{dir}/rec.crg.pdb"
            elif archive == "pdbbind":
                fin = f"{dir}/{dir.split(os.path.sep)[-1]}_protein.pdb"

            # Run the vina conf creation from box
            __run_create_vina_conf_from_box(dir, fin)

            # If there is not the same amount of box files as folders in vinaFiles folder (again)
            if len(glob(f"{dir}/vinaFiles/*")) == boxCount:
                octools.print_success(f"Conf files generated for '{dir}'.")
            else:
                octools.print_error(f"Unable to generate the conf files for '{dir}'...")
                octools.print_error_log(f"Unable to generate the conf files dir for '{dir}'...", f"{logdir}/PDBbind_integrity_report.log")
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

    # Get all dirs paths in the DUDEZ database
    dirs = glob(f"{chosenArchive}/*")

    # Set the allowed values
    allowed = ["ap", "ac", "bi", "db", "km", "ms", "mb", "na", "op", "sc"]

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

def prepare(archive, overwrite = False):
    '''
    Prepares the database.
    Input:
     archive   [string]                - Which archive will be processed. [dudez, pdbbind, astex]
     overwrite [bool]   DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
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
        return

    # Generate boxes for all receptors
    octools.printv("Generating information regarding possible ligand site.")

    # Get all dirs paths in the DUDEZ database
    dirs = glob(f"{chosenArchive}/*")

    # For each directory in the database folder
    for dir in tqdm(iterable=dirs, total=len(dirs)):
        # Find the protein name
        ptn = dir.split(os.path.sep)[-1]

        # Set the input file name path
        if archive == "astex":
            # Set the input file name path
            fin = f"{dir}/protein"

            # Set the ligand input file name path
            lfin = f"{dir}/ligand"

            # If the overwrite flag is true or the receptor pdb file already exists
            if overwrite or not os.path.isfile(f"{fin}.pdb"):
                # Convert the protein file from mol2 to pdb
                _ = octools.convertMols(f"{fin}.mol2", f"{fin}.pdb")

            # If the overwrite flag is true or the ligand mol2 file already exists
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

            # Split the file
            _ = octools.split_and_convert(f"{dudezDir}/dudez_0pt5LD_ligand_poses.mol2", dudezDirLigand, "mol2")
            _ = octools.split_and_convert(f"{dudezDir}/dudez_0pt5LD_decoy_poses.mol2", dudezDirDecoy, "mol2")
            _ = octools.split_and_convert(f"{extremaDir}/extrema_0pt5LD_decoy_poses.mol2", extremaDirDecoy, "mol2")
            _ = octools.split_and_convert(f"{goldilocksDir}/goldilocks_0pt5LD_decoy_poses.mol2", goldilocksDirDecoy, "mol2")

            # For each molecule in dudez ligand dir
            for mol in glob(f"{dudezDirLigand}/*.mol2"):
                # Find its name
                molName = ".".join(os.path.basename(mol).split(".")[:-1])
                # Create the ligand object
                l = ocl.Ligand(mol, molName)
                # Export its descriptors
                _ = l.to_json(overwrite)

            # For each molecule in dudez decoy dir
            for mol in glob(f"{dudezDirDecoy}/*.mol2"):
                # Find its name
                molName = ".".join(os.path.basename(mol).split(".")[:-1])
                # Create the ligand object
                l = ocl.Ligand(mol, molName)
                # Export its descriptors
                _ = l.to_json(overwrite)

            # For each molecule in extrema decoy dir
            for mol in glob(f"{extremaDirDecoy}/*.mol2"):
                # Find its name
                molName = ".".join(os.path.basename(mol).split(".")[:-1])
                # Create the ligand object
                l = ocl.Ligand(mol, molName, False)
                # Export its descriptors
                _ = l.to_json(overwrite)
                #see https://www.rdkit.org/docs/Cookbook.html

            # For each molecule in goldilocks decoy dir
            for mol in glob(f"{goldilocksDirDecoy}/*.mol2"):
                # Find its name
                molName = ".".join(os.path.basename(mol).split(".")[:-1])
                # Create the ligand object
                l = ocl.Ligand(mol, molName)
                # Export its descriptors
                _ = l.to_json(overwrite)

        elif archive == "pdbbind":
            # Set the input file name path
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
            __run_p2rank(dir, fin)
        else:
            octools.print_info(f"The protein '{dir}' already has its p2rank output generated, skipping its execution.")

        # If overwrite mode is on or there is not the same amount of box files as folders in vinaFiles folder
        if len(glob(f"{dir}/vinaFiles/*")) == boxCount or overwrite:
            # Create the vina inputs from the boxes
            ocvina.generate_vina_files_database(dir, fin)
        else:
            octools.print_info(f"The protein '{dir}' already has its vina file generated, skipping its execution.")

    return
