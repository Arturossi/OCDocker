#!/usr/lib/python3

# Imports
###############################################################################
import os
import time
import shutil
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

from OCDocker.Initialise import *
import OCDocker.Ligand as ocl
import OCDocker.Vina as ocvina
import OCDocker.Receptor as ocr
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
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = 1 if args.output_level >= 3 else 0)
    except Exception as e:
        octools.print_warning(f"The protein '{dir}' had a problem while running p2rank. Retrying to run p2rank. Exception: {e}  ")
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores, debug = False, boxMaxCutoff = p2rank_boxMaxCutoff, pocketCutoff = p2rank_pocketCutoff, verbose = 1 if args.output_level >= 3 else 0)

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

def __thread_prepare(arguments):
    '''
    Prepares the molecule.
    Input:
     arguments [tuple(string, bool, string, string, bool)] - Tuple containing, in this order:
        - [string] The molecule path
        - [bool]   Flag to tell if files should be overwritten
        - [string] The type of the molecule (ligant or receptor)
        - [string] The database name
        - [bool]   Flag to tell if the molecule should be sanitized
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Renaming arguments to what they are (making this just more readable)
        mol = arguments[0]
        overwrite = arguments[1]
        moltype = arguments[2]
        dbName = arguments[3]
        sanitize = arguments[4]
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
                    _ = errors.parse_molecule(f"The molecule '{mol}' could not be parsed!", "error")
                    octools.print_error_log(f"The molecule '{mol}' could not be parsed! .", f"{logdir}/{dbName}_error_Parse.log")
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

def __prepare_parallel(dirList, overwrite, moltype, dbName, desc, sanitize = True):
    '''
    Warper to prepare the parallel jobs, recieves a list of directories, creates the argument list and then pass it to the threads, afterwards waits all threads to finish.
    Input:
     dirList   [string] - List of molecule paths
     overwrite [bool]   - Flag to tell if files should be overwritten
     moltype   [string] - The type of the molecule (ligant or receptor)
     dbName    [string] - The database name (for proper logging)
     desc      [string] - The description
    Return:
      -
    '''
    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for filename in dirList:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((filename, overwrite, moltype, dbName, sanitize))
    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for _ in tqdm(p.imap_unordered(__thread_prepare, arguments), total = len(arguments), desc = desc):
            pass
    # Return
    return None

def __prepare_no_parallel(mol, overwrite, moltype, dbName, sanitize = True):
    '''
    # TODO:
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Find its name and path
    if type(mol) == tuple:
        molPath, molName = os.path.split(mol[0])
    else:
        molPath, molName = os.path.split(mol)
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
                _ = errors.parse_molecule(f"The molecule '{mol}' could not be parsed!", "error")
                octools.print_error_log(f"The molecule '{mol}' could not be parsed! .", f"{logdir}/{dbName}_error_Parse.log")
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

def __thread_prepare_pdbbind(arguments):
    '''
    # TODO:
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Renaming arguments to what they are (making this just more readable)
        dir = arguments[0]
        overwrite = arguments[1]
        archive = arguments[2]
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
        __prepare_no_parallel(fligand, overwrite, "ligand", archive, f"{ptn} PDBbind ligand")
        # For each Receptor
        __prepare_no_parallel((fin, fout), overwrite, "receptor", archive, f"{ptn} PDBbind receptor")
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

def __prepare_parallel_pdbbind(dirList, overwrite, desc):
    '''
    # TODO:
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for dir in dirList:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((dir, overwrite, "pdbbind"))
    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for _ in tqdm(p.imap_unordered(__thread_prepare_pdbbind, arguments), total = len(arguments), desc = desc):
            pass
    # Return
    return None

def __thread_get_parallel(arguments):
    '''
    # TODO:
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Renaming arguments to what they are (making this just more readable)
        dir = arguments[0]
        archive = arguments[1]
        # If is the index directory, ignore
        if dir not in ['index', 'db']:
            return
        # Find which kind of archive it will be
        if archive == "astex":
            chosenArchive = astex_archive
        elif archive == "dudez":
            chosenArchive = dudez_archive
        elif archive == "pdbbind":
            ptn = dir.split(os.path.sep)[-1]
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

def __get_parallel(dirList, chosenArchive, desc):
    '''
    # TODO:
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for dir in dirList:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((dir, chosenArchive))
    # Dict of elements
    databaseDict = dict()
    # Create a Thread pool with the maximum available_cores
    with Pool(args.available_cores) as p:
        # Perform the multi process
        for complexData in tqdm(p.imap_unordered(__thread_get_parallel, arguments), total = len(arguments), desc = desc):
            if complexData:
                databaseDict[complexData[0]] = (complexData[1], complexData[2])
    # Return
    return databaseDict

def __get_no_parallel(dirs, archive):
    '''
    # TODO:
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Dict of elements
    databaseDict = dict()
    for dir in dirs:
        # If is the index directory, ignore
        if dir not in ['index', 'db']:
            return
        # Find which kind of archive it will be
        if archive == "astex":
            chosenArchive = astex_archive
        elif archive == "dudez":
            chosenArchive = dudez_archive
        elif archive == "pdbbind":
            ptn = dir.split(os.path.sep)[-1]
            # Set the input file name path (to generate the box and data about the protein)
            receptorPath = f"{dir}/{ptn}_protein.pdb"
            # Set the ligand file name path (to generate data about the ligand)
            ligandPath = f"{dir}/{ptn}_ligand.mol2"
            # If the complex has all descriptors for protein AND ligand
            if os.path.isfile(f"{dir}/{ptn}_protein_descriptors.json") and os.path.isfile(f"{dir}/{ptn}_ligand_descriptors.json"):
                # Read the receptor and the ligand
                receptor = ocr.Receptor(receptorPath, from_json_descriptors = f"{dir}/{ptn}_protein_descriptors.json", name = f"{ptn}_receptor")
                ligand = ocl.Ligand(ligandPath, from_json_descriptors = f"{dir}/{ptn}_ligand_descriptors.json", name = f"{ptn}_ligand")
            # Add them to the dict using the protein as the key
            databaseDict[ptn] = (receptor, ligand)
    return databaseDict

def __thread_dock_parallel(arguments):
    '''
    # TODO:
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Redirect all prints to tqdm.write
    with octools.redirect_to_tqdm():
        # Renaming arguments to what they are (making this just more readable)
        dir = arguments[0]
        archive = arguments[1]
        dockingAlgorithm = arguments[2]
        overwrite = arguments[3]
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
                                # Parameterizing paths
                                vinaLog = f"{runPath}/vina_{runNumber}.log"
                                vinaOutput = f"{runPath}/vina_{runNumber}.pdbqt"
                                # Get the run number
                                runNumber = runPath.split(os.path.sep)[-1]
                                # Create the vina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                                vina = ocvina.Vina(f"{runPath}/conf_vina.txt", f"{dir}/p2rank/box{runNumber}.pdb", receptor, f"{dir}/{ptn}_protein.pdbqt", ligand, f"{dir}/{ptn}_ligand.pdbqt", vinaLog, vinaOutput, name=f"{ptn}_run_{runNumber}")
                                # Check if the vina object has been correctly created
                                if not vina:
                                    octools.print_error_log(f"Could not generate vina object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                                    return errors.docking_object_not_generated(f"Could not generate vina object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                                # If prepared ligand does not exsits or overwrite flag is true
                                if not os.path.isfile(vina.preparedLigand) or overwrite:
                                    # Run the prepare ligand
                                    _ = vina.run_prepare_ligand()
                                # If prepared receptor does not exists or overwrite flag is true
                                if not os.path.isfile(vina.preparedReceptor) or overwrite:
                                    # Run the prepare receptor
                                    _ = vina.run_prepare_receptor()
                                if overwrite or not os.path.isfile(vinaLog) or not os.path.isfile(vinaOutput):
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
                    # Create the smina dir
                    _ = octools.safe_create_dir(runPath)
                    # If is needed to run (overwrite is set or no output is produced)
                    if overwrite or not os.path.isfile(f"{runPath}/smina.log") or not os.path.isfile(f"{runPath}/smina.pdbqt"):
                        # Read the receptor and the ligand
                        receptor = ocr.Receptor(receptorPath, from_json_descriptors = f"{dir}/{ptn}_protein_descriptors.json", name = f"{ptn}_receptor")
                        ligand = ocl.Ligand(ligandPath, from_json_descriptors = f"{dir}/{ptn}_ligand_descriptors.json", name = f"{ptn}_ligand")
                        # If receptor and ligand are not null
                        if receptor and ligand:
                            # Parameterizing paths
                            sminaLog = f"{runPath}/smina.log"
                            sminaOutput = f"{runPath}/smina.pdbqt"
                            # Create the smina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                            smina = ocsmina.Smina(f"{runPath}/conf_smina.txt", receptor, f"{dir}/{ptn}_protein.pdbqt", ligand, f"{dir}/{ptn}_ligand.pdbqt", sminaLog, sminaOutput, name=f"{ptn}")
                            # Check if the smina object has been correctly created
                            if not smina:
                                octools.print_error_log(f"Could not generate smina object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                                return errors.docking_object_not_generated(f"Could not generate smina object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                            # If prepared ligand does not exsits or overwrite flag is true
                            if not os.path.isfile(smina.preparedLigand) or overwrite:
                                # Run the prepare ligand
                                _ = smina.run_prepare_ligand()
                            # If prepared receptor does not exists or overwrite flag is true
                            if not os.path.isfile(smina.preparedReceptor) or overwrite:
                                # Run the prepare receptor
                                _ = smina.run_prepare_receptor()
                            # If overwrite is true or the output is not generated
                            if overwrite or not os.path.isfile(sminaLog) or not os.path.isfile(sminaOutput):
                                # Run vina
                                smina.run_smina()
                            else:
                                octools.print_warning_log(f"The smina output for '{ptn}' is already generated and you can check it at the '{runPath}/smina.log' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                                octools.print_warning(f"The smina output for '{ptn}' is already generated and you can check it at the '{runPath}/smina.log' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
                        else:
                            octools.print_error_log(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_ERROR.log")
                            return errors.receptor_or_ligand_not_generated(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the '{dockingAlgorithm}' docking software.", level = "error")
                    else:
                        octools.print_warning_log(f"The smina output for '{ptn}' is already generated and you can check it at the '{runPath}/smina.log' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report_WARNING.log")
                        octools.print_warning(f"The smina output for '{ptn}' is already generated and you can check it at the '{runPath}/smina.log' path. Smina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true.")
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

def __run_dock_parallel(dirList, archive, dockingAlgorithm, overwrite, desc):
    '''
    # TODO:
    Input:
     archive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # Arguments to pass to each Thread in the Thread Pool
    arguments = []
    # For each file in the glob
    for dir in dirList:
        # Append a tuple containing the file name and ovewrite flag to the arguments list
        arguments.append((dir, archive, dockingAlgorithm, overwrite))
    # Define the number of used cores, limiting upper (max cores) and lowe bounds (1 core)
    #if dockingAlgorithm == "vina":
    #    cores = int(args.available_cores)/vina_exhaustiveness
    #    cores = int(cores) if cores > 0 else 1
    #else:
    #    cores = args.available_cores
    cores = args.available_cores
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
    with Pool(cores) as p:
        # Perform the multi process
        for _ in tqdm(p.imap_unordered(__thread_dock_parallel, arguments), total = len(arguments), desc = desc):
            pass
    # Return
    return None

def __run_dock_no_parallel(dirs, archive, dockingAlgorithm, overwrite):
    '''
    # TODO:
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
    Return:
      -
    '''
    # For each dir in dirs
    for dir in dirs:
        # If is the index directory, ignore
        if dir not in ['index', 'db']:
            continue
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
                # Read the receptor and the ligand
                receptor = ocr.Receptor(receptorPath, from_json_descriptors = f"{dir}/{ptn}_protein_descriptors.json", name = f"{ptn}_receptor")
                ligand = ocl.Ligand(ligandPath, from_json_descriptors = f"{dir}/{ptn}_ligand_descriptors.json", name = f"{ptn}_ligand")
                # If receptor and ligand are not null
                if receptor and ligand:
                    # Get the folder for each run
                    runPaths = glob(f"{dir}/vinaFiles/*")
                    # For each path in the paths array (will be more than on in case of multiple boxes)
                    for runPath in runPaths:
                        # Get the run number
                        runNumber = runPath.split(os.path.sep)[-1]
                        # If running vina
                        if dockingAlgorithm == "vina":
                            # Create the vina object (the pdbqt files will be in the father directory because it will be used multiple times, let's save some disk space, please)
                            vina = ocvina.Vina(f"{runPath}/conf_vina.txt", f"{dir}/p2rank/box{runNumber}.pdb", receptor, f"{dir}/{ptn}_protein.pdbqt", ligand, f"{dir}/{ptn}_ligand.pdbqt", f"{runPath}/vina_{runNumber}.log", f"{runPath}/vina_{runNumber}.pdbqt", name=f"{ptn}_run_{runNumber}")
                            # Check if the vina object has been correctly created
                            if not vina:
                                octools.print_error_log(f"Could not generate vina object for the protein in dir '{dir}'. Error found while trying to run the {dockingAlgorithm} docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report.log")
                                return errors.docking_object_not_generated(f"Could not generate vina object for the protein in dir '{dir}'. Error found while trying to run the {dockingAlgorithm} docking software.", level = "error")
                            # If prepared ligand does not exsits or overwrite flag is true
                            if not os.path.isfile(vina.preparedLigand) or overwrite:
                                # Run the prepare ligand
                                _ = vina.run_prepare_ligand()
                            # If prepared receptor does not exists or overwrite flag is true
                            if not os.path.isfile(vina.preparedReceptor) or overwrite:
                                # Run the prepare receptor
                                _ = vina.run_prepare_receptor()
                            # If the output does not exist or overwrite flag is true
                            if overwrite or (not os.path.isfile(f"{runPath}/vina_{runNumber}.log") and os.path.isfile(f"{runPath}/vina_{runNumber}.pdbqt")):
                                # Run vina
                                vina.run_vina()
                            else:
                                octools.print_warning(f"The vina output for {ptn} run {runNumber} is already generated and you can check it at the '{runPath}/vina_{runNumber}.log' path. Vina execution will be avoided to save processing time. If you want to generate these files, set the overwrite flag to true")
                else:
                    octools.print_error_log(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the {dockingAlgorithm} docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report.log")
                    _ = errors.receptor_or_ligand_not_generated(f"Could not generate receptor or ligand object for the protein in dir '{dir}'. Error found while trying to run the {dockingAlgorithm} docking software.", level = "error")
                    continue
            else:
                octools.print_error_log(f"There is no ligand or receptor descriptor for the protein in dir '{dir}'. Error found while trying to run the {dockingAlgorithm} docking software.", f"{logdir}/PDBbind_{dockingAlgorithm}_run_report.log")
                _ = errors.receptor_or_ligand_descriptor_does_not_exist(f"There is no ligand or receptor descriptor for the protein in dir '{dir}'. Error found while trying to run the {dockingAlgorithm} docking software.", level = "error")
                continue
    return None

## Public ##
def verify_integrity(chosenArchive):
    '''
    Verifies the integrity of the desired database
    Input:
     chosenArchive [string] - Which archive will be processed. [dudez, pdbbind, astex]
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
        if not os.path.isdir(f"{logdir}/pdbbind_past"):
            octools.safe_create_dir(f"{logdir}/pdbbind_past")
        os.rename(f"{logdir}/PDBbind_integrity_report.log", f"{logdir}/pdbbind_past/PDBbind_integrity_report_{time.strftime('%d%m%Y-%H%M%S')}.log")

    # Redirect output to tqdm.write
    with octools.redirect_to_tqdm():
        # For each directory in the database folder
        for dir in tqdm(iterable=dirs, total=lenDirs):
            # If is the index path
            if os.path.basename(dir) not in ['index', 'db']:
                # Skip it
                continue

            # Parameterizing paths
            p2rankDir = f"{dir}/p2rank"
            vinaDir = f"{dir}/vinaFiles"

            # Find protein name
            ptn = dir.split(os.path.sep)[-1]

            # Set the input file name path and set the input file name path
            if archive == "astex":
                fin = f"{dir}/protein.pdb"
            elif archive == "dudez":
                fin = f"{dir}/rec.crg.pdb"
            elif archive == "pdbbind":
                fin = f"{dir}/{dir.split(os.path.sep)[-1]}_protein.pdb"

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

                print(dir)

                # Run p2rank
                __run_p2rank(dir, fin)

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

            # If is the pdbbind files
            if archive == "pdbbind":
                # If there is no descriptor file for the ligand or its size is 0
                if not os.path.isfile(f"{dir}/{ptn}_ligand_descriptors.json") or os.path.getsize(f"{dir}/{ptn}_ligand_descriptors.json") == 0:
                    # Generate it
                    __prepare_no_parallel(f"{dir}/{ptn}_ligand.mol2", False, "ligand", archive, sanitize = True)
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
                    __prepare_no_parallel(f"{dir}/{ptn}_protein.pdb", False, "receptor", archive, sanitize = True)
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

    # Get all dirs paths in the database
    dirs = glob(f"{chosenArchive}/*")

    # Check if its pdbbind database and multiprocess flag is set as true
    if archive == "pdbbind" and args.multiprocess:
        # Let's go parallel (it's too slow without it)
        # NOTE: This is safe because pdbbind database is 1 ligand + 1 receptor.
        __prepare_parallel_pdbbind(dirs, overwrite, "PDBbind proteins")

    else:
        # For each directory in the database folder
        for dir in dirs:
            # If is the index path
            if os.path.basename(dir) == 'index' or os.path.basename(dir) == 'db':
                # Skip it
                continue
            # Find the protein name
            ptn = dir.split(os.path.sep)[-1]

            # Set the input file name path
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

                # For each molecule in dudez ligand dir
                mols = glob(f"{dudezDirLigand}/*.mol2")
                __prepare_parallel(mols, overwrite, moltype, f"{ptn} DUDEz ligand")
                # For each molecule in dudez decoy dir
                __prepare_parallel(glob(f"{dudezDirDecoy}/*.mol2"), overwrite, moltype, f"{ptn} DUDEz decoy")
                # For each molecule in extrema decoy dir
                __prepare_parallel(glob(f"{extremaDirDecoy}/*.mol2"), overwrite, moltype, f"{ptn} extrema decoy")
                # For each molecule in goldilocks decoy dir
                __prepare_parallel(glob(f"{goldilocksDirDecoy}/*.mol2"), overwrite, moltype, f"{ptn} goldilocks decoy")
            elif archive == "pdbbind":
                # Set the input file name path (to generate the box and data about the protein)
                fin = f"{dir}/{ptn}_protein.pdb"
                fout = f"{dir}/{ptn}_protein.mol2"
                # Convert the .pdb to .mol2 (for dock6 use)
                _ = octools.convertMols(fin, fout)
                # Set the ligand file name path (to generate data about the ligand)
                fligand = f"{dir}/{ptn}_ligand.mol2"
                # For each ligand
                __prepare_parallel([fligand], overwrite, "ligand", archive, f"{ptn} PDBbind ligand")
                # For each Receptor
                __prepare_parallel([(fin, fout)], overwrite, "receptor", archive, f"{ptn} PDBbind receptor")

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

def run_dock(archive, dockingAlgorithm, overwrite = False):
    '''
    Parse the database into a SINGLE serializable object. (Not so good)
    Input:
     archive [string]                         - Which archive will be processed. [dudez, pdbbind, astex]
     dockingAlgorithm [string]                - Which docking software will be run. [vina, smina, plants]
     overwrite [bool]          DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      [dict of tuples]
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
     overwrite [bool]   DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
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
