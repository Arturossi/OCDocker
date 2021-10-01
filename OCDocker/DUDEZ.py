#!/usr/lib/python3

# Imports
###############################################################################
import os

from OCDocker.Initialise import *
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
Sets of classes and functions that are used to process the DUDE-Z dataset.

They are imported as:

import OCDocker.DUDEZ as ocdudez
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private
def __get_vina_data_from_box(file):
    '''
    Get data to generate vina conf file from box file.
    Input:
     path [string] - Input path.
    Return:
     [string] - The center in the x axis.
     [string] - The center in the y axis.
     [string] - The center in the z axis.
     [string] - The size for the x axis.
     [string] - The size for the y axis.
     [string] - The size for the z axis.
    '''
    # Initialise variables with 0
    center_x = 0
    center_y = 0
    center_z = 0
    size_x = 0
    size_y = 0
    size_z = 0

    # Open file in read mode
    with open(file, "r") as fp:
        # For each line in file
        for line in fp:
            # If is a remark line with center data
            if line.startswith("REMARK") and "CENTER" in line:
                # Split each line in file using spaces
                splited = line.split()

                # Grab the last 3 informations which are x,y,z centers
                center_x = splited[-3].strip()
                center_y = splited[-2].strip()
                center_z = splited[-1].strip()
            # If is a remark line with dimensions data
            elif line.startswith("REMARK") and "DIMENSIONS" in line:
                # Split each line in file using spaces
                splited = line.split()

                # Grab the last 3 informations which are x,y,z sizes
                size_x = splited[-3].strip()
                size_y = splited[-2].strip()
                size_z = splited[-1].strip()

                # Since we have all useful data, break to optimize time/CPU
                break

    return center_x, center_y, center_z, size_x, size_y, size_z

def __generate_vina_conf_file_dudez(receptorPath, box, boxFolder):
    '''
    Generate the conf file required by vina to run.
    Input:
     receptorPath [string] - Receptor folder path.
     box          [string] - Box path.
     boxFolder    [string] - The folder to put files concerning current box.
    Return:
      -
    '''
    # Parameterize conf files
    vinaConf = f"{boxFolder}/{conf.txt}"

    ## Gather required info

    # The receptor name
    receptor =f"{receptorPath}/rec.crg.pdb"

    # TODO:Prepare the receptor

    # x,y,z center and size to create the conf.txt file
    center_x, center_y, center_z, size_x, size_y, size_z = __get_vina_data_from_box(box)

    with open(vinaConf, "w") as conf_file:
        conf_file.write(f"receptor = {receptor}\n\n")
        conf_file.write(f"center_x = {center_x}\n")
        conf_file.write(f"center_y = {center_y}\n")
        conf_file.write(f"center_z = {center_z}\n\n")
        conf_file.write(f"size_x = {size_x}\n")
        conf_file.write(f"size_y = {size_y}\n")
        conf_file.write(f"size_z = {size_z}\n\n")
        conf_file.write(f"energy_range = {energy_range}\n")
        conf_file.write(f"exhaustiveness = {exhaustiveness}\n")
        conf_file.write(f"num_modes = {num_modes}\n")

    return

## Public
def generate_vina_files(path):
    '''
    Generate all vina required files for provided protein.
    Input:
     path [string] - Input path.
    Return:
      -
    '''
    # Parameterize the vina and p2rank paths
    vinaPath = f"{path}/vinaFiles"
    prankPath = f"{path}/p2rank"

    # Create the vina folder inside protein's directory
    _ = octools.safe_create_dir(vinaPath)

    # Find all boxes
    boxes = glob(f"{prankPath}/box*")

    # TODO:Prepare the ligands

    # For each box
    for box in boxes:
        # Get box name
        boxName = os.path.basename(box)

        # Get box id
        boxId = boxName.split(".")[0].replace("box","")

        # Parameterize the box folder
        boxFolder = f"{vinaPath}/{boxId}"

        # Create vina execution folder
        _ = octools.safe_create_dir(boxFolder)

        __generate_vina_conf_file_dudez(path, box, boxFolder)

    return

def runprank(args):
    '''
    Generate all vina required files for provided protein.
    Input:
     path [string] - Input path.
    Return:
      -
    '''
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

    # Generate boxes for all receptors
    print("Generating information regarding possible ligand site.")

    # Get all dirs paths in the DUDEZ database
    dirs = glob(f"{dudez_archive}/*")

    # For each directory in the database folder
    for d in tqdm(iterable=dirs, total=len(dirs)):
        # Set the input file name path
        fin = f"{d}/rec.crg.pdb"

        # Find the protein name
        ptn = d.split("/")[-1]

        # Set the output path
        fout = f"{d}/p2rank"

        # Create the p2rank output dir
        _ = octools.safe_create_dir(fout)

        # Run p2rank
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores,
                           debug = False, boxMaxCutoff = 0.5, pocketCutoff = 0.1, verbose = args.verbosity)

        # Create the vina inputs from the boxes
        generate_vina_files(d)

        return
