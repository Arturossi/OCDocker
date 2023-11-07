#!/usr/bin/env python3

# Imports
###############################################################################
import inspect
import shutil
import os

import pandas as pd

from glob import glob

os.environ['OCDOCKER_CONFIG'] = 'OCDocker.cfg'

from OCDocker.Initialise import *

output_level = ocerror.ReportLevel.NONE

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust
import OCDocker.Rescoring.ODDT as ocoddt
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc

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
This script is used for fast import of all funcionalities in the OCDocker suite
making easier to debug and possibly allowing a future OCDocker console to enable
the user to perform the steps step by step.
'''

# Classes
###############################################################################


# Functions
###############################################################################
def clean_test_files(baseProtPath, baseLigPath, baseDecPath, baseCanPath) -> None:
    '''Rests the test_files folder to its original state

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    # Remove all files in the baseProtPath except the receptor.pdb
    for f in glob(f"{baseProtPath}/*"):
        # If the file is a file and not the receptor.pdb
        if os.path.isfile(f) and not f.endswith(f"{baseProtPath}/receptor.pdb"):
            os.remove(f)

    # For each ligand folder
    for ligFolder in [baseLigPath, baseDecPath, baseCanPath]:
        # Remove all the files inside all ligand folders except for the ligand.smi or ligand.mol2
        for f in glob(f"{ligFolder}/*/*"):
            # If the file is a file and not the ligand.smi
            if os.path.isfile(f) and not f.endswith("ligand.smi"):
                os.remove(f)
            # If the file is a folder and not the boxes folder
            elif os.path.isdir(f) and not f.endswith("boxes"):
                shutil.rmtree(f)
    return None

cpu_cores = 3
available_cores = cpu_cores - 1
multiprocess = 1
update = False

basePath = f"{os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))}/test_files" # type: ignore
ptn = "test_ptn1"
lig = "ligand"
baseProtPath = f"{basePath}/{ptn}"
baseLigPath = f"{baseProtPath}/compounds/ligands"
baseDecPath = f"{baseProtPath}/compounds/decoys"
baseCanPath = f"{baseProtPath}/compounds/candidates"

# If you want to clean the last run of the test files
clean_test_files(baseProtPath, baseLigPath, baseDecPath, baseCanPath)


########################
## Ligand and Receptor #
########################

# Create the Receptor and Ligand objects
receptorTest = ocr.Receptor(f"{baseProtPath}/receptor.pdb", relativeASAcutoff=0.7, name=f"{ptn}")
ligandTest = ocl.Ligand(f"{baseLigPath}/{lig}/ligand.smi", name=f"{lig}")


##############
##   Vina    #
##############

# Create the object
vinaTest = ocvina.Vina(f"{baseLigPath}/{lig}/vinaFiles/conf_vina.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.pdbqt", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.pdbqt", f"{baseLigPath}/{lig}/vinaFiles/{lig}.log", f"{baseLigPath}/{lig}/vinaFiles/{lig}.pdbqt", name=f"Vina {ptn}-{lig}")

# Prepare the receptor
vinaTest.run_prepare_receptor()

# Prepare the ligand
vinaTest.run_prepare_ligand()

# Run the docking
vinaTest.run_docking()

# Split the docking results into multiple files
vinaTest.split_poses(f"{baseLigPath}/{lig}/vinaFiles", logFile = "")

# Get the docking poses
vinadockingPoses = vinaTest.get_docked_poses()


###############
##   PLANTS   #
###############

# Create the object
plantsTest = ocplants.PLANTS(f"{baseLigPath}/{lig}/plantsFiles/conf_plants.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.mol2", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.mol2", f"{baseLigPath}/{lig}/plantsFiles/{lig}.log", f"{baseLigPath}/{lig}/plantsFiles", name=f"PLANTS {ptn}-{lig}")

# Prepare the receptor
plantsTest.run_prepare_receptor()

# Prepare the ligand
plantsTest.run_prepare_ligand()

# Run the docking
plantsTest.run_docking()

# Get the docking poses
plantsdockingPoses = plantsTest.get_docked_poses()


###############
##  Cluster   #
###############

# Concatenate the poses lists from vina and plants into a single list
poses_list = vinadockingPoses + plantsdockingPoses

# Get the rmsd matrix from the poses list
mols_mat = ocmolproc.get_rmsd_matrix(poses_list)

# Get the rmsd matrix from the poses list
rmsdMatrix = ocmolproc.get_rmsd_matrix(poses_list)

# Get the clusters
clusters = ocrmsdclust.cluster_rmsd(rmsdMatrix, algorithm = 'agglomerativeClustering', outputPlot = f"{basePath}/medoids.png")

# Get the medoids (The plot is just for visualization, it is not required)
medoids = ocrmsdclust.get_medoids(rmsdMatrix, clusters, onlyBiggest = True) # type: ignore


###############
##  Rescore   #
###############

# Create the processed_Medoids dict
processedMedoids = {"plants" : [], "vina" : [], "smina" : []}

## Prepare the files for rescoring
# For each file in the medoids list
for medoid in medoids:
    # Get the directory of the file
    directory = os.path.dirname(medoid)
    # If the file is a vina like file (.pdbqt)
    if medoid.endswith(".pdbqt"):
        ## Prepare it for PLANTS
        # Set the output and prepared output file path
        outfile = f"{directory}/{os.path.basename(medoid).replace('.pdbqt', '.mol2')}"
        preparedOutfile = f"{directory}/{os.path.basename(medoid).replace('.pdbqt', '_prepared.mol2')}"
        # Convert the file to mol2
        occonversion.convertMols(medoid, outfile)
        # Prepare the mol2 file for PLANTS
        ocplants.run_prepare_ligand(outfile, preparedOutfile)
        # Append the prepared file to the processedMedoids dict
        processedMedoids["vina"].append(medoid)
        processedMedoids["smina"].append(medoid)
        processedMedoids["plants"].append(preparedOutfile)
    # If the file is a plants like file (.mol2)
    elif medoid.endswith(".mol2"):
        ## Prepare it for Vina like programs
        # Set the prepared output file path
        preparedOutfile = f"{directory}/{os.path.basename(medoid).replace('.mol2', '_prepared.pdbqt')}"
        preparedOutfileMol2 = f"{directory}/{os.path.basename(medoid).replace('.mol2', '_prepared.mol2')}"
        # Prepare the pdbqt file for Vina
        ocvina.run_prepare_ligand(medoid, preparedOutfile)
        # Append the prepared file to the processedMedoids dict
        processedMedoids["vina"].append(preparedOutfile)
        processedMedoids["smina"].append(preparedOutfile)
        processedMedoids["plants"].append(preparedOutfileMol2)

# Dictionary with the medoids and its docking method (to be correctly parsed by the next function)
medoidsDict = {}

## Run rescoring
# Create smina object
sminaTest = ocsmina.Smina(f"{baseLigPath}/{lig}/sminaFiles/conf_smina.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.pdbqt", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.pdbqt", f"{baseLigPath}/{lig}/sminaFiles/{lig}.log", f"{baseLigPath}/{lig}/sminaFiles/{lig}.pdbqt", name=f"Smina {ptn}-{lig}")

# For vina
for sf in vina_scoring_functions:
    # Run the rescoring
    ocvina.run_rescore(vinaTest.config, processedMedoids["vina"], f"{baseLigPath}/{lig}/vinaFiles", sf, splitLigand = False)

# For smina
for sf in smina_scoring_functions:
    # Run the rescoring
    ocsmina.run_rescore(sminaTest.config, processedMedoids["smina"], f"{baseLigPath}/{lig}/sminaFiles", sf, splitLigand = False)

# For PLANTS
# Set the pose list path
poseListPath = f"{plantsTest.outputPlants}/pose_list.txt"

# Write the pose_list file for rescoring
pose_list = ocplants.write_pose_list(processedMedoids["plants"], poseListPath)

for sf in plants_scoring_functions:
    # Set the output path
    outPath = f"{plantsTest.outputPlants}/run_{sf}"
    # Set the config file
    confFile = f"{plantsTest.outputPlants}/{plantsTest.inputLigand.name}_rescoring_{sf}.txt"
    # Run the rescoring
    ocplants.run_rescore(confFile, pose_list, outPath, plantsTest.preparedReceptor, sf, logFile = "", overwrite = False) # type: ignore

## For ODDT
## WARNING: The ODDT is used only for rescoring, so it is REQUIRED that you run at least one docking before rescoring with ODDT. The following example will use vina as the docking algorithm.

# Run ODDT and get the result as a dataframe
df = ocoddt.run_oddt(vinaTest.preparedReceptor, medoids, vinaTest.inputLigand.name, f"{vinaTest.get_input_ligand_path()}/oddt") # type: ignore

## Get the rescoring results
# Vina
vinaRescoringResult = ocvina.read_rescore_logs(ocvina.get_rescore_log_paths(f"{baseLigPath}/{lig}/vinaFiles"))

# Smina
sminaRescoringResult = ocsmina.read_rescore_logs(ocsmina.get_rescore_log_paths((f"{baseLigPath}/{lig}/sminaFiles")))

# PLANTS
plantsRescoringResult = {}

# For each scoring function
for sf in plants_scoring_functions:
    # Read the rescoring results and save it in the dictionary
    plantsRescoringResult[sf] = ocplants.read_rescore_logs(f"{plantsTest.outputPlants}/run_{sf}")

# ODDT
# If df not exists
if 'df' not in locals():
    # Get the ODDT rescoring results
    df = pd.read_csv(f"{vinaTest.get_input_ligand_path()}/oddt/{vinaTest.inputLigand.name,}.csv")

# Convert the dataframe to a dictionary
ODDTRescoringResult = ocoddt.df_to_dict(df) # type: ignore

# Make a dictionary with all the rescoring results
rescoringResults = {'vina': vinaRescoringResult, 'smina': sminaRescoringResult, 'plants': plantsRescoringResult, 'oddt': ODDTRescoringResult}
