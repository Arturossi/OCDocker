#!/usr/lib/python3

# Imports
###############################################################################
import inspect
import shutil
import os

import textwrap as tw
from pprint import pprint
from glob import glob

from OCDocker.Initialise import *

args.output_level = 0

import OCDocker.Toolbox as octools

import OCDocker.Database as ocdb
import OCDocker.baseDB as ocbdb
import OCDocker.DUDEz as ocdudez
import OCDocker.PDBbind as ocpdbbind
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.ExternalTools.runprank as runprank
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust
import OCDocker.Rescoring.ODDT as ocoddt
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
def print_args() -> None:
    '''Prints the current args variable

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    print("args:")
    # Iterate over vars(args) dict
    for key, value in vars(args).items():
        print(f"{key}:\t\t{value}")
    print("\n")

    return None

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

message = tw.dedent("""\033[1;93m
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\033[1;96m
                                CONSOLE MODE\033[1;93m
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\033[1;0m
            This mode is intended to make debug easier and allow the
        user to interact with the OCDocker pipeline step by step.

        \033[1;91mWARNING\033[1;0m: This mode is still experimental, some unexpected
        behaviour might occur while using it.

        To check the args variable use print_args() function.\033[1;93m
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
\033[1;0m""")

print(message)

args.cpu_cores = 18
args.available_cores = args.cpu_cores - 1
args.multiprocess = 1
args.generate_report = False
args.zip_output = False
args.update = False

'''
basePath = f"{os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))}/test_files"
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
# Get receptor descriptors
receptorDescriptors = receptorTest.get_descriptors()
# Get ligand descriptors
ligandDescriptors = ligandTest.get_descriptors()


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
# Run the rescoring with vina
vinaTest.run_rescore(f"{baseLigPath}/{lig}/vinaFiles", skipDefaultScoring = True)
# Get Docking results
dockingResult = vinaTest.read_log()
# Get Rescoring results (skip the default scoring function)
rescoringResult = vinaTest.read_rescore_logs(f"{baseLigPath}/{lig}/vinaFiles")


###############
##   Smina    #
###############

# Create the object
sminaTest = ocsmina.Smina(f"{baseLigPath}/{lig}/sminaFiles/conf_smina.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.pdbqt", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.pdbqt", f"{baseLigPath}/{lig}/sminaFiles/{lig}.log", f"{baseLigPath}/{lig}/sminaFiles/{lig}.pdbqt", name=f"Smina {ptn}-{lig}")

## If you will run smina only for rescoring (no docking), skip this block! (ideally we can skip if the docking is already done using the same kind of search algorithm, like vina-smina-gnina)
###########################################################################

# Prepare the receptor (check if the receptor is already prepared before to avoid unnecessary work)
if not os.path.isfile(f"{baseProtPath}/prepared_receptor.pdbqt"):
    sminaTest.run_prepare_receptor()
# Prepare the ligand (check if the ligand is already prepared before to avoid unnecessary work)
if not os.path.isfile(f"{baseLigPath}/{lig}/prepared_ligand.pdbqt"):
    sminaTest.run_prepare_ligand()
# Run the docking
sminaTest.run_docking()
# Run the rescoring with smina
sminaTest.run_rescore(f"{baseLigPath}/{lig}/sminaFiles", skipDefaultScoring = True)
# Get Docking results
dockingResult = sminaTest.read_log()

## If you will run smina only for rescoring (no docking), run this block!
##########################################################################

# Get the docking results from the vinaTest (example) object
dockingPoses = vinaTest.get_docked_poses()

# Process each scoring function
for sf in smina_scoring_functions:
    ocsmina.run_rescore(sminaTest.config, dockingPoses, f"{baseLigPath}/{lig}/sminaFiles", sf, splitLigand = False)

## Now you can get the rescoring results
###########################################

# Get Rescoring results
rescoringResult = sminaTest.read_rescore_logs(f"{baseLigPath}/{lig}/sminaFiles")


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

# Get Docking results
dockingResult = plantsTest.read_log(onlyBest = False)

# Get Docking poses
dockingPoses = plantsTest.get_docked_poses()

# Write the pose_list file for rescoring
pose_list = plantsTest.write_pose_list(dockingPoses)

# Run the rescoring (will create the config file and the output folder)
plantsTest.run_rescore(pose_list, logFile = "", overwrite = False)

# Get Rescoring results
rescoringResult = plantsTest.read_rescore_logs(onlyBest = False)


#############
##   ODDT   #
#############

## WARNING: The ODDT is used only for rescoring, so it is REQUIRED that you run at least one docking before rescoring with ODDT. The following example will use vina as the docking algorithm.

# Run ODDT and get the result as a dataframe
df = ocoddt.run_oddt(vinaTest.preparedReceptor, vinaTest.get_docked_poses(), vinaTest.inputLigand.name, f"{vinaTest.get_input_ligand_path()}/oddt")

# If you want a dict, you can convert with this function
dt = ocoddt.df_to_dict(df)



###############
## Clustering #
###############

# Get the docked poses for vina and plants
vinaPoses = vinaTest.get_docked_poses()
plantsPoses = plantsTest.get_docked_poses()

# Make them one single list
poses_list = vinaPoses + plantsPoses

# Get the rmsd matrix from the poses list
rmsdMatrix = ocmolproc.get_rmsd_matrix(poses_list)

# Get the medoids (The plot is just for visualization, it is not required)
medoids = ocrmsdclust.get_medoids(rmsdMatrix, algorithm = 'agglomerativeClustering', outputPlot = f"{basePath}/medoids.png")

# Find which medoid has the lowest energy
# TODO: implement

##############
##   Gnina   #
##############

## TODO: Fix the entire Gnina

# Gnina
gninaTest = ocgnina.Gnina(f"{baseLigPath}/{lig}/gninaFiles/conf_gnina.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.pdbqt", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.pdbqt", f"{baseLigPath}/{lig}/gninaFiles/{lig}.log", f"{baseLigPath}/{lig}/gninaFiles/{lig}.pdbqt", name=f"Gnina {ptn}-{lig}")
'''