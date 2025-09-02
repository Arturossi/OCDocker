#!/usr/bin/env python3

# Imports
###############################################################################
import inspect
import shutil
import os

import textwrap as tw
from pprint import pprint
from glob import glob

# The environment variable OCDOCKER_CONFIG must be set to the OCDocker.cfg file before importing OCDocker
cfg_path = os.environ.get('OCDOCKER_CONFIG') or 'OCDocker.cfg'

from OCDocker.Initialise import *

output_level = ocerror.ReportLevel.NONE

import OCDocker.Toolbox as octools

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Future.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust
import OCDocker.Rescoring.ODDT as ocoddt
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Licensed under the Apache License, Version 2.0 (January 2004)
See: http://www.apache.org/licenses/LICENSE-2.0

Commercial use requires a separate license.  
Contact: Artur Duque Rossi - arturossi10@gmail.com
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
def print_args(program: str = "") -> None:
    '''Print environment variables and optionally program-specific settings.

    Console usage examples:
      - print_args()                 # environment overview
      - print_args('paths')          # relevant paths and binaries
      - print_args('db')             # database connections
      - print_args('vina')           # Vina parameters
      - print_args('smina')          # Smina parameters
      - print_args('plants')         # PLANTS parameters
      - print_args('gnina')          # Gnina parameters (if configured)
      - print_args('oddt')           # ODDT parameters
      - print_args('all')            # print all sections
    '''

    def _g(name, default='-'):
        return globals().get(name, default)

    def _p(label, value):
        try:
            print(f"{label:<28}: {value}")
        except Exception:
            print(f"{label:<28}: <unprintable>")

    prog = (program or "").strip().lower()
    show_all = prog in ("all", "*")

    # Overview
    if not prog or show_all:
        print("\n=== OCDocker Runtime Arguments ===")
        _p("config_file", _g('config_file'))
        _p("multiprocess", _g('multiprocess'))
        _p("update", _g('update'))
        ol = _g('output_level')
        try:
            ol_disp = ol.name if hasattr(ol, 'name') else ol
        except Exception:
            ol_disp = ol
        _p("output_level", ol_disp)
        _p("overwrite", _g('overwrite'))

    if prog in ("paths",) or show_all:
        print("\n=== Key Paths ===")
        _p("ocdb_path", _g('ocdb_path'))
        _p("pca_path", _g('pca_path'))
        _p("logdir", _g('logdir'))
        _p("oddt_models_dir", _g('oddt_models_dir'))

        print("\n=== Docking Binaries ===")
        _p("vina", _g('vina'))
        _p("smina", _g('smina'))
        _p("plants", _g('plants'))
        _p("gnina", _g('gnina'))
        _p("obabel", _g('obabel'))
        _p("pythonsh", _g('pythonsh'))
        _p("prepare_ligand", _g('prepare_ligand'))
        _p("prepare_receptor", _g('prepare_receptor'))

    if prog in ("db",) or show_all:
        print("\n=== Database URLs ===")
        _p("db_url", _g('db_url'))
        _p("optdb_url", _g('optdb_url'))

    if prog in ("vina",) or show_all:
        print("\n=== Vina Parameters ===")
        _p("vina_scoring", _g('vina_scoring'))
        _p("vina_scoring_functions", _g('vina_scoring_functions'))
        _p("vina_num_modes", _g('vina_num_modes'))
        _p("vina_energy_range", _g('vina_energy_range'))
        _p("vina_exhaustiveness", _g('vina_exhaustiveness'))

    if prog in ("smina",) or show_all:
        print("\n=== Smina Parameters ===")
        for n in (
            'smina_scoring','smina_scoring_functions','smina_num_modes','smina_energy_range',
            'smina_exhaustiveness','smina_custom_scoring','smina_custom_atoms','smina_local_only',
            'smina_minimize','smina_randomize_only','smina_minimize_iters','smina_accurate_line',
            'smina_minimize_early_term','smina_approximation','smina_factor','smina_force_cap',
            'smina_user_grid','smina_user_grid_lambda'
        ):
            _p(n, _g(n))

    if prog in ("plants",) or show_all:
        print("\n=== PLANTS Parameters ===")
        for n in (
            'plants_cluster_structures','plants_cluster_rmsd','plants_search_speed',
            'plants_scoring','plants_scoring_functions'
        ):
            _p(n, _g(n))

    if prog in ("gnina",) or show_all:
        print("\n=== Gnina Parameters ===")
        for n in (
            'gnina_exhaustiveness','gnina_num_modes','gnina_scoring','gnina_custom_scoring_file',
            'gnina_custom_atoms','gnina_local_only','gnina_minimize','gnina_randomize_only',
            'gnina_num_mc_steps','gnina_max_mc_steps','gnina_num_mc_saved','gnina_minimize_iters',
            'gnina_simple_ascent','gnina_accurate_line','gnina_minimize_early_term','gnina_approximation',
            'gnina_factor','gnina_force_cap','gnina_user_grid','gnina_user_grid_lambda','gnina_no_gpu'
        ):
            _p(n, _g(n))

    if prog in ("oddt",) or show_all:
        print("\n=== ODDT Parameters ===")
        for n in ('oddt_program','oddt_seed','oddt_chunk_size','oddt_scoring_functions'):
            _p(n, _g(n))

    if not prog and not show_all:
        print("")
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

if __name__ == "__main__":
    # Set the variables based on args
    set_argparse()
    pass
else:
    cpu_cores = 18
    available_cores = cpu_cores - 1
    multiprocess = True
    update = False

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
# Split the docking results into multiple files
vinaTest.split_poses(f"{baseLigPath}/{lig}/vinaFiles", logFile = "")
# Run the rescoring with vina
vinaTest.run_rescore(f"{baseLigPath}/{lig}/vinaFiles", skipDefaultScoring = True)
# Get Docking results
vinaDockingResult = vinaTest.read_log()
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
sminaDockingResult = sminaTest.read_log()

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
plantsDockingResult = plantsTest.read_log(onlyBest = False)

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

# TODO: Review the ODDT implementation (seems to not be working properly)
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

# Get the clusters
clusters = ocrmsdclust.cluster_rmsd(rmsdMatrix, algorithm = 'agglomerativeClustering', outputPlot = f"{basePath}/medoids.png")

# Get the medoids (The plot is just for visualization, it is not required)
medoids = ocrmsdclust.get_medoids(rmsdMatrix, clusters, onlyBiggest = True)

# Dictionary with the medoids and its docking method (to be correctly parsed by the next function)
medoidsDict = {}

## Find which medoid has the lowest energy
# For each medoid
for medoid in medoids:
    # Check if it is contained in vinaPoses list
    if medoid in vinaPoses:
        # Add it to the medoidsDict as a list with vina as the key
        medoidsDict[medoid] = vinaDockingResult[ocvina.get_pose_index_from_file_path(medoid)]
    # Check if it is contained in plantsPoses list
    elif medoid in plantsPoses:
        # Add it to the medoidsDict as a list with plants as the key
        medoidsDict[medoid] = plantsDockingResult[ocplants.get_pose_index_from_file_path(medoid)]

##############
##   Gnina   #
##############

# Gnina
gninaTest = ocgnina.Gnina(f"{baseLigPath}/{lig}/gninaFiles/conf_gnina.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.pdbqt", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.pdbqt", f"{baseLigPath}/{lig}/gninaFiles/{lig}.log", f"{baseLigPath}/{lig}/gninaFiles/{lig}.pdbqt", name=f"Gnina {ptn}-{lig}")
'''
