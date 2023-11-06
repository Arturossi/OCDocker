#!/usr/lib/python3

# Description
###############################################################################
'''
Argparsing for OCDocker in CLI mode. TODO: finish this

They are imported as:

from OCDocker.CLI import *
'''

# Imports
###############################################################################
import argparse
import multiprocessing
import os
import shutil

from OCDocker.Initialise import *
import OCDocker.Error as ocerror


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

# Parse command line arguments
###############################################################################
def argument_parsing(noArgs: bool = False) -> argparse.Namespace:
    '''Get data to generate vina conf file from box file.
    
    Parameters
    ----------
    noArgs : bool
        If True, no arguments will be parsed.

            Returns
    -------
    argparse.Namespace
        Namespace object containing the arguments.
    '''
    
    # Create the parser
    parser = argparse.ArgumentParser(prog="OCDocker",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    
    # Add the arguments
    parser.add_argument("--version",
                        action="version",
                        default=False,
                        version=f"%(prog)s {ocVersion}")

    parser.add_argument("--multiprocess",
                        dest="multiprocess",
                        action="store_true",
                        default=True,
                        help="Defines whether python multiprocessing should be enabled for compatible lenghty tasks")

    parser.add_argument("-u", "--update-databases",
                        dest="update",
                        action="store_true",
                        default=False,
                        help="Updates databases")

    parser.add_argument("--conf",
                        dest="config_file",
                        type=str,
                        metavar="",
                        help="Configuration file containing external executable paths")

    parser.add_argument("--output-level",
                        dest="output_level",
                        type=int,
                        default=1,
                        metavar="",
                        help="Define the log level:\n\t0: Critical\n\t1: Warning (default)\n\t2: Info\n\t3: Verbose mode\n\t4: Debug")
    # If noArgs is true
    if noArgs:
        # Return the parser
        return parser.parse_args([])
    # Return the parser
    return parser.parse_args()


# Create error class object (making all errors standard)
errors = ocerror.Error(args.output_level)

# Initialise
###############################################################################
print(description)

# Retrieve the paths from provided configuration file
if (not args.config_file or not os.path.isfile(args.config_file)) and not os.path.isfile("OCDocker.cfg"):
    print("OCDocker configuration file has not been found in the provided path")
    create_config = input("Do you wish to create it? (y/n) ")
    if create_config.lower() in ["y", "ye", "yes"]:
        create_ocdocker_conf()
        quit()
    else:
        print("\n\nNo positive confirmation, please provide a valid configuration file.\n")
        quit()

elif not args.config_file and os.path.isfile("OCDocker.cfg"):
    config_file = "OCDocker.cfg"

elif args.config_file:
    assert os.path.isfile(args.config_file), f"{clrs['r']}\n\n Not able to find configuration file.\n\n Does \"{args.config_file}\" exist?{clrs['n']}"
    config_file = args.config_file

# Set the ocdb path as an empty string
ocdb_path = ""

# Read the conf file and assign its data to its variables (The order matters here, if you follow the same order which is in the conf file less computation power will be needed! It is not much, but it is something.)
for line in open(config_file, 'r'): # type: ignore
    if line.startswith("ocdb ="):
        ocdb_path = line.split("=")[1].strip()
    elif line.startswith("pdbbind_KdKi_order ="):
        pdbbind_KdKi_order = line.split("=")[1].strip()
    elif line.startswith("pythonsh ="):
        pythonsh = line.split("=")[1].strip()
    elif line.startswith("prepare_ligand ="):
        prepare_ligand = line.split("=")[1].strip()
    elif line.startswith("prepare_receptor ="):
        prepare_receptor = line.split("=")[1].strip()
    elif line.startswith("prank ="):
        prank = line.split("=")[1].strip()
    elif line.startswith("boxMaxCutoff ="):
        p2rank_boxMaxCutoff = float(line.split("=")[1].strip())
    elif line.startswith("pocketCutoff ="):
        p2rank_pocketCutoff = float(line.split("=")[1].strip())
    elif line.startswith("vina ="):
        vina = line.split("=")[1].strip()
    elif line.startswith("vina_split ="):
        vina_split = line.split("=")[1].strip()
    elif line.startswith("vina_energy_range ="):
        vina_energy_range = line.split("=")[1].strip()
    elif line.startswith("vina_scoring ="):
        vina_scoring = line.split("=")[1].strip()
    elif line.startswith("vina_scoring_functions ="):
        vina_scoring_functions = [l.strip() for l in line.split("=")[1].strip().split(",")]
    elif line.startswith("vina_exhaustiveness ="):
        vina_exhaustiveness = int(line.split("=")[1].strip())
    elif line.startswith("vina_num_modes ="):
        vina_num_modes = line.split("=")[1].strip()
    elif line.startswith("smina ="):
        smina = line.split("=")[1].strip()
    elif line.startswith("smina_energy_range ="):
        smina_energy_range = line.split("=")[1].strip()
    elif line.startswith("smina_exhaustiveness ="):
        smina_exhaustiveness = line.split("=")[1].strip()
    elif line.startswith("smina_num_modes ="):
        smina_num_modes = line.split("=")[1].strip()
    elif line.startswith("smina_scoring ="):
        smina_scoring = line.split("=")[1].strip()
    elif line.startswith("smina_scoring_functions ="):
        smina_scoring_functions = [l.strip() for l in line.split("=")[1].strip().split(",")]
    elif line.startswith("smina_custom_scoring ="):
        smina_custom_scoring = line.split("=")[1].strip()
    elif line.startswith("smina_custom_atoms ="):
        smina_custom_atoms = line.split("=")[1].strip()
    elif line.startswith("smina_local_only ="):
        smina_local_only = line.split("=")[1].strip()
    elif line.startswith("smina_minimize ="):
        smina_minimize = line.split("=")[1].strip()
    elif line.startswith("smina_randomize_only ="):
        smina_randomize_only = line.split("=")[1].strip()
    elif line.startswith("smina_minimize_iters ="):
        smina_minimize_iters = line.split("=")[1].strip()
    elif line.startswith("smina_accurate_line ="):
        smina_accurate_line = line.split("=")[1].strip()
    elif line.startswith("smina_minimize_early_term ="):
        smina_minimize_early_term = line.split("=")[1].strip()
    elif line.startswith("smina_approximation ="):
        smina_approximation = line.split("=")[1].strip()
    elif line.startswith("smina_factor ="):
        smina_factor = line.split("=")[1].strip()
    elif line.startswith("smina_force_cap ="):
        smina_force_cap = line.split("=")[1].strip()
    elif line.startswith("smina_user_grid ="):
        smina_user_grid = line.split("=")[1].strip()
    elif line.startswith("smina_user_grid_lambda ="):
        smina_user_grid_lambda = line.split("=")[1].strip()
    elif line.startswith("gnina ="):
        gnina = line.split("=")[1].strip()
    elif line.startswith("gnina_exaustiveness ="):
        gnina_exhaustiveness = line.split("=")[1].strip()
    elif line.startswith("gnina_num_modes ="):
        gnina_num_modes = line.split("=")[1].strip()
    elif line.startswith("gnina_scoring ="):
        gnina_scoring = line.split("=")[1].strip()
    elif line.startswith("gnina_custom_scoring ="):
        gnina_custom_scoring = line.split("=")[1].strip()
    elif line.startswith("gnina_custom_atoms ="):
        gnina_custom_atoms = line.split("=")[1].strip()
    elif line.startswith("gnina_local_only ="):
        gnina_local_only = line.split("=")[1].strip()
    elif line.startswith("gnina_minimize ="):
        gnina_minimize = line.split("=")[1].strip()
    elif line.startswith("gnina_randomize_only ="):
        gnina_randomize_only = line.split("=")[1].strip()
    elif line.startswith("gnina_num_mc_steps ="):
        gnina_num_mc_steps = line.split("=")[1].strip()
    elif line.startswith("gnina_max_mc_steps ="):
        gnina_max_mc_steps = line.split("=")[1].strip()
    elif line.startswith("gnina_num_mc_saved ="):
        gnina_num_mc_saved = line.split("=")[1].strip()
    elif line.startswith("gnina_minimize_iters ="):
        gnina_minimize_iters = line.split("=")[1].strip()
    elif line.startswith("gnina_simple_ascent ="):
        gnina_simple_ascent = line.split("=")[1].strip()
    elif line.startswith("gnina_accurate_line ="):
        gnina_accurate_line = line.split("=")[1].strip()
    elif line.startswith("gnina_minimize_early_term ="):
        gnina_minimize_early_term = line.split("=")[1].strip()
    elif line.startswith("gnina_approximation ="):
        gnina_approximation = line.split("=")[1].strip()
    elif line.startswith("gnina_factor ="):
        gnina_factor = line.split("=")[1].strip()
    elif line.startswith("gnina_force_cap ="):
        gnina_force_cap = line.split("=")[1].strip()
    elif line.startswith("gnina_user_grid ="):
        gnina_user_grid = line.split("=")[1].strip()
    elif line.startswith("gnina_user_grid_lambda ="):
        gnina_user_grid_lambda = line.split("=")[1].strip()
    elif line.startswith("gnina_no_gpu ="):
        gnina_no_gpu = line.split("=")[1].strip()
    elif line.startswith("plants ="):
        plants = line.split("=")[1].strip()
    elif line.startswith("plants_cluster_structures ="):
        plants_cluster_structures = int(line.split("=")[1].strip())
    elif line.startswith("plants_cluster_rmsd ="):
        plants_cluster_rmsd = line.split("=")[1].strip()
    elif line.startswith("plants_search_speed ="):
        plants_search_speed = line.split("=")[1].strip()
    elif line.startswith("plants_scoring ="):
        plants_scoring = line.split("=")[1].strip()
    elif line.startswith("plants_scoring_functions ="):
        plants_scoring_functions = [l.strip() for l in line.split("=")[1].strip().split(",")]
    elif line.startswith("plants_rescoring_mode ="):
        plants_rescoring_mode = line.split("=")[1].strip()
    elif line.startswith("dock6 ="):
        dock6 = line.split("=")[1].strip()
    elif line.startswith("dock6_vdw_defn_file ="):
        dock6_vdw_defn_file = line.split("=")[1].strip()
    elif line.startswith("dock6_flex_defn_file ="):
        dock6_flex_defn_file = line.split("=")[1].strip()
    elif line.startswith("dock6_flex_drive_file ="):
        dock6_flex_drive_file = line.split("=")[1].strip()
    elif line.startswith("oddt ="):
        oddt = line.split("=")[1].strip()
    elif line.startswith("oddt_seed ="):
        oddt_seed = line.split("=")[1].strip()
    elif line.startswith("oddt_chunk_size ="):
        oddt_chunk_size = line.split("=")[1].strip()
    elif line.startswith("oddt_scoring_functions ="):
        oddt_scoring_functions = [l.strip() for l in line.split("=")[1].strip().split(",")]
    elif line.startswith("dssp ="):
        dssp = line.split("=")[1].strip()
    elif line.startswith("obabel ="):
        obabel = line.split("=")[1].strip()
    elif line.startswith("spores ="):
        spores = line.split("=")[1].strip()
    elif line.startswith("DUDEz ="):
        dudez_download = line.split("=")[1].strip()

# Root directory for OCDocker module
ocdocker_path = os.path.dirname(os.path.abspath( __file__ ))

# Check if the ocdb_path is defined in the config file (empty string means not defined)
if not ocdb_path:
    print(f"{clrs['r']}ERROR{clrs['n']}: The variable ocdb_path is not set in the config file '{args.config_file}'")
    quit()

# Directory containing the dudez archive
dudez_archive = os.path.join(ocdb_path, "DUDEz")

# Directory containing the pdbbind archive
pdbbind_archive = os.path.join(ocdb_path, "PDBbind")

# Directory containing the pdbbind archive
parsed_archive = os.path.join(ocdb_path, "Parsed")

# Set the log directory
logdir = f"{os.path.abspath(os.path.join(os.path.dirname(ocerror.__file__), os.pardir))}/logs"

# Set the oddt model directory
oddt_models_dir = f"{os.path.abspath(os.path.join(os.path.dirname(ocerror.__file__), os.pardir))}/ODDT_models"

# Check if logdir exists, if not, create-it
if not os.path.isdir(logdir):
    os.mkdir(logdir)

# Check if oddt_models_dir exists, if not, create-it
if not os.path.isdir(oddt_models_dir):
    os.mkdir(oddt_models_dir)

# Remove tmp path then create it again
tmpDir = f"{ocdocker_path}/tmp"

# If the dir exists
if os.path.isdir(tmpDir):
    # Remove it with all its contents
    shutil.rmtree(tmpDir)

# Then create it since it does not exist
os.mkdir(tmpDir)

# Get number of CPUs (minus one) with a minimum of one
if args.multiprocess:
    n_cpu = multiprocessing.cpu_count() - 1
    args.available_cores = n_cpu if n_cpu > 1 else 1
else:
    args.available_cores = 1

# Limit the output_level between acceptable values [0-4]
if args.output_level > 4:
    args.output_level = 4
elif args.output_level < 0:
    args.output_level = 0

# Create the ODDT models
initialise_oddt_models(oddt_models_dir, oddt_scoring_functions) # type: ignore

#TODO: Colocar uma lista de parâmetros do OCDocker
