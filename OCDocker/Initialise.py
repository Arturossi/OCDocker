#!/usr/lib/python3

# Imports
###############################################################################
import os
import argparse
import multiprocessing

import textwrap as tw

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

# Description
###############################################################################
description = tw.dedent("""\033[1;93m
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+- \033[1;96m┏━┓┏━╸╺┳━┓┏━┓┏━╸╻┏ ┏━╸┏━┓ \033[1;93m-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+- \033[1;96m┃ ┃┃   ┃ ┃┃ ┃┃  ┣┻┓┣╸ ┣┳┛ \033[1;93m-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+- \033[1;96m┗━┛┗━╸╺┻━┛┗━┛┗━╸╹ ╹┗━╸╹┗╸ \033[1;93m-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
\033[1;0m
      Copyright (C) 2021  Rossi, A.D; Torres, P.H.M.
\033[1;95m
                  [The Federal University of Rio de Janeiro]
\033[1;0m
          This program comes with ABSOLUTELY NO WARRANTY
      Add a description Here
     Please cite:
     Ainda não tem
\033[1;93m
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
\033[1;0m""")

# Functions
###############################################################################
def create_ocdocker_conf():
    '''
    Creates the 'ocdocker.conf' file.
    Input:
      -
    Return:
      -
    '''
    confOcdb = "/mnt/e/Documents/OCDocker/OCDocker/data/ocdb"
    confDock6 = "/mnt/e/Documents/OCDocker/software/docking/dock6/bin/dock6"
    confPlants = "/mnt/e/Documents/OCDocker/software/docking/plants/PLANTS1.2_64bit"
    confPythonsh = "/mnt/e/Documents/OCDocker/OCDocker/mgltools/bin/pythonsh"
    confPrepare_ligand = "/mnt/e/Documents/OCDocker/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
    confPrepare_receptor = "/mnt/e/Documents/OCDocker/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"

    confP2rankBoxMaxCutoff = "0.5"
    confP2RankPocketCutoff = "0.1"

    confObabel = "/usr/bin/obabel"
    confPrank = "/mnt/e/Documents/OCDocker/software/search/p2rank_2.3/prank"
    confDUDEz = "https://dudez.docking.org/DOCKING_GRIDS_AND_POSES.tgz"

    confVina = "/usr/bin/vina"
    confVina_energy_range = "10"
    confVina_exhaustiveness = "5"
    confVina_num_modes = "3"

    confSmina = "/mnt/e/Documents/OCDocker/software/docking/smina/build/smina"
    confSmina_energy_range = "10"
    confSmina_exhaustiveness = "5"
    confSmina_num_modes = "3"
    confSmina_scoring = "vinardo"
    confSmina_custom_scoring_file = "no"
    confSmina_custom_atoms = "no"
    confSmina_local_only = "no"
    confSmine_minimize = "no"
    confSmina_randomize_only = "no"
    confSmina_minimize_iters = "0"
    confSmina_accurate_line = "yes"
    confSmina_minimize_early_term = "no"
    confSmina_approximation = "spline"
    confSmina_factor = "32"
    confSmina_force_cap = "10"
    confSmina_user_grid = "no"
    confSmina_user_grid_lambda = "-1"

    confDssp = "/usr/bin/dssp"

    # General variables
    answer = input(f"Path to the OCDB. Default [{confOcdb}] (press enter to keep default): ")
    confOcdb = confOcdb if not answer else answer

    answer = input(f"Path to the Dock6 software. Default [{confDock6}] (press enter to keep default): ")
    confDock6 = confDock6 if not answer else answer

    answer = input(f"Path to the Plants software. Default [{confPlants}] (press enter to keep default): ")
    confPlants = confPlants if not answer else answer

    answer = input(f"Path to the obabel software. Default [{confObabel}] (press enter to keep default): ")
    confObabel = confObabel if not answer else answer

    answer = input(f"Path to the p2rank software. Default [{confPrank}] (press enter to keep default): ")
    confPrank = confPrank if not answer else answer

    answer = input(f"Link to the DUDEz database where you can download data. Default [{confDUDEz}] (press enter to keep default): ")
    confDUDEz = confDUDEz if not answer else answer

    # p2rank variables
    print("\np2rank configuration")
    answer = input(f"p2rank box max cutoff. Default [{confP2rankBoxMaxCutoff}] (press enter to keep default): ")
    confP2rankBoxMaxCutoff = confP2rankBoxMaxCutoff if not answer else answer

    answer = input(f"p2rank pocket cutoff. Default [{confP2RankPocketCutoff}] (press enter to keep default): ")
    confP2RankPocketCutoff = confP2RankPocketCutoff if not answer else answer

    # Smina variables
    print("\nSmina configuration")
    answer = input(f"Path to the Smina software. Default [{confSmina}] (press enter to keep default): ")
    confSmina = confSmina if not answer else answer

    answer = input(f"Smina energy range parameter. Default [{confSmina_energy_range}] (press enter to keep default): ")
    confSmina_energy_range = confSmina_energy_range if not answer else answer

    answer = input(f"Smina exhaustiveness parameter. Default [{confSmina_exhaustiveness}] (press enter to keep default): ")
    confSmina_exhaustiveness = confSmina_exhaustiveness if not answer else answer

    answer = input(f"Smina num modes parameter. Default [{confSmina_num_modes}] (press enter to keep default): ")
    confSmina_num_modes = confSmina_num_modes if not answer else answer

    answer = input(f"Smina scoring function parameter. Default [{confSmina_scoring}] (press enter to keep default): ")
    confSmina_scoring = confSmina_scoring if not answer else answer

    answer = input(f"Smina custom scoring file parameter ('no' to ignore this parameter, otherwise provide the path). Default [{confSmina_custom_scoring_file}] (press enter to keep default): ")
    confSmina_custom_scoring_file = confSmina_custom_scoring_file if not answer else answer

    answer = input(f"Smina custom atoms file parameter ('no' to ignore this parameter, otherwise provide the path). Default [{confSmina_custom_atoms}] (press enter to keep default): ")
    confSmina_custom_atoms = confSmina_custom_atoms if not answer else answer

    answer = input(f"Smina local only parameter [yes/no]. Default [{confSmina_local_only}] (press enter to keep default): ")
    confSmina_local_only = confSmina_local_only if not answer else answer.lower()

    answer = input(f"Smina minimize parameter [yes/no]. Default [{confSmine_minimize}] (press enter to keep default): ")
    confSmine_minimize = confSmine_minimize if not answer else answer.lower()

    answer = input(f"Smina randomize only parameter [yes/no]. Default [{confSmina_randomize_only}] (press enter to keep default): ")
    confSmina_randomize_only = confSmina_randomize_only if not answer else answer.lower()

    answer = input(f"Smina scoring function parameter. Default [{confSmina_minimize_iters}] (press enter to keep default): ")
    confSmina_minimize_iters = confSmina_minimize_iters if not answer else answer

    answer = input(f"Smina scoring function parameter [yes/no]. Default [{confSmina_accurate_line}] (press enter to keep default): ")
    confSmina_accurate_line = confSmina_accurate_line if not answer else answer.lower()

    answer = input(f"Smina minimize early parameter [yes/no]. Default [{confSmina_minimize_early_term}] (press enter to keep default): ")
    confSmina_minimize_early_term = confSmina_minimize_early_term if not answer else answer.lower()

    answer = input(f"Smina scoring function parameter. Default [{confSmina_approximation}] (press enter to keep default): ")
    confSmina_approximation = confSmina_approximation if not answer else answer

    answer = input(f"Smina factor parameter. Default [{confSmina_factor}] (press enter to keep default): ")
    confSmina_factor = confSmina_factor if not answer else answer

    answer = input(f"Smina force cap parameter. Default [{confSmina_force_cap}] (press enter to keep default): ")
    confSmina_force_cap = confSmina_force_cap if not answer else answer

    answer = input(f"Smina user grid parameter ('no' to ignore this parameter, otherwise provide the path). Default [{confSmina_user_grid}] (press enter to keep default): ")
    confSmina_user_grid = confSmina_user_grid if not answer else answer

    answer = input(f"Smina user grid lambda parameter. Default [{confSmina_user_grid_lambda}] (press enter to keep default): ")
    confSmina_user_grid_lambda = confSmina_user_grid_lambda if not answer else answer

    # Vina variables
    print("\nVina configuration")
    answer = input(f"Path to the Vina software. Default [{confVina}] (press enter to keep default): ")
    confVina = confVina if not answer else answer

    answer = input(f"Vina energy parameter. Default [{confVina_energy_range}] (press enter to keep default): ")
    confVina_energy_range = confVina_energy_range if not answer else answer

    answer = input(f"Vina exhaustiveness parameter. Default [{confVina_exhaustiveness}] (press enter to keep default): ")
    confVina_exhaustiveness = confVina_exhaustiveness if not answer else answer

    answer = input(f"Vina num modes parameter. Default [{confVina_num_modes}] (press enter to keep default): ")
    confVina_num_modes = confVina_num_modes if not answer else answer

    # MGLTools variables
    print("\nMGLTools configuration")
    answer = input(f"Path to the pythonsh env from MGLTools. Default [{confPythonsh}] (press enter to keep default): ")
    confPythonsh = confPythonsh if not answer else answer

    answer = input(f"Path to the prepare_ligand4.py script from MGLTools. Default [{confPrepare_ligand}] (press enter to keep default): ")
    confPrepare_ligand = confPrepare_ligand if not answer else answer

    answer = input(f"Path to the prepare_receptor4.py script from MGLTools. Default [{confPrepare_receptor}] (press enter to keep default): ")
    confPrepare_receptor = confPrepare_receptor if not answer else answer

    # Other software
    print("\nOther software configuration")
    answer = input(f"Path to the dssp file/command. Default [{confDssp}] (press enter to keep default): ")
    confDssp = confDssp if not answer else answer

    conf_file = "OCDocker.cfg"

    # Create the conf file
    with open(conf_file, "w") as cf:
        cf.write(tw.dedent("""
        # Root directory for the OCDocker Database
        ocdb = """ + confOcdb + """

        # dock6 path
        dock6 = """ + confDock6 + """

        # PLANTS path
        plants = """ + confPlants + """

        # Smina path
        smina = """ + confSmina + """

        # Vina path
        vina = """ + confVina + """

        # MGLTools's pythonsh path
        pythonsh = """ + confPythonsh + """

        # prepare_ligand4 path
        prepare_ligand = """ + confPrepare_ligand + """

        # prepare_receptor4 path
        prepare_receptor = """ + confPrepare_receptor + """

        # Open Babel path
        obabel = """ + confObabel + """

        # P2Rank path
        prank = """ + confPrank + """

        ################# P2RANK PARAMETERS #################

        # p2rank box cutoff
        boxMaxCutoff = """ + confP2rankBoxMaxCutoff + """

        # p2rank pocket cutoff
        pocketCutoff = """ + confP2RankPocketCutoff + """

        ############# DATABASE FETCH PARAMETERS #############

        # DUDEz download link
        DUDEz = """ + confDUDEz + """

        ################## VINA PARAMETERS ##################

        # Maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)
        vina_energy_range = """ + confVina_energy_range + """

        # Exhaustiveness of the global search
        vina_exhaustiveness = """ + confVina_exhaustiveness + """

        # Maximum number of binding modes to generate
        vina_num_modes = """ + confVina_num_modes + """

        ################# SMINA PARAMETERS ##################

        # Maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)
        smina_energy_range = """ + confSmina_energy_range + """

        # Exhaustiveness of the global search
        smina_exhaustiveness = """ + confSmina_exhaustiveness + """

        # Maximum number of binding modes to generate
        smina_num_modes = """ + confSmina_num_modes + """

        # Alternativa scoring function
        smina_scoring = """ + confSmina_scoring + """

        # Custom scoring file
        smina_custom_scoring = """ + confSmina_custom_scoring_file + """

        # Custom atoms
        smina_custom_atoms = """ + confSmina_custom_atoms + """

        # Local search only using autobox (you probably want to use --minimize)
        smina_local_only = """ + confSmina_local_only + """

        # Energy minimization
        smina_minimize = """ + confSmine_minimize + """

        # Generate random poses, attempting to avoid clashes
        smina_randomize_only = """ + confSmina_randomize_only + """

        # Number iterations of steepest descent; default scales with rotors and usually isn't sufficient for convergence
        smina_minimize_iters = """ + confSmina_minimize_iters + """

        # Stop minimization before convergence conditions are fully met
        smina_minimize_early_term = """ + confSmina_minimize_early_term + """

        # Use accurate line search
        smina_accurate_line = """ + confSmina_accurate_line + """

        # Approximation (linear, spline, or exact) to use
        smina_approximation = """ + confSmina_approximation + """

        # Approximation factor: higher results in a finer-grained approximation
        smina_factor = """ + confSmina_factor + """

        # Max allowed force; lower values more gently minimize clashing structures
        smina_force_cap = """ + confSmina_force_cap + """

        # Autodock map file for user grid data based calculations
        smina_user_grid = """ + confSmina_user_grid + """

        # Scales user_grid and functional scoring
        smina_user_grid_lambda = """ + confSmina_user_grid_lambda + """

        ################## OTHER SOFTWARE ###################

        # MSMS program for the surface calculation
        dssp = """ + confDssp + """

        """))

    print(f"{clrs['g']}Configuration file created!{clrs['n']} If you need to change the paths you might want to {clrs['y']}EDIT ITS CONTENTS{clrs['n']} or delete the file and execute this routine again so that your environment variables are correctly set. To ensure that all variables are correctly set, please restart OCDocker.")
    return

# Define Global Variables
###############################################################################
# General variables
global args
global clrs
global widgets
global workdir
global errors
global logdir

# Data from .cfg
global ocdb
global vina
global dock6
global prank
global smina
global obabel
global plants
global dudez_download
global pythonsh
global prepare_ligand
global prepare_receptor

# p2rank parameters
global p2rank_boxMaxCutoff
global p2rank_pocketCutoff

# Vina parameters
global vina_num_modes
global vina_energy_range
global vina_exhaustiveness

# Smina parameters
global smina_num_modes
global smina_energy_range
global smina_exhaustiveness
global smina_scoring
global smina_custom_scoring
global smina_custom_atoms
global smina_local_only
global smina_minimize
global smina_randomize_only
global smina_minimize_iters
global smina_accurate_line
global smina_minimize_early_term
global smina_approximation
global smina_factor
global smina_force_cap
global smina_user_grid
global smina_user_grid_lambda

# Database + OCDocker variables
global astex_archive
global dudez_archive
global ocdocker_path
global pdbbind_archive

# Other software
global dssp

# Aditional Variables
###############################################################################

# Dictionary for the output colors
clrs = {"r": "\033[1;91m",  # red
        "g": "\033[1;92m",  # green
        "y": "\033[1;93m",  # yellow
        "b": "\033[1;94m",  # blue
        "p": "\033[1;95m",  # purple
        "c": "\033[1;96m",  # cyan
        "n": "\033[1;0m"}   # default

# Parse command line arguments
###############################################################################
def argument_parsing():
    '''
    Get data to generate vina conf file from box file.
    Input:
      -
    Return:
     [argparse.ArgumentParser] - 'argparse' object with all arguments.
    '''
    # Create the parser
    parser = argparse.ArgumentParser(prog="OCDocker",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    # Add the arguments
    parser.add_argument("--version",
                        action="version",
                        version="%(prog)s 0.1.8")

    parser.add_argument("-f", "--file",
                        dest="input_file",
                        type=str,
                        metavar="",
                        help=".pdb file to input")

    parser.add_argument("--multiprocess",
                        dest="multiprocess",
                        action="store_true",
                        default=False,
                        help="Defines whether python multiprocessing should be enabled for compatible lenghty tasks")

    parser.add_argument("--generate-report",
                        dest="generate_report",
                        action="store_true",
                        default=False,
                        help="Creates a final HTML report for each generated model (forces -a MIG and --plot-topologies)")

    parser.add_argument("-z", "--zip-output",
                        dest="zip_output",
                        type=int,
                        default=0,
                        metavar="",
                        help="Defines the compression level. [0] No compression, [1] partial compression, [2] full compression")

    parser.add_argument("-u", "--update-databases",
                        dest="update",
                        action="store_true",
                        default=False,
                        help="Updates databases")

    parser.add_argument("-v", "--verbose",
                        dest="verbosity",
                        action="count",
                        default=0,
                        help="Controls verbosity")

    parser.add_argument("-d", "--debug",
                        dest="debug",
                        action="count",
                        default=0,
                        help="Controls debug mode")

    parser.add_argument("--conf",
                        dest="config_file",
                        type=str,
                        metavar="",
                        help="Configuration file containing external executable paths")

    # Return the parser
    return parser.parse_args()

# Set the args variable as the args from the argument_parsing function
args = argument_parsing()

# Create error class object (making all errors standard)
errors = ocerror.Error(args)

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

# Read the conf file and assign its data to its variables
for line in open(config_file, "r"):
    if line.startswith("ocdb ="):
        ocdb = line.split("=")[1].strip()
    elif line.startswith("dock6 ="):
        dock6 = line.split("=")[1].strip()
    elif line.startswith("plants ="):
        plants = line.split("=")[1].strip()
    elif line.startswith("smina ="):
        smina = line.split("=")[1].strip()
    elif line.startswith("vina ="):
        vina = line.split("=")[1].strip()
    elif line.startswith("prepare_ligand ="):
        prepare_ligand = line.split("=")[1].strip()
    elif line.startswith("pythonsh ="):
        pythonsh = line.split("=")[1].strip()
    elif line.startswith("prepare_receptor ="):
        prepare_receptor = line.split("=")[1].strip()
    elif line.startswith("obabel ="):
        obabel = line.split("=")[1].strip()
    elif line.startswith("DUDEz ="):
        dudez_download = line.split("=")[1].strip()
    elif line.startswith("prank ="):
        prank = line.split("=")[1].strip()
    elif line.startswith("boxMaxCutoff ="):
        p2rank_boxMaxCutoff = float(line.split("=")[1].strip())
    elif line.startswith("pocketCutoff ="):
        p2rank_pocketCutoff = float(line.split("=")[1].strip())
    elif line.startswith("vina_energy_range ="):
        vina_energy_range = line.split("=")[1].strip()
    elif line.startswith("vina_exhaustiveness ="):
        vina_exhaustiveness = line.split("=")[1].strip()
    elif line.startswith("vina_num_modes ="):
        vina_num_modes = line.split("=")[1].strip()
    elif line.startswith("smina_energy_range ="):
        smina_energy_range = line.split("=")[1].strip()
    elif line.startswith("smina_exhaustiveness ="):
        smina_exhaustiveness = line.split("=")[1].strip()
    elif line.startswith("smina_num_modes ="):
        smina_num_modes = line.split("=")[1].strip()
    elif line.startswith("smina_scoring ="):
        smina_scoring = line.split("=")[1].strip()
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
    elif line.startswith("dssp ="):
        dssp = line.split("=")[1].strip()

# Root directory for OCDocker module
ocdocker_path = os.path.dirname(os.path.abspath( __file__ ))

# Directory containing the astex archive
astex_archive = os.path.join(ocdb, "Astex")

# Directory containing the dudez archive
dudez_archive = os.path.join(ocdb, "DUDEZ")

# Directory containing the pdbbind archive
pdbbind_archive = os.path.join(ocdb, "PDBbind")

# Set the log directory
logdir = f"{os.path.abspath(os.path.join(os.path.dirname(ocerror.__file__), os.pardir))}/logs"

# Get number of CPUs (minus one) with a minimum of one
if args.multiprocess:
    n_cpu = multiprocessing.cpu_count() - 1
    args.available_cores = n_cpu if n_cpu > 1 else 1
else:
    args.available_cores = 1

#TODO: Colocar uma lista de parâmetros do OCDocker
