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
    confOcdb = "/mnt/d/Documents/OCDocker/OCDocker/data/ocdb"
    confDock6 = "/mnt/d/Documents/OCDocker/software/docking/dock6/bin/dock6"
    confPlants = "/mnt/d/Documents/OCDocker/software/docking/plants/PLANTS1.2_64bit"
    confSmina = "/mnt/d/Documents/OCDocker/software/docking/smina/build/smina"
    confVina = "/usr/bin/vina"
    confPythonsh = "/mnt/d/Documents/OCDocker/OCDocker/mgltools/bin/pythonsh"
    confPrepare_ligand = "/mnt/d/Documents/OCDocker/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
    confPrepare_receptor = "/mnt/d/Documents/OCDocker/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
    confObabel = "/usr/bin/obabel"
    confPrank = "/mnt/d/Documents/OCDocker/software/search/p2rank_2.3/prank"
    confDUDEz = "https://dudez.docking.org/DOCKING_GRIDS_AND_POSES.tgz"
    confVina_energy_range = "10"
    confVina_exhaustiveness = "5"
    confVina_num_modes = "3"
    confSmina_energy_range = "10"
    confSmina_exhaustiveness = "5"
    confSmina_num_modes = "3"


    answer = input(f"Path to the OCDB. Default [{confOcdb}] (press enter to keep default): ")
    confOcdb = confOcdb if not answer else answer

    answer = input(f"Path to the Dock6 software. Default [{confDock6}] (press enter to keep default): ")
    confDock6 = confDock6 if not answer else answer

    answer = input(f"Path to the Plants software. Default [{confPlants}] (press enter to keep default): ")
    confPlants = confPlants if not answer else answer

    answer = input(f"Path to the Smina software. Default [{confSmina}] (press enter to keep default): ")
    confSmina = confSmina if not answer else answer

    answer = input(f"Path to the Vina software. Default [{confVina}] (press enter to keep default): ")
    confVina = confVina if not answer else answer

    answer = input(f"Path to the pythonsh env from MGLTools. Default [{confPythonsh}] (press enter to keep default): ")
    confPythonsh = confPythonsh if not answer else answer

    answer = input(f"Path to the prepare_ligand4.py script from MGLTools. Default [{confPrepare_ligand}] (press enter to keep default): ")
    confPrepare_ligand = confPrepare_ligand if not answer else answer

    answer = input(f"Path to the prepare_receptor4.py script from MGLTools. Default [{confPrepare_receptor}] (press enter to keep default): ")
    confPrepare_receptor = confPrepare_receptor if not answer else answer

    answer = input(f"Path to the obabel software. Default [{confObabel}] (press enter to keep default): ")
    confObabel = confObabel if not answer else answer

    answer = input(f"Path to the p2rank software. Default [{confPrank}] (press enter to keep default): ")
    confPrank = confPrank if not answer else answer

    answer = input(f"Link to the DUDEz database where you can download data. Default [{confDUDEz}] (press enter to keep default): ")
    confDUDEz = confDUDEz if not answer else answer

    answer = input(f"Vina energy parameter. Default [{confVina_energy_range}] (press enter to keep default): ")
    confVina_energy_range = confVina_energy_range if not answer else answer

    answer = input(f"Vina exhaustiveness parameter. Default [{confVina_exhaustiveness}] (press enter to keep default): ")
    confVina_exhaustiveness = confVina_exhaustiveness if not answer else answer

    answer = input(f"Vina num modes parameter. Default [{confVina_num_modes}] (press enter to keep default): ")
    confVina_num_modes = confVina_num_modes if not answer else answer

    answer = input(f"Smina energy range parameter. Default [{confSmina_energy_range}] (press enter to keep default): ")
    confSmina_energy_range = confSmina_energy_range if not answer else answer

    answer = input(f"Smina exhaustiveness parameter. Default [{confSmina_exhaustiveness}] (press enter to keep default): ")
    confSmina_exhaustiveness = confSmina_exhaustiveness if not answer else answer

    answer = input(f"Smina num modes parameter. Default [{confSmina_num_modes}] (press enter to keep default): ")
    confSmina_num_modes = confSmina_num_modes if not answer else answer

    conf_file = "OCDocker.cfg"
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

        #

        """))

    '''Scoring and minimization options:
      --scoring arg                specify alternative builtin scoring function [e.g. vinardo]
      --custom_scoring arg         custom scoring function file
      --custom_atoms arg           custom atom type parameters file
      --score_only                 score provided ligand pose
      --local_only                 local search only using autobox (you probably
                                   want to use --minimize)
      --minimize                   energy minimization
      --randomize_only             generate random poses, attempting to avoid
                                   clashes
      --minimize_iters arg (=0)    number iterations of steepest descent; default
                                   scales with rotors and usually isn't sufficient
                                   for convergence
      --accurate_line              use accurate line search
      --minimize_early_term        Stop minimization before convergence conditions
                                   are fully met.
      --approximation arg          approximation (linear, spline, or exact) to use
      --factor arg                 approximation factor: higher results in a
                                   finer-grained approximation
      --force_cap arg              max allowed force; lower values more gently
                                   minimize clashing structures
      --user_grid arg              Autodock map file for user grid data based
                                   calculations
      --user_grid_lambda arg (=-1) Scales user_grid and functional scoring
      --print_terms                Print all available terms with default
                                   parameterizations
      --print_atom_types           Print all available atom types'''

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

# Vina parameters
global vina_num_modes
global vina_energy_range
global vina_exhaustiveness

# Vina parameters
global smina_num_modes
global smina_energy_range
global smina_exhaustiveness

# Database + OCDocker variables
global dudez_archive
global ocdocker_path
global pdbbind_archive

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
                        version="%(prog)s 0.1.1")

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

# Root directory for OCDocker module
ocdocker_path = os.path.dirname(os.path.abspath( __file__ ))

# Directory containing the pdb mirror in "divided" scheme
pdbbind_archive = os.path.join(ocdb, "pdbBind")

# Directory containing the pdb mirror in "divided" scheme
dudez_archive = os.path.join(ocdb, "DUDEZ")

# Get number of CPUs (minus one) with a minimum of one
if args.multiprocess:
    n_cpu = multiprocessing.cpu_count() - 1
    args.available_cores = n_cpu if n_cpu > 1 else 1
else:
    args.available_cores = 1

#TODO: Colocar uma lista de parâmetros do OCDocker
