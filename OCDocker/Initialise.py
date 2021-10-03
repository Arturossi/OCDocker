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
    conf_file = "OCDocker.cfg"
    with open(conf_file, "w") as cf:
        cf.write(tw.dedent("""
        # Root directory for the OCDocker Database
        ocdb = /mnt/d/Documents/OCDocker/OCDocker/data/ocdb

        # dock6 path
        dock6 = /mnt/d/Documents/OCDocker/software/docking/dock6/bin/dock6

        # PLANTS path
        plants = /mnt/d/Documents/OCDocker/software/docking/plants/PLANTS1.2_64bit

        # Smina path
        smina = /mnt/d/Documents/OCDocker/software/docking/smina/build/smina

        # Vina path
        vina = /usr/bin/vina

        # MGLTools's pythonsh path
        pythonsh = /mnt/d/Documents/OCDocker/OCDocker/mgltools/bin/pythonsh

        # prepare_ligand4 path
        prepare_ligand = /mnt/d/Documents/OCDocker/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py

        # prepare_receptor4 path
        prepare_receptor = /mnt/d/Documents/OCDocker/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py

        # Open Babel path
        obabel = /usr/bin/obabel

        # P2Rank path
        prank = /mnt/d/Documents/OCDocker/software/search/p2rank_2.3/prank

        ############# DATABASE FETCH PARAMETERS #############

        # DUDEz download link
        DUDEz = https://dudez.docking.org/DOCKING_GRIDS_AND_POSES.tgz

        ################## VINA PARAMETERS ##################

        # Maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)
        vina_energy_range = 10

        # Exhaustiveness of the global search
        vina_exhaustiveness = 5

        # Maximum number of binding modes to generate
        vina_num_modes = 3

        ################# SMINA PARAMETERS ##################

        # Maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)
        smina_energy_range = 10

        # Exhaustiveness of the global search
        smina_exhaustiveness = 5

        # Maximum number of binding modes to generate
        smina_num_modes = 3

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

    print(f"{clrs['g']}Configuration file created!{clrs['n']} Please{clrs['y']} EDIT ITS CONTENTS {clrs['n']}to match your environment and run OCDocker again.")
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
