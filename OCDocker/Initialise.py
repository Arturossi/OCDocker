#!/usr/lib/python3

# Imports
###############################################################################
import os
import argparse
import multiprocessing

import textwrap as tw

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
    conf_file = "OCDocker.cfg"
    with open(conf_file, 'w') as cf:
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
        energy_range = 10

        # Exhaustiveness of the global search
        exhaustiveness = 5

        # Maximum number of binding modes to generate
        num_modes = 3
        """))

    print(f"{clrs['g']}Configuration file created!{clrs['n']} Please{clrs['y']} EDIT ITS CONTENTS {clrs['n']}to match your environment and run OCDocker again.")

# Define Global Variables
###############################################################################
# General variables
global args
global clrs
global aa3to1
global aa1to3
global widgets
global workdir

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
global num_modes
global energy_range
global exhaustiveness

# Database + OCDocker variables
global dudez_archive
global ocdocker_path
global pdbbind_archive

# Aditional Variables
###############################################################################

# Dictionary for the output colors
clrs = {'r': "\033[1;91m",  # red
        'g': "\033[1;92m",  # green
        'y': "\033[1;93m",  # yellow
        'b': "\033[1;94m",  # blue
        'p': "\033[1;95m",  # purple
        'c': "\033[1;96m",  # cyan
        'n': "\033[1;0m"}   # default

# Parse command line arguments
###############################################################################
def argument_parsing():
    parser = argparse.ArgumentParser(prog='OCDocker',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)

    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s 0.0.1')

    parser.add_argument('-f', '--file',
                        dest='input_file',
                        type=str,
                        metavar='',
                        help='Pdb file to input')

    parser.add_argument('--multiprocess',
                        dest='multiprocess',
                        action='store_true',
                        default=False,
                        help='Defines whether python multiprocessing should be enabled for compatible lenghty tasks')

    parser.add_argument('--generate-report',
                        dest='generate_report',
                        action='store_true',
                        default=False,
                        help='Creates a final HTML report for each generated model (forces -a MIG and --plot-topologies)')

    parser.add_argument('-z', '--zip-output',
                        dest='zip_output',
                        type=int,
                        default=0,
                        metavar='',
                        help='Defines the compression level. [0] No compression, [1] partial compression, [2] full compression')

    parser.add_argument('-u', '--update-databases',
                        dest='update',
                        action='store_true',
                        default=False,
                        help='Updates databases')

    parser.add_argument('-v', '--verbose',
                        dest='verbosity',
                        action='count',
                        default=0,
                        help='Controls verbosity')

    parser.add_argument('--conf',
                        dest='config_file',
                        type=str,
                        metavar='',
                        help='Configuration file containing external executable paths')

    initial_args = parser.parse_args()

    return initial_args

initial_args = argument_parsing()

# Conversion AA 3 char code to 1 char code
aa3to1 = {'CYS': 'C', 'ASP': 'D', 'GLN': 'Q', 'ILE': 'I',
          'ALA': 'A', 'TYR': 'Y', 'TRP': 'W', 'HIS': 'H',
          'LEU': 'L', 'ARG': 'R', 'VAL': 'V', 'GLU': 'E',
          'PHE': 'F', 'GLY': 'G', 'MET': 'M', 'ASN': 'N',
          'PRO': 'P', 'SER': 'S', 'LYS': 'K', 'THR': 'T',
          'MSE': 'M', 'CSE': 'U', 'GLH': 'E', 'HID': 'H',
          'HIE': 'H', 'HIP': 'H', 'HYP': 'P', 'ASX': 'B',
          'GLX': 'Z', 'MME': 'M', 'LYZ': 'K'}

# Conversion AA 1 char code to 3 char code
aa1to3 = dict((v,k) for k,v in aa3to1.items())

# Initialise
###############################################################################
print(description)

# Retrieve the paths from provided configuration file
if (not initial_args.config_file or not os.path.isfile(initial_args.config_file)) and not os.path.isfile('OCDocker.cfg'):
    print('OCDocker configuration file not found in the provided path')
    create_config = input('Do you wish to create it? (y/n)')
    if create_config.lower()  in ['y', 'ye', 'yes']:
        create_ocdocker_conf()
        quit()
    else:
        print('\n\nNo positive confirmation, please provide a valid configuration file.\n')
        quit()

elif not initial_args.config_file and os.path.isfile('OCDocker.cfg'):
    config_file = 'OCDocker.cfg'

elif initial_args.config_file:
    assert os.path.isfile(initial_args.config_file), f"{clrs['r']}\n\n Not able to find configuration file.\n\n Does \"{initial_args.config_file}\" exist?{clrs['n']}"
    config_file = initial_args.config_file

for line in open(config_file, 'r'):
    if line.startswith('ocdb'):
        ocdb = line.split('=')[1].strip()
    elif line.startswith('dock6'):
        dock6 = line.split('=')[1].strip()
    elif line.startswith('plants'):
        plants = line.split('=')[1].strip()
    elif line.startswith('smina'):
        smina = line.split('=')[1].strip()
    elif line.startswith('vina'):
        vina = line.split('=')[1].strip()
    elif line.startswith('prepare_ligand'):
        prepare_ligand = line.split('=')[1].strip()
    elif line.startswith('pythonsh'):
        pythonsh = line.split('=')[1].strip()
    elif line.startswith('prepare_receptor'):
        prepare_receptor = line.split('=')[1].strip()
    elif line.startswith('obabel'):
        obabel = line.split('=')[1].strip()
    elif line.startswith('DUDEz'):
        dudez_download = line.split('=')[1].strip()
    elif line.startswith('prank'):
        prank = line.split('=')[1].strip()
    elif line.startswith('energy_range'):
        energy_range = line.split('=')[1].strip()
    elif line.startswith('exhaustiveness'):
        exhaustiveness = line.split('=')[1].strip()
    elif line.startswith('num_modes'):
        num_modes = line.split('=')[1].strip()

# Root directory for OCDocker module
ocdocker_path = os.path.dirname(os.path.abspath( __file__ ))

# Directory containing the pdb mirror in "divided" scheme
pdbbind_archive = os.path.join(ocdb, 'pdbBind')

# Directory containing the pdb mirror in "divided" scheme
dudez_archive = os.path.join(ocdb, 'DUDEZ')

# Get number of CPUs (minus one) with a minimum of one
if initial_args.multiprocess:
    n_cpu = multiprocessing.cpu_count() - 1
    initial_args.available_cores = n_cpu if n_cpu > 1 else 1
else:
    initial_args.available_cores = 1

#TODO: Colocar uma lista de parâmetros do OCDocker
