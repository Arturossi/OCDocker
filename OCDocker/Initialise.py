#!/usr/bin/env python3

# Description
###############################################################################
'''
First set of primordial variables and functions that are used to initialise the
OCDocker library.\n

They are imported as:

from OCDocker.Initialise import *
'''

# Imports
###############################################################################
#import argparse
import multiprocessing
import os
import shutil
import argparse

import textwrap as tw

import OCDocker.Toolbox.Constants as occ
import OCDocker.Error as ocerror
from OCDocker.DB.DBMinimal import create_database_if_not_exists, create_engine, create_session

from glob import glob

global output_level
output_level = ocerror.ReportLevel.NONE

import oddt
from oddt.scoring.functions.RFScore import rfscore
from oddt.scoring.functions.NNScore import nnscore
from oddt.scoring.functions.PLECscore import PLECscore
from sqlalchemy.engine.url import URL

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

# Splash, version & clear tmp
###############################################################################
ocVersion = "0.9.5"

_description = tw.dedent("""\033[1;93m
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+- \033[1;96m┏━┓┏━╸╺┳━┓┏━┓┏━╸╻┏ ┏━╸┏━┓ \033[1;93m-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+- \033[1;96m┃ ┃┃   ┃ ┃┃ ┃┃  ┣┻┓┣╸ ┣┳┛ \033[1;93m-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+- \033[1;96m┗━┛┗━╸╺┻━┛┗━┛┗━╸╹ ╹┗━╸╹┗╸ \033[1;93m-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
\033[1;0m
      Copyright (C) 2022  Rossi, A.D; Torres, P.H.M.
\033[1;95m
                  [The Federal University of Rio de Janeiro]
\033[1;0m
          This program comes with ABSOLUTELY NO WARRANTY

      OCDocker version: """ + ocVersion + """

     Please cite:
         -
\033[1;93m
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
\033[1;0m""")

# Functions
###############################################################################
def __inner_initialise_models(oddt_sf: str):
    '''Inner function that initialises the scoring functions from the ODDT

    Parameters
    ----------
    oddt_sf : str
        The scoring function to be initialised

    Returns
    -------
    None
    '''

    # Warn the user that the pickled model will be created
    print(f"{clrs['y']}WARNING{clrs['n']}: {oddt_sf} model is not pickled, it will be created now.")

    # Discover the scoring function
    if oddt_sf.lower().startswith('rfscore'):
        # Create the new kwargs dict
        new_kwargs = {}
        # For each bit in the oddt_sf string
        for bit in oddt_sf.lower().split('_'):
            # Fill the kwargs dict
            if bit.startswith('pdbbind'):
                new_kwargs['pdbbind_version'] = int(bit.replace('pdbbind', ''))
            elif bit.startswith('v'):
                new_kwargs['version'] = int(bit.replace('v', ''))
        # Load the scoring function (this will create the pickled model)
        _ = rfscore.load(**new_kwargs)
    elif oddt_sf.lower().startswith('nnscore'):
        # Create the new kwargs dict
        new_kwargs = {}
        # For each bit in the oddt_sf string
        for bit in oddt_sf.lower().split('_'):
            # Fill the kwargs dict
            if bit.startswith('pdbbind'):
                new_kwargs['pdbbind_version'] = int(bit.replace('pdbbind', ''))
        # Load the scoring function (this will create the pickled model)
        _ = nnscore.load(**new_kwargs)
    elif oddt_sf.lower().startswith('plec'):
        # Create the new kwargs dict
        new_kwargs = {}
        # For each bit in the oddt_sf string
        for bit in oddt_sf.lower().split('_'):
            # Fill the kwargs dict
            if bit.startswith('pdbbind'):
                new_kwargs['pdbbind_version'] = int(bit.replace('pdbbind', ''))
            elif bit.startswith('plec'):
                new_kwargs['version'] = bit.replace('plec', '')
            elif bit.startswith('p'):
                new_kwargs['depth_protein'] = int(bit.replace('p', ''))
            elif bit.startswith('l'):
                new_kwargs['depth_ligand'] = int(bit.replace('l', ''))
            elif bit.startswith('s'):
                new_kwargs['size'] = int(bit.replace('s', ''))
        # Load the scoring function (this will create the pickled model)
        _ = PLECscore.load(**new_kwargs)

    # Return
    return None

def get_argument_parsing() -> argparse.ArgumentParser:
    '''Get data to generate vina conf file from box file.
    
    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Argument parser object.
    '''
    
    # Create the parser
    parser = argparse.ArgumentParser(prog="OCDocker",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=_description)
    
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
                        help="Define the log level:\n\t0: Silent\n\t1: Critical\n\t2: Warning (default)\n\t3: Info\n\t4: Verbose mode\n\t5: Debug")
    
    parser.add_argument("--overwrite",
                        dest="overwrite",
                        action="store_true",
                        default=False,
                        help="Defines if OCDocker should overwrite existing files")

    # Return the parser
    return parser

def argument_parsing() -> argparse.Namespace:
    '''Parse the arguments from the command line.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Namespace object containing the arguments.
    '''

    # Return the parser
    return get_argument_parsing().parse_args()

def create_ocdocker_conf() -> None:
    '''Creates the 'ocdocker.conf' file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    #region Database config
    confHOST = 'localhost'
    confUSER = 'root'
    confPASSWORD = ''
    confDATABASE = 'ocdocker'
    confOPTIMIZEDB = 'optimization'
    confPORT = '3306'

    print("\nSQL database OCDocker configuration")
    answer = input(f"HOST. Default [{confHOST}] (press enter to keep default): ")
    confHOST = confHOST if not answer else answer

    answer = input(f"USER. Default [{confUSER}] (press enter to keep default): ")
    confUSER = confUSER if not answer else answer

    answer = input(f"PASSWORD. Default [{confPASSWORD}] (press enter to keep default): ")
    confPASSWORD = confPASSWORD if not answer else answer

    answer = input(f"DATABASE. Default [{confDATABASE}] (press enter to keep default): ")
    confDATABASE = confDATABASE if not answer else answer

    answer = input(f"OPTIMIZATION DATABASE. Default [{confOPTIMIZEDB}] (press enter to keep default): ")
    confOPTIMIZEDB = confOPTIMIZEDB if not answer else answer

    answer = input(f"PORT. Default [{confPORT}] (press enter to keep default): ")
    confPORT = confPORT if not answer else answer

    #endregion

    #region General config
    confOcdb = "/mnt/e/Documents/OCDocker/OCDocker/data/ocdb"
    confPCA = "/mnt/e/Documents/OCDocker/OCDocker/data/pca"
    confPDBbind_KdKi_order = "u"

    print("\nGeneral OCDocker configuration")
    answer = input(f"Path to the OCDB. Default [{confOcdb}] (press enter to keep default): ")
    confOcdb = confOcdb if not answer else answer

    answer = input(f"Path to the folder where the PCA models will be stored. Default [{confPCA}] (press enter to keep default): ")
    confPCA = confPCA if not answer else answer

    # Ensure that the answer is valid (reset its value to an known invalid value before checking)
    answer = ""
    while answer not in ["Y", "Z", "E", "P", "T", "G", "M", "k", "un", "c", "m", "u", "n", "pf", "a", "z", "y"]:
        answer = input(f"The default pdbbind KiKd magnitude [Y, Z, E, P, T, G, M, k, un, c, m, u, n, pf, a, z, y] (follow the unit prefix table). Default [{confPDBbind_KdKi_order}] (press enter to keep default): ")
        # If the enter has been pressed (answer == "")
        if answer == "":
            # Set the default value
            answer = "u"
        confPDBbind_KdKi_order = confPDBbind_KdKi_order if not answer else answer

    #endregion

    #region MGLTools config
    confPythonsh = "/mnt/e/Documents/OCDocker/OCDocker/mgltools/bin/pythonsh"
    confPrepare_ligand = "/mnt/e/Documents/OCDocker/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
    confPrepare_receptor = "/mnt/e/Documents/OCDocker/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"

    print("\nMGLTools configuration")
    answer = input(f"Path to the pythonsh env from MGLTools. Default [{confPythonsh}] (press enter to keep default): ")
    confPythonsh = confPythonsh if not answer else answer

    answer = input(f"Path to the prepare_ligand4.py script from MGLTools. Default [{confPrepare_ligand}] (press enter to keep default): ")
    confPrepare_ligand = confPrepare_ligand if not answer else answer

    answer = input(f"Path to the prepare_receptor4.py script from MGLTools. Default [{confPrepare_receptor}] (press enter to keep default): ")
    confPrepare_receptor = confPrepare_receptor if not answer else answer

    #endregion

    #region Vina config
    confVina = "/usr/bin/vina"
    confVina_split = "/usr/bin/vina_split"
    confVina_energy_range = "10"
    confVina_exhaustiveness = "5"
    confVina_num_modes = "3"
    confVina_scoring = "vina"
    confVina_scoring_functions = "vina,vinardo"

    print("\nVina configuration")
    answer = input(f"Path to the Vina software. Default [{confVina}] (press enter to keep default): ")
    confVina = confVina if not answer else answer

    answer = input(f"Path to the Vina split software. Default [{confVina_split}] (press enter to keep default): ")
    confVina_split = confVina_split if not answer else answer

    answer = input(f"Vina energy parameter. Default [{confVina_energy_range}] (press enter to keep default): ")
    confVina_energy_range = confVina_energy_range if not answer else answer

    answer = input(f"Vina exhaustiveness parameter. Default [{confVina_exhaustiveness}] (press enter to keep default): ")
    confVina_exhaustiveness = confVina_exhaustiveness if not answer else answer

    answer = input(f"Vina num modes parameter. Default [{confVina_num_modes}] (press enter to keep default): ")
    confVina_num_modes = confVina_num_modes if not answer else answer

    answer = input(f"Vina default scoring function. Default [{confVina_scoring}] (press enter to keep default): ")
    confVina_scoring = confVina_scoring if not answer else answer

    answer = input(f"Vina available scoring functions (separated by ','). Default [{confVina_scoring_functions}] (press enter to keep default): ")
    confVina_scoring_functions = confVina_scoring_functions if not answer else answer

    #endregion

    #region SMINA variables
    confSmina = "/mnt/e/Documents/OCDocker/software/docking/smina/build/smina"
    confSmina_energy_range = "10"
    confSmina_exhaustiveness = "5"
    confSmina_num_modes = "3"
    confSmina_scoring = "vinardo"
    confSmina_scoring_functions = "vina,vinardo,dkoes_scoring,dkoes_scoring_old,dkoes_fast,ad4_scoring"
    confSmina_custom_scoring_file = "no"
    confSmina_custom_atoms = "no"
    confSmina_local_only = "no"
    confSmina_minimize = "no"
    confSmina_randomize_only = "no"
    confSmina_minimize_iters = "0"
    confSmina_accurate_line = "yes"
    confSmina_minimize_early_term = "no"
    confSmina_approximation = "spline"
    confSmina_factor = "32"
    confSmina_force_cap = "10"
    confSmina_user_grid = "no"
    confSmina_user_grid_lambda = "-1"

    print("\nSmina configuration")
    answer = input(f"Path to the Smina software. Default [{confSmina}] (press enter to keep default): ")
    confSmina = confSmina if not answer else answer

    answer = input(f"Smina energy range parameter. Default [{confSmina_energy_range}] (press enter to keep default): ")
    confSmina_energy_range = confSmina_energy_range if not answer else answer

    answer = input(f"Smina exhaustiveness parameter. Default [{confSmina_exhaustiveness}] (press enter to keep default): ")
    confSmina_exhaustiveness = confSmina_exhaustiveness if not answer else answer

    answer = input(f"Smina num modes parameter. Default [{confSmina_num_modes}] (press enter to keep default): ")
    confSmina_num_modes = confSmina_num_modes if not answer else answer

    answer = input(f"Smina default scoring function parameter. Default [{confSmina_scoring}] (press enter to keep default): ")
    confSmina_scoring = confSmina_scoring if not answer else answer

    answer = input(f"Smina available scoring functions (separated by ','). Default [{confSmina_scoring_functions}] (press enter to keep default): ")
    confSmina_scoring_functions = confSmina_scoring_functions if not answer else answer

    answer = input(f"Smina custom scoring file parameter ('no' to ignore this parameter, otherwise provide the path). Default [{confSmina_custom_scoring_file}] (press enter to keep default): ")
    confSmina_custom_scoring_file = confSmina_custom_scoring_file if not answer else answer

    answer = input(f"Smina custom atoms file parameter ('no' to ignore this parameter, otherwise provide the path). Default [{confSmina_custom_atoms}] (press enter to keep default): ")
    confSmina_custom_atoms = confSmina_custom_atoms if not answer else answer

    answer = input(f"Smina local only parameter [yes/no]. Default [{confSmina_local_only}] (press enter to keep default): ")
    confSmina_local_only = confSmina_local_only if not answer else answer.lower()

    answer = input(f"Smina minimize parameter [yes/no]. Default [{confSmina_minimize}] (press enter to keep default): ")
    confSmina_minimize = confSmina_minimize if not answer else answer.lower()

    answer = input(f"Smina randomize only parameter [yes/no]. Default [{confSmina_randomize_only}] (press enter to keep default): ")
    confSmina_randomize_only = confSmina_randomize_only if not answer else answer.lower()

    answer = input(f"Smina scoring function parameter. Default [{confSmina_minimize_iters}] (press enter to keep default): ")
    confSmina_minimize_iters = confSmina_minimize_iters if not answer else answer

    answer = input(f"Smina use accurate line search parameter [yes/no]. Default [{confSmina_accurate_line}] (press enter to keep default): ")
    confSmina_accurate_line = confSmina_accurate_line if not answer else answer.lower()

    answer = input(f"Smina minimize early parameter [yes/no]. Default [{confSmina_minimize_early_term}] (press enter to keep default): ")
    confSmina_minimize_early_term = confSmina_minimize_early_term if not answer else answer.lower()

    answer = input(f"Smina approximation (linear, spline, or exact) to use parameter parameter. Default [{confSmina_approximation}] (press enter to keep default): ")
    confSmina_approximation = confSmina_approximation if not answer else answer

    answer = input(f"Smina factor parameter. Default [{confSmina_factor}] (press enter to keep default): ")
    confSmina_factor = confSmina_factor if not answer else answer

    answer = input(f"Smina force cap parameter. Default [{confSmina_force_cap}] (press enter to keep default): ")
    confSmina_force_cap = confSmina_force_cap if not answer else answer

    answer = input(f"Smina user grid parameter ('no' to ignore this parameter, otherwise provide the path). Default [{confSmina_user_grid}] (press enter to keep default): ")
    confSmina_user_grid = confSmina_user_grid if not answer else answer

    answer = input(f"Smina user grid lambda parameter. Default [{confSmina_user_grid_lambda}] (press enter to keep default): ")
    confSmina_user_grid_lambda = confSmina_user_grid_lambda if not answer else answer

    #endregion

    #region GNINA variables
    confGnina = "/data/hd4tb/OCDocker/software/docking/gnina/gnina"
    confGnina_exhaustiveness = "8"
    confGnina_num_modes = "9"
    confGnina_scoring = "default"
    confGnina_custom_scoring_file = "no"
    confGnina_custom_atoms = "no"
    confGnina_local_only = "no"
    confGnina_minimize = "no"
    confGnina_randomize_only = "no"
    confGnina_num_mc_steps = "no"
    confGnina_max_mc_steps = "no"
    confGnina_num_mc_saved = "no"
    confGnina_minimize_iters = "0"
    confGnina_simple_ascent = "no"
    confGnina_accurate_line = "yes"
    confGnina_minimize_early_term = "no"
    confGnina_approximation = "spline"
    confGnina_factor = "32"
    confGnina_force_cap = "10"
    confGnina_user_grid = "no"
    confGnina_user_grid_lambda = "-1"
    confGnina_no_gpu = "no"

    print("\nGnina configuration")
    answer = input(f"Path to the Gnina software. Default [{confGnina}] (press enter to keep default): ")
    confGnina = confGnina if not answer else answer

    answer = input(f"Gnina exhaustiveness parameter. Default [{confGnina_exhaustiveness}] (press enter to keep default): ")
    confGnina_exhaustiveness = confGnina_exhaustiveness if not answer else answer

    answer = input(f"Gnina num modes parameter. Default [{confGnina_num_modes}] (press enter to keep default): ")
    confGnina_num_modes = confGnina_num_modes if not answer else answer

    answer = input(f"Gnina scoring function parameter. Default [{confGnina_scoring}] (press enter to keep default): ")
    confGnina_scoring = confGnina_scoring if not answer else answer

    answer = input(f"Gnina custom scoring file parameter ('no' to ignore this parameter, otherwise provide the path). Default [{confGnina_custom_scoring_file}] (press enter to keep default): ")
    confGnina_custom_scoring_file = confGnina_custom_scoring_file if not answer else answer

    answer = input(f"Gnina custom atoms file parameter ('no' to ignore this parameter, otherwise provide the path). Default [{confGnina_custom_atoms}] (press enter to keep default): ")
    confGnina_custom_atoms = confGnina_custom_atoms if not answer else answer

    answer = input(f"Gnina local only parameter [yes/no]. Default [{confGnina_local_only}] (press enter to keep default): ")
    confGnina_local_only = confGnina_local_only if not answer else answer.lower()

    answer = input(f"Gnina minimize parameter [yes/no]. Default [{confGnina_minimize}] (press enter to keep default): ")
    confGnina_minimize = confGnina_minimize if not answer else answer.lower()

    answer = input(f"Gnina randomize only parameter [yes/no]. Default [{confGnina_randomize_only}] (press enter to keep default): ")
    confGnina_randomize_only = confGnina_randomize_only if not answer else answer.lower()

    answer = input(f"Gnina number of monte carlo steps parameter [yes/no]. Default [{confGnina_num_mc_steps}] (press enter to keep default): ")
    confGnina_num_mc_steps = confGnina_num_mc_steps if not answer else answer.lower()

    answer = input(f"Gnina cap on number of monte carlo steps to take in each chain. Default [{confGnina_max_mc_steps}] (press enter to keep default): ")
    confGnina_max_mc_steps = confGnina_max_mc_steps if not answer else answer.lower()

    answer = input(f"Gnina number of pose saves in each monte carlo chain parameter [yes/no]. Default [{confGnina_num_mc_saved}] (press enter to keep default): ")
    confGnina_num_mc_saved = confGnina_num_mc_saved if not answer else answer.lower()

    answer = input(f"Gnina number iterations of steepest descent parameter. Default [{confGnina_minimize_iters}] (press enter to keep default): ")
    confGnina_minimize_iters = confGnina_minimize_iters if not answer else answer

    answer = input(f"Gnina use simple gradient ascent parameter. Default [{confGnina_simple_ascent}] (press enter to keep default): ")
    confGnina_simple_ascent = confGnina_simple_ascent if not answer else answer

    answer = input(f"Gnina use accurate line search parameter [yes/no]. Default [{confGnina_accurate_line}] (press enter to keep default): ")
    confGnina_accurate_line = confGnina_accurate_line if not answer else answer.lower()

    answer = input(f"Gnina minimize early parameter [yes/no]. Default [{confGnina_minimize_early_term}] (press enter to keep default): ")
    confGnina_minimize_early_term = confGnina_minimize_early_term if not answer else answer.lower()

    answer = input(f"Gnina approximation (linear, spline, or exact) to use parameter. Default [{confGnina_approximation}] (press enter to keep default): ")
    confGnina_approximation = confGnina_approximation if not answer else answer.lower()

    answer = input(f"Gnina factor parameter. Default [{confGnina_factor}] (press enter to keep default): ")
    confGnina_factor = confGnina_factor if not answer else answer

    answer = input(f"Gnina force cap parameter. Default [{confGnina_force_cap}] (press enter to keep default): ")
    confGnina_force_cap = confGnina_force_cap if not answer else answer

    answer = input(f"Gnina user grid parameter ('no' to ignore this parameter, otherwise provide the path). Default [{confGnina_user_grid}] (press enter to keep default): ")
    confGnina_user_grid = confGnina_user_grid if not answer else answer

    answer = input(f"Gnina user grid lambda parameter. Default [{confGnina_user_grid_lambda}] (press enter to keep default): ")
    confGnina_user_grid_lambda = confGnina_user_grid_lambda if not answer else answer

    answer = input(f"Use CPU instead of GPU? Default [{confGnina_no_gpu}] (press enter to keep default): ")
    #endregion

    #region PLANTS variables
    confPlants = "/mnt/e/Documents/OCDocker/software/docking/plants/PLANTS1.2_64bit"
    confPlants_cluster_structures = 3
    confPlants_cluster_rmsd = 2.0
    confPlants_search_speed = "speed1"
    confPlants_scoring = "chemplp"
    confPlants_scoring_functions = "chemplp,plp,plp95"
    confPlants_rescoring_mode = "simplex"

    print("\nPLANTS configuration")
    answer = input(f"Path to the Plants software. Default [{confPlants}] (press enter to keep default): ")
    confPlants = confPlants if not answer else answer

    answer = input(f"How many structures will be generated. Default [{confPlants_cluster_structures}] (press enter to keep default): ")
    confPlants_cluster_structures = confPlants_cluster_structures if not answer else answer

    answer = input(f"PLANTS cluster RMSD parameter. Default [{confPlants_cluster_rmsd}] (press enter to keep default): ")
    confPlants_cluster_rmsd = confPlants_cluster_rmsd if not answer else answer

    answer = input(f"PLANTS search speed parameter. Default [{confPlants_search_speed}] (press enter to keep default): ")
    confPlants_search_speed = confPlants_search_speed if not answer else answer

    answer = input(f"PLANTS default scoring function. Default [{confPlants_scoring}] (press enter to keep default): ")
    confPlants_scoring = confPlants_scoring if not answer else answer

    answer = input(f"PLANTS available scoring functions (separated by ','). Default [{confPlants_scoring_functions}] (press enter to keep default): ")
    confPlants_scoring_functions = confPlants_scoring_functions if not answer else answer

    answer = input(f"PLANTS rescoring mode parameter. Default [{confPlants_rescoring_mode}] (press enter to keep default): ")
    confPlants_rescoring_mode = confPlants_rescoring_mode if not answer else answer

    #endregion

    #region DOCK6 variables
    confDock6 = "/mnt/e/Documents/OCDocker/software/docking/dock6/bin/dock6"
    confDock6_vdw_defn_file = "/mnt/e/Documents/OCDocker/software/docking/dock6/vdw_AMBER_parm99.defn"
    confDock6_flex_defn_file = "/mnt/e/Documents/OCDocker/software/docking/dock6/flex.defn"
    confDock6_flex_drive_file = "/mnt/e/Documents/OCDocker/software/docking/dock6/flex_drive.tbl"

    #print("\nDock6 configuration")
    answer = input(f"Path to the DOCK6 software. Default [{confDock6}] (press enter to keep default): ")
    confDock6 = confDock6 if not answer else answer

    answer = input(f"DOCK6 vdw_defn file path. Default [{confDock6_vdw_defn_file}] (press enter to keep default): ")
    confDock6_vdw_defn_file = confDock6_vdw_defn_file if not answer else answer

    answer = input(f"DOCK6 flex_defn file path. Default [{confDock6_flex_defn_file}] (press enter to keep default): ")
    confDock6_flex_defn_file = confDock6_flex_defn_file if not answer else answer

    answer = input(f"DOCK6 flex_drive file path. Default [{confDock6_flex_drive_file}] (press enter to keep default): ")
    confDock6_flex_drive_file = confDock6_flex_drive_file if not answer else answer

    #endregion

    #region Ledock variables
    confLedock = "/mnt/e/Documents/OCDocker/software/docking/ledock/ledock_linux_x86"
    confLepro = "/mnt/e/Documents/OCDocker/software/docking/ledock/lepro_linux_x86"
    confLedock_rmsd = "1.0"
    confLedock_num_poses = "3"

    print("\nLedock configuration")
    answer = input(f"Path to the Ledock software. Default [{confLedock}] (press enter to keep default): ")
    confLedock = confLedock if not answer else answer

    answer = input(f"Path to the Lepro software. Default [{confLepro}] (press enter to keep default): ")
    confLepro = confLepro if not answer else answer

    answer = input(f"Ledock RMSD parameter. Default [{confLedock_rmsd}] (press enter to keep default): ")
    confLedock_rmsd = confLedock_rmsd if not answer else answer

    answer = input(f"Ledock number of poses parameter. Default [{confLedock_num_poses}] (press enter to keep default): ")
    confLedock_num_poses = confLedock_num_poses if not answer else answer

    # endregion

    #region ODDT variables
    try:
        confODDT = os.popen("which oddt_cli").read().replace('\n', '').strip()
    except:
        confODDT = "/usr/bin/oddt_cli"
    
    confODDT_scoring_functions = "rfscore_v1_pdbbind2016,rfscore_v2_pdbbind2016,rfscore_v3_pdbbind2016,nnscore_pdbbind2016,plecrf_pdbbind2016"
    confODDT_seed = 42
    confODDT_chunk_size = 100

    print("\nODDT configuration")
    answer = input(f"Path to the ODDT file/command. Default [{confODDT}] (press enter to keep default): ")
    confODDT = confODDT if not answer else answer

    answer = input(f"ODDT seed parameter. Default [{confODDT_seed}] (press enter to keep default): ")
    confODDT_seed = confODDT_seed if not answer else answer

    answer = input(f"ODDT chunk size parameter. Default [{confODDT_chunk_size}] (press enter to keep default): ")
    confODDT_chunk_size = confODDT_chunk_size if not answer else answer

    answer = input(f"ODDT available scoring functions (separated by ',') (The supported scoring functions are: rfscore_v1_pdbbind2016, rfscore_v2_pdbbind2016, rfscore_v3_pdbbind2016, nnscore_pdbbind2016, plecrf_pdbbind2016). Default [{confODDT_scoring_functions}] (press enter to keep default): ")
    confODDT_scoring_functions = confODDT_scoring_functions if not answer else answer

    #endregion

    #region Other variables
    confDssp = "/usr/bin/dssp"
    confObabel = "/usr/bin/obabel"
    confSpores = "/mnt/e/Documents/OCDocker/software/docking/plants/SPORES_64bit"
    confDUDEz = "https://dudez.docking.org/DOCKING_GRIDS_AND_POSES.tgz" # this is WRONG
    confChimera = "/usr/bin/chimera"

    print("\nOther software configuration")
    answer = input(f"Path to the dssp file/command. Default [{confDssp}] (press enter to keep default): ")
    confDssp = confDssp if not answer else answer

    answer = input(f"Path to the obabel software. Default [{confObabel}] (press enter to keep default): ")
    confObabel = confObabel if not answer else answer

    answer = input(f"Path to the SPORES software. Default [{confSpores}] (press enter to keep default): ")
    confSpores = confSpores if not answer else answer

    answer = input(f"Link to the DUDEz database where you can download data. Default [{confDUDEz}] (press enter to keep default): ")
    confDUDEz = confDUDEz if not answer else answer

    answer = input(f"Path to the Chimera software. Default [{confChimera}] (press enter to keep default): ")
    confChimera = confChimera if not answer else answer

    #endregion

    # Define the config file (NOT CHANGABLE)
    conf_file = "OCDocker.cfg"

    # Create the conf file
    with open(conf_file, 'w') as cf:
        cf.write(tw.dedent("""#######################################################
###################### OCDocker #######################
#######################################################
                           
#################### SQL PARAMETERS ###################
HOST = """ + str(confHOST) + """
USER = """ + str(confUSER) + """
PASSWORD = """ + str(confPASSWORD) + """
DATABASE = """ + str(confDATABASE) + """
OPTIMIZEDB = """ + str(confOPTIMIZEDB) + """
PORT = """ + str(confPORT) + """

################### OCDB PARAMETERS ###################
                                 
# Root directory for the OCDocker Database
ocdb = """ + str(confOcdb) + """

# Directory for the PCA models
pca = """ + str(confPCA) + """

# The default pdbbind KiKd magnitude [Y, Z, E, P, T, G, M, k, un, c, m, u, n, pf, a, z, y] (follow the unit prefix table)
pdbbind_KdKi_order = """ + str(confPDBbind_KdKi_order) + """

################# MGLTools PARAMETERS #################

# MGLTools's pythonsh path
pythonsh = """ + str(confPythonsh) + """

# prepare_ligand4 path
prepare_ligand = """ + str(confPrepare_ligand) + """

# prepare_receptor4 path
prepare_receptor = """ + str(confPrepare_receptor) + """

################## VINA PARAMETERS ##################

# Vina path
vina = """ + str(confVina) + """

# Vina_split path
vina_split = """ + str(confVina_split) + """

# Maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)
vina_energy_range = """ + str(confVina_energy_range) + """

# Exhaustiveness of the global search
vina_exhaustiveness = """ + str(confVina_exhaustiveness) + """

# Maximum number of binding modes to generate
vina_num_modes = """ + str(confVina_num_modes) + """

# Default scoring function
vina_scoring = """ + str(confVina_scoring) + """

# Available scoring functions
vina_scoring_functions = """ + str(confVina_scoring_functions) + """

################# SMINA PARAMETERS ##################

# Smina path
smina = """ + str(confSmina) + """

# Maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)
smina_energy_range = """ + str(confSmina_energy_range) + """

# Exhaustiveness of the global search
smina_exhaustiveness = """ + str(confSmina_exhaustiveness) + """

# Maximum number of binding modes to generate
smina_num_modes = """ + str(confSmina_num_modes) + """

# Default scoring function
smina_scoring = """ + str(confSmina_scoring) + """

# Available scoring functions
smina_scoring_functions = """ + str(confSmina_scoring_functions) + """

# Custom scoring file
smina_custom_scoring = """ + str(confSmina_custom_scoring_file) + """

# Custom atoms
smina_custom_atoms = """ + str(confSmina_custom_atoms) + """

# Local search only using autobox (you probably want to use --minimize)
smina_local_only = """ + str(confSmina_local_only) + """

# Energy minimization
smina_minimize = """ + str(confSmina_minimize) + """

# Generate random poses, attempting to avoid clashes
smina_randomize_only = """ + str(confSmina_randomize_only) + """

# Number iterations of steepest descent; default scales with rotors and usually isn't sufficient for convergence
smina_minimize_iters = """ + str(confSmina_minimize_iters) + """

# Use accurate line search
smina_accurate_line = """ + str(confSmina_accurate_line) + """

# Stop minimization before convergence conditions are fully met
smina_minimize_early_term = """ + str(confSmina_minimize_early_term) + """

# Approximation (linear, spline, or exact) to use
smina_approximation = """ + str(confSmina_approximation) + """

# Approximation factor: higher results in a finer-grained approximation
smina_factor = """ + str(confSmina_factor) + """

# Max allowed force; lower values more gently minimize clashing structures
smina_force_cap = """ + str(confSmina_force_cap) + """

# Autodock map file for user grid data based calculations
smina_user_grid = """ + str(confSmina_user_grid) + """

# Scales user_grid and functional scoring
smina_user_grid_lambda = """ + str(confSmina_user_grid_lambda) + """

################# PLANTS PARAMETERS ##################

# PLANTS path
plants = """ + str(confPlants) + """

# Number of cluster structures
plants_cluster_structures = """ + str(confPlants_cluster_structures) + """

# RMSD value for plants
plants_cluster_rmsd = """ + str(confPlants_cluster_rmsd) + """

# Search speed
plants_search_speed = """ + str(confPlants_search_speed) + """

# Default scoring function
plants_scoring = """ + str(confPlants_scoring) + """

# Available scoring functions
plants_scoring_functions = """ + str(confPlants_scoring_functions) + """

# Plants rescoring mode
plants_rescoring_mode = """ + str(confPlants_rescoring_mode) + """

################# GNINA PARAMETERS ##################

# Gnina path
gnina = """ + str(confGnina) + """

# Exhaustiveness of the global search
gnina_exhaustiveness = """ + str(confGnina_exhaustiveness) + """

# Maximum number of binding modes to generate
gnina_num_modes = """ + str(confGnina_num_modes) + """

# Alternativa scoring function
gnina_scoring = """ + str(confGnina_scoring) + """

# Custom scoring file
gnina_custom_scoring = """ + str(confGnina_custom_scoring_file) + """

# Custom atoms
gnina_custom_atoms = """ + str(confGnina_custom_atoms) + """

# Local search only using autobox (you probably want to use --minimize)
gnina_local_only = """ + str(confGnina_local_only) + """

# Energy minimization
gnina_minimize = """ + str(confGnina_minimize) + """

# Generate random poses, attempting to avoid clashes
gnina_randomize_only = """ + str(confGnina_randomize_only) + """

# Number of monte carlo steps to take in each chain
gnina_num_mc_steps = """ + str(confGnina_num_mc_steps) + """

# Cap on number of monte carlo steps to take in each chain
gnina_max_mc_steps = """ + str(confGnina_max_mc_steps) + """

# Number of top poses saved in each monte carlo chain
gnina_num_mc_saved = """ + str(confGnina_num_mc_saved) + """

# Number iterations of steepest descent; default scales with rotors and usually isn't sufficient for convergence
gnina_minimize_iters = """ + str(confGnina_minimize_iters) + """

# Use simple gradient ascent
gnina_simple_ascent = """ + str(confGnina_simple_ascent) + """

# Use accurate line search
gnina_accurate_line = """ + str(confGnina_accurate_line) + """

# Stop minimization before convergence conditions are fully met
gnina_minimize_early_term = """ + str(confGnina_minimize_early_term) + """

# Approximation (linear, spline, or exact) to use
gnina_approximation = """ + str(confGnina_approximation) + """

# Approximation factor: higher results in a finer-grained approximation
gnina_factor = """ + str(confGnina_factor) + """

# Max allowed force; lower values more gently minimize clashing structures
gnina_force_cap = """ + str(confGnina_force_cap) + """

# Autodock map file for user grid data based calculations
gnina_user_grid = """ + str(confGnina_user_grid) + """

# Scales user_grid and functional scoring
gnina_user_grid_lambda = """ + str(confGnina_user_grid_lambda) + """

# Wether to use the GPU or not
gnina_no_gpu = """ + str(confGnina_no_gpu) + """

################# DOCK6 PARAMETERS ##################

# dock6 path
dock6 = """ + str(confDock6) + """

# Path to the vdw defn file
dock6_vdw_defn_file = """ + str(confDock6_vdw_defn_file) + """

# Path to the flex defn file
dock6_flex_defn_file = """ + str(confDock6_flex_defn_file) + """

# Path to the flex drive file
dock6_flex_drive_file = """ + str(confDock6_flex_drive_file) + """

################# LEDOCK PARAMETERS #################

# LeDock path
ledock = """ + str(confLedock) + """

# Path to the LePro software
lepro = """ + str(confLepro) + """

# LeDock RMSD parameter
ledock_rmsd = """ + str(confLedock_rmsd) + """

# Maximum number of poses to generate
ledock_num_poses = """ + str(confLedock_num_poses) + """

################## ODDT PARAMETERS ##################

# Path to the oddt_cli file
oddt = """ + str(confODDT) + """

# Seed for the ODDT software
oddt_seed = """ + str(confODDT_seed) + """

# Seed for the ODDT chunk size
oddt_chunk_size = """ + str(confODDT_chunk_size) + """

# Alternative scoring function
oddt_scoring_functions = """ + str(confODDT_scoring_functions) + """

################## OTHER SOFTWARE ###################

# Chimeta program for dock file preparation
chimera = """ + str(confChimera) + """

# MSMS program for the surface calculation
dssp = """ + str(confDssp) + """

# Open Babel path
obabel = """ + str(confObabel) + """

# SPORES path
spores = """ + str(confSpores) + """

# DUDEz download link
DUDEz = """ + str(confDUDEz) + """
"""))

    print(f"{clrs['g']}Configuration file created!{clrs['n']} If you need to change the paths you might want to {clrs['y']}EDIT ITS CONTENTS{clrs['n']} or delete the file and execute this routine again so that your environment variables are correctly set. To ensure that all variables are correctly set, please restart OCDocker.")
    return

def initialise_oddt_models(oddt_models_dir: str, oddt_scoring_functions_aux: list) -> None:
    '''Initialise the ODDT models.

    Parameters
    ----------
    oddt_models_dir : str
        The path to the ODDT models directory.
    oddt_scoring_functions_aux : list
        The list of scoring functions to initialise.

    Returns
    -------
    None
    '''

    # Flag to print the warning only once
    warning_flag = True

    # Find which models are already pickled
    oddt_models = glob(f"{oddt_models_dir}/*.pickle")

    # Process the scoring function names to match the ODDT models
    processedNames = [".".join(oddt_model.split(os.path.sep)[-1].split(".")[:-1]).lower() if "plecrf" not in oddt_model.lower() else "plecrf" for oddt_model in oddt_models]

    # For each model, check if it is in the list of scoring functions
    for oddt_scoring_function_aux in oddt_scoring_functions_aux:
        # If the scoring function is not in the list of models or if it is plecrf and the plecrf model is not in the list of models
        if (not oddt_scoring_function_aux.startswith("plecrf") and oddt_scoring_function_aux not in processedNames) or (oddt_scoring_function_aux.startswith("plecrf") and "plecrf" not in processedNames):
            # If the flag is True, print the warning
            if warning_flag:
                # Warn the user that this could take some time
                print(f"{clrs['c']}INFO{clrs['n']}: Generating ODDT models for the first time can take a while, please be patient.")
                # Set the flag to False
                warning_flag = False
                # Save the current dir in a variable
                current_dir = os.getcwd()
            # Change to the ODDT models dir
            os.chdir(oddt_models_dir)
            # Initialise the model
            __inner_initialise_models(oddt_scoring_function_aux)
            # Return to the previous dir
            os.chdir(current_dir) # type: ignore
    # Return
    return None
"""
def set_argparse() -> None:
    '''Parse the arguments and set them to the global variables.
    '''

    # Call the argument parser
    args = argument_parsing()

    global multiprocess
    global update
    global config_file
    global output_level
    global overwrite

    # Set the global variables
    '''multiprocess = args.multiprocess
    update = args.update
    config_file = args.config
    output_level = ocerror.ReportLevel(args.output_level)
    overwrite = args.overwrite'''
"""
def set_log_level(level: ocerror.ReportLevel) -> None:
    '''Set the log level.

    Parameters
    ----------
    level : ocerror.ReportLevel
        The level of the log.
    '''

    ocerror.Error.set_output_level(ocerror.ReportLevel.WARNING)

# Define Global Variables
###############################################################################
# General variables
global args
global clrs
global widgets
global workdir

global logdir
global tmpdir
global oddt_models_dir
global cpu_cores
global available_cores
global multiprocess
global update
global config_file
#global output_level
global overwrite

# DB variables
global db_url
global engine
global session
global optdb_url

# Order variable
global order
global pdbbind_KdKi_order

# Data from .cfg
global ocdb_path
global pca_path
global vina
global vina_split
global dock6
global smina
global gnina
global obabel
global plants
global dudez_download
global pythonsh
global prepare_ligand
global prepare_receptor

# Vina parameters
global vina_scoring
global vina_scoring_functions
global vina_num_modes
global vina_energy_range
global vina_exhaustiveness

# Smina parameters
global smina_num_modes
global smina_energy_range
global smina_exhaustiveness
global smina_scoring
global smina_scoring_functions
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

# Gnina parameters
global gnina_exhaustiveness
global gnina_num_modes
global gnina_scoring
global gnina_custom_scoring_file
global gnina_custom_atoms
global gnina_local_only
global gnina_minimize
global gnina_randomize_only
global gnina_num_mc_steps
global gnina_max_mc_steps
global gnina_num_mc_saved
global gnina_minimize_iters
global gnina_simple_ascent
global gnina_accurate_line
global gnina_minimize_early_term
global gnina_approximation
global gnina_factor
global gnina_force_cap
global gnina_user_grid
global gnina_user_grid_lambda
global gnina_no_gpu

# PLANTS parameters
global plants_cluster_structures
global plants_cluster_rmsd
global plants_search_speed
global plants_scoring
global plants_scoring_functions

# Dock6 parameters
global dock6_vdw_defn_file
global dock6_flex_defn_file
global dock6_flex_drive_file

# ODDT parameters
global oddt_program
global oddt_seed
global oddt_chunk_size
global oddt_scoring_functions

# Database + OCDocker variables
global dudez_archive
global ocdocker_path
global pdbbind_archive
global parsed_archive

# Other software
global dssp

# Aditional Variables
###############################################################################

# Dictionary for the output colors
clrs = {
    "r": "\033[1;91m",  # red
    "g": "\033[1;92m",  # green
    "y": "\033[1;93m",  # yellow
    "b": "\033[1;94m",  # blue
    "p": "\033[1;95m",  # purple
    "c": "\033[1;96m",  # cyan
    "n": "\033[1;0m"    # default
    }

# This structure is to define which will be used order, the first index will be the default magnitude and the other is element magnitude [Y, Z, E, P, T, G, M, k, un, c, m, u, n, p, f, a, z, y]
order = {
    "Y": {
        "Y": 1e0, "Z": 1e-3, "E": 1e-6, "P": 1e-9, "T": 1e-12, "G": 1e-15, "M": 1e-18, "k": 1e-21, "un": 1e-24, "c": 1e-26, "m": 1e-27, "u": 1e-30, "n": 1e-33, "p": 1e-36, "f": 1e-39, "a": 1e-42, "z": 1e-45, "y": 1e-48
    },
    "Z": {
        "Y": 1e3, "Z": 1e0, "E": 1e-3, "P": 1e-6, "T": 1e-9, "G": 1e-12, "M": 1e-15, "k": 1e-18, "un": 1e-21, "c": 1e-23, "m": 1e-24, "u": 1e-27, "n": 1e-30, "p": 1e-33, "f": 1e-36, "a": 1e-39, "z": 1e-42, "y": 1e-45
    },
    "E": {
        "Y": 1e6, "Z": 1e3, "E": 1e0, "P": 1e-3, "T": 1e-6, "G": 1e-9, "M": 1e-12, "k": 1e-15, "un": 1e-18, "c": 1e-20, "m": 1e-21, "u": 1e-24, "n": 1e-27, "p": 1e-30, "f": 1e-33, "a": 1e-36, "z": 1e-39, "y": 1e-42
    },
    "P": {
        "Y": 1e9, "Z": 1e6, "E": 1e3, "P": 1e0, "T": 1e-3, "G": 1e-6, "M": 1e-9, "k": 1e-12, "un": 1e-15, "c": 1e-17, "m": 1e-18, "u": 1e-21, "n": 1e-24, "p": 1e-27, "f": 1e-30, "a": 1e-33, "z": 1e-36, "y": 1e-39
    },
    "T": {
        "Y": 1e12, "Z": 1e9, "E": 1e6, "P": 1e3, "T": 1e0, "G": 1e-3, "M": 1e-6, "k": 1e-9, "un": 1e-12, "c": 1e-14, "m": 1e-15, "u": 1e-18, "n": 1e-21, "p": 1e-24, "f": 1e-27, "a": 1e-30, "z": 1e-33, "y": 1e-34
    },
    "G": {
        "Y": 1e15, "Z": 1e12, "E": 1e9, "P": 1e6, "T": 1e3, "G": 1e0, "M": 1e-3, "k": 1e-6, "un": 1e-9, "c": 1e-11, "m": 1e-12, "u": 1e-15, "n": 1e-18, "p": 1e-21, "f": 1e-24, "a": 1e-27, "z": 1e-30, "y": 1e-33
    },
    "M": {
        "Y": 1e18, "Z": 1e18, "E": 1e12, "P": 1e9, "T": 1e6, "G": 1e3, "M": 1e0, "k": 1e-3, "un": 1e-6, "c": 1e-8, "m": 1e-9, "u": 1e-12, "n": 1e-15, "p": 1e-18, "f": 1e-21, "a": 1e-24, "z": 1e-27, "y": 1e-30
    },
    "k": {
        "Y": 1e21, "Z": 1e18, "E": 1e15, "P": 1e12, "T": 1e9, "G": 1e6, "M": 1e3, "k": 1e0, "un": 1e-3, "c": 1e-5, "m": 1e-6, "u": 1e-9, "n": 1e-12, "p": 1e-15, "f": 1e-18, "a": 1e-21, "z": 1e-24, "y": 1e-27
    },
    "un": {
        "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "un": 1e0, "c": 1e-2, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18, "z": 1e-21, "y": 1e-24
    },
    "c": {
        "Y": 1e26, "Z": 1e23, "E": 1e20, "P": 1e17, "T": 1e14, "G": 1e11, "M": 1e8, "k": 1e5, "un": 1e2, "c": 1e0, "m": 1e-1, "u": 1e-4, "n": 1e-7, "p": 1e-10, "f": 1e-13, "a": 1e-16, "z": 1e-19, "y": 1e-22
    },
    "m": {
        "Y": 1e27, "Z": 1e24, "E": 1e21, "P": 1e18, "T": 1e15, "G": 1e12, "M": 1e9, "k": 1e6, "un": 1e3, "c": 1e1, "m": 1e0, "u": 1e-3, "n": 1e-6, "p": 1e-9, "f": 1e-12, "a": 1e-15, "z": 1e-18, "y": 1e-21
    },
    "u": {
        "Y": 1e30, "Z": 1e27, "E": 1e24, "P": 1e21, "T": 1e18, "G": 1e15, "M": 1e12, "k": 1e9, "un": 1e6, "c": 1e4, "m": 1e3, "u": 1e0, "n": 1e-3, "p": 1e-6, "f": 1e-9, "a": 1e-12, "z": 1e-15, "y": 1e-18
    },
    "n": {
        "Y": 1e33, "Z": 1e30, "E": 1e27, "P": 1e24, "T": 1e21, "G": 1e18, "M": 1e15, "k": 1e12, "un": 1e9, "c": 1e7, "m": 1e6, "u": 1e3, "n": 1e0, "p": 1e-3, "f": 1e-6, "a": 1e-9, "z": 1e-12, "y": 1e-15
    },
    "p": {
        "Y": 1e36, "Z": 1e33, "E": 1e30, "P": 1e27, "T": 1e24, "G": 1e21, "M": 1e18, "k": 1e15, "un": 1e12, "c": 1e10, "m": 1e9, "u": 1e6, "n": 1e3, "p": 1e0, "f": 1e-3, "a": 1e-6, "z": 1e-9, "y": 1e-12
    },
    "f": {
        "Y": 1e39, "Z": 1e36, "E": 1e33, "P": 1e30, "T": 1e27, "G": 1e24, "M": 1e21, "k": 1e18, "un": 1e15, "c": 1e13, "m": 1e12, "u": 1e9, "n": 1e6, "p": 1e3, "f": 1e0, "a": 1e-3, "z": 1e-6, "y": 1e-9
    },
    "a": {
        "Y": 1e42, "Z": 1e39, "E": 1e36, "P": 1e33, "T": 1e30, "G": 1e27, "M": 1e24, "k": 1e21, "un": 1e18, "c": 1e16, "m": 1e15, "u": 1e12, "n": 1e9, "p": 1e6, "f": 1e3, "a": 1e0, "z": 1e-3, "y": 1e-6
    },
    "z": {
        "Y": 1e45, "Z": 1e42, "E": 1e39, "P": 1e36, "T": 1e33, "G": 1e30, "M": 1e27, "k": 1e24, "un": 1e21, "c": 1e19, "m": 1e18, "u": 1e15, "n": 1e12, "p": 1e9, "f": 1e6, "a": 1e3, "z": 1e0, "y": 1e-3
    },
    "y": {
        "Y": 1e48, "Z": 1e45, "E": 1e42, "P": 1e39, "T": 1e36, "G": 1e33, "M": 1e30, "k": 1e27, "un": 1e24, "c": 1e22, "m": 1e21, "u": 1e18, "n": 1e15, "p": 1e12, "f": 1e9, "a": 1e6, "z": 1e3, "y": 1e0
    }
}

# Parse command line arguments
###############################################################################

'''
    args = argument_parsing()
    multiprocess = args.multiprocess
    update = args.update
    config_file = args.config_file or os.getenv('OCDOCKER_CONFIG', 'OCDocker.cfg')
    output_level = args.output_level
    overwrite = args.overwrite
'''

def is_doc_build() -> bool:
    '''Detects if the code is being run in a documentation (e.g., Sphinx) or test context.'''
    import sys
    import inspect

    # Check if common doc/test modules are loaded
    if any(m in sys.modules for m in ["sphinx", "sphinx.ext.autodoc", "pytest", "unittest", "doctest"]):
        return True

    # Check call stack for doc-related callers
    for frame in inspect.stack():
        if any(kw in frame.filename.lower() for kw in ["sphinx", "pytest", "unittest", "doctest"]):
            return True

    return False

if not is_doc_build():
    try:
        args = argument_parsing()
        multiprocess = args.multiprocess
        update = args.update
        config_file = args.config_file or os.getenv('OCDOCKER_CONFIG', 'OCDocker.cfg')
        output_level = args.output_level
        overwrite = args.overwrite
        ocerror.Error.set_output_level(output_level)
    except SystemExit:
        print(f"{clrs['r']}ERROR{clrs['n']}: Invalid command line arguments.")
        exit()

    # proceed with full environment setup...
    # config file loading, DB setup, oddt_models_dir, etc.
else:
    import sys
    sys.argv = [sys.argv[0]]
    # Minimal config to satisfy imports in Sphinx/test
    args = argparse.Namespace(
        multiprocess=False,
        update=False,
        config_file=os.getenv('OCDOCKER_CONFIG', 'OCDocker.cfg'),
        output_level=ocerror.ReportLevel.WARNING,
        overwrite=False
    )
    multiprocess = False
    update = False
    config_file = args.config_file
    output_level = args.output_level
    overwrite = False
    ocerror.Error.set_output_level(output_level)

# Initialise
###############################################################################
def print_description() -> None:
    ''' Print the description of the program.
    '''

    print(_description)
    
# Retrieve the paths from provided configuration file
if (not config_file or not os.path.isfile(config_file)) and not os.path.isfile("OCDocker.cfg"):
    print("OCDocker configuration file has not been found in the provided path")
    if __name__ == "__main__":
        create_config = input("Do you wish to create it? (y/n) ")
        if create_config.lower() in ["y", "ye", "yes"]:
            create_ocdocker_conf()
            exit()
        else:
            print("\n\nNo positive confirmation, please provide a valid configuration file.\n")
            exit()
    else:
        print("Create the configuration file manually or set the path to it in the environment variable \"OCDOCKER_CONFIG\".")
        exit()

elif not config_file and os.path.isfile("OCDocker.cfg"):
    config_file = "OCDocker.cfg"

elif config_file:
    assert os.path.isfile(config_file), f"{clrs['r']}\n\n Not able to find configuration file.\n\n Does \"{config_file}\" exist?{clrs['n']}"
    config_file = config_file

print_description()

# Set the ocdb path as an empty string
ocdb_path = ""

# Set db variables as empty strings
HOST = ""
USER = ""
PASSWORD = ""
DATABASE = ""
OPTIMIZEDB = ""
PORT = ""

# Read the conf file and assign its data to its variables (The order matters here, if you follow the same order which is in the conf file less computation power will be needed! It is not much, but it is something.)
for line in open(config_file, 'r'): # type: ignore
    if line.startswith("HOST ="):
        HOST = line.split("=")[1].strip()
    elif line.startswith("USER ="):
        USER = line.split("=")[1].strip()
    elif line.startswith("PASSWORD ="):
        PASSWORD = line.split("=")[1].strip()
    elif line.startswith("DATABASE ="):
        DATABASE = line.split("=")[1].strip()
    elif line.startswith("OPTIMIZEDB ="):
        OPTIMIZEDB = line.split("=")[1].strip()
    elif line.startswith("PORT ="):
        PORT = line.split("=")[1].strip()
        # Check if the port is a number
        if not PORT.isdigit():
            print(f"{clrs['r']}ERROR{clrs['n']}: The port number must be an integer.")
            exit()
        # Parse the port as an integer
        PORT = int(PORT)
    elif line.startswith("ocdb ="):
        ocdb_path = line.split("=")[1].strip()
    elif line.startswith("pca ="):
        pca_path = line.split("=")[1].strip()
    elif line.startswith("pdbbind_KdKi_order ="):
        pdbbind_KdKi_order = line.split("=")[1].strip()
    elif line.startswith("pythonsh ="):
        pythonsh = line.split("=")[1].strip()
    elif line.startswith("prepare_ligand ="):
        prepare_ligand = line.split("=")[1].strip()
    elif line.startswith("prepare_receptor ="):
        prepare_receptor = line.split("=")[1].strip()
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
    elif line.startswith("gnina_exhaustiveness ="):
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
    elif line.startswith("ledock ="):
        ledock = line.split("=")[1].strip()
    elif line.startswith("lepro ="):
        lepro = line.split("=")[1].strip()
    elif line.startswith("ledock_rmsd ="):
        ledock_rmsd = line.split("=")[1].strip()
    elif line.startswith("ledock_num_poses ="):
        ledock_num_poses = line.split("=")[1].strip()
    elif line.startswith("oddt ="):
        oddt_program = line.split("=")[1].strip()
    elif line.startswith("oddt_seed ="):
        oddt_seed = line.split("=")[1].strip()
    elif line.startswith("oddt_chunk_size ="):
        oddt_chunk_size = line.split("=")[1].strip()
    elif line.startswith("oddt_scoring_functions ="):
        oddt_scoring_functions = [l.strip() for l in line.split("=")[1].strip().split(",")]
    elif line.startswith("chimera ="):
        chimera = line.split("=")[1].strip()
    elif line.startswith("dssp ="):
        dssp = line.split("=")[1].strip()
    elif line.startswith("obabel ="):
        obabel = line.split("=")[1].strip()
    elif line.startswith("spores ="):
        spores = line.split("=")[1].strip()
    elif line.startswith("DUDEz ="):
        dudez_download = line.split("=")[1].strip()

# Check if the db_config variables are empty
if not HOST or not USER or not PASSWORD or not DATABASE or not PORT:
    print(f"{clrs['r']}ERROR{clrs['n']}: The variables HOST, USER, PASSWORD, DATABASE and PORT must be set in the config file '{config_file}'")
    exit()

# Create the database URLs
db_url = URL.create(
    drivername = 'mysql+pymysql',
    host = HOST,
    username = USER,
    password = PASSWORD,
    database = DATABASE,
    port = PORT
)

optdb_url = URL.create(
    drivername = 'mysql+pymysql',
    host = HOST,
    username = USER,
    password = PASSWORD,
    database = OPTIMIZEDB,
    port = PORT
)

# Set the engine
#engine = create_engine(db_url.render_as_string(hide_password=False))
#optengine = create_engine(optdb_url.render_as_string(hide_password=False))  
engine = create_engine(db_url)
#optengine = create_engine(optdb_url)

# Create the databases if it does not exist
create_database_if_not_exists(engine.url)
create_database_if_not_exists(optdb_url)

# Set the session factory as scoped to ensure that the session is thread-safe
session = create_session(engine)

# Root directory for OCDocker module
ocdocker_path = os.path.dirname(os.path.abspath( __file__ ))

# Check if the ocdb_path is defined in the config file (empty string means not defined)
if not ocdb_path:
    print(f"{clrs['r']}ERROR{clrs['n']}: The variable ocdb_path is not set in the config file '{config_file}'")
    exit()

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

# Check if pca_path exists, if not, create-it
if not os.path.isdir(pca_path):
    os.mkdir(pca_path)

# Remove tmp path then create it again
tmpDir = f"{ocdocker_path}/tmp"

try:
    # If the dir exists
    if os.path.isdir(tmpDir):
        # Remove it with all its contents
        shutil.rmtree(tmpDir)
        
    # Then create it since it does not exist
    os.mkdir(tmpDir)
except:
    pass

# Get number of CPUs (minus one) with a minimum of one
if multiprocess:
    n_cpu = multiprocessing.cpu_count() - 1
    available_cores = n_cpu if n_cpu > 1 else 1
else:
    available_cores = 1

# Limit the output_level between acceptable values [0-4]
if output_level > ocerror.ReportLevel.DEBUG:
    output_level = ocerror.ReportLevel.DEBUG
elif output_level < ocerror.ReportLevel.NONE:
    output_level = ocerror.ReportLevel.NONE
else:
    output_level = ocerror.ReportLevel.NONE

# Create error class object (making all errors standard)
ocerror.Error.set_output_level(output_level)

# Create the ODDT models
initialise_oddt_models(oddt_models_dir, oddt_scoring_functions) # type: ignore
