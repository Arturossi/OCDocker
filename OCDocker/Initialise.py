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

import multiprocessing
import os
import shutil
import argparse
import inspect
import atexit
import configparser
from pathlib import Path

import textwrap as tw
from typing import Optional, Any, Dict, Union

try:
    from unittest.mock import MagicMock
except Exception:
    MagicMock = None

import OCDocker.Toolbox.Constants as occ
import OCDocker.Error as ocerror
from OCDocker.DB.DBMinimal import create_database_if_not_exists, create_engine, create_session, cleanup_session, cleanup_engine

from glob import glob

output_level = ocerror.ReportLevel.NONE

from sqlalchemy.engine.url import URL

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Splash, version & clear tmp
###############################################################################
ocVersion = "0.11.1"

_description = tw.dedent("""\033[1;93m
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+- \033[1;96m┏━┓┏━╸╺┳━┓┏━┓┏━╸╻┏ ┏━╸┏━┓ \033[1;93m-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+- \033[1;96m┃ ┃┃   ┃ ┃┃ ┃┃  ┣┻┓┣╸ ┣┳┛ \033[1;93m-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+- \033[1;96m┗━┛┗━╸╺┻━┛┗━┛┗━╸╹ ╹┗━╸╹┗╸ \033[1;93m-+-+-+-+-+-+-+-+-+-+
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
\033[1;0m
      Copyright (C) 2025  Rossi, A.D; Torres, P.H.M.
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

# NOTE: Configuration values are now managed via OCDocker.Config module.
# Use `from OCDocker.Config import get_config` to access configuration.
# Runtime objects (db_url, engine, session, etc.) are still available as module-level globals.

# Functions
###############################################################################
def __inner_initialise_models(oddt_sf: str) -> None:
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

    # Discover the scoring function (lazy import to avoid hard dependency at import time)
    if oddt_sf.lower().startswith('rfscore'):
        from oddt.scoring.functions.RFScore import rfscore  # type: ignore
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
        from oddt.scoring.functions.NNScore import nnscore  # type: ignore
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
        from oddt.scoring.functions.PLECscore import PLECscore  # type: ignore
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
    confOcdb = "~/OCDocker/data/ocdb"
    confPCA = "~/OCDocker/data/pca"
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
    confPythonsh = "~/OCDocker/mgltools/bin/pythonsh"
    confPrepare_ligand = "~/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
    confPrepare_receptor = "~/OCDocker/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"

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
    confSmina = "~/software/docking/smina/build/smina"
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
    confPlants = "~/software/docking/plants/PLANTS1.2_64bit"
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
    confDock6 = "~/software/docking/dock6/bin/dock6"
    confDock6_vdw_defn_file = "~/software/docking/dock6/vdw_AMBER_parm99.defn"
    confDock6_flex_defn_file = "~/software/docking/dock6/flex.defn"
    confDock6_flex_drive_file = "~/software/docking/dock6/flex_drive.tbl"

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
    confLedock = "~/software/docking/ledock/ledock_linux_x86"
    confLepro = "~/software/docking/ledock/lepro_linux_x86"
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
    except (OSError, IOError):
        # Fallback to default path if 'which' command fails
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
    confSpores = "~/software/docking/plants/SPORES_64bit"
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

# Reference column order for OCScore mask application (comma-separated list)
# This list defines the exact column order used during model training
# CRITICAL: The order of scoring functions (SFs) must match the training data order
reference_column_order = name,receptor,ligand,SMINA_VINA,SMINA_SCORING_DKOES,SMINA_VINARDO,SMINA_OLD_SCORING_DKOES,SMINA_FAST_DKOES,SMINA_SCORING_AD4,VINA_VINA,VINA_VINARDO,PLANTS_CHEMPLP,PLANTS_PLP,PLANTS_PLP95,ODDT_RFSCORE_V1,ODDT_RFSCORE_V2,ODDT_RFSCORE_V3,ODDT_PLECRF_P5_L1_S65536,ODDT_NNSCORE,countA,countR,countN,countD,countC,countQ,countE,countG,countH,countI,countL,countK,countM,countF,countP,countS,countT,countW,countY,countV,TotalAALength,AvgAALength,countChain,SASA,DipoleMoment,IsoelectricPoint,GRAVY,Aromaticity,InstabilityIndex,AUTOCORR2D_1,AUTOCORR2D_2,AUTOCORR2D_3,AUTOCORR2D_4,AUTOCORR2D_5,AUTOCORR2D_6,AUTOCORR2D_7,AUTOCORR2D_8,AUTOCORR2D_9,AUTOCORR2D_10,AUTOCORR2D_11,AUTOCORR2D_12,AUTOCORR2D_13,AUTOCORR2D_14,AUTOCORR2D_15,AUTOCORR2D_16,AUTOCORR2D_17,AUTOCORR2D_18,AUTOCORR2D_19,AUTOCORR2D_20,AUTOCORR2D_21,AUTOCORR2D_22,AUTOCORR2D_23,AUTOCORR2D_24,AUTOCORR2D_25,AUTOCORR2D_26,AUTOCORR2D_27,AUTOCORR2D_28,AUTOCORR2D_29,AUTOCORR2D_30,AUTOCORR2D_31,AUTOCORR2D_32,AUTOCORR2D_33,AUTOCORR2D_34,AUTOCORR2D_35,AUTOCORR2D_36,AUTOCORR2D_37,AUTOCORR2D_38,AUTOCORR2D_39,AUTOCORR2D_40,AUTOCORR2D_41,AUTOCORR2D_42,AUTOCORR2D_43,AUTOCORR2D_44,AUTOCORR2D_45,AUTOCORR2D_46,AUTOCORR2D_47,AUTOCORR2D_48,AUTOCORR2D_49,AUTOCORR2D_50,AUTOCORR2D_51,AUTOCORR2D_52,AUTOCORR2D_53,AUTOCORR2D_54,AUTOCORR2D_55,AUTOCORR2D_56,AUTOCORR2D_57,AUTOCORR2D_58,AUTOCORR2D_59,AUTOCORR2D_60,AUTOCORR2D_61,AUTOCORR2D_62,AUTOCORR2D_63,AUTOCORR2D_64,AUTOCORR2D_65,AUTOCORR2D_66,AUTOCORR2D_67,AUTOCORR2D_68,AUTOCORR2D_69,AUTOCORR2D_70,AUTOCORR2D_71,AUTOCORR2D_72,AUTOCORR2D_73,AUTOCORR2D_74,AUTOCORR2D_75,AUTOCORR2D_76,AUTOCORR2D_77,AUTOCORR2D_78,AUTOCORR2D_79,AUTOCORR2D_80,AUTOCORR2D_81,AUTOCORR2D_82,AUTOCORR2D_83,AUTOCORR2D_84,AUTOCORR2D_85,AUTOCORR2D_86,AUTOCORR2D_87,AUTOCORR2D_88,AUTOCORR2D_89,AUTOCORR2D_90,AUTOCORR2D_91,AUTOCORR2D_92,AUTOCORR2D_93,AUTOCORR2D_94,AUTOCORR2D_95,AUTOCORR2D_96,AUTOCORR2D_97,AUTOCORR2D_98,AUTOCORR2D_99,AUTOCORR2D_100,AUTOCORR2D_101,AUTOCORR2D_102,AUTOCORR2D_103,AUTOCORR2D_104,AUTOCORR2D_105,AUTOCORR2D_106,AUTOCORR2D_107,AUTOCORR2D_108,AUTOCORR2D_109,AUTOCORR2D_110,AUTOCORR2D_111,AUTOCORR2D_112,AUTOCORR2D_113,AUTOCORR2D_114,AUTOCORR2D_115,AUTOCORR2D_116,AUTOCORR2D_117,AUTOCORR2D_118,AUTOCORR2D_119,AUTOCORR2D_120,AUTOCORR2D_121,AUTOCORR2D_122,AUTOCORR2D_123,AUTOCORR2D_124,AUTOCORR2D_125,AUTOCORR2D_126,AUTOCORR2D_127,AUTOCORR2D_128,AUTOCORR2D_129,AUTOCORR2D_130,AUTOCORR2D_131,AUTOCORR2D_132,AUTOCORR2D_133,AUTOCORR2D_134,AUTOCORR2D_135,AUTOCORR2D_136,AUTOCORR2D_137,AUTOCORR2D_138,AUTOCORR2D_139,AUTOCORR2D_140,AUTOCORR2D_141,AUTOCORR2D_142,AUTOCORR2D_143,AUTOCORR2D_144,AUTOCORR2D_145,AUTOCORR2D_146,AUTOCORR2D_147,AUTOCORR2D_148,AUTOCORR2D_149,AUTOCORR2D_150,AUTOCORR2D_151,AUTOCORR2D_152,AUTOCORR2D_153,AUTOCORR2D_154,AUTOCORR2D_155,AUTOCORR2D_156,AUTOCORR2D_157,AUTOCORR2D_158,AUTOCORR2D_159,AUTOCORR2D_160,AUTOCORR2D_161,AUTOCORR2D_162,AUTOCORR2D_163,AUTOCORR2D_164,AUTOCORR2D_165,AUTOCORR2D_166,AUTOCORR2D_167,AUTOCORR2D_168,AUTOCORR2D_169,AUTOCORR2D_170,AUTOCORR2D_171,AUTOCORR2D_172,AUTOCORR2D_173,AUTOCORR2D_174,AUTOCORR2D_175,AUTOCORR2D_176,AUTOCORR2D_177,AUTOCORR2D_178,AUTOCORR2D_179,AUTOCORR2D_180,AUTOCORR2D_181,AUTOCORR2D_182,AUTOCORR2D_183,AUTOCORR2D_184,AUTOCORR2D_185,AUTOCORR2D_186,AUTOCORR2D_187,AUTOCORR2D_188,AUTOCORR2D_189,AUTOCORR2D_190,AUTOCORR2D_191,AUTOCORR2D_192,BCUT2D_CHGHI,BCUT2D_CHGLO,BCUT2D_LOGPHI,BCUT2D_LOGPLOW,BCUT2D_MRHI,BCUT2D_MRLOW,BCUT2D_MWHI,BCUT2D_MWLOW,fr_Al_COO,fr_Al_OH,fr_Al_OH_noTert,fr_ArN,fr_Ar_COO,fr_Ar_N,fr_Ar_NH,fr_Ar_OH,fr_COO,fr_COO2,fr_C_O,fr_C_O_noCOO,fr_C_S,fr_HOCCN,fr_Imine,fr_NH0,fr_NH1,fr_NH2,fr_N_O,fr_Ndealkylation1,fr_Ndealkylation2,fr_Nhpyrrole,fr_SH,fr_aldehyde,fr_alkyl_carbamate,fr_alkyl_halide,fr_allylic_oxid,fr_amide,fr_amidine,fr_aniline,fr_aryl_methyl,fr_azide,fr_azo,fr_barbitur,fr_benzene,fr_benzodiazepine,fr_bicyclic,fr_diazo,fr_dihydropyridine,fr_epoxide,fr_ester,fr_ether,fr_furan,fr_guanido,fr_halogen,fr_hdrzine,fr_hdrzone,fr_imidazole,fr_imide,fr_isocyan,fr_isothiocyan,fr_ketone,fr_ketone_Topliss,fr_lactam,fr_lactone,fr_methoxy,fr_morpholine,fr_nitrile,fr_nitro,fr_nitro_arom,fr_nitro_arom_nonortho,fr_nitroso,fr_oxazole,fr_oxime,fr_para_hydroxylation,fr_phenol,fr_phenol_noOrthoHbond,fr_phos_acid,fr_phos_ester,fr_piperdine,fr_piperzine,fr_priamide,fr_prisulfonamd,fr_pyridine,fr_quatN,fr_sulfide,fr_sulfonamd,fr_sulfone,fr_term_acetylene,fr_tetrazole,fr_thiazole,fr_thiocyan,fr_thiophene,fr_unbrch_alkane,fr_urea,Chi0,Chi0v,Chi0n,Chi1,Chi1v,Chi1n,Chi2v,Chi2n,Chi3v,Chi3n,Chi4v,Chi4n,EState_VSA1,EState_VSA2,EState_VSA3,EState_VSA4,EState_VSA5,EState_VSA6,EState_VSA7,EState_VSA8,EState_VSA9,EState_VSA10,EState_VSA11,FpDensityMorgan1,FpDensityMorgan2,FpDensityMorgan3,Kappa1,Kappa2,Kappa3,MolLogP,MolMR,MolWt,NumAliphaticCarbocycles,NumAliphaticHeterocycles,NumAliphaticRings,NumAromaticCarbocycles,NumAromaticHeterocycles,NumAromaticRings,NumHAcceptors,NumHDonors,NumHeteroatoms,NumRadicalElectrons,NumRotatableBonds,NumSaturatedCarbocycles,NumSaturatedHeterocycles,NumSaturatedRings,NumValenceElectrons,NPR1,NPR2,PMI1,PMI2,PMI3,PEOE_VSA1,PEOE_VSA2,PEOE_VSA3,PEOE_VSA4,PEOE_VSA5,PEOE_VSA6,PEOE_VSA7,PEOE_VSA8,PEOE_VSA9,PEOE_VSA10,PEOE_VSA11,PEOE_VSA12,PEOE_VSA13,PEOE_VSA14,SMR_VSA1,SMR_VSA2,SMR_VSA3,SMR_VSA4,SMR_VSA5,SMR_VSA6,SMR_VSA7,SMR_VSA8,SMR_VSA9,SMR_VSA10,SlogP_VSA1,SlogP_VSA2,SlogP_VSA3,SlogP_VSA4,SlogP_VSA5,SlogP_VSA6,SlogP_VSA7,SlogP_VSA8,SlogP_VSA9,SlogP_VSA10,SlogP_VSA11,SlogP_VSA12,VSA_EState1,VSA_EState2,VSA_EState3,VSA_EState4,VSA_EState5,VSA_EState6,VSA_EState7,VSA_EState8,VSA_EState9,VSA_EState10,BalabanJ,BertzCT,ExactMolWt,FractionCSP3,HallKierAlpha,HeavyAtomMolWt,HeavyAtomCount,LabuteASA,TPSA,MaxAbsEStateIndex,MaxEStateIndex,MinAbsEStateIndex,MinEStateIndex,MaxAbsPartialCharge,MaxPartialCharge,MinAbsPartialCharge,MinPartialCharge,qed,RingCount,Asphericity,Eccentricity,InertialShapeFactor,RadiusOfGyration,SpherocityIndex,NHOHCount,NOCount,db,experimental,type

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


# Note: set_argparse() function removed - functionality moved to bootstrap()


def set_log_level(level: ocerror.ReportLevel) -> None:
    '''Set the log level.

    Parameters
    ----------
    level : ocerror.ReportLevel
        The level of the log.
    '''

    ocerror.Error.set_output_level(ocerror.ReportLevel.WARNING)



###############################################################################

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

# Import order dictionary from Constants module
from OCDocker.Toolbox.Constants import order

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



bootstrapped = False

# Sync star-import consumers
_SYNC_SKIP_NAMES = {
    "__annotations__", "__builtins__", "__cached__", "__doc__", "__file__", "__loader__", "__name__", "__package__", "__spec__"
}


def _sync_import_consumers() -> None:
    '''Push updated globals to caller modules that pulled names via star-import.
    '''

    # Inspect the current frame to find the caller
    frame = inspect.currentframe()

    # If frame is None, return
    if not frame:
        return
    try:
        # Get the parent frame
        parent = frame.f_back

        # If parent frame is None, return
        if not parent:
            return
    
        # Get the module name of the parent frame
        module_name = parent.f_globals.get("__name__")

        # If module name is invalid, return
        if not isinstance(module_name, str) or module_name in ("builtins", __name__):
            return
        
        # Push public items to the parent frame's globals
        public_items = {
            name: value
            for name, value in globals().items()
            if not name.startswith("_") and name not in _SYNC_SKIP_NAMES
        }

        # Find shared names
        shared = set(parent.f_globals.keys()).intersection(public_items.keys())

        # Update parent frame globals
        for name in shared:
            parent.f_globals[name] = public_items[name]
    finally:
        # Cleanup to avoid reference cycles
        del frame


# Initialise
###############################################################################
def print_description() -> None:
    ''' Print the description of the program.
    '''

    print(_description)


def cleanup_database_resources() -> None:
    '''Clean up database resources (sessions and engines) on shutdown.
    
    This function is automatically registered with atexit to ensure proper cleanup
    of database connections when the application exits.
    '''
    global session, engine
    try:
        if 'session' in globals() and session is not None:
            cleanup_session(session)
        if 'engine' in globals() and engine is not None:
            cleanup_engine(engine)
    except Exception:
        # Silently ignore errors during shutdown cleanup
        pass


def _register_db_cleanup() -> None:
    '''Register database cleanup handlers with atexit.
    
    This ensures that database sessions and engines are properly closed
    when the application exits, preventing connection leaks.
    '''
    atexit.register(cleanup_database_resources)


def _parse_config_file(config_file: str) -> Dict[str, Any]:
    '''Parse OCDocker configuration file using configparser.
    
    This function replaces the manual string parsing with a more robust
    approach using Python's configparser module. It handles:
    - Type conversion (int, bool, list)
    - Default values
    - Error handling
    - Comments and empty lines
    
    Parameters
    ----------
    config_file : str
        Path to the configuration file
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all configuration values
        
    Raises
    ------
    SystemExit
        If the configuration file cannot be read or parsed
    '''
    
    # Allow duplicate keys to maintain compatibility with legacy configs that
    # may define the same option multiple times (last one wins).
    config = configparser.ConfigParser(strict=False)
    
    # Read config file - configparser can handle files without sections
    # by using the DEFAULT section
    try:
        # Read the file as a single section (DEFAULT)
        with open(config_file, 'r') as f:
            # Prepend [DEFAULT] to make it a valid INI file
            config_content = '[DEFAULT]\n' + f.read()
        
        config.read_string(config_content)
    except (OSError, IOError, configparser.Error) as e:
        print(f"{clrs['r']}ERROR{clrs['n']}: Failed to read configuration file '{config_file}': {e}")
        raise SystemExit(2)
    
    # Helper function to get config values with type conversion
    def get_config(key: str, default: Any = "", value_type: type = str) -> Any:
        '''Get configuration value with type conversion.
        
        Parameters
        ----------
        key : str
            Configuration key name
        default : Any
            Default value if key is not found
        value_type : type
            Type to convert the value to (str, int, float, bool, list)
            
        Returns
        -------
        Any
            Configuration value converted to the specified type
        '''
        
        try:
            value = config.get('DEFAULT', key, fallback=default)
            
            # Handle empty strings
            if not value or value.strip() == "":
                return default
            
            # Type conversion
            if value_type == int:
                return int(value.strip())
            elif value_type == float:
                return float(value.strip())
            elif value_type == bool:
                val_lower = value.strip().lower()
                return val_lower in ('1', 'true', 'yes', 'y', 'on')
            elif value_type == list:
                # Split by comma and strip whitespace
                return [item.strip() for item in value.split(',') if item.strip()]
            else:
                return value.strip()
        except (ValueError, configparser.NoOptionError) as e:
            # If conversion fails, return default
            # Use a try-except to handle cases where output_level might not be set yet
            try:
                if output_level >= ocerror.ReportLevel.WARNING:
                    print(f"{clrs['y']}WARNING{clrs['n']}: Invalid value for '{key}', using default: {default}")
            except (NameError, AttributeError):
                # output_level not set yet, skip warning
                pass
            return default
    
    # Parse all configuration values
    config_dict = {
        # Database settings
        'HOST': get_config('HOST', ''),
        'USER': get_config('USER', ''),
        'PASSWORD': get_config('PASSWORD', ''),
        'DATABASE': get_config('DATABASE', ''),
        'OPTIMIZEDB': get_config('OPTIMIZEDB', ''),
        'PORT': get_config('PORT', None, int),
        'USE_SQLITE': get_config('USE_SQLITE', ''),
        'SQLITE_PATH': get_config('SQLITE_PATH', ''),
        
        # General paths
        'ocdb': get_config('ocdb', ''),
        'pca': get_config('pca', ''),
        'pdbbind_KdKi_order': get_config('pdbbind_KdKi_order', 'u'),
        'reference_column_order': get_config('reference_column_order', [], list),
        
        # MGLTools
        'pythonsh': get_config('pythonsh', 'pythonsh'),
        'prepare_ligand': get_config('prepare_ligand', 'prepare_ligand4.py'),
        'prepare_receptor': get_config('prepare_receptor', 'prepare_receptor4.py'),
        
        # Vina
        'vina': get_config('vina', 'vina'),
        'vina_split': get_config('vina_split', 'vina_split'),
        'vina_energy_range': get_config('vina_energy_range', '10'),
        'vina_exhaustiveness': get_config('vina_exhaustiveness', 5, int),
        'vina_num_modes': get_config('vina_num_modes', '3'),
        'vina_scoring': get_config('vina_scoring', 'vina'),
        'vina_scoring_functions': get_config('vina_scoring_functions', ['vina'], list),
        
        # Smina
        'smina': get_config('smina', 'smina'),
        'smina_energy_range': get_config('smina_energy_range', '10'),
        'smina_exhaustiveness': get_config('smina_exhaustiveness', '5'),
        'smina_num_modes': get_config('smina_num_modes', '3'),
        'smina_scoring': get_config('smina_scoring', 'vinardo'),
        'smina_scoring_functions': get_config('smina_scoring_functions', ['vinardo'], list),
        'smina_custom_scoring': get_config('smina_custom_scoring', 'no'),
        'smina_custom_atoms': get_config('smina_custom_atoms', 'no'),
        'smina_local_only': get_config('smina_local_only', 'no'),
        'smina_minimize': get_config('smina_minimize', 'no'),
        'smina_randomize_only': get_config('smina_randomize_only', 'no'),
        'smina_minimize_iters': get_config('smina_minimize_iters', '0'),
        'smina_accurate_line': get_config('smina_accurate_line', 'no'),
        'smina_minimize_early_term': get_config('smina_minimize_early_term', 'no'),
        'smina_approximation': get_config('smina_approximation', 'spline'),
        'smina_factor': get_config('smina_factor', '32'),
        'smina_force_cap': get_config('smina_force_cap', '10'),
        'smina_user_grid': get_config('smina_user_grid', 'no'),
        'smina_user_grid_lambda': get_config('smina_user_grid_lambda', 'no'),
        
        # Gnina
        'gnina': get_config('gnina', 'gnina'),
        'gnina_exhaustiveness': get_config('gnina_exhaustiveness', ''),
        'gnina_num_modes': get_config('gnina_num_modes', ''),
        'gnina_scoring': get_config('gnina_scoring', ''),
        'gnina_custom_scoring': get_config('gnina_custom_scoring', ''),
        'gnina_custom_atoms': get_config('gnina_custom_atoms', ''),
        'gnina_local_only': get_config('gnina_local_only', ''),
        'gnina_minimize': get_config('gnina_minimize', ''),
        'gnina_randomize_only': get_config('gnina_randomize_only', ''),
        'gnina_num_mc_steps': get_config('gnina_num_mc_steps', ''),
        'gnina_max_mc_steps': get_config('gnina_max_mc_steps', ''),
        'gnina_num_mc_saved': get_config('gnina_num_mc_saved', ''),
        'gnina_minimize_iters': get_config('gnina_minimize_iters', ''),
        'gnina_simple_ascent': get_config('gnina_simple_ascent', ''),
        'gnina_accurate_line': get_config('gnina_accurate_line', ''),
        'gnina_minimize_early_term': get_config('gnina_minimize_early_term', ''),
        'gnina_approximation': get_config('gnina_approximation', ''),
        'gnina_factor': get_config('gnina_factor', ''),
        'gnina_force_cap': get_config('gnina_force_cap', ''),
        'gnina_user_grid': get_config('gnina_user_grid', ''),
        'gnina_user_grid_lambda': get_config('gnina_user_grid_lambda', ''),
        'gnina_no_gpu': get_config('gnina_no_gpu', ''),
        
        # PLANTS
        'plants': get_config('plants', 'plants'),
        'plants_cluster_structures': get_config('plants_cluster_structures', 3, int),
        'plants_cluster_rmsd': get_config('plants_cluster_rmsd', '2.0'),
        'plants_search_speed': get_config('plants_search_speed', 'speed1'),
        'plants_scoring': get_config('plants_scoring', 'chemplp'),
        'plants_scoring_functions': get_config('plants_scoring_functions', ['chemplp'], list),
        'plants_rescoring_mode': get_config('plants_rescoring_mode', 'simplex'),
        
        # Dock6
        'dock6': get_config('dock6', ''),
        'dock6_vdw_defn_file': get_config('dock6_vdw_defn_file', ''),
        'dock6_flex_defn_file': get_config('dock6_flex_defn_file', ''),
        'dock6_flex_drive_file': get_config('dock6_flex_drive_file', ''),
        
        # LeDock
        'ledock': get_config('ledock', ''),
        'lepro': get_config('lepro', ''),
        'ledock_rmsd': get_config('ledock_rmsd', ''),
        'ledock_num_poses': get_config('ledock_num_poses', ''),
        
        # ODDT
        'oddt': get_config('oddt', ''),
        'oddt_seed': get_config('oddt_seed', ''),
        'oddt_chunk_size': get_config('oddt_chunk_size', ''),
        'oddt_scoring_functions': get_config('oddt_scoring_functions', [], list),
        
        # Other software
        'chimera': get_config('chimera', ''),
        'dssp': get_config('dssp', 'dssp'),
        'obabel': get_config('obabel', 'obabel'),
        'spores': get_config('spores', 'spores'),
        'DUDEz': get_config('DUDEz', ''),
    }
    
    # Validate PORT if provided (already converted to int by get_config, but double-check)
    if config_dict['PORT'] is not None and not isinstance(config_dict['PORT'], int):
        try:
            config_dict['PORT'] = int(config_dict['PORT'])
        except (ValueError, TypeError):
            print(f"{clrs['r']}ERROR{clrs['n']}: The port number must be an integer.")
            raise SystemExit(2)
    
    return config_dict


def bootstrap(ns: Optional[argparse.Namespace] = None) -> None:
    """Explicitly bootstrap OCDocker environment (config, DB, paths).

    Must be called before using modules that depend on Initialise globals.
    """
    global bootstrapped
    if bootstrapped:
        return

    # Resolve arguments
    if ns is None:
        try:
            ns = argument_parsing()
        except SystemExit as e:
            print(f"{clrs['r']}ERROR{clrs['n']}: Invalid command line arguments.")
            raise

    # Expose key settings (runtime globals for CLI access)
    global args, update, config_file, output_level, overwrite
    args = ns
    update = bool(getattr(ns, 'update', False))
    config_file = getattr(ns, 'config_file', None) or os.getenv('OCDOCKER_CONFIG', 'OCDocker.cfg')

    # Ensure output_level is ALWAYS a ReportLevel enum
    raw_level = getattr(ns, 'output_level', ocerror.ReportLevel.WARNING)
    try:
        output_level = raw_level if isinstance(raw_level, ocerror.ReportLevel) else ocerror.ReportLevel(int(raw_level))
    except (ValueError, TypeError, AttributeError):
        # Fallback to WARNING if conversion fails
        output_level = ocerror.ReportLevel.WARNING

    overwrite = bool(getattr(ns, 'overwrite', False))
    ocerror.Error.set_output_level(output_level)

    # Locate configuration file and convert to absolute path
    # Try multiple locations if file not found
    if config_file and os.path.isfile(config_file):
        # File exists at specified path, convert to absolute
        config_file = os.path.abspath(config_file)
    elif os.path.isfile("OCDocker.cfg"):
        # Found in current directory
        config_file = os.path.abspath("OCDocker.cfg")
    else:
        # Try to find in common locations
        _module_dir = os.path.dirname(os.path.abspath(__file__))
        _package_root = os.path.dirname(os.path.dirname(_module_dir))  # Go up from OCDocker/Initialise.py to project root
        possible_paths = [
            os.path.join(_package_root, "OCDocker.cfg"),  # Project root
            os.path.join(_module_dir, "..", "..", "OCDocker.cfg"),  # Alternative path
        ]
        
        found = False
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.isfile(abs_path):
                config_file = abs_path
                found = True
                break
        
        if not found:
            print(f"{clrs['r']}ERROR{clrs['n']}: OCDocker configuration file not found.")
            print(f"  Searched in: current directory, {_package_root}")
            print(f"  Set OCDOCKER_CONFIG environment variable to specify the config file path.")
            raise SystemExit(2)

    # Splash
    print_description()

    # Create and set Config object first
    from OCDocker.Config import OCDockerConfig, set_config
    config = OCDockerConfig.from_config_file(config_file)
    
    # Update config with command-line arguments
    config.output_level = output_level
    config.multiprocess = bool(getattr(ns, 'multiprocess', True))
    config.overwrite = overwrite
    
    # Determine DB backend (MySQL default; optional SQLite fallback)
    use_sqlite_env = str(os.getenv('OCDOCKER_USE_SQLITE', '')).lower() in ('1', 'true', 'yes', 'y')
    use_sqlite_cfg = str(config.database.use_sqlite).lower() in ('1', 'true', 'yes', 'y', 'on', 'sqlite') if config.database.use_sqlite else False
    use_sqlite = use_sqlite_env or use_sqlite_cfg

    # Build DB URLs and connections
    global db_url, optdb_url, engine, session
    if use_sqlite:
        _module_dir = os.path.dirname(os.path.abspath(__file__))
        # Env var takes precedence, then config, then default path
        sqlite_path_env = os.getenv('OCDOCKER_SQLITE_PATH', '').strip()
        if sqlite_path_env:
            sqlite_path = sqlite_path_env
        elif config.database.sqlite_path:
            sqlite_path = config.database.sqlite_path
        else:
            sqlite_path = os.path.join(_module_dir, 'ocdocker.db')
        db_url = URL.create(drivername='sqlite', database=sqlite_path)
        optdb_url = db_url
        # Warn user about SQLite limitations
        try:
            print(f"{clrs['y']}WARNING{clrs['n']}: SQLite backend enabled. This is suitable for development/tests only. For performance and concurrency, a full MySQL installation is strongly recommended.")
            print(f"{clrs['c']}INFO{clrs['n']}: To use MySQL, unset OCDOCKER_USE_SQLITE (or set USE_SQLITE = no in your OCDocker.cfg) and configure HOST/USER/PASSWORD/DATABASE/PORT.")
        except (OSError, IOError, BrokenPipeError):
            # Ignore stdout errors (e.g., when output is redirected to a broken pipe)
            pass
    else:
        # Ensure DB settings exist (MySQL mode)
        if not config.database.host or not config.database.user or not config.database.password or not config.database.database or not config.database.port:
            print(f"{clrs['r']}ERROR{clrs['n']}: The variables HOST, USER, PASSWORD, DATABASE and PORT must be set in the config file '{config_file}'")
            raise SystemExit(2)
        db_url = URL.create(
            drivername='mysql+pymysql', 
            host=config.database.host, 
            username=config.database.user, 
            password=config.database.password, 
            database=config.database.database, 
            port=config.database.port  # type: ignore
        )
        optdb_url = URL.create(
            drivername='mysql+pymysql', 
            host=config.database.host, 
            username=config.database.user, 
            password=config.database.password, 
            database=config.database.optimizedb, 
            port=config.database.port  # type: ignore
        )

    engine = create_engine(db_url)
    create_database_if_not_exists(engine.url)
    create_database_if_not_exists(optdb_url)
    session = create_session(engine)

    # Paths and dirs (runtime values - stored in config only, no globals)
    ocdocker_path = os.path.dirname(os.path.abspath(__file__))
    if not config.paths.ocdb_path:
        print(f"{clrs['r']}ERROR{clrs['n']}: The variable ocdb_path is not set in the config file '{config_file}'")
        raise SystemExit(2)

    dudez_archive = os.path.join(config.paths.ocdb_path, "DUDEz")
    pdbbind_archive = os.path.join(config.paths.ocdb_path, "PDBbind")
    parsed_archive = os.path.join(config.paths.ocdb_path, "Parsed")
    logdir = f"{os.path.abspath(os.path.join(os.path.dirname(ocerror.__file__), os.pardir))}/logs"
    oddt_models_dir = f"{os.path.abspath(os.path.join(os.path.dirname(ocerror.__file__), os.pardir))}/ODDT_models"
    for d in (logdir, oddt_models_dir, config.paths.pca_path):
        if d and not os.path.isdir(d):
            os.mkdir(d)

    # Reset tmp dir
    tmpDir = f"{ocdocker_path}/tmp"
    try:
        if os.path.isdir(tmpDir):
            shutil.rmtree(tmpDir)
        os.mkdir(tmpDir)
    except (OSError, PermissionError, shutil.Error):
        # Ignore errors during tmp directory cleanup/creation (non-critical)
        pass

    # CPU cores (stored in config only, no global)
    if config.multiprocess:
        n_cpu = multiprocessing.cpu_count() - 1
        available_cores = n_cpu if n_cpu > 1 else 1
    else:
        available_cores = 1

    # Clamp output level
    if config.output_level.value > ocerror.ReportLevel.DEBUG.value:
        config.output_level = ocerror.ReportLevel.DEBUG
    elif config.output_level.value < ocerror.ReportLevel.NONE.value:
        config.output_level = ocerror.ReportLevel.NONE
    ocerror.Error.set_output_level(config.output_level)

    # Update config with computed runtime paths
    config.tmp_dir = tmpDir
    config.ocdocker_path = ocdocker_path
    config.dudez_archive = dudez_archive
    config.pdbbind_archive = pdbbind_archive
    config.parsed_archive = parsed_archive
    config.logdir = logdir
    config.oddt_models_dir = oddt_models_dir
    config.available_cores = available_cores
    
    # Set as global config
    set_config(config)

    # Ensure ODDT models folder contents (allow skipping for slim environments)
    if not str(os.getenv('OCDOCKER_SKIP_ODDT', '')).lower() in ('1','true','yes','y'):
        initialise_oddt_models(config.oddt_models_dir, config.oddt.scoring_functions)

    # Note: _sync_import_consumers() exists but is not called - no longer needed since we've eliminated star imports
    # If any legacy code still uses star imports, uncomment the line below:
    # _sync_import_consumers()
    
    # Register cleanup handlers for database connections
    _register_db_cleanup()
    
    bootstrapped = True

# Autobootstrap on first import (non‑CLI contexts), unless disabled via env
try:
    # Provide harmless defaults when building docs (or tests)
    if os.getenv("OC_BUILD_DOCS") == "1":
        if "session" not in globals() and MagicMock:
            session = MagicMock(name="session")
        if "db_url" not in globals():
            db_url = "sqlite:///:memory:"

    if not bootstrapped and not is_doc_build() and not os.getenv('OCDOCKER_NO_AUTO_BOOTSTRAP'):
        default_ns = argparse.Namespace(
            multiprocess=True,
            update=False,
            config_file=os.getenv('OCDOCKER_CONFIG', 'OCDocker.cfg'),
            output_level=ocerror.ReportLevel.WARNING,
            overwrite=False,
        )
        bootstrap(default_ns)
except SystemExit:
    # Propagate the failure semantics as before: importing without a valid config exits
    raise
