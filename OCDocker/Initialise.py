#!/usr/lib/python3

# Description
###############################################################################
'''
Sets of primordial variables and functions that are used to initialise the
OCDocker library.\n
All scripts that use OCDocker must import this file.

They are imported as:

from OCDocker.Initialise import *
'''

# Imports
###############################################################################
import argparse
import multiprocessing
import os
import shutil

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

# Splash, version & clear tmp
###############################################################################
ocVersion = "0.8.0"

description = tw.dedent("""\033[1;93m
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
def create_ocdocker_conf() -> None:
    '''Creates the 'ocdocker.conf' file.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None
    '''

    #region General config
    confOcdb = "/mnt/e/Documents/OCDocker/OCDocker/data/ocdb"
    confPDBbind_KdKi_order = "u"

    print("\nGeneral OCDocker configuration")
    answer = input(f"Path to the OCDB. Default [{confOcdb}] (press enter to keep default): ")
    confOcdb = confOcdb if not answer else answer

    # Ensure that the answer is valid (reset its value to an known invalid value before checking)
    answer = ""
    while answer not in ["Y", "Z", "E", "P", "T", "G", "M", "k", "un", "c", "m", "u", "n", "pf", "a", "z", "y"]:
        answer = input(f"The default pdbbind KiKd magnitude [Y, Z, E, P, T, G, M, k, un, c, m, u, n, pf, a, z, y] (follow the unit prefix table). Default [{confPDBbind_KdKi_order}] (press enter to keep default): ")
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

    #region P2rank config
    confPrank = "/mnt/e/Documents/OCDocker/software/search/p2rank_2.3/prank"
    confP2rankBoxMaxCutoff = "0.5"
    confP2RankPocketCutoff = "0.1"

    print("\np2rank configuration")
    answer = input(f"Path to the p2rank software. Default [{confPrank}] (press enter to keep default): ")
    confPrank = confPrank if not answer else answer

    answer = input(f"p2rank box max cutoff. Default [{confP2rankBoxMaxCutoff}] (press enter to keep default): ")
    confP2rankBoxMaxCutoff = confP2rankBoxMaxCutoff if not answer else answer

    answer = input(f"p2rank pocket cutoff. Default [{confP2RankPocketCutoff}] (press enter to keep default): ")
    confP2RankPocketCutoff = confP2RankPocketCutoff if not answer else answer

    #endregion

    #region Vina config
    confVina = "/usr/bin/vina"
    confVina = "/usr/bin/vina_split"
    confVina_split = "/usr/bin/vina_split"
    confVina_energy_range = "10"
    confVina_exhaustiveness = "5"
    confVina_num_modes = "3"
    confVina_scoring = "vina"
    confVina_scoring_functions = "ad4,vina,vinardo"


    print("\nVina configuration")
    answer = input(f"Path to the Vina software. Default [{confVina}] (press enter to keep default): ")
    confVina = confVina if not answer else answer

    answer = input(f"Vina energy parameter. Default [{confVina_energy_range}] (press enter to keep default): ")
    confVina_energy_range = confVina_energy_range if not answer else answer

    answer = input(f"Vina exhaustiveness parameter. Default [{confVina_exhaustiveness}] (press enter to keep default): ")
    confVina_exhaustiveness = confVina_exhaustiveness if not answer else answer

    answer = input(f"Vina num modes parameter. Default [{confVina_num_modes}] (press enter to keep default): ")
    confVina_num_modes = confVina_num_modes if not answer else answer

    answer = input(f"Vina scoring function. Default [{confVina_scoring}] (press enter to keep default): ")
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
    confSmina_scoring = "vina"
    confSmina_scoring_functions = "vina,vinardo,dkoes_scoring,dkoes_scoring_old,dkoes_fast,ad4_scoring"

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
    confPlants_cluster_structures = 10
    confPlants_cluster_rmsd = 2.0
    confPlants_search_speed = "speed1"

    print("\nPLANTS configuration")
    answer = input(f"Path to the Plants software. Default [{confPlants}] (press enter to keep default): ")
    confPlants = confPlants if not answer else answer

    answer = input(f"How many structures will be generated. Default [{confPlants_cluster_structures}] (press enter to keep default): ")
    confPlants_cluster_structures = confPlants_cluster_structures if not answer else answer

    answer = input(f"PLANTS cluster RMSD parameter. Default [{confPlants_cluster_rmsd}] (press enter to keep default): ")
    confPlants_cluster_rmsd = confPlants_cluster_rmsd if not answer else answer

    answer = input(f"PLANTS search speed parameter. Default [{confPlants_search_speed}] (press enter to keep default): ")
    confPlants_search_speed = confPlants_search_speed if not answer else answer

    #endregion

    #region DOCK6 variables
    confDock6 = "/mnt/e/Documents/OCDocker/software/docking/dock6/bin/dock6"
    confDock6_vdw_defn_file = "/mnt/e/Documents/OCDocker/software/docking/dock6/vdw_AMBER_parm99.defn"
    confDock6_flex_defn_file = "/mnt/e/Documents/OCDocker/software/docking/dock6/flex.defn"
    confDock6_flex_drive_file = "/mnt/e/Documents/OCDocker/software/docking/dock6/flex_drive.tbl"

    #print("\nVina configuration")
    answer = input(f"Path to the DOCK6 software. Default [{confDock6}] (press enter to keep default): ")
    confDock6 = confDock6 if not answer else answer

    answer = input(f"DOCK6 vdw_defn file path. Default [{confDock6_vdw_defn_file}] (press enter to keep default): ")
    confDock6_vdw_defn_file = confDock6_vdw_defn_file if not answer else answer

    answer = input(f"DOCK6 flex_defn file path. Default [{confDock6_flex_defn_file}] (press enter to keep default): ")
    confDock6_flex_defn_file = confDock6_flex_defn_file if not answer else answer

    answer = input(f"DOCK6 flex_drive file path. Default [{confDock6_flex_drive_file}] (press enter to keep default): ")
    confDock6_flex_drive_file = confDock6_flex_drive_file if not answer else answer

    #endregion

    #region Other variables
    confDssp = "/usr/bin/dssp"
    confObabel = "/usr/bin/obabel"
    confSpores = "/mnt/e/Documents/OCDocker/software/docking/plants/SPORES_64bit"
    confDUDEz = "https://dudez.docking.org/DOCKING_GRIDS_AND_POSES.tgz" # this is WRONG

    print("\nOther software configuration")
    answer = input(f"Path to the dssp file/command. Default [{confDssp}] (press enter to keep default): ")
    confDssp = confDssp if not answer else answer

    answer = input(f"Path to the obabel software. Default [{confObabel}] (press enter to keep default): ")
    confObabel = confObabel if not answer else answer

    answer = input(f"Path to the SPORES software. Default [{confSpores}] (press enter to keep default): ")
    confSpores = confSpores if not answer else answer

    answer = input(f"Link to the DUDEz database where you can download data. Default [{confDUDEz}] (press enter to keep default): ")
    confDUDEz = confDUDEz if not answer else answer

    #endregion

    # Define the config file (NOT CHANGABLE)
    conf_file = "OCDocker.cfg"

    # Create the conf file
    with open(conf_file, 'w') as cf:
        cf.write(tw.dedent("""# Root directory for the OCDocker Database
        ocdb = """ + str(confOcdb) + """

        # The default pdbbind KiKd magnitude [Y, Z, E, P, T, G, M, k, un, c, m, u, n, pf, a, z, y] (follow the unit prefix table)
        pdbbind_KdKi_order = """ + str(confPDBbind_KdKi_order) + """

        ################# MGLTools PARAMETERS #################

        # MGLTools's pythonsh path
        pythonsh = """ + str(confPythonsh) + """

        # prepare_ligand4 path
        prepare_ligand = """ + str(confPrepare_ligand) + """

        # prepare_receptor4 path
        prepare_receptor = """ + str(confPrepare_receptor) + """

        ################# P2RANK PARAMETERS #################

        # P2Rank path
        prank = """ + str(confPrank) + """

        # p2rank box cutoff
        boxMaxCutoff = """ + str(confP2rankBoxMaxCutoff) + """

        # p2rank pocket cutoff
        pocketCutoff = """ + str(confP2RankPocketCutoff) + """

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

        ################# SMINA PARAMETERS ##################

        # Smina path
        smina = """ + str(confSmina) + """

        # Maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)
        smina_energy_range = """ + str(confSmina_energy_range) + """

        # Exhaustiveness of the global search
        smina_exhaustiveness = """ + str(confSmina_exhaustiveness) + """

        # Maximum number of binding modes to generate
        smina_num_modes = """ + str(confSmina_num_modes) + """

        # Alternative scoring function
        smina_scoring = """ + str(confSmina_scoring) + """

        # Dis

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

        ################## OTHER SOFTWARE ###################

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

# Define Global Variables
###############################################################################
# General variables
global args
global clrs
global widgets
global workdir
global errors
global logdir
global tmpdir

# Order variable
global order
global pdbbind_KdKi_order

# Data from .cfg
global ocdb_path
global vina
global vina_split
global dock6
global prank
global smina
global gnina
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

# Dock6 parameters
global dock6_vdw_defn_file
global dock6_flex_defn_file
global dock6_flex_drive_file

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
    "n": "\033[1;0m"   # default
    }

# This structure is to define which will be used order, the first index will be the default magnitude and the other is element magnitude [Y, Z, E, P, T, G, M, k, un, c, m, u, n, p, f, a, z, y]
order = {
    "Y": {
        "Y": 10e0, "Z": 10e-3, "E": 10e-6, "P": 10e-9, "T": 10e-12, "G": 10e-15, "M": 10e-18, "k": 10e-21, "un": 10e-24, "c": 10e-26, "m": 10e-27, "u": 10e-30, "n": 10e-33, "p": 10e-36, "f": 10e-39, "a": 10e-42, "z": 10e-45, "y": 10e-48
    },
    "Z": {
        "Y": 10e3, "Z": 10e0, "E": 10e-3, "P": 10e-6, "T": 10e-9, "G": 10e-12, "M": 10e-15, "k": 10e-18, "un": 10e-21, "c": 10e-23, "m": 10e-24, "u": 10e-27, "n": 10e-30, "p": 10e-33, "f": 10e-36, "a": 10e-39, "z": 10e-42, "y": 10e-45
    },
    "E": {
        "Y": 10e6, "Z": 10e3, "E": 10e0, "P": 10e-3, "T": 10e-6, "G": 10e-9, "M": 10e-12, "k": 10e-15, "un": 10e-18, "c": 10e-20, "m": 10e-21, "u": 10e-24, "n": 10e-27, "p": 10e-30, "f": 10e-33, "a": 10e-36, "z": 10e-39, "y": 10e-42
    },
    "P": {
        "Y": 10e9, "Z": 10e6, "E": 10e3, "P": 10e0, "T": 10e-3, "G": 10e-6, "M": 10e-9, "k": 10e-12, "un": 10e-15, "c": 10e-17, "m": 10e-18, "u": 10e-21, "n": 10e-24, "p": 10e-27, "f": 10e-30, "a": 10e-33, "z": 10e-36, "y": 10e-39
    },
    "T": {
        "Y": 10e12, "Z": 10e9, "E": 10e6, "P": 10e3, "T": 10e0, "G": 10e-3, "M": 10e-6, "k": 10e-9, "un": 10e-12, "c": 10e-14, "m": 10e-15, "u": 10e-18, "n": 10e-21, "p": 10e-24, "f": 10e-27, "a": 10e-30, "z": 10e-33, "y": 10e-34
    },
    "G": {
        "Y": 10e15, "Z": 10e12, "E": 10e9, "P": 10e6, "T": 10e3, "G": 10e0, "M": 10e-3, "k": 10e-6, "un": 10e-9, "c": 10e-11, "m": 10e-12, "u": 10e-15, "n": 10e-18, "p": 10e-21, "f": 10e-24, "a": 10e-27, "z": 10e-30, "y": 10e-33
    },
    "M": {
        "Y": 10e18, "Z": 10e18, "E": 10e12, "P": 10e9, "T": 10e6, "G": 10e3, "M": 10e0, "k": 10e-3, "un": 10e-6, "c": 10e-8, "m": 10e-9, "u": 10e-12, "n": 10e-15, "p": 10e-18, "f": 10e-21, "a": 10e-24, "z": 10e-27, "y": 10e-30
    },
    "k": {
        "Y": 10e21, "Z": 10e18, "E": 10e15, "P": 10e12, "T": 10e9, "G": 10e6, "M": 10e3, "k": 10e0, "un": 10e-3, "c": 10e-5, "m": 10e-6, "u": 10e-9, "n": 10e-12, "p": 10e-15, "f": 10e-18, "a": 10e-21, "z": 10e-24, "y": 10e-27
    },
    "un": {
        "Y": 10e24, "Z": 10e21, "E": 10e18, "P": 10e15, "T": 10e12, "G": 10e9, "M": 10e6, "k": 10e3, "un": 10e0, "c": 10e-2, "m": 10e-3, "u": 10e-6, "n": 10e-9, "p": 10e-12, "f": 10e-15, "a": 10e-18, "z": 10e-21, "y": 10e-24
    },
    "c": {
        "Y": 10e26, "Z": 10e23, "E": 10e20, "P": 10e17, "T": 10e14, "G": 10e11, "M": 10e8, "k": 10e5, "un": 10e2, "c": 10e0, "m": 10e-1, "u": 10e-4, "n": 10e-7, "p": 10e-10, "f": 10e-13, "a": 10e-16, "z": 10e-19, "y": 10e-22
    },
    "m": {
        "Y": 10e27, "Z": 10e24, "E": 10e21, "P": 10e18, "T": 10e15, "G": 10e12, "M": 10e9, "k": 10e6, "un": 10e3, "c": 10e1, "m": 10e0, "u": 10e-3, "n": 10e-6, "p": 10e-9, "f": 10e-12, "a": 10e-15, "z": 10e-18, "y": 10e-21
    },
    "u": {
        "Y": 10e30, "Z": 10e27, "E": 10e24, "P": 10e21, "T": 10e18, "G": 10e15, "M": 10e12, "k": 10e9, "un": 10e6, "c": 10e4, "m": 10e3, "u": 10e0, "n": 10e-3, "p": 10e-6, "f": 10e-9, "a": 10e-12, "z": 10e-15, "y": 10e-18
    },
    "n": {
        "Y": 10e33, "Z": 10e30, "E": 10e27, "P": 10e24, "T": 10e21, "G": 10e18, "M": 10e15, "k": 10e12, "un": 10e9, "c": 10e7, "m": 10e6, "u": 10e3, "n": 10e0, "p": 10e-3, "f": 10e-6, "a": 10e-9, "z": 10e-12, "y": 10e-15
    },
    "p": {
        "Y": 10e36, "Z": 10e33, "E": 10e30, "P": 10e27, "T": 10e24, "G": 10e21, "M": 10e18, "k": 10e15, "un": 10e12, "c": 10e10, "m": 10e9, "u": 10e6, "n": 10e3, "p": 10e0, "f": 10e-3, "a": 10e-6, "z": 10e-9, "y": 10e-12
    },
    "f": {
        "Y": 10e39, "Z": 10e36, "E": 10e33, "P": 10e30, "T": 10e27, "G": 10e24, "M": 10e21, "k": 10e18, "un": 10e15, "c": 10e13, "m": 10e12, "u": 10e9, "n": 10e6, "p": 10e3, "f": 10e0, "a": 10e-3, "z": 10e-6, "y": 10e-9
    },
    "a": {
        "Y": 10e42, "Z": 10e39, "E": 10e36, "P": 10e33, "T": 10e30, "G": 10e27, "M": 10e24, "k": 10e21, "un": 10e18, "c": 10e16, "m": 10e15, "u": 10e12, "n": 10e9, "p": 10e6, "f": 10e3, "a": 10e0, "z": 10e-3, "y": 10e-6
    },
    "z": {
        "Y": 10e45, "Z": 10e42, "E": 10e39, "P": 10e36, "T": 10e33, "G": 10e30, "M": 10e27, "k": 10e24, "un": 10e21, "c": 10e19, "m": 10e18, "u": 10e15, "n": 10e12, "p": 10e9, "f": 10e6, "a": 10e3, "z": 10e0, "y": 10e-3
    },
    "y": {
        "Y": 10e48, "Z": 10e45, "E": 10e42, "P": 10e39, "T": 10e36, "G": 10e33, "M": 10e30, "k": 10e27, "un": 10e24, "c": 10e22, "m": 10e21, "u": 10e18, "n": 10e15, "p": 10e12, "f": 10e9, "a": 10e6, "z": 10e3, "y": 10e0
    }
}

# Parse command line arguments
###############################################################################
def argument_parsing() -> argparse.Namespace:
    '''Get data to generate vina conf file from box file.
    
    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Namespace object containing the arguments.

    Raises
    ------
    None
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
    elif line.startswith("dock6 ="):
        dock6 = line.split("=")[1].strip()
    elif line.startswith("dock6_vdw_defn_file ="):
        dock6_vdw_defn_file = line.split("=")[1].strip()
    elif line.startswith("dock6_flex_defn_file ="):
        dock6_flex_defn_file = line.split("=")[1].strip()
    elif line.startswith("dock6_flex_drive_file ="):
        dock6_flex_drive_file = line.split("=")[1].strip()
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

# Check if logdir exists, if not, create-it
if not os.path.isdir(logdir):
    os.mkdir(logdir)

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
    
#TODO: Colocar uma lista de parâmetros do OCDocker
