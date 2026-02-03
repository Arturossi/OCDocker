#!/usr/bin/env python3

# Description
###############################################################################
'''
This script is used for fast import of all funcionalities in the OCDocker suite
making easier to debug and possibly allowing a future OCDocker console to enable
the user to perform the steps step by step.
'''

# Imports
###############################################################################
import inspect
import os
import shutil

import textwrap as tw

from glob import glob
from pprint import pprint

import OCDocker.Docking.Future.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Ligand as ocl
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust
import OCDocker.Receptor as ocr
import OCDocker.Rescoring.ODDT as ocoddt
import OCDocker.Toolbox as octools
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc

from OCDocker.Initialise import *

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##
def clean_test_files(baseProtPath: str, baseLigPath: str, baseDecPath: str, baseCanPath: str) -> None:
    '''Rests the test_files folder to its original state

    Parameters
    ----------
    baseProtPath : str
        Path to the base protein folder
    baseLigPath : str
        Path to the base ligand folder
    baseDecPath : str
        Path to the base decoy folder
    baseCanPath : str
        Path to the base candidates folder
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

    from OCDocker.Config import get_config
    
    # Get config object
    try:
        cfg = get_config()
    except Exception:
        cfg = None

    def _g(name: str, default: str = '-') -> str:
        '''Get value from globals (for runtime values like config_file, db_url, etc.)

        Parameters
        ----------
        name : str
            Name of the global variable
        default : str
            Default value if the variable is not found
            
        Returns
        -------
        str
            Value of the global variable
        '''

        return globals().get(name, default)
    
    def _c(attr_path: str, default: str = '-') -> str:
        '''Get value from config object using dot notation (e.g., 'vina.executable')

        Parameters
        ----------
        attr_path : str
            Path to the attribute in the config object
        default : str
            Default value if the attribute is not found
            
        Returns
        -------
        str
            Value of the attribute
        '''

        if cfg is None:
            return default
        try:
            obj = cfg
            for attr in attr_path.split('.'):
                obj = getattr(obj, attr, None)
                if obj is None:
                    return default
            return obj if obj != "" else default
        except (AttributeError, TypeError):
            return default

    def _p(label: str, value: str) -> None:
        '''Print the label and value

        Parameters
        ----------
        label : str
            Label to print
        value : str
            Value to print
        '''

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
        _p("multiprocess", _c('multiprocess', _g('multiprocess')))
        _p("update", _g('update'))
        ol = _g('output_level') or _c('output_level')
        try:
            if hasattr(ol, 'name'):
                ol_disp = ol.name
            elif hasattr(ol, 'value'):
                ol_disp = ocerror.ReportLevel(ol.value).name
            else:
                ol_disp = ol
        except Exception:
            ol_disp = ol
        _p("output_level", ol_disp)
        _p("overwrite", _c('overwrite', _g('overwrite')))

    if prog in ("paths",) or show_all:
        print("\n=== Key Paths ===")
        _p("ocdb_path", _c('paths.ocdb_path'))
        _p("pca_path", _c('paths.pca_path'))
        _p("logdir", _c('logdir'))
        _p("oddt_models_dir", _c('oddt_models_dir'))

        print("\n=== Docking Binaries ===")
        _p("vina", _c('vina.executable'))
        _p("smina", _c('smina.executable'))
        _p("plants", _c('plants.executable'))
        _p("gnina", _c('gnina.executable'))
        _p("obabel", _c('tools.obabel'))
        _p("pythonsh", _c('tools.pythonsh'))
        _p("prepare_ligand", _c('tools.prepare_ligand'))
        _p("prepare_receptor", _c('tools.prepare_receptor'))

    if prog in ("db",) or show_all:
        print("\n=== Database URLs ===")
        _p("db_url", _g('db_url'))
        _p("optdb_url", _g('optdb_url'))

    if prog in ("vina",) or show_all:
        print("\n=== Vina Parameters ===")
        _p("vina_scoring", _c('vina.scoring'))
        _p("vina_scoring_functions", _c('vina.scoring_functions'))
        _p("vina_num_modes", _c('vina.num_modes'))
        _p("vina_energy_range", _c('vina.energy_range'))
        _p("vina_exhaustiveness", _c('vina.exhaustiveness'))

    if prog in ("smina",) or show_all:
        print("\n=== Smina Parameters ===")
        _p("smina_scoring", _c('smina.scoring'))
        _p("smina_scoring_functions", _c('smina.scoring_functions'))
        _p("smina_num_modes", _c('smina.num_modes'))
        _p("smina_energy_range", _c('smina.energy_range'))
        _p("smina_exhaustiveness", _c('smina.exhaustiveness'))
        _p("smina_custom_scoring", _c('smina.custom_scoring'))
        _p("smina_custom_atoms", _c('smina.custom_atoms'))
        _p("smina_local_only", _c('smina.local_only'))
        _p("smina_minimize", _c('smina.minimize'))
        _p("smina_randomize_only", _c('smina.randomize_only'))
        _p("smina_minimize_iters", _c('smina.minimize_iters'))
        _p("smina_accurate_line", _c('smina.accurate_line'))
        _p("smina_minimize_early_term", _c('smina.minimize_early_term'))
        _p("smina_approximation", _c('smina.approximation'))
        _p("smina_factor", _c('smina.factor'))
        _p("smina_force_cap", _c('smina.force_cap'))
        _p("smina_user_grid", _c('smina.user_grid'))
        _p("smina_user_grid_lambda", _c('smina.user_grid_lambda'))

    if prog in ("plants",) or show_all:
        print("\n=== PLANTS Parameters ===")
        _p("plants_cluster_structures", _c('plants.cluster_structures'))
        _p("plants_cluster_rmsd", _c('plants.cluster_rmsd'))
        _p("plants_search_speed", _c('plants.search_speed'))
        _p("plants_scoring", _c('plants.scoring'))
        _p("plants_scoring_functions", _c('plants.scoring_functions'))

    if prog in ("gnina",) or show_all:
        print("\n=== Gnina Parameters ===")
        _p("gnina_exhaustiveness", _c('gnina.exhaustiveness'))
        _p("gnina_num_modes", _c('gnina.num_modes'))
        _p("gnina_scoring", _c('gnina.scoring'))
        _p("gnina_custom_scoring_file", _c('gnina.custom_scoring'))
        _p("gnina_custom_atoms", _c('gnina.custom_atoms'))
        _p("gnina_local_only", _c('gnina.local_only'))
        _p("gnina_minimize", _c('gnina.minimize'))
        _p("gnina_randomize_only", _c('gnina.randomize_only'))
        _p("gnina_num_mc_steps", _c('gnina.num_mc_steps'))
        _p("gnina_max_mc_steps", _c('gnina.max_mc_steps'))
        _p("gnina_num_mc_saved", _c('gnina.num_mc_saved'))
        _p("gnina_minimize_iters", _c('gnina.minimize_iters'))
        _p("gnina_simple_ascent", _c('gnina.simple_ascent'))
        _p("gnina_accurate_line", _c('gnina.accurate_line'))
        _p("gnina_minimize_early_term", _c('gnina.minimize_early_term'))
        _p("gnina_approximation", _c('gnina.approximation'))
        _p("gnina_factor", _c('gnina.factor'))
        _p("gnina_force_cap", _c('gnina.force_cap'))
        _p("gnina_user_grid", _c('gnina.user_grid'))
        _p("gnina_user_grid_lambda", _c('gnina.user_grid_lambda'))
        _p("gnina_no_gpu", _c('gnina.no_gpu'))

    if prog in ("oddt",) or show_all:
        print("\n=== ODDT Parameters ===")
        _p("oddt_program", _c('oddt.executable'))
        _p("oddt_seed", _c('oddt.seed'))
        _p("oddt_chunk_size", _c('oddt.chunk_size'))
        _p("oddt_scoring_functions", _c('oddt.scoring_functions'))

    if not prog and not show_all:
        print("")
    return None

# The environment variable OCDOCKER_CONFIG must be set to the OCDocker.cfg file before importing OCDocker
cfg_path = os.environ.get('OCDOCKER_CONFIG') or 'OCDocker.cfg'

output_level = ocerror.ReportLevel.NONE

message = tw.dedent(f"""{clrs["y"]}
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{clrs["c"]}
                              CONSOLE MODE{clrs["y"]}
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{clrs["n"]}
          Welcome to the OCDocker interactive console!

      This console allows you to interact with the OCDocker pipeline
      step by step.

      {clrs["g"]}TIP{clrs["n"]} It's an interesting way to learn OCDocker, useful
      for debugging, and great for quick API interaction and
      experimentation.

      To check the args variable use print_args() function.{clrs["y"]}
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{clrs["n"]}""")

print(message)

if __name__ == "__main__":
    # Set the variables based on args
    bootstrap(argument_parsing())
else:
    # CPU cores -2, and 1 if cpu has only one or two cores
    cpu_cores = os.cpu_count() - 2 if os.cpu_count() and os.cpu_count() > 2 else 1
    available_cores = cpu_cores - 1
    multiprocess = True
    update = False
