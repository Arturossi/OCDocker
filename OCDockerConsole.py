#!/usr/lib/python3

# Imports
###############################################################################
import textwrap as tw
from pprint import pprint

from OCDocker.Initialise import *

args.output_level = 0

import OCDocker.Toolbox as octools

import OCDocker.Database as ocdb
import OCDocker.baseDB as ocbdb
import OCDocker.DUDEz as ocdudez
import OCDocker.PDBbind as ocpdbbind
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Gnina as ocgnina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.ExternalTools.runprank as runprank

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
'''
This script is used for fast import of all funcionalities in the OCDocker suite
making easier to debug and possibly allowing a future OCDocker console to enable
the user to perform the steps step by step.
'''

# Classes
###############################################################################


# Functions
###############################################################################
def print_args() -> None:
    '''Prints the current args variable

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    print("args:")
    # Iterate over vars(args) dict
    for key, value in vars(args).items():
        print(f"{key}:\t\t{value}")
    print("\n")

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

args.cpu_cores = 18
args.available_cores = args.cpu_cores - 1
args.multiprocess = 1
args.generate_report = False
args.zip_output = False
args.update = False

'''
basePath = f"/data/hd4tb/OCDocker/OCDocker/test_files"
ptn = "test_ptn1"
lig = "ligand"
baseProtPath = f"{basePath}/{ptn}"
baseLigPath = f"{baseProtPath}/compounds/ligands"
baseDecPath = f"{baseProtPath}/compounds/decoys"
baseCanPath = f"{baseProtPath}/compounds/candidates"

receptorTest = ocr.Receptor(f"{baseProtPath}/receptor.pdb", relativeASAcutoff=0.7, name=f"{ptn}")
ligandTest = ocl.Ligand(f"{baseLigPath}/{lig}/ligand.mol2", name="{lig}")

gninaTest = ocgnina.Gnina(f"{baseLigPath}/{lig}/gninaFiles/conf_gnina.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.pdbqt", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.pdbqt", f"{baseLigPath}/{lig}/gninaFiles/gnina_0.log", f"{baseLigPath}/{lig}/gninaFiles/gnina_0.pdbqt", name=f"Gnina {ptn}-{lig}")
vinaTest = ocvina.Vina(f"{baseLigPath}/{lig}/vinaFiles/conf_vina.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.pdbqt", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.pdbqt", f"{baseLigPath}/{lig}/vinaFiles/vina_0.log", f"{baseLigPath}/{lig}/vinaFiles/vina_0.pdbqt", name=f"Vina {ptn}-{lig}")
sminaTest = ocsmina.Smina(f"{baseLigPath}/{lig}/sminaFiles/conf_smina.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.pdbqt", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.pdbqt", f"{baseLigPath}/{lig}/sminaFiles/smina_0.log", f"{baseLigPath}/{lig}/sminaFiles/smina_0.pdbqt", name=f"Smina {ptn}-{lig}")
plantsTest = ocplants.PLANTS(f"{baseLigPath}/{lig}/plantsFiles/conf_plants.txt", f"{baseLigPath}/{lig}/boxes/box0.pdb", receptorTest, f"{baseProtPath}/prepared_receptor.mol2", ligandTest, f"{baseLigPath}/{lig}/prepared_ligand.mol2", f"{baseLigPath}/{lig}/plantsFiles/plants_0.log", f"{baseLigPath}/{lig}/plantsFiles/", name=f"PLANTS {ptn}-{lig}")
'''