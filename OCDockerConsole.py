#!/usr/lib/python3

# Imports
###############################################################################
import textwrap as tw
from pprint import pprint

from OCDocker.Initialise import *

args.output_level = 0

import OCDocker.Toolbox as octools

import OCDocker.Vina as ocvina
import OCDocker.Smina as ocsmina
import OCDocker.Gnina as ocgnina
import OCDocker.PLANTS as ocplants
import OCDocker.Database as ocdb
import OCDocker.baseDB as ocbdb
import OCDocker.DUDEz as ocdudez
import OCDocker.PDBbind as ocpdbbind
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
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
    pprint(vars(args))
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

basePath = "/mnt/e/Documents/OCDocker/OCDocker"
basePath = "/mnt/sda/artur/OCDocker"
basePath = "/data/hd4tb/OCDocker/OCDocker"

'''
dbsampledir = f"{basePath}/data/ocdb/DUDEz/AA2AR"

import pickle
ligands = []
with (open("ligands.pickle", "rb")) as openfile:
    while True:
        try:
            ligands = pickle.load(openfile)
        except EOFError:
            break
arguments = []
for i in range(len(ligands)):
    innerToCompare = []
    for j in range(i + 1, len(ligands)):
        innerToCompare.append(ligands[j])
    if innerToCompare:
        arguments.append((ligands[i], innerToCompare))
'''

# Testing the classes and objects
#receptorTest = ocr.Receptor(f"{basePath}/test_files/rec.crg.pdb", relativeASAcutoff=0.7, name="Receptor teste")
#ligandTest = ocl.Ligand(f"{basePath}/test_files/xtal-lig.pdb", name="Ligante teste")

#gninaTest = = ocgnina.Gnina(f"{basePath}/test_files/conf_gnina.txt", f"{basePath}/test_files/box.pdb", receptorTest, f"{basePath}/test_files/rec.crg.pdbqt", ligandTest, f"{basePath}/test_files/xtal-lig.pdbqt", f"{basePath}/test_files/gnina.log", f"{basePath}/test_files/gnina.pdbqt", name="Gnina Test")
#vinaTest = ocvina.Vina(f"{basePath}/test_files/conf_vina.txt", f"{basePath}/test_files/box.pdb", receptorTest, f"{basePath}/test_files/rec.crg.pdbqt", ligandTest, f"{basePath}/test_files/xtal-lig.pdbqt", f"{basePath}/test_files/vina.log", f"{basePath}/test_files/vina.pdbqt", name="Vina Test")
#sminaTest = ocsmina.Smina(f"{basePath}/test_files/conf_smina.txt", f"{basePath}/test_files/box.pdb", receptorTest, f"{basePath}/test_files/rec.crg2.pdbqt", ligandTest, f"{basePath}/test_files/xtal-lig2.pdbqt", f"{basePath}/test_files/smina.log", f"{basePath}/test_files/smina.pdbqt", name="Smina Test")
#plantsTest = ocplants.PLANTS(f"{basePath}/test_files/conf_plants.txt", f"{basePath}/test_files/box.pdb", receptorTest, f"{basePath}/test_files/rec.crg2_prepared_spores.mol2", ligandTest, f"{basePath}/test_files/xtal-lig2_prepared_spores.mol2", f"{basePath}/test_files/plants.log", f"{basePath}/test_files/plants.pdb", name="PLANTS Test")
