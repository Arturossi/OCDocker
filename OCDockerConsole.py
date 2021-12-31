#!/usr/lib/python3

# Imports
###############################################################################
import textwrap as tw
from pprint import pprint

from OCDocker.Initialise import *
import OCDocker.Toolbox as octools

import OCDocker.Vina as ocvina
import OCDocker.Smina as ocsmina
import OCDocker.Database as ocdb
import OCDocker.Astex as ocastex
import OCDocker.PDBbind as ocpdbbind
import OCDocker.Ligand as ocl
import OCDocker.Complex as occ
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
message = tw.dedent("""\033[1;93m
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\033[1;96m
                                CONSOLE MODE\033[1;93m
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\033[1;0m
            This mode is intended to make debug easier and allow the
        user to interact with the OCDocker pipeline step by step.

        \033[1;91mWARNING\033[1;0m: This mode is still experimental, some unexpected
        behaviour might occur while using it.\033[1;93m
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
\033[1;0m""")

print(message)

args.debug = 1
args.verbosity = 1

basePath = '/mnt/e/Documents/OCDocker/OCDocker'

# Testing the classes and objects
receptorTest = ocr.Receptor(f"{basePath}/test/rec.crg.pdb", relativeASAcutoff=0.7, name="Receptor teste")
ligandTest = ocl.Ligand(f"{basePath}/test/xtal-lig.pdb", name="Ligante teste")

vinaTest = ocvina.Vina(f"{basePath}/test/conf_vina.txt", f"{basePath}/test/box.pdb", receptorTest, f"{basePath}/test/rec.crg.pdbqt", ligandTest, f"{basePath}/test/xtal-lig.pdbqt", f"{basePath}/test/vina.log", f"{basePath}/test/vina.pdbqt", name="Vina Test")
sminaTest = ocsmina.Smina(f"{basePath}/test/conf_smina.txt", receptorTest, f"{basePath}/test/rec.crg2.pdbqt", ligandTest, f"{basePath}/test/xtal-lig2.pdbqt", f"{basePath}/test/smina.log", f"{basePath}/test/smina.pdbqt", name="Smina Test")
