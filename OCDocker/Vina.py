#!/usr/lib/python3

# Imports
###############################################################################
import sys
import shutil
import tarfile
import datetime
from Initialise import *

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
Sets of classes and functions that are used .

They are imported as:

import OCDocker.Vina as ocvina
'''

# Classes
###############################################################################


# Functions
###############################################################################
def box_to_vina(path):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
     path     [string] - Path to the box file
    Return:
      -
    '''
    return

def prepare_ligand():
    '''
    '''
    #subprocess.run([f"{progPath}/ADT_scripts/prepare_ligand4.py", "-l", f"{outf}/ligand.mol2", "-C", "-o", f"{outf}/ligand.pdbqt"]) #change /home/ocean/Softwares/mgltools/bin/python automatic script

    return

def prepare_receptor():
    '''
    '''
    #subprocess.run([f"{progPath}/ADT_scripts/prepare_receptor4.py", "-r", f"{outf}/receptor.pdb", "-o", f"{outf}/receptor.pdbqt", "-A", "hydrogens", "-U", "nphs_lps_waters"])

    return

def run_vina():
    '''
    '''
    return
