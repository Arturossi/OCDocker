#!/usr/lib/python3

# Imports
###############################################################################
import sys
import shutil
import tarfile
import datetime
from Initialise import *
import OCDocker.Toolbox as octools

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
def box_to_vina(boxFile, confFile, receptor = "receptor_noH"):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
     boxFile   [string]                       - Path to the box file
     confFile  [string]                       - Path to the conf file
     receptor  [string] Default: receptor_noH - Receptor name to be used in conf file
    Return:
      0 - No problems were found
      1 - Problems while working with the box file
      2 - Problems while working with the conf file
    '''
    try:
        # Open the box file
        with open(str(boxFile), "r") as box_file:
            # List to hold all the data
            lines = []

            # For each line in the file
            for line in box_file:
                # If it starts with REMARK
                if line.startswith("REMARK"):
                    # Split the line (using spaces as delimiters)
                    l = line.split()
                    # Append the last 3 elements as a tuple to the list
                    lines.append((l[-3], l[-2], l[-1]))

                    # If the length of the lines element is 2 or greater
                    if len(lines) >= 2:
                        # Break the loop (optimization)
                        break
    except Exception as e:
        octools.print_warning(f"Found a problem while opening box file: {e}")
        return

    try:
        # Now open the conf file to write
        with open(confFile, 'w') as conf_file:
            conf_file.write(f"receptor = {receptor}.pdbqt\n\n");
            conf_file.write(f"center_x = {lines[0][0]}\n")
            conf_file.write(f"center_y = {lines[0][1]}\n")
            conf_file.write(f"center_z = {lines[0][2]}\n\n")
            conf_file.write(f"size_x = {lines[1][0]}\n")
            conf_file.write(f"size_y = {lines[1][1]}\n")
            conf_file.write(f"size_z = {lines[1][2]}\n\n")
            conf_file.write(f"energy_range = {energy_range}\n")
            conf_file.write(f"exhaustiveness = {exhaustiveness}\n")
            conf_file.write(f"num_modes = {num_modes}\n")
    except Exception as e:
        octools.print_warning(f"Found a problem while opening conf file: {e}")
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
