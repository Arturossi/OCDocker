#!/usr/lib/python3

# Imports
###############################################################################
import os
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
class Vina:
    """
    Vina object with methods for easy run
    """
    def __init__(self, configPath, boxFile, receptorPath, preparedReceptorPath, ligandPath, preparedLigandPath, vinaLog, outputVina, name=""):
        self.name = name
        self.config = configPath
        self.boxFile = boxFile
        # Receptor
        self.inputReceptor = receptorPath
        self.preparedReceptor = preparedReceptorPath
        self.prepareReceptorCmd = __prepare_receptor_cmd()
        # Ligand
        self.inputLigand = ligandPath
        self.preparedLigand = preparedLigandPath
        self.prepareLigandCmd = __prepare_ligand_cmd()
        # Vina
        self.vinaLog = vinaLog
        self.outputVina = outputVina
        self.vinaCmd = __vina_cmd()

    def __vina_cmd(self):
        '''
        Generate the vina command
        Input:
          -
        Return:
          -
        '''
        cmd = ['vina', '--config', self.config, '--ligand', self.inputLigand, '--out', outpath, '--log', self.vinaLog, "--cpu", "1"]
        return cmd

    def __prepare_ligand_cmd(self):
        '''
        Generate the prepare ligand command
        Input:
          -
        Return:
          -
        '''
        cmd = [pythonsh, prepare_ligand, "-l", self.inputLigand, "-C", "-o", self.preparedLigand]
        return cmd

    def __prepare_receptor_cmd(self):
        '''
        Generate the prepare ligand command
        Input:
          -
        Return:
          -
        '''
        cmd = [pythonsh, prepare_receptor, "-r", self.inputReceptor, "-o", self.inputReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]
        return cmd

    def run_vina(self, logFile = ""):
        '''
        Run vina (warper for run)
        Input:
          logfile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
        Return:
          0 - No problems were found
          1 - self.vinaCommand is not set or is empty list
          2 - self.vinaCommand has wrong type
          3 - Problems while running the self.vinaCommand
        '''
        return run(self.vinaCommand, logFile=logFile)

    def run_prepare_ligand(self, logFile = ""):
        '''
        Run prepare_ligand4 (warper for run)
        Input:
          logfile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
        Return:
          0 - No problems were found
          1 - self.prepareLigand is not set or is empty list
          2 - self.prepareLigand has wrong type
          3 - Problems while running the self.prepareLigand
        '''
        return run(self.prepareLigand, logFile=logFile)

    def run_prepare_receptor(self, logFile = ""):
        '''
        Run prepare_receptor4 (warper for run)
        Input:
          logfile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
        Return:
          0 - No problems were found
          1 - self.prepareLigand is not set or is empty list
          2 - self.prepareLigand has wrong type
          3 - Problems while running the self.prepareLigand
        '''
        return run(self.prepareReceptor, logFile=logFile)

    def box_to_vina(self):
        '''
        Convert a box (DUDE like format) to vina input.
        Input:
          -
        Return:
          0 - No problems were found
          1 - Problems while working with the box file
          2 - Problems while working with the conf file
        '''
        try:
            # Open the box file
            with open(str(self.boxFile), "r") as box_file:
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
            octools.print_error(f"Found a problem while opening box file: {e}")
            return

        try:
            # Now open the conf file to write
            with open(self.confFile, 'w') as conf_file:
                conf_file.write(f"receptor = {self.preparedReceptor}.pdbqt\n\n");
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
            octools.print_error(f"Found a problem while opening conf file: {e}")
        return

# Functions
###############################################################################
def run(cmd, logFile = ""):
    '''
    Run the command (generic)
    Input:
      cmd     [list(string)]             - List containing the strings of the command
      logfile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
    Return:
      0 - No problems were found
      1 - The var cmd is not set or is empty list
      2 - The command list has wrong type
      3 - Problems while running the command
    '''
    if not cmd:
        octools.print_error(f"The variable cmd is not set or is an empty list!")
        return 1

    if type(cmd) == list:
        octools.print_error(f"The argument cmd has to be a list! Found {type(cmd)} instead...")
        return 2

    if logFile == "":
        logFile = open(os.devnull, 'w')

    try:
        with open(logFile, "w") as outfile:
            subprocess.run(cmd, stdout=outfile)
    except Exception as e:
        octools.print_error(f"Found a problem while executing the command '{' '.join(cmd)}': {e}")
        return 3
    return 0


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
        octools.print_error(f"Found a problem while opening box file: {e}")
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
        octools.print_error(f"Found a problem while opening conf file: {e}")
    return

def prepare_ligand(inputLigand, outputLigand):
    '''
    Prepares the ligand using prepare_ligand using MGLTools suite
    Input:
      inputLigand  [string]  - Path to the input ligand file
      outputLigand [string]  - Path to the output ligand file
    Return:
      -
    '''
    # Create the command list
    cmd = [pythonsh, prepare_ligand, "-l", inputLigand, "-C", "-o", outputLigand]

    # Open the output file
    with open("./prepare_ligand.log", "w") as outfile:
        subprocess.run(cmd, stdout=outfile)
    return

def prepare_receptor(inputReceptor, outputReceptor):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
      inputReceptor  [string]  - Path to the input receptor file
      outputReceptor [string]  - Path to the output receptor file
    Return:
      -
    '''
    # Create the command list
    cmd = [pythonsh, prepare_receptor, "-r", inputReceptor, "-o", outputReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]

    # Open the log and output script text to it
    with open("./prepare_receptor.log", "w") as outfile:
        subprocess.run(cmd, stdout=outfile)
    return

def run_vina(config, ligand, outpath, logpath):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
      config   [string]  - Path to the config file
      ligand   [string]  - Path to the ligand file
      outpath  [string]  - Path to the receptor file
      logpath  [string]  - Path to the log file
    Return:
      -
    '''
    # Create the command list
    command = ['vina', '--config', config, '--ligand', ligand, '--out', outpath, '--log', logpath, "--cpu", "1"]

    # Open the log and output script text to it
    with open("./run_vina.log", "w") as outfile:
        subprocess.run(command, stdout=outfile)
    return
