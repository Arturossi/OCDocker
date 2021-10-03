#!/usr/lib/python3

# Imports
###############################################################################
import os
import sys
import shutil
import tarfile
import datetime
import subprocess

from OCDocker.Initialise import *
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
Sets of classes and functions that are used to prepare vina files and run it.

They are imported as:

import OCDocker.Vina as ocvina
'''

# Classes
###############################################################################
class Vina:
    """
    Vina object with methods for easy run.
    """
    def __init__(self, configPath, boxFile, receptorPath, preparedReceptorPath, ligandPath, preparedLigandPath, vinaLog, outputVina, name=""):
        self.name = str(name)
        self.config = str(configPath)
        self.boxFile = str(boxFile)
        # Receptor
        self.inputReceptor = str(receptorPath)
        self.preparedReceptor = str(preparedReceptorPath)
        self.prepareReceptorCmd = self.__prepare_receptor_cmd()
        # Ligand
        self.preparedLigand = str(preparedLigandPath)
        self.convert2mol2log = ""
        self.inputLigand = self.__process_ligand(ligandPath)
        self.prepareLigandCmd = self.__prepare_ligand_cmd()
        # Vina
        self.vinaLog = str(vinaLog)
        self.outputVina = str(outputVina)
        self.vinaCmd = self.__vina_cmd()
        # Create the box
        self.__box_to_vina()

    def __process_ligand(self, ligandPath):
        '''
        Process the ligand to output to mol2 if needed.
        Input:
          ligandPath [list(string)] - The path for the ligand.
        Return:
          [string] The Path of the ligand with mol2 extension.
        '''
        # Get the extension
        ligandExtension = os.path.splitext(ligandPath)[1]

        # If its mol2 we do not need to convert it
        if ligandExtension == "mol2":
            # So return the ligandPath
            return ligandPath

        # Create the output path
        outputLigandPath = f"{os.path.dirname(ligandPath)}/{os.path.splitext(os.path.basename(ligandPath))[0]}.mol2"

        # Process the ligand
        octools.convert2mol2(ligandPath, outputLigandPath, logFile = self.convert2mol2log)

        return outputLigandPath

    def __vina_cmd(self):
        '''
        Generate the vina command.
        Input:
          -
        Return:
          list[string] - List of strings of the command.
        '''
        cmd = [vina, "--config", self.config, "--ligand", self.preparedLigand, "--out", self.outputVina, "--log", self.vinaLog, "--cpu", "1"]
        return cmd

    def __prepare_ligand_cmd(self):
        '''
        Generate the prepare ligand command.
        Input:
          -
        Return:
          list[string] - List of strings of the command.
        '''
        cmd = [pythonsh, prepare_ligand, "-l", self.inputLigand, "-C", "-o", self.preparedLigand]
        return cmd

    def __prepare_receptor_cmd(self):
        '''
        Generate the prepare receptor command.
        Input:
          -
        Return:
          list[string] - List of strings of the command.
        '''
        cmd = [pythonsh, prepare_receptor, "-r", self.inputReceptor, "-o", self.preparedReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]
        return cmd

    def __box_to_vina(self):
        '''
        Convert a box (DUDE like format) to vina input.
        Input:
          -
        Return:
          [int]
          See Error.py for all return codes.
        '''
        return box_to_vina(self.boxFile, self.config, self.preparedReceptor)

    def run_vina(self, logFile = ""):
        '''
        Run vina.
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
        Return:
          [int]
          See Error.py for all return codes.
        '''
        # Print verboosity
        octools.printv(f"Running vina using the '{self.confFile}' configurations.")
        return octools.run(self.vinaCmd, logFile=logFile)

    def run_prepare_ligand(self, logFile = ""):
        '''
        Run 'prepare_ligand4'.
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
        Return:
          [int]
          See Error.py for all return codes.
        '''
        # Print verboosity
        octools.printv(f"Running '{prepare_ligand}' for '{self.inputLigand}'.")
        return octools.run(self.prepareLigandCmd, logFile=logFile)

    def run_prepare_receptor(self, logFile = ""):
        '''
        Run 'prepare_receptor4'.
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
        Return:
          [int]
          See Error.py for all return codes.
        '''
        # Print verboosity
        octools.printv(f"Running '{prepare_receptor}' for '{self.inputReceptor}'.")
        return octools.run(self.prepareReceptorCmd, logFile=logFile)

    def print_attributes(self):
        '''
        Print the class attributes.
        Input:
          -
        Return:
          -
        '''
        print(f"Name:                        '{self.name if self.name else '-' }'")
        print(f"Box path:                    '{self.boxFile if self.boxFile else '-' }'")
        print(f"Config path:                 '{self.config if self.config else '-' }'")
        print(f"Input receptor path:         '{self.inputReceptor if self.inputReceptor else '-' }'")
        print(f"Prepared receptor path:      '{self.preparedReceptor if self.preparedReceptor else '-' }'")
        print(f"Prepared receptor command:   '{' '.join(self.prepareReceptorCmd) if self.prepareReceptorCmd else '-' }'")
        print(f"Input ligand path:           '{self.inputLigand if self.inputLigand else '-' }'")
        print(f"Prepared ligand path:        '{self.preparedLigand if self.preparedLigand else '-' }'")
        print(f"Prepared ligand command:     '{' '.join(self.prepareLigandCmd) if self.prepareLigandCmd else '-' }'")
        print(f"Conversion to mol2 log path: '{self.convert2mol2log if self.convert2mol2log else '-' }'")
        print(f"Vina execution log path:     '{self.vinaLog if self.vinaLog else '-' }'")
        print(f"Vina output path:            '{self.outputVina if self.outputVina else '-' }'")
        print(f"Vina command:                '{' '.join(self.vinaCmd) if self.vinaCmd else '-' }'")
        return

# Functions
###############################################################################
def box_to_vina(boxFile, confFile, receptor = "receptor_noH"):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
      boxFile   [string]                         - Path to the box file.
      confFile  [string]                         - Path to the conf file.
      receptor  [string] DEFAULT: "receptor_noH" - Receptor name to be used in conf file.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    octools.printv(f"Converting the box file '{boxFile}' to vina conf file as '{confFile}' file.")
    # Test if the file boxFile exists
    if not os.path.exists(boxFile):
        return errors.file_do_not_exist(message=f"The box file in the path {boxFile} does not exists! Please ensure that the file exsits and the path is correct. If you have no box file, try to run the function 'runprank' from the 'runprank' library to create it before calling this function or creating a Vina class object.", level="error")
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
        return errors.read_file(message=f"Found a problem while reading the box file: {e}", level="error")

    try:
        # Now open the conf file to write
        with open(confFile, 'w') as conf_file:
            conf_file.write(f"receptor = {receptor}\n\n");
            conf_file.write(f"center_x = {lines[0][0]}\n")
            conf_file.write(f"center_y = {lines[0][1]}\n")
            conf_file.write(f"center_z = {lines[0][2]}\n\n")
            conf_file.write(f"size_x = {lines[1][0]}\n")
            conf_file.write(f"size_y = {lines[1][1]}\n")
            conf_file.write(f"size_z = {lines[1][2]}\n\n")
            conf_file.write(f"energy_range = {vina_energy_range}\n")
            conf_file.write(f"exhaustiveness = {vina_exhaustiveness}\n")
            conf_file.write(f"num_modes = {vina_num_modes}\n")
    except Exception as e:
        return errors.write_file(message=f"Found a problem while opening conf file: {e}.", level="error")
    return errors.ok()

def run_prepare_ligand(inputLigand, outputLigand, logFile=""):
    '''
    Prepares the ligand using 'prepare_ligand' from MGLTools suite.
    Input:
      inputLigand  [string]                   - Path to the input ligand file.
      outputLigand [string]                   - Path to the output ligand file.
      logFile      [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [pythonsh, prepare_ligand, "-l", inputLigand, "-C", "-o", outputLigand]
    # Print verboosity
    octools.printv(f"Running '{prepare_ligand}' for '{inputLigand}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_prepare_receptor(inputReceptor, outputReceptor, logFile=""):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
      inputReceptor  [string]                   - Path to the input receptor file.
      outputReceptor [string]                   - Path to the output receptor file.
      logFile        [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [pythonsh, prepare_receptor, "-r", inputReceptor, "-o", outputReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]
    # Print verboosity
    octools.printv(f"Running '{prepare_receptor}' for '{inputReceptor}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_vina(confFile, ligand, outpath, logpath, logFile=""):
    '''
    Run vina.
    Input:
      confFile [string]                   - Path to the config file.
      ligand   [string]                   - Path to the ligand file.
      outpath  [string]                   - Path to the receptor file.
      logpath  [string]                   - Path to the log file.
      logFile  [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [vina, "--config", confFile, "--ligand", ligand, "--out", outpath, "--log", logpath, "--cpu", "1"]
    # Print verboosity
    octools.printv(f"Running vina using the '{confFile}' configurations.")
    # Run the command
    return octools.run(cmd, logFile=logFile)
