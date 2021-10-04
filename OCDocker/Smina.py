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
Sets of classes and functions that are used to prepare smina files and run it.

They are imported as:

import OCDocker.Smina as ocsmina
'''

# Classes
###############################################################################
class Smina:
    """
    Smina object with methods for easy run.
    """
    def __init__(self, configPath, receptorPath, preparedReceptorPath, ligandPath, preparedLigandPath, sminaLog, outputSmina, name=""):
        self.name = str(name)
        self.config = str(configPath)
        # Receptor
        self.inputReceptor = str(receptorPath)
        self.preparedReceptor = str(preparedReceptorPath)
        self.prepareReceptorCmd = self.__prepare_receptor_cmd()
        # Ligand
        self.preparedLigand = str(preparedLigandPath)
        self.inputLigand = ligandPath
        self.prepareLigandCmd = self.__prepare_ligand_cmd()
        # Vina
        self.sminaLog = str(sminaLog)
        self.outputSmina = str(outputSmina)
        self.sminaCmd = self.__smina_cmd()
        self.__gen_smina_conf()

    def __smina_cmd(self):
        '''
        Generate the vina command.
        Input:
          -
        Return:
          cmd [list[string]] - List of strings of the command.
        '''
        cmd = [smina, "--config", self.config, "--ligand", self.preparedLigand, "--autobox_ligand", self.preparedLigand]

        if smina_local_only.lower() in ["y", "ye", "yes"]:
            cmd.append("--score_only")
        if smina_minimize.lower() in ["y", "ye", "yes"]:
            cmd.append("--minimize")
        if smina_randomize_only.lower() in ["y", "ye", "yes"]:
            cmd.append("--randomize_only")
        if smina_accurate_line.lower() in ["y", "ye", "yes"]:
            cmd.append("--accurate_line")
        if smina_minimize_early_term.lower() in ["y", "ye", "yes"]:
            cmd.append("--minimize_early_term")

        cmd.extend(["--out", self.outputSmina, "--log", self.sminaLog, "--cpu", "1"])
        return cmd

    def __prepare_ligand_cmd(self):
        '''
        Generate the prepare ligand command.
        Input:
          -
        Return:
          cmd [list[string]] - List of strings of the command.
        '''

        cmd = [obabel, self.inputLigand, "-O", self.preparedLigand]

        return cmd

    def __prepare_receptor_cmd(self):
        '''
        Generate the prepare receptor command.
        Input:
          -
        Return:
          cmd [list[string]] - List of strings of the command.
        '''

        cmd = [obabel, self.inputReceptor, "-xr", "-O", self.preparedReceptor]
        return cmd

    def run_smina(self, logFile = ""):
        '''
        Run smina.
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
        Return:
          [int]
          See Error.py for all return codes.
        '''
        return octools.run(self.sminaCmd, logFile=logFile)

    def run_prepare_ligand(self, logFile = ""):
        '''
        Run obabel convert ligand to pdbqt script.
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
        Return:
          [int]
          See Error.py for all return codes.
        '''
        return octools.run(self.prepareLigandCmd, logFile=logFile)

    def run_prepare_receptor(self, logFile = ""):
        '''
        Run obabel convert receptor to pdbqt script.
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
        Return:
          [int]
          See Error.py for all return codes.
        '''
        return octools.run(self.prepareReceptorCmd, logFile=logFile)

    def __gen_smina_conf(self):
        '''
        Creates a conf file for smina.
        Input:
          -
        Return:
          [int]
          See Error.py for all return codes.
        '''
        return gen_smina_conf(self.config, self.preparedReceptor)

    def print_attributes(self):
        '''
        Print the class attributes.
        Input:
          -
        Return:
          -
        '''
        print(f"Name:                        '{self.name if self.name else '-' }'")
        print(f"Config path:                 '{self.config if self.config else '-' }'")
        print(f"Input receptor path:         '{self.inputReceptor if self.inputReceptor else '-' }'")
        print(f"Prepared receptor path:      '{self.preparedReceptor if self.preparedReceptor else '-' }'")
        print(f"Prepared receptor command:   '{' '.join(self.prepareReceptorCmd) if self.prepareReceptorCmd else '-' }'")
        print(f"Input ligand path:           '{self.inputLigand if self.inputLigand else '-' }'")
        print(f"Prepared ligand path:        '{self.preparedLigand if self.preparedLigand else '-' }'")
        print(f"Prepared ligand command:     '{' '.join(self.prepareLigandCmd) if self.prepareLigandCmd else '-' }'")
        print(f"Smina execution log path:    '{self.sminaLog if self.sminaLog else '-' }'")
        print(f"Smina output path:           '{self.outputSmina if self.outputSmina else '-' }'")
        print(f"Smina command:               '{' '.join(self.sminaCmd) if self.sminaCmd else '-' }'")
        return

# Functions
###############################################################################
def gen_smina_conf(confFile, receptor = "receptor_noH"):
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
    octools.printv(f"Creating smina conf file in the path '{confFile}'.")
    try:
        # Now open the conf file to write
        with open(confFile, 'w') as conf_file:
            conf_file.write(f"receptor = {receptor}\n\n");
            if smina_custom_scoring.lower() != "no":
                conf_file.write(f"custom_scoring = {smina_custom_scoring}\n")
            if smina_custom_atoms.lower() != "no":
                conf_file.write(f"custom_atoms = {smina_custom_atoms}\n")

            conf_file.write(f"minimize_iters = {smina_minimize_iters}\n")
            conf_file.write(f"approximation = {smina_approximation}\n")
            conf_file.write(f"factor = {smina_factor}\n")
            conf_file.write(f"force_cap = {smina_force_cap}\n")

            if smina_user_grid.lower() != "no":
                conf_file.write(f"user_grid = {smina_custom_scoring}\n")
                conf_file.write(f"user_grid_lambda = {smina_user_grid_lambda}\n")

            conf_file.write(f"energy_range = {smina_energy_range}\n")
            conf_file.write(f"exhaustiveness = {smina_exhaustiveness}\n")
            conf_file.write(f"num_modes = {smina_num_modes}\n")
    except Exception as e:
        return errors.write_file(message=f"Found a problem while opening conf file: {e}.", level="error")
    return errors.ok()

def run_prepare_ligand(inputLigand, preparedLigand, logFile = ""):
    '''
    Converts the ligand to .pdbqt using obabel.
    Input:
      inputLigand    [string]                   - Path to the input ligand file.
      preparedLigand [string]                   - Path to the output ligand file.
      logFile        [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [obabel, inputLigand, "-O", preparedLigand]

    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_prepare_receptor(inputReceptor, outputReceptor, logFile=""):
    '''
    Converts the receptor to .pdbqt using obabel.
    Input:
      inputReceptor    [string]                   - Path to the input receptor file.
      preparedReceptor [string]                   - Path to the output receptor file.
      logFile          [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [obabel, inputReceptor, "-xr", "-O", preparedReceptor]

    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_smina(config, preparedLigand, outputSmina, sminaLog, logpath):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
      config          [string]                   - Path to the config file.
      preparedLigand  [string]                   - Path to the ligand file.
      outputSmina     [string]                   - Path to the receptor file.
      sminaLog        [string]                   - Path to the smina log file.
      logFile         [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [smina, "--config", config, "--ligand", preparedLigand, "--autobox_ligand", preparedLigand]

    if smina_local_only.lower() in ["y", "ye", "yes"]:
        cmd.append("--score_only")
    if smina_minimize.lower() in ["y", "ye", "yes"]:
        cmd.append("--minimize")
    if smina_randomize_only.lower() in ["y", "ye", "yes"]:
        cmd.append("--randomize_only")
    if smina_accurate_line.lower() in ["y", "ye", "yes"]:
        cmd.append("--accurate_line")
    if smina_minimize_early_term.lower() in ["y", "ye", "yes"]:
        cmd.append("--minimize_early_term")

    cmd.extend(["--out", outputSmina, "--log", sminaLog, "--cpu", "1"])
    # Run the command
    return octools.run(cmd, logFile=logFile)
