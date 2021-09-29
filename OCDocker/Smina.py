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
    Smina object with methods for easy run
    """
    def __init__(self, configPath, boxFile, receptorPath, preparedReceptorPath, ligandPath, preparedLigandPath, sminaLog, outputSmina, name=""):
        self.name = str(name)
        self.config = str(configPath)
        self.boxFile = str(boxFile)
        # Receptor
        self.inputReceptor = str(receptorPath)
        self.preparedReceptor = str(preparedReceptorPath)
        self.prepareReceptorCmd = self.__prepare_receptor_cmd()
        # Ligand
        self.preparedLigand = str(preparedLigandPath)
        self.inputLigand = self.ligandPath
        self.prepareLigandCmd = self.__prepare_ligand_cmd()
        # Vina
        self.sminaLog = str(sminaLog)
        self.outputSmina = str(outputSmina)
        self.sminaCmd = self.__smina_cmd()

    def __smina_cmd(self):
        '''
        Generate the vina command
        Input:
          -
        Return:
          -
        '''
        cmd = [smina, '--config', self.config, '--ligand', self.preparedLigand, '--autobox_ligand', self.preparedLigand, '--out', self.outputSmina, '--log', self.sminaLog, "--cpu", "1"]
        return cmd

    def __prepare_ligand_cmd(self):
        '''
        Generate the prepare ligand command
        Input:
          -
        Return:
          The Path of the ligand
        '''

        cmd = ['obabel', self.inputLigand, '-O', self.preparedLigand]

        return cmd

    def __prepare_receptor_cmd(self):
        '''
        Generate the prepare receptor command
        Input:
          -
        Return:
          -
        '''

        cmd = ["obabel", self.inputReceptor, "-xr", "-O", self.preparedReceptor]
        return cmd

    def run_smina(self, logFile = ""):
        '''
        Run vina
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
        Return:
          0 - No problems were found
          1 - self.sminaCommand is not set or is empty list
          2 - self.sminaCommand has wrong type
          3 - Problems while running the self.vinaCommand
        '''
        return octools.run(self.sminaCmd, logFile=logFile)

    def run_prepare_ligand(self, logFile = ""):
        '''
        Run prepare_ligand4
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
        Return:
          0 - No problems were found
          1 - self.prepareLigand is not set or is empty list
          2 - self.prepareLigand has wrong type
          3 - Problems while running the self.prepareLigand
        '''
        return octools.run(self.prepareLigandCmd, logFile=logFile)

    def run_prepare_receptor(self, logFile = ""):
        '''
        Run prepare_receptor4
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
        Return:
          0 - No problems were found
          1 - self.prepareLigand is not set or is empty list
          2 - self.prepareLigand has wrong type
          3 - Problems while running the self.prepareLigand
        '''
        return octools.run(self.prepareReceptorCmd, logFile=logFile)

    def print_attributes(self):
        '''
        Run prepare_receptor4 (warper for run)
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
        print(f"Smina execution log path:    '{self.sminaLog if self.sminaLog else '-' }'")
        print(f"Smina output path:           '{self.outputSmina if self.outputSmina else '-' }'")
        print(f"Smina command:               '{' '.join(self.sminaCmd) if self.sminaCmd else '-' }'")
        return

# Functions
###############################################################################
def run_prepare_ligand(inputLigand, preparedLigand, logFile = ""):
    '''
    Prepares the ligand using prepare_ligand using MGLTools suite.
    Input:
      inputLigand  [string]              - Path to the input ligand file.
      preparedLigand [string]            - Path to the output ligand file.
      logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      -
    '''
    # Create the command list
    cmd = ['obabel', inputLigand, '-O', preparedLigand]

    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_prepare_receptor(inputReceptor, outputReceptor, logFile=""):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
      inputReceptor    [string]          - Path to the input receptor file.
      preparedReceptor [string]          - Path to the output receptor file.
      logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      -
    '''
    # Create the command list
    cmd = ["obabel", inputReceptor, "-xr", "-O", preparedReceptor]

    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_smina(config, ligand, outpath, logpath):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
      config  [string]  - Path to the config file.
      ligand  [string]  - Path to the ligand file.
      outpath [string]  - Path to the receptor file.
      logpath [string]  - Path to the log file.
      logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
    Return:
      -
    '''
    # Create the command list
    command = ['vina', '--config', config, '--ligand', ligand, '--out', outpath, '--log', logpath, "--cpu", "1"]

    # Run the command
    return octools.run(cmd, logFile=logFile)
