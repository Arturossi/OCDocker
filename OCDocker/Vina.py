#!/usr/lib/python3

# Imports
###############################################################################
import os
import sys
import shutil
import tarfile
import datetime
import subprocess

import numpy as np
import pandas as pd

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
from OCDocker.Initialise import *
import OCDocker.Toolbox as octools

from Bio.PDB import *
from glob import glob
from rdkit import Chem

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
    def __init__(self, configPath, boxFile, receptor, preparedReceptorPath, ligand, preparedLigandPath, vinaLog, outputVina, name=""):
        self.name = str(name)
        self.config = str(configPath)
        self.boxFile = str(boxFile)
        # Receptor
        self.inputReceptor = self.__parse_receptor(receptor)
        self.inputReceptorPath = self.__parse_receptor_path(receptor)
        self.preparedReceptor = str(preparedReceptorPath)
        self.prepareReceptorCmd = self.__prepare_receptor_cmd()
        # Ligand
        self.preparedLigand = str(preparedLigandPath)
        self.convert2mol2log = ""
        self.inputLigand = self.__parse_ligand(ligand)
        self.inputLigandPath = self.__parse_ligand_path(ligand)
        self.prepareLigandCmd = self.__prepare_ligand_cmd()
        # Vina
        self.vinaLog = str(vinaLog)
        self.outputVina = str(outputVina)
        self.vinaCmd = self.__vina_cmd()
        # Create the box
        self.__box_to_vina()

    ## Private ##

    def __parse_receptor(self, receptor):
        '''
        Parse the receptor as input, handling its type.
        Input:
          receptor [ocr.Receptor] - The path for the receptor or its ocr.Receptor object.
        Return:
          [ocr.Receptor]
           [object] The ocr.Receptor object.
           [None]   If is a path, returns None (no linkage to Receptor object) NOT RECOMENDED.
        '''
        # Check the type of the receptor
        if type(receptor) == ocr.Receptor:
            octools.printv(f"The receptor '{receptor}' has been loaded.")
            return receptor

        octools.print_warning(f"The receptor '{receptor}' is not the type 'ocr.Receptor'. It is STRONGLY recomended that you provide an 'ocr.Receptor' object.")
        return None

    def __parse_receptor_path(self, receptor):
        '''
        Parse the receptor path, handling its type.
        Input:
          receptor [string/ocr.Receptor] - The path for the receptor or its receptor object.
        Return:
          [string] The receptor path.
        '''
        # Check the type of receptor variable
        if type(receptor) == ocr.Receptor:
            return receptor.path
        elif type(receptor) == str:
            # Since is a string, check if the file exists
            if os.path.isfile(receptor):
                # Exists! Return it!
                return receptor
            else:
                _ = errors.file_do_not_exist(message=f"The receptor '{receptor}' has not a valid path.", level="error")
                return ""

        _ = errors.wrong_type(message=f"The receptor '{receptor}' has not a supported type. Expected 'string' or 'ocr.Receptor' but got {type(receptor)} instead.", level="error")
        return ""

    def __parse_ligand(self, ligand):
        '''
        Parse the ligand as input, handling its type.
        Input:
          receptor [ocl.Ligand] - The path for the receptor or its receptor object.
        Return:
          [ocl.Ligand]
           [object] The ocr.Ligand object.
           [None]   If is a path, returns None (no linkage to Ligand object) NOT RECOMENDED.
        '''
        # Check the type of the ligand
        if type(ligand) == ocl.Ligand:
            octools.printv(f"The ligand '{ligand}' has been loaded.")
            return ligand

        octools.print_warning(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.")
        return None

    def __parse_ligand_path(self, ligand):
        '''
        Parse the ligand path, handling its type.
        Input:
          ;igand [string/ocl.Ligand] - The path for the ligand or its ocl.Ligand object.
        Return:
          [string] The ligand object.
        '''
        # Check the type of ligand variable
        if type(ligand) == ocl.Ligand:
            return ligand.path
        elif type(ligand) == str:
            # Since is a string, check if the file exists
            if os.path.isfile(ligand):
                # Exists! Process it then!
                return __process_ligand(ligand)
            else:
                _ = errors.file_do_not_exist(message=f"The ligand '{ligand}' has not a valid path.", level="error")
                return ""

        _ = errors.wrong_type(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.", level="error")
        return ""

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
        octools.convertMols(ligandPath, outputLigandPath, logFile = self.convert2mol2log)

        return outputLigandPath

    def __vina_cmd(self):
        '''
        Generate the vina command.
        Input:
          -
        Return:
          list[string] - List of strings of the command.
        '''
        cmd = [vina, "--config", self.config, "--ligand", self.preparedLigand, "--out", self.outputVina, "--cpu", "1"]
        return cmd

    def __prepare_ligand_cmd(self):
        '''
        Generate the prepare ligand command.
        Input:
          -
        Return:
          list[string] - List of strings of the command.
        '''
        cmd = [pythonsh, prepare_ligand, "-l", self.inputLigandPath, "-C", "-o", self.preparedLigand]
        return cmd

    def __prepare_receptor_cmd(self):
        '''
        Generate the prepare receptor command.
        Input:
          -
        Return:
          list[string] - List of strings of the command.
        '''
        cmd = [pythonsh, prepare_receptor, "-r", self.inputReceptorPath, "-o", self.preparedReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]
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

    ## Public ##

    def read_vina_log(self):
        '''
        Read the vina log path, returning a pd.dataframe with data from complexes.
        Input:
          -
        Return:
          [pd.dataframe]
        '''
        return read_vina_log(self.vinaLog)

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
        octools.printv(f"Running vina using the '{self.config}' configurations.")
        return octools.run(self.vinaCmd, logFile=self.vinaLog)

    def run_prepare_ligand(self, logFile = "", useOpenBabel = False):
        '''
        Run 'prepare_ligand4' or openbabel to prepare the ligand.
        Input:
          logFile      [list(string)] DEFAULT: ""    - Path to the logFile. If empty, suppress the output
          useOpenBabel [list(string)] DEFAULT: False - If True will try to run openbabel, otherwise will run prepare_ligand4
        Return:
          [int]
           See Error.py for all return codes.
        '''
        # Print verboosity
        octools.printv(f"Running '{prepare_ligand}' for '{self.inputLigandPath}'.")
        # If True, use openbabel
        if useOpenBabel:
            return octools.convertMols(self.inputLigandPath, self.preparedLigand)
        else:
            return octools.run(self.prepareLigandCmd, logFile=logFile, cwd=os.path.dirname(self.inputLigandPath))
        return None

    def run_prepare_receptor(self, logFile = "", useOpenBabel = False):
        '''
        Run 'prepare_receptor4' or openbabel to prepare the receptor.
        Input:
          logFile      [list(string)] DEFAULT: ""    - Path to the logFile. If empty, suppress the output
          useOpenBabel [list(string)] DEFAULT: False - If True will try to run openbabel, otherwise will run prepare_receptor4
        Return:
          [int]
           See Error.py for all return codes.
        '''
        # Print verboosity
        octools.printv(f"Running '{prepare_receptor}' for '{self.inputReceptorPath}'.")
        # If True, use openbabel
        if useOpenBabel:
            return octools.convertMols(self.inputReceptorPath, self.preparedReceptor)
        else:
            return octools.run(self.prepareReceptorCmd, logFile=logFile, cwd=os.path.dirname(self.inputReceptorPath))
        return None

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
        print(f"Input receptor:              '{self.inputReceptor if self.inputReceptor else '-' }'")
        print(f"Input receptor path:         '{self.inputReceptorPath if self.inputReceptorPath else '-' }'")
        print(f"Prepared receptor path:      '{self.preparedReceptor if self.preparedReceptor else '-' }'")
        print(f"Prepared receptor command:   '{' '.join(self.prepareReceptorCmd) if self.prepareReceptorCmd else '-' }'")
        print(f"Input ligand:                '{self.inputLigand if self.inputLigand else '-' }'")
        print(f"Input ligand path:           '{self.inputLigandPath if self.inputLigandPath else '-' }'")
        print(f"Prepared ligand path:        '{self.preparedLigand if self.preparedLigand else '-' }'")
        print(f"Prepared ligand command:     '{' '.join(self.prepareLigandCmd) if self.prepareLigandCmd else '-' }'")
        print(f"Conversion to mol2 log path: '{self.convert2mol2log if self.convert2mol2log else '-' }'")
        print(f"Vina execution log path:     '{self.vinaLog if self.vinaLog else '-' }'")
        print(f"Vina output path:            '{self.outputVina if self.outputVina else '-' }'")
        print(f"Vina command:                '{' '.join(self.vinaCmd) if self.vinaCmd else '-' }'")
        return

# Functions
###############################################################################
## Private ##

## Public ##
def box_to_vina(boxFile, confFile, receptor):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
      boxFile   [string] - Path to the box file.
      confFile  [string] - Path to the conf file.
      receptor  [string] - Receptor name to be used in conf file.
    Return:
      [int]
       See Error.py for all return codes.
    '''
    octools.printv(f"Converting the box file '{boxFile}' to Vina conf file as '{confFile}' file.")
    # Test if the file boxFile exists
    if not os.path.exists(boxFile):
        return errors.file_do_not_exist(message=f"The box file in the path {boxFile} does not exists! Please ensure that the file exsits and the path is correct. If you have no box file, try to run the function 'runprank' from the 'runprank' library to create it before calling this function or creating a Vina class object.", level="error")
    # List to hold all the data
    lines = []

    try:
        # Open the box file
        with open(str(boxFile), "r") as box_file:
            # For each line in the file
            for line in box_file:
                # If it starts with REMARK
                if line.startswith("REMARK"):
                    # Slice the line in right positions
                    lines.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

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

def run_prepare_ligand(inputLigandPath, outputLigand, logFile=""):
    '''
    Prepares the ligand using 'prepare_ligand' from MGLTools suite.
    Input:
      inputLigandPath  [string]                   - Path to the input ligand file.
      outputLigand     [string]                   - Path to the output ligand file.
      logFile          [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      [int]
       See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [pythonsh, prepare_ligand, "-l", inputLigandPath, "-C", "-o", outputLigand]
    # Print verboosity
    octools.printv(f"Running '{prepare_ligand}' for '{inputLigandPath}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_prepare_receptor(inputReceptorPath, outputReceptor, logFile=""):
    '''
    Convert a box (DUDE like format) to vina input.
    Input:
      inputReceptorPath  [string]                   - Path to the input receptor file.
      outputReceptor     [string]                   - Path to the output receptor file.
      logFile            [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      [int]
       See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [pythonsh, prepare_receptor, "-r", inputReceptorPath, "-o", outputReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]
    # Print verboosity
    octools.printv(f"Running '{prepare_receptor}' for '{inputReceptorPath}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_vina(confFile, ligand, outpath, logFile=""):
    '''
    Run vina.
    Input:
      confFile [string]                   - Path to the config file.
      ligand   [string]                   - Path to the ligand file.
      outpath  [string]                   - Path to the receptor file.
      logFile  [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      [int]
       See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [vina, "--config", confFile, "--ligand", ligand, "--out", outpath, "--cpu", "1"]
    # Print verboosity
    octools.printv(f"Running vina using the '{confFile}' configurations.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def generate_vina_files_database(path, protein, prankPath = ""):
    '''
    Generate all vina required files for provided protein.
    Input:
     path      [string]             - Input path.
     protein   [string]             - Protein path.
     prankPath [string] DEFAULT: "" - If the prank dir is different than path, pass it here.
    Return:
      -
    '''
    # Parameterize the vina and p2rank paths
    vinaPath = f"{path}/vinaFiles"
    # Check if prankPath is an empty string
    if prankPath == "":
      # Set is as the path + p2rank
      prankPath = f"{path}/p2rank"
    # Create the vina folder inside protein's directory
    _ = octools.safe_create_dir(vinaPath)
    # Find all boxes
    boxes = glob(f"{prankPath}/box*.pdb")
    # For each box
    for box in boxes:
        # Get box name
        boxName = os.path.basename(box)
        # Get box id
        boxId = boxName.split(".")[0].replace("box", "").replace(".pdb", "")
        # Parameterize the box folder
        boxFolder = f"{vinaPath}/{boxId}"
        # Create vina execution folder
        _ = octools.safe_create_dir(boxFolder)
        confPath = f"{boxFolder}/conf_vina.txt"
        box_to_vina(box, confPath, protein)
    return None

def read_vina_log(path):
    '''
    Read the vina log path, returning a pd.dataframe with data from complexes.
    Input:
      path [string] - Path to the vina output log file.
    Return:
      [pd.dataframe]
    '''
    # Check if file exists
    if os.path.isfile(path):
        # Open the log file
        with open(path, "r") as f:
            # Read ALL the lines in file (there should not be lots of lines, so no problem)
            lines = f.readlines()
        # Create a dataframe to store the info
        df = pd.DataFrame(columns=['mode','affinity','rmsd_lb_best_mode','rmsd_ub_best_mode'])
        # For each line from the end to the beggining (reverse iteration since the intresting data is in the end of the file)
        for i in range(len(lines)-1, -1, -1):
            # If the line starts with a -
            if lines[i].startswith("-"):
                # Stop iteration, because it does not contain useful information and neither the upper lines do
                break
            # If useless information is in our way ignore it
            if "Writing output ... done." in lines[i]:
                continue
            try:
                # Add the reversed list to the end of
                df.loc[len(df), df.columns] = lines[i].strip().split()
            except Exception as e:
                octools.print_error(f"Problems while reading file '{path}'. Error: {e}")
                octools.print_error_log(f"Problems while reading file '{path}'. Error: {e}", f"{logdir}/vina_read_log_ERROR.log")
        # Return the df reversing the order and reseting the index
        return df.reindex(index=df.index[::-1]).reset_index(drop=True)
    # Throw an error
    return errors.file_do_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.")
