#!/usr/lib/python3

# Imports
###############################################################################
import os
import sys
import shutil
import tarfile
import datetime
import subprocess

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
Sets of classes and functions that are used to prepare dock6 files and run it.

They are imported as:

import OCDocker.PLANTS as ocplants
'''

# Classes
###############################################################################
class PLANTS:
    """
    PLANTS object with methods for easy run.
    """
    def __init__(self, configPath, boxFile, receptor, preparedReceptorPath, ligand, preparedLigandPath, plantsLog, outputPlants, name=""):
        self.name = str(name)
        self.config = str(configPath)
        self.bindingSiteCenter, self.bindingSiteRadius = __get_binding_site(boxFile)
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
        self.plantsLog = str(plantsLog)
        self.outputPlants = str(outputPlants)
        self.plantsCmd = self.__plants_cmd()
        # Create the box
        self.__write_config_file()

    ## Private ##
    def __get_binding_site(boxFile, spacing = 0.33):
        '''
        Get the binding site from a box file.
        Input:
          boxFile   [string]               - Path to the box file.
          spacing   [float]  DEFAULT: 0.33 - Extra spacing
        Return:
          [tuple of mixed tuple of floats and floats]
           Binding center (x, y, z) and binding radius.
        '''
        octools.printv(f"Parsing '{boxFile}' to binding center data.")
        # Test if the file boxFile exists
        if not os.path.exists(boxFile):
            return errors.file_do_not_exist(message=f"The box file in the path {boxFile} does not exists! Please ensure that the file exsits and the path is correct. If you have no box file, try to run the function 'runprank' from the 'runprank' library to create it before calling this function or creating a PLANTS class object.", level="error")
        # List to hold all the data
        lines = []
        try:
            # Open the box file
            with open(str(boxFile), "r") as box_file:
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
        # Get the biggest value among the coordinates, divide for 2 (because its a size)
        radius = (max(lines[1][0], lines[1][1], lines[1][2])/2)
        # Add some extra space
        radius += spacing * radius
        # Return the data
        return ((lines[0][0], lines[0][1], lines[0][2]), radius)

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
          ligand [string/ocl.Ligand] - The path for the ligand or its ocl.Ligand object.
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

    def __plants_cmd(self):
        '''
        Generate the vina command.
        Input:
          -
        Return:
          list[string] - List of strings of the command.
        '''
        cmd = [plants, "--mode", "screen", self.config]
        return cmd

    def __prepare_ligand_cmd(self):
        '''
        Generate the prepare ligand command.
        Input:
          -
        Return:
          list[string] - List of strings of the command.
        '''
        cmd = [spores, "--mode", "complete", self.inputLigandPath, self.preparedLigand]
        return cmd

    def __prepare_receptor_cmd(self):
        '''
        Generate the prepare receptor command.
        Input:
          -
        Return:
          list[string] - List of strings of the command.
        '''
        cmd = [spores, "--mode", "complete", self.inputReceptor, self.preparedReceptor]
        return cmd

    def __write_config_file(self):
        '''
        Write the config file.
        Input:
          -
        Return:
          [int]
           See Error.py for all return codes.
        '''
        try:
            with open(self.config, "w") as f:
                #f.write("# scoring function and search settings")
                f.write("scoring_function chemplp")
                f.write(f"search_speed {plants_search_speed}")
                #f.write("# input")
                f.write(f"protein_file {self.preparedReceptor}")
                f.write(f"ligand_file {self.inputReceptor}")
                #f.write("# output")
                f.write(f"output_dir {outputPlants}")
                #f.write("# write single mol2 files (e.g. for RMSD calculation)")
                f.write("write_multi_mol2 0")
                #f.write("# binding site definition")
                f.write(f"bindingsite_center {self.bindingSiteCenter[0]} {self.bindingSiteCenter[1]} {self.bindingSiteCenter[2]}")
                f.write(f"bindingsite_radius {self.bindingSiteRadius}")
                f.write("# cluster algorithm")
                f.write(f"cluster_structures {plants_cluster_structures}")
                f.write(f"cluster_rmsd {plants_cluster_rmsd}")
        except Exception as e:
            return errors.write_file(f"Problems while writing the file {self.config}: {e}")

    ## Public ##
    def run_plants(self, logFile = ""):
        '''
        Run plants.
        Input:
          logFile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
        Return:
          [int]
           See Error.py for all return codes.
        '''
        # Print verboosity
        octools.printv(f"Running PLANTS using the '{self.config}' configurations.")
        return octools.run(self.plantsCmd, logFile=self.plantsLog)

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
        octools.printv(f"Running '{spores}' for '{self.inputLigandPath}'.")
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
        octools.printv(f"Running '{spores}' for '{self.inputReceptorPath}'.")
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
def run_prepare_ligand(inputLigandPath, outputLigand, logFile=""):
    '''
    Prepares the ligand using 'prepare_ligand4' from MGLTools suite.
    Input:
      inputLigandPath  [string]                   - Path to the input ligand file.
      outputLigand     [string]                   - Path to the output ligand file.
      logFile          [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      [int]
       See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [spores, "--mode", "complete", inputLigandPath, outputLigand]
    # Print verboosity
    octools.printv(f"Running '{spores}' for '{inputLigandPath}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_prepare_receptor(inputReceptorPath, outputReceptor, logFile=""):
    '''
    Prepares the receptor using 'prepare_receptor4' from MGLTools suite.
    Input:
      inputReceptorPath  [string]                   - Path to the input receptor file.
      outputReceptor     [string]                   - Path to the output receptor file.
      logFile            [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      [int]
       See Error.py for all return codes.
    '''
    # Create the command list
    cmd = [spores, "--mode", "complete", inputReceptorPath, outputReceptor]
    # Print verboosity
    octools.printv(f"Running '{spores}' for '{inputReceptorPath}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_plants(confFile, ligand, outpath, logFile=""):
    '''
    Run PLANTS.
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
    cmd = [plants, "--mode", "screen", confFile]
    # Print verboosity
    octools.printv(f"Running PLANTS using the '{confFile}' configurations.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

"""def generate_vina_files_database(path, protein):
    '''
    Generate all vina required files for provided protein.
    Input:
     path    [string] - Input path.
     protein [string] - Protein path.
    Return:
      -
    '''
    # Parameterize the vina and p2rank paths
    vinaPath = f"{path}/vinaFiles"
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

    return"""
