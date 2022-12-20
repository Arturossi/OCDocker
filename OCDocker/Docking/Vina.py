#!/usr/lib/python3

# Imports
###############################################################################
import errno
import os

import numpy as np
import pandas as pd

from Bio.PDB import *
from typing import Dict, List, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
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

import OCDocker.Docking.Vina as ocvina
'''

# Classes
###############################################################################
class Vina:
    """Vina object with methods for easy run."""
    def __init__(self, configPath: str, boxFile: str, receptor: ocr.Receptor, preparedReceptorPath: str, ligand: ocl.Ligand, preparedLigandPath: str, vinaLog: str, outputVina: str, name: str = "", overwriteConfig: bool = False) -> None:
        '''Constructor of the class Vina.
        
        Parameters
        ----------
        configPath : str
            The path for the config file.
        boxFile : str
            The path for the box file.
        receptor : ocr.Receptor
            The receptor object.
        preparedReceptorPath : str
            The path for the prepared receptor.
        ligand : ocl.Ligand
            The ligand object.
        preparedLigandPath : str
            The path for the prepared ligand.
        vinaLog : str
            The path for the vina log file.
        outputVina : str
            The path for the vina output files.
        name : str, optional
            The name of the vina object, by default "".

        Returns
        -------
        None

        Raises
        ------
        None
        '''

        self.name = str(name)
        self.config = str(configPath)
        self.boxFile = str(boxFile)

        # Receptor
        if type(receptor) == ocr.Receptor:
            self.inputReceptor = receptor
        else:
            errors.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'ocr.Receptor' but got {type(receptor)} instead.", level="error")
            return None

        self.inputReceptorPath = self.__parse_receptor_path(receptor)
        self.preparedReceptor = str(preparedReceptorPath)
        self.prepareReceptorCmd = [pythonsh, prepare_receptor, "-r", self.inputReceptorPath, "-o", self.preparedReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]

        # Ligand
        self.preparedLigand = str(preparedLigandPath)
        # Check the type of the ligand
        if type(ligand) == ocl.Ligand:
            self.inputLigand = ligand
            # Create the plantsFiles folder
            _ = octools.safe_create_dir(os.path.join(ligand.path, "plantsFiles"))
        else:
            errors.wrong_type(f"The ligand '{ligand}' has not a supported type. Expected 'ocl.Ligand' but got {type(ligand)} instead.", level="error")
            return None

        self.inputLigandPath = self.__parse_ligand_path(ligand)
        self.prepareLigandCmd = [pythonsh, prepare_ligand, "-l", self.inputLigandPath, "-C", "-o", self.preparedLigand]

        # Vina
        self.vinaLog = str(vinaLog)
        self.outputVina = str(outputVina)
        self.vinaCmd = [vina, "--config", self.config, "--ligand", self.preparedLigand, "--out", self.outputVina, "--cpu", "1"]
        
        # Check if the config file exists or if it should be overwritten
        if not os.path.isfile(self.config) or overwriteConfig:
            # Create the box
            box_to_vina(self.boxFile, self.config, self.preparedReceptor)

    ## Private ##
    def __parse_receptor_path(self, receptor: Union[str, ocr.Receptor]) -> str:
        '''Parse the receptor path, handling its type.
        
        Parameters
        ----------
        receptor : str | ocr.Receptor
            The path for the receptor or its receptor object.

        Returns
        -------
        str
            The receptor path.

        Raises
        ------
        None
        '''

        # Check the type of receptor variable
        if type(receptor) == ocr.Receptor:
            return receptor.path  # type: ignore
        elif type(receptor) == str:
            # Since is a string, check if the file exists
            if os.path.isfile(receptor): # type: ignore
                # Exists! Return it!
                return receptor # type: ignore
            else:
                _ = errors.file_do_not_exist(message=f"The receptor '{receptor}' has not a valid path.", level="error")
                return ""

        _ = errors.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'string' or 'ocr.Receptor' but got {type(receptor)} instead.", level = "error")
        return ""

    def __parse_ligand_path(self, ligand: Union[str, ocl.Ligand]) -> str:
        '''Parse the ligand path, handling its type.
        
        Parameters
        ----------
        ligand : str | ocl.Ligand
            The path for the ligand or its ocl.Ligand object.

        Returns
        -------
            The ligand path. If fails, return an empty string.
        
        Raises
        ------
        None
        '''

        # Check the type of ligand variable
        if type(ligand) == ocl.Ligand:
            return ligand.path # type: ignore
        elif type(ligand) == str:
            # Since is a string, check if the file exists
            if os.path.isfile(ligand): # type: ignore
                # Exists! Process it then!
                return self.__process_ligand(ligand) # type: ignore
            else:
                _ = errors.file_do_not_exist(message=f"The ligand '{ligand}' has not a valid path.", level="error")
                return ""

        _ = errors.wrong_type(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.", level="error")
        return ""

    def __process_ligand(self, ligandPath: str) -> str:
        '''Process the ligand to output to mol2 if needed.

        Parameters
        ----------
        ligandPath : str
            The path for the ligand.

        Returns
        -------
        str
            The Path of the ligand with mol2 extension.

        Raises
        ------
        None
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
        octools.convertMols(ligandPath, outputLigandPath)

        return outputLigandPath

    ## Public ##

    def read_vina_log(self) -> Union[Dict[str, List[Union[str, float]]], int]:
        '''Read the vina log path, returning a pd.dataframe with data from complexes.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, List[Union[str, float]]] | int
            A dictionary with the data from the vina log file. If any error occurs, it will return the exit code of the command (based on the Error.py code table).
        
        Raises
        ------
        None
        '''

        return read_vina_log(self.vinaLog)

    def run_vina(self) -> Union[int, Tuple[int, str]]:
        '''Run vina.

        Parameters
        ----------
        None

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        
        Raises
        ------
        None
        '''

        # Print verboosity
        octools.printv(f"Running vina using the '{self.config}' configurations.")
        return octools.run(self.vinaCmd, logFile=self.vinaLog)

    def run_prepare_ligand(self, logFile: str = "", useOpenBabel: bool = False) -> Union[int, str, Tuple[int, str]]:
        '''Run 'prepare_ligand4' or openbabel to prepare the ligand.

        Parameters
        ----------
        logFile : str
            Path to the logFile. If empty, suppress the output.
        useOpenBabel : bool
            If True, use openbabel instead of prepare_ligand4.
        
        Returns
        -------
        int | str | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command. If fails, return the file extension. 
        
        Raises
        ------
        None
        '''

        # Print verboosity
        octools.printv(f"Running '{prepare_ligand}' for '{self.inputLigandPath}'.")
        # If True, use openbabel
        if useOpenBabel:
            return octools.convertMols(self.inputLigandPath, self.preparedLigand)
        return octools.run(self.prepareLigandCmd, logFile=logFile, cwd=os.path.dirname(self.inputLigandPath))

    def run_prepare_receptor(self, logFile:str = "", useOpenBabel:bool = False) -> Union[int, str, Tuple[int, str]]:
        '''Run 'prepare_receptor4' or openbabel to prepare the receptor.

        Parameters
        ----------
        logFile : str
            Path to the logFile. If empty, suppress the output.
        useOpenBabel : bool
            If True, use openbabel instead of prepare_receptor4.

        Returns
        -------
        int | str | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command. If fails, return the file extension.

        Raises
        ------
        None
        '''

        # Print verboosity
        octools.printv(f"Running '{prepare_receptor}' for '{self.inputReceptorPath}'.")
        # If True, use openbabel
        if useOpenBabel:
            return octools.convertMols(self.inputReceptorPath, self.preparedReceptor)
        return octools.run(self.prepareReceptorCmd, logFile=logFile, cwd=os.path.dirname(self.inputReceptorPath))

    def print_attributes(self) -> None:
        '''Print the class attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
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
        print(f"Vina execution log path:     '{self.vinaLog if self.vinaLog else '-' }'")
        print(f"Vina output path:            '{self.outputVina if self.outputVina else '-' }'")
        print(f"Vina command:                '{' '.join(self.vinaCmd) if self.vinaCmd else '-' }'")

        return None

# Functions
###############################################################################
## Private ##

## Public ##
def box_to_vina(boxFile: str, confFile: str, receptor: str) -> int:
    '''Convert a box (DUDE like format) to vina input.

    Parameters
    ----------
    boxFile : str
        The path to the box file.
    confFile : str
        The path to the vina configuration file.
    receptor : str
        The path to the receptor file.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    octools.printv(f"Converting the box file '{boxFile}' to Vina conf file as '{confFile}' file.")
    # Test if the file boxFile exists
    if not os.path.exists(boxFile):
        return errors.file_do_not_exist(message=f"The box file in the path {boxFile} does not exist! Please ensure that the file exists and the path is correct.", level="error")
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

def run_prepare_ligand(inputLigandPath: str, outputLigand: str, logFile: str = ""):
    '''Prepares the ligand using 'prepare_ligand' from MGLTools suite.

    Parameters
    ----------
    inputLigandPath : str
        The path to the input ligand.
    outputLigand : str
        The path to the output ligand.
    logFile : str
        The path to the log file. If empty, suppress the output.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Create the command list
    cmd = [pythonsh, prepare_ligand, "-l", inputLigandPath, "-C", "-o", outputLigand]
    # Print verboosity
    octools.printv(f"Running '{prepare_ligand}' for '{inputLigandPath}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile, cwd = os.path.dirname(inputLigandPath))

def run_prepare_receptor(inputReceptorPath: str, outputReceptor: str, logFile: str = ""):
    '''Convert a box (DUDE like format) to vina input.

    Parameters
    ----------
    inputReceptorPath : str
        The path to the input receptor file.
    outputReceptor : str
        The path to the output receptor file.
    logFile : str
        The path to the log file. If empty, suppress the output.
    
    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Create the command list
    cmd = [pythonsh, prepare_receptor, "-r", inputReceptorPath, "-o", outputReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]
    # Print verboosity
    octools.printv(f"Running '{prepare_receptor}' for '{inputReceptorPath}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_vina(confFile: str, ligand: str, outpath: str, logFile: str = ""):
    '''Run vina.

    Parameters
    ----------
    confFile : str
        The path to the vina configuration file.
    ligand : str
        The path to the ligand file.
    outpath : str
        The path to the output file.
    logFile : str
        The path to the log file. If empty, suppress the output.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''
    
    # Create the command list
    cmd = [vina, "--config", confFile, "--ligand", ligand, "--out", outpath, "--cpu", "1"]
    # Print verboosity
    octools.printv(f"Running vina using the '{confFile}' configurations.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def generate_vina_files_database(path: str, protein: str, boxPath: str = "") -> None:
    '''Generate all vina required files for provided protein.

    Parameters
    ----------
    path : str
        The path to the folder where the files will be generated.
    protein : str
        The path of the protein.
    boxPath : str
        The path to the box file. If empty, it will try to look for a p2rank dir inside <path>.
    
    Returns
    -------
    None

    Raises
    ------
    None
    '''
    
    # Parameterize the vina and p2rank paths
    vinaPath = f"{path}/vinaFiles"
    # Check if boxPath is an empty string
    if boxPath == "":
      # Set is as the path + p2rank
      boxPath = f"{path}/p2rank"
    # Create the vina folder inside protein's directory
    _ = octools.safe_create_dir(vinaPath)
    
    # TODO: Implement multiple box support here
    box = f"{boxPath}/box0.pdb"
    confPath = f"{vinaPath}/conf_vina.conf"
    box_to_vina(box, confPath, protein)

    return None

def read_vina_log(path: str) -> Dict[str, List[Union[str, float]]]:
    '''Read the vina log path, returning the data from complexes.

    Parameters
    ----------
    path : str
        The path to the vina log file.

    Returns
    -------
    Dict[str, List[str | float]]
        A dictionary with the data from the vina log file.

    Raises
    ------
    None
    '''

    # Check if file exists
    if os.path.isfile(path):
        # Create a dictionary to store the info
        data = {"vina_pose": [], "vina_affinity": []}

        # Initiate the last read line as empty
        lastReadLine = ""

        # Try except to avoid broken pipe errors
        try:
            # Read the file reversely
            for line in octools.lazyread_reverse_order_mmap(path):
                # If a stop line is found, means that the last read line is the one that is wanted
                if line.startswith("-----+"):
                    # Split the last line
                    lastLine = lastReadLine.split()
                    data["vina_pose"].append(lastLine[0])
                    data["vina_affinity"].append(lastLine[1])
                    break

                # Assign the last read line as the current line
                lastReadLine = line
        except IOError as e:
            if e.errno == errno.EPIPE:
                octools.print_error(f"Problems while reading file '{path}'. Error: {e}")
                octools.print_error_log(f"Problems while reading file '{path}'. Error: {e}", f"{logdir}/vina_read_log_ERROR.log")
        
        # Check if the len of the data["vina_affinity"] is 0
        if len(data["vina_pose"]) == 0:
            data["vina_pose"].append(np.NaN)
            data["vina_affinity"].append(np.NaN)

        # Return the df reversing the order and reseting the index
        return data

    # Throw an error
    _ = errors.file_do_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.")

    # Return a dict with a NaN value
    return {"vina_pose": [np.NaN], "vina_affinity": [np.NaN]}
