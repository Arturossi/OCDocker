#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to prepare smina files and run it.

They are imported as:

import OCDocker.Docking.Smina as ocsmina
'''

# Imports
###############################################################################
import errno
import json
import os

import numpy as np

from glob import glob
from typing import Dict, List, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.IO as ocio
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun
import OCDocker.Toolbox.Validation as ocvalidation
from OCDocker.Docking.BaseVinaLike import (
    read_smina_log as read_log,
    read_smina_rescoring_log as read_rescoring_log,
    generate_smina_digest as generate_digest,
    get_smina_docked_poses as get_docked_poses,
)

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Licensed under the Apache License, Version 2.0 (January 2004)
See: http://www.apache.org/licenses/LICENSE-2.0

Commercial use requires a separate license.  
Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################
class Smina:
    """Smina object with methods for easy run."""
    def __init__(self, configPath: str, boxFile: str, receptor: ocr.Receptor, preparedReceptorPath: str, ligand: ocl.Ligand, preparedLigandPath: str, sminaLog: str, outputSmina: str, name: str = "", overwriteConfig: bool = False) -> None:
        '''Constructor of the class Smina.

        Parameters
        ----------
        configPath : str
            Path to the configuration file.
        boxFile : str
            The path for the box file.
        receptor : ocr.Receptor
            The receptor object.
        preparedReceptorPath : str 
            Path to the prepared receptor.
        ligand : ocl.Ligand
            The ligand object.
        preparedLigandPath : str
            Path to the prepared ligand.
        sminaLog : str
            Path to the smina log file.
        outputSmina : str
            Path to the output smina file.
        name : str, optional
            Name of the smina object, by default "".
        overwriteConfig : bool, optional
            If the config file should be overwritten, by default False.

        Returns
        -------
        None
        '''

        self.name = str(name)
        self.config = str(configPath)
        self.boxFile = str(boxFile)
        
        # Receptor
        if type(receptor) == ocr.Receptor:
            self.inputReceptor = receptor
        else:
            ocerror.Error.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR) # type: ignore
            return None
        
        # Check if the folder where the configPath is located exists (remove the file name from the path)
        _ = ocff.safe_create_dir(os.path.dirname(self.config))

        self.inputReceptorPath = self.__parse_receptor_path(receptor)
        
        self.preparedReceptor = str(preparedReceptorPath)
        self.prepareReceptorCmd = [pythonsh, prepare_receptor, "-r", self.inputReceptorPath, "-o", self.preparedReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]
        #self.prepareReceptorCmd = [obabel, self.inputReceptorPath, "-xr", "-O", self.preparedReceptor]

        # Ligand
        if type(ligand) == ocl.Ligand:
            self.inputLigand = ligand
            # Create the sminaFiles folder
            _ = ocff.safe_create_dir(os.path.join(os.path.dirname(ligand.path), "sminaFiles"))
        else:
            ocerror.Error.wrong_type(f"The ligand '{ligand}' has not a supported type. Expected 'ocl.Ligand' but got {type(ligand)} instead.", level = ocerror.ReportLevel.ERROR) # type: ignore
            return None

        self.inputLigandPath = self.__parse_ligand_path(ligand)
        self.preparedLigand = str(preparedLigandPath)
        self.prepareLigandCmd = [pythonsh, prepare_ligand, "-l", self.inputLigandPath, "-C", "-o", self.preparedLigand]
        #self.prepareLigandCmd = [obabel, self.inputLigandPath, "-O", self.preparedLigand]

        # Smina
        self.sminaLog = str(sminaLog)
        self.outputSmina = str(outputSmina)
        self.sminaCmd = self.__smina_cmd()
        
        # Check if config file exists to avoid useless processing
        if not os.path.isfile(self.config) or overwriteConfig:
            # Create the conf file
            gen_smina_conf(self.boxFile, self.config, self.preparedReceptor)

        # Aliases
        ############
        self.run_docking = self.run_smina

    ## Private ##
    def __parse_receptor_path(self, receptor: Union[str, ocr.Receptor]) -> str:
        '''Parse the receptor path, handling its type.

        Parameters
        ----------
        receptor : ocr.Receptor | str
            The path for the receptor or its receptor object.

        Returns
        -------
        str
            The receptor path.
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
                _ = ocerror.Error.file_not_exist(message=f"The receptor '{receptor}' has not a valid path.", level = ocerror.ReportLevel.ERROR) # type: ignore
                return ""

        _ = ocerror.Error.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'string' or 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR) # type: ignore
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
                _ = ocerror.Error.file_not_exist(message=f"The ligand '{ligand}' has not a valid path.", level = ocerror.ReportLevel.ERROR) # type: ignore
                return ""

        _ = ocerror.Error.wrong_type(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.", level = ocerror.ReportLevel.ERROR) # type: ignore
        return ""

    def __smina_cmd(self) -> List[str]:
        '''Generate the smina command.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            The smina command.
        '''

        cmd = [smina, "--config", self.config, "--ligand", self.preparedLigand]#, "--autobox_ligand", self.preparedLigand]

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

    ## Public ##
    def read_log(self, onlyBest: bool = False) -> Dict[int, Dict[int, float]]:
        '''Read the SMINA log path, returning a dict with data from complexes.

        Parameters
        ----------
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default False.

        Returns
        -------
        Dict[int, Dict[int, float]]
            A dictionary with the data from the SMINA log file. If any error occurs, it will return the exit code of the command (based on the Error.py code table).
        '''

        return read_log(self.sminaLog, onlyBest = onlyBest) # type: ignore

    def run_smina(self, logFile: str = "") -> Union[int, Tuple[int, str]]:
        '''Run smina.

        Parameters
        ----------
        logFile : str
            The path for the log file.
        
        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table).   
        '''

        return ocrun.run(self.sminaCmd, logFile=logFile)

    def run_prepare_ligand_from_cmd(self, logFile: str = "") -> Union[int, Tuple[int, str]]:
        '''Run obabel convert ligand to pdbqt using the 'self.inputLigandPath' attribute. [DEPRECATED]

        Parameters
        ----------
        logFile : str
            The path for the log file.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return ocrun.run(self.prepareLigandCmd, logFile=logFile)

    def run_prepare_ligand(self) -> Union[int, Tuple[int, str]]:
        '''Run the convert ligand command to pdbqt.

        Parameters
        ----------
        None

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return run_prepare_ligand(self.inputLigandPath, self.preparedLigand)

    def run_prepare_receptor_from_cmd(self, logFile: str = "") -> Union[int, Tuple[int, str]]:
        '''Run obabel convert receptor to pdbqt script using the 'self.prepareReceptorCmd' attribute. [DEPRECATED]

        Parameters
        ----------
        logFile : str
            The path for the log file.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return ocrun.run(self.prepareReceptorCmd, logFile=logFile)

    def run_prepare_receptor(self) -> Union[int, Tuple[int, str]]:
        '''Run obabel convert receptor to pdbqt using the openbabel python library.

        Parameters
        ----------
        None

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return run_prepare_receptor(self.inputReceptorPath, self.preparedReceptor)
    
    def run_rescore(self, outPath: str, logFile: str = "", skipDefaultScoring: bool = False, overwrite = False) -> None:
        '''Run smina to rescore the ligand.

        Parameters
        ----------
        outPath : str
            Path to the output folder.
        logFile : str, optional
            Path to the logFile. If empty, suppress the output. By default "".
        skipDefaultScoring : bool, optional
            If True, skip the default scoring function. By default False.
        overwrite : bool, optional
            If True, overwrite the logFile. By default False.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        # Set the splitLigand as True
        splitLigand = True

        # For each scoring function
        for scoring_function in smina_scoring_functions:
            # If is the default scoring function and skipDefaultScoring is True
            if not (scoring_function == smina_scoring and skipDefaultScoring):
                # Run smina to rescore
                _ = run_rescore(self.config, self.outputSmina, outPath, scoring_function, logFile = logFile, splitLigand = splitLigand, overwrite = overwrite)

                # Set the splitLigand as False (to avoid running it again without need)
                splitLigand = False

        return None
    
    def get_docked_poses(self) -> List[str]:
        '''Get the paths for the docked poses.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            A list with the paths for the docked poses.
        '''

        return get_docked_poses(os.path.dirname(self.outputSmina))

    def get_input_ligand_path(self) -> str:
        ''' Get the input ligand path.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The input ligand path.
        '''

        return os.path.dirname(self.inputLigandPath)
    
    def get_input_receptor_path(self) -> str:
        ''' Get the input receptor path.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The input receptor path.
        '''

        return os.path.dirname(self.inputReceptorPath)

    def read_rescore_logs(self, outPath: str, onlyBest: bool = False) -> Dict[str, List[Union[str, float]]]:
        ''' Reads the data from the rescore log files.

        Parameters
        ----------
        outPath : str
            Path to the output folder where the rescoring logs are located.
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default False.

        Returns
        -------
        Dict[str, List[Union[str, float]]]
            A dictionary with the data from the rescore log files.
        '''

        # Get the rescore log paths
        rescoreLogPaths = get_rescore_log_paths(outPath)

        # Call the function
        return read_rescore_logs(rescoreLogPaths, onlyBest = onlyBest)

    def split_poses(self, outPath: str = "", logFile: str = "") -> int:
        '''Split the ligand resulted from smina into its poses.

        Parameters
        ----------
        outPath : str, optional
            Path to the output folder. By default "". If empty, the poses will be saved in the same folder as the vina output.
        logFile : str, optional
            Path to the logFile. If empty, suppress the output. By default "".

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).
        '''

        # If the outPath is empty
        if not outPath:
            # Set the outPath as the same folder as the smina output
            outPath = os.path.dirname(self.outputSmina)

        return ocmolproc.split_poses(self.outputSmina, self.inputLigand.name, outPath, logFile = logFile, suffix = "_split_") # type: ignore

    def print_attributes(self) -> None:
        '''Print the class attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print(f"Name:                        '{self.name if self.name else '-' }'")
        print(f"Config path:                 '{self.config if self.config else '-' }'")
        print(f"Input receptor:              '{self.inputReceptor if self.inputReceptor else '-' }'")
        print(f"Input receptor path:         '{self.inputReceptorPath if self.inputReceptorPath else '-' }'")
        print(f"Prepared receptor path:      '{self.preparedReceptor if self.preparedReceptor else '-' }'")
        print(f"Prepared receptor command:   '{' '.join(self.prepareReceptorCmd) if self.prepareReceptorCmd else '-' }'")
        print(f"Input ligand:                '{self.inputLigand if self.inputLigand else '-' }'")
        print(f"Input ligand path:           '{self.inputLigandPath if self.inputLigandPath else '-' }'")
        print(f"Prepared ligand path:        '{self.preparedLigand if self.preparedLigand else '-' }'")
        print(f"Prepared ligand command:     '{' '.join(self.prepareLigandCmd) if self.prepareLigandCmd else '-' }'")
        print(f"Smina execution log path:    '{self.sminaLog if self.sminaLog else '-' }'")
        print(f"Smina output path:           '{self.outputSmina if self.outputSmina else '-' }'")
        print(f"Smina command:               '{' '.join(self.sminaCmd) if self.sminaCmd else '-' }'")
        return

# Functions
###############################################################################
## Private ##

## Public ##
def gen_smina_conf(boxFile: str, confFile: str, receptor: str) -> int:
    '''Convert a box (DUDE like format) to smina input.

    Parameters
    ----------
    boxFile : str
        The path to the box file.
    confFile : str
        The path for the conf file.
    receptor : str
        The path for the receptor.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Test if the file boxFile exists
    if not os.path.exists(boxFile):
        return ocerror.Error.file_not_exist(message=f"The box file in the path {boxFile} does not exist! Please ensure that the file exists and the path is correct.", level = ocerror.ReportLevel.ERROR) # type: ignore
    # List to hold all the data
    lines = []

    try:
        # Open the box file
        with open(str(boxFile), 'r') as box_file:
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
        return ocerror.Error.read_file(message=f"Found a problem while reading the box file: {e}", level = ocerror.ReportLevel.ERROR) # type: ignore

    ocprint.printv(f"Creating smina conf file in the path '{confFile}'.")
    try:
        # Now open the conf file to write
        with open(confFile, 'w') as conf_file:
            conf_file.write(f"receptor = {receptor}\n\n")

            if smina_custom_scoring.lower() != "no":
                conf_file.write(f"custom_scoring = {smina_custom_scoring}\n")

            if smina_custom_atoms.lower() != "no":
                conf_file.write(f"custom_atoms = {smina_custom_atoms}\n")

            conf_file.write(f"center_x = {lines[0][0]}\n")
            conf_file.write(f"center_y = {lines[0][1]}\n")
            conf_file.write(f"center_z = {lines[0][2]}\n\n")
            conf_file.write(f"size_x = {lines[1][0]}\n")
            conf_file.write(f"size_y = {lines[1][1]}\n")
            conf_file.write(f"size_z = {lines[1][2]}\n\n")

            if smina_minimize_iters.lower() != "no":
                conf_file.write(f"minimize_iters = {smina_minimize_iters}\n")

            conf_file.write(f"approximation = {smina_approximation}\n")
            conf_file.write(f"factor = {smina_factor}\n")
            conf_file.write(f"force_cap = {smina_force_cap}\n")

            if smina_user_grid.lower() != "no":
                conf_file.write(f"user_grid = {smina_user_grid}\n")

            if smina_user_grid_lambda.lower() != "no":
                conf_file.write(f"user_grid_lambda = {smina_user_grid_lambda}\n")

            conf_file.write(f"energy_range = {smina_energy_range}\n")
            conf_file.write(f"exhaustiveness = {smina_exhaustiveness}\n")
            conf_file.write(f"num_modes = {smina_num_modes}\n")
    except Exception as e:
        return ocerror.Error.write_file(message=f"Found a problem while opening conf file: {e}.", level = ocerror.ReportLevel.ERROR) # type: ignore

    return ocerror.Error.ok() # type: ignore

def run_prepare_ligand_from_cmd(inputLigandPath: str, preparedLigand: str, logFile: str = "") -> Union[int, Tuple[int, str]]:
    '''Converts the ligand to .pdbqt using obabel. [DEPRECATED]

    Parameters
    ----------
    inputLigandPath : str
        The path for the input ligand.
    preparedLigand : str
        The path for the prepared ligand.
    logFile : str
        The path for the log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    # Create the command list
    cmd = [obabel, inputLigandPath, "-O", preparedLigand]

    # Run the command
    return ocrun.run(cmd, logFile=logFile)

def run_prepare_ligand(inputLigandPath: str, preparedLigand: str) -> Union[int, Tuple[int, str]]:
    '''Run obabel convert ligand to pdbqt using the openbabel python library.

    Parameters
    ----------
    inputLigandPath : str
        The path for the input ligand.
    preparedLigand : str
        The path for the prepared ligand.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    # Find the extension for input and output
    extension = ocvalidation.validate_obabel_extension(inputLigandPath)
    outExtension = os.path.splitext(preparedLigand)[1]

    # Check if the extension is valid
    if type(extension) != str:
        ocprint.print_error(f"Problems while reading the ligand file '{inputLigandPath}'.")
        return extension # type: ignore

    # Discover if the output extension is pdbqt (to warn user if it is not)
    if outExtension != ".pdbqt":
        ocprint.print_warning(f"The output extension is not '.pdbqt', is {outExtension}. This function converts {clrs['r']}ONLY{clrs['n']} to '.pdbqt'. Please pay attention, since this might be a problem in the future for you!")

    try:
        if extension in ["smi", "smiles"]:
            ocprint.print_warning(f"The input ligand is a smiles file, it is supposed that there will be also a mol2 file within the same folder, so I am changing the file extension to '.mol2' to be able to read it.")
            # Change it to mol2 in the inputLigandPath
            # get the path
            inputLigandPath = f"{os.path.dirname(inputLigandPath)}/ligand.mol2"
        
        # Create the command list
        cmd = [pythonsh, prepare_ligand, "-l", inputLigandPath, "-C", "-o", preparedLigand]
        return ocrun.run(cmd, cwd = os.path.dirname(inputLigandPath))
    except Exception as e:
        return ocerror.Error.subprocess(message=f"Error while running ligand conversion using obabel python lib. Error: {e}", level = ocerror.ReportLevel.ERROR) # type: ignore

def run_prepare_receptor_from_cmd(inputReceptorPath: str, outputReceptor: str, logFile: str = "") -> Union[int, Tuple[int, str]]:
    '''Converts the receptor to .pdbqt using obabel. [DEPRECATED]

    Parameters
    ----------
    inputReceptorPath : str
        The path for the input receptor.
    outputReceptor : str
        The path for the output receptor.
    logFile : str
        The path for the log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    # Create the command list
    cmd = [obabel, inputReceptorPath, "-xr", "-O", outputReceptor]
    # Run the command
    return ocrun.run(cmd, logFile=logFile)

def run_prepare_receptor(inputReceptorPath: str, preparedReceptor: str) -> Union[int, Tuple[int, str]]:
    '''Run obabel convert receptor to pdbqt using the openbabel python library.

    Parameters
    ----------
    inputReceptorPath : str
        The path for the input receptor.
    preparedReceptor : str
        The path for the prepared receptor.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    # Find the extension for input and output
    extension = ocvalidation.validate_obabel_extension(inputReceptorPath)
    outExtension = os.path.splitext(preparedReceptor)[1]

    # Check if the extension is valid
    if type(extension) != str:
        ocprint.print_error(f"Problems while reading the receptor file '{inputReceptorPath}'.")
        return extension # type: ignore

    # Discover if the output extension is pdbqt (to warn user if it is not)
    if outExtension != ".pdbqt":
        ocprint.print_warning(f"The output extension is not '.pdbqt', is {outExtension}. This function converts {clrs['r']}ONLY{clrs['n']} to '.pdbqt'. Please pay attention, since this might be a problem in the future for you!")

    return occonversion.convertMols(inputReceptorPath, preparedReceptor) # type: ignore

def run_smina(config: str, preparedLigand: str, outputSmina: str, sminaLog: str, logPath: str) -> Union[int, Tuple[int, str]]:
    '''Convert a box (DUDE like format) to smina input.

    Parameters
    ----------
    config : str
        The path for the config file.
    preparedLigand : str
        The path for the prepared ligand.
    outputSmina : str
        The path for the output smina file.
    sminaLog : str
        The path for the smina log file.
    logPath : str
        The path for the log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
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
    return ocrun.run(cmd, logFile = logPath)

def run_rescore(confFile: str, ligands: Union[List[str], str], outPath: str, scoring_function: str, logFile: str = "", splitLigand: bool = True, overwrite: bool = False) -> None:
    '''Run smina to rescore the ligand.

    Parameters
    ----------
    confFile : str
        The path to the smina configuration file.
    ligands : Union[List[str], str]
        The path to a List of ligand files or the ligand file.
    outPath : str
        The path to the output file.
    scoring_function : str
        The scoring function to use.
    logFile : str, optional
        The path to the log file. If empty, suppress the output. By default "".
    splitLigand : bool, optional
        If True, split the ligand before running smina. By default True.
    overwrite : bool, optional
        If True, overwrite the logFile. By default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Print verboosity
    ocprint.printv(f"Running smina using the '{confFile}' configurations and scoring function '{scoring_function}'.")

    # Check if the ligands is a string
    if isinstance(ligands, str):
        # Convert to list
        ligands = [ligands]
    
    # Ligand name list
    ligandNames = []
    
    # For each ligand
    for ligand in ligands:
        # If need to split the ligand or overwrite is True
        if splitLigand or overwrite:
            # Get the ligand name
            ligandName = os.path.splitext(os.path.basename(ligand))[0]
            
            # Split the ligand
            _ = ocmolproc.split_poses(ligand, ligandName, outPath, logFile = "", suffix = "_split_")

            # Add the ligand name to the list
            ligandNames.append(ligandName)
        
    # If splitLigand or overwrite is True means that it is needed to get the splited ligands again
    if splitLigand or overwrite:
        # Reset the ligand list
        ligands = []
        # Append the splited ligands to the ligands list (using the glob function)
        ligands.extend(glob(f"{outPath}/*_split_*.pdbqt"))

    # For each ligand in the ligands list (newly splited ligands)
    for ligand in ligands:
        # Get the splited ligand name
        ligand_name = os.path.splitext(os.path.basename(ligand))[0]

        # Create the command list
        cmd = [smina, "--scoring", scoring_function, "--score_only", "--config", confFile, "--ligand", ligand, "--log", f"{outPath}/{ligand_name}_{scoring_function}_rescoring.log", "--cpu", "1"]

        # Create the log file path
        logFile = f"{outPath}/{ligand_name}_{scoring_function}_rescoring.log"

        # If the logFile already exists, check also if the user wants to overwrite it
        if not os.path.isfile(logFile) or overwrite:
            # Print verboosity
            ocprint.printv(f"Running smina using the '{confFile}' configurations and scoring function '{scoring_function}'.")

            # Run the command
            _ = ocrun.run(cmd, logFile = logFile)

            # Check if the logFile exists and it has the string "Estimated Free Energy of Binding" inside it
            if not os.path.isfile(logFile) or not "Estimated Free Energy of Binding" in open(logFile).read():
                # Print an error
                ocprint.print_error(f"Problems while running smina for the ligand '{ligand_name}' using the scoring function '{scoring_function}'.")

                # Remove the file
                _ = ocff.safe_remove_file(logFile)
        else:
            # Print verboosity
            ocprint.printv(f"The log file '{logFile}' already exists. Skipping the smina run for the ligand '{ligand_name}' using the scoring function '{scoring_function}'.")
    
    # Think about how can this be done to deal with multiple runs
    return None


def run_rescore_old(confFile: str, ligands: Union[List[str], str], outPath: str, scoring_function: str, logFile: str = "", splitLigand: bool = True, overwrite: bool = False) -> None:
    '''Run smina to rescore the ligand.

    Parameters
    ----------
    confFile : str
        The path to the smina configuration file.
    ligands : Union[List[str], str]
        The path to a List of ligand files or the ligand file.
    outPath : str
        The path to the output file.
    scoring_function : str
        The scoring function to use.
    logFile : str, optional
        The path to the log file. If empty, suppress the output. By default "".
    splitLigand : bool, optional
        If True, split the ligand before running smina. By default True.
    overwrite : bool, optional
        If True, overwrite the logFile. By default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Print verboosity
    ocprint.printv(f"Running smina using the '{confFile}' configurations and scoring function '{scoring_function}'.")

    # Check if the ligands is a string
    if isinstance(ligands, str):
        # Convert to list
        ligands = [ligands]

    # Ligand name list
    ligandNames = []
    
    # For each ligand
    for ligand in ligands:
        # If need to split the ligand or overwrite is True
        if splitLigand or overwrite:
            # Get the ligand name
            ligandName = os.path.splitext(os.path.basename(ligand))[0]

            _ = ocmolproc.split_poses(ligand, ligandName, outPath, logFile = "", suffix = "_split_")

            # Add the ligand name to the list
            ligandNames.append(ligandName)

    # If splitLigand or overwrite is True means that it is needed to get the splited ligands again
    if splitLigand or overwrite:
        # Reset the ligand list
        ligands = []
        # Append the splited ligands to the ligands list (using the glob function)
        ligands.extend(glob(f"{outPath}/*_split_*.pdbqt"))
    
    # For each ligand in the ligands list (newly splited ligands)
    for ligand in ligands:
        # Get the splited ligand name
        ligand_name = os.path.splitext(os.path.basename(ligand))[0]

        # Create the command list
        cmd = [smina, "--scoring", scoring_function, "--score_only", "--config", confFile, "--ligand", ligand, "--log", f"{outPath}/{ligand_name}_{scoring_function}_rescoring.log", "--cpu", "1"]

        # Run the command
        _ = ocrun.run(cmd, logFile = logFile)

        if not logFile:
            # Set it as cmd log
            logFile = f"{outPath}/{ligand_name}_{scoring_function}_rescoring.log"

        # Check if the logFile exists and it has the string "Affinity:" inside it
        if not os.path.isfile(logFile) or not "Affinity:" in open(logFile).read():
            # Print an error
            ocprint.print_error(f"Problems while running smina for the ligand '{ligand_name}' using the scoring function '{scoring_function}'.")
            # Remove the file
            _ = ocff.safe_remove_file(logFile)
    
    # Think about how can this be done to deal with multiple runs
    return None

def get_pose_index_from_file_path(filePath: str) -> int:
    '''Get the pose index from the file path.

    Parameters
    ----------
    filePath : str
        The path to the file.

    Returns
    -------
    int
        The pose index.
    '''

    # Get the filename from the file path
    filename = os.path.splitext(os.path.basename(filePath))[0]

    # Split the filename using the '_split_' string as delimiter then grab the end of the string
    filename = filename.split("_split_")[-1]

    # Return the filename
    return int(filename)

def get_rescore_log_paths(outPath: str) -> List[str]:
    ''' Get the paths for the rescore log files.

    Parameters
    ----------
    outPath : str
        Path to the output folder where the rescoring logs are located.

    Returns
    -------
    List[str]
        A list with the paths for the rescoring log files.
    '''

    return [f for f in glob(f"{outPath}/*.log") if os.path.isfile(f)]

def read_rescore_logs(rescoreLogPaths: Union[List[str], str], onlyBest: bool = False) -> Dict[str, List[Union[str, float]]]:
    ''' Reads the data from the rescore log files.

    Parameters
    ----------
    rescoreLogPaths : List[str] | str
        A list with the paths for the rescoring log files.
    onlyBest : bool, optional
        If True, only the best pose will be returned. By default False.

    Returns
    -------
    Dict[str, List[Union[str, float]]]
        A dictionary with the data from the rescore log files.
    '''

    # Create the dictionary
    rescoreLogData = {}

    # If the rescoreLogPaths is not a list
    if not isinstance(rescoreLogPaths, list):
        # Make it a list
        rescoreLogPaths = [rescoreLogPaths]

    # For each rescore log path
    for rescoreLogPath in rescoreLogPaths:
        # Get the filename from the log path
        filename = os.path.splitext(os.path.basename(rescoreLogPath))[0]
        # Split the filename using the split string as delimiter then grab the end of the string
        filename = filename.split("_split_")[-1]
        # Remove the extension from the filename
        filename = os.path.splitext(filename)[0]
        # If onlyBest is True and the filename does not start with "1"
        if onlyBest and not filename.startswith("1"):
            # Skip this iteration
            continue
        # Reverse the filename with the delimiter as the underscore
        filename = "_".join(reversed(filename.split("_")))
        # Get the rescore log data
        rescoreLogData[filename] = read_rescoring_log(rescoreLogPath)
    
    # Return the dictionary
    return rescoreLogData

# Aliases
###############################################################################
run_docking = run_smina
