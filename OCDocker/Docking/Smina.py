#!/usr/lib/python3

# Imports
###############################################################################
import errno
import os

import numpy as np
import pandas as pd

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
Sets of classes and functions that are used to prepare smina files and run it.

They are imported as:

import OCDocker.Docking.Smina as ocsmina
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
        #self.prepareReceptorCmd = [obabel, self.inputReceptorPath, "-xr", "-O", self.preparedReceptor]

        # Ligand
        if type(ligand) == ocl.Ligand:
            self.inputLigand = ligand
            # Create the sminaFiles folder
            _ = octools.safe_create_dir(os.path.join(os.path.dirname(ligand.path), "plantsFiles"))
        else:
            errors.wrong_type(f"The ligand '{ligand}' has not a supported type. Expected 'ocl.Ligand' but got {type(ligand)} instead.", level="error")
            return None

        self.inputLigandPath = self.__parse_ligand_path(ligand)
        self.preparedLigand = str(preparedLigandPath)
        self.prepareLigandCmd = [pythonsh, prepare_ligand, "-l", self.inputLigandPath, "-C", "-o", self.preparedLigand]
        #self.prepareLigandCmd = [obabel, self.inputLigandPath, "-O", self.preparedLigand]

        # Vina
        self.sminaLog = str(sminaLog)
        self.outputSmina = str(outputSmina)
        self.sminaCmd = self.__smina_cmd()
        
        # Check if config file exists to avoid useless processing
        if not os.path.isfile(self.config) or overwriteConfig:
            # Create the conf file
            gen_smina_conf(self.boxFile, self.config, self.preparedReceptor)

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

    def __smina_cmd(self) -> List[str]:
        '''Generate the smina command.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            The smina command.

        Raises
        ------
        None
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
    def read_smina_log(self) -> Union[pd.DataFrame, int]:
        '''Read the smina log path, returning a pd.dataframe with data from complexes.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame | int
            The dataframe with the data from the smina log, or the error code.

        Raises
        ------
        None
        '''

        return read_smina_log(self.sminaLog) # type: ignore

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

        Raises
        ------
        None
        '''

        return octools.run(self.sminaCmd, logFile=logFile)

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

        Raises
        ------
        None
        '''

        return octools.run(self.prepareLigandCmd, logFile=logFile)

    def run_prepare_ligand(self) -> Union[int, Tuple[int, str]]:
        '''Run the convert ligand command to pdbqt.

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

        Raises
        ------
        None
        '''

        return octools.run(self.prepareReceptorCmd, logFile=logFile)

    def run_prepare_receptor(self) -> Union[int, Tuple[int, str]]:
        '''Run obabel convert receptor to pdbqt using the openbabel python library.

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

        return run_prepare_receptor(self.inputReceptorPath, self.preparedReceptor)

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

    Raises
    ------
    None
    '''

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

    octools.printv(f"Creating smina conf file in the path '{confFile}'.")
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
        return errors.write_file(message=f"Found a problem while opening conf file: {e}.", level="error")

    return errors.ok()

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

    Raises
    ------
    None
    '''

    # Create the command list
    cmd = [obabel, inputLigandPath, "-O", preparedLigand]

    # Run the command
    return octools.run(cmd, logFile=logFile)

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

    Raises
    ------
    None
    '''

    # Find the extension for input and output
    extension = octools.validate_obabel_extension(inputLigandPath)
    outExtension = os.path.splitext(preparedLigand)[1]

    # Check if the extension is valid
    if type(extension) != str:
        octools.print_error(f"Problems while reading the ligand file '{inputLigandPath}'.")
        return extension # type: ignore

    # Discover if the output extension is pdbqt (to warn user if it is not)
    if outExtension != ".pdbqt":
        octools.print_warning(f"The output extension is not '.pdbqt', is {outExtension}. This function converts {clrs['r']}ONLY{clrs['n']} to '.pdbqt'. Please pay attention, since this might be a problem in the future for you!")

    try:
        if extension in ["smi", "smiles"]:
            octools.print_warning(f"The input ligand is a smiles file, it is supposed that there will be also a mol2 file within the same folder, so I am changing the file extension to '.mol2' to be able to read it.")
            # Change it to mol2 in the inputLigandPath
            # get the path
            inputLigandPath = f"{os.path.dirname(inputLigandPath)}/ligand.mol2"
        
        # Create the command list
        cmd = [pythonsh, prepare_ligand, "-l", inputLigandPath, "-C", "-o", preparedLigand]
        return octools.run(cmd, cwd = os.path.dirname(inputLigandPath))
    except Exception as e:
        return errors.subprocess(message=f"Error while running ligand conversion using obabel python lib. Error: {e}", level="error")

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

    Raises
    ------
    None
    '''

    # Create the command list
    cmd = [obabel, inputReceptorPath, "-xr", "-O", outputReceptor]
    # Run the command
    return octools.run(cmd, logFile=logFile)

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

    Raises
    ------
    None
    '''

    # Find the extension for input and output
    extension = octools.validate_obabel_extension(inputReceptorPath)
    outExtension = os.path.splitext(preparedReceptor)[1]

    # Check if the extension is valid
    if type(extension) != str:
        octools.print_error(f"Problems while reading the receptor file '{inputReceptorPath}'.")
        return extension # type: ignore

    # Discover if the output extension is pdbqt (to warn user if it is not)
    if outExtension != ".pdbqt":
        octools.print_warning(f"The output extension is not '.pdbqt', is {outExtension}. This function converts {clrs['r']}ONLY{clrs['n']} to '.pdbqt'. Please pay attention, since this might be a problem in the future for you!")

    return octools.convertMols(inputReceptorPath, preparedReceptor) # type: ignore

def run_smina(config: str, preparedLigand: str, outputSmina: str, sminaLog: str, logPath: str) -> Union[int, Tuple[int, str]]:
    '''Convert a box (DUDE like format) to vina input.

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

    Raises
    ------
    None
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
    return octools.run(cmd, logFile = logPath)

def read_smina_log(path: str) -> Dict[str, List[Union[str, float]]]:
    '''Read the smina log path, returning the data from complexes.

    Parameters
    ----------
    path : str
        The path to the smina log file.

    Returns
    -------
    Dict[str, List[str | float]]
        A dictionary with the data from the smina log file.

    Raises
    ------
    None
    '''

    # Check if file exists
    if os.path.isfile(path):
        # Catch any error that might occur
        try:
            # Check if file is empty
            if os.stat(path).st_size == 0:
                # Print the error
                _ = errors.empty_file(f"The smina log file '{path}' is empty.", "error")
                # Return the dictionary with invalid default data
                return {"smina_pose": [np.NaN], "smina_affinity": [np.NaN]}

            # Create a dictionary to store the info
            data = {"smina_pose": [], "smina_affinity": []}

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
                        data["smina_pose"].append(lastLine[0])
                        data["smina_affinity"].append(lastLine[1])
                        break

                    # Assign the last read line as the current line
                    lastReadLine = line
            except IOError as e:
                if e.errno == errno.EPIPE:
                    octools.print_error(f"Problems while reading file '{path}'. Error: {e}")
                    octools.print_error_log(f"Problems while reading file '{path}'. Error: {e}", f"{logdir}/smina_read_log_ERROR.log")
            
            # Check if the len of the data["smina_affinity"] is 0
            if len(data["smina_pose"]) == 0:
                data["smina_pose"].append(np.NaN)
                data["smina_affinity"].append(np.NaN)

            # Return the df reversing the order and reseting the index
            return data
        
        except Exception as e:
            _ = errors.read_docking_log_error(f"Problems while reading the vina log file '{path}'. Error: {e}", "error")
            return {"smina_pose": [np.NaN], "smina_affinity": [np.NaN]}

    # Throw an error
    _ = errors.file_do_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.")

    # Return a dict with a NaN value
    return {"smina_pose": [np.NaN], "smina_affinity": [np.NaN]}

