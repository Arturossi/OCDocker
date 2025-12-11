#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to prepare gnina files and run it.

They are imported as:

import OCDocker.Docking.Gnina as ocgnina
'''

# Imports
###############################################################################
import errno
import json
import os

import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Union

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.IO as ocio
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun
import OCDocker.Toolbox.Validation as ocvalidation
from OCDocker.Toolbox.Preparation import OpenBabelPreparationStrategy

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################
class Gnina:
    """Gnina object with methods for easy run."""
    def __init__(self, config_path: str, box_file: str, receptor: ocr.Receptor, prepared_receptor_path: str, ligand: ocl.Ligand, prepared_ligand_path: str, gnina_log: str, output_gnina: str, name: str = "", overwrite_config: bool = False) -> None:
        '''Constructor of the class Gnina.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        box_file : str
            The path for the box file.
        receptor : ocr.Receptor
            The receptor object.
        prepared_receptor_path : str 
            Path to the prepared receptor.
        ligand : ocl.Ligand
            The ligand object.
        prepared_ligand_path : str
            Path to the prepared ligand.
        gnina_log : str
            Path to the gnina log file.
        output_gnina : str
            Path to the output gnina file.
        name : str, optional
            Name of the gnina object, by default "".
        overwrite_config : bool, optional
            If the config file should be overwritten, by default False.
        '''

        self.name = str(name)
        self.config = str(config_path)
        self.box_file = str(box_file)
        
        # Receptor
        if type(receptor) == ocr.Receptor:
            self.input_receptor = receptor
        else:
            ocerror.Error.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR)
            return None

        # Check if the folder where the config_path is located exists (remove the file name from the path)
        _ = ocff.safe_create_dir(os.path.dirname(self.config))

        self.input_receptor_path = self.__parse_receptor_path(receptor)
        self.prepared_receptor = str(prepared_receptor_path)

        # Ligand
        if type(ligand) == ocl.Ligand:
            self.input_ligand = ligand
        else:
            ocerror.Error.wrong_type(f"The ligand '{ligand}' has not a supported type. Expected 'ocl.Ligand' but got {type(ligand)} instead.", level = ocerror.ReportLevel.ERROR)
            return None

        self.input_ligand_path = self.__parse_ligand_path(ligand)
        self.prepared_ligand = str(prepared_ligand_path)
        
        # Initialize preparation strategy
        self.preparation_strategy = OpenBabelPreparationStrategy()

        # Gnina
        self.gnina_log = str(gnina_log)
        self.output_gnina = str(output_gnina)
        self.gnina_cmd = self.__gnina_cmd()
        
        # Check if config file exists to avoid useless processing
        if not os.path.isfile(self.config) or overwrite_config:
            # Create the conf file

            gen_gnina_conf(self.box_file, self.config, self.prepared_receptor)

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
                _ = ocerror.Error.file_not_exist(message=f"The receptor '{receptor}' has not a valid path.", level = ocerror.ReportLevel.ERROR)
                return ""

        _ = ocerror.Error.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'string' or 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR)

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
                _ = ocerror.Error.file_not_exist(message=f"The ligand '{ligand}' has not a valid path.", level = ocerror.ReportLevel.ERROR)
                return ""

        _ = ocerror.Error.wrong_type(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.", level = ocerror.ReportLevel.ERROR)

        return ""

    def __gnina_cmd(self) -> List[str]:
        '''Generate the gnina command.

        Returns
        -------
        List[str]
            The gnina command.
        '''

        config = get_config()
        cmd = [config.gnina.executable, "--config", self.config, "--ligand", self.prepared_ligand]

        if config.gnina.local_only.lower() in ["y", "ye", "yes"]:
            cmd.append("--score_only")
        if config.gnina.minimize.lower() in ["y", "ye", "yes"]:
            cmd.append("--minimize")
        if config.gnina.randomize_only.lower() in ["y", "ye", "yes"]:
            cmd.append("--randomize_only")
        if config.gnina.accurate_line.lower() in ["y", "ye", "yes"]:
            cmd.append("--accurate_line")
        if config.gnina.minimize_early_term.lower() in ["y", "ye", "yes"]:
            cmd.append("--minimize_early_term")
        
        # Check if the no_gpu flag is set
        if config.gnina.no_gpu.lower() in ["y", "ye", "yes"]:
            # Set the no gpu flag
            cmd.append("--no_gpu")
        else:
            # Check if CUDA_VISIBLE_DEVICES is set
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                # Set the GPU variable
                CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
                # Check if it is a list
                if "," in CUDA_VISIBLE_DEVICES: # type: ignore
                    # It is a list, get the first element
                    CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")[0] # type: ignore
                # Set the GPU
                cmd.extend(["--device", CUDA_VISIBLE_DEVICES]) # type: ignore

        cmd.extend(["--out", self.output_gnina, "--log", self.gnina_log, "--cpu", "1"])

        return cmd

    ## Public ##
    def read_log(self) -> Union[pd.DataFrame, int]:
        '''Read the gnina log path, returning a pd.dataframe with data from complexes.

        Returns
        -------
        pd.DataFrame | int
            The dataframe with the data from the gnina log, or the error code.
        '''

        return read_log(self.gnina_log) # type: ignore

    def run_gnina(self, logFile: str = "") -> Union[int, Tuple[int, str]]:
        '''Run gnina.

        Parameters
        ----------
        logFile : str
            The path for the log file.
        
        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table).   
        '''

        return ocrun.run(self.gnina_cmd, logFile=logFile)

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

        # DEPRECATED: Use run_prepare_ligand() instead
        return self.preparation_strategy.prepare_ligand(
            self.input_ligand_path,
            self.prepared_ligand,
            logFile
        )

    def run_prepare_ligand(self) -> Union[int, Tuple[int, str]]:
        '''Run the convert ligand command to pdbqt.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return run_prepare_ligand(self.input_ligand_path, self.prepared_ligand)

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

        # DEPRECATED: This method uses MGLTools which is not the standard for Gnina
        # Keeping for backward compatibility but should use OpenBabel strategy instead
        config = get_config()
        cmd = [config.tools.pythonsh, config.tools.prepare_receptor, "-r", self.input_receptor_path, "-o", self.prepared_receptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]
        return ocrun.run(cmd, logFile=logFile)

    def run_prepare_receptor(self) -> Union[int, Tuple[int, str]]:
        '''Run obabel convert receptor to pdbqt using the openbabel python library.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return self.preparation_strategy.prepare_receptor(
            self.input_receptor_path,
            self.prepared_receptor,
            ""
        )

    def print_attributes(self) -> None:
        '''Print the class attributes.'''

        print(f"Name:                        '{self.name if self.name else '-' }'")
        print(f"Config path:                 '{self.config if self.config else '-' }'")
        print(f"Input receptor:              '{self.input_receptor if self.input_receptor else '-' }'")
        print(f"Input receptor path:         '{self.input_receptor_path if self.input_receptor_path else '-' }'")
        print(f"Prepared receptor path:      '{self.prepared_receptor if self.prepared_receptor else '-' }'")
        prep_receptor_cmd = self.preparation_strategy.get_receptor_command(self.input_receptor_path, self.prepared_receptor)
        print(f"Prepared receptor command:   '{' '.join(prep_receptor_cmd) if prep_receptor_cmd else '-' }'")
        print(f"Input ligand:                '{self.input_ligand if self.input_ligand else '-' }'")
        print(f"Input ligand path:           '{self.input_ligand_path if self.input_ligand_path else '-' }'")
        print(f"Prepared ligand path:        '{self.prepared_ligand if self.prepared_ligand else '-' }'")
        prep_ligand_cmd = self.preparation_strategy.get_ligand_command(self.input_ligand_path, self.prepared_ligand)
        print(f"Prepared ligand command:     '{' '.join(prep_ligand_cmd) if prep_ligand_cmd else '-' }'")
        print(f"Gnina execution log path:    '{self.gnina_log if self.gnina_log else '-' }'")
        print(f"Gnina output path:           '{self.output_gnina if self.output_gnina else '-' }'")
        print(f"Gnina command:               '{' '.join(self.gnina_cmd) if self.gnina_cmd else '-' }'")
        
        return


# Functions
###############################################################################
## Private ##

## Public ##
def gen_gnina_conf(box_file: str, conf_file: str, receptor: str) -> int:
    '''Convert a box (DUDE like format) to gnina input.

    Parameters
    ----------
    box_file : str
        The path to the box file.
    conf_file : str
        The path for the conf file.
    receptor : str
        The path for the receptor.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    '''

    # Test if the file box_file exists
    if not os.path.exists(box_file):
        return ocerror.Error.file_not_exist(message=f"The box file in the path {box_file} does not exist! Please ensure that the file exists and the path is correct.", level = ocerror.ReportLevel.ERROR)
    # List to hold all the data
    lines = []

    try:
        # Open the box file
        with open(str(box_file), 'r') as box_file_obj:
            # For each line in the file
            for line in box_file_obj:
                # If it starts with REMARK
                if line.startswith("REMARK"):
                    # Slice the line in right positions
                    lines.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

                    # If the length of the lines element is 2 or greater
                    if len(lines) >= 2:
                        # Break the loop (optimization)
                        break
    except Exception as e:
        return ocerror.Error.read_file(message=f"Found a problem while reading the box file: {e}", level = ocerror.ReportLevel.ERROR)

    ocprint.printv(f"Creating gnina conf file in the path '{conf_file}'.")
    try:
        # Now open the conf file to write
        config = get_config()
        with open(conf_file, 'w') as conf_file_obj:
            conf_file_obj.write(f"receptor = {receptor}\n\n")

            if config.gnina.custom_scoring.lower() != "no":
                conf_file_obj.write(f"custom_scoring = {config.gnina.custom_scoring}\n")

            if config.gnina.custom_atoms.lower() != "no":
                conf_file_obj.write(f"custom_atoms = {config.gnina.custom_atoms}\n")

            conf_file_obj.write(f"center_x = {lines[0][0]}\n")
            conf_file_obj.write(f"center_y = {lines[0][1]}\n")
            conf_file_obj.write(f"center_z = {lines[0][2]}\n\n")
            conf_file_obj.write(f"size_x = {lines[1][0]}\n")
            conf_file_obj.write(f"size_y = {lines[1][1]}\n")
            conf_file_obj.write(f"size_z = {lines[1][2]}\n\n")

            if config.gnina.user_grid.lower() != "no":
                conf_file_obj.write(f"user_grid = {config.gnina.user_grid}\n")

            if config.gnina.user_grid_lambda.lower() != "no":
                conf_file_obj.write(f"user_grid_lambda = {config.gnina.user_grid_lambda}\n")
            
            if config.gnina.num_mc_steps.lower() != "no":
                conf_file_obj.write(f"num_mc_steps = {config.gnina.num_mc_steps}\n")

            if config.gnina.max_mc_steps.lower() != "no":
                conf_file_obj.write(f"max_mc_steps = {config.gnina.max_mc_steps}\n")

            if config.gnina.num_mc_saved.lower() != "no":
                conf_file_obj.write(f"num_mc_saved = {config.gnina.num_mc_saved}\n")

            if config.gnina.approximation.lower() != "no":
                conf_file_obj.write(f"approximation = {config.gnina.approximation}\n")

            if config.gnina.minimize_iters.lower() != "no":
                conf_file_obj.write(f"minimize_iters = {config.gnina.minimize_iters}\n")

            if config.gnina.exhaustiveness:
                conf_file_obj.write(f"exhaustiveness = {config.gnina.exhaustiveness}\n")
            if config.gnina.num_modes:
                conf_file_obj.write(f"num_modes = {config.gnina.num_modes}\n")
            if config.gnina.factor:
                conf_file_obj.write(f"factor = {config.gnina.factor}\n")
            if config.gnina.force_cap:
                conf_file_obj.write(f"force_cap = {config.gnina.force_cap}\n")
            
    except Exception as e:
        return ocerror.Error.write_file(message=f"Found a problem while opening conf file: {e}.", level = ocerror.ReportLevel.ERROR)

    return ocerror.Error.ok()


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
    config = get_config()
    cmd = [config.tools.obabel, inputLigandPath, "-O", preparedLigand]

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
    strategy = OpenBabelPreparationStrategy()
    return strategy.prepare_ligand(inputLigandPath, preparedLigand, "")


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
    config = get_config()
    cmd = [config.tools.obabel, inputReceptorPath, "-xr", "-O", outputReceptor]
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

    strategy = OpenBabelPreparationStrategy()
    return strategy.prepare_receptor(inputReceptorPath, preparedReceptor, "")


def run_gnina(config: str, preparedLigand: str, outputGnina: str, gninaLog: str, logPath: str) -> Union[int, Tuple[int, str]]:
    '''Convert a box (DUDE like format) to vina input.

    Parameters
    ----------
    config : str
        The path for the config file.
    preparedLigand : str
        The path for the prepared ligand.
    outputGnina : str
        The path for the output gnina file.
    gninaLog : str
        The path for the gnina log file.
    logPath : str
        The path for the log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.

    '''

    # Create the command list
    cfg = get_config()
    cmd = [cfg.gnina.executable, "--config", config, "--ligand", preparedLigand, "--autobox_ligand", preparedLigand]

    if cfg.gnina.local_only.lower() in ["y", "ye", "yes"]:
        cmd.append("--score_only")
    if cfg.gnina.minimize.lower() in ["y", "ye", "yes"]:
        cmd.append("--minimize")
    if cfg.gnina.randomize_only.lower() in ["y", "ye", "yes"]:
        cmd.append("--randomize_only")
    if cfg.gnina.accurate_line.lower() in ["y", "ye", "yes"]:
        cmd.append("--accurate_line")
    if cfg.gnina.minimize_early_term.lower() in ["y", "ye", "yes"]:
        cmd.append("--minimize_early_term")

    cmd.extend(["--out", outputGnina, "--log", gninaLog, "--cpu", "1"])
    
    # Run the command
    return ocrun.run(cmd, logFile = logPath)


def read_log(path: str) -> Dict[str, List[Union[str, float]]]:
    '''Read the gnina log path, returning the data from complexes.

    Parameters
    ----------
    path : str
        The path to the gnina log file.

    Returns
    -------
    str, List[str | float]
        A dictionary with the data from the gnina log file.

    '''

    # Check if file exists
    if os.path.isfile(path):
        # Catch any error that might occur
        try:
            # Check if file is empty
            if os.stat(path).st_size == 0:
                # Print the error
                _ = ocerror.Error.empty_file(f"The gnina log file '{path}' is empty.", level = ocerror.ReportLevel.ERROR)
                # Return the dictionary with invalid default data
                return {"gnina_pose": [np.nan], "gnina_affinity": [np.nan]}

            # Create a dictionary to store the info
            data = {"gnina_pose": [], "gnina_affinity": []}

            # Initiate the last read line as empty
            lastReadLine = ""

            # Try except to avoid broken pipe ocerror.Error
            try:
                # Read the file reversely
                for line in ocio.lazyread_reverse_order_mmap(path):
                    # If a stop line is found, means that the last read line is the one that is wanted
                    if line.startswith("-----+"):
                        # Split the last line
                        lastLine = lastReadLine.split()
                        data["gnina_pose"].append(lastLine[0])
                        data["gnina_affinity"].append(lastLine[1])
                        break

                    # Assign the last read line as the current line
                    lastReadLine = line
            except IOError as e:
                if e.errno == errno.EPIPE:
                    ocprint.print_error(f"Problems while reading file '{path}'. Error: {e}")
                    config = get_config()
                    ocprint.print_error_log(f"Problems while reading file '{path}'. Error: {e}", f"{config.logdir}/gnina_read_log_ERROR.log")
            
            # Check if the len of the data["gnina_affinity"] is 0
            if len(data["gnina_pose"]) == 0:
                data["gnina_pose"].append(np.nan)
                data["gnina_affinity"].append(np.nan)

            # Return the df reversing the order and reseting the index
            return data

        except Exception as e:
            _ = ocerror.Error.read_docking_log_error(f"Problems while reading the gnina log file '{path}'. Error: {e}", level = ocerror.ReportLevel.ERROR)
            # Return the dictionary with invalid default data
            return {"gnina_pose": [np.nan], "gnina_affinity": [np.nan]}

    # Throw an error
    _ = ocerror.Error.file_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.")

    # Return a dict with a NaN value
    return {"gnina_pose": [np.nan], "gnina_affinity": [np.nan]}


def generate_digest(digestPath: str, logPath: str, overwrite: bool = False, digestFormat : str = "json") -> int:
    '''Generate the docking digest.
    
    Parameters
    ----------
    digestPath : str
        Where to store the digest file.
    logPath : str
        The log path.
    overwrite : bool, optional
        If True, overwrites the output files if they already exist. (default is False)
    digestFormat : str, optional
        The format of the digest file. The options are: [ json (default), hdf5 (not implemented) ]

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Check if the file does not exists or if the overwrite flag is true
    if not os.path.isdir(digestPath) or overwrite:
        # Check if the digest extension is supported
        if ocvalidation.validate_digest_extension(digestPath, digestFormat):
        
            # Create the digest variable
            digest = None

            # Check if the file exists
            if os.path.isfile(digestPath):
                # Read it
                if digestFormat == "json":
                    # Read the json file
                    try:
                        # Open the json file in read mode
                        with open(digestPath, 'r') as f:
                            # Load the data
                            digest = json.load(f)
                            # Check if the digest variable is fine
                            if not isinstance(digest, dict):
                                return ocerror.Error.wrong_type(f"The digest file '{digestPath}' is not valid.", level = ocerror.ReportLevel.ERROR)
                    except Exception as e:
                        return ocerror.Error.file_not_exist(f"Could not read the digest file '{digestPath}'.", level = ocerror.ReportLevel.ERROR)
            else:
                # Since it does not exists, create it
                digest = ocff.empty_docking_digest(digestPath, overwrite)

            # Read the docking object log to generate the docking digest
            dockingDigest = read_log(logPath)

            # Check if the digest variable is fine
            if not isinstance(digest, dict):
                return ocerror.Error.wrong_type(f"The docking digest file '{digestPath}' is not valid.", level = ocerror.ReportLevel.ERROR)
            
            # Merge the digest and the docking digest
            digest = { **digest, **dockingDigest } # type: ignore

            # Write the digest file
            if digestFormat == "json":
                # Write the json file
                try:
                    # Open the json file in write mode
                    with open(digestPath, 'w') as f:
                        # Dump the data
                        json.dump(digest, f)
                except Exception as e:
                    return ocerror.Error.write_file(f"Could not write the digest file '{digestPath}'.", level = ocerror.ReportLevel.ERROR)

            return ocerror.Error.ok()
        return ocerror.Error.unsupported_extension(f"The provided extension '{digestFormat}' is not supported.", level = ocerror.ReportLevel.ERROR)
    
    return ocerror.Error.file_exists(f"The file '{digestPath}' already exists. If you want to overwrite it yse the overwrite flag.", level = ocerror.ReportLevel.WARNING)

# Aliases
###############################################################################
run_docking = run_gnina
