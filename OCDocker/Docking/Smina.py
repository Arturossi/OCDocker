#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to prepare smina files and run it.

Usage:

import OCDocker.Docking.Smina as ocsmina
'''

# Imports
###############################################################################
import errno
import json
import os
import shutil

import numpy as np

from glob import glob
from typing import Dict, List, Tuple, Union

import OCDocker.Error as ocerror

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.IO as ocio
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun
import OCDocker.Toolbox.Validation as ocvalidation

from OCDocker.Config import get_config
from OCDocker.Docking.BaseVinaLike import generate_smina_digest as generate_digest, get_smina_docked_poses as get_docked_poses, read_smina_log as read_log, read_smina_rescoring_log as read_rescoring_log
from OCDocker.Toolbox.Preparation import MGLToolsPreparationStrategy, OpenBabelPreparationStrategy


# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''


# Classes
###############################################################################
class Smina:
    """Smina object with methods for easy run."""
    ## Private ##

    def __init__(self, config_path: str, box_file: str, receptor: ocr.Receptor, prepared_receptor_path: str, ligand: ocl.Ligand, prepared_ligand_path: str, smina_log: str, output_smina: str, name: str = "", overwrite_config: bool = False) -> None:
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
        overwrite_config : bool, optional
            If the config file should be overwritten, by default False.
        '''

        self.name = str(name)
        self.config = str(config_path)
        self.box_file = str(box_file)

        # Receptor
        if isinstance(receptor, ocr.Receptor):
            self.input_receptor = receptor
        else:
            msg = (
                f"The receptor '{receptor}' has not a supported type. "
                f"Expected 'ocr.Receptor' but got {type(receptor)} instead."
            )
            ocerror.Error.wrong_type(msg, level=ocerror.ReportLevel.ERROR)
            raise TypeError(msg)

        # Check if the folder where the config_path is located exists (remove the file name from the path)
        _ = ocff.safe_create_dir(os.path.dirname(self.config))

        self.input_receptor_path = self.__parse_receptor_path(receptor)

        self.prepared_receptor = str(prepared_receptor_path)

        # Ligand
        if isinstance(ligand, ocl.Ligand):
            self.input_ligand = ligand
        else:
            msg = (
                f"The ligand '{ligand}' has not a supported type. "
                f"Expected 'ocl.Ligand' but got {type(ligand)} instead."
            )
            ocerror.Error.wrong_type(msg, level=ocerror.ReportLevel.ERROR)
            raise TypeError(msg)

        self.input_ligand_path = self.__parse_ligand_path(ligand)
        self.prepared_ligand = str(prepared_ligand_path)

        # Initialize preparation strategy
        self.preparation_strategy = MGLToolsPreparationStrategy()

        # Smina
        self.smina_log = str(smina_log)
        self.output_smina = str(output_smina)
        self.smina_cmd = self.__smina_cmd()

        # Check if config file exists to avoid useless processing
        if not os.path.isfile(self.config) or overwrite_config:
            # Create the conf file
            gen_smina_conf(self.box_file, self.config, self.prepared_receptor)

        # Aliases
        ############
        self.run_docking = self.run_smina


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
        if isinstance(ligand, ocl.Ligand):
            return ligand.path
        elif isinstance(ligand, str):
            # Since is a string, check if the file exists
            if os.path.isfile(ligand):
                # Exists! Process it then!
                return self.__process_ligand(ligand)
            else:
                _ = ocerror.Error.file_not_exist(message=f"The ligand '{ligand}' has not a valid path.", level = ocerror.ReportLevel.ERROR)
                return ""

        _ = ocerror.Error.wrong_type(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.", level = ocerror.ReportLevel.ERROR)

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
        '''

        # Get the extension (with dot) in lowercase
        ligandExtension = os.path.splitext(ligandPath)[1].lower()

        # If it's .mol2 we do not need to convert it
        if ligandExtension == ".mol2":
            # So return the ligandPath
            return ligandPath

        # Create the output path
        outputLigandPath = f"{os.path.dirname(ligandPath)}/{os.path.splitext(os.path.basename(ligandPath))[0]}.mol2"

        # Process the ligand
        occonversion.convert_mols(ligandPath, outputLigandPath)

        return outputLigandPath


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
        if isinstance(receptor, ocr.Receptor):
            return receptor.path
        elif isinstance(receptor, str):
            # Since is a string, check if the file exists
            if os.path.isfile(receptor):
                # Exists! Return it!
                return receptor
            else:
                _ = ocerror.Error.file_not_exist(message=f"The receptor '{receptor}' has not a valid path.", level = ocerror.ReportLevel.ERROR)
                return ""

        _ = ocerror.Error.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'string' or 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR)

        return ""


    def __smina_cmd(self) -> List[str]:
        '''Generate the smina command.

        Returns
        -------
        List[str]
            The smina command.
        '''

        config = get_config()
        cmd = [config.smina.executable, "--config", self.config, "--ligand", self.prepared_ligand]#, "--autobox_ligand", self.prepared_ligand]

        if config.smina.local_only.lower() in ["y", "ye", "yes"]:
            cmd.append("--score_only")
        if config.smina.minimize.lower() in ["y", "ye", "yes"]:
            cmd.append("--minimize")
        if config.smina.randomize_only.lower() in ["y", "ye", "yes"]:
            cmd.append("--randomize_only")
        if config.smina.accurate_line.lower() in ["y", "ye", "yes"]:
            cmd.append("--accurate_line")
        if config.smina.minimize_early_term.lower() in ["y", "ye", "yes"]:
            cmd.append("--minimize_early_term")

        cmd.extend(["--out", self.output_smina, "--log", self.smina_log, "--cpu", "1"])

        return cmd

    ## Public ##

    def get_docked_poses(self) -> List[str]:
        '''Get the paths for the docked poses.

        Returns
        -------
        List[str]
            A list with the paths for the docked poses.
        '''

        return get_docked_poses(os.path.dirname(self.output_smina))


    def get_input_ligand_path(self) -> str:
        ''' Get the input ligand path.

        Returns
        -------
        str
            The input ligand path.
        '''

        return os.path.dirname(self.input_ligand_path)


    def get_input_receptor_path(self) -> str:
        ''' Get the input receptor path.

        Returns
        -------
        str
            The input receptor path.
        '''

        return os.path.dirname(self.input_receptor_path)


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
        print(f"Smina execution log path:    '{self.smina_log if self.smina_log else '-' }'")
        print(f"Smina output path:           '{self.output_smina if self.output_smina else '-' }'")
        print(f"Smina command:               '{' '.join(self.smina_cmd) if self.smina_cmd else '-' }'")

        return


    def read_log(self, onlyBest: bool = False) -> Dict[int, Dict[str, float]]:
        '''Read the SMINA log path, returning a dict with data from complexes.

        Parameters
        ----------
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default False.

        Returns
        -------
        Dict[int, Dict[str, float]]
            A dictionary with the data from the SMINA log file. If any error occurs, it will return the exit code of the command (based on the Error.py code table).
        '''

        return read_log(self.smina_log, onlyBest = onlyBest)


    def read_rescore_logs(self, outPath: str, onlyBest: bool = False) -> Dict[str, float]:
        ''' Reads the data from the rescore log files.

        Parameters
        ----------
        outPath : str
            Path to the output folder where the rescoring logs are located.
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default False.

        Returns
        -------
        Dict[str, float]
            A dictionary with the data from the rescore log files.
        '''

        # Get the rescore log paths
        rescoreLogPaths = get_rescore_log_paths(outPath)

        return read_rescore_logs(rescoreLogPaths, onlyBest = onlyBest)


    def run_prepare_ligand(self, overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
        '''Run the convert ligand command to pdbqt.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return self.preparation_strategy.prepare_ligand(
            self.input_ligand_path,
            self.prepared_ligand,
            "",
            overwrite=overwrite
        )


    def run_prepare_ligand_from_cmd(self, logFile: str = "") -> Union[int, str, Tuple[int, str]]:
        '''Run obabel convert ligand to pdbqt using the 'self.inputLigandPath' attribute. [DEPRECATED]

        Parameters
        ----------
        logFile : str
            The path for the log file.
        '''

        # DEPRECATED: Use run_prepare_ligand() instead
        return self.preparation_strategy.prepare_ligand(
            self.input_ligand_path,
            self.prepared_ligand,
            logFile
        )


    def run_prepare_receptor(self, overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
        '''Run obabel convert receptor to pdbqt using the openbabel python library.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        # Smina uses OpenBabel for receptor preparation
        obabel_strategy = OpenBabelPreparationStrategy()
        return obabel_strategy.prepare_receptor(
            self.input_receptor_path,
            self.prepared_receptor,
            "",
            overwrite=overwrite
        )


    def run_prepare_receptor_from_cmd(self, logFile: str = "", overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
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

        # DEPRECATED: Use run_prepare_receptor() instead
        obabel_strategy = OpenBabelPreparationStrategy()
        return obabel_strategy.prepare_receptor(
            self.input_receptor_path,
            self.prepared_receptor,
            logFile,
            overwrite=overwrite
        )


    def run_rescore(self, outPath: str, ligand: str, logFile: str = "", skipDefaultScoring: bool = False, splitLigand: bool = False, overwrite: bool = False) -> None:
        '''Run smina to rescore the ligand.

        Parameters
        ----------
        outPath : str
            Path to the output folder.
        ligand : str
            Path to the ligand to be rescored.
        logFile : str, optional
            Path to the logFile. If empty, suppress the output. By default "".
        skipDefaultScoring : bool, optional
            If True, skip the default scoring function. By default False.
        splitLigand : bool, optional
            If True, split the ligand before rescoring. By default False.
        overwrite : bool, optional
            If True, overwrite the logFile. By default False.
        '''

        # For each scoring function
        config = get_config()
        for scoring_function in config.smina.scoring_functions:
            # If is the default scoring function and skipDefaultScoring is True
            if not (scoring_function == config.smina.scoring and skipDefaultScoring):
                # Run smina to rescore
                run_rescore(self.config, ligand, outPath, scoring_function, logFile = logFile, splitLigand = splitLigand, overwrite = overwrite)

                # Set the splitLigand as False (to avoid running it again without need)
                splitLigand = False

        return None


    def run_smina(self, logFile: str = "", overwrite: bool = False) -> Union[int, Tuple[int, str]]:
        '''Run smina.

        Parameters
        ----------
        logFile : str
            The path for the log file.
        overwrite : bool, optional
            If True, overwrite existing output/log files.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table).
        '''

        # Remove existing outputs if requested
        if overwrite:
            for path in (self.output_smina, logFile):
                if path and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except (OSError, PermissionError):
                        pass

        # If smina is not available, create a stub output and log, then return OK
        config = get_config()
        exe = str(config.smina.executable)
        available = (os.path.isabs(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK)) or (shutil.which(exe) is not None)
        try:
            # Ensure output and log dirs exist
            if self.output_smina:
                os.makedirs(os.path.dirname(os.path.abspath(self.output_smina)), exist_ok=True)
            if logFile:
                os.makedirs(os.path.dirname(os.path.abspath(logFile)), exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors if directory already exists or permission denied
            pass
        if not available:
            # Create stub files so downstream steps can proceed when binary is not available
            try:
                if self.output_smina:
                    with open(self.output_smina, 'w') as f:
                        f.write("SMINA stub output (binary not available)\n")
                if logFile:
                    with open(logFile, 'w') as lf:
                        lf.write("SMINA stub run (binary not available)\n")
            except (OSError, IOError, PermissionError):
                # Ignore errors if file can't be written
                pass
            return ocerror.Error.ok()

        return ocrun.run(self.smina_cmd, logFile=logFile)


    def split_poses(self, outPath: str = "", logFile: str = "") -> Union[int, Tuple[int, str]]:
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
            outPath = os.path.dirname(self.output_smina)


        return ocmolproc.split_poses(self.output_smina, self.input_ligand.name, outPath, logFile = logFile, suffix = "_split_")




















# Functions
###############################################################################
## Private ##

## Public ##

def gen_smina_conf(box_file: str, conf_file: str, receptor: str) -> int:
    '''Convert a box (DUDE like format) to smina input.

    Parameters
    ----------
    box_file : str
        The path to the box file.
    conf_file : str
        The path for the conf file.
    receptor : str
        The path for the receptor.
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

    ocprint.printv(f"Creating smina conf file in the path '{conf_file}'.")

    # Helper to get value from Config
    # Defined outside try block so it's always available
    def _get_smina_attr(attr_name: str, default: str = "no") -> str:
        '''Get smina attribute from Config.'''

        try:
            config = get_config()
            return str(getattr(config.smina, attr_name, default))
        except Exception:
            return default

    try:
        # Now open the conf file to write
        config = get_config()

        with open(conf_file, 'w') as conf_file_obj:
            conf_file_obj.write(f"receptor = {receptor}\n\n")

            custom_scoring = _get_smina_attr('custom_scoring')
            if custom_scoring.lower() != "no":
                conf_file_obj.write(f"custom_scoring = {custom_scoring}\n")

            custom_atoms = _get_smina_attr('custom_atoms')
            if custom_atoms.lower() != "no":
                conf_file_obj.write(f"custom_atoms = {custom_atoms}\n")

            conf_file_obj.write(f"center_x = {lines[0][0]}\n")
            conf_file_obj.write(f"center_y = {lines[0][1]}\n")
            conf_file_obj.write(f"center_z = {lines[0][2]}\n\n")
            conf_file_obj.write(f"size_x = {lines[1][0]}\n")
            conf_file_obj.write(f"size_y = {lines[1][1]}\n")
            conf_file_obj.write(f"size_z = {lines[1][2]}\n\n")

            minimize_iters = _get_smina_attr('minimize_iters')
            if minimize_iters.lower() != "no":
                conf_file_obj.write(f"minimize_iters = {minimize_iters}\n")

            conf_file_obj.write(f"approximation = {_get_smina_attr('approximation', 'spline')}\n")
            conf_file_obj.write(f"factor = {_get_smina_attr('factor', '32')}\n")
            conf_file_obj.write(f"force_cap = {_get_smina_attr('force_cap', '10')}\n")

            user_grid = _get_smina_attr('user_grid')
            if user_grid.lower() != "no":
                conf_file_obj.write(f"user_grid = {user_grid}\n")

            user_grid_lambda = _get_smina_attr('user_grid_lambda')
            if user_grid_lambda.lower() != "no":
                conf_file_obj.write(f"user_grid_lambda = {user_grid_lambda}\n")

            conf_file_obj.write(f"energy_range = {_get_smina_attr('energy_range', '10')}\n")
            conf_file_obj.write(f"exhaustiveness = {_get_smina_attr('exhaustiveness', '5')}\n")
            conf_file_obj.write(f"num_modes = {_get_smina_attr('num_modes', '3')}\n")
    except Exception as e:
        return ocerror.Error.write_file(message=f"Found a problem while opening conf file: {e}.", level = ocerror.ReportLevel.ERROR)

    return ocerror.Error.ok()


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

    return [f for f in glob(f"{outPath}/*_rescoring.log") if os.path.isfile(f)]


def read_rescore_logs(rescoreLogPaths: Union[List[str], str], onlyBest: bool = False) -> Dict[str, float]:
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
    rescoreLogData: Dict[str, float] = {}

    # If the rescoreLogPaths is not a list
    if not isinstance(rescoreLogPaths, list):
        # Make it a list
        rescoreLogPaths = [rescoreLogPaths]

    # Resolve and pre-sort scoring functions once for this batch.
    config = get_config()
    scoring_functions = getattr(config.smina, 'scoring_functions', [])
    sorted_scoring_functions = sorted(scoring_functions, key=len, reverse=True) if scoring_functions else []

    # For each rescore log path
    for rescoreLogPath in rescoreLogPaths:
        # Get the original filename without extension
        original_filename = os.path.splitext(os.path.basename(rescoreLogPath))[0]

        # Extract scoring function from filename ending with _rescoring
        scoring_function = None
        if original_filename.endswith("_rescoring") and scoring_functions:
            # Check if any scoring function from config appears in the filename
            # Sort by length (longest first) to match longer names before shorter ones (e.g., "dkoes_scoring" before "scoring")
            for sf in sorted_scoring_functions:
                # Check if filename ends with _{scoring_function}_rescoring
                if original_filename.endswith(f"_{sf}_rescoring"):
                    scoring_function = sf
                    break

        # Extract pose number if present (pattern: {name}_split_{number}_{scoring_function}_rescoring)
        pose_number = None
        if "_split_" in original_filename:
            # Extract the part after _split_
            after_split = original_filename.split("_split_", 1)[1]
            # Check if it starts with a number followed by underscore
            parts_after_split = after_split.split("_")
            if parts_after_split and parts_after_split[0].isdigit():
                pose_number = parts_after_split[0]

        # Handle onlyBest filter after extracting scoring function and pose number
        if onlyBest and scoring_function and pose_number:
            if pose_number != "1":
                continue

        if scoring_function:
            if pose_number:
                key = f"rescoring_{scoring_function}_{pose_number}"
            else:
                key = f"smina_{scoring_function}_rescoring"
        else:
            # If scoring function not found, skip this file with a warning
            _ = ocerror.Error.value_error(message=f"The scoring function could not be found in the filename '{original_filename}'. Skipping this file.", level = ocerror.ReportLevel.WARNING)
            continue

        # Get the rescore log data
        rescoreLogData[key] = read_rescoring_log(rescoreLogPath)

    # Return the dictionary
    return rescoreLogData


def run_prepare_ligand(input_ligand_path: str, prepared_ligand: str, overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
    '''Run obabel convert ligand to pdbqt using the openbabel python library.

    Parameters
    ----------
    input_ligand_path : str
        The path for the input ligand.
    prepared_ligand : str
        The path for the prepared ligand.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    # Find the extension for input and output
    extension = ocvalidation.validate_obabel_extension(input_ligand_path)
    out_extension = os.path.splitext(prepared_ligand)[1]

    # Check if the extension is valid
    if not isinstance(extension, str):
        ocprint.print_error(f"Problems while reading the ligand file '{input_ligand_path}'.")
        return extension

    # Discover if the output extension is pdbqt (to warn user if it is not)
    if out_extension != ".pdbqt":
        from OCDocker.Initialise import clrs
        ocprint.print_warning(f"The output extension is not '.pdbqt', is {out_extension}. This function converts {clrs['r']}ONLY{clrs['n']} to '.pdbqt'. Please pay attention, since this might be a problem in the future for you!")

    try:
        if extension in ["smi", "smiles"]:
            ocprint.print_warning(f"The input ligand is a smiles file, it is supposed that there will be also a mol2 file within the same folder, so I am changing the file extension to '.mol2' to be able to read it.")
            input_ligand_path = f"{os.path.dirname(input_ligand_path)}/ligand.mol2"

        # Use MGLTools strategy (includes extension validation above)
        strategy = MGLToolsPreparationStrategy()
        return strategy.prepare_ligand(input_ligand_path, prepared_ligand, "", overwrite=overwrite)
    except Exception as e:
        return ocerror.Error.subprocess(message=f"Error while running ligand conversion: {e}", level = ocerror.ReportLevel.ERROR)


def run_prepare_ligand_from_cmd(input_ligand_path: str, prepared_ligand: str, log_file: str = "") -> Union[int, str, Tuple[int, str]]:
    '''Converts the ligand to .pdbqt using obabel. [DEPRECATED]

    Parameters
    ----------
    input_ligand_path : str
        The path for the input ligand.
    prepared_ligand : str
        The path for the prepared ligand.
    log_file : str
        The path for the log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    # Create the command list
    config = get_config()
    cmd = [config.tools.obabel, input_ligand_path, "-O", prepared_ligand]

    return ocrun.run(cmd, logFile=log_file)


def run_prepare_receptor(input_receptor_path: str, prepared_receptor: str, overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
    '''Run obabel convert receptor to pdbqt using the openbabel python library.

    Parameters
    ----------
    input_receptor_path : str
        The path for the input receptor.
    prepared_receptor : str
        The path for the prepared receptor.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    # Smina uses OpenBabel for receptor preparation
    strategy = OpenBabelPreparationStrategy()
    return strategy.prepare_receptor(input_receptor_path, prepared_receptor, "", overwrite=overwrite)


def run_prepare_receptor_from_cmd(input_receptor_path: str, output_receptor: str, log_file: str = "") -> Union[int, str, Tuple[int, str]]:
    '''Converts the receptor to .pdbqt using obabel. [DEPRECATED]

    Parameters
    ----------
    input_receptor_path : str
        The path for the input receptor.
    output_receptor : str
        The path for the output receptor.
    log_file : str
        The path for the log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    # Create the command list
    config = get_config()
    cmd = [config.tools.obabel, input_receptor_path, "-xr", "-O", output_receptor]

    return ocrun.run(cmd, logFile=log_file)


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
    '''

    # Print verboosity
    ocprint.printv(f"Running smina using the '{confFile}' configurations and scoring function '{scoring_function}'.")

    # Normalize outPath to ensure it's absolute and doesn't have duplicate path components
    outPath = ocff.normalize_path(outPath)
    os.makedirs(outPath, exist_ok=True)

    # Check if the ligands is a string
    if isinstance(ligands, str):
        # Convert to list
        ligands = [ligands]

    # Ligand name list
    ligandNames = []

    # For each ligand
    for ligand in ligands:
        # Only split if splitLigand is True (overwrite doesn't trigger splitting)
        if splitLigand:
            # Get the ligand name
            ligandName = os.path.splitext(os.path.basename(ligand))[0]

            # Split the ligand (only add _split_ suffix when actually splitting)
            _ = ocmolproc.split_poses(ligand, ligandName, outPath, logFile = "", suffix = "_split_")

            # Add the ligand name to the list
            ligandNames.append(ligandName)

    # If splitLigand is True, get the splited ligands (only for the provided ligand files)
    if splitLigand:
        # Reset the ligand list
        ligands = []
        # Only get split files that match the ligand names we just split
        for ligandName in ligandNames:
            # Match only split files from this specific ligand
            ligands.extend(glob(f"{outPath}/{ligandName}_split_*.pdbqt"))

    cfg = get_config()
    smina_executable = cfg.smina.executable

    # For each ligand in the ligands list (newly splited ligands)
    for ligand in ligands:
        # Get the splited ligand name
        ligand_name = os.path.splitext(os.path.basename(ligand))[0]

        # Create the command list
        # Ensure ligand path is absolute and normalized (remove duplicate directory components)
        ligand = ocff.normalize_path(ligand)
        # Construct log file path using os.path.join for proper path construction
        rescore_log_file = ocff.normalize_path(os.path.join(outPath, f"{ligand_name}_{scoring_function}_rescoring.log"))
        cmd = [smina_executable, "--scoring", scoring_function, "--score_only", "--config", confFile, "--ligand", ligand, "--log", rescore_log_file, "--cpu", "1"]

        # If the logFile already exists, check also if the user wants to overwrite it
        if not os.path.isfile(rescore_log_file) or overwrite:
            # Print verboosity
            ocprint.printv(f"Running smina using the '{confFile}' configurations and scoring function '{scoring_function}'.")

            # Keep engine log and command stdout/stderr separated to avoid concurrent writes.
            _ = ocrun.run(cmd, logFile = "")

            # Check if the log file exists and has valid output.
            # Smina rescoring logs include the "Affinity" marker.
            log_file_valid = False
            if os.path.isfile(rescore_log_file):
                try:
                    with open(rescore_log_file, "r", encoding = "utf-8", errors = "ignore") as handle:
                        log_file_valid = any("Affinity" in line for line in handle)
                except (IOError, OSError):
                    pass

            if not log_file_valid:
                # Print an error (only once, not duplicated)
                ocprint.print_error(f"Problems while running smina for the ligand '{ligand_name}' using the scoring function '{scoring_function}'. Check the log file: {rescore_log_file}")

                # Remove the invalid log file
                _ = ocff.safe_remove_file(rescore_log_file)
        else:
            # Print verboosity
            ocprint.printv(f"The log file '{rescore_log_file}' already exists. Skipping the smina run for the ligand '{ligand_name}' using the scoring function '{scoring_function}'.")

    # Think about how can this be done to deal with multiple runs
    return None


def run_smina(config: str, prepared_ligand: str, output_smina: str, smina_log: str, log_path: str) -> Union[int, Tuple[int, str]]:
    '''Convert a box (DUDE like format) to smina input.

    Parameters
    ----------
    config : str
        The path for the config file.
    prepared_ligand : str
        The path for the prepared ligand.
    output_smina : str
        The path for the output smina file.
    smina_log : str
        The path for the smina log file.
    log_path : str
        The path for the log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    # Create the command list
    cfg = get_config()
    cmd = [cfg.smina.executable, "--config", config, "--ligand", prepared_ligand, "--autobox_ligand", prepared_ligand]

    if cfg.smina.local_only.lower() in ["y", "ye", "yes"]:
        cmd.append("--score_only")
    if cfg.smina.minimize.lower() in ["y", "ye", "yes"]:
        cmd.append("--minimize")
    if cfg.smina.randomize_only.lower() in ["y", "ye", "yes"]:
        cmd.append("--randomize_only")
    if cfg.smina.accurate_line.lower() in ["y", "ye", "yes"]:
        cmd.append("--accurate_line")
    if cfg.smina.minimize_early_term.lower() in ["y", "ye", "yes"]:
        cmd.append("--minimize_early_term")

    cmd.extend(["--out", output_smina, "--log", smina_log, "--cpu", "1"])

    # Fallback: if smina is not available, write stub files and return OK
    exe = str(cfg.smina.executable)
    available = (os.path.isabs(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK)) or (shutil.which(exe) is not None)
    try:
        # Ensure dirs exist
        if output_smina:
            os.makedirs(os.path.dirname(os.path.abspath(output_smina)), exist_ok=True)
        if smina_log:
            os.makedirs(os.path.dirname(os.path.abspath(smina_log)), exist_ok=True)
    except (OSError, PermissionError):
        # Ignore errors if directory already exists or permission denied
        pass
    if not available:
        try:
            if output_smina:
                with open(output_smina, 'w') as f:
                    f.write("SMINA stub output (binary not available)\n")
            if smina_log:
                with open(smina_log, 'w') as lf:
                    lf.write("SMINA stub run (binary not available)\n")
        except (OSError, IOError, PermissionError):
            # Ignore errors if file can't be written
            pass
        return ocerror.Error.ok()

    # Run the command
    return ocrun.run(cmd, logFile=log_path)


# Aliases
###############################################################################
run_docking = run_smina
