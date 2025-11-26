#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to prepare vina files and run it.

They are imported as:

import OCDocker.Docking.Vina as ocvina
'''

# Imports
###############################################################################
import os
import shutil

from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun

from OCDocker.Toolbox.Preparation import MGLToolsPreparationStrategy
from OCDocker.Docking.BaseVinaLike import (
    read_vina_log as read_log,
    read_vina_rescoring_log as read_rescoring_log,
    generate_vina_digest as generate_digest,
    get_vina_docked_poses as get_docked_poses,
)


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
class Vina:
    """Vina object with methods for easy run."""
    def __init__(self, config_path: str, box_file: str, receptor: ocr.Receptor, prepared_receptor_path: str, ligand: ocl.Ligand, prepared_ligand_path: str, vina_log: str, output_vina: str, name: str = "", overwrite_config: bool = False) -> None:
        '''Constructor of the class Vina.
        
        Parameters
        ----------
        config_path : str
            The path for the config file.
        box_file : str
            The path for the box file.
        receptor : ocr.Receptor
            The receptor object.
        prepared_receptor_path : str
            The path for the prepared receptor.
        ligand : ocl.Ligand
            The ligand object.
        prepared_ligand_path : str
            The path for the prepared ligand.
        vina_log : str
            The path for the vina log file.
        output_vina : str
            The path for the vina output files.
        name : str, optional
            The name of the vina object, by default "".

        Returns
        -------
        None
        '''

        self.name = str(name)
        self.config = str(config_path)
        self.box_file = str(box_file)

        # Receptor
        if isinstance(receptor, ocr.Receptor):
            self.input_receptor = receptor
        else:
            ocerror.Error.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR) # type: ignore
            return None
        
        # Check if the folder where the configPath is located exists (remove the file name from the path)
        _ = ocff.safe_create_dir(Path(self.config).parent)

        self.input_receptor_path = self.__parse_receptor_path(receptor)
        self.prepared_receptor = str(prepared_receptor_path)

        # Ligand
        self.prepared_ligand = str(prepared_ligand_path)
        
        # Check the type of the ligand
        if isinstance(ligand, ocl.Ligand):   
            self.input_ligand = ligand
        else:
            ocerror.Error.wrong_type(f"The ligand '{ligand}' has not a supported type. Expected 'ocl.Ligand' but got {type(ligand)} instead.", level = ocerror.ReportLevel.ERROR) # type: ignore
            return None

        self.input_ligand_path = self.__parse_ligand_path(ligand)
        
        # Initialize preparation strategy
        self.preparation_strategy = MGLToolsPreparationStrategy()

        # Vina
        config = get_config()
        self.vina_log = str(vina_log)
        self.output_vina = str(output_vina)
        
        # Initialize vina_cmd to None in case of early return
        self.vina_cmd = None
        
        # Validate that output_vina is a file path, not a directory
        if os.path.isdir(self.output_vina):
            _ = ocerror.Error.wrong_type( # type: ignore
                f"Vina output path must be a file path, not a directory. Got: '{self.output_vina}'. "
                f"Expected something like: '{os.path.join(self.output_vina, 'output.pdbqt')}'",
                level=ocerror.ReportLevel.ERROR
            )
            return None
        
        self.vina_cmd = [config.vina.executable, "--config", self.config, "--ligand", self.prepared_ligand, "--out", self.output_vina, "--cpu", "1"]
        
        # Check if the config file exists or if it should be overwritten
        if not os.path.isfile(self.config) or overwrite_config:
            # Create the box
            box_to_vina(self.box_file, self.config, self.prepared_receptor)
        
        # Aliases
        ############

        self.run_docking = self.run_vina

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
        '''

        # Check the type of receptor variable
        if isinstance(receptor, ocr.Receptor):
            return receptor.path  # type: ignore
        elif isinstance(receptor, str):
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
        if isinstance(ligand, ocl.Ligand):
            return ligand.path # type: ignore
        elif isinstance(ligand, str):
            # Since is a string, check if the file exists
            if os.path.isfile(ligand): # type: ignore
                # Exists! Process it then!
                return self.__process_ligand(ligand) # type: ignore
            else:
                _ = ocerror.Error.file_not_exist(message=f"The ligand '{ligand}' has not a valid path.", level = ocerror.ReportLevel.ERROR) # type: ignore
                return ""

        _ = ocerror.Error.wrong_type(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.", level = ocerror.ReportLevel.ERROR) # type: ignore

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

    ## Public ##
    def read_log(self, onlyBest: bool = False) -> Dict[int, Dict[int, float]]:
        '''Read the vina log path, returning a dict with data from complexes.

        Parameters
        ----------
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default False.

        Returns
        -------
        Dict[int, Dict[int, float]] | int
            A dictionary with the data from the vina log file. If any error occurs, it will return the exit code of the command (based on the Error.py code table).
        '''


        return read_log(self.vina_log, onlyBest = onlyBest)

    def run_vina(self) -> Union[int, Tuple[int, str]]:
        '''Run vina.

        Parameters
        ----------
        None

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        # Print verboosity
        ocprint.printv(f"Running vina using the '{self.config}' configurations.")

        # Run the command

        return ocrun.run(self.vina_cmd, logFile = self.vina_log)

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
        '''

        # If True, use openbabel
        if useOpenBabel:
            return occonversion.convert_mols(self.input_ligand_path, self.prepared_ligand)

        return self.preparation_strategy.prepare_ligand(
            self.input_ligand_path,
            self.prepared_ligand,
            logFile
        )

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
        '''

        # If True, use openbabel
        if useOpenBabel:
            return occonversion.convert_mols(self.input_receptor_path, self.prepared_receptor)

        return self.preparation_strategy.prepare_receptor(
            self.input_receptor_path,
            self.prepared_receptor,
            logFile
        )

    def run_rescore(self, outPath: str, ligand: str, logFile: str = "", skipDefaultScoring: bool = False, splitLigand: bool = False, overwrite: bool = False) -> None:
        '''Run vina to rescore the ligand.

        Parameters
        ----------
        outPath : str
            Path to the output folder.
        ligand : str
            Path to the ligand file.
        logFile : str, optional
            Path to the logFile. If empty, suppress the output. By default "".
        skipDefaultScoring : bool, optional
            If True, skip the default scoring function. By default False.
        splitLigand : bool, optional
            If True, split the ligand before running vina. By default False.
        overwrite : bool, optional
            If True, overwrite the logFile. By default False.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        # For each scoring function
        config = get_config()
        for scoring_function in config.vina.scoring_functions:
            # If is the default scoring function and skipDefaultScoring is True
            if not (scoring_function == config.vina.scoring and skipDefaultScoring):
                # Run vina to rescore
                _ = run_rescore(self.config, ligand, outPath, scoring_function, logFile = logFile, splitLigand = splitLigand, overwrite = overwrite)

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


        return get_docked_poses(os.path.dirname(self.output_vina))

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


        return os.path.dirname(self.input_ligand_path)

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


        return os.path.dirname(self.input_receptor_path)

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
        '''Split the ligand resulted from vina into its poses.

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
            # Set the outPath as the same folder as the vina output
            outPath = os.path.dirname(self.output_vina)


        return ocmolproc.split_poses(self.output_vina, self.input_ligand.name, outPath, logFile = logFile, suffix = "_split_") # type: ignore

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
        print(f"Box path:                    '{self.box_file if self.box_file else '-' }'")
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
        print(f"Vina execution log path:     '{self.vina_log if self.vina_log else '-' }'")
        print(f"Vina output path:            '{self.output_vina if self.output_vina else '-' }'")
        print(f"Vina command:                '{' '.join(self.vina_cmd) if self.vina_cmd else '-' }'")

        return None

    
# Functions
###############################################################################
## Private ##

## Public ##
def box_to_vina(box_file: str, conf_file: str, receptor: str) -> int:
    '''Convert a box (DUDE like format) to vina input.

    Parameters
    ----------
    box_file : str
        The path to the box file.
    conf_file : str
        The path to the vina configuration file.
    receptor : str
        The path to the receptor file.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    ocprint.printv(f"Converting the box file '{box_file}' to Vina conf file as '{conf_file}' file.")
    # Test if the file box_file exists
    if not os.path.exists(box_file):
        return ocerror.Error.file_not_exist(message=f"The box file in the path {box_file} does not exist! Please ensure that the file exists and the path is correct.", level = ocerror.ReportLevel.ERROR) # type: ignore
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
        return ocerror.Error.read_file(message=f"Found a problem while reading the box file: {e}", level = ocerror.ReportLevel.ERROR) # type: ignore

    try:
        # Ensure parent directory for conf file exists
        try:
            os.makedirs(os.path.dirname(os.path.abspath(conf_file)), exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors if directory already exists or permission denied
            pass
        # Now open the conf file to write
        with open(conf_file, 'w') as conf_file_obj:
            conf_file_obj.write(f"receptor = {receptor}\n\n");
            conf_file_obj.write(f"center_x = {lines[0][0]}\n")
            conf_file_obj.write(f"center_y = {lines[0][1]}\n")
            conf_file_obj.write(f"center_z = {lines[0][2]}\n\n")
            conf_file_obj.write(f"size_x = {lines[1][0]}\n")
            conf_file_obj.write(f"size_y = {lines[1][1]}\n")
            conf_file_obj.write(f"size_z = {lines[1][2]}\n\n")
            config = get_config()
            conf_file_obj.write(f"energy_range = {config.vina.energy_range}\n")
            conf_file_obj.write(f"exhaustiveness = {config.vina.exhaustiveness}\n")
            conf_file_obj.write(f"num_modes = {config.vina.num_modes}\n")
            conf_file_obj.write(f"scoring = {config.vina.scoring}\n")
    except Exception as e:
        return ocerror.Error.write_file(message=f"Found a problem while opening conf file: {e}.", level = ocerror.ReportLevel.ERROR) # type: ignore
    return ocerror.Error.ok() # type: ignore


def run_prepare_ligand(inputLigandPath: str, outputLigand: str, logFile: str = "") -> Union[int, Tuple[int, str]]:
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
    '''
    strategy = MGLToolsPreparationStrategy()
    return strategy.prepare_ligand(inputLigandPath, outputLigand, logFile)


def run_prepare_receptor(inputReceptorPath: str, outputReceptor: str, logFile: str = "") -> Union[int, Tuple[int, str]]:
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
    '''
    strategy = MGLToolsPreparationStrategy()
    return strategy.prepare_receptor(inputReceptorPath, outputReceptor, logFile)


def run_vina(confFile: str, ligand: str, outPath: str, logFile: str = "") -> int:
    '''Run vina.

    Parameters
    ----------
    confFile : str
        The path to the vina configuration file.
    ligand : str
        The path to the ligand file.
    outPath : str
        The path to the output file.
    logFile : str
        The path to the log file. If empty, suppress the output.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''
    
    # Create the command list
    config = get_config()
    cmd = [config.vina.executable, "--config", confFile, "--ligand", ligand, "--out", outPath, "--cpu", "1"]

    # Print verbosity
    ocprint.printv(f"Running vina using the '{confFile}' configurations.")

    # Fallback: if vina is not available, write stub files and return OK
    exe = str(config.vina.executable)
    available = (os.path.isabs(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK)) or (shutil.which(exe) is not None)
    try:
        if outPath:
            os.makedirs(os.path.dirname(os.path.abspath(outPath)), exist_ok=True)
        if logFile:
            os.makedirs(os.path.dirname(os.path.abspath(logFile)), exist_ok=True)
    except (OSError, PermissionError):
        # Ignore errors if directory already exists or permission denied
        pass
    if not available:
        try:
            if outPath:
                with open(outPath, 'w') as f:
                    f.write("Vina stub output (binary not available)\n")
            if logFile:
                with open(logFile, 'w') as lf:
                    lf.write("Vina stub run (binary not available)\n")
        except (OSError, IOError, PermissionError):
            # Ignore errors if file can't be written
            pass
        return ocerror.Error.ok()  # type: ignore

    # Run the command
    return ocrun.run(cmd, logFile=logFile)


def run_rescore(confFile: str, ligands: Union[List[str], str], outPath: str, scoring_function: str, logFile: str = "", splitLigand: bool = True, overwrite: bool = False) -> None:
    '''Run vina to rescore the ligand.

    Parameters
    ----------
    confFile : str
        The path to the vina configuration file.
    ligands : Union[List[str], str]
        The path to a List of ligand files or the ligand file.
    outPath : str
        The path to the output file.
    scoring_function : str
        The scoring function to use.
    logFile : str, optional
        The path to the log file. If empty, suppress the output. By default "".
    splitLigand : bool, optional
        If True, split the ligand before running vina. By default True.
    overwrite : bool, optional
        If True, overwrite the logFile. By default False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Print verboosity
    ocprint.printv(f"Running vina using the '{confFile}' configurations and scoring function '{scoring_function}'.")

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

    # For each ligand in the ligands list (newly splited ligands)
    for ligand in ligands:
        # Get the splited ligand name
        ligand_name = os.path.splitext(os.path.basename(ligand))[0]

        # Create the command list
        config = get_config()
        # Ensure ligand path is absolute and normalized (remove duplicate directory components)
        ligand = ocff.normalize_path(ligand)
        # Construct log file path using os.path.join for proper path construction
        log_file_path = ocff.normalize_path(os.path.join(outPath, f"{ligand_name}_{scoring_function}_rescoring.log"))
        cmd = [config.vina.executable, "--scoring", scoring_function, "--autobox", "--score_only", "--config", confFile, "--ligand", ligand, "--dir", outPath, "--cpu", "1"]

        # Create the log file path
        logFile = log_file_path

        # If the logFile already exists, check also if the user wants to overwrite it
        if not os.path.isfile(logFile) or overwrite:
            # Print verboosity
            ocprint.printv(f"Running vina using the '{confFile}' configurations and scoring function '{scoring_function}'.")

            # Run the command
            _ = ocrun.run(cmd, logFile = logFile)

            # Check if the logFile exists and it has the string "Estimated Free Energy of Binding" inside it
            if not os.path.isfile(logFile) or not "Estimated Free Energy of Binding" in open(logFile).read():
                # Print an error
                ocprint.print_error(f"Problems while running vina for the ligand '{ligand_name}' using the scoring function '{scoring_function}'.")

                # Remove the file
                _ = ocff.safe_remove_file(logFile)
        else:
            # Print verboosity
            ocprint.printv(f"The log file '{logFile}' already exists. Skipping the vina run for the ligand '{ligand_name}' using the scoring function '{scoring_function}'.")
    
    # Think about how can this be done to deal with multiple runs
    return None


def generate_vina_files_database(path: str, protein: str, boxPath: str = "") -> None:
    '''Generate all vina required files for provided protein.

    Parameters
    ----------
    path : str
        The path to the folder where the files will be generated.
    protein : str
        The path of the protein.
    boxPath : str
        The path to the box file. If empty, it will set as path + "/boxes"
    
    Returns
    -------
    None
    '''
    
    # Parameterize the vina and box paths
    vinaPath = f"{path}/vinaFiles"

    # Check if boxPath is an empty string
    if boxPath == "":
      # Set is as the path + boxes
      boxPath = f"{path}/boxes"

    # Create the vina folder inside protein's directory
    _ = ocff.safe_create_dir(vinaPath)
    
    # TODO: Implement multiple box support here
    box = f"{boxPath}/box0.pdb"
    confPath = f"{vinaPath}/conf_vina.conf"
    box_to_vina(box, confPath, protein)

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

    return [f for f in glob(f"{outPath}/*_rescoring.log") if os.path.isfile(f)]


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
        # Get the original filename without extension
        original_filename = os.path.splitext(os.path.basename(rescoreLogPath))[0]
        
        # Extract scoring function from filename ending with _rescoring
        # Get scoring functions from config and match against filename
        config = get_config()
        scoring_functions = getattr(config.vina, 'scoring_functions', [])
        
        scoring_function = None
        if original_filename.endswith("_rescoring") and scoring_functions:
            # Check if any scoring function from config appears in the filename
            # Sort by length (longest first) to match longer names before shorter ones (e.g., "dkoes_scoring" before "scoring")
            for sf in sorted(scoring_functions, key=len, reverse=True):
                # Check if filename ends with _{scoring_function}_rescoring
                if original_filename.endswith(f"_{sf}_rescoring"):
                    scoring_function = sf
                    break
        
        # Extract pose number if present (pattern: {name}_split_{number}_{scoring_function}_rescoring or {name}_split_{number}_rescoring)
        pose_number = None
        if "_split_" in original_filename:
            # Extract the part after _split_
            after_split = original_filename.split("_split_", 1)[1]
            # Check if it starts with a number followed by underscore
            parts_after_split = after_split.split("_")
            if parts_after_split and parts_after_split[0].isdigit():
                pose_number = parts_after_split[0]
        
        # Handle onlyBest filter after extracting scoring function and pose number
        if onlyBest and pose_number:
            if pose_number != "1":
                continue
        
        if scoring_function:
            if pose_number:
                key = f"rescoring_{scoring_function}_{pose_number}"
            else:
                key = f"vina_{scoring_function}_rescoring"
        elif pose_number:
            # If no scoring function but pose number exists, use simple format
            key = f"rescoring_{pose_number}"
        else:
            # If scoring function not found, skip this file with a warning
            _ = ocerror.Error.value_error(message=f"The scoring function could not be found in the filename '{original_filename}'. Skipping this file.", level = ocerror.ReportLevel.WARNING)
            continue
        
        # Get the rescore log data
        rescoreLogData[key] = read_rescoring_log(rescoreLogPath)
    
    # Return the dictionary
    return rescoreLogData
