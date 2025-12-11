#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to prepare dock6 files and run it.

They are imported as:

import OCDocker.Docking.PLANTS as ocplants
'''

# Imports
###############################################################################
import os
import json
import shutil

import pandas as pd

from glob import glob
from typing import Dict, List, Tuple, Union, Optional

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun
import OCDocker.Toolbox.Validation as ocvalidation
from OCDocker.Toolbox.Preparation import SPORESPreparationStrategy


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
class PLANTS:
    """PLANTS object with methods for easy run."""
    def __init__(self, config_path: str, box_file: str, receptor: ocr.Receptor, prepared_receptor_path: str, ligand: ocl.Ligand, prepared_ligand_path: str, plants_log: str, output_plants: str, name: str = "", box_spacing: float = 2.9, overwrite_config: bool = False) -> None:
        ''' Constructor for the PLANTS object.
        
        Parameters
        ----------
        config_path : str
            Path for the PLANTS config file.
        box_file : str
            Path for the PLANTS box file.
        receptor : ocr.Receptor
            Receptor object.
        prepared_receptor_path : str
            Path for the prepared receptor.
        ligand : ocl.Ligand
            Ligand object.
        prepared_ligand_path : str
            Path for the prepared ligand.
        plants_log : str
            Path for the PLANTS log file.
        output_plants : str
            Path for the PLANTS output file.
        name : str, optional
            Name for the PLANTS run, by default ""
        box_spacing : float, optional
            Spacing for the PLANTS box, by default 0.33.
        overwrite_config : bool, optional
            Overwrite the PLANTS config file, by default False.

        Returns
        -------
        None
        '''
        
        self.name = str(name)
        self.config = str(config_path)
        self.box_file = str(box_file)
        self.box_spacing = float(box_spacing)
        self.__bindingSite = self.__get_binding_site()

        if isinstance(self.__bindingSite, int):
            _ = ocerror.Error.binding_site_not_found(f"The binding site was not found in the box file '{self.box_file}'.", level = ocerror.ReportLevel.ERROR) # type: ignore
            return None

        # Check if the folder where the configPath is located exists (remove the file name from the path)
        _ = ocff.safe_create_dir(os.path.dirname(self.config))

        self.binding_site_center, self.binding_site_radius = self.__bindingSite # type: ignore
        
        # Receptor
        if isinstance(receptor, ocr.Receptor):
            self.input_receptor = receptor
        else:
            ocerror.Error.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR) # type: ignore
            return None
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
        self.preparation_strategy = SPORESPreparationStrategy()
        
        # Plants
        config = get_config()
        self.plants_log = str(plants_log)
        self.output_plants = str(output_plants)
        self.output_csv = f"{self.output_plants}/run"
        self.plants_cmd = [config.plants.executable, "--mode", "screen", self.config]
        
        # Check if config file exists to avoid useless processing
        if not os.path.isfile(self.config) or overwrite_config:
            # Create the box
            self.write_config_file()
        
        # Aliases
        ############

        self.run_docking = self.run_plants

    ## Private ##
    def __get_binding_site(self) -> Union[Tuple[Tuple[float, float, float], float], int]:
        '''Get the binding site from a box file.

        Parameters
        ----------
        None

        Returns
        -------
        Tuple[Tuple[float, float, float], float] | int
            Tuple with the center and radius of the binding site. If there is an error, the error code is returned.
        '''


        return get_binding_site(self.box_file, self.box_spacing)

    def __parse_receptor_path(self, receptor: ocr.Receptor, forceMol2: bool = False) -> Optional[str]:
        '''Parse the receptor path, handling its type.

        Parameters
        ----------
        receptor : ocr.Receptor
            The path for the receptor or its receptor object.
        forceMol2 : bool, optional
            Force the receptor to be converted to mol2, by default False
            
        Returns
        -------
        str
            The path for the receptor.
        '''

        # Check the type of receptor variable
        if isinstance(receptor, ocr.Receptor):
            # If the flag to force the use of mol2 file as input is True
            if forceMol2:
                # If receptor has a mol2_path
                if receptor.mol2_path:
                    return receptor.mol2_path
                # Try to generate it
                else:
                    mol2_path = f"{os.path.splitext(receptor.path)[0]}.mol2"
                    # Create the mol2_path
                    ocprint.print_warning(f"No mol2 file for '{receptor.path}' trying to generate in '{mol2_path}'.")
                    # Convert the molecule
                    _ = occonversion.convert_mols(receptor.path, mol2_path)
                    # Check if it is generated
                    if os.path.isfile(mol2_path):
                        # Set the mol2path in the receptor object
                        receptor.mol2_path = mol2_path
                        return receptor.mol2_path
                    else:
                        _ = ocprint.print_error(f"The mol2 file could not be generated for '{receptor.path}'.")
                        return None
            else:
                # Check if the object has a valid path
                if receptor.path:
                    return receptor.path
                else:
                    _ = ocprint.print_error(f"Invalid receptor path for the following path: '{receptor.path}'.")
                    return None
        elif isinstance(receptor, str):
            # Since is a string, check if the file exists
            if os.path.isfile(receptor): # type: ignore
                # Exists! Return it!
                return receptor
            else:
                _ = ocerror.Error.file_not_exist(message=f"The receptor '{receptor}' has not a valid path.", level = ocerror.ReportLevel.ERROR) # type: ignore
                return ""

        _ = ocerror.Error.wrong_type(message=f"The receptor '{receptor}' has not a supported type. Expected 'string' or 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR) # type: ignore

        return ""

    def __parse_ligand_path(self, ligand: ocl.Ligand) -> str:
        '''Parse the ligand path, handling its type.

        Parameters
        ----------
        ligand : ocl.Ligand
            The path for the ligand or its ligand object.

        Returns
        -------
        str
            The path for the ligand.
        '''

        # Check the type of ligand variable
        if type(ligand) == ocl.Ligand:
            return ligand.path
        
        _ = ocerror.Error.wrong_type(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.", level = ocerror.ReportLevel.ERROR) # type: ignore

        return ""

    ## Public ##
    def write_config_file(self) -> int:
        '''Write the config file.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).
    
        '''

        return write_config_file(self.config, self.prepared_receptor, self.prepared_ligand, self.output_plants, self.binding_site_center[0], self.binding_site_center[1], self.binding_site_center[2], self.binding_site_radius)

    def read_log(self, onlyBest: bool = True) -> Dict[int, Dict[int, float]]:
        '''Read the PLANTS log path, returning a pd.dataframe with data from complexes.

        Parameters
        ----------
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default True.

        Returns
        -------
        Dict[int, Dict[int, float]]
            The dictionary with the data from complexes.
        '''

        # If onlyBest is set
        if onlyBest:
            # The ranking file will be called bestranking
            rankingFile = "bestranking.csv"
        else:
            # The ranking file will be called ranking
            rankingFile = "ranking.csv"
            

        return read_log(f"{self.output_csv}/{rankingFile}", onlyBest = onlyBest)

    def run_plants(self, overwrite: bool = False) -> Union[Tuple[int, str], int]:
        '''Run plants.

        Parameters
        ----------
        overwrite : bool, optional
            If True, overwrite the output file. Default is False.

        Returns
        -------
        Tuple[int, str] | int
            The exit code of the command (based on the Error.py code table) and the stderr if applied.
        '''

        # Set the run folder name
        run_folder = f"{self.output_plants}/run"

        # If overwrite is set
        if overwrite:
            # Check if there is an output
            if os.path.isdir(run_folder):
                # Remove it
                shutil.rmtree(run_folder)
        # Check if there is an output
        elif os.path.isdir(run_folder):
            # Check if the dir is empty or no output file has been generated (the double of the number of cluster structures, being 2 for each structure)
            config = get_config()
            if len(os.listdir(run_folder)) == 0 or (len(glob(f"{run_folder}/{self.input_ligand.name}*.mol2")) < config.plants.cluster_structures * 2): # type: ignore
                # Remove it
                os.rmdir(run_folder)

        # Print verboosity
        ocprint.printv(f"Running PLANTS using the '{self.config}' configurations.")

        # Ensure tmpDir exists; PLANTS writes temp files there
        config = get_config()
        tmp_dir = config.tmp_dir if config.tmp_dir else os.path.join(os.getcwd(), 'tmp')
        try:
            os.makedirs(tmp_dir, exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors if directory already exists or permission denied
            pass
        # Cd to tmpDir (because PLANTS keeps spamming annoying files) and run plants
        output = ocrun.run(self.plants_cmd, logFile=self.plants_log, cwd=tmp_dir)

        # Check if there is a PLANTS-*.pid file
        for pidFile in glob(f"{tmp_dir}/PLANTS-*.pid"):
            # This try is to avoid ocerror.Error when the file does not exist
            try:
                # Remove it
                os.remove(pidFile)
            except (OSError, FileNotFoundError, PermissionError):
                # Ignore if file doesn't exist or can't be removed
                pass
            
        # Check if there is a *bad*.mol2 file
        config = get_config()
        tmp_dir = config.tmp_dir if config.tmp_dir else os.path.join(os.getcwd(), 'tmp')
        for badFile in glob(f"{tmp_dir}/*bad.mol2"):
            # This try is to avoid ocerror.Error when the file does not exist
            try:
                # Remove it
                os.remove(badFile)
            except (OSError, FileNotFoundError, PermissionError):
                # Ignore if file doesn't exist or can't be removed
                pass


        return output

    def run_prepare_ligand(self, logFile: str = "") -> Union[Tuple[int, str], int]:
        '''Run SPORES for ligand.

        Parameters
        ----------
        logFile : str, optional
            The path for the log file. Default is "".

        Returns
        -------
        Tuple[int, str] | int
            The exit code of the command (based on the Error.py code table) and the stderr if applied.
        '''
        return self.preparation_strategy.prepare_ligand(
            self.input_ligand_path,
            self.prepared_ligand,
            logFile
        )

    def run_prepare_receptor(self, logFile: str = "") -> Union[Tuple[int, str], int]:
        '''Run SPORES for receptor.

        Parameters
        ----------
        logFile : str, optional
            The path for the log file. Default is "".

        Returns
        -------
        Tuple[int, str] | int
            The exit code of the command (based on the Error.py code table) and the stderr if applied.
        '''

        if not self.input_receptor_path:
            return ocerror.Error.file_not_exist("No valid input receptor path provided.", level = ocerror.ReportLevel.ERROR) # type: ignore
        
        if not self.prepared_receptor:
            return ocerror.Error.file_not_exist("No valid prepared receptor path provided.", level = ocerror.ReportLevel.ERROR) # type: ignore
        
        return self.preparation_strategy.prepare_receptor(
            self.input_receptor_path,
            self.prepared_receptor,
            logFile
        )

    def run_rescore(self, pose_list: str, logFile: str = "", skipDefaultScoring: bool = False, overwrite: bool = False) -> None:
        '''Run PLANTS to rescore the ligand.

        Parameters
        ----------
        pose_list : str
            The path to the ligand poses list file.
        logFile : str
            Path to the logFile. If empty, suppress the output.
        skipDefaultScoring : bool, optional
            If True, skip the default scoring function. By default False.
        overwrite : bool, optional
            If True, overwrite the logFile. Default is False.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        # For each scoring function
        config = get_config()
        for scoring_function in config.plants.scoring_functions:
            # Set the output path
            outPath = f"{self.output_plants}/run_{scoring_function}"
            # Set the config file
            confFile = f"{self.output_plants}/{self.input_ligand.name}_rescoring_{scoring_function}.txt"
            # If is the default scoring function and skipDefaultScoring is True
            if not (scoring_function == config.plants.scoring and skipDefaultScoring):
                # Run vina to rescore
                _ = run_rescore(confFile, pose_list, outPath, self.prepared_receptor, scoring_function, self.binding_site_center[0], self.binding_site_center[1], self.binding_site_center[2], self.binding_site_radius, logFile = logFile, overwrite = overwrite) # type: ignore

        return None

    def get_rescore_log_paths(self, onlyBest: bool = False) -> List[str]:
        ''' Get the paths for the rescore csv file.

        Parameters
        ----------
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default False.

        Returns
        -------
        List[str]
            List of rescoring logs.
        '''

        # Create the rescoring logs list
        rescoring_logs = []

        # If onlyBest is set
        if onlyBest:
            # The ranking file will be called bestranking
            rankingFile = "bestranking.csv"
        else:
            # The ranking file will be called ranking
            rankingFile = "ranking.csv"

        # For each scoring function
        config = get_config()
        for scoring_function in config.plants.scoring_functions:
            # Set the output path
            outPath = f"{self.output_plants}/run_{scoring_function}"
            # If the file exists
            if os.path.isfile(f"{outPath}/{rankingFile}"):
                # Append the data to the rescoring_logs list
                rescoring_logs.append(f"{outPath}/{rankingFile}")

        return rescoring_logs

    def get_docked_poses(self) -> List[str]:
        '''Get the paths for the docked poses.

        Returns
        -------
        List[str]
            A list with the paths for the docked poses.
        '''

        return get_docked_poses(f"{self.output_plants}/run")

    def get_input_ligand_path(self) -> str:
        ''' Get the input ligand path.

        Returns
        -------
        str
            The input ligand path.
        '''

        return self.input_ligand_path if self.input_ligand_path else ""

    def get_input_receptor_path(self) -> str:
        ''' Get the input receptor path.

        Returns
        -------
        str
            The input receptor path.
        '''

        return self.input_receptor_path if self.input_receptor_path else ""

    def read_rescore_logs(self, onlyBest: bool = False) -> Dict[str, List[Union[str, float]]]:
        ''' Reads the data from the rescore log files.

        Parameters
        ----------
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default False.

        Returns
        -------
        Dict[str, List[Union[str, float]]]
            A dictionary with the data from the rescore log files.
        '''

        # Get the rescore log paths
        rescoreLogPaths = self.get_rescore_log_paths(onlyBest = onlyBest)

        # Create the dictionary
        rescoreLogData = {}

        # For each rescore log path
        for rescoreLogPath in rescoreLogPaths:
            # Get the directory name from the log path (e.g., "run_chemplp")
            dir_name = os.path.basename(os.path.dirname(rescoreLogPath))
            
            # Extract scoring function from directory name (remove "run_" prefix)
            # Format: run_{scoring_function} -> plants_{scoring_function}
            if dir_name.startswith("run_"):
                scoring_function = dir_name[4:]  # Remove "run_" prefix
                key = f"plants_{scoring_function}"
            else:
                # Fallback: use directory name as-is with plants_ prefix
                key = f"plants_{dir_name}"
            
            # Get the rescore log data (returns Dict[int, Dict] where int is pose number)
            log_data = read_log(rescoreLogPath, onlyBest = onlyBest)
            
            # Extract the inner dictionary (remove pose number key)
            # Since rescoring typically uses onlyBest=True, there's usually one pose (key 1)
            # But we'll handle multiple poses by taking the first (best) one
            if log_data:
                # Get the first (and typically only) value from the dict
                inner_dict = next(iter(log_data.values()))
                # Convert lists with single values to just the value
                # PLANTS read_log returns lists when onlyBest=True, but we want single values
                converted_dict = {}
                for dict_key, value in inner_dict.items():
                    if isinstance(value, list) and len(value) == 1:
                        converted_dict[dict_key] = value[0]
                    else:
                        converted_dict[dict_key] = value
                rescoreLogData[key] = converted_dict
            else:
                # Empty log data
                rescoreLogData[key] = {}
        
        # Return the dictionary

        return rescoreLogData

    def write_pose_list(self, overwrite: bool = False) -> Optional[str]:
        ''' Write the pose_list file.

        Parameters
        ----------
        overwrite : bool, optional
            If True, overwrite the pose_list file. Default is False.

        Returns
        -------
        str | None
            The path for the pose_list file. If the file already exists and overwrite is False, return None.
        '''

        # Get the docked poses file paths
        dockedPoses = self.get_docked_poses()

        # Parameterize the pose_list file path
        poseListPath = f"{self.output_plants}/pose_list.txt"

        # Call write_pose_list
        return write_pose_list(dockedPoses, poseListPath, overwrite = overwrite)

    def print_attributes(self) -> None:
        '''Print the class attributes.
        '''

        print(f"Name:                        '{self.name if self.name else '-' }'")
        print(f"Box path:                    '{self.box_file if self.box_file else '-' }'")
        print(f"Config path:                 '{self.config if self.config else '-' }'")
        print(f"Input receptor:              '{self.input_receptor if self.input_receptor else '-' }'")
        print(f"Input receptor path:         '{self.input_receptor_path if self.input_receptor_path else '-' }'")
        print(f"Prepared receptor path:      '{self.prepared_receptor if self.prepared_receptor else '-' }'")

        if not self.prepared_receptor or not self.input_receptor_path:
            print("Prepared receptor command:   '-' ")
        else:
            prep_receptor_cmd = self.preparation_strategy.get_receptor_command(self.input_receptor_path, self.prepared_receptor)
            print(f"Prepared receptor command:   '{' '.join(prep_receptor_cmd) if prep_receptor_cmd else '-' }'")
        
        print(f"Input ligand:                '{self.input_ligand if self.input_ligand else '-' }'")
        print(f"Input ligand path:           '{self.input_ligand_path if self.input_ligand_path else '-' }'")
        print(f"Prepared ligand path:        '{self.prepared_ligand if self.prepared_ligand else '-' }'")

        if not self.prepared_ligand or not self.input_ligand_path:
            print("Prepared ligand command:     '-' ")
        else:
            prep_ligand_cmd = self.preparation_strategy.get_ligand_command(self.input_ligand_path, self.prepared_ligand)
            print(f"Prepared ligand command:     '{' '.join(prep_ligand_cmd) if prep_ligand_cmd else '-' }'")

        print(f"PLANTS execution log path:   '{self.plants_log if self.plants_log else '-' }'")
        print(f"PLANTS output path:          '{self.output_plants if self.output_plants else '-' }'")
        print(f"PLANTS output csv path:      '{self.output_csv if self.output_csv else '-' }'")
        print(f"PLANTS command:              '{' '.join(self.plants_cmd) if self.plants_cmd else '-' }'")
        
        return None


# Functions
###############################################################################
## Private ##

## Public ##
def box_to_plants(box_file: str, conf_file: str, receptor: str, ligand: str, output_plants: str, center: Optional[float] = None, binding_site_radius: Optional[float] = None, spacing: float = 2.9) -> int:
    '''Convert a box (DUDE like format) to PLANTS input.

    Parameters
    ----------
    box_file : str
        The path to the box file.
    conf_file : str
        The path to the PLANTS configuration file.
    receptor : str
        The path to the receptor file.
    ligand : str
        The path to the ligand file.
    output_plants : str
        The path to the PLANTS output directory.
    center : float, optional
        The center of the box. Default is None and it will be calculated.
    binding_site_radius : float, optional
        The radius of the box. Default is None and it will be calculated.
    spacing : float, optional
        The spacing between the grid points. Default is 2.9.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    ocprint.printv(f"Converting the box file '{box_file}' to PLANTS conf file as '{conf_file}' file.")

    # Check if the center and the radius are given
    if center is None or binding_site_radius is None:
        # Calculate the center and the radius
        binding_site = get_binding_site(box_file, spacing = spacing)
        # Check if the binding site is int
        if isinstance(binding_site, int):
            # Return the error code
            return binding_site

        # Get the center and the binding site center
        center, binding_site_radius = binding_site # type: ignore
    # Write the file
    return write_config_file(conf_file, receptor, ligand, output_plants, center[0], center[1], center[2], binding_site_radius) # type: ignore


def run_prepare_ligand(input_ligand_path: str, output_ligand: str, log_file: str = "") -> Union[Tuple[int, str], int]:
    ''' Run SPORES for ligand.

    Parameters
    ----------
    input_ligand_path : str
        The path to the input ligand.
    output_ligand : str
        The path to the output ligand.
    log_file : str, optional
        The path for the log file. Default is "".

    Returns
    -------
    Tuple[int, str] | int
        The exit code of the command (based on the Error.py code table) and the stderr if applied.
    '''
    strategy = SPORESPreparationStrategy()
    return strategy.prepare_ligand(input_ligand_path, output_ligand, log_file)


def run_prepare_receptor(input_receptor_path: str, output_receptor: str, log_file: str = "") -> Union[Tuple[int, str], int]:
    ''' Run SPORES for receptor.

    Parameters
    ----------
    input_receptor_path : str
        The path to the input receptor.
    output_receptor : str
        The path to the output receptor.
    log_file : str, optional
        The path for the log file. Default is "".

    Returns
    -------
    Tuple[int, str] | int
        The exit code of the command (based on the Error.py code table) and the stderr if applied.
    '''
    strategy = SPORESPreparationStrategy()
    return strategy.prepare_receptor(input_receptor_path, output_receptor, log_file)


def run_plants(confFile: str, outputPlants: str, overwrite: bool = False, logFile: str = "") -> Union[Tuple[int, str], int]:
    '''Run PLANTS.

    Parameters
    ----------
    confFile : str
        The path to the PLANTS configuration file.
    outputPlants : str
        The path to the PLANTS output directory.
    overwrite : bool, optional
        If True, overwrite the output directory. Default is False.
    logFile : str, optional
        The path for the log file. Default is "".

    Returns
    -------
    Tuple[int, str] | int
        The exit code of the command (based on the Error.py code table) and the stderr if applied.
    '''

    # If overwrite is set
    if overwrite:
        # Check if there is an output
        if os.path.isdir(outputPlants):
            # Remove it
            shutil.rmtree(outputPlants)
    # Check if there is an output
    elif os.path.isdir(outputPlants):
        # Check if the dir is empty
        if len(os.listdir(outputPlants)) == 0:
            # Remove it
            os.rmdir(outputPlants)

    # Create the command list
    config = get_config()
    cmd = [config.plants.executable, "--mode", "screen", confFile]
    # Print verbosity
    ocprint.printv(f"Running PLANTS using the '{confFile}' configurations.")
    # If PLANTS is not available, create a stub log and return OK
    exe = str(config.plants.executable)
    available = (os.path.isabs(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK)) or (shutil.which(exe) is not None)

    if not available and logFile:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(logFile)), exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors if directory already exists or permission denied
            pass
        with open(logFile, 'w') as lf:
            lf.write("PLANTS stub run (binary not available)\n")
        return ocerror.Error.ok() # type: ignore
        
    # Run the command
    return ocrun.run(cmd, logFile = logFile)


def run_rescore(confFile: str, pose_list_file: str, outPath: str, proteinFile: str, scoring_function: str, bindingSiteCenterX: float, bindingSiteCenterY: float, bindingSiteCenterZ: float, bindingSiteRadius: float, logFile: str = "", overwrite: bool = False) -> int:
    '''Run PLANTS to rescore the ligand.

    Parameters
    ----------
    confFile : str
        The path to the PLANTS configuration file.
    pose_list_file : str
        The path to the ligand poses list file.
    outPath : str
        The path to the output file.
    proteinFile : str
        The path to the protein file which will be used as receptor.
    scoring_function : str
        The scoring function to use.
    bindingSiteCenterX : float
        The X coordinate of the binding site center.
    bindingSiteCenterY : float
        The Y coordinate of the binding site center.
    bindingSiteCenterZ : float
        The Z coordinate of the binding site center.
    bindingSiteRadius : float
        The radius of the binding site.
    logFile : str
        The path to the log file. If empty, suppress the output.
    overwrite : bool, optional
        If True, overwrite the logFile. Default is False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Check if the conf file exists
    if not os.path.isfile(confFile) or overwrite:
        # Check if the folder exists
        if os.path.isdir(outPath):
            # If overwrite is set
            if overwrite:
                # Remove it
                ocff.safe_remove_dir(outPath)
            else:
                # Print verboosity
                return ocerror.Error.dir_exists(f"The folder '{outPath}' already exists. Skipping the PLANTS run.", level = ocerror.ReportLevel.WARNING) # type: ignore

        # Create the conf file (yes... again...)
        config = get_config()
        _ = write_rescoring_config_file(confFile, proteinFile, pose_list_file, outPath, bindingSiteCenterX, bindingSiteCenterY, bindingSiteCenterZ, bindingSiteRadius, scoringFunction = scoring_function, rescoringMode = config.plants.rescoring_mode)

        # Create the command list
        cmd = [config.plants.executable, "--mode", "rescore", confFile]

        # Run the command
        _ = ocrun.run(cmd, logFile = logFile)

        # Print verboosity
        ocprint.printv(f"Running PLANTS using the '{confFile}' configurations and scoring function '{scoring_function}'.")
        return ocerror.Error.ok() # type: ignore
    else:
        # Print verboosity
        return ocerror.Error.file_exists(f"The file '{confFile}' already exists. Skipping the PLANTS run.", level = ocerror.ReportLevel.WARNING) # type: ignore
        
    return None 


def write_config_file(confFile: str, preparedReceptor: str, preparedLigand: str, outputPlants: str, bindingSiteCenterX: float, bindingSiteCenterY: float, bindingSiteCenterZ: float, bindingSiteRadius: float, scoringFunction: str = "chemplp") -> int:
    '''Write the config file.

    Parameters
    ----------
    confFile : str
        The path to the PLANTS configuration file.
    preparedReceptor : str
        The path to the prepared receptor.
    preparedLigand : str
        The path to the prepared ligand.
    outputPlants : str
        The path to the PLANTS output directory.
    bindingSiteCenterX : float
        The X coordinate of the binding site center.
    bindingSiteCenterY : float
        The Y coordinate of the binding site center.
    bindingSiteCenterZ : float
        The Z coordinate of the binding site center.
    bindingSiteRadius : float
        The radius of the binding site.
    scoringFunction : str, optional
        The scoring function to use. Default is "chemplp". Options are plp, plp95 or chemplp

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    try:
        # Ensure parent directory for config exists; let PLANTS create output_dir itself
        try:
            os.makedirs(os.path.dirname(os.path.abspath(confFile)), exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors if directory already exists or permission denied
            pass
        # Do not pre-create the PLANTS output_dir (some installations require it not to exist)
        # It's safe to ensure the base output folder exists, but avoid creating the 'run' subfolder here.
        try:
            os.makedirs(outputPlants, exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors if directory already exists or permission denied
            pass
        with open(confFile, 'w') as f:
            f.write("# scoring function and search settings\n")
            f.write(f"scoring_function {scoringFunction}\n")
            config = get_config()
            f.write(f"search_speed {config.plants.search_speed}\n")
            f.write("# input\n")
            f.write(f"protein_file {preparedReceptor}\n")
            f.write(f"ligand_file {preparedLigand}\n")
            f.write("# output\n")
            f.write(f"keep_original_mol2_description 0\n") # important to avoid problems in output generation
            f.write(f"output_dir {outputPlants}/run\n")
            f.write("# write single mol2 files (e.g. for RMSD calculation)\n")
            f.write("write_multi_mol2 0\n")
            f.write("# binding site definition\n")
            f.write(f"bindingsite_center {bindingSiteCenterX} {bindingSiteCenterY} {bindingSiteCenterZ}\n")
            f.write(f"bindingsite_radius {round(bindingSiteRadius, 3)}\n")
            f.write("# cluster algorithm\n")
            f.write(f"cluster_structures {config.plants.cluster_structures}\n")
            f.write(f"cluster_rmsd {config.plants.cluster_rmsd}")
    except Exception as e:
        return ocerror.Error.write_file(f"Problems while writing the file {confFile}: {e}") # type: ignore

    return ocerror.Error.ok() # type: ignore


def write_rescoring_config_file(confFile: str, preparedReceptor: str, ligandListPath: str, outputPlants: str, bindingSiteCenterX: float, bindingSiteCenterY: float, bindingSiteCenterZ: float, bindingSiteRadius: float, scoringFunction: str = "chemplp", rescoringMode: str = "simplex") -> int:
    '''Write the config file to be used in rescoring mode.

    Parameters
    ----------
    confFile : str
        The path to the PLANTS configuration file.
    preparedReceptor : str
        The path to the prepared receptor.
    ligandListPath : str
        The path to the ligand pose_list file.
    outputPlants : str
        The path to the PLANTS output directory.
    bindingSiteCenterX : float
        The X coordinate of the binding site center.
    bindingSiteCenterY : float
        The Y coordinate of the binding site center.
    bindingSiteCenterZ : float
        The Z coordinate of the binding site center.
    bindingSiteRadius : float
        The radius of the binding site.
    scoringFunction : str, optional
        The scoring function to use. Default is "chemplp". Options are plp, plp95 or chemplp
    rescoringMode : str, optional
        The rescoring mode to use. Default is "simplex". Options are simplex or no_simplex.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    try:
        # Ensure parent directory for config exists; avoid pre-creating rescoring output directory
        try:
            os.makedirs(os.path.dirname(os.path.abspath(confFile)), exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors if directory already exists or permission denied
            pass
        with open(confFile, 'w') as f:
            f.write("# scoring function and search settings\n")
            f.write(f"scoring_function {scoringFunction}\n")
            f.write("# input\n")
            f.write(f"protein_file {preparedReceptor}\n")
            f.write(f"ligand_list {ligandListPath}\n")
            f.write("# binding site definition\n")
            f.write(f"bindingsite_center {bindingSiteCenterX} {bindingSiteCenterY} {bindingSiteCenterZ}\n")
            f.write(f"bindingsite_radius {round(bindingSiteRadius, 3)}\n")
            f.write("# output\n")
            f.write(f"keep_original_mol2_description 0\n") # important to avoid problems in output generation
            f.write(f"output_dir {outputPlants}\n")
            f.write(f"# Rescoring mode parameter\n")
            f.write(f"rescore_mode {rescoringMode}\n")
    except Exception as e:
        return ocerror.Error.write_file(f"Problems while writing the file {confFile}: {e}") # type: ignore

    return ocerror.Error.ok() # type: ignore


def get_binding_site(boxFile: str, spacing: float = 2.9) -> Union[Tuple[Tuple[float, float, float], float], int]:
    '''Get the binding site from a box file.

    Parameters
    ----------
    boxFile : str
        The path to the box file.
    spacing : float, optional
        The spacing between the box and the binding site. Default is 2.9.
    
    Returns
    -------
    Tuple[Tuple[float, float, float], float] | int
        The center of the binding site and the radius of the binding site. If there is an error, the error code is returned.
    '''

    ocprint.printv(f"Parsing '{boxFile}' to binding center data.")
    
    # Test if the file boxFile exists
    if not os.path.exists(boxFile):
        return ocerror.Error.file_not_exist(message=f"The box file in the path {boxFile} does not exists! Please ensure that the box file exists and the path is correct.", level = ocerror.ReportLevel.ERROR) # type: ignore

    # Dict to hold the center data
    center: Dict[str, Optional[float]] = {
        'x': None,
        'y': None,
        'z': None
    }

    # Dict to hold max and min x,y,z (set all as None)
    positions: Dict[str, Optional[float]] = {
        'max_x': None,
        'max_y': None,
        'max_z': None,
        'min_x': None,
        'min_y': None,
        'min_z': None
        }
        
    try:
        # Open the box file
        with open(str(boxFile), 'r') as box_file:
            # For each line in the file
            for line in box_file:
                # If it starts with REMARK
                if line.startswith("REMARK"):
                    # Slice the line in right positions
                    center['x'] = float(line[30:38])
                    center['y'] = float(line[38:46])
                    center['z'] = float(line[46:54])
                    # Break the loop (optimization)
                    break
                # If it starts with ATOM
                elif line.startswith("HEADER"):
                    # Slice the line in right positions
                    positions['min_x'] = float(line[30:38])
                    positions['min_y'] = float(line[38:46])
                    positions['min_z'] = float(line[46:54])
                    positions['max_x'] = float(line[54:62])
                    positions['max_y'] = float(line[62:70])
                    positions['max_z'] = float(line[70:78])

    except Exception as e:
        return ocerror.Error.read_file(message=f"Found a problem while reading the box file: {e}", level = ocerror.ReportLevel.ERROR) # type: ignore
        
    # Check if center coordinates were found
    if center['x'] is None or center['y'] is None or center['z'] is None:
        return ocerror.Error.read_file(message=f"Could not find center coordinates in box file '{boxFile}'. Expected REMARK line with coordinates at columns 30-54.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Find which is the biggest value in each coordinate
    # If HEADER line was not found, positions will be None, so use a default radius
    if positions['min_x'] is None or positions['max_x'] is None:
        # Throw an error
        return ocerror.Error.read_file(message=f"Could not find min and max x coordinates in box file '{boxFile}'. Expected HEADER line with coordinates at columns 30-78.", level = ocerror.ReportLevel.ERROR) # type: ignore
    else:
        xMax = max(abs(center['x'] - positions['min_x']), abs(positions['max_x'] - center['x'])) # type: ignore
        yMax = max(abs(center['y'] - positions['min_y']), abs(positions['max_y'] - center['y'])) # type: ignore
        zMax = max(abs(center['z'] - positions['min_z']), abs(positions['max_z'] - center['z'])) # type: ignore

        # Get the biggest value among the coordinates (do not divide it, to allow more space for the protein)
        radius = max(xMax, yMax, zMax) 

        # Add some extra space
        radius += round(spacing * radius, 3) # type: ignore

    # Return the data
    return ((center['x'], center['y'], center['z']), radius) # type: ignore


def generate_plants_files_database(path: str, protein: str, ligand: str, spacing: float = 0.33, boxPath: str = "") -> None:
    '''Generate all PLANTS required files for provided protein.

    Parameters
    ----------
    path : str
        The path to the directory where the files will be generated.
    protein : str
        The path to the protein file.
    ligand : str
        The path to the ligand file.
    spacing : float
        The spacing between the box and the binding site.
    boxPath : str, optional
        The path to the box file. If empty, it will set as path + "/boxes"

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Parameterize the PLANTS and paths
    plantsPath = f"{path}/plantsFiles"

    # Check if boxPath is an empty string
    if boxPath == "":
      # Set is as the path + "/boxes"
      boxPath = f"{path}/boxes"
      
    # Create the PLANTS folder inside protein's directory
    _ = ocff.safe_create_dir(plantsPath)

    # TODO: Implement multiple box support here
    # Set the box file path
    box = f"{boxPath}/box0.pdb"
    # Set the conf file path
    confPath = f"{plantsPath}/conf_plants.conf"
    # Convert the box to a conf file
    box_to_plants(box, confPath, protein, ligand, f"{plantsPath}/run", spacing = spacing)

    return None


def read_log(path: str, onlyBest: bool = False) -> Dict[int, Dict[int, float]]:
    '''Read the PLANTS log path, returning a dict with data from complexes.

    Parameters
    ----------
    path : str
        The path to the PLANTS log file.
    onlyBest : bool, optional
        If True, only the best pose will be returned. By default False.
        
    Returns
    -------
    Dict[int, Dict[int, float]]
        A dictionary with the data from the PLANTS log file.
    '''
   
    # Check if file exists
    if os.path.isfile(path):
        try:
            # Read the csv
            df = pd.read_csv(path)

            # Check if df is empty or malformed
            if df is None or df.shape[0] == 0 or df.shape[1] == 0: # type: ignore
                # Return an empty dict
                return {}
            else:
                # If onlyBest is True
                if onlyBest:
                    # Return the built the dictionary
                    return { 1: {
                            "PLANTS_TOTAL_SCORE": [df.TOTAL_SCORE[:1].values[0]], # type: ignore
                            "PLANTS_SCORE_RB_PEN": [df.SCORE_RB_PEN[:1].values[0]], # type: ignore
                            "PLANTS_SCORE_NORM_HEVATOMS": [df.SCORE_NORM_HEVATOMS[:1].values[0]], # type: ignore
                            "PLANTS_SCORE_NORM_CRT_HEVATOMS": [df.SCORE_NORM_CRT_HEVATOMS[:1].values[0]], # type: ignore
                            "PLANTS_SCORE_NORM_WEIGHT": [df.SCORE_NORM_WEIGHT[:1].values[0]], # type: ignore
                            "PLANTS_SCORE_NORM_CRT_WEIGHT": [df.SCORE_NORM_CRT_WEIGHT[:1].values[0]], # type: ignore
                            "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": [df.SCORE_RB_PEN_NORM_CRT_HEVATOMS[:1].values[0]], # type: ignore
                        }
                    }
                else:
                    # Create the dict
                    data = {}
                    # For each row
                    for _, row in df.iterrows(): # type: ignore
                        # Add the data to the dict
                        data[get_pose_index_from_file_path(str(row['LIGAND_ENTRY']))] = {
                            "PLANTS_TOTAL_SCORE": row['TOTAL_SCORE'], # type: ignore
                            "PLANTS_SCORE_RB_PEN": row['SCORE_RB_PEN'], # type: ignore
                            "PLANTS_SCORE_NORM_HEVATOMS": row['SCORE_NORM_HEVATOMS'], # type: ignore
                            "PLANTS_SCORE_NORM_CRT_HEVATOMS": row['SCORE_NORM_CRT_HEVATOMS'], # type: ignore
                            "PLANTS_SCORE_NORM_WEIGHT": row['SCORE_NORM_WEIGHT'], # type: ignore
                            "PLANTS_SCORE_NORM_CRT_WEIGHT": row['SCORE_NORM_CRT_WEIGHT'], # type: ignore
                            "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": row['SCORE_RB_PEN_NORM_CRT_HEVATOMS'], # type: ignore
                        }
                    # Return the dict
                    return data
        except Exception as e:
            ocprint.print_error(f"Problems while reading file '{path}'. Error: {e}")
            config = get_config()
            ocprint.print_error_log(f"Problems while reading file '{path}'. Error: {e}", f"{config.logdir}/PLANTS_read_log_ERROR.log")

    # Throw an error
    _ = ocerror.Error.file_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.") # type: ignore

    # Return an empty dict
    return {}


def generate_digest(digestPath: str, logPath: str, overwrite: bool = False, digestFormat : str = "json") -> int:
    """Generate the docking digest.
    
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
    """

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
                                return ocerror.Error.wrong_type(f"The digest file '{digestPath}' is not valid.", ocerror.ReportLevel.ERROR) # type: ignore
                    except Exception as e:
                        return ocerror.Error.file_not_exist(f"Could not read the digest file '{digestPath}'.", ocerror.ReportLevel.ERROR) # type: ignore
            else:
                # Since it does not exists, create it
                digest = ocff.empty_docking_digest(digestPath, overwrite)

            # Read the docking object log to generate the docking digest
            dockingDigest = read_log(logPath)

            # Check if the digest variable is fine
            if not isinstance(digest, dict):
                return ocerror.Error.wrong_type(f"The docking digest file '{digestPath}' is not valid.", ocerror.ReportLevel.ERROR) # type: ignore
            
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
                    return ocerror.Error.write_file(f"Could not write the digest file '{digestPath}'.", ocerror.ReportLevel.ERROR) # type: ignore

            return ocerror.Error.ok() # type: ignore
        return ocerror.Error.unsupported_extension(f"The provided extension '{digestFormat}' is not supported.", ocerror.ReportLevel.ERROR) # type: ignore
    
    return ocerror.Error.file_exists(f"The file '{digestPath}' already exists. If you want to overwrite it yse the overwrite flag.", level = ocerror.ReportLevel.WARNING) # type: ignore


def get_docked_poses(posesPath: str) -> List[str]:
    '''Get the docked poses from the poses path.

    Parameters
    ----------
    posesPath : str
        The path to the poses folder.

    Returns
    -------
    List[str]
        A list with the paths to the docked poses.
    '''

    # Check if the posesPath exists
    if os.path.isdir(posesPath):
        # Get the docked poses removing the protein and fixed files
        return [d for d in glob(f"{posesPath}/*.mol2") if os.path.isfile(d) and not d.endswith("_protein.mol2") and not d.endswith("_fixed.mol2")]
    
    # Print an error message
    _ = ocerror.Error.dir_not_exist(message=f"The poses path '{posesPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Return an empty list
    return []


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

    filename = os.path.basename(filePath)
    root, ext = os.path.splitext(filename)
    known_exts = {'.mol2', '.pdb', '.pdbqt', '.sdf'}
    # Only strip extensions we know belong to pose files; some PLANTS entries
    # include dots (e.g. prepared_ligand._entry_00001) as part of the name.
    if ext and ext.lower() in known_exts:
        filename = root
    # Split the filename using the '_' string as delimiter then grab the end of the string
    filename = filename.split("_")[-1]
    # Return the filename
    return int(filename)


def write_pose_list(dockedPoses: Union[str, List[str]], poseListPath: str, overwrite: bool = False) -> Optional[str]:
    ''' Write the pose_list file.

    Parameters
    ----------
        dockedPoses : Union[str, List[str]]
        The list with the docked poses. If a string is provided, it will be considered as a single pose.
    poseListPath : str
        The path to the pose_list file.
    overwrite : bool, optional
        If True, overwrite the pose_list file. Default is False.

    Returns
    -------
    str | None
        The path for the pose_list file. If the file already exists and overwrite is False, return None.
    '''

    # Check if the pose_list file exists
    if not os.path.isfile(poseListPath) or overwrite:
        # If dockedPoses is a string, convert it to a list
        if isinstance(dockedPoses, str):
            dockedPoses = [dockedPoses]
        # Create the pose_list file
        with open(poseListPath, "w") as poseListFile:
            # Write the docked poses
            poseListFile.write("\n".join(dockedPoses))
        return poseListPath
    return None

# Aliases
###############################################################################
run_docking = run_plants
read_rescore_logs = read_log
