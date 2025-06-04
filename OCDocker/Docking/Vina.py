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

from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun
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

Licensed under the Apache License, Version 2.0 (January 2004)
See: http://www.apache.org/licenses/LICENSE-2.0

Commercial use requires a separate license.  
Contact: Artur Duque Rossi - arturossi10@gmail.com
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
        '''

        self.name = str(name)
        self.config = str(configPath)
        self.boxFile = str(boxFile)

        # Receptor
        if isinstance(receptor, ocr.Receptor):
            self.inputReceptor = receptor
        else:
            ocerror.Error.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR) # type: ignore
            return None
        
        # Check if the folder where the configPath is located exists (remove the file name from the path)
        _ = ocff.safe_create_dir(Path(self.config).parent)

        self.inputReceptorPath = self.__parse_receptor_path(receptor)
        self.preparedReceptor = str(preparedReceptorPath)
        self.prepareReceptorCmd = [pythonsh, prepare_receptor, "-r", self.inputReceptorPath, "-o", self.preparedReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]

        # Ligand
        self.preparedLigand = str(preparedLigandPath)
        
        # Check the type of the ligand
        if isinstance(ligand, ocl.Ligand):   
            self.inputLigand = ligand
            # Create the vinaFiles folder
            _ = ocff.safe_create_dir(os.path.join(os.path.dirname(ligand.path), "vinaFiles"))
        else:
            ocerror.Error.wrong_type(f"The ligand '{ligand}' has not a supported type. Expected 'ocl.Ligand' but got {type(ligand)} instead.", level = ocerror.ReportLevel.ERROR) # type: ignore
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

        # Get the extension
        ligandExtension = os.path.splitext(ligandPath)[1]

        # If its mol2 we do not need to convert it
        if ligandExtension == "mol2":
            # So return the ligandPath
            return ligandPath

        # Create the output path
        outputLigandPath = f"{os.path.dirname(ligandPath)}/{os.path.splitext(os.path.basename(ligandPath))[0]}.mol2"

        # Process the ligand
        occonversion.convertMols(ligandPath, outputLigandPath)

        return outputLigandPath

    ## Public ##

    def read_log(self, onlyBest = False) -> Dict[int, Dict[int, float]]:
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

        return read_log(self.vinaLog, onlyBest = onlyBest)

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
        return ocrun.run(self.vinaCmd, logFile = self.vinaLog)

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

        # Print verboosity
        ocprint.printv(f"Running '{prepare_ligand}' for '{self.inputLigandPath}'.")
        # If True, use openbabel
        if useOpenBabel:
            return occonversion.convertMols(self.inputLigandPath, self.preparedLigand)
        return ocrun.run(self.prepareLigandCmd, logFile=logFile, cwd=os.path.dirname(self.inputLigandPath))

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

        # Print verboosity
        ocprint.printv(f"Running '{prepare_receptor}' for '{self.inputReceptorPath}'.")
        # If True, use openbabel
        if useOpenBabel:
            return occonversion.convertMols(self.inputReceptorPath, self.preparedReceptor)
        return ocrun.run(self.prepareReceptorCmd, logFile=logFile, cwd=self.get_input_receptor_path())

    def run_rescore(self, outPath: str, logFile: str = "", skipDefaultScoring: bool = False, overwrite = False) -> None:
        '''Run vina to rescore the ligand.

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
        for scoring_function in vina_scoring_functions:
            # If is the default scoring function and skipDefaultScoring is True
            if not (scoring_function == smina_scoring and skipDefaultScoring):
                # Run smina to rescore
                _ = run_rescore(self.config, self.outputVina, outPath, scoring_function, logFile = logFile, splitLigand = splitLigand, overwrite = overwrite)

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

        return get_docked_poses(os.path.dirname(self.outputVina))

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
            outPath = os.path.dirname(self.outputVina)

        return ocmolproc.split_poses(self.outputVina, self.inputLigand.name, outPath, logFile = logFile, suffix = "_split_") # type: ignore
        
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
    '''

    ocprint.printv(f"Converting the box file '{boxFile}' to Vina conf file as '{confFile}' file.")
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
            conf_file.write(f"scoring = {vina_scoring}\n")
    except Exception as e:
        return ocerror.Error.write_file(message=f"Found a problem while opening conf file: {e}.", level = ocerror.ReportLevel.ERROR) # type: ignore
    return ocerror.Error.ok() # type: ignore

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
    '''

    # Create the command list
    cmd = [pythonsh, prepare_ligand, "-l", inputLigandPath, "-C", "-o", outputLigand]

    # Print verboosity
    ocprint.printv(f"Running '{prepare_ligand}' for '{inputLigandPath}'.")

    # Run the command
    return ocrun.run(cmd, logFile=logFile, cwd = os.path.dirname(inputLigandPath))

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
    '''

    # Create the command list
    cmd = [pythonsh, prepare_receptor, "-r", inputReceptorPath, "-o", outputReceptor, "-A", "hydrogens", "-U", "nphs_lps_waters"]

    # Print verboosity
    ocprint.printv(f"Running '{prepare_receptor}' for '{inputReceptorPath}'.")

    # Run the command
    return ocrun.run(cmd, logFile=logFile)

def run_vina(confFile: str, ligand: str, outPath: str, logFile: str = ""):
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
    cmd = [vina, "--config", confFile, "--ligand", ligand, "--out", outPath, "--cpu", "1"]

    # Print verboosity
    ocprint.printv(f"Running vina using the '{confFile}' configurations.")

    # Get the result of the command
    return ocrun.run(cmd, logFile = logFile)

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
        cmd = [vina, "--scoring", scoring_function, "--autobox", "--score_only", "--config", confFile, "--ligand", ligand, "--dir", f"{outPath}", "--cpu", "1"]

        # Create the log file path
        logFile = f"{outPath}/{ligand_name}_{scoring_function}_rescoring.log"

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
run_docking = run_vina
