#!/usr/lib/python3

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
import vaex

import numpy as np

from glob import glob
from typing import Dict, List, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun
import OCDocker.Toolbox.Validation as ocvalidation

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

# Classes
###############################################################################
class PLANTS:
    """PLANTS object with methods for easy run."""
    def __init__(self, configPath: str, boxFile: str, receptor: ocr.Receptor, preparedReceptorPath: str, ligand: ocl.Ligand, preparedLigandPath: str, plantsLog: str, outputPlants: str, name: str = "", boxSpacing: float = 2.9, overwriteConfig: bool = False) -> None:
        ''' Constructor for the PLANTS object.
        
        Parameters
        ----------
        configPath : str
            Path for the PLANTS config file.
        boxFile : str
            Path for the PLANTS box file.
        receptor : ocr.Receptor
            Receptor object.
        preparedReceptorPath : str
            Path for the prepared receptor.
        ligand : ocl.Ligand
            Ligand object.
        preparedLigandPath : str
            Path for the prepared ligand.
        plantsLog : str
            Path for the PLANTS log file.
        outputPlants : str
            Path for the PLANTS output file.
        name : str, optional
            Name for the PLANTS run, by default ""
        boxSpacing : float, optional
            Spacing for the PLANTS box, by default 0.33.
        overwriteConfig : bool, optional
            Overwrite the PLANTS config file, by default False.

        Returns
        -------
        None
        '''
        
        self.name = str(name)
        self.config = str(configPath)
        self.boxFile = str(boxFile)
        self.boxSpacing = float(boxSpacing)
        self.__bindingSite = self.__get_binding_site()

        if type(self.__bindingSite) == int:
            _ = errors.binding_site_not_found(f"The binding site was not found in the box file '{self.boxFile}'.", level="error")
            return None

        # Check if the folder where the configPath is located exists (remove the file name from the path)
        _ = ocff.safe_create_dir(os.path.dirname(self.config))

        self.bindingSiteCenter, self.bindingSiteRadius = self.__bindingSite # type: ignore
        
        # Receptor
        if type(receptor) == ocr.Receptor:
            self.inputReceptor = receptor
        else:
            errors.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'ocr.Receptor' but got {type(receptor)} instead.", level="error")
            return None
        self.inputReceptorPath = self.__parse_receptor_path(receptor)
        self.preparedReceptor = str(preparedReceptorPath)
        self.prepareReceptorCmd = [spores, "--mode", "complete", self.inputReceptorPath, self.preparedReceptor]
        
        # Ligand
        self.preparedLigand = str(preparedLigandPath)
        # Check the type of the ligand
        if type(ligand) == ocl.Ligand:
            self.inputLigand = ligand
            # Create the plantsFiles folder
            _ = ocff.safe_create_dir(os.path.join(os.path.dirname(ligand.path), "plantsFiles"))
        else:
            errors.wrong_type(f"The ligand '{ligand}' has not a supported type. Expected 'ocl.Ligand' but got {type(ligand)} instead.", level="error")
            return None

        self.inputLigandPath = self.__parse_ligand_path(ligand)
        self.prepareLigandCmd = [spores, "--mode", "complete", self.inputLigandPath, self.preparedLigand]
        
        # Plants
        self.plantsLog = str(plantsLog)
        self.outputPlants = str(outputPlants)
        self.plantsCmd = [plants, "--mode", "screen", self.config]
        
        # Check if config file exists to avoid useless processing
        if not os.path.isfile(self.config) or overwriteConfig:
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

        return get_binding_site(self.boxFile, self.boxSpacing)

    def __parse_receptor_path(self, receptor: ocr.Receptor, forceMol2: bool = False):
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
        if type(receptor) == ocr.Receptor:
            # If the flag to force the use of mol2 file as input is True
            if forceMol2:
                # If receptor has a mol2Path
                if receptor.mol2Path:
                    return receptor.mol2Path
                # Try to generate it
                else:
                    mol2Path = f"{os.path.splitext(receptor.path)[0]}.mol2"
                    # Create the mol2Path
                    ocprint.print_warning(f"No mol2 file for '{receptor.path}' trying to generate in '{mol2Path}'.")
                    # Convert the molecule
                    _ = occonversion.convertMols(receptor.path, mol2Path)
                    # Check if it is generated
                    if os.path.isfile(mol2Path):
                        # Set the mol2path in the receptor object
                        receptor.mol2Path = mol2Path
                        return receptor.mol2Path
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
        elif type(receptor) == str:
            # Since is a string, check if the file exists
            if os.path.isfile(receptor): # type: ignore
                # Exists! Return it!
                return receptor
            else:
                _ = errors.file_do_not_exist(message=f"The receptor '{receptor}' has not a valid path.", level="error")
                return ""

        _ = errors.wrong_type(message=f"The receptor '{receptor}' has not a supported type. Expected 'string' or 'ocr.Receptor' but got {type(receptor)} instead.", level="error")
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
        
        _ = errors.wrong_type(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.", level="error")
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

        return write_config_file(self.config, self.preparedReceptor, self.preparedLigand, self.outputPlants, self.bindingSiteCenter[0], self.bindingSiteCenter[1], self.bindingSiteCenter[2], self.bindingSiteRadius)

    def read_log(self) -> Union[Dict[str, List[Union[str, float]]], int]:
        '''Read the PLANTS log path, returning a pd.dataframe with data from complexes.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, List[str | float]] | int
            The dictionary with the data from complexes or the error code.
        '''

        return read_log(self.plantsLog)

    def run_plants(self, overwrite: bool =False) -> Union[Tuple[int, str], int]:
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
        runfolder = f"{self.outputPlants}/run"

        # If overwrite is set
        if overwrite:
            # Check if there is an output
            if os.path.isdir(runfolder):
                # Remove it
                shutil.rmtree(runfolder)
        # Check if there is an output
        elif os.path.isdir(runfolder):
            # Check if the dir is empty or no output file has been generated (the double of the number of cluster structures, being 2 for each structure)
            if len(os.listdir(runfolder)) == 0 or (len(glob(f"{runfolder}/{self.inputLigand.name}*.mol2")) < plants_cluster_structures * 2): # type: ignore
                # Remove it
                os.rmdir(runfolder)

        # Print verboosity
        ocprint.printv(f"Running PLANTS using the '{self.config}' configurations.")
        # Cd to tmpDir (because PLANTS keeps spamming annoying files)
        os.chdir(tmpDir)
        # Run plants
        output = ocrun.run(self.plantsCmd, logFile=self.plantsLog)
        # Check if there is a PLANTS-*.pid file
        for pidFile in glob(f"{tmpDir}/PLANTS-*.pid"):
            # This try is to avoid errors when the file does not exist
            try:
                # Remove it
                os.remove(pidFile)
            except:
                pass
        # Check if there is a *bad*.mol2 file
        for badFile in glob(f"{tmpDir}/*bad.mol2"):
            # This try is to avoid errors when the file does not exist
            try:
                # Remove it
                os.remove(badFile)
            except:
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

        # Print verboosity
        ocprint.printv(f"Running '{spores}' for '{self.inputLigandPath}'.")

        return ocrun.run(self.prepareLigandCmd, logFile=logFile)

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

        # Print verboosity
        ocprint.printv(f"Running '{spores}' for '{self.inputReceptorPath}'.")
        return ocrun.run(self.prepareReceptorCmd, logFile=logFile)

    def run_rescore(self, posePath: str, outPath: str, logFile: str = "", sanitize: bool = True, overwrite: bool = False) -> None:
        '''Run smina to rescore the ligand.

        Parameters
        ----------
        outPath : str
            Path to the output folder.
        posePath : str
            Path to the pose file.
        logFile : str
            Path to the logFile. If empty, suppress the output.
        sanitize : bool, optional
            If True, sanitize the ligand. Default is True.
        overwrite : bool, optional
            If True, overwrite the logFile. Default is False.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        # For each scoring function
        for scoring_function in vina_scoring_functions:
            # If it is not the one used to find the pose
            if scoring_function != vina_scoring:
                # Run vina to rescore
                _ = run_rescore(self.preparedLigand, posePath, outPath, scoring_function, logFile = logFile, boxSpacing = self.boxSpacing, sanitize = sanitize, overwrite = overwrite)

        return None
    
    """

    def get_rescore_log_paths(self, outPath: str) -> List[str]:
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

        return [f for f in glob(f"{outPath}/*_split_*.log") if os.path.isfile(f)]
    
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

    def read_rescore_logs(self, outPath: str, onlyBest: bool = True) -> Dict[str, List[Union[str, float]]]:
        ''' Reads the data from the rescore log files.

        Parameters
        ----------
        outPath : str
            Path to the output folder where the rescoring logs are located.
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default True.

        Returns
        -------
        Dict[str, List[Union[str, float]]]
            A dictionary with the data from the rescore log files.
        '''

        # Get the rescore log paths
        rescoreLogPaths = self.get_rescore_log_paths(outPath)

        # Create the dictionary
        rescoreLogData = {}

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
    
    """
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
        print(f"PLANTS execution log path:   '{self.plantsLog if self.plantsLog else '-' }'")
        print(f"PLANTS output path:          '{self.outputPlants if self.outputPlants else '-' }'")
        print(f"PLANTS command:              '{' '.join(self.plantsCmd) if self.plantsCmd else '-' }'")
        return None

# Functions
###############################################################################
## Private ##

## Public ##
def box_to_plants(boxFile: str, confFile: str, receptor: str, ligand: str, outputPlants: str, center: Union[float, None] = None, bindingSiteRadius: Union[float, None] = None, spacing: float = 2.9) -> int:
    '''Convert a box (DUDE like format) to PLANTS input.

    Parameters
    ----------
    boxFile : str
        The path to the box file.
    confFile : str
        The path to the PLANTS configuration file.
    receptor : str
        The path to the receptor file.
    ligand : str
        The path to the ligand file.
    outputPlants : str
        The path to the PLANTS output directory.
    center : float, optional
        The center of the box. Default is None and it will be calculated.
    bindingSiteRadius : float, optional
        The radius of the box. Default is None and it will be calculated.
    spacing : float, optional
        The spacing between the grid points. Default is 2.9.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    ocprint.printv(f"Converting the box file '{boxFile}' to PLANTS conf file as '{confFile}' file.")

    # Check if the center and the radius are given
    if center is None or bindingSiteRadius is None:
        # Calculate the center and the radius
        bindingSite = get_binding_site(boxFile, spacing = spacing)
        # Check if the binding site is int
        if isinstance(bindingSite, int):
            # Return the error code
            return bindingSite

        # Get the center and the binding site center
        center, bindingSiteRadius = bindingSite # type: ignore
    # Write the file
    return write_config_file(confFile, receptor, ligand, outputPlants, center[0], center[1], center[2], bindingSiteRadius) # type: ignore

def run_prepare_ligand(inputLigandPath: str, outputLigand: str, logFile: str = "") -> Union[Tuple[int, str], int]:
    ''' Run SPORES for ligand.

    Parameters
    ----------
    inputLigandPath : str
        The path to the input ligand.
    outputLigand : str
        The path to the output ligand.
    logFile : str, optional
        The path for the log file. Default is "".

    Returns
    -------
    Tuple[int, str] | int
        The exit code of the command (based on the Error.py code table) and the stderr if applied.

    Raises
    ------
    None
    '''

    # Create the command list
    cmd = [spores, "--mode", "complete", inputLigandPath, outputLigand]
    # Print verboosity
    ocprint.printv(f"Running '{spores}' for '{inputLigandPath}'.")
    # Run the command
    return ocrun.run(cmd, logFile=logFile)

def run_prepare_receptor(inputReceptorPath: str, outputReceptor: str, logFile: str = "") -> Union[Tuple[int, str], int]:
    ''' Run SPORES for receptor.

    Parameters
    ----------
    inputReceptorPath : str
        The path to the input receptor.
    outputReceptor : str
        The path to the output receptor.
    logFile : str, optional
        The path for the log file. Default is "".

    Returns
    -------
    Tuple[int, str] | int
        The exit code of the command (based on the Error.py code table) and the stderr if applied.

    Raises
    ------
    None
    '''
    # Create the command list
    cmd = [spores, "--mode", "complete", inputReceptorPath, outputReceptor]
    # Print verboosity
    ocprint.printv(f"Running '{spores}' for '{inputReceptorPath}'.")
    # Run the command
    return ocrun.run(cmd, logFile=logFile)

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

    Raises
    ------
    None
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
    cmd = [plants, "--mode", "screen", confFile]
    # Print verboosity
    ocprint.printv(f"Running PLANTS using the '{confFile}' configurations.")
    # Run the command
    return ocrun.run(cmd, logFile = logFile)

def run_rescore(ligands: Union[List[str], str], posePath: str, outpath: str, scoring_function: str, logFile: str = "", boxSpacing: float = 2.9, sanitize: bool = True, overwrite: bool = False) -> None:
    '''Run PLANTS to rescore the ligand.

    Parameters
    ----------
    ligands : Union[List[str], str]
        The path to a List of ligand files or the ligand file.
    posePath : str
        The path to the pose file.
    outpath : str
        The path to the output file.
    scoring_function : str
        The scoring function to use.
    logFile : str
        The path to the log file. If empty, suppress the output.
    boxSpacing : float, optional
        The spacing to be used to expand the box. Default is 2.9.
    sanitize : bool, optional
        If True, sanitize the ligand. Default is True.
    overwrite : bool, optional
        If True, overwrite the logFile. Default is False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Check if the ligands is a string
    if isinstance(ligands, str):
        # Convert to list
        ligands = [ligands]

    # For each ligand
    for ligand in ligands:
        # Get the ligand name
        ligandName = os.path.splitext(os.path.basename(ligand))[0]

        # Get the conf file name
        confFile = f"{outpath}/{ligandName}_{scoring_function}.conf"

        # Get the protein file name
        proteinFile = f"{outpath}/{ligandName}_protein.mol2"

        # Get the centroid of the ligand
        bindingSiteCenterX, bindingSiteCenterY, bindingSiteCenterZ = ocl.get_centroid(ligand, sanitize = sanitize)

        # Get boxSpacing times the gyration radius of the ligand
        bindingSiteRadius = ocl.findRadiusOfGyration(ligand) * boxSpacing # type: ignore

        # Create the conf file (yes... again...)
        _ = write_config_file(confFile, proteinFile, ligand, outpath, bindingSiteCenterX, bindingSiteCenterY, bindingSiteCenterZ, bindingSiteRadius, scoringFunction = scoring_function)
    
        # Create the command list
        cmd = [plants, "--mode", "rescore", confFile]


        # Run the command
        _ = ocrun.run(cmd, logFile = logFile)

        # Print verboosity
        ocprint.printv(f"Running PLANTS using the '{confFile}' configurations and scoring function '{scoring_function}'.")
    
    # Think about how can this be done to deal with multiple runs
    return None

def write_config_file(confFile: str, preparedReceptor: str, preparedLigand: str, outputPlants: str, bindingSiteCenterX: float, bindingSiteCenterY: float, bindingSiteCenterZ: float, bindingSiteRadius: float, scoringFunction: str = "chemplp", rescoringMode: bool = False) -> int:
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
    rescoringMode : bool, optional
        If True, the config file will be written for rescoring. Default is False.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    if rescoringMode:
        try:
            with open(confFile, 'w') as f:
                #f.write("# scoring function and search settings\n")
                f.write(f"scoring_function {scoringFunction}\n")
                #f.write("# input\n")
                f.write(f"protein_file {preparedReceptor}\n")
                f.write(f"ligand_file {preparedLigand}\n")
                #f.write("# output\n")
                f.write(f"keep_original_mol2_description 0\n") # important to avoid problems in output generation
                f.write(f"output_dir {outputPlants}/run\n")
                #f.write(f"# Rescoring mode parameter\n")
                f.write(f"rescoring_mode simplex\n")
        except Exception as e:
            return errors.write_file(f"Problems while writing the file {confFile}: {e}")
    else:
        try:
            with open(confFile, 'w') as f:
                #f.write("# scoring function and search settings\n")
                f.write(f"scoring_function {scoringFunction}\n")
                f.write(f"search_speed {plants_search_speed}\n")
                #f.write("# input\n")
                f.write(f"protein_file {preparedReceptor}\n")
                f.write(f"ligand_file {preparedLigand}\n")
                #f.write("# output\n")
                f.write(f"keep_original_mol2_description 0\n") # important to avoid problems in output generation
                f.write(f"output_dir {outputPlants}/run\n")
                #f.write("# write single mol2 files (e.g. for RMSD calculation)\n")
                f.write("write_multi_mol2 0\n")
                #f.write("# binding site definition\n")
                f.write(f"bindingsite_center {bindingSiteCenterX} {bindingSiteCenterY} {bindingSiteCenterZ}\n")
                f.write(f"bindingsite_radius {round(bindingSiteRadius, 3)}\n")
                #f.write("# cluster algorithm\n")
                f.write(f"cluster_structures {plants_cluster_structures}\n")
                f.write(f"cluster_rmsd {plants_cluster_rmsd}")
        except Exception as e:
            return errors.write_file(f"Problems while writing the file {confFile}: {e}")

    return errors.ok()

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

    Raises
    ------
    None
    '''

    ocprint.printv(f"Parsing '{boxFile}' to binding center data.")
    # Test if the file boxFile exists
    if not os.path.exists(boxFile):
        return errors.file_do_not_exist(message=f"The box file in the path {boxFile} does not exists! Please ensure that the box file exists and the path is correct.", level="error")

    # Dict to hold the center data
    center: Dict[str, Union[float, None]] = {
        'x': None,
        'y': None,
        'z': None
    }

    # Dict to hold max and min x,y,z (set all as None)
    positions: Dict[str, Union[float, None]] = {
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
        return errors.read_file(message=f"Found a problem while reading the box file: {e}", level="error")
        
    # Find which is the biggest value in each coordinate
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
        The path to the box file. If empty, it will try to look for a p2rank dir inside <path>.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Parameterize the PLANTS and p2rank paths
    plantsPath = f"{path}/plantsFiles"
    # Check if boxPath is an empty string
    if boxPath == "":
      # Set is as the path + p2rank
      boxPath = f"{path}/p2rank"
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

def read_log(path: str) -> Dict[str, List[Union[str, float]]]:
    '''Read the PLANTS log path, returning a pd.dataframe with data from complexes.

    Parameters
    ----------
    path : str
        The path to the PLANTS log file.
        
    Returns
    -------
    Dict[str, List[str | float]]
        A dictionary with the data from the PLANTS log file.

    Raises
    ------
    None
    '''
    
    # Check if file exists
    if os.path.isfile(path):
        try:
            # Read the csv
            df = vaex.read_csv(path)

            # Check if df is empty or malformed
            if df is None or df.shape[0] == 0 or df.shape[1] == 0: # type: ignore
                # Return the dict filled with np.NaN
                return {
                    "PLANTS_TOTAL_SCORE": [np.NaN],
                    "PLANTS_SCORE_RB_PEN": [np.NaN],
                    "PLANTS_SCORE_NORM_HEVATOMS": [np.NaN],
                    "PLANTS_SCORE_NORM_CRT_HEVATOMS": [np.NaN], 
                    "PLANTS_SCORE_NORM_WEIGHT": [np.NaN],
                    "PLANTS_SCORE_NORM_CRT_WEIGHT": [np.NaN],
                    "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": [np.NaN],
                }
            else:
                # Return the built the dictionary
                return {
                    "PLANTS_TOTAL_SCORE": [df.TOTAL_SCORE[:1].values[0]], # type: ignore
                    "PLANTS_SCORE_RB_PEN": [df.SCORE_RB_PEN[:1].values[0]], # type: ignore
                    "PLANTS_SCORE_NORM_HEVATOMS": [df.SCORE_NORM_HEVATOMS[:1].values[0]], # type: ignore
                    "PLANTS_SCORE_NORM_CRT_HEVATOMS": [df.SCORE_NORM_CRT_HEVATOMS[:1].values[0]], # type: ignore
                    "PLANTS_SCORE_NORM_WEIGHT": [df.SCORE_NORM_WEIGHT[:1].values[0]], # type: ignore
                    "PLANTS_SCORE_NORM_CRT_WEIGHT": [df.SCORE_NORM_CRT_WEIGHT[:1].values[0]], # type: ignore
                    "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": [df.SCORE_RB_PEN_NORM_CRT_HEVATOMS[:1].values[0]], # type: ignore
                }
        except Exception as e:
            ocprint.print_error(f"Problems while reading file '{path}'. Error: {e}")
            ocprint.print_error_log(f"Problems while reading file '{path}'. Error: {e}", f"{logdir}/PLANTS_read_log_ERROR.log")

    # Throw an error
    _ = errors.file_do_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.")

    # Return a dict with a NaN value
    return {
               "PLANTS_TOTAL_SCORE": [np.NaN],
               "PLANTS_SCORE_RB_PEN": [np.NaN],
               "PLANTS_SCORE_NORM_HEVATOMS": [np.NaN],
               "PLANTS_SCORE_NORM_CRT_HEVATOMS": [np.NaN],
               "PLANTS_SCORE_NORM_WEIGHT": [np.NaN],
               "PLANTS_SCORE_NORM_CRT_WEIGHT": [np.NaN],
               "PLANTS_SCORE_RB_PEN_NORM_CRT_HEVATOMS": [np.NaN],
           }

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

    Raises
    ------
    None
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
                                return errors.wrong_type(f"The digest file '{digestPath}' is not valid.", "error")
                    except Exception as e:
                        return errors.file_do_not_exist(f"Could not read the digest file '{digestPath}'.", "error")
            else:
                # Since it does not exists, create it
                digest = ocff.empty_docking_digest(digestPath, overwrite)

            # Read the docking object log to generate the docking digest
            dockingDigest = read_log(logPath)

            # Check if the digest variable is fine
            if not isinstance(digest, dict):
                return errors.wrong_type(f"The docking digest file '{digestPath}' is not valid.", "error")
            
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
                    return errors.write_file(f"Could not write the digest file '{digestPath}'.", "error")

            return errors.ok()
        return errors.unsupported_extension(f"The provided extension '{digestFormat}' is not supported.", "error")
    
    return errors.file_exists(f"The file '{digestPath}' already exists. If you want to overwrite it yse the overwrite flag.", "warn")

# Aliases
###############################################################################
run_docking = run_plants
