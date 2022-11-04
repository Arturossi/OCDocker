#!/usr/lib/python3

# Imports
###############################################################################
import os
import shutil
from typing import List, Tuple

import pandas as pd

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
from OCDocker.Initialise import *
import OCDocker.Toolbox as octools

from Bio.PDB import *
from glob import glob

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
    """PLANTS object with methods for easy run."""
    def __init__(self, configPath: str, boxFile: str, receptor: ocr.Receptor, preparedReceptorPath: str, ligand: ocl.Ligand, preparedLigandPath: str, plantsLog: str, outputPlants: str, name: str = "", boxSpacing: float = 0.33) -> None:
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
            Spacing for the PLANTS box, by default 0.33

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
        self.boxSpacing = float(boxSpacing)
        self.bindingSiteCenter, self.bindingSiteRadius = self.__get_binding_site()
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
        # Plants
        self.plantsLog = str(plantsLog)
        self.outputPlants = str(outputPlants)
        self.plantsCmd = self.__plants_cmd()
        # Check if config file exists to avoid useless processing
        if not os.path.isfile(self.config):
            # Create the box
            self.write_config_file()

    ## Private ##
    def __get_binding_site(self) -> Tuple[Tuple[float, float, float], float]:
        '''Get the binding site from a box file.

        Parameters
        ----------
        None

        Returns
        -------
        Tuple[Tuple[float, float, float], float]
            Tuple with the center and radius of the binding site.

        Raises
        ------
        None
        '''
        return get_binding_site(self.boxFile, self.boxSpacing)

    def __parse_receptor(self, receptor: ocr.Receptor) -> ocr.Receptor:
        '''Parse the receptor as input, handling its type.

        Parameters
        ----------
        receptor : ocr.Receptor
            The path for the receptor or its receptor object.

        Returns
        -------
        ocr.Receptor
            The ocr.Receptor object.

        Raises
        ------
        None
        '''

        # Check the type of the receptor
        if type(receptor) == ocr.Receptor:
            octools.printv(f"The receptor '{receptor}' has been loaded.")
            return receptor

        octools.print_warning(f"The receptor '{receptor}' is not the type 'ocr.Receptor'. It is STRONGLY recomended that you provide an 'ocr.Receptor' object.")
        return None

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

        Raises
        ------
        None
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
                    octools.print_warning(f"No mol2 file for '{receptor.path}' trying to generate in '{mol2Path}'.")
                    # Convert the molecule
                    _ = octools.convertMols(receptor.path, mol2Path)
                    # Check if it is generated
                    if os.path.isfile(mol2Path):
                        # Set the mol2path in the receptor object
                        receptor.mol2Path = mol2Path
                        return receptor.mol2Path
                    else:
                        _ = octools.print_error(f"The mol2 file could not be generated for '{receptor.path}'.")
                        return None
            else:
                # Check if the object has a valid path
                if receptor.path:
                    return receptor.path
                else:
                    _ = octools.print_error(f"Invalid receptor path for the following path: '{receptor.path}'.")
                    return None
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

    def __parse_ligand(self, ligand: ocl.Ligand) -> ocl.Ligand:
        '''Parse the ligand as input, handling its type.

        Parameters
        ----------
        ligand : ocl.Ligand
            The path for the ligand or its ligand object.

        Returns
        -------
        ocl.Ligand
            The ocl.Ligand object.

        Raises
        ------
        None
        '''

        # Check the type of the ligand
        if type(ligand) == ocl.Ligand:
            octools.printv(f"The ligand '{ligand}' has been loaded.")
            return ligand

        octools.print_warning(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.")
        return None

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

        Raises
        ------
        None
        '''

        # Check the type of ligand variable
        if type(ligand) == ocl.Ligand:
            return ligand.path
        
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
            The path for the ligand.

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
        octools.convertMols(ligandPath, outputLigandPath, logFile = self.convert2mol2log)

        return outputLigandPath

    def __plants_cmd(self) -> List[str]:
        '''Generate the vina command.

        Parameters
        ----------
        None
        
        Returns
        -------
        List[str]
            The vina command.

        Raises
        ------
        None
        '''

        cmd = [plants, "--mode", "screen", self.config]
        return cmd

    def __prepare_ligand_cmd(self) -> List[str]:
        '''Generate the prepare ligand command.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            The prepare ligand command.

        Raises
        ------
        None
        '''

        cmd = [spores, "--mode", "complete", self.inputLigandPath, self.preparedLigand]
        return cmd

    def __prepare_receptor_cmd(self):
        '''Generate the prepare receptor command.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            The prepare receptor command.

        Raises
        ------
        None
        '''

        cmd = [spores, "--mode", "complete", self.inputReceptorPath, self.preparedReceptor]
        return cmd

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
        
        Raises
        ------
        None
        '''

        return write_config_file(self.config, self.preparedReceptor, self.preparedLigand, self.outputPlants, self.bindingSiteCenter[0], self.bindingSiteCenter[1], self.bindingSiteCenter[2], self.bindingSiteRadius)

    def read_plants_log(self) -> pd.DataFrame:
        '''Read the PLANTS log path, returning a pd.dataframe with data from complexes.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            The dataframe with the data from PLANTS.

        Raises
        ------
        None
        '''

        return read_plants_log(self.plantsLog)

    def run_plants(self, overwrite: bool =False) -> int:
        '''Run plants.

        Parameters
        ----------
        overwrite : bool, optional
            If True, overwrite the output file. Default is False.

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).

        Raises
        ------
        None
        '''

        # If overwrite is set
        if overwrite:
            # Check if there is an output
            if os.path.isdir(self.outputPlants):
                # Remove it
                shutil.rmtree(self.outputPlants)
        # Check if there is an output
        elif os.path.isdir(self.outputPlants):
            # Check if the dir is empty or no output file has been generated (the double of the number of cluster structures, being 2 for each structure)
            if len(os.listdir(self.outputPlants)) == 0 or (len(glob(f"{self.outputPlants}/{self.inputLigand.name}*.mol2")) < plants_cluster_structures * 2):
                # Remove it
                os.rmdir(self.outputPlants)

        # Print verboosity
        octools.printv(f"Running PLANTS using the '{self.config}' configurations.")
        # Cd to tmpDir (because PLANTS keeps spamming annoying files)
        os.chdir(tmpDir)
        # Run plants
        output = octools.run(self.plantsCmd, logFile=self.plantsLog)
        # Check if there is a PLANTS-*.pid file
        if len(glob(f"{tmpDir}/PLANTS-*.pid")) > 0:
            # Remove it
            _ = octools.run(["rm", f"{tmpDir}/PLANTS-*.pid"])
        # Check if there is a *bad*.mol2 file
        if len(glob(f"{tmpDir}/*bad.mol2")) > 0:
            # Remove it
            _ = octools.run(["rm", f"{tmpDir}/*bad.mol2"])
        return output

    def run_prepare_ligand(self, logFile: str = "") -> int:
        '''Run SPORES for ligand.

        Parameters
        ----------
        logFile : str, optional
            The path for the log file. Default is "".

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).

        Raises
        ------
        None
        '''

        # Print verboosity
        octools.printv(f"Running '{spores}' for '{self.inputLigandPath}'.")
        return octools.run(self.prepareLigandCmd, logFile=logFile)

    def run_prepare_receptor(self, logFile: str = "") -> int:
        '''Run SPORES for receptor.

        Parameters
        ----------
        logFile : str, optional
            The path for the log file. Default is "".

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).

        Raises
        ------
        None
        '''

        # Print verboosity
        octools.printv(f"Running '{spores}' for '{self.inputReceptorPath}'.")
        return octools.run(self.prepareReceptorCmd, logFile=logFile)

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
        print(f"Conversion to mol2 log path: '{self.convert2mol2log if self.convert2mol2log else '-' }'")
        print(f"PLANTS execution log path:   '{self.plantsLog if self.plantsLog else '-' }'")
        print(f"PLANTS output path:          '{self.outputPlants if self.outputPlants else '-' }'")
        print(f"PLANTS command:              '{' '.join(self.plantsCmd) if self.plantsCmd else '-' }'")
        return None

# Functions
###############################################################################
## Private ##

## Public ##
def box_to_plants(boxFile: str, confFile: str, receptor: str, ligand: str, outputPlants: str, spacing: float = 0.33) -> int:
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
    spacing : float, optional
        The spacing between the grid points. Default is 0.33.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    octools.printv(f"Converting the box file '{boxFile}' to PLANTS conf file as '{confFile}' file.")
    # Get the center and the binding site center
    center, bindingSiteRadius = get_binding_site(boxFile, spacing = spacing)
    # Write the file
    return write_config_file(confFile, receptor, ligand, outputPlants, center[0], center[1], center[2], bindingSiteRadius)

def run_prepare_ligand(inputLigandPath: str, outputLigand: str, logFile: str = "") -> int:
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
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Create the command list
    cmd = [spores, "--mode", "complete", inputLigandPath, outputLigand]
    # Print verboosity
    octools.printv(f"Running '{spores}' for '{inputLigandPath}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_prepare_receptor(inputReceptorPath: str, outputReceptor: str, logFile: str = "") -> int:
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
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''
    # Create the command list
    cmd = [spores, "--mode", "complete", inputReceptorPath, outputReceptor]
    # Print verboosity
    octools.printv(f"Running '{spores}' for '{inputReceptorPath}'.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_plants(confFile: str, outputPlants: str, overwrite: bool = False, logFile: str = "") -> int:
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
    int
        The exit code of the command (based on the Error.py code table).

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
    octools.printv(f"Running PLANTS using the '{confFile}' configurations.")
    # Run the command
    return octools.run(cmd, logFile=logFile)

def write_config_file(confFile: str, preparedReceptor: str, preparedLigand: str, outputPlants: str, bindingSiteCenterX: int, bindingSiteCenterY: int, bindingSiteCenterZ: int, bindingSiteRadius: int) -> int:
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
    bindingSiteCenterX : int
        The X coordinate of the binding site center.
    bindingSiteCenterY : int
        The Y coordinate of the binding site center.
    bindingSiteCenterZ : int
        The Z coordinate of the binding site center.
    bindingSiteRadius : int
        The radius of the binding site.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    try:
        with open(confFile, "w") as f:
            #f.write("# scoring function and search settings\n")
            f.write("scoring_function chemplp\n")
            f.write(f"search_speed {plants_search_speed}\n")
            #f.write("# input\n")
            f.write(f"protein_file {preparedReceptor}\n")
            f.write(f"ligand_file {preparedLigand}\n")
            #f.write("# output\n")
            f.write(f"keep_original_mol2_description 0\n") # important to avoid problems in output generation
            f.write(f"output_dir {outputPlants}\n")
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

def get_binding_site(boxFile: str, spacing: float = 0.33) -> Tuple[Tuple[int, int, int], int]:
    '''Get the binding site from a box file.

    Parameters
    ----------
    boxFile : str
        The path to the box file.
    spacing : float, optional
        The spacing between the box and the binding site. Default is 0.33.
    
    Returns
    -------
    Tuple[Tuple[int, int, int], int]
        The center of the binding site and the radius of the binding site.

    Raises
    ------
    None
    '''

    octools.printv(f"Parsing '{boxFile}' to binding center data.")
    # Test if the file boxFile exists
    if not os.path.exists(boxFile):
        return errors.file_do_not_exist(message=f"The box file in the path {boxFile} does not exists! Please ensure that the file exsits and the path is correct. If you have no box file, try to run the function 'runprank' from the 'runprank' library to create it before calling this function or creating a PLANTS class object.", level="error")
    # Dict to hold all the data
    center = {
        'x': None,
        'y': None,
        'z': None
    }
    # Dict to hold max and min x,y,z (set all as None)
    positions = {
        'max_x': None,
        'max_y': None,
        'max_z': None,
        'min_x': None,
        'min_y': None,
        'min_z': None
        }
    try:
        # Open the box file
        with open(str(boxFile), "r") as box_file:
            # For each line in the file
            for line in box_file:
                # If it starts with REMARK
                if line.startswith("REMARK"):
                    # Split the line (using spaces as delimiters)
                    l = line.split()
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
    xMax = max(abs(center['x'] - positions['min_x']), abs(positions['max_x'] - center['x']))
    yMax = max(abs(center['y'] - positions['min_y']), abs(positions['max_y'] - center['y']))
    zMax = max(abs(center['z'] - positions['min_z']), abs(positions['max_z'] - center['z']))
    # Get the biggest value among the coordinates, divide for 2 (because its a size)
    radius = max(xMax, yMax, zMax)/2
    # Add some extra space
    radius += round(spacing * radius, 3)
    # Return the data
    return ((center['x'], center['y'], center['z']), radius)

def generate_plants_files_database(path: str, protein: str, ligand: str, spacing: float, boxPath: str = "") -> None:
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

    # Parameterize the vina and p2rank paths
    plantsPath = f"{path}/plantsFiles"
    # Check if boxPath is an empty string
    if boxPath == "":
      # Set is as the path + p2rank
      boxPath = f"{path}/p2rank"
    # Create the vina folder inside protein's directory
    _ = octools.safe_create_dir(plantsPath)
    # Find all boxes
    boxes = glob(f"{boxPath}/box*.pdb")
    # For each box
    for box in boxes:
        # Get box name
        boxName = os.path.basename(box)
        # Get box id
        boxId = os.path.splitext(boxName)[0].replace("box", "").replace(".pdb", "")
        # Parameterize the box folder
        outputPlants = f"{plantsPath}/{boxId}"
        # Create vina execution folder
        _ = octools.safe_create_dir(outputPlants)
        confPath = f"{outputPlants}/conf_plants.txt"
        # Convert the box to a conf file
        box_to_plants(box, confPath, protein, ligand, f"{outputPlants}/run", spacing = spacing)

    return None

def read_plants_log(path: str ) -> pd.DataFrame:
    '''Read the PLANTS log path, returning a pd.dataframe with data from complexes.

    Parameters
    ----------
    path : str
        The path to the PLANTS log file.
        
    Returns
    -------
    pd.DataFrame
        A pandas dataframe with the data from the PLANTS log file.

    Raises
    ------
    None
    '''
    
    # Check if file exists
    if os.path.isfile(path):
        try:
            # Read the csv
            df = pd.read_csv(path)
            # Remove EVAL and TIME columns
            df.drop("EVAL", axis=1, inplace=True)
            df.drop("TIME", axis=1, inplace=True)
            # Remove also the SCORE_NORM_CONTACT because it was being problematic
            df.drop("SCORE_NORM_CONTACT", axis=1, inplace=True)
            return df
        except Exception as e:
            octools.print_error(f"Problems while reading file '{path}'. Error: {e}")
            octools.print_error_log(f"Problems while reading file '{path}'. Error: {e}", f"{logdir}/PLANTS_read_log_ERROR.log")
    # Throw an error
    return errors.file_do_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.")
