#!/usr/lib/python3

# Imports
###############################################################################
import os

import pandas as pd

from typing import List, Union
from openbabel import openbabel

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
from OCDocker.Initialise import *
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

import OCDocker.Smina as ocsmina
'''

# Classes
###############################################################################
class Smina:
    """Smina object with methods for easy run."""
    def __init__(self, configPath: str, receptor: ocr.Receptor, preparedReceptorPath: str, ligand: ocl.Ligand, preparedLigandPath: str, sminaLog: str, outputSmina: str, name: str = "") -> None:
        '''Constructor of the class Smina.

        Parameters
        ----------
        configPath : str
            Path to the configuration file.
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
            Name of the smina object, by default ""

        Returns
        -------
        None

        Raises
        ------
        None
        '''

        self.name = str(name)
        self.config = str(configPath)
        # Receptor
        self.inputReceptor = self.__parse_receptor(receptor)
        self.inputReceptorPath = self.__parse_receptor_path(receptor)
        self.preparedReceptor = str(preparedReceptorPath)
        self.prepareReceptorCmd = self.__prepare_receptor_cmd()
        # Ligand
        self.inputLigand = self.__parse_ligand(ligand)
        self.inputLigandPath = self.__parse_ligand_path(ligand)
        self.preparedLigand = str(preparedLigandPath)
        self.prepareLigandCmd = self.__prepare_ligand_cmd()
        # Vina
        self.sminaLog = str(sminaLog)
        self.outputSmina = str(outputSmina)
        self.sminaCmd = self.__smina_cmd()
        self.__gen_smina_conf()

    ## Private ##

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

    def __parse_receptor_path(self, receptor: ocr.Receptor) -> str:
        '''Parse the receptor path, handling its type.

        Parameters
        ----------
        receptor : ocr.Receptor
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
            return receptor.path

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
            The ligand path.

        Raises
        ------
        None
        '''

        # Check the type of ligand variable
        if type(ligand) == ocl.Ligand:
            return ligand.path
        elif type(ligand) == str:
            # Since is a string, check if the file exists
            if os.path.isfile(ligand):
                # Exists! Process it then!
                return self.__process_ligand(ligand)
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

        cmd = [smina, "--config", self.config, "--ligand", self.preparedLigand, "--autobox_ligand", self.preparedLigand]

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

        cmd = [obabel, self.inputLigandPath, "-O", self.preparedLigand]
        return cmd

    def __prepare_receptor_cmd(self) -> List[str]:
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

        cmd = [obabel, self.inputReceptorPath, "-xr", "-O", self.preparedReceptor]
        return cmd

    def __gen_smina_conf(self) -> int:
        '''Creates a conf file for smina.

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
        
        return gen_smina_conf(self.config, self.preparedReceptor)

    ## Public ##

    def read_smina_log(self) -> pd.DataFrame:
        '''Read the smina log path, returning a pd.dataframe with data from complexes.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            The dataframe with the data from the smina log.

        Raises
        ------
        None
        '''

        return read_smina_log(self.sminaLog)

    def run_smina(self, logFile: str = "") -> int:
        '''Run smina.

        Parameters
        ----------
        logFile : str
            The path for the log file.
        
        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).

        Raises
        ------
        None
        '''

        return octools.run(self.sminaCmd, logFile=logFile)

    def run_prepare_ligand_from_cmd(self, logFile: str = "") -> int:
        '''Run obabel convert ligand to pdbqt using the 'self.inputLigandPath' attribute. [DEPRECATED]

        Parameters
        ----------
        logFile : str
            The path for the log file.

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).

        Raises
        ------
        None
        '''

        return octools.run(self.prepareLigandCmd, logFile=logFile)

    def run_prepare_ligand(self) -> int:
        '''Run obabel convert ligand to pdbqt using the openbabel python library.

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

        return run_prepare_ligand(self.inputLigandPath, self.preparedLigand)

    def run_prepare_receptor_from_cmd(self, logFile: str = "") -> int:
        '''Run obabel convert receptor to pdbqt script using the 'self.prepareReceptorCmd' attribute. [DEPRECATED]

        Parameters
        ----------
        logFile : str
            The path for the log file.

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).

        Raises
        ------
        None
        '''

        return octools.run(self.prepareReceptorCmd, logFile=logFile)

    def run_prepare_receptor(self) -> int:
        '''Run obabel convert receptor to pdbqt using the openbabel python library.

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
def gen_smina_conf(confFile: str, receptor: str) -> int:
    '''Convert a box (DUDE like format) to vina input.

    Parameters
    ----------
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

    octools.printv(f"Creating smina conf file in the path '{confFile}'.")
    try:
        # Now open the conf file to write
        with open(confFile, 'w') as conf_file:
            conf_file.write(f"receptor = {receptor}\n\n");
            if smina_custom_scoring.lower() != "no":
                conf_file.write(f"custom_scoring = {smina_custom_scoring}\n")
            if smina_custom_atoms.lower() != "no":
                conf_file.write(f"custom_atoms = {smina_custom_atoms}\n")

            conf_file.write(f"minimize_iters = {smina_minimize_iters}\n")
            conf_file.write(f"approximation = {smina_approximation}\n")
            conf_file.write(f"factor = {smina_factor}\n")
            conf_file.write(f"force_cap = {smina_force_cap}\n")

            if smina_user_grid.lower() != "no":
                conf_file.write(f"user_grid = {smina_custom_scoring}\n")
                conf_file.write(f"user_grid_lambda = {smina_user_grid_lambda}\n")

            conf_file.write(f"energy_range = {smina_energy_range}\n")
            conf_file.write(f"exhaustiveness = {smina_exhaustiveness}\n")
            conf_file.write(f"num_modes = {smina_num_modes}\n")
    except Exception as e:
        return errors.write_file(message=f"Found a problem while opening conf file: {e}.", level="error")

    return errors.ok()

def run_prepare_ligand_from_cmd(inputLigandPath: str, preparedLigand: str, logFile: str = "") -> int:
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
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Create the command list
    cmd = [obabel, inputLigandPath, "-O", preparedLigand]

    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_prepare_ligand(inputLigandPath: str, preparedLigand: str) -> int:
    '''Run obabel convert ligand to pdbqt using the openbabel python library.

    Parameters
    ----------
    inputLigandPath : str
        The path for the input ligand.
    preparedLigand : str
        The path for the prepared ligand.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

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
        return extension
    # Discover if the output extension is pdbqt (to warn user if it is not)
    if outExtension != ".pdbqt":
        octools.print_warn(f"The output extension is not '.pdbqt', is {outExtension}. This function converts {clrs['r']}ONLY{clrs['n']} to '.pdbqt'. Please pay attention, since this might be a problem in the future for you!")
    try:
        # Create a conversor object
        obConversion = openbabel.OBConversion()
        # Set the conversion from the extension to pdbqt
        obConversion.SetInAndOutFormats(extension, "pdbqt")
        # Create an empty OBMol object
        mol = openbabel.OBMol()
        # Load the input file to the prebiusly loaded OBMol object
        obConversion.ReadFile(mol, inputLigandPath)
        # Write the mol object to the output performing the conversion
        obConversion.WriteFile(mol, preparedLigand)
    except Exception as e:
        return errors.subprocess(message=f"Error while running ligand conversion using obabel python lib. Error: {e}", level="error")
    return errors.ok()

def run_prepare_receptor_from_cmd(inputReceptorPath: str, outputReceptor: str, logFile: str = "") -> int:
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
    int
        The exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Create the command list
    cmd = [obabel, inputReceptorPath, "-xr", "-O", outputReceptor]
    # Run the command
    return octools.run(cmd, logFile=logFile)

def run_prepare_receptor(inputReceptorPath: str, preparedReceptor: str) -> int:
    '''Run obabel convert receptor to pdbqt using the openbabel python library.

    Parameters
    ----------
    inputReceptorPath : str
        The path for the input receptor.
    preparedReceptor : str
        The path for the prepared receptor.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).

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
        return extension
    # Discover if the output extension is pdbqt (to warn user if it is not)
    if outExtension != ".pdbqt":
        octools.print_warn(f"The output extension is not '.pdbqt', is {outExtension}. This function converts {clrs['r']}ONLY{clrs['n']} to '.pdbqt'. Please pay attention, since this might be a problem in the future for you!")
    # Try to convert (if fails, throw exception for subprocess failing)
    try:
        # Create a conversor object
        obConversion = openbabel.OBConversion()
        # Set the conversion from the extension to pdbqt
        obConversion.SetInAndOutFormats(extension, "pdbqt")
        # Create an empty OBMol object
        mol = openbabel.OBMol()
        # Load the input file to the previously loaded OBMol object
        obConversion.ReadFile(mol, inputReceptorPath)
        # Write the mol object to the output performing the conversion
        obConversion.WriteFile(mol, preparedReceptor)
    except Exception as e:
        return errors.subprocess(message=f"Error while running receptor conversion using obabel python lib. Error: {e}", level="error")
    return errors.ok()

def run_smina(config: str, preparedLigand: str, outputSmina: str, sminaLog: str, logPath: str) -> int:
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
    int
        The exit code of the command (based on the Error.py code table).

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

def read_smina_log(path: str) -> Union[pd.DataFrame, int]:
    '''Read the smina log path, returning a pd.dataframe with data from complexes.

    Parameters
    ----------
    path : str
        The path for the smina log file.

    Returns
    -------
    pd.DataFrame | int
        The dataframe with the data from the smina log file or the exit code of the command (based on the Error.py code table).

    Raises
    ------
    None
    '''

    # Check if file exists
    if os.path.isfile(path):
        # Open the log file
        with open(path, "r") as f:
            # Read ALL the lines in file (there should not be lots of lines, so no problem)
            lines = f.readlines()
        # Create a dataframe to store the info
        df = pd.DataFrame(columns=["mode", "affinity", "rmsd_lb_best_mode", "rmsd_ub_best_mode"])
        # For each line from the end to the beggining (reverse iteration since the intresting data is in the end of the file)
        for i in range(len(lines)-1, -1, -1):
            # If the line starts with a -
            if lines[i].startswith("-"):
                # Stop iteration, because it does not contain useful information and neither the upper lines do
                break
            # If useless information is in our way ignore it
            if "Writing output ... done." in lines[i]:
                continue
            try:
                # Add the reversed list to the end of
                df.loc[len(df), df.columns] = lines[i].strip().split()
            except Exception as e:
                octools.print_error(f"Problems while reading file '{path}'. Error: {e}")
                octools.print_error_log(f"Problems while reading file '{path}'. Error: {e}", f"{logdir}/smina_read_log_ERROR.log")
        # Return the df reversing the order and reseting the index
        return df.reindex(index=df.index[::-1]).reset_index(drop=True)
    # Throw an error
    return errors.file_do_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.")
