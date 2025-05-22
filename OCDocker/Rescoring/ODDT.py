#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions to perform rescoring of docking results using the ODDT.

They are imported as:

import OCDocker.Rescoring.ODDT as ocoddt
'''

# Imports
###############################################################################
import os
import six

import oddt as od
import pandas as pd

from glob import glob
from typing import Dict, List, Tuple, Union

from oddt.scoring import scorer
from oddt.virtualscreening import virtualscreening as vs

from OCDocker.Initialise import *

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun

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

# Functions
###############################################################################
## Private ##
def __build_cmd(receptorPath: str, ligandPath: str, outputFile: str) -> Union[List[str], int]:
    '''Builds the command to run ODDT.

    Parameters
    ----------
    receptorPath : str
        The path to the receptor file.
    ligandPath : str
        The path to the ligand file.
    outputFile : str
        The path to the output file.

    Returns
    -------
    List[str] | int
        The command to run ODDT or an error code (based on the Error.py code table).
    '''

    # Check if the output file is a csv
    if not outputFile.endswith(".csv"):
        return ocerror.Error.unsupported_extension("The output file must be a csv file.", level = ocerror.ReportLevel.ERROR) # type: ignore

    # Start building the command
    cmd = [oddt, ligandPath, "-O", outputFile, "--receptor", receptorPath, "-i", "pdbqt", "-n", "1"]

    # Check if there are scoring functions to be used
    if isinstance(oddt_scoring_functions, list) and len(oddt_scoring_functions) > 0:
        # Add the scoring functions
        for score in oddt_scoring_functions:
            cmd.append("--score")
            cmd.append(score)
    else:
        ocprint.print_error("No scoring functions were provided to ODDT. Please check your configuration file.")

    return cmd

## Public ##
def get_models(outputPath: str) -> List[str]:
    '''Get the models from the output path.

    Parameters
    ----------
    outputPath : str
        The path to the output folder.

    Returns
    -------
    List[str]
        A list with the paths to the models.
    '''

    # Get the models
    models = glob(f"{outputPath}/*.pickle")

    return models

def run_oddt_from_cli(receptor: Union[ocr.Receptor, str], ligand: Union[ocl.Ligand, str], outputPath: str, overwrite: bool = False, logFile: str = "", cleanModels: bool = False) -> Union[int, Tuple[int, str]]:
    '''Run ODDT using the oddt_cli command. UNSTABLE FUNCTION DO NOT USE.

    Parameters
    ----------
    receptor : ocr.Receptor | str
        The receptor to be used in the docking.
    ligand : ocl.Ligand | str
        The ligand to be used in the docking.
    outputPath : str
        The path where the output file will be saved.
    overwrite : bool, optional
        If True, the output file will be overwritten. The default is False.
    logFile : str, optional
        The path to the log file. The default is "" (no log file).
    cleanModels : bool, optional
        If True, the models will be deleted after the rescoring. The default is False. If set to False, this can speed up the rescoring process for multiple ligands.
    
    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table).   
    '''

    # Check if the output dir exists
    if not os.path.isdir(outputPath):
        return ocerror.Error.dir_not_exist(f"The output directory '{outputPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the receptor is an ocr.Receptor object
    if isinstance(receptor, ocr.Receptor):
        # Get the receptor path
        receptorPath = receptor.path
    # Check if the receptor is a string
    elif isinstance(receptor, str):
        # Get the receptor path
        receptorPath = receptor
    else:
        return ocerror.Error.wrong_type(f"The receptor must be a string or an ocr.Receptor object. The type {type(receptor)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the ligand is an ocl.Ligand object
    if isinstance(ligand, ocl.Ligand):
        # Get the ligand path
        ligandPath = ligand.path
        # Output file name
        outputFile = f"{outputPath}/{ligand.name}.csv"
    # Check if the ligand is a string
    elif isinstance(ligand, str):
        # Get the ligand path
        ligandPath = ligand
        # Get the ligand name from the path
        ligandName = ".".join(os.path.basename(ligandPath).split(".")[:-1])
        # Output file name
        outputFile = f"{outputPath}/{ligandName}.csv"
    else:
        return ocerror.Error.wrong_type(f"The ligand must be a string or an ocl.Ligand object. The type {type(ligand)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the output file exists
    if os.path.isfile(outputFile) and not overwrite:
        return ocerror.Error.file_exists(f"The output file '{outputFile}' already exists. Please use the overwrite option if you want to overwrite it.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the receptor exists
    if not os.path.isfile(receptorPath):
        return ocerror.Error.file_not_exist(f"The receptor file '{receptorPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore

    # Check if the ligand exists
    if not os.path.isfile(ligandPath):
        return ocerror.Error.file_not_exist(f"The ligand file '{ligandPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Create the output file path
    
    # Get the command
    cmd = __build_cmd(receptorPath, ligandPath, outputFile)

    # If the command is an int, it is an error code
    if isinstance(cmd, int):
        return cmd
    
    # Run the command
    exitCode = ocrun.run(cmd, logFile = logFile, cwd = oddt_models_dir)

    # If the models should be deleted
    if cleanModels:
        # Get the models
        models = get_models(outputPath)

        # For each model
        for model in models:
            # Delete it
            ocff.safe_remove_file(model)
    
    return exitCode

def run_oddt(preparedReceptorPath: str, preparedLigandPath: Union[str, List[str]], ligandName: str, outputPath: str, returnData: bool = True, overwrite: bool = False, cleanModels: bool = False) -> Union[int, pd.DataFrame]:
    '''Run ODDT programatically.

    Parameters
    ----------
    preparedReceptorPath : str
        The receptor to be used in the rescoring.
    preparedLigandPath : str | List[str]
        The ligand to be used in the rescoring. If a list is given, the rescoring will be performed for each ligand in the list.
    ligandName : str
        The name of the ligand.
    outputPath : str
        The path where the output file will be saved.
    returnData : bool, optional
        If True, the data will be returned. The default is True.
    overwrite : bool, optional
        If True, the output file will be overwritten. The default is False.
    cleanModels : bool, optional
        If True, the models will be deleted after the rescoring. The default is False. If set to False, this can speed up the rescoring process for multiple ligands (you probably will not want to set this to True).
    
    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).   
    '''

    # Check if the output dir exists
    if not os.path.isdir(outputPath):
        # Try to create it
        try:
            _ = ocff.safe_create_dir(outputPath)
        except Exception as e:
            return ocerror.Error.dir_not_exist(f"The output directory '{outputPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore
        
    # If the ligand path is a string
    if isinstance(preparedLigandPath, str):
        # Transform it into a list
        preparedLigandPath = [preparedLigandPath]

    # Get the models (only files)
    models = [model for model in glob(f"{oddt_models_dir}/*.pickle") if os.path.isfile(model)]

    # Check if are there any model
    if len(models) <= 0:
        return ocerror.Error.missing_oddt_models("There are no models in the models folder. Please run the initialise_oddt() function (with proper arguments) to download the models.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the receptor is a string
    if not isinstance(preparedReceptorPath, str):
        return ocerror.Error.wrong_type(f"The receptor must be a string. The type {type(preparedReceptorPath)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the receptor exists
    if not os.path.isfile(preparedReceptorPath):
        return ocerror.Error.file_not_exist(f"The receptor file '{preparedReceptorPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the ligand is not a string
    if not isinstance(preparedLigandPath, list):
        return ocerror.Error.wrong_type(f"The ligand must be a string or a list. The type {type(preparedLigandPath)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Set the output file name
    outputFile = f"{outputPath}/{ligandName}.csv"

    # Check if the output file exists and if it should be overwritten
    if os.path.isfile(outputFile) and not overwrite:
        # Check if the returnData is True
        if returnData:
            try:
                # Read the output file
                return pd.read_csv(outputFile, sep = ",")
            except Exception as e:
                return ocerror.Error.corrputed_file(f"Failed to read output file '{outputFile}'.", level=ocerror.ReportLevel.ERROR) # type: ignore
        else:
            return ocerror.Error.file_exists(f"The output file '{outputFile}' already exists. Please use the overwrite option if you want to overwrite it.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Create the vs object
    pipeline = vs()

    # Load the receptor
    receptorObj = six.next(od.toolkit.readfile(preparedReceptorPath.split('.')[-1], preparedReceptorPath))

    # Check if the receptor is None
    if receptorObj is None:
        return ocerror.Error.empty(f"The rescoring of the ligand '{ligandName}' failed.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    receptorObj.protein = True

    # Find missing ligands
    missing = [ligand for ligand in preparedLigandPath if not os.path.isfile(ligand)]

    # Check if there are missing ligands
    if missing:
        return ocerror.Error.file_not_exist(f"Missing ligands: {missing}", level=ocerror.ReportLevel.ERROR)  # type: ignore

    # Check if all the ligands exist
    for ligand in preparedLigandPath:
        # Load the ligand
        pipeline.load_ligands(ligand.split('.')[-1], ligand)

    # Create a set with the scoring functions
    sf_set = {'nnscore', 'rfscore', 'plec'}

    # Check if the desired models are in the models folder
    for model in models:
        # Extract the model name and convert it to lower case
        model_name = os.path.basename(model).lower()

        # Check if the model name is in the set of scoring functions
        if any(sf in model_name for sf in sf_set):
            # Load the model
            sf = scorer.load(model)

            # Add the model to the pipeline performing the rescoring
            pipeline.score(sf, receptorObj)

    # Create the empty data list
    datas = []

    # Process the results
    for mol in pipeline.fetch():
        # Transform the results into a dict
        data = mol.data.to_dict()
        # Add the ligand name
        data["ligand_name"] = ".".join(os.path.basename(mol.title).split(".")[:-1])
        # Set the blacklist keys
        blacklist_keys = ['OpenBabel Symmetry Classes', 'MOL Chiral Flag', 'PartialCharges', 'TORSDO', 'REMARK']
        
        # For each key in the blacklist
        for b in blacklist_keys:
            # Check if the key is in the data
            if b in data:
                # Delete it
                del data[b]
            
            # Check if there is anything in the data dict
            if len(data) <= 0:
                # Show an error
                return ocerror.Error.rescoring_failed(f"The rescoring of the ligand '{ligandName}' failed.", level = ocerror.ReportLevel.ERROR) # type: ignore
        
        # Append the data to the datas list
        datas.append(data)

    # Check if datas is empty
    if len(datas) <= 0:
        return ocerror.Error.rescoring_failed(f"The rescoring of the ligand '{ligandName}' failed.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Create the dataframe
    df = pd.DataFrame(datas)

    # Set the ligand_name as the first column and remove all columns with vina in the name (maybe there is a better way to fix this)
    df = df[["ligand_name"] + [col for col in df.columns if col != "ligand_name" and "vina" not in col]]

    # Set the index to the ligand name
    df = df.set_index("ligand_name")

    # Write the output csv file
    df.to_csv(outputFile, sep = ",", index = False)

    # If the models should be deleted
    if cleanModels:
        # Get the models
        models = get_models(outputPath)

        # For each model
        for model in models:
            # Delete it
            ocff.safe_remove_file(model)
    
    # Check if the returnData is True
    if returnData:
        # Return the dataframe
        return df
    
    # Just return an ok code
    return ocerror.Error.ok() # type: ignore

def df_to_dict(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    '''Convert the data from a pandas dataframe to a dict.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be converted.

    Returns
    -------
    Dict[str, Dict[str, float]]
        The converted data.
    '''

    # Check if the data is a dataframe
    if not isinstance(data, pd.DataFrame):
        return ocerror.Error.wrong_type(f"The data must be a pandas dataframe. The type {type(data)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Convert the dataframe to dict, one row per index
    return data.to_dict(orient = "index") # type: ignore

def read_log(path: str) -> Union[pd.DataFrame, None]:
    '''Read the oddt log path, returning the data from complexes.

    Parameters
    ----------
    path : str
        The path to the oddt csv log file.

    Returns
    -------
    pd.DataFrame | None
        A pd.DataFrame with the data from the vina log file. If the file does not exist, None is returned.
    '''

    # Check if file exists
    if os.path.isfile(path):
        # Read the dataframe
        data = pd.read_csv(path, sep = ",")

        # Return the dataframe
        return data

    # Throw an error
    _ = ocerror.Error.file_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.") # type: ignore

    # Return None
    return None
