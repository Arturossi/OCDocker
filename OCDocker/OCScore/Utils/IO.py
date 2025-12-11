#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage I/O operations in OCDocker in the context of scoring 
functions.

They are imported as:

import OCDocker.OCScore.Utils.IO as ocscoreio
'''

# Imports
###############################################################################

import joblib
import os
import pandas as pd
import pickle
import numpy as np

from typing import Any, Optional, Union

import OCDocker.Error as ocerror



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

# Methods
###############################################################################


def load_object(file_name : str, serialization_method : str = "auto") -> Any:
    ''' Load an object from a file using pickle, joblib, or torch.

    Parameters
    ----------
    file_name : str
        The name of the file from which to load the object.
    serialization_method : str
        The serialization method used to save the object. Options are:
        - "auto": Automatically detect from file extension (.pt/.pth -> torch, .pkl -> joblib/pickle)
        - "joblib": Use joblib to load
        - "pickle": Use pickle to load
        - "torch": Use torch.load to load (for PyTorch models)

    Returns
    -------
    Any
        The loaded object.
    
    Raises
    ------
    ValueError
        If the serialization method is not recognized.
    '''

    # Auto-detect format from file extension if "auto" is specified
    if serialization_method == "auto":
        if file_name.endswith('.pt') or file_name.endswith('.pth'):
            serialization_method = "torch"
        elif file_name.endswith('.pkl'):
            serialization_method = "joblib"  # Default to joblib for .pkl
        else:
            # Default to joblib for unknown extensions
            serialization_method = "joblib"

    # Load based on method
    if serialization_method == "torch":
        try:
            import torch
            # Explicitly set weights_only=False to suppress FutureWarning
            # This is safe for trusted model files
            return torch.load(file_name, map_location='cpu', weights_only=False)
        except ImportError:
            ocerror.Error.value_error("PyTorch is not installed. Cannot load .pt/.pth files.") # type: ignore
            raise ValueError("PyTorch is not installed. Cannot load .pt/.pth files.")
    elif serialization_method == "joblib":
        return joblib.load(file_name)
    elif serialization_method == "pickle":
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    else:
        ocerror.Error.value_error(f"Invalid serialization method: '{serialization_method}'. Must be 'auto', 'joblib', 'pickle', or 'torch'.") # type: ignore
        raise ValueError(f"Invalid serialization method: '{serialization_method}'. Must be 'auto', 'joblib', 'pickle', or 'torch'.")


def load_data(file_name : str, exclude_column : str = 'experimental') -> pd.DataFrame:
    ''' Loads a CSV file into a DataFrame, removes rows with NaNs (except in a specified column), and notifies the user.

    Parameters
    ----------
    file_name: str
        Name of the CSV file to load.
    exclude_column: str
        Column to exclude from the NaN removal process. 

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the CSV file.
    '''

    # Read the csv file into a DataFrame
    df = pd.read_csv(file_name)
    
    # Identify columns to check for NaNs (excluding the specified column)
    columns_to_check = [col for col in df.columns if col != exclude_column]
    
    if df[columns_to_check].isnull().values.any():
        # Count the number of rows with NaN values in the columns to check
        original_size = len(df)
        rows_with_nan = df[columns_to_check].isnull().any(axis=1).sum()
        
        # Calculate the percentage of rows that will be removed
        percentage_lost = (rows_with_nan / original_size) * 100
        
        # Notify the user TODO: integrate with OCDocker
        print(f'Warning: {rows_with_nan} rows contain NaN values in columns other than "{exclude_column}".')
        print(f'These rows will be removed, which is {percentage_lost:.2f}% of the original dataset.')
        
        # Remove rows with NaN values (except in the specified column)
        df = df.dropna(subset=columns_to_check)
    
    return df


def save_object(obj : Any, filename : str, serialization_method : str = "auto") -> None:
    ''' Save an object to a file using pickle, joblib, or torch.

    Parameters
    ----------
    obj : Any
        The object to be saved.
    filename : str
        The name of the file where the object will be stored.
    serialization_method : str
        The serialization method to use. Options are:
        - "auto": Automatically detect from file extension (.pt/.pth -> torch, .pkl -> joblib)
        - "joblib": Use joblib to save (recommended for sklearn models, XGBoost)
        - "pickle": Use pickle to save
        - "torch": Use torch.save to save (for PyTorch models)
    '''

    # Auto-detect format from file extension if "auto" is specified
    if serialization_method == "auto":
        if filename.endswith('.pt') or filename.endswith('.pth'):
            serialization_method = "torch"
        elif filename.endswith('.pkl'):
            serialization_method = "joblib"  # Default to joblib for .pkl
        else:
            # Default to joblib for unknown extensions
            serialization_method = "joblib"

    # Save based on method
    if serialization_method == "torch":
        try:
            import torch
            torch.save(obj, filename)
        except ImportError:
            ocerror.Error.value_error("PyTorch is not installed. Cannot save .pt/.pth files.") # type: ignore
            raise ValueError("PyTorch is not installed. Cannot save .pt/.pth files.")
    elif serialization_method == "joblib":
        joblib.dump(obj, filename)
    elif serialization_method == "pickle":
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
    else:
        ocerror.Error.value_error(f"Invalid serialization method: '{serialization_method}'. Must be 'auto', 'joblib', 'pickle', or 'torch'.") # type: ignore
        raise ValueError(f"Invalid serialization method: '{serialization_method}'. Must be 'auto', 'joblib', 'pickle', or 'torch'.")

    return None


def get_models_dir() -> str:
    ''' Get the path to the OCScore models directory.
    
    This directory is used to store models and masks that are shipped with the code.
    The directory is located at the project root level (same level as ODDT_models),
    separate from the code folder. The directory is created if it doesn't exist.
    
    Returns
    -------
    str
        Path to the models directory.
    '''
    
    # Get the directory where this module is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to OCScore directory
    ocscore_dir = os.path.dirname(current_dir)
    # Go up to OCDocker package directory
    ocdocker_dir = os.path.dirname(ocscore_dir)
    # Go up to project root (where ODDT_models is located)
    project_root = os.path.dirname(ocdocker_dir)
    # Create models directory path at project root
    models_dir = os.path.join(project_root, "OCScore_models")
    
    # Create directory if it doesn't exist
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    return models_dir


def save_mask(mask: Union[list, np.ndarray], name: str, models_dir: Optional[str] = None) -> str:
    ''' Save a mask to a file in the models directory.
    
    Parameters
    ----------
    mask : list | np.ndarray
        The mask array of 0s and 1s to save.
    name : str
        Name for the mask file (without extension). The file will be saved as
        '{name}_mask.pkl' in the models directory.
    models_dir : str, optional
        Custom directory to save the mask. If None, uses the default OCScore
        models directory. Default is None.
    
    Returns
    -------
    str
        Path to the saved mask file.
    
    Raises
    ------
    ValueError
        If the mask is not a valid array of 0s and 1s.
    '''
    
    # Convert mask to numpy array
    mask_array = np.asarray(mask, dtype=int)
    
    # Validate mask contains only 0s and 1s
    if not np.all((mask_array == 0) | (mask_array == 1)):
        ocerror.Error.value_error("Mask must contain only 0s and 1s.") # type: ignore
        raise ValueError("Mask must contain only 0s and 1s.")
    
    # Get models directory
    if models_dir is None:
        models_dir = get_models_dir()
    else:
        # Ensure the custom directory exists
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir, exist_ok=True)
    
    # Create filename
    filename = os.path.join(models_dir, f"{name}_mask.pkl")
    
    # Save the mask
    save_object(mask_array, filename)
    
    return filename


def load_mask(name: str, models_dir: Optional[str] = None) -> np.ndarray:
    ''' Load a mask from a file in the models directory.
    
    Parameters
    ----------
    name : str
        Name of the mask file (without extension). The function will look for
        '{name}_mask.pkl' in the models directory.
    models_dir : str, optional
        Custom directory to load the mask from. If None, uses the default OCScore
        models directory. Default is None.
    
    Returns
    -------
    np.ndarray
        The loaded mask array.
    
    Raises
    ------
    FileNotFoundError
        If the mask file is not found.
    '''
    
    # Get models directory
    if models_dir is None:
        models_dir = get_models_dir()
    
    # Create filename
    filename = os.path.join(models_dir, f"{name}_mask.pkl")
    
    # Check if file exists
    if not os.path.isfile(filename):
        ocerror.Error.file_not_exist(f"Mask file not found: {filename}") # type: ignore
        raise FileNotFoundError(f"Mask file not found: {filename}")
    
    # Load the mask - try different serialization methods
    try:
        # First try joblib (most common for masks)
        mask = load_object(filename, serialization_method="joblib")
    except (ValueError, EOFError, pickle.UnpicklingError) as e:
        # If joblib fails, try pickle
        try:
            mask = load_object(filename, serialization_method="pickle")
        except (ValueError, EOFError, pickle.UnpicklingError) as e2:
            ocerror.Error.value_error(f"Failed to load mask from {filename}: {e}. Tried both joblib and pickle.") # type: ignore
            raise ValueError(f"Failed to load mask from {filename}. The file may be corrupted or in an unsupported format. Error: {e}")
    
    # Ensure it's a numpy array
    # Handle different mask formats
    if isinstance(mask, dict):
        # If mask is a dict, try to extract the array
        if 'mask' in mask:
            mask = mask['mask']
        elif 'array' in mask:
            mask = mask['array']
        else:
            # Try to get the first value that looks like an array
            for key, value in mask.items():
                if isinstance(value, (list, np.ndarray)):
                    mask = value
                    break
            else:
                ocerror.Error.value_error(f"Mask loaded as dict but no array found. Keys: {list(mask.keys())}") # type: ignore
                raise ValueError(f"Mask loaded as dict but no array found. Keys: {list(mask.keys())}")
    
    mask_array = np.asarray(mask, dtype=int)
    
    # Validate mask contains only 0s and 1s
    if not np.all((mask_array == 0) | (mask_array == 1)):
        ocerror.Error.value_error("Loaded mask must contain only 0s and 1s.") # type: ignore
        raise ValueError("Loaded mask must contain only 0s and 1s.")
    
    return mask_array
