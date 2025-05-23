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
import pandas as pd
import pickle

from typing import Any

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

# Methods
###############################################################################

def load_object(file_name : str, serialization_method : str = "joblib") -> Any:
    ''' Load an object from a file using pickle.

    Parameters
    ----------
    file_name : str
        The name of the file from which to load the object.
    serialization_method : str
        The serialization method used to save the object. Options are "joblib" and "pickle".

    Returns
    -------
    Any
        The unpickled object.
    
    Raises
    ------
    ValueError
        If the serialization method is not "joblib" or "pickle".
    '''

    # If the serialization method is not joblib or pickle, raise an error
    if serialization_method not in ["joblib", "pickle"]:
        raise ValueError("The serialization method must be either 'joblib' or 'pickle'.")

    if serialization_method == "joblib":
        return joblib.load(file_name)
    
    with open(file_name, 'rb') as file:
        return pickle.load(file)
    
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

def save_object(obj : Any, filename : str) -> None:
    ''' Save an object to a file using pickle.

    Parameters
    ----------
    obj : Any
        The object to be pickled.
    filename : str
        The name of the file where the object will be stored.
    '''

    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

    return None
