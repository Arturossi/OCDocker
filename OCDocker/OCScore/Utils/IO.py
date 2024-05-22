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

def load_object(file_name: str) -> Any:
    ''' Load an object from a file using pickle.

    Parameters
    ----------
    file_name : str
        The name of the file from which to load the object.

    Returns
    -------
    Any
        The unpickled object.
    '''

    with open(file_name, 'rb') as file:
        return pickle.load(file)
    
def load_data(file_name: str) -> pd.DataFrame:
    ''' Loads a CSV file into a DataFrame.

    Parameters
    ----------
    file_name: str
        Name of the CSV file to load.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the CSV file.
    '''

    return pd.read_csv(file_name)

def save_object(obj: Any, filename: str) -> None:
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
