#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage data processment in OCDocker in the context of
scoring functions.

They are imported as:

import OCDocker.OCScore.Utils.IO as ocscoreio
'''

# Imports
###############################################################################

import os
import pandas as pd
import pickle
import seaborn as sns
import time
import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from typing import Any, Union

import OCDocker.OCScore.Utils.IO as ocscoreio

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

def apply_pca(df: pd.DataFrame, pca_model_path: str, columns_to_skip_pca: list[str] = [], inplace: bool = False) -> Union[None, pd.DataFrame]:
    ''' Applies PCA to a DataFrame using a pre-trained PCA model.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    pca_model_path: str
        Path to the pre-trained PCA model.
    columns_to_skip_pca: list[str], optional
        List of columns to keep in the DataFrame before applying PCA. The default is [].
    inplace: bool, optional
        If True, the original DataFrame is modified. If False, a new DataFrame
        is returned. The default is False.
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame with PCA applied if inplace is False. None if inplace is True.
    '''
    pca = ocscoreio.load_object(pca_model_path)

    # Transform the data (excluding columns to keep)
    pca_data = pca.transform(df.drop(columns = columns_to_skip_pca, errors = 'ignore'))

    # Make it a DataFrame
    pca_data_df = pd.DataFrame(pca_data, columns = [f"PC_{i}" for i in range(pca_data.shape[1])])

    # Create a DataFrame with the metadata and reset the indexes
    metadata_df = df[columns_to_skip_pca].reset_index(drop = True)

    # Combine metadata and PCA data
    combined_df = pd.concat([metadata_df, pca_data_df], axis = 1)

    if inplace:
        # Modify the original DataFrame
        df.drop(df.columns, axis = 1, inplace = True)
        for col in combined_df.columns:
            df[col] = combined_df[col]
        return None
    else:
        # Return a new DataFrame
        return combined_df

def calculate_metrics(df: pd.DataFrame, selected_columns: list) -> tuple[pd.DataFrame, list]:
    ''' Calculates additional metrics for a DataFrame. The metrics include average, median, 
    maximum, minimum, standard deviation, variance, sum, range, 25th and 75th percentiles.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    selected_columns: list
        List of columns to calculate metrics for.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional metrics.
    list
        List of additional metrics column names.
    '''

    # Check if selected columns are present in the DataFrame
    for col in selected_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")

    # Calculate metrics
    df['mean'] = df[selected_columns].mean(axis = 1)                  # The mean of the selected columns
    df['median'] = df[selected_columns].median(axis = 1)              # The median of the selected columns
    df['max'] = df[selected_columns].max(axis = 1)                    # The maximum value of the selected columns
    df['min'] = df[selected_columns].min(axis = 1)                    # The minimum value of the selected columns
    df['std'] = df[selected_columns].std(axis = 1)                    # The standard deviation of the selected columns
    df['variance'] = df[selected_columns].var(axis = 1)               # The variance of the selected columns
    df['sum'] = df[selected_columns].sum(axis = 1)                    # The sum of the selected columns
    df['range'] = df['max'] - df['min']                               # The range of the selected columns
    df['quantile_25'] = df[selected_columns].quantile(0.25, axis = 1) # The 25th percentile of the selected columns (lower quartile)
    df['quantile_75'] = df[selected_columns].quantile(0.75, axis = 1) # The 75th percentile of the selected columns (upper quartile)
    df['iqr'] = df['quantile_75'] - df['quantile_25']                 # The interquartile range of the selected columns (IQR)
    df['skewness'] = df[selected_columns].skew(axis = 1)              # The skewness of the selected columns (measure of asymmetry)
    df['kurtosis'] = df[selected_columns].kurtosis(axis = 1)          # The kurtosis of the selected columns (measure of tailedness)

    # Return DataFrame with additional metrics
    return df, ['mean', 'median', 'max', 'min', 'std', 'variance', 'sum', 'range', 'quantile_25', 'quantile_75', 'iqr', 'skewness', 'kurtosis']

def compute_zscore(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    ''' Computes the z-score for the specified columns in a DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    columns: list
        List of columns to compute the z-score for.

    Returns
    -------
    pd.DataFrame
        DataFrame with z-score values for the specified columns.
    '''

    # Check if the specified columns are present in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")

    # Compute the z-score for the specified columns
    zscore_df = df.copy()
    zscore_df[["z_" + s for s in columns]] = (zscore_df[columns] - zscore_df[columns].mean()) / zscore_df[columns].std()

    return zscore_df

def invert_values_conditionally(df: pd.DataFrame, regex_pattern = r"^(VINA|SMINA).*|^experimental$", inplace: bool = False) -> Union[pd.DataFrame, None]:
    ''' Inverts the values of specific columns in a DataFrame. The inversion 
    is applied to columns that start with 'VINA' or 'SMINA', as well as the 
    column named 'experimental'.

    This function multiplies the values in these columns by -1, effectively 
    inverting them. It's particularly useful in scenarios where the sign of 
    these values needs to be reversed for analysis or data processing.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    regex_pattern: str
        Regular expression pattern to match the columns to invert. The default
        pattern matches columns that start with 'VINA' or 'SMINA', as well as
        the column named 'experimental'. (r"^(VINA|SMINA).*|^experimental$")
    inplace: bool
        If True, the original DataFrame is modified. If False, a new DataFrame
        is returned.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with inverted values, ensuring not to modify the original DataFrame.
    '''

    # Get the columns to invert
    invert_columns = df.filter(regex = regex_pattern).columns

    if not inplace:
        # Create a copy of the DataFrame to avoid modifying the original
        df_modified = df.copy()

        # For each column, multiply the values by -1
        for col in invert_columns:
            df_modified.loc[:, col] *= -1
    
        return df_modified
    else:
        # For each column, multiply the values by -1
        for col in invert_columns:
            df.loc[:, col] *= -1
    
    return None

def norm_data(df: pd.DataFrame, scaler: str = 'standard', inplace: bool = False) -> Union[Any, pd.DataFrame]:
    ''' Preprocesses the input DataFrame by scaling selected feature columns using a Scaler.
    The metadata columns ('receptor', 'ligand', 'name', 'type', 'db') are preserved.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    scaler: str
        Scaler to use. Options are 'standard' and 'minmax'.
    inplace: bool
        If True, the original DataFrame is modified. If False, a new DataFrame is returned.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized features while preserving metadata.
    '''

    # Check the chosen scaler
    if scaler not in ['standard', 'minmax']:
        raise ValueError("Invalid scaler. Please choose 'standard' or 'minmax'")
    
    # Initialize the scaler
    scaler_model = StandardScaler() if scaler == 'standard' else MinMaxScaler()

    # Select columns to be scaled
    feature_columns = df.columns.difference(['receptor', 'ligand', 'name', 'type', 'db'])

    if inplace:
        # Scale only the selected feature columns in the original DataFrame
        df[feature_columns] = scaler_model.fit_transform(df[feature_columns])
        return df
    
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Scale only the selected feature columns
    df_copy[feature_columns] = scaler_model.fit_transform(df_copy[feature_columns])

    return df_copy

def remove_other_columns(df: pd.DataFrame, columns_to_keep: list, inplace: bool = False) -> Union[Any, pd.DataFrame]:
    ''' Removes columns from a DataFrame that are not in the specified list.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    columns_to_keep: list
        List of columns to keep.
    inplace: bool
        If True, the original DataFrame is modified. If False, a new DataFrame is returned.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the specified columns.
    '''

    # Check if the specified columns are present in the DataFrame
    for col in columns_to_keep:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")

    if inplace:
        # Remove columns that are not in the specified list
        df.drop(columns = df.columns.difference(columns_to_keep), axis = 1, inplace = True)
        return df
    
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Remove columns that are not in the specified list
    df_copy.drop(columns = df_copy.columns.difference(columns_to_keep), axis = 1, inplace = True)

    return df_copy

def preprocess_df(file_name: str, score_columns_list: list[str] = ['SMINA', 'VINA', 'ODDT', 'PLANTS']) -> tuple[pd.DataFrame, list[str]]:
    ''' Load a DataFrame from a file and preprocess it.

    Parameters
    ----------
    file_name : str
        The name of the file to load the DataFrame from.
    score_columns_list : list[str], optional
        The list of columns to be considered as score columns. The default is ['SMINA', 'VINA', 'ODDT', 'PLANTS'].

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    list[str]
        The list of score columns.
    '''

    # Load and preprocess data
    df = ocscoreio.load_data(file_name)

    # Check if the score columns list is not empty
    if score_columns_list:
        # Define the score columns
        score_columns = df.filter(regex=f"^({'|'.join(score_columns_list)})").columns.to_list()
    else:
        # Define the score columns
        score_columns = score_columns_list

    # Split DUDEz data from PDBbind
    dudez_data = df[df['db'] == 'DUDEz']
    pdbbind_data = df[df['db'] == 'PDBbind']

    # Inverting values
    dudez_data = invert_values_conditionally(dudez_data)

    # Inverting values
    pdbbind_data = invert_values_conditionally(pdbbind_data)

    # Drop the experimental column from DUDEz data
    dudez_data = dudez_data.drop(columns = 'experimental')

    return df, score_columns

def split_dataset(X, y, test_size = 0.2, random_state = 42) -> list[Any]:
    ''' Split the data into training and testing sets.

    Parameters
    ----------
    X : pd.DataFrame
        The input features.
    y : pd.Series
        The target variable.
    test_size : float, optional
        The proportion of the dataset to include in the test split. The default is 0.2.
    random_state : int, optional
        The seed used by the random number generator. The default is 42.

    Returns
    -------
    X_train : pd.DataFrame
        The training input features.
    X_test : pd.DataFrame
        The testing input features.
    y_train : pd.Series
        The training target variable.
    y_test : pd.Series
        The testing target variable.
    '''
    
    # Split the data into training and testing sets
    return train_test_split(X, y, test_size = test_size, random_state = random_state)
