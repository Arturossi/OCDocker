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
    pca_data = pca.transform(df.drop(columns = columns_to_skip_pca, errors = "ignore"))

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
    df["mean"] = df[selected_columns].mean(axis = 1)                  # The mean of the selected columns
    df["median"] = df[selected_columns].median(axis = 1)              # The median of the selected columns
    df["max"] = df[selected_columns].max(axis = 1)                    # The maximum value of the selected columns
    df["min"] = df[selected_columns].min(axis = 1)                    # The minimum value of the selected columns
    df["std"] = df[selected_columns].std(axis = 1)                    # The standard deviation of the selected columns
    df["variance"] = df[selected_columns].var(axis = 1)               # The variance of the selected columns
    df["sum"] = df[selected_columns].sum(axis = 1)                    # The sum of the selected columns
    df["range"] = df["max"] - df["min"]                               # The range of the selected columns
    df["quantile_25"] = df[selected_columns].quantile(0.25, axis = 1) # The 25th percentile of the selected columns (lower quartile)
    df["quantile_75"] = df[selected_columns].quantile(0.75, axis = 1) # The 75th percentile of the selected columns (upper quartile)
    df["iqr"] = df["quantile_75"] - df["quantile_25"]                 # The interquartile range of the selected columns (IQR)
    df["skewness"] = df[selected_columns].skew(axis = 1)              # The skewness of the selected columns (measure of asymmetry)
    df["kurtosis"] = df[selected_columns].kurtosis(axis = 1)          # The kurtosis of the selected columns (measure of tailedness)

    # Return DataFrame with additional metrics
    return df, ["mean", "median", "max", "min", "std", "variance", "sum", "range", "quantile_25", "quantile_75", "iqr", "skewness", "kurtosis"]

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

def load_data(base_models_folder: str, storage_id: int, df_path: str, optimization_type: str, no_scores: bool = False, only_scores: bool = False, use_PCA: bool = False, pca_type: Union[str, int] = 95, use_pdb_train: bool = True, random_seed: int = 42) -> dict:
    ''' Process the data for training and testing the models.

    Parameters
    ----------
    base_models_folder: str
        The base folder to store the models.
    storage_id: int
        The storage ID for the models.
    df_path: str
        The path to the DataFrame file.
    optimization_type: str
        The optimization type.
    no_scores: bool, optional
        If True, no scores are used. The default is False. (Will override only_scores)
    only_scores: bool, optional
        If True, only the score columns are used. The default is False.
    use_PCA: bool, optional
        If True, PCA is applied to the data. The default is False.
    pca_type: str | int, optional
        The PCA type. The default is "95". Options are "95", "90", "85", and "80".
    use_pdb_train: bool, optional
        If True, the PDBbind data is used for training. The default is True.
    random_seed: int, optional
        The random seed for splitting the data. The default is 42.

    Returns
    -------
    dict
        Dictionary containing the processed data. The keys are:
        - models_folder: The models folder.
        - study_name: The study name.
        - X_train: The training input features.
        - X_test: The testing input features.
        - y_train: The training target variable.
        - y_test: The testing target variable.
        - X_val: The validation input features.
        - y_val: The validation target variable.
    '''

    # TODO: expose this variable to the user in the future
    pca_model = f"/data/hd4tb/OCDocker/OCDocker/OCDocker/OCScore/pca{pca_type}.pkl"

    # Set the models folder
    models_folder = f"{base_models_folder}/{optimization_type}_{storage_id}"

    ############################################################################################################

    # Load and preprocess data returning the DataFrame and the score columns
    df, score_columns = preprocess_df(df_path)

    # Invert the values conditionally
    invert_values_conditionally(df, inplace = True)

    # Split DUDEz data from PDBbind
    dudez_data = df[df["db"] == "DUDEz"]
    pdbbind_data = df[df["db"] == "PDBbind"]

    # Drop the experimental column from DUDEz data
    dudez_data = dudez_data.drop(columns = "experimental")

    pdbbind_norm_df = norm_data(pdbbind_data, scaler = "standard")
    dudez_norm_df = norm_data(dudez_data, scaler = "standard")

    # Filter the columns to keep
    if no_scores:
            # Remove the score columns from the dfs
            dudez_norm_df = dudez_norm_df.drop(columns = score_columns)
            pdbbind_norm_df = pdbbind_norm_df.drop(columns = score_columns)

            # Set the study name
            study_name = f"NoScores_{optimization_type}_Optimization"
    elif only_scores:
        # Remove all columns except the score columns and metadata
        remove_other_columns(
            dudez_norm_df,
            ["receptor", "ligand", "name", "type", "db"] + score_columns, 
            inplace = True
        )
        remove_other_columns(
            pdbbind_norm_df,
            ["receptor", "ligand", "name", "type", "db", "experimental"] + score_columns,
            inplace = True
        )

        # Set the study name
        study_name = f"ScoreOnly_{optimization_type}_Optimization"
    else:
        # Set the study name
        study_name = f"{optimization_type}_Optimization"
    
    if use_PCA:
        apply_pca(pdbbind_norm_df, pca_model, columns_to_skip_pca=["receptor", "ligand", "name", "type", "db", "experimental"] + score_columns, inplace=True)

        # Transform the data (validation)
        if use_pdb_train:
            apply_pca(dudez_norm_df, pca_model, columns_to_skip_pca=["receptor", "ligand", "name", "type", "db"] + score_columns, inplace=True)
        
        # Set the study name
        study_name = f"PCA{pca_type}_{study_name}"

    if use_pdb_train:
        # Split the PDBbind data into training and testing sets
        X_train, X_test, y_train, y_test = split_dataset(
            pdbbind_norm_df.drop(
                columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
                errors = "ignore"
            ), 
            pdbbind_norm_df["experimental"], 
            test_size = 0.25,
            random_state = random_seed
        )

        # Split the DUDEz data into validation X and y
        X_val = dudez_norm_df.drop(
            columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
            errors = "ignore"
        )

        y_val = dudez_norm_df["type"].map(
            {
                "ligand": 1,
                "decoy": 0
            }
        )
    else:
        # Set the test size to 0.0 to use the entire dataset for training
        X_train = dudez_norm_df.drop(
            columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
            errors = "ignore"
        )
        y_train = dudez_norm_df["experimental"]

        X_test = dudez_norm_df.drop(
            columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
            errors = "ignore"
        )
        y_test = dudez_norm_df["type"].map(
            {
                "ligand": 1, 
                "decoy": 0
            }
        )

        # Set X and y for validation to None
        X_val = None
        y_val = None
    
    # If models folder does not exist, create it
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    return {
        "models_folder": models_folder,
        "study_name": study_name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_val": X_val,
        "y_val": y_val
    }

def norm_data(df: pd.DataFrame, scaler: str = "standard", inplace: bool = False) -> Union[Any, pd.DataFrame]:
    ''' Preprocesses the input DataFrame by scaling selected feature columns using a Scaler.
    The metadata columns ("receptor", "ligand", "name", "type", "db") are preserved.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    scaler: str
        Scaler to use. Options are "standard" and "minmax".
    inplace: bool
        If True, the original DataFrame is modified. If False, a new DataFrame is returned.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized features while preserving metadata.
    '''

    # Check the chosen scaler
    if scaler not in ["standard", "minmax"]:
        raise ValueError("Invalid scaler. Please choose 'standard' or 'minmax'.")
    
    # Initialize the scaler
    scaler_model = StandardScaler() if scaler == "standard" else MinMaxScaler()

    # Select columns to be scaled
    feature_columns = df.columns.difference(["receptor", "ligand", "name", "type", "db"])

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

def preprocess_df(file_name: str, score_columns_list: list[str] = ["SMINA", "VINA", "ODDT", "PLANTS"]) -> tuple[pd.DataFrame, list[str]]:
    ''' Load a DataFrame from a file and preprocess it.

    Parameters
    ----------
    file_name : str
        The name of the file to load the DataFrame from.
    score_columns_list : list[str], optional
        The list of columns to be considered as score columns. The default is ["SMINA", "VINA", "ODDT", "PLANTS"].

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
    dudez_data = df[df["db"] == "DUDEz"]
    pdbbind_data = df[df["db"] == "PDBbind"]

    # Inverting values
    dudez_data = invert_values_conditionally(dudez_data)

    # Inverting values
    pdbbind_data = invert_values_conditionally(pdbbind_data)

    # Drop the experimental column from DUDEz data
    dudez_data = dudez_data.drop(columns = "experimental") # type: ignore 

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
