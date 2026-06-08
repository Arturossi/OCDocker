#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage data processment in OCDocker in the context of
legacy OCScore training/optimization data helpers.

Shared inference helpers live in ``OCDocker.OCScore.Utils.Data``.

Usage:

import OCDocker.OCScore.Utils.legacy.Data as ocscoredata
'''

# Imports
###############################################################################

import itertools
import math
import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Any, Optional, Union, overload, Literal, cast

import OCDocker.Error as ocerror
import OCDocker.OCScore.Utils.IO as ocscoreio

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##


import OCDocker.OCScore.Utils.Data as _data_inference
from OCDocker.OCScore.Utils.Data import (
    apply_pca,
    get_column_order,
    invert_values_conditionally,
    norm_data,
    remove_other_columns,
    reorder_columns_to_match_data_order,
)

_get_reference_column_order_from_cfg_file = _data_inference._get_reference_column_order_from_cfg_file

def calculate_metrics(df : pd.DataFrame, selected_columns : list) -> tuple[pd.DataFrame, list]:
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
            # User-facing error: missing required data in DataFrame
            ocerror.Error.data_not_found(f"Column '{col}' not found in DataFrame")
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


def chunkenize_dataset(data : Union[list[Any], np.ndarray, pd.DataFrame], id : int, num_machines : int) -> Union[list[Any], np.ndarray, pd.DataFrame]:
    '''
    Split a dataset in multiple chunks.

    Parameters
    ----------
    data : list[Any] | np.ndarray | pd.DataFrame
        The dataset to split (can be a list, numpy array, or pandas DataFrame).
    id : int
        The ID of the current machine (1-based index).
    num_machines : int
        The total number of machines (integer).

    Returns
    -------
    list[Any] | np.ndarray | pd.Dataframe
        A subset of the data that corresponds to the given id.
    '''

    # Sanity checks
    if id < 1 or id > num_machines:
        # User-facing error: invalid id parameter value
        ocerror.Error.value_error(f"Invalid id: {id}. It should be between 1 and {num_machines}.")
        raise ValueError(f"Invalid id. It should be between 1 and {num_machines}.")

    # Calculate the size of each chunk
    total_data_size = len(data)
    chunk_size = math.ceil(total_data_size / num_machines)

    # Calculate the start and end indices for the id
    start_idx = (id - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, total_data_size)

    # Return the corresponding chunk of data removing empty elements
    return data[start_idx:end_idx]
def compute_zscore(df : pd.DataFrame, columns : list) -> pd.DataFrame:
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
            # User-facing error: missing required data in DataFrame
            ocerror.Error.data_not_found(f"Column '{col}' not found in DataFrame")
            raise ValueError(f"Column {col} not found in DataFrame")

    # Compute the z-score for the specified columns
    zscore_df = df.copy()
    zscore_df[["z_" + s for s in columns]] = (zscore_df[columns] - zscore_df[columns].mean()) / zscore_df[columns].std()

    return zscore_df


def detect_extreme_outliers_iqr_columns_positive(df : pd.DataFrame, columns : list[str], extreme_factor : float = 3.0) -> dict:
    ''' Detects extreme outliers in specified columns of a DataFrame using the IQR method.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    columns: list[str]
        List of columns to check for extreme outliers.
    extreme_factor: float, optional
        The factor to determine extreme outliers. The default is 3.0.

    Returns
    -------
    dict
        Dictionary containing the extreme outliers for each specified column.
    '''

    # Initialize a dictionary to store extreme outliers for each specified column
    extreme_outliers_dict = {}

    # Loop through the specified columns
    for column in columns:
        if column in df.select_dtypes(include=['float64', 'int64']).columns:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Define the extreme outlier upper bound
            upper_bound = Q3 + extreme_factor * IQR

            # Filter rows where the value is an extreme outlier and is positive
            extreme_outliers = df[(df[column] > upper_bound) & (df[column] > 0)]

            # Store extreme outliers for this column
            extreme_outliers_dict[column] = extreme_outliers

    return extreme_outliers_dict


def generate_mask(column_names : Union[list[str], pd.Index], score_columns : list[str]) -> list[np.ndarray]:
    '''
    Generates masks with combinations of 0s and 1s for columns that match a regex pattern.
    Columns that don't match the regex are filled with 1s.

    Parameters
    ----------
    column_names : list[str] | pd.Index
        A list of strings, pandas series or pandas index representing column names.
    score_columns : list[str]
        Column names that should have combinations of 0s and 1s.

    Returns
    -------
        list[np.ndarray]
            A list of numpy arrays, where columns matching the regex pattern
            have combinations of 0s and 1s, and columns that don't match are filled with 1s.
    '''

    # Identify the indices of the columns that match the list
    matching_indices = [i for i, name in enumerate(column_names) if name in score_columns]

    # Number of total columns and the columns to apply combinations to
    total_elements = len(column_names)
    num_combinations_elements = len(matching_indices)

    # Generate all possible combinations of 0s and 1s for the matching columns
    combinations = itertools.product([0, 1], repeat=num_combinations_elements)

    # Prepare the list to store results
    results = []

    # For each combination, fill the matching columns with 0s/1s and the rest with 1s
    for combination in combinations:
        # Start with all 1s
        arr = np.ones(total_elements, dtype=int)

        # Set the matching columns to the current combination of 0s/1s
        for idx, value in zip(matching_indices, combination):
            arr[idx] = value

        # Append this mask to the results
        results.append(arr)

    return results


def load_data(
        base_models_folder : str,
        storage_id : int,
        df_path : str,
        optimization_type : str,
        pca_model : Union[str, PCA] = "",
        no_scores : bool = False,
        only_scores : bool = False,
        use_PCA : bool = False,
        pca_type : Union[str, int] = 95,
        use_pdb_train : bool = True,
        random_seed : int = 42,
        invert_conditionally : bool = True,
        normalize : bool = True,
        scaler : str = "standard",
        enforce_reference_order : bool = False
    ) -> dict:
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
    pca_model: str | PCA, optional
        The PCA model or the path to the PCA model. The default is "".
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
    invert_conditionally: bool, optional
        If True, invert score-like columns conditionally during preprocessing.
        The default is True.
    normalize: bool, optional
        If True, normalize data during preprocessing. The default is True.
    scaler: str, optional
        Scaler used when normalize=True. Options are "standard" and "minmax".
        The default is "standard".
    enforce_reference_order: bool, optional
        If True, reorder columns using ``reference_column_order`` from config
        before split, ensuring stable feature/mask alignment. The default is False.

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

    # Check if the PCA model is an empty string
    if use_PCA and pca_model == "":
        # Set the PCA model path
        pca_model = os.path.join(base_models_folder, f"pca{pca_type}.pkl")

    # Set the models folder
    models_folder = f"{base_models_folder}/{optimization_type}_{storage_id}"

    ############################################################################################################

    # Load and preprocess data returning DataFrames/score columns and, when
    # available, the fitted scaler (for consistent inference preprocessing).
    fitted_scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
    if normalize:
        preprocessed = preprocess_df(
            df_path,
            scaler = scaler,
            invert_conditionally = invert_conditionally,
            normalize = normalize,
            return_scaler = True
        )
        # Keep backward compatibility with tests/mocks that may still return
        # only 3 values.
        if len(preprocessed) == 4:
            dudez_data, pdbbind_data, score_columns, fitted_scaler = preprocessed
        else:
            dudez_data, pdbbind_data, score_columns = preprocessed
    else:
        dudez_data, pdbbind_data, score_columns = preprocess_df(
            df_path,
            scaler = scaler,
            invert_conditionally = invert_conditionally,
            normalize = normalize
        )

    # Filter the columns to keep
    if no_scores:
        # Remove the score columns from the dfs
        dudez_data = dudez_data.drop(columns = score_columns)
        pdbbind_data = pdbbind_data.drop(columns = score_columns)

        # Set the study name
        study_name = f"NoScores_{optimization_type}_Optimization"
    elif only_scores:
        # Remove all columns except the score columns and metadata
        remove_other_columns(
            dudez_data,
            ["receptor", "ligand", "name", "type", "db"] + score_columns,
            inplace = True
        )
        remove_other_columns(
            pdbbind_data,
            ["receptor", "ligand", "name", "type", "db", "experimental"] + score_columns,
            inplace = True
        )

        # Set the study name
        study_name = f"ScoreOnly_{optimization_type}_Optimization"
    else:
        # Set the study name
        study_name = f"{optimization_type}_Optimization"

    if use_PCA:
        apply_pca(pdbbind_data, pca_model, columns_to_skip_pca=["receptor", "ligand", "name", "type", "db", "experimental"] + score_columns, inplace=True)

        # Transform the data (validation)
        if use_pdb_train:
            apply_pca(dudez_data, pca_model, columns_to_skip_pca=["receptor", "ligand", "name", "type", "db"] + score_columns, inplace=True)

        # Set the study name
        study_name = f"PCA{pca_type}_{study_name}"

    # Ensure feature order matches the configured training order when requested.
    # This is critical for mask alignment at train/inference time.
    if enforce_reference_order:
        reference_cols = _get_reference_column_order_from_cfg_file()
        if reference_cols:
            reference_df = pd.DataFrame(columns=reference_cols)
            dudez_data = reorder_columns_to_match_data_order(
                dudez_data,
                data_source = reference_df,
                keep_extra_columns = True,
                fill_missing_columns = False
            )
            pdbbind_data = reorder_columns_to_match_data_order(
                pdbbind_data,
                data_source = reference_df,
                keep_extra_columns = True,
                fill_missing_columns = False
            )
        else:
            # Fallback to runtime config object (requires bootstrap/get_config).
            dudez_data = reorder_columns_to_match_data_order(
                dudez_data,
                data_source = None,
                keep_extra_columns = True,
                fill_missing_columns = False
            )
            pdbbind_data = reorder_columns_to_match_data_order(
                pdbbind_data,
                data_source = None,
                keep_extra_columns = True,
                fill_missing_columns = False
            )

    if use_pdb_train:
        # Split the PDBbind data into training and testing sets
        X_train, X_test, y_train, y_test = split_dataset(
            pdbbind_data.drop(
                columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
                errors = "ignore"
            ),
            pdbbind_data["experimental"],
            test_size = 0.25,
            random_state = random_seed
        )

        # Split the DUDEz data into validation X and y
        X_val = dudez_data.drop(
            columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
            errors = "ignore"
        )

        y_val = dudez_data["type"].map(
            {
                "ligand": 1,
                "decoy": 0
            }
        )
    else:
        # Set the test size to 0.0 to use the entire dataset for training
        X_train = dudez_data.drop(
            columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
            errors = "ignore"
        )
        y_train = dudez_data["experimental"]

        X_test = dudez_data.drop(
            columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
            errors = "ignore"
        )
        y_test = dudez_data["type"].map(
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
        "y_val": y_val,
        "scaler": fitted_scaler


    }


@overload
def preprocess_df(
    file_name: str,
    score_columns_list: list[str] = ["SMINA", "VINA", "ODDT", "PLANTS"],
    outliers_columns_list: Optional[list[str]] = None,
    scaler: str = "standard",
    invert_conditionally: bool = True,
    normalize: bool = True,
    return_scaler: Literal[False] = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    ...


@overload
def preprocess_df(
    file_name: str,
    score_columns_list: list[str] = ["SMINA", "VINA", "ODDT", "PLANTS"],
    outliers_columns_list: Optional[list[str]] = None,
    scaler: str = "standard",
    invert_conditionally: bool = True,
    normalize: bool = True,
    return_scaler: Literal[True] = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Union[StandardScaler, MinMaxScaler]]:
    ...


def preprocess_df(
    file_name: str,
    score_columns_list: list[str] = ["SMINA", "VINA", "ODDT", "PLANTS"],
    outliers_columns_list: Optional[list[str]] = None,
    scaler: str = "standard",
    invert_conditionally: bool = True,
    normalize: bool = True,
    return_scaler: bool = False,
) -> Union[
    tuple[pd.DataFrame, pd.DataFrame, list[str]],
    tuple[pd.DataFrame, pd.DataFrame, list[str], Union[StandardScaler, MinMaxScaler]],
]:
    ''' Load a DataFrame from a file and preprocess it.

    Parameters
    ----------
    file_name : str
        The name of the file to load the DataFrame from.
    score_columns_list : list[str], optional
        The list of columns to be considered as score columns. The default is ["SMINA", "VINA", "ODDT", "PLANTS"].
    outliers_columns_list : list[str], optional
        The list of columns to analyze for outliers. If None, defaults to 'PLANTS' columns. The default is None.
    scaler : str, optional
        The scaler to use. The default is "standard". Options are "standard" and "minmax".
    invert_conditionally : bool, optional
        If True, the values in the score columns are inverted conditionally. The default is True.
    normalize : bool, optional
        If True, the data is normalized. The default is True.
    return_scaler : bool, optional
        If True, returns the fitted scaler along with the data. The scaler is fitted on
        PDBbind data (training data) and used to transform both datasets. Default is False.

    Returns
    -------
    tuple
        If return_scaler is False: (dudez_data, pdbbind_data, score_columns)
        If return_scaler is True: (dudez_data, pdbbind_data, score_columns, fitted_scaler)
    '''

    # Load the data
    df = ocscoreio.load_data(file_name)

    # If outliers_columns_list is not empty, remove extreme outliers
    if outliers_columns_list:
        df = remove_extreme_outliers_iqr_columns_positive(df, outliers_columns_list, extreme_factor=3.0)

    # Check if the score columns list is not empty
    if score_columns_list:
        # Define the score columns
        score_columns = df.filter(regex=f"^({'|'.join(score_columns_list)})").columns.to_list()
    else:
        # Define the score columns
        score_columns = score_columns_list

    # Split DUDEz data from PDBbind
    dudez_data = df[df["db"].str.upper() == "DUDEZ"]
    pdbbind_data = df[df["db"].str.upper() == "PDBBIND"]

    if invert_conditionally:
        # Inverting values for DUDEz data
        dudez_data = invert_values_conditionally(dudez_data)

        # Inverting values for PDBbind data
        pdbbind_data = invert_values_conditionally(pdbbind_data)

    # Drop the 'experimental' column from DUDEz data if it exists
    if "experimental" in dudez_data.columns:
        dudez_data = dudez_data.drop(columns="experimental")

    if normalize:
        # Fit scaler on PDBbind data (training data) and transform it
        pdbbind_data, fitted_scaler = norm_data(pdbbind_data, scaler=scaler, inplace=False)

        # Use the same scaler to transform DUDEz data (validation/test data)
        dudez_data = norm_data(dudez_data, scaler=fitted_scaler, inplace=False)

        if return_scaler:
            return dudez_data, pdbbind_data, score_columns, fitted_scaler
        else:
            return dudez_data, pdbbind_data, score_columns
    else:
        return dudez_data, pdbbind_data, score_columns
def remove_extreme_outliers_iqr_columns_positive(df : pd.DataFrame, columns : list[str], extreme_factor : float = 3.0) -> pd.DataFrame:
    ''' Removes rows with extreme outliers in specified columns of a DataFrame using the IQR method.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    columns: list[str]
        List of columns to check for extreme outliers.
    extreme_factor: float, optional
        The factor to determine extreme outliers. The default is 3.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing extreme outliers removed.
    '''

    # Get extreme outliers for the specified columns
    extreme_outliers_dict = detect_extreme_outliers_iqr_columns_positive(df, columns, extreme_factor)

    # Get the indices of all rows that contain extreme outliers in any of the specified columns
    outlier_indices = set()

    for column, outliers_df in extreme_outliers_dict.items():
        outlier_indices.update(outliers_df.index)

    # Remove rows with extreme outliers by filtering out those indices
    df_cleaned = df.drop(list(outlier_indices))

    return df_cleaned


def split_dataset(X : pd.DataFrame, y : pd.Series, test_size : float = 0.2, random_state : int = 42) -> tuple[Any, Any, Any, Any]:
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
    return cast(tuple[Any, Any, Any, Any], train_test_split(X, y, test_size = test_size, random_state = random_state))
