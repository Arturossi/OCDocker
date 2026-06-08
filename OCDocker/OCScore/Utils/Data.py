#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage data processment in OCDocker in the context of
scoring functions.

Usage:

import OCDocker.OCScore.Utils.Data as ocscoredata
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

def apply_pca(df : pd.DataFrame, pca_model : Union[str, PCA], columns_to_skip_pca : list[str] = [], inplace : bool = False) -> Union[None, pd.DataFrame]:
    ''' Applies PCA to a DataFrame using a pre-trained PCA model.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    pca_model: str
        Path to the pre-trained PCA model or the PCA model.
    columns_to_skip_pca: list[str], optional
        List of columns to keep in the DataFrame before applying PCA. The default is [].
    inplace: bool, optional
        If True, the original DataFrame is modified. If False, a new DataFrame
        is returned. The default is False.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with PCA applied if inplace is False. None if inplace is True.

    Raises
    ------
    FileNotFoundError
        If the PCA model path is not found.
    TypeError
        If the PCA model type is invalid. Must be a string or a PCA model.
    '''

    # Check if the PCA model is a string
    if isinstance(pca_model, str):
        # Check if pca_model_path is a valid file
        if not os.path.isfile(pca_model):
            # User-facing error: file not found
            ocerror.Error.file_not_exist(f"PCA model file not found: {pca_model}")
            raise FileNotFoundError(f"File {pca_model} not found")

        # Load the pre-trained PCA model
        pca = ocscoreio.load_object(pca_model, trusted=True)
    elif isinstance(pca_model, PCA):
        # Use the PCA model directly
        pca = pca_model
    else:
        raise TypeError("Invalid PCA model type. Please provide a path to a pre-trained PCA model or a PCA model.")

    # Apply PCA transformation (excluding columns to keep)
    pca_data = pca.transform(
        df.drop(columns = columns_to_skip_pca, errors = 'ignore')
    )

    # Convert PCA-transformed data to DataFrame
    pca_data_df = pd.DataFrame(pca_data, columns=[f"PC_{i}" for i in range(pca_data.shape[1])])

    # Retrieve the metadata columns (columns to skip PCA) and reset their index
    metadata_df = df[columns_to_skip_pca].reset_index(drop=True)

    # Concatenate the metadata and the PCA-transformed data
    combined_df = pd.concat([metadata_df, pca_data_df], axis = 1)

    if inplace:
        # Modify the original DataFrame in place
        df.drop(df.columns, axis = 1, inplace = True)

        # For each column in the combined DataFrame
        for col in combined_df.columns:
            # Add the columns from the combined DataFrame to the original DataFrame
            df[col] = combined_df[col].values
        return None
    else:
        # Return a new DataFrame with PCA applied
        return combined_df


def _get_reference_column_order_from_cfg_file(config_file: Optional[str] = None) -> Optional[list[str]]:
    '''Read ``reference_column_order`` directly from an OCDocker config file.

    This is a lightweight fallback used when config bootstrap/get_config is not
    available (for example, when DB bootstrap fails but the local config file
    exists).

    Parameters
    ----------
    config_file : str | None, optional
        Path to config file. If None, uses ``OCDOCKER_CONFIG`` env var or
        ``OCDocker.cfg``.

    Returns
    -------
    list[str] | None
        Parsed reference column order, or None if not found/invalid.
    '''

    cfg_path = config_file or os.getenv("OCDOCKER_CONFIG") or "OCDocker.cfg"
    if not os.path.isfile(cfg_path):
        return None

    try:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("reference_column_order"):
                    parts = raw_line.split("=", 1)
                    if len(parts) != 2:
                        return None
                    cols = [c.strip() for c in parts[1].strip().split(",") if c.strip()]
                    return cols if cols else None
    except (OSError, UnicodeDecodeError):
        return None

    return None


def get_column_order(data: Optional[Union[str, pd.DataFrame]] = None) -> list[str]:
    '''Get the column order from a data source (file path or DataFrame) or from config.

    This function extracts the column order from either a file path, an existing
    DataFrame, or from the config file if no data source is provided. This ensures
    consistency with the order used during model training. This is critical for
    proper mask application and feature alignment.

    Parameters
    ----------
    data : str | pd.DataFrame | None, optional
        Either:
        - A file path (CSV or gzipped CSV) to load column order from
        - A pandas DataFrame to extract column order from
        - None to use the column order from config (default: None)

    Returns
    -------
    list[str]
        List of column names in the exact order they appear in the data source or config.

    Raises
    ------
    FileNotFoundError
        If data is a string path and the file is not found.
    TypeError
        If data is neither a string, DataFrame, nor None.
    ValueError
        If data is None and config does not have reference_column_order set.
    '''

    # If no data provided, try to get from config
    if data is None:
        try:
            from OCDocker.Config import get_config
            config = get_config()
            if config.paths.reference_column_order:
                return list(config.paths.reference_column_order)
            else:
                ocerror.Error.value_error("No data source provided and 'reference_column_order' not set in config file.")
                raise ValueError("No data source provided and 'reference_column_order' not set in config file. Please provide a data source or set 'reference_column_order' in OCDocker.cfg")
        except (ImportError, AttributeError) as e:
            ocerror.Error.value_error(f"Could not load config: {e}. Please provide a data source.")
            raise ValueError(f"Could not load config: {e}. Please provide a data source.")

    if isinstance(data, pd.DataFrame):
        # Extract column order directly from DataFrame
        return list(data.columns)
    elif isinstance(data, str):
        # Load column order from file
        if not os.path.isfile(data):
            ocerror.Error.file_not_exist(f"Data file not found: {data}")
            raise FileNotFoundError(f"Data file not found: {data}")

        # Load just the header to get column order
        try:
            df = pd.read_csv(data, compression='infer', nrows=0)
        except Exception as e:
            # Fallback: try loading with ocscoreio
            df = ocscoreio.load_data(data)
            if len(df) > 0:
                df = df.iloc[:0]  # Keep only column structure

        return list(df.columns)
    else:
        ocerror.Error.value_error(f"Invalid data type: {type(data)}. Expected str (file path), pd.DataFrame, or None.")
        raise TypeError(f"Invalid data type: {type(data)}. Expected str (file path), pd.DataFrame, or None.")


def invert_values_conditionally(df : pd.DataFrame, regex_pattern : str = r"^(VINA|SMINA|PLANTS).*|^experimental$", inplace : bool = False) -> Optional[pd.DataFrame]:
    ''' Inverts the values of specific columns in a DataFrame. The inversion
    is applied to columns that start with 'VINA', 'SMINA', or 'PLANTS' as well
    as the column named 'experimental'.

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


def norm_data(df : pd.DataFrame, scaler : Union[str, StandardScaler, MinMaxScaler] = "standard", inplace : bool = False) -> Union[Any, pd.DataFrame, tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]]:
    ''' Preprocesses the input DataFrame by scaling selected feature columns using a Scaler.
    The metadata columns ("receptor", "ligand", "name", "type", "db") and target variable
    ("experimental") are preserved and excluded from scaling.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    scaler: str | StandardScaler | MinMaxScaler
        Scaler to use. Options are:
        - "standard" or "minmax": Creates and fits a new scaler
        - StandardScaler or MinMaxScaler object: Uses the provided pre-fitted scaler
    inplace: bool
        If True, the original DataFrame is modified. If False, a new DataFrame is returned.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]
        DataFrame with normalized features while preserving metadata and target variable.
        If scaler is a string (new scaler), returns tuple of (DataFrame, fitted_scaler) if inplace=False,
        or just DataFrame if inplace=True.
        If scaler is a pre-fitted object, returns only the DataFrame.
    '''

    # Select columns to be scaled (exclude metadata and target variable)
    # Metadata: receptor, ligand, name, type, db
    # Target: experimental (should not be scaled)
    feature_columns = df.columns.difference(["receptor", "ligand", "name", "type", "db", "experimental"])

    # Check if scaler is a pre-fitted object
    if isinstance(scaler, (StandardScaler, MinMaxScaler)):
        # Use the provided pre-fitted scaler
        scaler_model = scaler
        use_fit = False
    else:
        # Check the chosen scaler string
        if scaler not in ["standard", "minmax"]:
            # User-facing error: invalid value for scaler parameter
            ocerror.Error.value_error(f"Invalid scaler: '{scaler}'. Please choose 'standard' or 'minmax'.")
            raise ValueError("Invalid scaler. Please choose 'standard' or 'minmax'.")

        # Initialize a new scaler
        scaler_model = StandardScaler() if scaler == "standard" else MinMaxScaler()
        use_fit = True

    if inplace:
        # Scale only the selected feature columns in the original DataFrame
        if use_fit:
            df[feature_columns] = scaler_model.fit_transform(df[feature_columns])
        else:
            df[feature_columns] = scaler_model.transform(df[feature_columns])
        return df

    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Scale only the selected feature columns
    if use_fit:
        df_copy[feature_columns] = scaler_model.fit_transform(df_copy[feature_columns])
        return df_copy, scaler_model
    else:
        df_copy[feature_columns] = scaler_model.transform(df_copy[feature_columns])
        return df_copy


def remove_other_columns(df : pd.DataFrame, columns_to_keep : list, inplace : bool = False) -> Union[Any, pd.DataFrame]:
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
            # User-facing error: missing required data in DataFrame
            ocerror.Error.data_not_found(f"Column '{col}' not found in DataFrame")
            raise ValueError(f"Column {col} not found in DataFrame")

    if inplace:
        # Remove columns that are not in the specified list
        df.drop(columns = df.columns.difference(columns_to_keep), inplace = True)
        return df

    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Remove columns that are not in the specified list
    df_copy.drop(columns = df_copy.columns.difference(columns_to_keep), inplace = True)

    return df_copy


def reorder_columns_to_match_data_order(
    df: pd.DataFrame,
    data_source: Optional[Union[str, pd.DataFrame]] = None,
    keep_extra_columns: bool = True,
    fill_missing_columns: bool = False
) -> pd.DataFrame:
    '''Reorder DataFrame columns to match the column order from another data source.

    !!! CRITICAL: This function ensures that all columns are in the exact same order
    as the data source, which is essential for proper mask application and model
    inference. The order of scoring functions (SFs) is particularly important for masks.

    This is typically used to ensure prediction data has the same column order as
    the training data, ensuring masks and models work correctly.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to reorder.
    data_source : str | pd.DataFrame | None, optional
        Data source to match column order from. Either:
        - A file path (CSV or gzipped CSV) to load column order from
        - A pandas DataFrame to extract column order from
        - None to use reference_column_order from config (default: None)
    keep_extra_columns : bool, optional
        If True, columns not in data_source are kept at the end (default: True).
        If False, extra columns are dropped.
    fill_missing_columns : bool, optional
        If True, missing columns from data_source are added as NaN (default: False).
        If False, missing columns are simply not included.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns reordered to match data_source column order.

    Raises
    ------
    FileNotFoundError
        If data_source is a string path and the file is not found.
    TypeError
        If data_source is neither a string nor a DataFrame.
    '''

    # Get the data source column order
    source_order = get_column_order(data_source)

    # Get columns that exist in both DataFrames
    common_cols = [col for col in source_order if col in df.columns]

    # Build the ordered column list
    ordered_cols = common_cols.copy()

    # Add missing columns from data_source if requested
    if fill_missing_columns:
        missing_cols = [col for col in source_order if col not in df.columns]
        ordered_cols.extend(missing_cols)

    # Add extra columns (not in data_source) if requested
    if keep_extra_columns:
        extra_cols = [col for col in df.columns if col not in source_order]
        # Sort extra columns alphabetically for consistency
        extra_cols.sort()
        ordered_cols.extend(extra_cols)

    # Reorder the DataFrame
    # First, add missing columns as NaN if fill_missing_columns is True
    if fill_missing_columns:
        missing_cols = [col for col in source_order if col not in df.columns]
        for col in missing_cols:
            df[col] = np.nan

    # Select columns in the correct order (only existing columns)
    existing_ordered_cols = [col for col in ordered_cols if col in df.columns]
    df_reordered = df[existing_ordered_cols].copy()

    return df_reordered


