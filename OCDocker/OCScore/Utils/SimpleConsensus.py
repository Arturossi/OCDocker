#!/usr/bin/env python3

# Description
###############################################################################
''' Module to perform the simple consensus calculation for the given dataset.

It is imported as:

import OCDocker.OCScore.Utils.SimpleConsensus as ocsimple
'''

# Imports
###############################################################################

import pandas as pd
import numpy as np

from sklearn.metrics import auc, mean_squared_error, roc_curve

import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.Toolbox.Printing as ocprint

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - G1-19,
University City - Rio de Janeiro, RJ, ZIP: 21941-902
E-mail address:
This project is licensed under Creative Commons (CC-BY-4.0).
'''

# Classes
###############################################################################

# Methods
###############################################################################


def simple_consensus(
        data : pd.DataFrame,
        score_columns : list[str]
    ) -> pd.DataFrame:
    ''' Perform the consensus calculation for the given dataset. The metrics are: mean, median, max, min, std, variance, sum, range, 25th and 75th percentiles, kurtoisis, skewness.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the dataset.
    score_columns : list[str]
        The list of columns containing the scores.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the combined metrics.
    '''

    # Create a DataFrame to store the combined metrics
    df = pd.DataFrame()

    # For each row in the DataFrame, calculate the combined metric (mean, median, max, min, std, variance, sum, range, 25th and 75th percentiles, kurtoisis, skewness)
    df['mean'] = data[score_columns].mean(axis = 1)
    df['median'] = data[score_columns].median(axis = 1)
    df['max'] = data[score_columns].max(axis = 1)
    df['min'] = data[score_columns].min(axis = 1)
    df['std'] = data[score_columns].std(axis = 1)
    df['variance'] = data[score_columns].var(axis = 1)
    df['sum'] = data[score_columns].sum(axis = 1)
    df['range'] = data[score_columns].max(axis = 1) - data[score_columns].min(axis = 1)
    df['quantile_25'] = data[score_columns].quantile(0.25, axis = 1)
    df['quantile_75'] = data[score_columns].quantile(0.75, axis = 1)
    df['iqr'] = df['quantile_75'] - df['quantile_25']
    df['skewness'] = data[score_columns].skew(axis = 1)
    df['kurtosis'] = data[score_columns].kurtosis(axis = 1)
    
    # If the experimental column is present in input dataframe
    if 'experimental' in data.columns:
        # Add the experimental column to the stats DataFrame
        df['experimental'] = data['experimental']

    # If the type column is present in input dataframe
    if 'type' in data.columns:
        # Add the type column to the stats DataFrame
        df['type'] = data['type']

    return df


def perform_simple_consensus(
        df_path : str,
        threshold : float = 1.2,
        metrics : list[str] = ['mean', 'median', 'max', 'min', 'std', 'variance', 'sum', 'range', 'quantile_25', 'quantile_75', 'iqr', 'skewness', 'kurtosis'],
        verbose : bool = False
    ) -> pd.DataFrame:
    ''' Perform the simple consensus calculation for the given dataset.
    
    Parameters
    ----------
    df_path : str
        The path to the DataFrame.
    threshold : float, optional
        The threshold to filter the results. Default is 1.2.
    metrics : list[str], optional
        The list of metrics to calculate. Default is ['mean', 'median', 'max', 'min', 'std', 'variance', 'sum', 'range', 'quantile_25', 'quantile_75', 'iqr', 'skewness', 'kurtosis'].
        If empty, all metrics will be calculated.
    verbose : bool, optional
        Whether to print the results. Default is False.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the AUC and Error values
    '''

    # Parse the data from the CSV files
    dudez_data, pdbbind_data, score_columns = ocscoredata.preprocess_df(df_path)

    # Get the data for the DUDEz dataset
    dudez_stats_df = simple_consensus(dudez_data, score_columns)

    # Get the data for the pdbbind dataset
    pdbbind_stats_df = simple_consensus(pdbbind_data, score_columns)

    # Create the final df to hold Error and AUC values
    final_df = pd.DataFrame()

    # Check if the metrics list is empty
    if not metrics:
        # If empty, use all columns
        metrics = ['mean', 'median', 'max', 'min', 'std', 'variance', 'sum', 'range', 'quantile_25', 'quantile_75', 'iqr', 'skewness', 'kurtosis']

    # Calculate the AUC for each new metric
    for col in metrics:
        fpr, tpr, _ = roc_curve(dudez_stats_df['type'].map({'ligand': 1, 'decoy': 0}), dudez_stats_df[col])

        # Calculate the AUC
        final_df.loc[col, 'AUC'] = float(auc(fpr, tpr))

        # Calculate the mean squared error (from pdbbind_stats_df)
        final_df.loc[col, 'RMSE'] = np.sqrt(mean_squared_error(pdbbind_stats_df['experimental'], pdbbind_stats_df[col]))
        
    if verbose:
        # Print the results only for the rows with error below the threshold (to avoid the plot to have outliers)
        ocprint.printv(f"The rows with error smaller than the threshold of {threshold}:\n{final_df[final_df['Error'] < threshold]}")

    return final_df
