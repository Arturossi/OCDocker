#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage evaluation operations in OCDocker in the context of
scoring functions.

They are imported as:

import OCDocker.OCScore.Utils.Evaluation as ocseval
'''

# Imports
###############################################################################

import numpy as np
import pandas as pd

from sklearn.metrics import auc, mean_squared_error, roc_curve
from typing import Union



import OCDocker.OCScore.Utils.Data as ocscoredata

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

def compute_auc(
        df : pd.DataFrame,
        positive_class_names : Union[str, list[str]],
        score_columns : list[str],
        class_column_name : str
    ) -> pd.DataFrame:
    ''' Compute the AUC for the scores in given score_columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the scores.
    positive_class_names : Union[str, list[str]]
        The name/names of the positive class. If a string is given, it will be converted to a list. (All other classes will be considered as negative)
    score_columns : list[str]
        The list of columns containing the scores.
    class_column_name : str
        The name of the column containing the class.
    
    Returns
    -------
    pd.DataFrame
        The DataFrame with the computed AUC.
    '''

    # Check if positive_class_names is a string
    if isinstance(positive_class_names, str):
        positive_class_names = [positive_class_names]

    metrics = []

    df[class_column_name] = df[class_column_name].apply(lambda x: 1 if x in positive_class_names else 0)

    for score_column in score_columns:
        # Compute the AUC
        # Calculate AUC for each feature and store the results
        fpr, tpr, _ = roc_curve(df[class_column_name], df[score_column], pos_label = 1)
        roc_auc = auc(fpr, tpr)

        # Append the metrics to the list
        metrics.append({
            "score_column": score_column,
            "AUC": roc_auc
        })
    
    # Return the DataFrame with the metrics
    return pd.DataFrame(metrics)


def compute_rmse(
        df : pd.DataFrame,
        score_columns : list[str],
        target_column_name : str
    ) -> pd.DataFrame:
    ''' Compute the RMSE for the scores in given score_columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the scores.
    score_columns : list[str]
        The list of columns containing the scores.
    target_column_name : str
        The name of the column containing the target values.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the computed RMSE.
    '''
    
    metrics = []

    for score_column in score_columns:
        # Compute the RMSE
        rmse = np.sqrt(mean_squared_error(df[target_column_name], df[score_column]))

        # Append the metrics to the list
        metrics.append({
            "score_column": score_column,
            "RMSE": rmse
        })
    
    # Return the DataFrame with the metrics
    return pd.DataFrame(metrics)


def compute_metrics(
        df: pd.DataFrame, 
        score_columns: list[str],
        target_column_name: str, 
        db_column_name: str, 
        metric_db_name: tuple[str, str], 
        class_column_name: str, 
        positive_class_names: Union[str, list[str]],
        invert_conditionally: bool = True
    ) -> pd.DataFrame:
    ''' Compute the metrics for the scores in given score_columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the scores.
    score_columns : list[str]
        The list of columns containing the scores.
    target_column_name : str
        The name of the column containing the target values.
    db_column_name : str
        The name of the column containing the database name which will be assigned to RMSE or AUC.
    metric_db_name : tuple[str, str]
        The pair of values which will be assigned to RMSE and AUC respectively.
    class_column_name : str
        The name of the column containing the class.
    positive_class_names : Union[str, list[str]]
        The name/names of the positive class. If a string is given, it will be converted to a list. (All other classes will be considered as negative)

    Returns
    -------
    pd.DataFrame
        The DataFrame with the computed metrics.
    
    Raises
    ------
    ValueError
        If metric_db_name does not have two elements.
    '''

    # Check if metric_db_name has two elements
    if len(metric_db_name) != 2:
        # User-facing error: invalid metric_db_name format
        ocerror.Error.value_error(f"metric_db_name must have two elements. Got {len(metric_db_name)} elements: {metric_db_name}") # type: ignore
        raise ValueError("metric_db_name must have two elements.")

    # Check if positive_class_names is a string
    if isinstance(positive_class_names, str):
        positive_class_names = [positive_class_names]

    # Create the metrics list
    metrics = []

    # Check if the metrics should be inverted
    if invert_conditionally:
        # Inverting values for DUDEz data
        df = ocscoredata.invert_values_conditionally(df) # type: ignore

    # Split the dataframe into groups to compute the metrics
    df_rmse = df[df[db_column_name] == metric_db_name[0]]
    df_auc = df[df[db_column_name] == metric_db_name[1]]

    # Set the class column as 1 for the positive class and 0 for the negative class
    df_auc[class_column_name] = df_auc[class_column_name].apply(lambda x: 1 if x in positive_class_names else 0)
    #df_auc.loc[:, class_column_name] = df_auc[class_column_name].apply(lambda x: int(1) if x in positive_class_names else int(0)).astype(int)

    # Set the class column as integer
    #df_auc.loc[:, class_column_name] = df_auc.loc[:, class_column_name].astype(int)

    for score_column in score_columns:
        # Compute the RMSE
        rmse = np.sqrt(mean_squared_error(df[target_column_name], df[score_column]))

        # Compute the AUC
        # Calculate AUC for each feature and store the results
        fpr, tpr, _ = roc_curve(df_auc[class_column_name], df_auc[score_column], pos_label = 1)
        roc_auc = auc(fpr, tpr)

        # Append the metrics to the list
        metrics.append({
            "score_column": score_column,
            "RMSE": rmse,
            "AUC": roc_auc
        })
    
    # Return the DataFrame with the metrics
    return pd.DataFrame(metrics)
