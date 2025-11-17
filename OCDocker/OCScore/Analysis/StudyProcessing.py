#!/usr/bin/env python3

# Description
###############################################################################
'''
This module provides utilities to parse and structure Optuna study data
into best-RMSE, best-AUC, and best-combined views, with consensus scores merged in.

Usage:

from OCDocker.OCScore.Analysis.StudyProcessing import get_study_data
'''

# Imports
###############################################################################

import pandas as pd

from typing import Union



import OCDocker.OCScore.Utils.StudyParser as ocstudy

# Methods
###############################################################################


def get_study_data(
    snames : list[str],
    storage : str,
    final_metrics : pd.DataFrame,
    n_trials : int,
    error_threshold : float = 1.5,
    nn_ae_start: Union[int, None] = None,
    nn_ae_end: Union[int, None] = None,
    xgb_ga_start: Union[int, None] = None,
    xgb_ga_end: Union[int, None] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, float, float, float, float]:
    '''
    Retrieve Optuna study data and structure it by best RMSE, AUC, and combined metrics.

    Parameters
    ----------
    snames : list[str]
        List of study names.
    storage : str
        SQLAlchemy storage string.
    final_metrics : pd.DataFrame
        Consensus and raw metric dataframe.
    n_trials : int
        Number of trials per study.
    error_threshold : float
        Threshold to filter maximum RMSE.
    nn_ae_start : int | None, optional
        Start index for "NN + AE" labeling.
    nn_ae_end : int | None, optional
        End index for "NN + AE" labeling.
    xgb_ga_start : int | None, optional
        Start index for "XGB + GA" labeling.
    xgb_ga_end : int | None, optional
        End index for "XGB + GA" labeling.

    Returns
    -------
    tuple
        Filtered RMSE, AUC, combined metric dataframes + full results_df + ranges.
    '''
    
    results_df = ocstudy.analyze_studies(snames, storage=storage, n_trials=n_trials)

    if nn_ae_start is not None and nn_ae_end is not None:
        if nn_ae_start >= nn_ae_end:
            # User-facing error: invalid index range
            ocerror.Error.value_error(f"Invalid index range for 'NN + AE': start ({nn_ae_start}) must be less than end ({nn_ae_end})") # type: ignore
            raise ValueError("The start index for 'NN + AE' must be less than the end index.")
        results_df.loc[nn_ae_start:nn_ae_end - 1, 'study_type'] = 'NN + AE'

    if xgb_ga_start is not None and xgb_ga_end is not None:
        if xgb_ga_start >= xgb_ga_end:
            # User-facing error: invalid index range
            ocerror.Error.value_error(f"Invalid index range for 'XGB + GA': start ({xgb_ga_start}) must be less than end ({xgb_ga_end})") # type: ignore
            raise ValueError("The start index for 'XGB + GA' must be less than the end index.")
        results_df.loc[xgb_ga_start:xgb_ga_end - 1, 'study_type'] = 'XGB + GA'

    # Extract views
    best_rmse_df = results_df[["study_name", "study_type", "best_rmse_number", "best_rmse_value", "best_rmse_auc"]].copy()
    best_auc_df = results_df[["study_name", "study_type", "best_auc_number", "best_auc_value", "best_auc"]].copy()
    best_combined_df = results_df[["study_name", "study_type", "best_combined_number", "best_combined_metric", "best_combined_value", "best_combined_auc"]].copy()

    # Rename columns
    best_rmse_df.rename(columns={"study_type": "Methodology", "best_rmse_number": "Experiment", "best_rmse_value": "RMSE", "best_rmse_auc": "AUC"}, inplace=True)
    best_auc_df.rename(columns={"study_type": "Methodology", "best_auc_number": "Experiment", "best_auc_value": "RMSE", "best_auc": "AUC"}, inplace=True)
    best_combined_df.rename(columns={"study_type": "Methodology", "best_combined_number": "Experiment", "best_combined_metric": "combined_metric", "best_combined_value": "RMSE", "best_combined_auc": "AUC"}, inplace=True)

    # Add combined metric to RMSE and AUC views
    best_rmse_df["combined_metric"] = best_rmse_df["RMSE"] - best_rmse_df["AUC"]
    best_auc_df["combined_metric"] = best_auc_df["RMSE"] - best_auc_df["AUC"]

    # Merge in final metrics
    best_rmse_df = pd.concat([best_rmse_df, final_metrics], axis=0)
    best_auc_df = pd.concat([best_auc_df, final_metrics], axis=0)
    best_combined_df = pd.concat([best_combined_df, final_metrics], axis=0)

    # AUC normalization
    for df in [best_rmse_df, best_auc_df, best_combined_df]:
        df['AUC New'] = df['AUC'].apply(lambda x: 1 - x if x < 0.5 else x)
        df.reset_index(drop=True, inplace=True)

    min_error = min([df['RMSE'].min() for df in [best_rmse_df, best_auc_df, best_combined_df]])
    max_error = max([df['RMSE'].max() for df in [best_rmse_df, best_auc_df, best_combined_df]])
    min_auc = min([df['AUC New'].min() for df in [best_rmse_df, best_auc_df, best_combined_df]])
    max_auc = max([df['AUC New'].max() for df in [best_rmse_df, best_auc_df, best_combined_df]])

    # Filter
    best_rmse_df_filtered = best_rmse_df[best_rmse_df['RMSE'] <= error_threshold]
    best_auc_df_filtered = best_auc_df[best_auc_df['RMSE'] <= error_threshold]
    best_combined_df_filtered = best_combined_df[best_combined_df['RMSE'] <= error_threshold]

    return (
        best_rmse_df_filtered,
        best_auc_df_filtered,
        best_combined_df_filtered,
        results_df,
        min_auc, max_auc,
        min_error, max_error,
        max_error - min_error,
        max_auc - min_auc
    )
